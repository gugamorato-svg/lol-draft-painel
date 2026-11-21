"""Smoke tests offline. Roda com:  python -m unittest tests.test_smoke -v

Cobertura:
  - JsonCache: set/get com TTL.
  - LeaguepediaClient: cargo() chama o transporte HTTP correto e parsa o JSON;
    get_player_year_stats e get_team_year_stats calculam medias certas.
  - get_h2h: matematica de winrate e diff de kills.

A camada HTTP e' mockada (monkey-patch em session.get) - nao precisa de rede.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from api.cache import JsonCache
from api.leaguepedia import LeaguepediaClient
from api.livefeed import _lf_champion_to_oe
from api.oracle_stats import _champion_key
from api.player_resolver import PlayerResolver, norm_name
from models.trained_predictor import _read_coefs
from ui.worker import _draft_is_complete


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _wrap_rows(rows):
    """A Cargo API retorna { 'cargoquery': [ {'title': row}, ...] }."""
    return {"cargoquery": [{"title": r} for r in rows]}




class ChampionNormalizationTests(unittest.TestCase):
    def test_wukong_livefeed_alias_matches_oracle_name(self):
        self.assertEqual(_lf_champion_to_oe("MonkeyKing"), "Wukong")
        self.assertEqual(_champion_key("MonkeyKing"), _champion_key("Wukong"))


class PlayerResolverTests(unittest.TestCase):
    class FakeOracle:
        def __init__(self):
            rows = [
                {
                    "__year": 2026,
                    "position": "top",
                    "teamname": "Other Team",
                    "playername": "Target",
                },
                {
                    "__year": 2026,
                    "position": "top",
                    "teamname": "Known Team",
                    "playername": "RealTop",
                },
            ]
            self.players = pd.DataFrame(rows)
            self.players["__playername_norm"] = self.players["playername"].map(norm_name)

        def _resolve_team(self, team):
            return {"Known Team": "Known Team", "Other Team": "Other Team"}.get(team, "")

        def _resolve_player(self, player):
            return "Target" if norm_name(player) == "target" else ""

    def test_context_does_not_fall_back_to_wrong_team(self):
        resolver = PlayerResolver(self.FakeOracle())
        result = resolver.resolve("Target", team="Known Team", role="Top", year=2026)
        self.assertEqual(result.reason, "unresolved")
        self.assertEqual(result.canonical, "")
        self.assertEqual(result.display, "Target")

    def test_unknown_team_context_does_not_use_global_player(self):
        resolver = PlayerResolver(self.FakeOracle())
        result = resolver.resolve("Target", team="Unknown Team", role="Top", year=2026)
        self.assertEqual(result.reason, "unresolved")
        self.assertEqual(result.canonical, "")

    def test_context_resolves_matching_team_player(self):
        resolver = PlayerResolver(self.FakeOracle())
        result = resolver.resolve("Target", team="Other Team", role="Top", year=2026)
        self.assertEqual(result.reason, "exact/context")
        self.assertEqual(result.canonical, "Target")


class LiveDraftTests(unittest.TestCase):
    def test_draft_requires_all_ten_picks(self):
        complete = {
            f"{side}_{role}": f"{side}_{role}_champ"
            for side in ("blue", "red")
            for role in ("top", "jng", "mid", "bot", "sup")
        }
        self.assertTrue(_draft_is_complete(complete))

        incomplete = complete.copy()
        incomplete["red_sup"] = ""
        self.assertFalse(_draft_is_complete(incomplete))


class CoefficientFileTests(unittest.TestCase):
    def test_duplicate_equal_coefficients_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "coefs.csv"
            path.write_text(
                "feature,coefficient\n"
                "__intercept__,0.1\n"
                "x,1.0\n"
                "x,1.0\n"
                "__prematch_cap__,0.0\n",
                encoding="utf-8",
            )
            self.assertEqual(_read_coefs(path, set_global_cap=False)["x"], 1.0)

    def test_duplicate_conflicting_coefficients_fail_fast(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "coefs.csv"
            path.write_text(
                "feature,coefficient\n"
                "__intercept__,0.1\n"
                "x,1.0\n"
                "x,2.0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _read_coefs(path, set_global_cap=False)


class CacheTests(unittest.TestCase):
    def test_set_get_ttl(self):
        with tempfile.TemporaryDirectory() as d:
            with JsonCache(Path(d) / "c.sqlite") as c:
                c.set("ns", "k", {"a": 1}, ttl_seconds=60)
                self.assertEqual(c.get("ns", "k"), {"a": 1})

    def test_expired(self):
        with tempfile.TemporaryDirectory() as d:
            with JsonCache(Path(d) / "c.sqlite") as c:
                c.set("ns", "k", {"a": 1}, ttl_seconds=0)
                time.sleep(0.01)
                self.assertIsNone(c.get("ns", "k"))


class LeaguepediaParseTests(unittest.TestCase):
    def _client(self, rows):
        lp = LeaguepediaClient(cache=None)
        fake = FakeResponse(_wrap_rows(rows))
        lp.session = SimpleNamespace(get=lambda *a, **k: fake, headers={})
        return lp

    def test_player_year_stats(self):
        # 3 jogos: K=3/4/2, D=1/2/3, A=5/7/6 - vitoria nos 2 primeiros (Blue side W=1)
        rows = [
            {"K": "3", "D": "1", "A": "5", "W": "1", "Side": "Blue"},
            {"K": "4", "D": "2", "A": "7", "W": "1", "Side": "Blue"},
            {"K": "2", "D": "3", "A": "6", "W": "1", "Side": "Red"},  # derrota (Red mas Blue venceu)
        ]
        lp = self._client(rows)
        s = lp.get_player_year_stats("Faker", 2026)
        self.assertEqual(s["games"], 3)
        self.assertAlmostEqual(s["kills_avg"], 3.0)
        self.assertAlmostEqual(s["deaths_avg"], 2.0)
        self.assertAlmostEqual(s["assists_avg"], 6.0)
        self.assertAlmostEqual(s["winrate"], round(2 / 3, 3))

    def test_player_champion_career_stats(self):
        rows = [
            {"K": "2", "D": "1", "A": "5", "W": "1", "Side": "Blue"},
            {"K": "4", "D": "3", "A": "7", "W": "2", "Side": "Blue"},
            {"K": "6", "D": "2", "A": "1", "W": "2", "Side": "Red"},
        ]
        lp = self._client(rows)
        s = lp.get_player_champion_career_stats("Stefan", "Olaf")
        self.assertEqual(s["games"], 3)
        self.assertEqual(s["wins"], 2)
        self.assertAlmostEqual(s["kills_avg"], 4.0)
        self.assertAlmostEqual(s["deaths_avg"], 2.0)
        self.assertAlmostEqual(s["assists_avg"], round(13 / 3, 2))
        self.assertAlmostEqual(s["winrate"], 66.7)

    def test_team_year_stats_medians(self):
        rows = [
            {"GameId": "g1", "T1": "T1", "T2": "GenG", "W": "1",
             "K1": "10", "K2": "8", "Len": "30"},
            {"GameId": "g2", "T1": "T1", "T2": "KT",   "W": "2",
             "K1": "12", "K2": "20", "Len": "40"},
            {"GameId": "g3", "T1": "HLE","T2": "T1",  "W": "1",
             "K1": "5",  "K2": "5", "Len": "35"},
        ]
        lp = self._client(rows)
        s = lp.get_team_year_stats("T1", 2026)
        self.assertEqual(s["games"], 3)
        # totals = [18, 32, 10] -> media = 20.0; mediana = 18
        self.assertAlmostEqual(s["total_kills_avg"], 20.0)
        # over kills (>18) = 1 (32) -> 33.3%
        self.assertAlmostEqual(s["over_kills_pct"], 33.3)
        # durations = [30,40,35] -> media 35.0, mediana 35
        self.assertAlmostEqual(s["duration_avg_min"], 35.0)

    def test_h2h(self):
        # Team_a = "T1", duas vitorias e uma derrota contra "GenG"
        rows = [
            {"T1": "T1", "T2": "GenG", "W": "1", "K1": "12", "K2": "8"},   # T1 venceu, +4 kills
            {"T1": "GenG", "T2": "T1", "W": "1", "K1": "10", "K2": "5"},   # T1 perdeu, -5
            {"T1": "T1", "T2": "GenG", "W": "1", "K1": "15", "K2": "9"},   # T1 venceu, +6
        ]
        lp = self._client(rows)
        h = lp.get_h2h("T1", "GenG", 2026)
        self.assertEqual(h["games"], 3)
        # 2 vitorias de 3 = 66.7%
        self.assertAlmostEqual(h["winrate_a"], 66.7)
        # kill diff = (4 + -5 + 6) / 3 = 5/3 ~ 1.67
        self.assertAlmostEqual(h["kills_diff"], round(5 / 3, 2))

    def test_team_roster_picks_latest_per_role(self):
        rows = [
            {"Player": "Doran",  "Role": "Top",     "Team": "T1", "Date": "2026-04-01"},
            {"Player": "OldTop", "Role": "Top",     "Team": "T1", "Date": "2026-01-01"},
            {"Player": "Oner",   "Role": "Jungle",  "Team": "T1", "Date": "2026-04-01"},
            {"Player": "Faker",  "Role": "Mid",     "Team": "T1", "Date": "2026-04-01"},
            {"Player": "Gumayusi","Role":"Bot",     "Team": "T1", "Date": "2026-04-01"},
            {"Player": "Keria",  "Role": "Support", "Team": "T1", "Date": "2026-04-01"},
        ]
        lp = self._client(rows)
        roster = lp.get_team_roster("T1", 2026)
        names = [p["player"] for p in roster]
        # As linhas chegam em ordem DESC pela query; teste so pega o primeiro de cada role
        self.assertIn("Doran", names)
        self.assertNotIn("OldTop", names)
        self.assertEqual(len(roster), 5)


if __name__ == "__main__":
    unittest.main()
