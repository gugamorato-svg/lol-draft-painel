"""Forca da liga com ligas de base (24/09/2026): hierarquia mae/filha, liga de casa recente,
nome antigo de liga e Elo entre ligas.

Roda com:  python -m unittest tests.test_league_strength -v
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.temporal_context import TemporalContext  # noqa: E402

ROLES = ("top", "jng", "mid", "bot", "sup")


def jogo(blue, red, blue_won, league):
    g = {"blue_won": blue_won, "blue_teamid": blue, "red_teamid": red, "patch": 16.1,
         "league": league, "game": 1}
    for r in ROLES:
        g[f"blue_{r}"], g[f"red_{r}"] = f"B{r}", f"R{r}"
        g[f"blue_player_{r}"], g[f"red_player_{r}"] = f"{blue}{r}", f"{red}{r}"
    return g


def cruzados(pares, seed=1):
    """[(liga_a, liga_b, n, taxa_vitoria_a)] -> lista (a, b, venceu) p/ o Bradley-Terry."""
    rnd = random.Random(seed)
    out = []
    for a, b, n, p in pares:
        for _ in range(n):
            venceu_a = int(rnd.random() < p)
            # lado aleatorio: senao o intercepto (vantagem do azul) engole a forca
            out.append((a, b, venceu_a) if rnd.random() < 0.5 else (b, a, 1 - venceu_a))
    return out


class HierarchicalLeagueTests(unittest.TestCase):
    PARES = [("LCK", "LPL", 200, 0.60), ("LPL", "LDL", 80, 0.80), ("LCK", "LEC", 150, 0.75)]

    def test_child_without_games_follows_parent_order(self):
        ctx = TemporalContext()
        ctx.league_parents = {"LCKC": "LCK", "LDL": "LPL"}
        ctx.cross_league_games = cruzados(self.PARES)
        ctx._refit_league_ratings()
        r = ctx.league_rating
        self.assertLess(r["LCKC"], r["LCK"])          # base abaixo da principal
        self.assertLess(r["LDL"], r["LPL"])
        self.assertGreater(r["LCKC"], r["LDL"])       # LCK > LPL => LCKC > LDL sem jogo direto

    def test_flat_fit_unchanged_without_parents(self):
        ctx = TemporalContext()
        ctx.cross_league_games = cruzados(self.PARES)
        ctx._refit_league_ratings()
        self.assertNotIn("LCKC", ctx.league_rating)    # como antes: liga sem jogo nao tem forca


class HomeLeagueTests(unittest.TestCase):
    def jogar(self, ctx, liga, n):
        for _ in range(n):
            ctx.update(jogo("T", "X", 1, liga))

    def test_recent_league_wins_with_decay(self):
        velho, novo = TemporalContext(), TemporalContext()
        velho.home_decay, novo.home_decay = None, 0.97
        for ctx in (velho, novo):
            self.jogar(ctx, "PCS", 40)
            self.jogar(ctx, "LCP", 25)
        self.assertEqual(velho.home_league("T"), "PCS")   # vida toda
        self.assertEqual(novo.home_league("T"), "LCP")    # onde joga hoje

    def test_renamed_league_is_the_same_league(self):
        ctx = TemporalContext()
        ctx.league_aliases = {"LVP SL": "LES"}
        self.jogar(ctx, "LVP SL", 10)
        self.assertEqual(ctx.home_league("T"), "LES")


class LeagueInRatingTests(unittest.TestCase):
    def test_rating_diff_includes_league_strength(self):
        ctx = TemporalContext()
        ctx.team_league_games["forte"]["LCKC"] = 20.0
        ctx.team_league_games["fraco"]["LRN"] = 20.0
        ctx.league_rating.update({"LCKC": 1.0, "LRN": -0.5})
        g = jogo("forte", "fraco", 1, "WSCI")
        ctx.league_in_rating = True
        com = ctx.build_features(g)["rating_diff"]
        ctx.league_in_rating = False
        sem = ctx.build_features(g)["rating_diff"]
        self.assertEqual(sem, 0.0)                                    # Elo igual (1500 x 1500)
        self.assertAlmostEqual(com, 1.5 * 400 / math.log(10) / 250, places=9)


class LeagueEloTests(unittest.TestCase):
    def queda(self, league_elo):
        """Elo que um time de base perde ao perder p/ um time de liga bem mais forte."""
        ctx = TemporalContext()
        ctx.league_elo = league_elo
        ctx.team_league_games["forte"]["LCKC"] = 20.0
        ctx.team_league_games["fraco"]["LDL"] = 20.0
        ctx.league_rating.update({"LCKC": 1.0, "LDL": 0.0})
        ctx.update(jogo("forte", "fraco", 1, "WSCI"))
        return 1500.0 - ctx.team_rating["fraco"]

    def test_losing_to_stronger_league_costs_less(self):
        self.assertAlmostEqual(self.queda(False), 12.0, places=6)     # K=24, jogo "parelho"
        esperado_forte = 1 / (1 + math.exp(-1.0))                      # 1 logit de diferenca
        self.assertAlmostEqual(self.queda(True), 24 * (1 - esperado_forte), places=6)

    def test_offset_zero_in_same_league(self):
        ctx = TemporalContext()
        ctx.team_league_games["a"]["LCK"] = 20.0
        ctx.team_league_games["b"]["LCK"] = 20.0
        ctx.league_rating["LCK"] = 1.5
        self.assertEqual(ctx.league_elo_offset("a", "b"), 0.0)


if __name__ == "__main__":
    unittest.main()
