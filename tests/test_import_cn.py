"""Importador das bases chinesas da Leaguepedia (scripts/import_leaguepedia_cn.py), 24/09/2026.

Roda com:  python -m unittest tests.test_import_cn -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import import_leaguepedia_cn as imp  # noqa: E402

ROLES = ("Top", "Jungle", "Mid", "Bot", "Support")


def oracle():
    """Oracle minimo: EDG Youth (LDL) com o mid 'Xiaoxu', e um homonimo 'Relax' so na LRN."""
    rows = [
        {"gameid": "g1", "date": "2026-06-01 10:00:00", "league": "AM", "participantid": 100,
         "teamname": "EDG Youth Team", "teamid": "oe:edgy", "patch": 16.11},
        {"gameid": "g1", "date": "2026-06-01 10:00:00", "league": "AM", "participantid": 3,
         "playername": "Xiaoxu", "playerid": "oe:xiaoxu", "teamname": "EDG Youth Team"},
        {"gameid": "g2", "date": "2026-06-20 10:00:00", "league": "LRN", "participantid": 1,
         "playername": "Relax", "playerid": "oe:relax_lrn", "teamname": "Fuego"},
        {"gameid": "g2", "date": "2026-06-20 10:00:00", "league": "LRN", "participantid": 100,
         "teamname": "Fuego", "teamid": "oe:fuego", "patch": 16.12},
    ]
    return pd.DataFrame(rows)


def jogo_wiki():
    g = {"GameId": "T/X_1_1", "Tournament": "WSCI 2026 China Qualifier", "Winner": 2,
         "Team1": "EDward Gaming Youth Team", "Team2": "Team WE Academy",
         "DateTime UTC": "2026-06-25 09:00:00", "N GameInMatch": 1, "Gamelength Number": 30.5,
         "Team1Picks": ["A", "B", "C", "D", "E"], "Team2Picks": ["F", "G", "H", "I", "J"],
         "Team1Bans": [], "Team2Bans": [], "Team1Dragons": 1, "Team2Dragons": 3}
    ps = []
    for side, time_ in ((1, g["Team1"]), (2, g["Team2"])):
        for i, r in enumerate(ROLES):
            nome = "Xiaoxu" if (side, r) == (1, "Mid") else ("Relax" if (side, r) == (2, "Top") else f"p{side}{i}")
            ps.append({"GameId": g["GameId"], "Name": nome, "Link": f"{nome} (Wiki)", "Team": time_,
                       "Champion": f"c{side}{i}", "Kills": 1, "Deaths": 2, "Assists": 3,
                       "Role": r, "Side": side})
    return g, ps


class ImportCnTests(unittest.TestCase):
    def setUp(self):
        o = oracle()
        self.nomes = imp.Nomes(o)
        g, ps = jogo_wiki()
        self.df = imp.montar([g], ps, self.nomes, imp.patch_por_data(o), {g["Tournament"]: "LDL"})

    def test_oracle_format_12_rows_red_won(self):
        self.assertEqual(len(self.df), 12)
        t = self.df[self.df["participantid"].isin([100, 200])].set_index("side")
        self.assertEqual(t.loc["Blue", "result"], 0)
        self.assertEqual(t.loc["Red", "result"], 1)
        self.assertEqual(t.loc["Blue", "opp_dragons"], 3)
        self.assertEqual(set(self.df["league"]), {"LDL"})
        self.assertEqual(set(self.df["datacompleteness"]), {"partial"})

    def test_ids_and_names_follow_oracle(self):
        t = self.df[self.df["participantid"] == 100].iloc[0]
        self.assertEqual((t["teamid"], t["teamname"]), ("oe:edgy", "EDG Youth Team"))  # alias
        mid = self.df[(self.df["side"] == "Blue") & (self.df["position"] == "mid")].iloc[0]
        self.assertEqual(mid["playerid"], "oe:xiaoxu")

    def test_foreign_namesake_is_not_merged(self):
        top = self.df[(self.df["side"] == "Red") & (self.df["position"] == "top")].iloc[0]
        self.assertEqual(top["playerid"], "lp:player:relaxwiki")    # nao o "Relax" da LRN

    def test_patch_comes_from_oracle_date(self):
        self.assertEqual(set(self.df["patch"]), {16.12})           # ultimo dia <= 25/06

    def test_duplicate_of_oracle_game_is_detected(self):
        o = oracle()
        dup = self.df.copy()
        dup["date"] = "2026-06-01 11:00:00"
        dup.loc[dup["participantid"] == 200, "teamname"] = "Fuego"
        o.loc[o["gameid"] == "g1", "teamname"] = "EDG Youth Team"
        o2 = pd.concat([o, pd.DataFrame([{"gameid": "g1", "date": "2026-06-01 10:00:00",
                                          "participantid": 200, "teamname": "Fuego"}])])
        self.assertEqual(imp.ja_no_oracle(dup, o2), set(dup["gameid"]))


if __name__ == "__main__":
    unittest.main()
