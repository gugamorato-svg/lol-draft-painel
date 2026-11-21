"""Nome de time (Riot e outras fontes) -> time do Oracle, sem trocar time de base pelo principal.

Roda com:  python -m unittest tests.test_team_resolution -v
Bug de 23/09: "Edward Gaming Youth Team" (LDL) virava "Edward Gaming" (LPL) pelo trecho.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import trained_predictor as tp  # noqa: E402

NOMES = {"edward gaming": "EDG", "edg youth team": "EDGY", "furia": "FUR", "t1": "T1",
         "t1 esports academy": "T1A", "only the family": "OTF", "red canids": "RED",
         "team liquid": "TL", "team liquid challengers": "TLC", "top esports": "TES",
         "top esports challenger": "TESC", "skillcamp": "SKC"}


class TeamResolutionTests(unittest.TestCase):
    def setUp(self):
        idx = {tp._norm_name(k): v for k, v in NOMES.items()}
        self.p = [patch.object(tp, "_team_name_index", idx),
                  patch.object(tp, "_team_core_index", tp._build_team_core_index(NOMES))]
        for p in self.p:
            p.start()

    def tearDown(self):
        for p in self.p:
            p.stop()

    def test_youth_team_never_becomes_main_team(self):
        self.assertEqual(tp._resolve_team_id("Edward Gaming Youth Team"), "EDGY")
        self.assertEqual(tp._resolve_team_id("EDward Gaming Youth Team"), "EDGY")
        self.assertIsNone(tp._resolve_team_id("Weibo Gaming Youth Team"))   # nao existe: nada
        self.assertEqual(tp._resolve_team_id("Edward Gaming"), "EDG")

    def test_academy_and_challengers_kept_apart(self):
        self.assertEqual(tp._resolve_team_id("T1 Academy"), "T1A")
        self.assertEqual(tp._resolve_team_id("T1"), "T1")
        self.assertEqual(tp._resolve_team_id("Team Liquid Challengers"), "TLC")
        self.assertEqual(tp._resolve_team_id("Team Liquid"), "TL")

    def test_core_name_without_generic_words(self):
        self.assertEqual(tp._resolve_team_id("FURIA Esports"), "FUR")
        self.assertEqual(tp._resolve_team_id("Skillcamp Esport"), "SKC")

    def test_aliases(self):
        self.assertEqual(tp._resolve_team_id("RED Kalunga"), "RED")
        self.assertEqual(tp._resolve_team_id("ØTF"), "OTF")

    def test_ambiguous_core_is_dropped(self):
        idx = tp._build_team_core_index({"alpha": "A1", "alpha esports": "A2"})
        self.assertNotIn("alpha", idx)


class AppTeamResolutionTests(unittest.TestCase):
    """O app (api/oracle_stats) usa as mesmas regras do modelo (24/09: "kt Challengers" n=0)."""

    def setUp(self):
        import pandas as pd
        from api.oracle_stats import OracleStatsClient, _norm
        self.o = OracleStatsClient.__new__(OracleStatsClient)        # sem carregar o Oracle
        nomes = ["KT Rolster", "KT Rolster Challengers", "Edward Gaming", "EDG Youth Team",
                 "Movistar KOI Fénix", "Movistar KOI", "FURIA", "T1", "T1 Esports Academy"]
        self.o.teams = pd.DataFrame({"teamname": nomes, "__teamname_norm": [_norm(n) for n in nomes]})

    def test_riot_short_names(self):
        self.assertEqual(self.o._resolve_team("kt Challengers"), "KT Rolster Challengers")
        self.assertEqual(self.o._resolve_team("Movistar KOI Fenix"), "Movistar KOI Fénix")
        self.assertEqual(self.o._resolve_team("FURIA Esports"), "FURIA")

    def test_academy_never_becomes_main_team(self):
        self.assertEqual(self.o._resolve_team("Edward Gaming Youth Team"), "EDG Youth Team")
        self.assertEqual(self.o._resolve_team("T1 Academy"), "T1 Esports Academy")
        self.assertEqual(self.o._resolve_team("Weibo Gaming Youth Team"), "")
        self.assertEqual(self.o._resolve_team("KT Rolster"), "KT Rolster")


if __name__ == "__main__":
    unittest.main()
