"""Live names must not silently inherit another team's or starter's stats."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from api.player_resolver import PlayerResolver, norm_name


class OracleFixture:
    def __init__(self, rows):
        self.players = pd.DataFrame(rows)
        self.players["__playername_norm"] = self.players["playername"].map(norm_name)

    def _resolve_team(self, team):
        return team if team in set(self.players.teamname) else ""

    def _resolve_player(self, name):
        matches = self.players[self.players["__playername_norm"] == norm_name(name)]
        return str(matches.iloc[0].playername) if len(matches) else ""


def appearance(name, team, year=2026):
    return {"playername": name, "teamname": team, "position": "top", "__year": year}


class PlayerResolverContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.alias_path = Path(self.temp.name) / "absent_aliases.json"

    def resolver(self, rows):
        return PlayerResolver(OracleFixture(rows), alias_path=self.alias_path)

    def test_transfer_recorded_in_new_team_remains_resolvable(self):
        resolver = self.resolver([
            appearance("Transfer", "Old Team"),
            appearance("Starter", "New Team"),
            appearance("Transfer", "New Team"),
        ])
        result = resolver.resolve("Transfer", team="New Team", role="Top", year=2026)
        self.assertEqual((result.canonical, result.reason), ("Transfer", "exact/context"))

    def test_unconfirmed_transfer_keeps_live_name_without_foreign_stats(self):
        resolver = self.resolver([
            appearance("Transfer", "Old Team"),
            appearance("Starter", "New Team"),
        ])
        result = resolver.resolve("Transfer", team="New Team", role="Top", year=2026)
        self.assertEqual((result.canonical, result.display, result.reason), ("", "Transfer", "unresolved"))

    def test_unknown_substitute_is_not_replaced_by_regular_starter(self):
        resolver = self.resolver([appearance("Starter", "Known Team")])
        result = resolver.resolve("KT Substitute", team="Known Team", role="Top", year=2026,
                                  fallback_name="Starter")
        self.assertEqual((result.canonical, result.display, result.reason), ("", "Substitute", "unresolved"))

    def test_empty_live_name_uses_roster_for_display_only(self):
        resolver = self.resolver([appearance("Starter", "Known Team")])
        result = resolver.resolve("", team="Known Team", role="Top", year=2026,
                                  fallback_name="Starter")
        self.assertEqual((result.canonical, result.display, result.reason), ("", "Starter", "unresolved"))

    def test_global_lookup_remains_available_without_team_context(self):
        resolver = self.resolver([appearance("Transfer", "Old Team", year=2025)])
        result = resolver.resolve("Transfer", year=2026)
        self.assertEqual((result.canonical, result.reason), ("Transfer", "loose"))


if __name__ == "__main__":
    unittest.main()
