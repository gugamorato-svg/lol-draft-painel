import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from models.model_meta import UPSTREAM_SOURCES, upstream_source_sha256
from scripts.train_live_wr import temporal_features, past_only_probabilities, split_live_validation


class FakeContext:
    def __init__(self):
        self.wins = 0

    def build_features(self, game):
        return {"pre_match_raw": self.wins, "known_wins": self.wins}

    def update(self, game):
        self.wins += int(game["blue_won"])


class TemporalTrainingTests(unittest.TestCase):
    def test_second_stage_excludes_game_ending_after_year_boundary(self):
        data = pd.DataFrame([
            {"gameid": "past", "date": "2025-12-31T20:00Z", "available_at": "2025-12-31T21:00Z"},
            {"gameid": "overlap", "date": "2025-12-31T23:50Z", "available_at": "2026-01-01T00:30Z"},
            {"gameid": "test", "date": "2026-01-01T01:00Z", "available_at": "2026-01-01T01:30Z"},
        ])
        train, test = split_live_validation(data, 2026)
        self.assertEqual(train.gameid.tolist(), ["past"])
        self.assertEqual(test.gameid.tolist(), ["test"])

    def test_parallel_game_result_not_available_at_start(self):
        games = pd.DataFrame([
            dict(gameid="a", date="2025-01-01T10:00Z", gamelength=1800, blue_won=1),
            dict(gameid="b", date="2025-01-01T10:10Z", gamelength=1800, blue_won=0),
            dict(gameid="c", date="2025-01-01T10:30Z", gamelength=1800, blue_won=0),
            dict(gameid="d", date="2025-01-01T11:10Z", gamelength=None, blue_won=1),
            dict(gameid="e", date="2025-01-02T11:10Z", gamelength=1800, blue_won=0),
        ])
        df, features = temporal_features(games, FakeContext)
        self.assertEqual(features.known_wins.tolist(), [0, 0, 1, 1, 1])
        self.assertTrue(pd.isna(df.loc[3, "available_at"]))

    def test_month_model_cannot_use_future_or_unfinished_labels(self):
        games = pd.DataFrame([
            dict(gameid="a", date="2025-01-01T10:00Z", gamelength=1800, blue_won=1),
            dict(gameid="b", date="2025-01-01T11:00Z", gamelength=1800, blue_won=0),
            dict(gameid="c", date="2025-01-31T23:55Z", gamelength=1800, blue_won=1),
            dict(gameid="d", date="2025-02-02T11:00Z", gamelength=1800, blue_won=1),
            dict(gameid="e", date="2025-02-03T11:00Z", gamelength=1800, blue_won=0),
        ])
        df, features = temporal_features(games, FakeContext)
        seen = []

        class AuditModel:
            def fit(self, X, y):
                seen.append(X.index.tolist())
                self.p = (sum(y) + 1) / (len(y) + 2)
                return self

            def predict_proba(self, X):
                return np.tile([1-self.p, self.p], (len(X), 1))

        p, folds = past_only_probabilities(df, features, min_games=2, model_factory=AuditModel)
        changed = df.copy()
        changed.loc[3:, "blue_won"] = 1 - changed.loc[3:, "blue_won"]
        p2, _ = past_only_probabilities(changed, features, min_games=2, model_factory=AuditModel)
        self.assertEqual(seen, [[0, 1], [0, 1]])
        np.testing.assert_allclose(p, p2, equal_nan=True)
        self.assertTrue(p.iloc[:3].isna().all())
        self.assertEqual(folds[0]["month"], "2025-02")
        self.assertLess(folds[0]["latest_training_result"], folds[0]["prediction_cutoff"])

    def test_source_hash_detects_change_and_ignores_windows_newlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in UPSTREAM_SOURCES:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"example\n")
            original = upstream_source_sha256(root)
            p.write_bytes(b"example\r\n")
            self.assertEqual(original, upstream_source_sha256(root))
            p.write_bytes(b"different\n")
            self.assertNotEqual(original, upstream_source_sha256(root))
            p.unlink()
            with self.assertRaises(FileNotFoundError):
                upstream_source_sha256(root)


if __name__ == "__main__":
    unittest.main()
