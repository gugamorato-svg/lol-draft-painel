"""Synthetic checks only; never train or write production models."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from data_pipeline.training_timeline import build_training_timeline, temporal_split, infer_series_groups
from data_pipeline.temporal_context import TemporalContext
from scripts import train_model, train_team_only, train_prematch_sweep, train_pure_draft


AS_OF = "2026-10-15T00:00:00Z"


class CountingContext:
    def __init__(self):
        self.results = []

    def build_features(self, game):
        return {"known_wins": sum(g["blue_won"] for g in self.results),
                "known_games": len(self.results), "pre_match_raw": 0.0}

    def update(self, game):
        self.results.append(game)


def game(gid, start, duration=1800, win=1, pair=None):
    return {"gameid": gid, "date": start, "gamelength": duration, "blue_won": win,
            "blue_teamname": (pair or gid)+"_blue", "red_teamname": (pair or gid)+"_red",
            "league": "TEST", "year": 2026, "split": "synthetic"}


def synthetic_games(n=10):
    rows = []
    for i in range(n):
        row = game(str(i), pd.Timestamp("2026-01-01T00:00Z")+pd.Timedelta(hours=2*i), win=i % 2)
        row.update(blue_teamid=row["blue_teamname"], red_teamid=row["red_teamname"], patch="16.1", game=1)
        for side in ("blue", "red"):
            for role in ("top", "jng", "mid", "bot", "sup"):
                row[f"{side}_{role}"] = f"{side}_{role}_champ"
                row[f"{side}_player_{role}"] = f"{side}_{role}_id_{i % 2}"
                row[f"{side}_playername_{role}"] = f"{side}_{role}_shared_name"
        rows.append(row)
    return pd.DataFrame(rows)


class FakeEstimator:
    def fit(self, X, y):
        self.coef_ = np.zeros((1, X.shape[1]))
        self.intercept_ = np.zeros(1)
        return self

    def predict_proba(self, X):
        return np.tile([.5, .5], (len(X), 1))


class TrainingTimelineTests(unittest.TestCase):
    def replay(self, rows):
        return build_training_timeline(pd.DataFrame(rows), as_of=AS_OF, context_factory=CountingContext)

    def test_overlapping_and_equal_start_maps_do_not_see_unfinished_results(self):
        rows = [game("a", "2026-01-01T10:00Z"), game("b", "2026-01-01T10:00Z"),
                game("c", "2026-01-01T10:10Z"), game("d", "2026-01-01T10:30Z"),
                game("e", "2026-01-01T10:31Z")]
        t = self.replay(rows)
        self.assertEqual([0, 0, 0, 0, 2], t.features.known_games.tolist())
        self.assertEqual(5, len(t.context.results))

    def test_updates_follow_completion_order_and_final_queue_is_drained(self):
        t = self.replay([game("long", "2026-01-01T10:00Z", 3600),
                         game("short", "2026-01-01T10:10Z", 600),
                         game("after", "2026-01-01T10:30Z", 1800)])
        self.assertEqual([0, 0, 1], t.features.known_games.tolist())
        self.assertEqual(["short", "long", "after"], [g["gameid"] for g in t.context.results])

    def test_future_labels_and_missing_durations_never_reach_fit_or_snapshot(self):
        rows = [game("valid", "2026-01-01T10:00Z"), game("unknown", "2026-01-01T11:00Z", None),
                game("zero", "2026-01-01T11:00Z", 0), game("invalid", "2026-01-01T11:00Z", 20000),
                game("future", "2026-10-15T00:00Z")]
        t = self.replay(rows)
        self.assertEqual(["valid"], t.games.gameid.tolist())
        self.assertEqual(3, t.audit["excluded_invalid_duration"])
        self.assertEqual(1, t.audit["excluded_not_finished_as_of"])
        self.assertEqual(1, len(t.context.results))

    def test_future_outcome_change_cannot_change_earlier_features(self):
        rows = [game("a", "2026-01-01T10:00Z"), game("b", "2026-01-01T10:10Z"),
                game("c", "2026-01-01T11:00Z")]
        before = self.replay(rows)
        rows[1]["blue_won"] = 0
        after = self.replay(rows)
        pd.testing.assert_frame_equal(before.features.iloc[:2], after.features.iloc[:2])
        self.assertNotEqual(before.features.iloc[2].known_wins, after.features.iloc[2].known_wins)

    def test_final_snapshot_equals_sequential_context_when_no_games_overlap(self):
        games = synthetic_games()
        sequential = TemporalContext()
        expected = []
        for row in games.to_dict("records"):
            expected.append(sequential.build_features(row))
            sequential.update(row)
        t = build_training_timeline(games, as_of=AS_OF)
        pd.testing.assert_frame_equal(pd.DataFrame(expected), t.features)
        self.assertEqual(dict(sequential.team_rating), dict(t.context.team_rating))
        self.assertEqual(dict(sequential.champion_matchups), dict(t.context.champion_matchups))
        self.assertEqual(dict(sequential.player_champion), dict(t.context.player_champion))
        self.assertEqual(sequential.n_games_seen, t.context.n_games_seen)

    def test_undated_prior_is_never_loaded_even_when_legacy_flag_requests_it(self):
        with patch.object(TemporalContext, "with_prior", side_effect=AssertionError("undated prior forbidden")), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t = build_training_timeline(synthetic_games(3), use_prior=True, as_of=AS_OF)
        self.assertTrue(caught)
        self.assertEqual({}, t.context.prior_champion_matchups)
        self.assertEqual({}, t.context.prior_synergies)

    def test_bad_ids_dates_and_labels_fail_instead_of_silent_repair(self):
        cases = [[game("same", "2026-01-01"), game("same", "2026-01-02")],
                 [game("a", "not a date")], [game("a", "2026-01-01", win=2)]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.replay(rows)

    def test_split_purges_results_at_or_after_prediction_cutoff(self):
        rows = [game("a", "2026-01-01T08:00Z"), game("b", "2026-01-01T09:00Z"),
                game("ends_at_cut", "2026-01-01T09:50Z", 600),
                game("test", "2026-01-01T10:00Z"), game("later", "2026-01-01T11:00Z")]
        t = self.replay(rows)
        split = temporal_split(t.games, .6)
        self.assertEqual(["a", "b"], t.games.iloc[split.train].gameid.tolist())
        self.assertEqual(1, split.audit["purged_unfinished"])

    def test_series_crossing_cutoff_and_midnight_is_not_in_regression_fit(self):
        rows = [game("old", "2026-01-01T12:00Z"), game("map1", "2026-01-01T23:00Z", pair="series"),
                game("map2", "2026-01-02T00:00Z", pair="series"), game("other", "2026-01-02T02:00Z")]
        t = self.replay(rows)
        split = temporal_split(t.games, .5)
        self.assertEqual(["old"], t.games.iloc[split.train].gameid.tolist())
        self.assertEqual(1, split.audit["purged_same_series"])
        groups = infer_series_groups(t.games)
        self.assertEqual(groups[1], groups[2])

    def test_missing_id_on_one_map_and_side_swap_do_not_split_series(self):
        a, b = game("a", "2026-01-01T23:00Z", pair="s"), game("b", "2026-01-02T00:00Z", pair="s")
        a.update(blue_teamid="id1", red_teamid="id2")
        b["blue_teamname"], b["red_teamname"] = b["red_teamname"], b["blue_teamname"]
        groups = infer_series_groups(pd.DataFrame([a, b]))
        self.assertEqual(groups[0], groups[1])

    def test_development_never_scores_holdout_or_series_crossing_its_end(self):
        games = synthetic_games(10)
        games.loc[7:8, ["blue_teamname", "red_teamname"]] = ["sharedblue", "sharedred"]
        t = build_training_timeline(games, as_of=AS_OF, context_factory=CountingContext)
        split = temporal_split(t.games, .6, .8)
        self.assertEqual([6], split.test.tolist())
        self.assertLess(max(split.test), 8)

    def test_explicit_series_id_keeps_resumed_maps_together(self):
        games = pd.DataFrame([game("a", "2026-01-01"), game("b", "2026-01-05")])
        games["series_id"] = "series-resumed"
        self.assertEqual(*infer_series_groups(games))

    def test_legacy_trainer_cannot_overwrite_production(self):
        with patch.object(train_model, "load_years", side_effect=AssertionError("must reject before loading")):
            with self.assertRaises(ValueError):
                train_model.train([2026], False, train_model.DATA_DIR)

    def test_team_snapshot_keeps_all_homonym_ids_and_timing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                patch.object(train_team_only, "DATA_DIR", Path(tmp)), \
                patch.object(train_team_only, "load_years", return_value=synthetic_games()), \
                patch("sklearn.linear_model.LogisticRegression", side_effect=lambda **kw: FakeEstimator()), \
                patch("models.model_meta.registrar"):
            train_team_only.run([2026], False, False)
            with (Path(tmp)/"temporal_state.pkl").open("rb") as stream:
                saved = pickle.load(stream)
        self.assertEqual(["blue_top_id_0", "blue_top_id_1"], saved["player_name_to_ids"]["blue_top_shared_name"])
        self.assertIn("blue_top_shared_name", saved["player_name_to_id"])
        self.assertEqual({"league_in_rating": True}, saved["context_settings"])
        self.assertEqual(10, saved["n_games_seen"])
        self.assertEqual(saved["training_timeline"]["latest_result_available_at"], saved["latest_result_available_at"])

    def test_sweep_scores_development_only_and_fixed_cap_scores_once(self):
        t = build_training_timeline(synthetic_games(), as_of=AS_OF, context_factory=CountingContext)
        seen = []

        def evaluate(X, y, names, split):
            seen.append(split.test.tolist())
            return {"coefs": {}, "log_loss": .69, "brier": .25, "acc_test": .5,
                    "log_loss_raw": .69, "brier_raw": .25}

        with redirect_stdout(io.StringIO()), patch.object(train_prematch_sweep, "run_temporal", return_value=t), \
                patch.object(train_prematch_sweep, "_fit_eval", side_effect=evaluate), \
                patch("sys.argv", ["train_prematch_sweep"]):
            train_prematch_sweep.main()
        self.assertEqual(7, len(seen))  # baseline + six caps, all development only
        self.assertTrue(all(indices == [6, 7] for indices in seen))
        seen.clear()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                patch.object(train_prematch_sweep, "DATA_DIR", Path(tmp)), \
                patch.object(train_prematch_sweep, "run_temporal", return_value=t), \
                patch.object(train_prematch_sweep, "_fit_eval", side_effect=evaluate), \
                patch("models.model_meta.ajuste_final", side_effect=lambda X, y: FakeEstimator().fit(X, y)), \
                patch("models.model_meta.registrar"), \
                patch("sys.argv", ["train_prematch_sweep", "--save-cap", "1.6"]):
            train_prematch_sweep.main()
        self.assertEqual([[8, 9]], seen)

    def test_full_and_comfort_metrics_apply_deployed_probability_clip(self):
        t = build_training_timeline(synthetic_games(), as_of=AS_OF, context_factory=CountingContext)
        split = temporal_split(t.games)
        class ExtremeEstimator(FakeEstimator):
            def predict_proba(self, X):
                return np.tile([.005, .995], (len(X), 1))
        with patch("sklearn.linear_model.LogisticRegression", side_effect=lambda **kw: ExtremeEstimator()):
            full = train_prematch_sweep._fit_eval(t.features, t.games.blue_won.to_numpy(), ["known_wins"], split)
            draft = train_pure_draft.fit(t.features, t.games.blue_won.to_numpy(), ["known_wins"], split)
        np.testing.assert_allclose(full["p_test"], .9)
        np.testing.assert_allclose(draft["p"], .9)
        self.assertLess(full["log_loss"], full["log_loss_raw"])
        self.assertLess(draft["ll"], draft["ll_raw"])


if __name__ == "__main__":
    unittest.main()
