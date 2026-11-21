import unittest
from unittest.mock import patch

from models import predictor, trained_predictor as tp


class PlayerIdentityPipelineTests(unittest.TestCase):
    def test_full_model_does_not_restore_rejected_name_from_draft(self):
        draft = {"blue_top": "Aatrox", "blue_playername_top": "WrongPlayer"}
        data = {"live_draft": draft,
                "blue": {"name": "Blue", "players": [{"player": "WrongPlayer", "resolved_player": ""}]},
                "red": {"name": "Red", "players": []}}
        with patch.object(tp, "is_available", return_value=True), \
             patch.object(tp, "predict_full_from_draft", return_value={"blue_resolved": True}) as full:
            predictor.predict_win_prob(data)
        args = full.call_args.kwargs
        self.assertEqual(args["blue_player_names"], [""] * 5)
        self.assertEqual(args["draft"]["blue_playername_top"], "")
        self.assertEqual(draft["blue_playername_top"], "WrongPlayer")

    def test_team_model_preserves_explicit_unresolved_identity(self):
        data = {"blue": {"name": "Blue", "players": [{"player": "WrongPlayer", "resolved_player": ""}]},
                "red": {"name": "Red", "players": []}}
        with patch.object(tp, "_load"), patch.object(tp, "is_available", return_value=True), \
             patch.object(tp, "_resolve_player_id", side_effect=lambda name: "wrong-id" if name else None), \
             patch.object(tp, "predict") as predict:
            tp.predict_from_app_data(data)
        self.assertEqual(predict.call_args.kwargs["blue_player_ids"], [])


if __name__ == "__main__":
    unittest.main()
