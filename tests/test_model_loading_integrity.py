import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import trained_predictor as tp
from tests.test_prediction_parity import history, synthetic_model


class ModelLoadingIntegrityTests(unittest.TestCase):
    def test_manual_elo_floor_does_not_modify_model_state(self):
        state = {'team_rating': {'dk': 1682}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'curated.csv'
            path.write_text('team,elo\nDplus Kia,1820\n', encoding='utf-8')
            with patch.object(tp, '_state', state), patch.object(tp, '_CURATED_PATH', path), \
                 patch.object(tp, '_resolve_team_id', return_value='dk'):
                tp._apply_curated_baselines()
        self.assertEqual(state['team_rating']['dk'], 1682)

    def test_nonfinite_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.csv'
            for value in ('nan', 'inf', '-inf'):
                path.write_text(f'feature,coefficient\n__intercept__,{value}\n', encoding='utf-8')
                with self.assertRaises(ValueError):
                    tp._read_coefs(path)

    def test_failed_load_cannot_leave_partial_model_available(self):
        with patch.object(tp, '_load_files', side_effect=ValueError('invalid')), \
             patch.multiple(tp, _state={'partial': True}, _coefs={'__intercept__': 0},
                            _full_coefs={}, _zero_coefs={}, _comfort_coefs={},
                            _ctx_for_lookup=object(), _model_version={'partial': True}):
            with self.assertRaises(ValueError):
                tp._load()
            self.assertIsNone(tp._state)
            self.assertIsNone(tp._coefs)
            self.assertIsNone(tp._ctx_for_lookup)
            self.assertEqual(tp._model_version, {})

    def test_unknown_explicit_id_cannot_fall_back_to_known_nickname(self):
        from tests.test_prediction_parity import game, live_draft
        with synthetic_model(history()):
            draft = live_draft(game())
            draft['blue_playerid_top'] = 'unknown-transfer-id'
            pred = tp.predict_full_from_draft('Blue Team', 'Red Team', draft)
        self.assertEqual(pred['players_resolved_blue'], 4)


if __name__ == '__main__':
    unittest.main()
