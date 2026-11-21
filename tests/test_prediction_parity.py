"""Synthetic train/inference parity checks. Never trains or loads production data.

Regression cases assert the corrected contract. All snapshots and model
coefficients are synthetic; the probability checks preserve full precision.
"""
import copy
import csv
import dataclasses
import math
import pickle
import tempfile
import unittest
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from api.player_resolver import PlayerResolver, norm_name
from data_pipeline.temporal_context import TemporalContext
from models import predictor
from models import trained_predictor as tp

ROLES = ("top", "jng", "mid", "bot", "sup")
BLUE = ("Aatrox", "Lee Sin", "Ahri", "Jinx", "Rell")
RED = ("Gnar", "Vi", "Syndra", "Ashe", "Braum")


def game():
    g = {"blue_teamid": "blue-id", "red_teamid": "red-id", "league": "EM",
         "patch": 16.1, "game": 2, "blue_won": 1,
         "blue_picks_draft": list(BLUE), "red_picks_draft": list(RED),
         "blue_bans": ["Gnar"], "red_bans": ["Aatrox"],
         "blue_golddiffat15": 1000, "red_golddiffat15": -1000,
         "blue_xpdiffat15": 500, "red_xpdiffat15": -500,
         "blue_csdiffat15": 20, "red_csdiffat15": -20,
         "blue_killsat15": 6, "red_killsat15": 3,
         "blue_deathsat15": 3, "red_deathsat15": 6,
         "blue_firstblood": 1, "red_firstblood": 0,
         "blue_firstdragon": 1, "red_firstdragon": 0,
         "blue_dragons": 3, "red_dragons": 1,
         "blue_opp_dragons": 1, "red_opp_dragons": 3}
    for side, champs in (("blue", BLUE), ("red", RED)):
        for role, champ in zip(ROLES, champs):
            g[f"{side}_{role}"] = champ
            g[f"{side}_player_{role}"] = f"{side}-{role}-id"
            g[f"{side}_kills_{role}"] = 5 if side == "blue" else 2
            g[f"{side}_deaths_{role}"] = 2 if side == "blue" else 5
            g[f"{side}_assists_{role}"] = 9 if side == "blue" else 4
    return g


def history():
    ctx = TemporalContext()
    for i in range(15):
        g = game()
        g["blue_won"] = int(i % 5 != 0)
        g["blue_kills_mid"] += i
        ctx.update(g)
    ctx.team_league_games["blue-id"] = {"LCK": 20}
    ctx.team_league_games["red-id"] = {"LEC": 20}
    ctx.league_rating.update({"LCK": .5, "LEC": -.25})
    return ctx


def live_draft(g):
    d = {k: copy.deepcopy(v) for k, v in g.items()
         if k in {"patch", "league", "blue_picks_draft", "red_picks_draft", "blue_bans", "red_bans"}
         or k in {f"{s}_{r}" for s in ("blue", "red") for r in ROLES}}
    d["patch"] = "16.10.1"
    d["game_number"] = g["game"]
    for side in ("blue", "red"):
        for role in ROLES:
            d[f"{side}_playername_{role}"] = f"{side}-{role}"
    return d


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, deque)):
        return [plain(x) for x in value]
    return value


@contextmanager
def synthetic_model(ctx, full=None, comfort=None, draft_only=None, names=None, cap=1.6):
    """Use the real snapshot loader against temporary synthetic files only."""
    snapshot = {f.name: plain(getattr(ctx, f.name)) for f in dataclasses.fields(ctx)}
    snapshot["golgg_champion_matchups"] = snapshot.pop("prior_champion_matchups")
    snapshot["golgg_synergies"] = snapshot.pop("prior_synergies")
    snapshot["team_name_to_id"] = {"Blue Team": "blue-id", "Red Team": "red-id"}
    snapshot["player_name_to_id"] = names or {f"{s}-{r}": f"{s}-{r}-id" for s in ("blue", "red") for r in ROLES}
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        snapshot_path = directory / "state.pkl"
        snapshot_path.write_bytes(pickle.dumps(snapshot))
        paths = {"_SNAPSHOT_PATH": snapshot_path, "_DATA": directory,
                 "_CURATED_PATH": directory / "absent_curated.csv"}
        for key, filename, coefs in (
            ("_COEF_PATH", "team.csv", {"__intercept__": 0.0}),
            ("_FULL_COEF_PATH", "full.csv", full or {"__intercept__": 0.0}),
            ("_COMFORT_COEF_PATH", "comfort.csv", comfort or {"__intercept__": 0.0}),
            ("_CHAMPS_ONLY_COEF_PATH", "draft.csv", draft_only or {"__intercept__": 0.0}),
        ):
            path = directory / filename
            with path.open("w", newline="", encoding="utf8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["feature", "coefficient"])
                writer.writerow(["__prematch_cap__", cap if key == "_FULL_COEF_PATH" else 0])
                writer.writerows(coefs.items())
            paths[key] = path
        with patch.multiple(tp, **paths, _state=None, _coefs=None, _full_coefs=None,
                            _zero_coefs=None, _comfort_coefs=None, _ctx_for_lookup=None,
                            _team_name_index=None, _team_core_index=None, _player_name_index=None,
                            _champ_role_counts=None, _model_version={}, _zero_model_type="", _prematch_cap=1.6), \
             patch.object(tp, "_version_info", return_value={"synthetic": True}):
            tp._load()
            yield


class PredictionParityTests(unittest.TestCase):
    def test_snapshot_roundtrip_preserves_every_context_feature(self):
        ctx = history()
        g = game()
        expected = ctx.build_features(g)
        with synthetic_model(ctx):
            actual = tp._ctx_for_lookup.build_features(g)
        self.assertEqual(set(actual), set(expected))
        for feature, value in expected.items():
            with self.subTest(feature=feature):
                self.assertAlmostEqual(actual[feature], value, places=12)

    def test_live_wrapper_preserves_all_shared_features_and_prematch_cap(self):
        ctx = history()
        expected = ctx.build_features(game())
        with synthetic_model(ctx, cap=.2):
            pred = tp.predict_full_from_draft("Blue Team", "Red Team", live_draft(game()))
        for feature, value in expected.items():
            with self.subTest(feature=feature):
                if feature == "pre_match_raw":
                    self.assertEqual(pred["features"]["pre_match_block"], round(max(-.2, min(.2, value)), 4))
                else:
                    self.assertEqual(pred["features"][feature], round(value, 4))

    def test_every_feature_uses_only_its_trained_weight(self):
        ctx = history()
        expected = ctx.build_features(game())
        expected["pre_match_block"] = max(-1.6, min(1.6, expected.pop("pre_match_raw")))
        with synthetic_model(ctx):
            for feature, value in expected.items():
                with self.subTest(feature=feature), patch.object(tp, "_full_coefs", {"__intercept__": .08, feature: .3}):
                    pred = tp.predict_full_from_draft("Blue Team", "Red Team", live_draft(game()))
                    probability = max(.1, min(.9, 1/(1+math.exp(-(.08+.3*value)))))
                    self.assertAlmostEqual(pred["win_prob_blue"], probability, places=12)

    def test_comfort_sample_kda_and_draft_models_apply_their_own_coefficients(self):
        ctx = history()
        feats = ctx.build_features(game())
        comfort = {"__intercept__": -.1, "player_champion_diff": .17,
                   "sample_size_confidence_diff": .31, "comfort_kda_diff": .43,
                   "player_champion_rate_diff": .23, "pre_match_block": 99}
        draft = {"__intercept__": .12, "lane_matchup_diff": .27, "pre_match_block": 99}
        with synthetic_model(ctx, comfort=comfort, draft_only=draft):
            pred = tp.predict_full_from_draft("Blue Team", "Red Team", live_draft(game()))
        for coefs, output in [(comfort, "win_prob_blue_comfort"), (draft, "win_prob_blue_no_pre")]:
            z = coefs["__intercept__"] + sum(w*feats[k] for k,w in coefs.items() if not k.startswith("__") and k!="pre_match_block")
            self.assertAlmostEqual(pred[output], max(.1,min(.9,1/(1+math.exp(-z)))), places=12)

    def test_probability_clamps_are_symmetric_for_all_three_models(self):
        for intercept, expected in [(30,.9),(-30,.1)]:
            with self.subTest(intercept=intercept), synthetic_model(history(),
                    full={"__intercept__":intercept}, comfort={"__intercept__":intercept},
                    draft_only={"__intercept__":intercept}):
                pred=tp.predict_full_from_draft("Blue Team","Red Team",live_draft(game()))
                for key in ["win_prob_blue","win_prob_blue_comfort","win_prob_blue_no_pre"]:
                    self.assertEqual(pred[key],expected)

    def test_patch_float_training_and_live_string_have_identical_score(self):
        ctx=history()
        expected=ctx.build_features(game())["patch_champion_wr_diff"]
        self.assertNotEqual(expected,0)
        with synthetic_model(ctx):
            for representation in ["16.10","16.10.1",16.1]:
                d=live_draft(game());d["patch"]=representation
                pred=tp.predict_full_from_draft("Blue Team","Red Team",d)
                self.assertEqual(pred["features"]["patch_champion_wr_diff"],round(expected,4))

    def test_unknown_players_have_neutral_comfort_sample_kda_and_form(self):
        ctx=history();d=live_draft(game());g=game()
        for side in ("blue","red"):
            for role in ROLES:
                d[f"{side}_playername_{role}"]=""
                g[f"{side}_player_{role}"]=""
        with synthetic_model(ctx):
            pred=tp.predict_full_from_draft("Blue Team","Red Team",d)
        for key in ["player_champion_diff","sample_size_confidence_diff","comfort_kda_diff",
                    "player_champion_rate_diff","player_form_diff"]:
            self.assertEqual(pred["features"][key],0)
        self.assertEqual(pred["players_resolved_blue"],0)

    def test_explicit_unresolved_reaches_real_full_model_as_neutral(self):
        ctx=history();d=live_draft(game())
        data={"live_draft":d}
        for side,name in [("blue","Blue Team"),("red","Red Team")]:
            data[side]={"name":name,"players":[{"role":role,"player":f"{side}-{role}","resolved_player":""} for role in ROLES]}
        with synthetic_model(ctx, full={"__intercept__":0,"player_champion_diff":1}):
            pred=predictor.predict_win_prob(data)
        self.assertEqual(pred["players_resolved_blue"],0)
        self.assertEqual(pred["players_resolved_red"],0)
        self.assertEqual(pred["win_prob_blue"],.5)

    def test_draft_pick_order_difference_has_no_effect_when_weight_is_excluded(self):
        ctx=history()
        d=live_draft(game());reverse=copy.deepcopy(d)
        reverse["blue_picks_draft"].reverse();reverse["red_picks_draft"].reverse()
        with synthetic_model(ctx,full={"__intercept__":0,"lane_matchup_diff":.7,"counter_diff":.5}):
            one=tp.predict_full_from_draft("Blue Team","Red Team",d)
            two=tp.predict_full_from_draft("Blue Team","Red Team",reverse)
        self.assertEqual(one["win_prob_blue"],two["win_prob_blue"])


class PredictionParityRegressionTests(unittest.TestCase):
    def test_team_only_partial_roster_neutralizes_each_missing_slot(self):
        ctx=history();ctx.player_form["blue-top-id"]=deque([1]*10,maxlen=10)
        g=game()
        for side in ("blue","red"):
            for role in ROLES:
                g[f"{side}_player_{role}"]=""
        g["blue_player_top"]="blue-top-id"
        expected=ctx.build_features(g)["player_form_diff"]
        self.assertAlmostEqual(expected,.1)
        with synthetic_model(ctx), patch.object(tp,"_coefs",{"__intercept__":0,"player_form_diff":1}):
            pred=tp.predict("Blue Team","Red Team",blue_player_ids=["blue-top-id"],red_player_ids=[])
        actual=next(f["signal"] for f in pred["factors"] if f["name"]=="Forma jogador")
        self.assertAlmostEqual(actual,expected)

    def test_team_only_real_zero_winrate_is_not_replaced_by_neutral_form(self):
        ctx=history();g=game()
        for role in ROLES:
            ctx.player_form[f"blue-{role}-id"]=deque([0]*10,maxlen=10)
            g[f"red_player_{role}"]=""
        expected=ctx.build_features(g)["player_form_diff"]
        self.assertAlmostEqual(expected,-.5)
        with synthetic_model(ctx), patch.object(tp,"_coefs",{"__intercept__":0,"player_form_diff":1}):
            pred=tp.predict("Blue Team","Red Team",blue_player_ids=[f"blue-{r}-id" for r in ROLES],red_player_ids=[])
        actual=next(f["signal"] for f in pred["factors"] if f["name"]=="Forma jogador")
        self.assertAlmostEqual(actual,expected)

    def test_live_league_name_matches_oracle_league_code(self):
        ctx=history();g=game();d=live_draft(g);d["league"]="EMEA Masters"
        expected=ctx.build_features(g)["league_champion_wr_diff"]
        self.assertNotEqual(expected,0)
        with synthetic_model(ctx):
            pred=tp.predict_full_from_draft("Blue Team","Red Team",d)
        self.assertEqual(pred["features"]["league_champion_wr_diff"],round(expected,4))

    def test_player_order_follows_role_not_position_in_list(self):
        data={"live_draft":live_draft(game()),"blue":{"name":"Blue Team","players":[
            {"role":"Mid","player":"blue-mid","resolved_player":"blue-mid"},
            {"role":"Top","player":"blue-top","resolved_player":""}]},
              "red":{"name":"Red Team","players":[]}}
        with patch.object(tp,"is_available",return_value=True), \
             patch.object(tp,"predict_full_from_draft",return_value={"blue_resolved":True}) as full:
            predictor.predict_win_prob(data)
        d=full.call_args.kwargs["draft"]
        self.assertEqual(d["blue_playername_top"],"")
        self.assertEqual(d["blue_playername_mid"],"blue-mid")

    def test_contextual_identity_must_not_be_replaced_by_global_homonym(self):
        ctx=history();g=game();g["blue_player_top"]="correct-id"
        ctx.player_champion[("correct-id","Aatrox")]={"games":50,"wins":50}
        ctx.player_champion[("wrong-id","Aatrox")]={"games":50,"wins":0}
        ctx.player_form["correct-id"]=deque([1]*10,maxlen=10)
        ctx.player_form["wrong-id"]=deque([0]*10,maxlen=10)
        class Oracle:
            players=pd.DataFrame([
                {"playername":"SharedNick","playerid":pid,"__playername_norm":"sharednick",
                 "teamname":team,"position":"top","__year":2026}
                for team,pid in [("Blue Team","correct-id"),("Other Team","wrong-id")]])
            def _resolve_team(self,team):return team
        with tempfile.TemporaryDirectory() as tmp:
            resolved=PlayerResolver(Oracle(),alias_path=Path(tmp)/"absent.json").resolve(
                "SharedNick",team="Blue Team",role="Top",year=2026)
        names={f"{s}-{r}":f"{s}-{r}-id" for s in ("blue","red") for r in ROLES}
        names["SharedNick"]="wrong-id"
        data={"live_draft":live_draft(game())}
        for side,name in [("blue","Blue Team"),("red","Red Team")]:
            data[side]={"name":name,"players":[{"role":role,"player":f"{side}-{role}",
                        "resolved_player":f"{side}-{role}"} for role in ROLES]}
        data["blue"]["players"][0].update(player="SharedNick",resolved_player=resolved.canonical,
                                         resolved_player_id=resolved.player_id)
        expected=ctx.build_features(g)["player_champion_diff"]
        with synthetic_model(ctx,names=names):
            pred=predictor.predict_win_prob(data)
        self.assertEqual(pred["features"]["player_champion_diff"],round(expected,4))

    def test_ambiguous_contextual_substring_is_unresolved(self):
        class Oracle:
            players=pd.DataFrame([
                {"playername":name,"__playername_norm":norm_name(name),"teamname":"Known Team",
                 "position":"top","__year":2026} for name in ["Knight","Knighty"]])
            def _resolve_team(self,team):return team
        with tempfile.TemporaryDirectory() as tmp:
            resolver=PlayerResolver(Oracle(),alias_path=Path(tmp)/"absent.json")
            result=resolver.resolve("Kni",team="Known Team",role="Top",year=2026)
        self.assertEqual(result.reason,"unresolved")


if __name__=="__main__":
    unittest.main()
