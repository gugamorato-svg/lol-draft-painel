"""Predictor treinado: usa modelo team-only (4 features) + snapshot temporal.

Carrega:
  data/model_team_only.csv  -- coeficientes aprendidos
  data/temporal_state.pkl   -- snapshot do TemporalContext apos treino

Expoe:
  is_available()                 -- True se modelo + snapshot existem
  predict(blue_name, red_name, blue_players, red_players) -> dict
"""

from __future__ import annotations

import csv
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DATA = Path(__file__).parent.parent / "data"
_COEF_PATH      = _DATA / "model_team_only.csv"
_FULL_COEF_PATH = _DATA / "model_coefficients.csv"
_CHAMPS_ONLY_COEF_PATH = _DATA / "model_coefficients_champs_only.csv"  # draft puro: sem time/player
# draft + COMFORT do jogador (WR no campeao, volume nele, % do tempo nele). Sem forca do
# time. Foi o modelo de draft que melhor se sustentou na validacao de 2026,
# acima do so-draft na maioria dos meses.
_COMFORT_COEF_PATH = _DATA / "model_coefficients_zero.csv"
# 29/10 (dono): os modelos "draft" oficiais ficaram SEM a forca da liga (informacao do time).
# As versoes COM ela viram probabilidades a parte (estrategias "draft + forca da liga" e insumo
# da candidata do teste prospectivo).
_CHAMPS_LIGA_COEF_PATH = _DATA / "model_coefficients_champs_only_liga.csv"
_COMFORT_LIGA_COEF_PATH = _DATA / "model_coefficients_zero_liga.csv"
_SNAPSHOT_PATH  = _DATA / "temporal_state.pkl"
_CURATED_PATH   = _DATA / "curated_baselines.csv"  # override manual de Elo (domain)

# Constantes que precisam bater com o temporal_context
_ELO_CLAMP = 250.0
_SMOOTHING_PRIOR_H2H = 10
_SCORE_SCALE_H2H = 10.0
_SCORE_CAP_H2H = 2.0
_TEAM_FORM_DEFAULT   = 0.5
_PLAYER_FORM_DEFAULT = 0.5
_CHAMPION_META_WR_PRIOR = 12.0
_CHAMPION_META_WR_WEIGHT = 0.30


# ---------- lazy loading ----------

_state: Optional[Dict[str, Any]] = None
_coefs: Optional[Dict[str, float]] = None
_full_coefs: Optional[Dict[str, float]] = None
_zero_coefs: Optional[Dict[str, float]] = None    # alternativo exibido na segunda barra
_comfort_coefs: Optional[Dict[str, float]] = None  # draft + comfort
_champs_liga_coefs: Optional[Dict[str, float]] = None   # draft + forca da liga (29/10)
_comfort_liga_coefs: Optional[Dict[str, float]] = None  # draft + comfort + forca da liga (29/10)
_zero_model_type: str = ""
_zero_prematch_cap: float = 0.0
_model_version: Dict[str, Any] = {}   # 13/10: versao dos pesos CARREGADOS (vai em cada captura)
_team_name_index: Optional[Dict[str, str]] = None    # lower-norm -> id
_player_name_index: Optional[Dict[str, str]] = None  # lower-norm -> id
_ctx_for_lookup = None  # TemporalContext reconstituido do snapshot
_champ_role_counts: Optional[Dict[str, Dict[str, float]]] = None
_prematch_cap: float = 1.6  # cap do pre_match_block (lido do CSV se presente)

# Pesos internos do pre_match_block (devem casar com PREMATCH_WEIGHTS no
# temporal_context). Usados para reconstruir o bloco na predicao.
_PREMATCH_WEIGHTS = {
    "player_form_diff": 1.04,
    "rating_diff":      0.87,
    "recent_form_diff": -0.26,
    "h2h_diff":         0.14,
}

_ROLES = ("top", "jng", "mid", "bot", "sup")
_ROLE_LABEL = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}

_ENHANCED_WEIGHTS: Dict[str, float] = {
    "__intercept__": 0.0259,
    "champion_meta_diff": 0.05,
    "player_comfort_diff": 0.08,
    "role_fit_diff": 0.02,
    "draft_balance_diff": 0.22,
    "lane_matchup_diff": 0.12,
    "rating_diff": 0.05,
    "ban_score_diff": 0.04,
    "synergy_diff": 0.60,
    "counter_diff": 0.60,
    "last_pick_counter_diff": 0.08,
    "player_form_diff": 0.04,
    "head_to_head_diff": 0.06,
    "recent_form_diff": 0.08,
}

# Trava da previsao ao vivo. Era 0.20-0.90 (ASSIMETRICA: vermelho no maximo 80%, azul
# 90%). No backtest os extremos sao calibrados (previsto 16% -> real 13.5%; 92% -> 96%),
# entao a trava so corta o absurdo, igual dos dois lados.
_P_MIN, _P_MAX = 0.10, 0.90

_COEF_ALIASES = {
    "player_comfort_diff": "player_champion_diff",
    "head_to_head_diff": "h2h_diff",
}


def _live_intercept() -> float:
    if _full_coefs and "__intercept__" in _full_coefs:
        return _full_coefs["__intercept__"]
    return _ENHANCED_WEIGHTS["__intercept__"]


def _live_weight(feature: str) -> float:
    if _full_coefs:
        if feature in _full_coefs:
            return _full_coefs[feature]
        alias = _COEF_ALIASES.get(feature)
        if alias and alias in _full_coefs:
            return _full_coefs[alias]
    return _ENHANCED_WEIGHTS.get(feature, 0.0)


def _live_feature_order() -> List[str]:
    # Com modelo treinado, SO as features dele. Antes somava role_fit (0.02) e
    # draft_balance (0.22) com peso chutado, que nunca foram treinados nem entram no
    # backtest: mexiam ate 3pp na previsao ao vivo (auditoria 23/09).
    if _full_coefs:
        return [k for k in _full_coefs.keys() if k != "__intercept__"]
    return [k for k in _ENHANCED_WEIGHTS.keys() if k != "__intercept__"]

# Curadoria inicial para champs comuns no competitivo. Quando um campeao nao
# estiver aqui, as features historicas (counter/synergy/meta/role fit) continuam
# funcionando e o draft_balance fica neutro para esse pick.
_CHAMPION_TRAITS: Dict[str, Dict[str, float]] = {
    "aatrox": {"ad": 1, "frontline": 0.6, "cc": 0.4, "side": 0.8, "teamfight": 0.5, "scaling": 0.6},
    "ahri": {"ap": 1, "cc": 0.5, "pick": 0.8, "waveclear": 0.6, "early": 0.6},
    "akali": {"ap": 0.8, "ad": 0.2, "pick": 0.8, "side": 0.8, "scaling": 0.5},
    "alistar": {"engage": 0.9, "frontline": 0.7, "cc": 0.9, "peel": 0.7, "teamfight": 0.8},
    "ambessa": {"ad": 1, "engage": 0.5, "side": 0.8, "early": 0.6, "frontline": 0.4},
    "annie": {"ap": 1, "engage": 0.7, "cc": 0.8, "pick": 0.6, "teamfight": 0.7},
    "aphelios": {"ad": 1, "scaling": 0.9, "teamfight": 0.8, "objective": 0.5},
    "azir": {"ap": 1, "waveclear": 0.9, "scaling": 0.9, "teamfight": 0.9, "disengage": 0.5},
    "bard": {"ap": 0.2, "engage": 0.7, "cc": 0.8, "pick": 0.8, "roam": 1.0, "peel": 0.4},
    "braum": {"frontline": 0.7, "cc": 0.7, "peel": 1.0, "disengage": 0.8, "teamfight": 0.7},
    "caitlyn": {"ad": 1, "early": 0.9, "objective": 0.7, "waveclear": 0.6, "siege": 0.9},
    "corki": {"ap": 0.6, "ad": 0.4, "waveclear": 0.8, "scaling": 0.8, "poke": 0.7},
    "ezreal": {"ad": 0.8, "ap": 0.2, "poke": 0.8, "scaling": 0.6, "safe": 0.9},
    "gnar": {"ad": 1, "frontline": 0.5, "engage": 0.7, "cc": 0.8, "side": 0.7, "teamfight": 0.8},
    "gragas": {"ap": 1, "engage": 0.7, "frontline": 0.6, "cc": 0.8, "disengage": 0.8, "teamfight": 0.7},
    "gwen": {"ap": 1, "side": 0.9, "scaling": 0.9, "objective": 0.5},
    "hwei": {"ap": 1, "waveclear": 0.9, "cc": 0.5, "poke": 0.8, "teamfight": 0.7, "scaling": 0.7},
    "jarvaniv": {"ad": 1, "engage": 0.9, "frontline": 0.7, "cc": 0.7, "early": 0.8, "teamfight": 0.8},
    "jax": {"ad": 1, "side": 1.0, "scaling": 0.8, "frontline": 0.4},
    "jinx": {"ad": 1, "scaling": 0.9, "teamfight": 0.9, "objective": 0.7},
    "jhin": {"ad": 1, "pick": 0.7, "cc": 0.5, "early": 0.6, "objective": 0.4},
    "kaisa": {"ad": 0.8, "ap": 0.2, "scaling": 0.8, "pick": 0.5, "teamfight": 0.7},
    "kalista": {"ad": 1, "early": 1.0, "objective": 0.9, "engage": 0.4},
    "karma": {"ap": 0.7, "poke": 0.7, "peel": 0.8, "waveclear": 0.6, "early": 0.7},
    "ksante": {"ad": 0.6, "frontline": 0.9, "cc": 0.7, "side": 0.7, "teamfight": 0.7, "scaling": 0.6},
    "leblanc": {"ap": 1, "pick": 0.9, "poke": 0.6, "side": 0.6, "early": 0.7},
    "leesin": {"ad": 1, "early": 1.0, "engage": 0.5, "pick": 0.7, "objective": 0.5},
    "leona": {"engage": 1.0, "frontline": 0.7, "cc": 1.0, "early": 0.8, "teamfight": 0.8},
    "lucian": {"ad": 1, "early": 0.9, "side": 0.5, "objective": 0.5},
    "lulu": {"ap": 0.4, "peel": 1.0, "scaling": 0.8, "disengage": 0.7},
    "maokai": {"ap": 0.4, "engage": 0.8, "frontline": 0.8, "cc": 0.9, "objective": 0.8, "teamfight": 0.8},
    "nautilus": {"engage": 1.0, "frontline": 0.7, "cc": 1.0, "pick": 0.8, "teamfight": 0.8},
    "neeko": {"ap": 1, "engage": 0.8, "cc": 0.8, "waveclear": 0.6, "teamfight": 0.8},
    "nocturne": {"ad": 1, "engage": 0.8, "pick": 0.8, "early": 0.6, "side": 0.6},
    "orianna": {"ap": 1, "waveclear": 0.8, "scaling": 0.8, "teamfight": 0.9, "peel": 0.5},
    "ornn": {"frontline": 1.0, "engage": 0.8, "cc": 0.9, "scaling": 1.0, "teamfight": 0.9},
    "poppy": {"ad": 0.7, "frontline": 0.8, "cc": 0.8, "disengage": 0.9, "early": 0.6},
    "rakan": {"ap": 0.3, "engage": 0.9, "cc": 0.8, "peel": 0.6, "teamfight": 0.9},
    "rell": {"engage": 1.0, "frontline": 0.8, "cc": 1.0, "teamfight": 0.9},
    "renataglasc": {"ap": 0.4, "peel": 0.9, "disengage": 0.8, "teamfight": 0.8, "scaling": 0.6},
    "renekton": {"ad": 1, "frontline": 0.5, "early": 1.0, "side": 0.7, "cc": 0.4},
    "rumble": {"ap": 1, "teamfight": 0.9, "early": 0.7, "objective": 0.7, "waveclear": 0.5},
    "ryze": {"ap": 1, "waveclear": 0.8, "side": 0.8, "scaling": 0.8, "teamfight": 0.5},
    "sejuani": {"ap": 0.3, "frontline": 0.9, "engage": 0.8, "cc": 0.9, "objective": 0.7, "teamfight": 0.8},
    "senna": {"ad": 0.8, "scaling": 0.9, "poke": 0.7, "objective": 0.4},
    "sivir": {"ad": 1, "waveclear": 1.0, "scaling": 0.8, "teamfight": 0.7},
    "skarner": {"frontline": 0.9, "engage": 0.8, "cc": 0.9, "pick": 0.7, "objective": 0.7},
    "smolder": {"ad": 0.8, "ap": 0.2, "scaling": 1.0, "waveclear": 0.7, "teamfight": 0.8},
    "syndra": {"ap": 1, "waveclear": 0.7, "pick": 0.7, "scaling": 0.7, "teamfight": 0.6},
    "tahmkench": {"frontline": 0.8, "peel": 1.0, "disengage": 0.8, "scaling": 0.5},
    "taliyah": {"ap": 1, "waveclear": 0.8, "cc": 0.7, "pick": 0.6, "roam": 0.8},
    "tristana": {"ad": 1, "early": 0.8, "objective": 0.8, "scaling": 0.7, "side": 0.6},
    "twistedfate": {"ap": 0.7, "ad": 0.3, "pick": 0.8, "roam": 1.0, "side": 0.7},
    "varus": {"ad": 0.8, "ap": 0.2, "poke": 0.9, "early": 0.8, "objective": 0.6},
    "vi": {"ad": 1, "engage": 0.9, "pick": 0.8, "early": 0.7, "objective": 0.5},
    "viego": {"ad": 1, "teamfight": 0.7, "scaling": 0.6, "objective": 0.6},
    "xayah": {"ad": 1, "scaling": 0.8, "teamfight": 0.8, "safe": 0.8},
    "xinzhao": {"ad": 1, "early": 0.9, "frontline": 0.6, "objective": 0.8, "engage": 0.5},
    "yone": {"ad": 0.8, "ap": 0.2, "engage": 0.7, "side": 0.8, "scaling": 0.7, "teamfight": 0.7},
    "zeri": {"ad": 1, "scaling": 0.9, "teamfight": 0.8, "safe": 0.7},
    "ziggs": {"ap": 1, "waveclear": 1.0, "siege": 1.0, "objective": 0.7, "scaling": 0.6},
}


def _version_info() -> Dict[str, Any]:
    """Versao do que foi carregado: hash do conteudo dos pesos e do snapshot, e a
    data do ultimo jogo de treino de cada arquivo (data/model_meta.json). 13/10 (auditoria): os
    pesos ficam em memoria ate o coletor reiniciar (ate 12 h) -> sem isto nao dava p/ saber qual
    versao gerou cada sinal."""
    import hashlib
    import json
    from datetime import datetime, timezone
    h = hashlib.sha256()
    for p in (_SNAPSHOT_PATH, _COEF_PATH, _FULL_COEF_PATH, _CHAMPS_ONLY_COEF_PATH, _COMFORT_COEF_PATH,
              _CHAMPS_LIGA_COEF_PATH, _COMFORT_LIGA_COEF_PATH):
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    try:
        meta = json.loads((_DATA / "model_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    from models.model_meta import upstream_source_sha256
    train_dates = {p.name: (meta.get(p.name) or {}).get("treino_ate")
                   for p in (_COEF_PATH, _FULL_COEF_PATH, _COMFORT_COEF_PATH, _CHAMPS_ONLY_COEF_PATH,
                             _CHAMPS_LIGA_COEF_PATH, _COMFORT_LIGA_COEF_PATH)}
    train_dates[_SNAPSHOT_PATH.name] = (_state or {}).get("latest_result_available_at")
    return {"versao": h.hexdigest()[:10],
            "source_sha256": upstream_source_sha256(),
            "carregado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "treino_ate": train_dates}


def model_version() -> Dict[str, Any]:
    _load()
    return dict(_model_version)


def _load() -> None:
    """Nao deixa uma leitura incompleta parecer um modelo pronto no proximo acesso."""
    global _state, _coefs, _full_coefs, _zero_coefs, _comfort_coefs, _ctx_for_lookup
    global _champs_liga_coefs, _comfort_liga_coefs
    try:
        _load_files()
    except Exception:
        _state = _coefs = _full_coefs = _zero_coefs = _comfort_coefs = None
        _champs_liga_coefs = _comfort_liga_coefs = None
        _ctx_for_lookup = None
        _model_version.clear()
        raise


def _load_files() -> None:
    """Carrega snapshot + coefs uma unica vez."""
    global _state, _coefs, _full_coefs, _zero_coefs, _zero_model_type, _team_name_index, _player_name_index
    global _comfort_coefs, _champs_liga_coefs, _comfort_liga_coefs
    global _ctx_for_lookup, _champ_role_counts
    if _state is not None:
        return

    if not _SNAPSHOT_PATH.exists() or not _COEF_PATH.exists():
        return

    with open(_SNAPSHOT_PATH, "rb") as f:
        _state = pickle.load(f)

    _coefs = _read_coefs(_COEF_PATH)
    if _FULL_COEF_PATH.exists():
        _full_coefs = _read_coefs(_FULL_COEF_PATH)
    # Segunda barra: usa somente o modelo realmente draft-only/champs-only.
    # O antigo model_coefficients_zero.csv removia pre-jogo, mas ainda vazava
    # forca por identidade dos players via comfort/sample_size, entao nao entra
    # como fallback.
    if _CHAMPS_ONLY_COEF_PATH.exists():
        _zero_coefs = _read_coefs(_CHAMPS_ONLY_COEF_PATH, set_global_cap=False)
        _zero_model_type = "champs_only"
    if _COMFORT_COEF_PATH.exists():
        _comfort_coefs = _read_coefs(_COMFORT_COEF_PATH, set_global_cap=False)
    if _CHAMPS_LIGA_COEF_PATH.exists():
        _champs_liga_coefs = _read_coefs(_CHAMPS_LIGA_COEF_PATH, set_global_cap=False)
    if _COMFORT_LIGA_COEF_PATH.exists():
        _comfort_liga_coefs = _read_coefs(_COMFORT_LIGA_COEF_PATH, set_global_cap=False)
    _model_version.clear()
    _model_version.update(_version_info())

    # Indexes de nomes -> ids (lowercase normalizado)
    global _team_core_index
    _team_name_index = {}
    for raw_name, tid in _state.get("team_name_to_id", {}).items():
        key = _norm_name(raw_name)
        _team_name_index[key] = tid
    _team_core_index = _build_team_core_index(_state.get("team_name_to_id", {}))

    # Homonimos nunca podem herdar o ultimo ID gravado no dicionario.
    names = _state.get("player_name_to_ids")
    if names is None:
        names = {name: [pid] for name, pid in _state.get("player_name_to_id", {}).items()}
    ids_by_name = {}
    for raw_name, ids in names.items():
        ids_by_name.setdefault(_norm_name(raw_name), set()).update(
            str(pid) for pid in ids if str(pid).lower() not in ("", "nan", "none"))
    _player_name_index = {name: next(iter(ids)) for name, ids in ids_by_name.items() if len(ids) == 1}

    # Reconstitui TemporalContext do snapshot para reaproveitar build_features
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from collections import defaultdict, deque
        from data_pipeline.temporal_context import TemporalContext, FORM_WINDOW
        ctx = TemporalContext()
        for key in ("league_in_rating", "home_decay", "league_elo", "league_parents"):
            if key in _state.get("context_settings", {}):
                setattr(ctx, key, _state["context_settings"][key])
        # Repopula com o estado salvo
        for k, v in _state["champion_matchups"].items():
            ctx.champion_matchups[k] = dict(v)
        for k, v in _state["synergies"].items():
            ctx.synergies[k] = dict(v)
        for k, v in _state.get("golgg_champion_matchups", {}).items():
            ctx.prior_champion_matchups[k] = dict(v)
        for k, v in _state.get("golgg_synergies", {}).items():
            ctx.prior_synergies[k] = dict(v)
        for k, v in _state["team_rating"].items():
            ctx.team_rating[k] = v
        for k, lst in _state["team_form"].items():
            ctx.team_form[k] = deque(lst, maxlen=FORM_WINDOW)
        for k, lst in _state["player_form"].items():
            ctx.player_form[k] = deque(lst, maxlen=FORM_WINDOW)
        for k, v in _state["h2h"].items():
            ctx.h2h[k] = dict(v)
        for k, v in _state["champion_pickrate"].items():
            ctx.champion_pickrate[k] = v
        for k, v in _state.get("champion_recent_meta", {}).items():
            ctx.champion_recent_meta[k] = dict(v)
        for k, v in _state.get("player_champion", {}).items():
            ctx.player_champion[k] = dict(v)
        for k, v in _state.get("team_early_profile", {}).items():
            ctx.team_early_profile[k] = dict(v)
        for k, v in _state.get("team_objective_profile", {}).items():
            ctx.team_objective_profile[k] = dict(v)
        for k, v in _state.get("team_side", {}).items():
            ctx.team_side[k] = dict(v)
        for k, v in _state.get("team_series", {}).items():
            ctx.team_series[k] = dict(v)
        for k, v in _state.get("team_champion", {}).items():
            ctx.team_champion[k] = dict(v)
        for k, v in _state.get("patch_champion", {}).items():
            ctx.patch_champion[k] = dict(v)
        for k, v in _state.get("league_rating", {}).items():
            ctx.league_rating[k] = float(v)
        ctx.cross_league_games = list(_state.get("cross_league_games", []))
        for k, v in _state.get("team_league_games", {}).items():
            ctx.team_league_games[k].update(v)
        for k, v in _state.get("league_champion", {}).items():
            ctx.league_champion[k] = dict(v)
        for k, v in _state.get("player_total_games", {}).items():
            ctx.player_total_games[k] = int(v)
        for k, v in _state.get("player_champion_kda", {}).items():
            ctx.player_champion_kda[k] = dict(v)
        for k, v in _state.get("champion_role_kda", {}).items():
            ctx.champion_role_kda[k] = dict(v)
        ctx.n_games_seen = _state.get("n_games_seen", 0)
        _ctx_for_lookup = ctx
        _champ_role_counts = _build_champion_role_counts(_state.get("champion_matchups", {}))
    except Exception:
        # Sem o contexto inteiro nao existe garantia de paridade de features.
        raise

    # O Elo vem exclusivamente do mesmo estado historico usado pelo treino.
    # O antigo piso manual nao tinha data de expiracao nem avaliacao temporal.


def _apply_curated_baselines() -> None:
    """Pisos manuais desativados: nao integravam treino/avaliacao temporal."""
    return


def _read_coefs(path: Path, set_global_cap: bool = True) -> Dict[str, float]:
    """Le coeficientes do CSV. set_global_cap=False para nao sobrescrever
    o _prematch_cap global (usado ao carregar o modelo alternativo)."""
    global _prematch_cap
    out: Dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            feat = row["feature"]
            val = float(row["coefficient"])
            if not math.isfinite(val):
                raise ValueError(f"Coeficiente nao finito em {path.name}: {feat}")
            if feat == "__prematch_cap__":
                if val < 0:
                    raise ValueError(f"Cap negativo em {path.name}")
                if set_global_cap:
                    _prematch_cap = val
                continue
            if feat in out:
                if math.isclose(out[feat], val, rel_tol=1e-12, abs_tol=1e-12):
                    continue
                raise ValueError(
                    f"Coeficiente duplicado conflitante em {path.name}: {feat}"
                )
            out[feat] = val
    return out


def _build_champion_role_counts(matchups: Dict[Any, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, float]] = {}
    for key, rec in matchups.items():
        if not isinstance(key, tuple) or len(key) != 3:
            continue
        champ, _opp, role = key
        if role not in _ROLES:
            continue
        games = float(rec.get("games", 0) or 0)
        if games <= 0:
            continue
        champ_key = _champ_key(champ)
        counts.setdefault(champ_key, {r: 0.0 for r in _ROLES})
        counts[champ_key][role] += games
    return counts


def is_available() -> bool:
    _load()
    return _state is not None and _coefs is not None


def stats() -> Dict[str, int]:
    _load()
    if not _state:
        return {"available": 0}
    return {
        "available":      1,
        "games_seen":     _state.get("n_games_seen", 0),
        "teams_known":    len(_state.get("team_rating", {})),
        "players_known":  len(_state.get("player_form", {})),
        "h2h_pairs":      len(_state.get("h2h", {})),
        "oracle_matchups": len(_state.get("champion_matchups", {})),
        "oracle_synergies": len(_state.get("synergies", {})),
        "golgg_matchups": len(_state.get("golgg_champion_matchups", {})),
        "golgg_synergies": len(_state.get("golgg_synergies", {})),
        "champion_recent_meta": len(_state.get("champion_recent_meta", {})),
        "team_early_profiles": len(_state.get("team_early_profile", {})),
        "team_objective_profiles": len(_state.get("team_objective_profile", {})),
        "team_side_profiles": len(_state.get("team_side", {})),
        "team_series_profiles": len(_state.get("team_series", {})),
        "team_champion_pairs": len(_state.get("team_champion", {})),
        "patch_champion_pairs": len(_state.get("patch_champion", {})),
    }


# ---------- name resolution ----------

def _norm_name(s: str) -> str:
    if not s:
        return ""
    return "".join(c for c in s.lower().strip() if c.isalnum())


# Nome da Riot/outras fontes -> nome do Oracle, quando nem o trecho resolve (normalizado).
_TEAM_ALIASES = {
    "edwardgamingyouthteam": "EDG Youth Team",
    "dnschallengers": "DN SOOPers Challengers",
    "ktchallengers": "KT Rolster Challengers",
    "redkalunga": "RED Canids",
    "øtf": "Only The Family",
    "gamespacemc": "Gamespace Mediterranean College Esports",
}
# Palavras de CATEGORIA: time de base nunca pode virar o time principal (e vice-versa).
# Bug de 23/09: "Edward Gaming Youth Team" (LDL) casava com "Edward Gaming" (LPL) pelo
# trecho -> Elo e forca de liga do time errado na previsao.
_TEAM_TIER_WORDS = ("academy", "youth", "junior", "challengers", "rookies", "young", "academia")


def _same_tier(query_key: str, candidate_key: str) -> bool:
    return all((w in query_key) == (w in candidate_key) for w in _TEAM_TIER_WORDS)


_TEAM_GENERIC_WORDS = {"esports", "esport", "e", "sports", "gaming", "team", "club", "the", "gg"}
_team_core_index: Optional[Dict[str, str]] = None   # nucleo do nome -> id (so se UNICO)


def _team_core(name: str) -> str:
    """Nucleo do nome: sem acento e sem palavra generica ("FURIA Esports" -> "furia",
    "T1 Academy" e "T1 Esports Academy" -> "t1academy"). Categoria (academy...) fica."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name or "").lower().replace("ø", "o").replace("æ", "ae"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(w for w in re.split(r"[^a-z0-9]+", s) if w and w not in _TEAM_GENERIC_WORDS)


def _build_team_core_index(names: Dict[str, str]) -> Dict[str, str]:
    seen: Dict[str, set] = {}
    for raw, tid in names.items():
        core = _team_core(raw)
        if len(core) >= 2:
            seen.setdefault(core, set()).add(tid)
    return {c: next(iter(ids)) for c, ids in seen.items() if len(ids) == 1}


def _resolve_team_id(team_name: str) -> Optional[str]:
    """Nome do time -> teamid OE: exato, apelido, nucleo do nome (unico) e so entao trecho
    (sem nunca trocar time de base pelo principal)."""
    alias = _TEAM_ALIASES.get(_norm_name(team_name))
    name = alias or team_name
    if _team_name_index and _norm_name(name) in _team_name_index:
        return _team_name_index[_norm_name(name)]
    core = _team_core(name)
    if _team_core_index and core in _team_core_index:
        return _team_core_index[core]
    return _resolve_in_index(name, _team_name_index, allow=_same_tier)


def _resolve_player_id(player_name: str) -> Optional[str]:
    """Nome exato e nao ambiguo; resolucao contextual ocorre no worker."""
    return (_player_name_index or {}).get(_norm_name(player_name))


def _known_player_id(value: Any) -> str:
    """Aceita ID contextual apenas se existir no historico carregado."""
    pid = str(value or "")
    state = _state or {}
    return pid if pid and (pid in state.get("player_form", {})
                           or pid in state.get("player_total_games", {})) else ""


def _canonical_league(value: str) -> str:
    name = str(value or "").strip()
    # Nome da mesma competicao nos feeds Riot e Oracle (nao liga de origem).
    alias = {"emeamasters": "EM", "europeanmasters": "EM"}
    if _norm_name(name) in alias:
        return alias[_norm_name(name)]
    ctx = _ctx_for_lookup
    known = {lg for lg, _champ in ctx.league_champion} if ctx is not None else set()
    matches = [lg for lg in known if _norm_name(lg) == _norm_name(name)]
    return matches[0] if len(matches) == 1 else name


_MIN_SUBSTRING_SIMILARITY = 0.5


def _resolve_in_index(name: str, index: Optional[Dict[str, str]], allow=None) -> Optional[str]:
    """allow(chave_pedida, chave_candidata) -> bool filtra candidatos por trecho."""
    if not name or not index:
        return None
    key = _norm_name(name)
    if not key:
        return None
    if key in index:
        return index[key]
    # Match por trecho (qualquer direcao): o MAIS PARECIDO, nunca o 1o da lista.
    # Bug de 23/09: "DK Wayne" pegava o jogador "Way", "SGW Aomine" o "Min" e
    # "VKS SUPERCLEBER" o "Leb" (1 em cada ~9 jogadores ao vivo trocado por outro).
    best, best_sim = None, 0.0
    for k, v in index.items():
        if allow is not None and not allow(key, k):
            continue
        if len(k) >= 3 and (k in key or key in k):
            sim = min(len(k), len(key)) / max(len(k), len(key))
            if sim > best_sim:
                best, best_sim = v, sim
    return best if best_sim >= _MIN_SUBSTRING_SIMILARITY else None


# ---------- score helpers (espelham temporal_context) ----------

def _smoothed_h2h(wins: float, games: int) -> float:
    if games <= 0:
        return 0.0
    centered = wins - 0.5 * games
    val = centered / (games + _SMOOTHING_PRIOR_H2H) * _SCORE_SCALE_H2H
    return max(-_SCORE_CAP_H2H, min(_SCORE_CAP_H2H, val))


def _h2h_score(state, team_a: str, team_b: str) -> float:
    if not team_a or not team_b:
        return 0.0
    a, b = (team_a, team_b) if team_a <= team_b else (team_b, team_a)
    swapped = a != team_a
    rec = state["h2h"].get((a, b))
    if not rec or rec["games"] <= 0:
        return 0.0
    wins_a = rec["wins_a"] if not swapped else (rec["games"] - rec["wins_a"])
    return _smoothed_h2h(wins_a, rec["games"])


def _team_form_wr(state, team_id: str) -> float:
    if not team_id:
        return _TEAM_FORM_DEFAULT
    dq = state["team_form"].get(team_id)
    if not dq:
        return _TEAM_FORM_DEFAULT
    return sum(dq) / len(dq)


def _player_form_wr(state, player_id: str) -> float:
    if not player_id:
        return _PLAYER_FORM_DEFAULT
    dq = state["player_form"].get(player_id)
    if not dq:
        return _PLAYER_FORM_DEFAULT
    return sum(dq) / len(dq)


def _league_pts(blue_id: Optional[str], red_id: Optional[str]) -> float:
    """Forca da liga de casa (azul - vermelho) em pts de Elo, como no treino (24/09)."""
    ctx = _ctx_for_lookup
    if ctx is None or not blue_id or not red_id or not getattr(ctx, "league_in_rating", False):
        return 0.0
    return ctx.league_elo_offset(blue_id, red_id)


def _liga_txt(pts: float) -> str:
    return f" (liga {pts:+.0f})" if abs(pts) >= 1 else ""


def _sigmoid(x: float) -> float:
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# ---------- predict ----------

def predict(
    blue_name: str,
    red_name: str,
    blue_player_ids: Optional[List[str]] = None,
    red_player_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Estima P(blue_wins) usando modelo team-only.

    Args:
        blue_name: nome do time blue (Leaguepedia ou parecido)
        red_name:  nome do time red
        blue_player_ids: lista de player IDs OE (opcional, usa defaults se None)
        red_player_ids:  lista de player IDs OE (opcional)

    Retorna None se modelo nao esta carregado, senao um dict com:
        win_prob_blue, win_prob_red, confidence, factors, z_score, model_type
    """
    _load()
    if not is_available():
        return None

    state = _state
    coefs = _coefs

    # Resolve teams
    blue_id = _resolve_team_id(blue_name)
    red_id  = _resolve_team_id(red_name)

    # Features ---
    # rating_diff
    blue_rating = state["team_rating"].get(blue_id, 1500.0)
    red_rating  = state["team_rating"].get(red_id, 1500.0)
    liga_pts = _league_pts(blue_id, red_id)
    rating_raw = blue_rating - red_rating + liga_pts
    rating_clamped = max(-_ELO_CLAMP * 1.5, min(_ELO_CLAMP * 1.5, rating_raw))
    rating_diff = rating_clamped / _ELO_CLAMP

    # recent_form_diff
    blue_tf = _team_form_wr(state, blue_id) if blue_id else _TEAM_FORM_DEFAULT
    red_tf  = _team_form_wr(state, red_id)  if red_id  else _TEAM_FORM_DEFAULT
    recent_form_diff = blue_tf - red_tf

    # h2h_diff
    h2h_diff = _h2h_score(state, blue_id, red_id) if (blue_id and red_id) else 0.0

    # player_form_diff
    def lineup_form(ids):
        slots = list(ids or [])[:5]
        slots += [""] * (5 - len(slots))
        return sum(_player_form_wr(state, pid) for pid in slots)
    blue_pf = lineup_form(blue_player_ids)
    red_pf = lineup_form(red_player_ids)
    player_form_diff = (blue_pf - red_pf) / 5.0

    # Combina via logistic
    intercept = coefs.get("__intercept__", 0.0)
    z = intercept
    z += coefs.get("rating_diff", 0.0) * rating_diff
    z += coefs.get("recent_form_diff", 0.0) * recent_form_diff
    z += coefs.get("h2h_diff", 0.0) * h2h_diff
    z += coefs.get("player_form_diff", 0.0) * player_form_diff

    p_blue = _sigmoid(z)

    # Confianca: depende de quanto dado real tinha
    has_blue = blue_id is not None
    has_red  = red_id  is not None
    h2h_games = 0
    if blue_id and red_id:
        a, b = (blue_id, red_id) if blue_id <= red_id else (red_id, blue_id)
        rec = state["h2h"].get((a, b))
        if rec:
            h2h_games = int(rec["games"])
    team_games_blue = len(state["team_form"].get(blue_id, []))
    team_games_red  = len(state["team_form"].get(red_id, []))
    min_team_games = min(team_games_blue, team_games_red)

    if not has_blue or not has_red:
        confidence = "baixa"
    elif min_team_games >= 8 and h2h_games >= 2:
        confidence = "alta"
    elif min_team_games >= 5:
        confidence = "media"
    else:
        confidence = "baixa"

    # Nota: recent_form_diff (Forma time) tem coef negativo no modelo -
    # provavelmente captura mean reversion ou colinearidade com Elo. Esta
    # incluido no z mas escondido do display para nao confundir o usuario.
    factors = [
        _factor("Elo (rating)",  rating_diff,      coefs.get("rating_diff", 0.0),
                f"{blue_rating:.0f} vs {red_rating:.0f}{_liga_txt(liga_pts)}"),
        _factor("Forma jogador", player_form_diff, coefs.get("player_form_diff", 0.0),
                f"{blue_pf/5*100:.0f}% vs {red_pf/5*100:.0f}%"),
        _factor("H2H",           h2h_diff,         coefs.get("h2h_diff", 0.0),
                f"{h2h_games} jogos"),
    ]

    return {
        "win_prob_blue":  p_blue,
        "win_prob_red":   1.0 - p_blue,
        "confidence":     confidence,
        "factors":        factors,
        "z_score":        round(z, 3),
        "model_type":     "trained_team_only",
        "blue_resolved":  has_blue,
        "red_resolved":   has_red,
        "team_games":     min_team_games,
        "h2h_games":      h2h_games,
    }


def _factor(name: str, signal: float, coef: float, detail: str) -> Dict[str, Any]:
    contribution = coef * signal
    favors = "neutro"
    if contribution > 0.05:
        favors = "blue"
    elif contribution < -0.05:
        favors = "red"
    return {
        "name":         name,
        "signal":       round(signal, 3),
        "contribution": round(contribution, 3),
        "detail":       detail,
        "favors":       favors,
    }


def _resolve_player_id_from_candidates(candidates: List[str]) -> Optional[str]:
    values = [v for c in candidates if c for v in _player_name_candidates(c)]
    # 1) nome EXATO em qualquer variante (com e sem a tag do time) antes de
    #    aceitar trecho: "DK Wayne" tem que achar "Wayne" antes de "Way".
    index = _player_name_index or {}
    for value in values:
        pid = index.get(_norm_name(value))
        if pid:
            return pid
    return None


def _player_name_candidates(name: str) -> List[str]:
    out: List[str] = []

    def add(value: str) -> None:
        value = str(value or "").strip()
        if value and value.lower() not in {"-", "nan", "none", "null"} and value not in out:
            out.append(value)

    add(name)
    parts = str(name or "").split()
    if len(parts) >= 2:
        add(" ".join(parts[1:]))   # sem a tag do time ("OTF Leo S Man" -> "Leo S Man")
        add(parts[-1])
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _build_enhanced_features(
    base_feats: Dict[str, float],
    game: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, float]:
    blue_champs = [game.get(f"blue_{r}", "") for r in _ROLES]
    red_champs = [game.get(f"red_{r}", "") for r in _ROLES]
    blue_players = [game.get(f"blue_player_{r}", "") for r in _ROLES]
    red_players = [game.get(f"red_player_{r}", "") for r in _ROLES]
    blue_bans = game.get("blue_bans") or []
    red_bans = game.get("red_bans") or []

    return {
        "champion_meta_diff": base_feats.get("champion_meta_diff", _team_meta_score(blue_champs, state) - _team_meta_score(red_champs, state)),
        "player_champion_diff": base_feats.get("player_champion_diff", 0.0),
        "player_comfort_diff": base_feats.get("player_champion_diff", 0.0),
        "role_fit_diff": _team_role_fit_score(blue_champs) - _team_role_fit_score(red_champs),
        "draft_balance_diff": _draft_balance_score(blue_champs) - _draft_balance_score(red_champs),
        "lane_matchup_diff": base_feats.get("lane_matchup_diff", 0.0),
        "rating_diff": base_feats.get("rating_diff", 0.0),
        # FIX 23/07: usa o MESMO ban_score_diff que o treino (base_feats via
        # _ban_value/pickrate). Antes recomputava com _ban_denial_score (formula
        # e escala diferentes) -> o coef treinado era aplicado sobre valor errado.
        "ban_score_diff": base_feats.get("ban_score_diff", 0.0),
        "synergy_diff": base_feats.get("synergy_diff", 0.0),
        "counter_diff": base_feats.get("counter_diff", 0.0),
        "last_pick_counter_diff": base_feats.get("last_pick_counter_diff", 0.0),
        "player_form_diff": base_feats.get("player_form_diff", 0.0),
        "h2h_diff": base_feats.get("h2h_diff", 0.0),
        "head_to_head_diff": base_feats.get("h2h_diff", 0.0),
        "recent_form_diff": base_feats.get("recent_form_diff", 0.0),
        "flex_matchup_diff": base_feats.get("flex_matchup_diff", 0.0),
        "side_advantage": base_feats.get("side_advantage", 1.0),
        "first_pick_diff": base_feats.get("first_pick_diff", 0.0),
        "sample_size_confidence_diff": base_feats.get("sample_size_confidence_diff", 0.0),
        "team_patch_meta_diff": base_feats.get("team_patch_meta_diff", 0.0),
        "jungle_mid_synergy_diff": base_feats.get("jungle_mid_synergy_diff", 0.0),
        "bot_2v2_synergy_diff": base_feats.get("bot_2v2_synergy_diff", 0.0),
        "early_game_team_diff": base_feats.get("early_game_team_diff", 0.0),
        "objective_control_diff": base_feats.get("objective_control_diff", 0.0),
        "side_specific_team_wr_diff": base_feats.get("side_specific_team_wr_diff", 0.0),
        "series_context_diff": base_feats.get("series_context_diff", 0.0),
        "patch_champion_wr_diff": base_feats.get("patch_champion_wr_diff", 0.0),
        "league_champion_wr_diff": base_feats.get("league_champion_wr_diff", 0.0),
        "league_strength_diff": base_feats.get("league_strength_diff", 0.0),
        "player_champion_rate_diff": base_feats.get("player_champion_rate_diff", 0.0),
        "comfort_kda_diff": base_feats.get("comfort_kda_diff", 0.0),
    }


def _team_meta_score(champs: List[str], state: Dict[str, Any]) -> float:
    recent = state.get("champion_recent_meta", {}) or {}
    if recent:
        scores = [_champ_recent_meta_score(recent, champ) for champ in champs if champ]
        return sum(scores) / max(len(scores), 1) if scores else 0.0

    pickrate = state.get("champion_pickrate", {}) or {}
    if not pickrate:
        return 0.0
    max_count = max(float(v or 0) for v in pickrate.values()) or 1.0
    scores = []
    for champ in champs:
        count = _champ_pick_count(pickrate, champ)
        scores.append(math.log1p(count) / math.log1p(max_count))
    return sum(scores) / max(len(scores), 1)


def _champ_recent_meta_score(recent: Dict[str, Dict[str, float]], champ: str) -> float:
    rec = _lookup_champ_recent_record(recent, champ)
    games = float((rec or {}).get("games", 0.0) or 0.0)
    if games <= 0:
        return 0.0
    max_games = max((float(v.get("games", 0.0) or 0.0) for v in recent.values()), default=1.0) or 1.0
    pick_score = math.log1p(games) / math.log1p(max_games)
    wins = float((rec or {}).get("wins", 0.0) or 0.0)
    wr = (wins + _CHAMPION_META_WR_PRIOR * 0.5) / (games + _CHAMPION_META_WR_PRIOR)
    volume = _clamp(math.log1p(games) / math.log1p(40.0), 0.0, 1.0)
    return pick_score + _CHAMPION_META_WR_WEIGHT * volume * ((wr - 0.5) * 2.0)


def _lookup_champ_recent_record(recent: Dict[str, Dict[str, float]], champ: str) -> Optional[Dict[str, float]]:
    if champ in recent:
        return recent[champ]
    key = _champ_key(champ)
    for raw, rec in recent.items():
        if _champ_key(raw) == key:
            return rec
    return None


def _team_role_fit_score(champs: List[str]) -> float:
    scores = [
        _role_fit_score(champ, role)
        for champ, role in zip(champs, _ROLES)
    ]
    return sum(scores) / max(len(scores), 1)


def _role_fit_score(champ: str, role: str) -> float:
    counts = (_champ_role_counts or {}).get(_champ_key(champ), {})
    total = sum(float(v or 0) for v in counts.values())
    if total <= 0:
        return 0.5
    # Prior leve evita que um pick rarissimo vire 100% fit por acidente.
    role_count = float(counts.get(role, 0.0))
    share = (role_count + 3.0) / (total + 3.0 * len(_ROLES))
    return _clamp((share - 0.20) / 0.80, 0.0, 1.0)


def _draft_balance_score(champs: List[str]) -> float:
    profiles = [_champ_traits(champ) for champ in champs]
    known = [p for p in profiles if p]
    if not known:
        return 0.5

    def avg(key: str) -> float:
        return sum(p.get(key, 0.0) for p in known) / len(known)

    def peak(key: str) -> float:
        return max((p.get(key, 0.0) for p in known), default=0.0)

    ap = sum(p.get("ap", 0.0) for p in known)
    ad = sum(p.get("ad", 0.0) for p in known)
    damage_total = ap + ad
    damage_mix = 0.5 if damage_total <= 0 else 1.0 - abs(ap - ad) / damage_total

    components = {
        "engage": peak("engage"),
        "frontline": peak("frontline"),
        "cc": max(avg("cc"), peak("cc") * 0.65),
        "waveclear": max(avg("waveclear"), peak("waveclear") * 0.75),
        "damage_mix": damage_mix,
        "scaling": avg("scaling"),
        "objective": max(avg("objective"), peak("objective") * 0.70),
        "pick": max(avg("pick"), peak("pick") * 0.70),
        "teamfight": avg("teamfight"),
        "side": peak("side"),
        "peel": peak("peel"),
    }
    weights = {
        "engage": 0.90,
        "frontline": 0.80,
        "cc": 0.80,
        "waveclear": 0.60,
        "damage_mix": 0.70,
        "scaling": 0.50,
        "objective": 0.50,
        "pick": 0.40,
        "teamfight": 0.40,
        "side": 0.30,
        "peel": 0.30,
    }
    raw = sum(components[k] * w for k, w in weights.items()) / sum(weights.values())
    penalty = 0.0
    if components["engage"] < 0.25:
        penalty += 0.08
    if components["frontline"] < 0.25:
        penalty += 0.08
    if components["waveclear"] < 0.25:
        penalty += 0.05
    if components["cc"] < 0.25:
        penalty += 0.05
    if damage_mix < 0.25:
        penalty += 0.07
    return _clamp(raw - penalty, 0.0, 1.0)


def _ban_denial_score(bans: List[str], enemy_player_ids: List[str], state: Dict[str, Any]) -> float:
    pickrate = state.get("champion_pickrate", {}) or {}
    max_count = max((float(v or 0) for v in pickrate.values()), default=1.0) or 1.0
    score = 0.0
    for ban in bans:
        key = _champ_key(ban)
        if not key:
            continue
        count = _champ_pick_count(pickrate, ban)
        meta_value = math.log1p(count) / math.log1p(max_count)
        comfort_value = max(
            (_player_champ_comfort_value(player_id, ban, state) for player_id in enemy_player_ids),
            default=0.0,
        )
        score += meta_value + 0.65 * comfort_value
    return score / 5.0


def _player_champ_comfort_value(player_id: str, champ: str, state: Dict[str, Any]) -> float:
    if not player_id or not champ:
        return 0.0
    rec = _lookup_player_champ_record(player_id, champ, state)
    if not rec:
        return 0.0
    games = float(rec.get("games", 0) or 0)
    if games <= 0:
        return 0.0
    wins = float(rec.get("wins", 0) or 0)
    smoothed_wr = (wins + 5.0) / (games + 10.0)
    volume = _clamp(math.log1p(games) / math.log1p(25.0), 0.0, 1.0)
    return volume * (0.50 + 0.50 * smoothed_wr)


def _lookup_player_champ_record(player_id: str, champ: str, state: Dict[str, Any]) -> Optional[Dict[str, float]]:
    pc = state.get("player_champion", {}) or {}
    exact = pc.get((player_id, champ))
    if exact:
        return exact
    champ_key = _champ_key(champ)
    for (pid, raw_champ), rec in pc.items():
        if pid == player_id and _champ_key(raw_champ) == champ_key:
            return rec
    return None


def _champ_traits(champ: str) -> Dict[str, float]:
    return _CHAMPION_TRAITS.get(_champ_key(champ), {})


def _champ_pick_count(pickrate: Dict[Any, Any], champ: Any) -> float:
    if champ in pickrate:
        return float(pickrate.get(champ, 0) or 0)
    key = _champ_key(champ)
    for raw, value in pickrate.items():
        if _champ_key(raw) == key:
            return float(value or 0)
    return 0.0


def _champ_key(champ: Any) -> str:
    return "".join(c for c in str(champ or "").casefold() if c.isalnum())


def predict_full_from_draft(
    blue_name: str,
    red_name: str,
    draft: Dict[str, Any],
    blue_player_names: Optional[List[str]] = None,
    red_player_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Predicao com 10 features usando picks/players da live feed.

    Args:
        blue_name, red_name: nomes dos times (resolvidos via fuzzy match)
        draft: dict de api.livefeed.extract_draft() com picks por role
        blue_player_names, red_player_names: opcional, lista de summonerNames
            extras (pra fuzzy match adicional caso player IDs nao batam)
    """
    _load()
    if not is_available() or _full_coefs is None or _ctx_for_lookup is None:
        return None

    state = _state

    # Resolve teams
    blue_id = _resolve_team_id(blue_name)
    red_id  = _resolve_team_id(red_name)

    blue_player_names = blue_player_names or []
    red_player_names = red_player_names or []

    # Monta dict no formato que temporal_context.build_features espera
    game = {
        "blue_teamid": blue_id or "",
        "red_teamid":  red_id  or "",
        "blue_picks_draft": draft.get("blue_picks_draft", []),
        "red_picks_draft":  draft.get("red_picks_draft", []),
        "blue_bans": draft.get("blue_bans", []),
        "red_bans":  draft.get("red_bans", []),
        "patch": draft.get("patch", ""),
        "league": _canonical_league(draft.get("league", "") or draft.get("league_name", "")),
        "game": draft.get("game_number", draft.get("game", 1)),
    }
    for idx, r in enumerate(_ROLES):
        game[f"blue_{r}"] = draft.get(f"blue_{r}", "")
        game[f"red_{r}"]  = draft.get(f"red_{r}", "")
        bn = draft.get(f"blue_playername_{r}", "")
        rn = draft.get(f"red_playername_{r}", "")
        bp_hint = blue_player_names[idx] if idx < len(blue_player_names) else ""
        rp_hint = red_player_names[idx] if idx < len(red_player_names) else ""
        for side, hint, name in (("blue", bp_hint, bn), ("red", rp_hint, rn)):
            id_key = f"{side}_playerid_{r}"
            game[f"{side}_player_{r}"] = (
                _known_player_id(draft[id_key]) if id_key in draft
                else (_resolve_player_id_from_candidates([hint, name]) or ""))

    # Calcula features com o snapshot
    base_feats = _ctx_for_lookup.build_features(game)
    feats = _build_enhanced_features(base_feats, game, state)
    if not all(math.isfinite(float(value)) for value in feats.values()):
        raise ValueError("Features nao finitas: previsao recusada")

    # pre_match_block: consolida rating/h2h/forma num bloco capado (descoberta
    # do Dota). Reconstrui a partir das features cruas e aplica o cap, para que
    # o pre-jogo nao domine e o draft tenha espaco.
    pre_match_raw = base_feats.get("pre_match_raw")
    if pre_match_raw is None:
        pre_match_raw = sum(
            w * feats.get(f, 0.0) for f, w in _PREMATCH_WEIGHTS.items()
        )
    feats["pre_match_block"] = _clamp(pre_match_raw, -_prematch_cap, _prematch_cap)

    # Combina via formula Blue - Red adaptada do modelo de Dota.
    z = _live_intercept()
    for fname in _live_feature_order():
        z += _live_weight(fname) * feats.get(fname, 0.0)

    p_blue = _clamp(_sigmoid(z), _P_MIN, _P_MAX)

    # ALTERNATIVA: previsao do modelo cap=0 (pre-jogo zerado, so draft).
    # Usa coeficientes proprios (treinados separadamente com cap=0), com
    # pre_match_block = 0. Util para caçar valor em underdog com draft bom.
    def _prob_sem_pre(coefs: Optional[Dict[str, float]]) -> Optional[float]:
        if not coefs:
            return None
        z_alt = coefs.get("__intercept__", 0.0)
        for fname, coef in coefs.items():
            if fname == "__intercept__":
                continue
            z_alt += coef * (0.0 if fname == "pre_match_block" else feats.get(fname, 0.0))
        return _clamp(_sigmoid(z_alt), _P_MIN, _P_MAX)

    p_blue_no_pre = _prob_sem_pre(_zero_coefs)
    p_blue_comfort = _prob_sem_pre(_comfort_coefs)
    p_blue_draft_liga = _prob_sem_pre(_champs_liga_coefs)
    p_blue_comfort_liga = _prob_sem_pre(_comfort_liga_coefs)

    # Quantos players foram resolvidos
    blue_resolved_p = sum(1 for r in _ROLES if game[f"blue_player_{r}"])
    red_resolved_p  = sum(1 for r in _ROLES if game[f"red_player_{r}"])
    profiled_champs = sum(
        1
        for r in _ROLES
        for champ in (game.get(f"blue_{r}", ""), game.get(f"red_{r}", ""))
        if _champ_traits(champ)
    )

    # Confianca: melhor quanto mais coisas resolvidas
    if blue_id and red_id and blue_resolved_p >= 4 and red_resolved_p >= 4 and profiled_champs >= 6:
        confidence = "alta"
    elif blue_id and red_id and blue_resolved_p >= 2 and red_resolved_p >= 2:
        confidence = "media"
    else:
        confidence = "baixa"

    blue_rating = state["team_rating"].get(blue_id, 1500.0) if blue_id else 1500.0
    red_rating  = state["team_rating"].get(red_id,  1500.0) if red_id  else 1500.0
    liga = _liga_txt(_league_pts(blue_id, red_id))
    factor_specs = [
        ("Meta picks", "champion_meta_diff", "patch/pro pick proxy"),
        ("Comfort champ", "player_champion_diff", f"{blue_resolved_p}/5 vs {red_resolved_p}/5 players"),
        ("Amostra comfort", "sample_size_confidence_diff", "volume player/champ"),
        ("Team meta", "team_patch_meta_diff", "time nesses picks"),
        ("Patch WR", "patch_champion_wr_diff", "champ no patch"),
        ("Liga WR", "league_champion_wr_diff", "champ nessa liga"),
        ("Comfort %", "player_champion_rate_diff", "% tempo no champ"),
        ("JNG+MID", "jungle_mid_synergy_diff", "sinergia core mapa"),
        ("Bot 2v2", "bot_2v2_synergy_diff", "ADC+SUP"),
        ("Early game", "early_game_team_diff", "gold/xp/cs/FB@15"),
        ("Objetivos", "objective_control_diff", "drag/herald/baron/torres"),
        ("Side WR", "side_specific_team_wr_diff", "time no lado atual"),
        ("Lane matchup", "lane_matchup_diff", f"signal {feats.get('lane_matchup_diff', 0.0):+.2f}"),
        ("Elo", "rating_diff", f"{blue_rating:.0f} vs {red_rating:.0f}{liga}"),
        ("Bans", "ban_score_diff", f"signal {feats.get('ban_score_diff', 0.0):+.2f}"),
        ("Synergy", "synergy_diff", f"signal {feats.get('synergy_diff', 0.0):+.2f}"),
        ("Counters", "counter_diff", f"signal {feats.get('counter_diff', 0.0):+.2f}"),
        ("Pre-jogo", "pre_match_block",
         f"Elo {blue_rating:.0f}v{red_rating:.0f}{liga} | forma+h2h (cap {_prematch_cap:.1f})"),
    ]
    factors = [
        _factor(name, feats.get(feature, 0.0), _live_weight(feature), detail)
        for name, feature, detail in factor_specs
    ]
    factors = sorted(factors, key=lambda f: abs(f.get("contribution", 0.0)), reverse=True)[:5]

    return {
        "win_prob_blue":   p_blue,
        "win_prob_red":    1.0 - p_blue,
        "confidence":      confidence,
        "factors":         factors,
        "z_score":         round(z, 3),
        "model_type":      "enhanced_live_feature_v4",
        "blue_resolved":   blue_id is not None,
        "red_resolved":    red_id  is not None,
        "players_resolved_blue": blue_resolved_p,
        "players_resolved_red":  red_resolved_p,
        "features":        {k: round(v, 4) for k, v in feats.items()},
        # ALTERNATIVA cap=0: previsao "so draft" (pre-jogo zerado).
        # None se modelo alternativo nao foi treinado/salvo.
        "win_prob_blue_no_pre": (
            p_blue_no_pre if p_blue_no_pre is not None else None
        ),
        "win_prob_red_no_pre":  (
            1.0 - p_blue_no_pre if p_blue_no_pre is not None else None
        ),
        "no_pre_model_type": _zero_model_type or None,
        # draft + comfort do jogador (sem forca do time; 29/10: e sem a forca da liga)
        "win_prob_blue_comfort": (
            p_blue_comfort if p_blue_comfort is not None else None
        ),
        # 29/10: as mesmas duas COM a forca da liga de casa dos times
        "win_prob_blue_draft_liga": p_blue_draft_liga,
        "win_prob_blue_comfort_liga": p_blue_comfort_liga,
    }


def _strip_team_prefix(summoner_name: str) -> str:
    """Tenta remover prefixo de time (e.g. 'WECube' -> 'Cube').

    Em LoL pro o summoner pode ter prefixo de 2-4 chars do team code.
    Estrategia: retorna o nome original (sera tentado por fuzzy match);
    se nao bater, o resolver tenta strings menores.
    """
    return summoner_name  # fuzzy match no resolver ja lida com isso


def predict_from_app_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Adapta o dict emitido pelo MatchupWorker para o predict() interno.

    O worker emite:
        data["blue"]["name"], data["blue"]["players"][{role, player, ...}]
        data["red"]["name"],  data["red"]["players"][...]
    """
    _load()
    if not is_available():
        return None

    blue_name = data.get("blue", {}).get("name", "")
    red_name  = data.get("red", {}).get("name", "")

    blue_players = data.get("blue", {}).get("players", []) or []
    red_players  = data.get("red", {}).get("players", []) or []

    blue_ids = []
    for p in blue_players:
        pid = (_known_player_id(p["resolved_player_id"]) if "resolved_player_id" in p else
               _resolve_player_id(p.get("resolved_player", p.get("player", "")) or ""))
        if pid:
            blue_ids.append(pid)
    red_ids = []
    for p in red_players:
        pid = (_known_player_id(p["resolved_player_id"]) if "resolved_player_id" in p else
               _resolve_player_id(p.get("resolved_player", p.get("player", "")) or ""))
        if pid:
            red_ids.append(pid)

    return predict(
        blue_name=blue_name,
        red_name=red_name,
        blue_player_ids=blue_ids,
        red_player_ids=red_ids,
    )


if __name__ == "__main__":
    print("Modelo treinado:", "OK" if is_available() else "INDISPONIVEL")
    print("Stats:", stats())
    print()
    # Teste rapido
    for blue, red in [("Gen.G", "T1"), ("Hanwha Life Esports", "KT Rolster"),
                      ("Fnatic", "G2 Esports"), ("desconhecido X", "desconhecido Y")]:
        p = predict(blue, red)
        if p:
            print(f"{blue} vs {red}: P(blue)={p['win_prob_blue']*100:>5.1f}%  "
                  f"conf={p['confidence']:<6}  resolved={p['blue_resolved']},{p['red_resolved']}")


