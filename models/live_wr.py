"""WR ao vivo: combina ESTADO do jogo (gold/kills/tempo do feed) com a COMP
(P pre-draft do nosso modelo) numa unica logistica calibrada.

Treino com snapshots do Oracle (@10/15/20/25). O treinador gera P_comp
fora da amostra e registra a validacao em data/live_wr_validation.json.
XP ausente no feed usa zero. As metricas antigas usavam P_comp ajustado
com dados do proprio periodo avaliado e nao comprovam calibracao futura.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Optional

_MODEL_PATH = Path(__file__).parent.parent / "data" / "live_wr_model.csv"
_coefs: Optional[Dict[str, float]] = None


def _load() -> Dict[str, float]:
    global _coefs
    if _coefs is None:
        _coefs = {}
        with open(_MODEL_PATH, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                _coefs[r["feature"]] = float(r.get("coefficient", r.get("coef")))
    return _coefs


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sig(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def estimate_game_minute(state: Dict[str, Any]) -> float:
    """Minuto do jogo. Usa o relogio do feed se houver; senao estima pelo ouro
    total no mapa (~5k + 3.6k/min combinado)."""
    gt = state.get("game_time_s")
    if isinstance(gt, (int, float)) and gt > 0:
        return gt / 60.0
    b = state.get("blue", {}) or {}
    r = state.get("red", {}) or {}
    total = (b.get("gold", 0) or 0) + (r.get("gold", 0) or 0)
    return max(0.0, min(45.0, (total - 5000.0) / 3600.0))


def compute(state: Dict[str, Any], p_comp: float,
            minute: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """P(blue vence) dado o estado ao vivo + a comp (P_comp pre-draft do blue).

    Retorna dict com win_prob_blue/red, minute, gold_lead e os componentes.
    """
    c = _load()
    b = state.get("blue", {}) or {}
    r = state.get("red", {}) or {}
    if minute is None:
        minute = estimate_game_minute(state)

    gold_lead = int(b.get("gold", 0) or 0) - int(r.get("gold", 0) or 0)
    kills_lead = int(b.get("kills", 0) or 0) - int(r.get("kills", 0) or 0)
    # xp derivado dos niveis (do details); 0 se indisponivel
    bxp, rxp = b.get("xp"), r.get("xp")
    xp_lead = (bxp - rxp) if (bxp is not None and rxp is not None) else 0
    gd = gold_lead / 1000.0
    xpd = xp_lead / 1000.0
    kd = float(kills_lead)
    tn = (minute - 17.5) / 7.5
    gd_t = gd * tn
    cl = _logit(p_comp)

    z = (c.get("__intercept__", 0.0)
         + c.get("gd", 0.0) * gd
         + c.get("xpd", 0.0) * xpd
         + c.get("kd", 0.0) * kd
         + c.get("tn", 0.0) * tn
         + c.get("gd_t", 0.0) * gd_t
         + c.get("cl", 0.0) * cl)
    p = _sig(z)
    return {
        "win_prob_blue": round(p, 3),
        "win_prob_red": round(1 - p, 3),
        "minute": round(minute, 1),
        "gold_lead_blue": gold_lead,
        "xp_lead_blue": int(xp_lead),
        "kills_lead_blue": kills_lead,
        "p_comp_blue": round(p_comp, 3),
        "components": {
            "gold": round(c.get("gd", 0.0) * gd, 3),
            "xp": round(c.get("xpd", 0.0) * xpd, 3),
            "kills": round(c.get("kd", 0.0) * kd, 3),
            "comp": round(c.get("cl", 0.0) * cl, 3),
        },
    }
