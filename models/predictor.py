"""Estimador heuristico de probabilidade de vitoria para confrontos LoL pro.

Aplica os padroes aprendidos no Dota Draft Analyzer:
  - Padrao 1: features como diferenca normalizada (blue - red)
  - Padrao 2: Bayesian smoothing para H2H com poucos jogos
  - Padrao 4: Bayesian smoothing para matchups com poucos jogos
  - Padrao 7: threshold de confianca baseado em amostra

Feature 4 (champion matchup) e ativada automaticamente quando
data/champion_matchups.csv existir com dados (rodou download_golgg.py).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Permite importar data.matchup_store independente de como o app e chamado
sys.path.insert(0, str(Path(__file__).parent.parent))

# Prior bayesiano para H2H: equivale a 10 jogos em 50/50
_H2H_PRIOR = 10

# Abaixo deste valor absoluto de sinal, o fator e considerado "neutro"
_FAVOR_THRESHOLD = 0.08

# Pesos heuristicos (ver skills_win_rate_modeling.md)
# Quando champion_matchup esta disponivel, seu peso e 0.6
# e os demais sao levemente reduzidos para manter escala total similar.
_W_H2H     = 1.2
_W_KILLS   = 0.8
_W_LANES   = 0.5
_W_MATCHUP = 0.6   # feature 4 - so ativa quando CSVs tiverem dados


def _ordered_players(players):
    """Preserva cada papel; lista incompleta nao desloca o jogador seguinte."""
    roles = ("top", "jng", "mid", "bot", "sup")
    aliases = {"jungle": "jng", "support": "sup", "adc": "bot"}
    by_role = {}
    for i, player in enumerate(players or []):
        raw = str(player.get("role", "")).casefold()
        role = aliases.get(raw, raw) if raw else (roles[i] if i < 5 else "")
        if role in roles:
            # Duplicata de papel torna identidade incerta, nao escolhe a ultima.
            by_role[role] = player if role not in by_role else {"resolved_player": "", "resolved_player_id": ""}
    return [by_role.get(role, {}) for role in roles]


def predict_win_prob(data: Dict[str, Any]) -> Dict[str, Any]:
    """Estima P(blue_wins) a partir dos dados ja coletados do matchup.

    Chain de fallback:
      1. Modelo treinado team-only (acc 63.5% no teste 2025) - quando ambos
         os times sao reconhecidos no historico OE.
      2. Heuristico H2H + kills + lanes + champion MU         - caso contrario.
    """
    # 1) Modelo FULL (10 features) quando ha live_draft (partida ao vivo)
    try:
        from models import trained_predictor
        if trained_predictor.is_available() and data.get("live_draft"):
            # Uma resolucao explicita vazia significa identidade nao confirmada.
            # Nao ressuscitar o nome cru pelo fallback da draft.
            draft_for_model = dict(data["live_draft"])
            ordered = {side: _ordered_players(data.get(side, {}).get("players", []))
                       for side in ("blue", "red")}
            for side in ("blue", "red"):
                for role, player in zip(("top", "jng", "mid", "bot", "sup"),
                                        ordered[side]):
                    if "resolved_player" in player:
                        draft_for_model[f"{side}_playername_{role}"] = player.get("resolved_player") or ""
                    if "resolved_player_id" in player:
                        draft_for_model[f"{side}_playerid_{role}"] = player.get("resolved_player_id") or ""
            blue_player_names = [
                p.get("resolved_player", p.get("player", "")) or ""
                for p in ordered["blue"]
            ]
            red_player_names = [
                p.get("resolved_player", p.get("player", "")) or ""
                for p in ordered["red"]
            ]
            pred = trained_predictor.predict_full_from_draft(
                blue_name=data.get("blue", {}).get("name", ""),
                red_name=data.get("red", {}).get("name", ""),
                draft=draft_for_model,
                blue_player_names=blue_player_names,
                red_player_names=red_player_names,
            )
            if pred and (pred.get("blue_resolved") or pred.get("red_resolved")):
                return pred
    except Exception as exc:
        print(f"[predict] predict_full falhou: {exc}")

    # 2) Modelo team-only (4 features) quando time conhecido mas sem picks
    try:
        from models import trained_predictor
        if trained_predictor.is_available():
            pred = trained_predictor.predict_from_app_data(data)
            if pred and pred.get("blue_resolved") and pred.get("red_resolved"):
                blue_players = data.get("blue", {}).get("players", [])
                red_players  = data.get("red", {}).get("players", [])
                lane_matchups = data.get("lane_matchups", [])
                champ_signal, champ_detail, _ = _champion_matchup_signal(
                    blue_players, red_players, lane_matchups
                )
                if champ_signal is not None:
                    pred["factors"].append({
                        "name":         "Champion MU",
                        "signal":       champ_signal,
                        "contribution": 0.0,
                        "detail":       champ_detail + " (info)",
                        "favors":       _direction(champ_signal),
                    })
                pred["champion_data"] = champ_signal is not None
                return pred
    except Exception:
        pass

    # 2) Heuristico (fallback quando time desconhecido)
    h2h      = data.get("h2h", {})
    blue_ts  = data.get("blue", {}).get("team_stats", {})
    red_ts   = data.get("red", {}).get("team_stats", {})
    blue_players = data.get("blue", {}).get("players", [])
    red_players  = data.get("red", {}).get("players", [])
    lane_matchups = data.get("lane_matchups", [])

    h2h_games = h2h.get("games", 0)
    h2h_wr    = h2h.get("winrate_a", 50.0)
    blue_games = blue_ts.get("games", 0)
    red_games  = red_ts.get("games", 0)

    f_h2h   = _smoothed_h2h(h2h_wr, h2h_games)
    f_kills = _kills_signal(
        blue_ts.get("total_kills_avg", 0.0),
        red_ts.get("total_kills_avg", 0.0),
    )
    f_lanes = _lane_signal(lane_matchups)

    z = _W_H2H * f_h2h + _W_KILLS * f_kills + _W_LANES * f_lanes

    blue_kills = blue_ts.get("total_kills_avg", 0.0)
    red_kills  = red_ts.get("total_kills_avg", 0.0)

    factors = [
        {
            "name": "Hist. H2H",
            "signal": f_h2h,
            "contribution": round(_W_H2H * f_h2h, 3),
            "detail": f"{h2h_wr:.0f}% ({h2h_games}j)",
            "favors": _direction(f_h2h),
        },
        {
            "name": "Kills/jogo",
            "signal": f_kills,
            "contribution": round(_W_KILLS * f_kills, 3),
            "detail": f"{blue_kills:.1f} vs {red_kills:.1f}",
            "favors": _direction(f_kills),
        },
        {
            "name": "Lanes",
            "signal": f_lanes,
            "contribution": round(_W_LANES * f_lanes, 3),
            "detail": f"score {f_lanes:+.2f}",
            "favors": _direction(f_lanes),
        },
    ]

    # Feature 4: champion matchup score (ativo quando gol.gg data disponivel)
    champ_signal, champ_detail, champ_pairs = _champion_matchup_signal(
        blue_players, red_players, lane_matchups
    )
    if champ_signal is not None:
        z += _W_MATCHUP * champ_signal
        factors.append({
            "name": "Champion MU",
            "signal": champ_signal,
            "contribution": round(_W_MATCHUP * champ_signal, 3),
            "detail": champ_detail,
            "favors": _direction(champ_signal),
        })

    p_blue = _sigmoid(z)

    # Confianca: armadilha 50-55% do Dota => exige amostra minima
    min_games = min(blue_games, red_games)
    if min_games < 5:
        confidence = "baixa"
    elif min_games >= 15 and h2h_games >= 3:
        confidence = "alta"
    else:
        confidence = "media"

    return {
        "win_prob_blue": round(p_blue, 3),
        "win_prob_red":  round(1.0 - p_blue, 3),
        "confidence":    confidence,
        "factors":       factors,
        "z_score":       round(z, 3),
        "champion_data": champ_signal is not None,
    }


# ---------- feature 4: champion matchup ----------

def _champion_matchup_signal(
    blue_players: List[Dict],
    red_players: List[Dict],
    lane_matchups: List[Dict],
) -> tuple[Optional[float], str, int]:
    """Calcula score agregado de matchup de campeoes (gol.gg data).

    Retorna (signal, detail_str, n_pairs).
    signal=None quando dados nao estao disponiveis.
    """
    try:
        from data.matchup_store import matchup_score, is_loaded
        if not is_loaded():
            return None, "", 0
    except ImportError:
        return None, "", 0

    # Constroi mapa role -> champion para cada time
    def champ_by_role(players: List[Dict]) -> Dict[str, str]:
        out = {}
        for p in players:
            role    = p.get("role", "")
            champion = p.get("champion_name") or p.get("champion") or ""
            if role and champion:
                out[role] = champion
        return out

    blue_champs = champ_by_role(blue_players)
    red_champs  = champ_by_role(red_players)

    # Para lane_matchups que ja tem role mapeado
    scores = []
    roles_with_data = []
    for m in lane_matchups:
        role = m.get("role", "")
        bc = blue_champs.get(role, "")
        rc = red_champs.get(role, "")
        if bc and rc:
            s = matchup_score(bc, rc, "direct")
            if s != 0.0:
                scores.append(s)
                roles_with_data.append(role)

    if not scores:
        return None, "sem campeoes", 0

    agg = sum(scores) / len(scores)
    detail = f"{len(scores)} lanes ({agg:+.2f})"
    return agg, detail, len(scores)


# ---------- helpers ----------

def _sigmoid(x: float) -> float:
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = max(0.01, min(0.99, p))
    return math.log(p / (1.0 - p))


def _smoothed_h2h(winrate_pct: float, games: int) -> float:
    """Bayesian smoothing: com games=0 retorna 0.0 (prior 50/50)."""
    wins = winrate_pct / 100.0 * games
    smoothed = (wins + _H2H_PRIOR * 0.5) / (games + _H2H_PRIOR)
    return _logit(smoothed)


def _kills_signal(blue_avg: float, red_avg: float) -> float:
    return (blue_avg - red_avg) / 10.0


def _lane_signal(lane_matchups: List[Dict]) -> float:
    if not lane_matchups:
        return 0.0
    total = sum((m.get("delta_kills") or 0.0) for m in lane_matchups)
    return total / (len(lane_matchups) * 2.0)


def _direction(signal: float) -> str:
    if signal > _FAVOR_THRESHOLD:
        return "blue"
    if signal < -_FAVOR_THRESHOLD:
        return "red"
    return "neutro"

