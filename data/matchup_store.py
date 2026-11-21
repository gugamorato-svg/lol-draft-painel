"""Carrega champion_matchups.csv e champion_synergies.csv em memória.

Expõe:
    matchup_score(own, opp)  → float [-2, +2]  (positivo = own favorecido)
    synergy_score(a, b)      → float [-2, +2]  (positivo = boa sinergia)
    is_loaded()              → bool

Bayesian smoothing idêntico ao Dota (Padrão 4 das skills):
    PRIOR_GAMES = 20  — pull to 50% WR quando jogos são poucos
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

_DATA = Path(__file__).parent

_MATCHUP_FILE  = _DATA / "champion_matchups.csv"
_SYNERGY_FILE  = _DATA / "champion_synergies.csv"

PRIOR_GAMES = 20   # partidas virtuais em 50% (Bayesian smoothing)
SCORE_SCALE = 10   # multiplica (wr - 0.5) antes de clampar
SCORE_CAP   = 2.0  # score máximo/mínimo

# Chave: (own_lower, opp_lower, matchup_type) -> (wins_smoothed, games)
# Armazenamos wins/games para poder fazer smoothing correto
_matchups: Dict[Tuple[str, str, str], Tuple[float, int]] = {}
_synergies: Dict[Tuple[str, str], Tuple[float, int]] = {}
_loaded = False


def _norm(name: str) -> str:
    return name.strip().lower()


def _smoothed_score(winrate_pct: float, games: int) -> float:
    """Bayesian smoothed score: 0.0 quando games→0, escala [-CAP, +CAP]."""
    wins = winrate_pct / 100.0 * games
    smoothed_wr = (wins + PRIOR_GAMES * 0.5) / (games + PRIOR_GAMES)
    raw = (smoothed_wr - 0.5) * SCORE_SCALE
    return max(-SCORE_CAP, min(SCORE_CAP, raw))


def load(force: bool = False) -> None:
    """Carrega os CSVs. Chamado automaticamente na primeira consulta."""
    global _loaded
    if _loaded and not force:
        return

    _matchups.clear()
    _synergies.clear()

    # Matchups (direct + indirect)
    if _MATCHUP_FILE.exists():
        with open(_MATCHUP_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                own  = _norm(row.get("own_champion", ""))
                opp  = _norm(row.get("opp_champion", ""))
                mtype = row.get("matchup_type", "direct")
                games = int(row.get("games", 0) or 0)
                wr    = float(row.get("winrate_pct", 50) or 50)
                if own and opp and games > 0:
                    key = (own, opp, mtype)
                    # Se já existe, soma (pode haver múltiplos roles)
                    prev_wr, prev_g = _matchups.get(key, (50.0, 0))
                    # Média ponderada de winrate
                    total_g = prev_g + games
                    avg_wr  = (prev_wr * prev_g + wr * games) / total_g
                    _matchups[key] = (avg_wr, total_g)

    # Synergies
    if _SYNERGY_FILE.exists():
        with open(_SYNERGY_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                own   = _norm(row.get("own_champion", ""))
                ally  = _norm(row.get("opp_champion", ""))
                games = int(row.get("games", 0) or 0)
                wr    = float(row.get("winrate_pct", 50) or 50)
                if own and ally and games > 0:
                    # Chave canônica: ordem alfabética para evitar duplicatas
                    key = (min(own, ally), max(own, ally))
                    prev_wr, prev_g = _synergies.get(key, (50.0, 0))
                    total_g = prev_g + games
                    avg_wr  = (prev_wr * prev_g + wr * games) / total_g
                    _synergies[key] = (avg_wr, total_g)

    _loaded = True


def is_loaded() -> bool:
    """Retorna True se os CSVs foram carregados e têm dados."""
    load()
    return bool(_matchups or _synergies)


def matchup_score(own: str, opp: str, mtype: str = "direct") -> float:
    """Score de matchup own vs opp. Positivo = own favorecido."""
    load()
    key = (_norm(own), _norm(opp), mtype)
    if key in _matchups:
        wr, games = _matchups[key]
        return _smoothed_score(wr, games)
    # Tenta inverso (opp vs own = sinal negativo)
    key_inv = (_norm(opp), _norm(own), mtype)
    if key_inv in _matchups:
        wr, games = _matchups[key_inv]
        return -_smoothed_score(wr, games)
    return 0.0


def synergy_score(champ_a: str, champ_b: str) -> float:
    """Score de sinergia entre dois campeões do mesmo time."""
    load()
    a, b = _norm(champ_a), _norm(champ_b)
    key = (min(a, b), max(a, b))
    if key in _synergies:
        wr, games = _synergies[key]
        return _smoothed_score(wr, games)
    return 0.0


def stats() -> Dict[str, int]:
    """Retorna contagens de registros carregados."""
    load()
    return {
        "matchup_pairs": len(_matchups),
        "synergy_pairs": len(_synergies),
    }
