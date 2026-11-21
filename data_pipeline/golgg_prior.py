"""Carrega dados gol.gg como prior para o TemporalContext.

Resolve o problema de cold-start: nas primeiras partidas do treino o
TemporalContext nao tinha dados de matchup ainda (todas as features
de counter/synergy ficavam em 0). Com o prior, ja comecamos com 12k+
matchups e 10k+ synergies vindos do gol.gg.

CSVs esperados (gerados por scripts/download_golgg.py):
  data/champion_matchups.csv   (direct + indirect)
  data/champion_synergies.csv  (good_with)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

_DATA = Path(__file__).parent.parent / "data"

# gol.gg usa nomes maiusculos; nosso state usa o que vem do OE (top/jng/mid/bot/sup)
_ROLE_MAP = {
    "TOP":     "top",
    "JUNGLE":  "jng",
    "MID":     "mid",
    "ADC":     "bot",
    "SUPPORT": "sup",
}


def _synergy_key(c1: str, c2: str) -> Tuple[str, str]:
    return (c1, c2) if c1 <= c2 else (c2, c1)


def load_matchups_prior(
    csv_path: Path = _DATA / "champion_matchups.csv",
) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """Carrega matchups gol.gg em dict {(own, opp, role): {wins, games}}.

    - "direct": preenche com role mapeado (top/jng/mid/bot/sup)
    - "indirect": preenche com role "" (cross-role agregado)
    - Cria entrada reciproca automaticamente (rc, bc, role) com (1 - wr)
    """
    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    if not csv_path.exists():
        return out

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            own = row.get("own_champion", "").strip()
            opp = row.get("opp_champion", "").strip()
            if not own or not opp:
                continue
            try:
                games = int(row.get("games") or 0)
                wr    = float(row.get("winrate_pct") or 50.0)
            except ValueError:
                continue
            if games <= 0:
                continue

            mtype = row.get("matchup_type", "direct")
            role_raw = row.get("role", "").upper()
            role = _ROLE_MAP.get(role_raw, "") if mtype == "direct" else ""

            wins = wr / 100.0 * games
            key  = (own, opp, role)
            prev = out.get(key)
            if prev:
                # Soma se ja existe (e.g. mesmo par em multiplos arquivos)
                prev["games"] += games
                prev["wins"]  += wins
            else:
                out[key] = {"games": games, "wins": wins}

            # Entrada reciproca: (opp, own, role) tem (games - wins) wins
            rev_key = (opp, own, role)
            rev_prev = out.get(rev_key)
            rev_wins = games - wins
            if rev_prev:
                rev_prev["games"] += games
                rev_prev["wins"]  += rev_wins
            else:
                out[rev_key] = {"games": games, "wins": rev_wins}

    return out


def load_synergies_prior(
    csv_path: Path = _DATA / "champion_synergies.csv",
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Carrega synergies gol.gg em dict {(c_min, c_max): {wins, games}}."""
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    if not csv_path.exists():
        return out

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = row.get("own_champion", "").strip()
            b = row.get("opp_champion", "").strip()
            if not a or not b or a == b:
                continue
            try:
                games = int(row.get("games") or 0)
                wr    = float(row.get("winrate_pct") or 50.0)
            except ValueError:
                continue
            if games <= 0:
                continue

            key = _synergy_key(a, b)
            wins = wr / 100.0 * games
            prev = out.get(key)
            if prev:
                prev["games"] += games
                prev["wins"]  += wins
            else:
                out[key] = {"games": games, "wins": wins}

    return out


def stats():
    mu = load_matchups_prior()
    sy = load_synergies_prior()
    return {
        "matchup_keys":     len(mu),
        "matchup_games":    sum(r["games"] for r in mu.values()),
        "synergy_keys":     len(sy),
        "synergy_games":    sum(r["games"] for r in sy.values()),
    }


if __name__ == "__main__":
    s = stats()
    print("gol.gg prior:")
    for k, v in s.items():
        print(f"  {k:<18} {v:>10,}")
