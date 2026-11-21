"""Logger de snapshots de jogo AO VIVO -> dataset p/ validar o gold efetivo.

Cada poll grava 1 linha JSONL com estado (gold cru, gold efetivo, xp, kills,
objetivos), o minuto e o P_comp. O RESULTADO (quem venceu) e resolvido depois
(scripts/resolve_live_log.py) por game_id. Assim juntamos dado real p/ testar se
gold EFETIVO (itens completos) preve melhor que gold cru.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_LOG_DIR = Path(__file__).parent.parent / "data" / "live_log"


def append_snapshot(game_id: str, state: Dict[str, Any], wr: Dict[str, Any],
                    p_comp_blue: float) -> None:
    """Grava um snapshot (1 linha JSONL em data/live_log/{game_id}.jsonl)."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        b = state.get("blue", {}) or {}
        r = state.get("red", {}) or {}
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "game_id": str(game_id),
            "minute": wr.get("minute"),
            "game_state": state.get("game_state", ""),
            "p_comp_blue": round(float(p_comp_blue), 4),
            "wr_blue": wr.get("win_prob_blue"),
            "blue_gold": b.get("gold"), "red_gold": r.get("gold"),
            "blue_eff_gold": b.get("eff_gold"), "red_eff_gold": r.get("eff_gold"),
            "blue_xp": b.get("xp"), "red_xp": r.get("xp"),
            "blue_kills": b.get("kills"), "red_kills": r.get("kills"),
            "blue_dragons": b.get("dragon_count"), "red_dragons": r.get("dragon_count"),
            "blue_barons": b.get("barons"), "red_barons": r.get("barons"),
            "blue_towers": b.get("towers"), "red_towers": r.get("towers"),
            "blue_soul": b.get("has_soul"), "red_soul": r.get("has_soul"),
        }
        path = _LOG_DIR / f"{game_id}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass  # logging nunca deve quebrar o polling
