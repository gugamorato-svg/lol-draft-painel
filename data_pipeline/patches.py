"""Normalizacao de patch entre as fontes.

O Oracle's Elixir guarda o patch como FLOAT: 16.09, 16.1 (=16.10), 16.2 (=16.20).
O live feed da Riot manda STRING: "16.9.769.5709", "16.10.1". Comparar o texto
cru quebra justamente nos patches .01-.09, .10 e .20 ("16.1" do live e o patch 1,
mas "16.1" do Oracle e o 16.10). O TIPO diz a fonte: float = Oracle, str = live.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


def patch_tuple(value) -> Optional[Tuple[int, int]]:
    """(major, minor) de qualquer representacao de patch, ou None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):          # Oracle (np.float64 e float)
        if isinstance(value, float) and math.isnan(value):
            return None
        major = int(value)
        return major, int(round((float(value) - major) * 100))
    parts = str(value).strip().split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def patch_label(value) -> str:
    """'16.9.769' / 16.09 -> '16.09' (para exibir)."""
    t = patch_tuple(value)
    return f"{t[0]}.{t[1]:02d}" if t else ""


def oe_patch_key(value) -> str:
    """Chave no formato que o treino grava (str do float do Oracle): 16.10 -> '16.1'."""
    t = patch_tuple(value)
    return str(float(f"{t[0]}.{t[1]:02d}")) if t else ""
