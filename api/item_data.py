"""Custo e status de item (completo vs componente) via Data Dragon.

Usado p/ o "gold EFETIVO": soma so o gold de itens COMPLETOS (nao componentes,
consumiveis ou trinkets). Um lead de gold em componentes/nao-gasto vale menos
que em itens fechados (insight do usuario).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import requests

DDRAGON = "https://ddragon.leagueoflegends.com"
_MIN_COMPLETED_COST = 1100  # abaixo disso nao conta como power-spike de item


class ItemData:
    def __init__(self, cache_dir: str | Path | None = None, timeout: float = 8.0) -> None:
        self.cache_dir = Path(cache_dir or Path.home() / ".lol_painel" / "item_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self._map: Optional[Dict[str, Tuple[int, bool]]] = None  # id -> (custo_total, completo)

    def _load(self) -> Dict[str, Tuple[int, bool]]:
        if self._map is not None:
            return self._map
        cache = self.cache_dir / "items.json"
        if cache.exists():
            try:
                self._map = {k: (v[0], bool(v[1])) for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
                return self._map
            except (OSError, ValueError):
                pass
        try:
            ver = self.session.get(f"{DDRAGON}/api/versions.json", timeout=self.timeout).json()[0]
            data = self.session.get(f"{DDRAGON}/cdn/{ver}/data/en_US/item.json", timeout=self.timeout).json().get("data", {})
        except Exception:
            self._map = {}
            return self._map
        m: Dict[str, Tuple[int, bool]] = {}
        for iid, it in data.items():
            total = int(it.get("gold", {}).get("total", 0) or 0)
            into = it.get("into") or []           # se builda em algo -> e componente
            tags = it.get("tags", []) or []
            completed = (not into) and total >= _MIN_COMPLETED_COST \
                and "Consumable" not in tags and "Trinket" not in tags
            m[iid] = (total, completed)
        self._map = m
        try:
            cache.write_text(json.dumps({k: [v[0], v[1]] for k, v in m.items()}), encoding="utf-8")
        except OSError:
            pass
        return m

    def effective_gold(self, item_ids: Iterable) -> int:
        """Soma o custo dos itens COMPLETOS (power spikes reais)."""
        m = self._load()
        tot = 0
        for iid in item_ids:
            c, comp = m.get(str(iid), (0, False))
            if comp:
                tot += c
        return tot

    def total_item_gold(self, item_ids: Iterable) -> int:
        """Soma o custo de TODOS os itens (inclui componentes)."""
        m = self._load()
        return sum(m.get(str(iid), (0, False))[0] for iid in item_ids)
