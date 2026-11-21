"""Cache de icones de ITENS via Data Dragon (espelho do champion_icons).

Itens sao indexados por ID numerico (o feed da Riep retorna items=[3006, 3153,...]),
entao nao precisa de indice nome->id: baixa direto /img/item/{id}.png.
Auto-cura a versao igual ao champion_icons (novo patch/item = re-busca 1x).
"""

from __future__ import annotations

from pathlib import Path

import requests

DDRAGON = "https://ddragon.leagueoflegends.com"


class ItemIconCache:
    def __init__(self, cache_dir: str | Path | None = None, timeout: float = 8.0) -> None:
        self.cache_dir = Path(cache_dir or Path.home() / ".lol_painel" / "item_icons")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self._refreshed = False
        self._missing: set[str] = set()  # ids que o DDragon nao tem (evita re-tentar)

    def path_for(self, item_id: str | int) -> str:
        """Caminho local do icone do item. Baixa se preciso. '' se nao existir."""
        item_id = str(item_id or "").strip()
        if not item_id or item_id in ("0",) or item_id in self._missing:
            return ""
        try:
            path = self.cache_dir / f"{item_id}.png"
            if path.exists() and path.stat().st_size > 500:
                return str(path)
            version = self._version()
            if self._download(version, item_id, path):
                return str(path)
            # pode ser item de patch mais novo -> refaz versao 1x e tenta de novo
            if not self._refreshed:
                version = self._refresh()
                if self._download(version, item_id, path):
                    return str(path)
            self._missing.add(item_id)
            return ""
        except Exception:
            return ""

    def _download(self, version: str, item_id: str, path: Path) -> bool:
        url = f"{DDRAGON}/cdn/{version}/img/item/{item_id}.png"
        try:
            r = self.session.get(url, timeout=self.timeout)
            if r.status_code == 200 and r.content:
                path.write_bytes(r.content)
                return True
        except Exception:
            pass
        return False

    def _refresh(self) -> str:
        self._refreshed = True
        p = self.cache_dir / "version.txt"
        try:
            p.unlink()
        except OSError:
            pass
        return self._version()

    def _version(self) -> str:
        path = self.cache_dir / "version.txt"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        response = self.session.get(f"{DDRAGON}/api/versions.json", timeout=self.timeout)
        response.raise_for_status()
        versions = response.json()
        version = str(versions[0]) if isinstance(versions, list) and versions else "15.10.1"
        path.write_text(version, encoding="utf-8")
        return version
