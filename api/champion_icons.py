"""Cache simples de icones de campeoes via Data Dragon.

Auto-cura: se um campeao nao esta no indice cacheado (ex.: campeao NOVO lancado
depois do ultimo cache), re-busca a versao + indice mais recentes UMA vez por
instancia e tenta de novo. Antes o version.txt era gravado pra sempre e nenhum
campeao novo era reconhecido ate limpar o cache na mao.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


DDRAGON = "https://ddragon.leagueoflegends.com"


class ChampionIconCache:
    def __init__(self, cache_dir: str | Path | None = None, timeout: float = 8.0) -> None:
        self.cache_dir = Path(cache_dir or Path.home() / ".lol_painel" / "champion_icons")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self._refreshed = False  # evita re-buscar toda hora por nome desconhecido/typo

    def path_for(self, champion_name: str) -> str:
        if not champion_name:
            return ""
        try:
            version = self._version()
            champion_id = self._champion_index(version).get(_normal(champion_name), "")

            # Campeao nao esta no indice cacheado -> pode ser um champ NOVO.
            # Refaz versao+indice UMA vez e tenta de novo.
            if not champion_id and not self._refreshed:
                version = self._refresh()
                champion_id = self._champion_index(version).get(_normal(champion_name), "")

            if not champion_id:
                return ""
            path = self.cache_dir / f"{champion_id}.png"
            if not path.exists():
                self._download_icon(version, champion_id, path)
            return str(path) if path.exists() else ""
        except Exception:
            return ""

    def _refresh(self) -> str:
        """Forca re-download da versao mais recente + indice de campeoes."""
        self._refreshed = True
        # apaga caches de metadados (nao as PNGs) e re-busca
        for name in ("version.txt", "index.json"):
            p = self.cache_dir / name
            try:
                p.unlink()
            except OSError:
                pass
        version = self._version()
        self._champion_index(version)  # re-popula index.json com a versao nova
        return version

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

    def _champion_index(self, version: str) -> dict[str, str]:
        path = self.cache_dir / "index.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

        response = self.session.get(
            f"{DDRAGON}/cdn/{version}/data/en_US/champion.json",
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        data = payload.get("data", {})
        index: dict[str, str] = {}
        if isinstance(data, dict):
            for champion_id, champion in data.items():
                if isinstance(champion, dict):
                    index[_normal(str(champion.get("name", "")))] = str(champion_id)
                    index[_normal(str(champion_id))] = str(champion_id)
        path.write_text(json.dumps(index), encoding="utf-8")
        return index

    def _download_icon(self, version: str, champion_id: str, path: Path) -> None:
        response = self.session.get(
            f"{DDRAGON}/cdn/{version}/img/champion/{champion_id}.png",
            timeout=self.timeout,
        )
        response.raise_for_status()
        path.write_bytes(response.content)


def _normal(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())
