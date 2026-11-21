"""Cliente para a API "unofficial" da Riot Esports (lolesports.com).

Usada para descobrir partidas ao vivo / proximas / completadas e roster por evento.

Endpoints principais (https://esports-api.lolesports.com/persisted/gw/):
  - getLeagues
  - getSchedule?leagueId=...   (proximas/recentes partidas da liga)
  - getLive                    (partidas ao vivo agora)
  - getEventDetails?id=...     (detalhes de UMA partida: times, jogadores)
  - getTeams?id=...            (info do time + roster atual)

Auth: header `x-api-key` com a chave publica (a mesma usada pelo site oficial).
Sem essa chave a API responde 403. Em maio/2026 a chave abaixo ainda funciona,
mas Riot pode trocar a qualquer momento - se cair, basta abrir o devtools em
lolesports.com, copiar a chave do header, e atualizar PUBLIC_API_KEY.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from .cache import JsonCache

DEFAULT_PUBLIC_API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
PUBLIC_API_KEY = os.getenv("LOL_ESPORTS_API_KEY", DEFAULT_PUBLIC_API_KEY)
BASE = "https://esports-api.lolesports.com/persisted/gw"
# Mirror oficial caso o principal tenha problemas:
MIRROR = "https://prod-relapi.ewp.gg/persisted/gw"


def _par_times(ev: Dict[str, Any]) -> frozenset:
    """Os 2 times de um evento ao vivo, normalizados (sem espaco/pontuacao, minusculo)."""
    import re as _re
    teams = ((ev or {}).get("match") or {}).get("teams") or []
    return frozenset(_re.sub(r"[^a-z0-9]", "", str(t.get("name") or t.get("code") or "").lower())
                     for t in teams[:2])


class EsportsClient:
    def __init__(
        self,
        cache: Optional[JsonCache] = None,
        api_key: str = PUBLIC_API_KEY,
        hl: str = "en-US",
        timeout: float = 10.0,
    ) -> None:
        self.cache = cache
        self.api_key = api_key
        self.hl = hl
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "x-api-key": self.api_key,
                "User-Agent": "LoLPainelBot/0.1 (+https://github.com/local)",
                "Accept": "application/json",
            }
        )

    # ---------- helpers ----------

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl: int = 300,
    ) -> Dict[str, Any]:
        params = {"hl": self.hl, **(params or {})}
        cache_key = f"{path}?{sorted(params.items())}"

        if self.cache and cache_ttl > 0:
            cached = self.cache.get("esports", cache_key)
            if cached is not None:
                return cached

        last_err: Optional[Exception] = None
        for base in (BASE, MIRROR):
            try:
                resp = self.session.get(f"{base}/{path}", params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if self.cache and cache_ttl > 0:
                        self.cache.set("esports", cache_key, data, ttl_seconds=cache_ttl)
                    return data
                last_err = RuntimeError(f"{path} HTTP {resp.status_code}: {resp.text[:200]}")
            except requests.RequestException as e:
                last_err = e
        raise RuntimeError(f"Esports API falhou em {path}: {last_err}")

    # ---------- endpoints ----------

    def get_leagues(self) -> List[Dict[str, Any]]:
        data = self._get("getLeagues", cache_ttl=86400)
        return data.get("data", {}).get("leagues", [])

    def get_live(self, include_tencent: bool = True) -> List[Dict[str, Any]]:
        """Partidas acontecendo AGORA. Cache curto (30s).

        15/10: a Tencent tambem lista jogo de Worlds (rotulado "LPL", nomes sem espaco:
        "Cloud9Kia"); a partida aparecia 2x no app e o clique na copia da Tencent ia p/ o
        servidor chines (lento, ~15 s; timeout virava janela de erro). Agora a copia da
        Tencent so entra se a Riot NAO tiver a mesma partida ao vivo. `include_tencent=False`
        (quem so precisa da API oficial) pula a consulta lenta."""
        data = self._get("getLive", cache_ttl=30)
        events = list(data.get("data", {}).get("schedule", {}).get("events", []))
        if not include_tencent:
            return events
        try:
            from .lplqq import LplQqClient
            riot = {_par_times(e) for e in events}
            events.extend(e for e in LplQqClient(timeout=self.timeout).live_events()
                          if _par_times(e) not in riot)
        except Exception:
            # A fonte Tencent/LPL e fallback: se cair, mantem Riot getLive.
            pass
        return events

    def get_schedule(
        self, league_ids: Optional[List[str]] = None, page_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Proximas + recentes. Se passar league_ids, filtra; senao retorna global."""
        params: Dict[str, Any] = {}
        if league_ids:
            params["leagueId"] = ",".join(league_ids)
        if page_token:
            params["pageToken"] = page_token
        data = self._get("getSchedule", params=params, cache_ttl=120)
        return data.get("data", {}).get("schedule", {})

    def get_event_details(self, event_id: str, fresh: bool = False) -> Dict[str, Any]:
        """Detalhes de UMA partida (times + games + estados).

        fresh=True ignora o cache. Essencial no fluxo ao vivo: numa serie BO,
        quando o mapa 1 termina e o mapa 2 comeca, o cache de 300s ainda
        mostraria o mapa 1 como atual e o draft nao atualizaria.
        """
        ttl = 0 if fresh else 300
        data = self._get("getEventDetails", params={"id": event_id}, cache_ttl=ttl)
        # evento que saiu da agenda volta {"data": {"event": null}}: .get(..., {}) nao cobre
        # chave presente com valor nulo (23/09: o coletor repetia o erro a cada 30 s)
        return ((data or {}).get("data") or {}).get("event") or {}

    def get_teams(self, team_slugs: List[str]) -> List[Dict[str, Any]]:
        data = self._get("getTeams", params={"id": ",".join(team_slugs)}, cache_ttl=3600)
        return data.get("data", {}).get("teams", [])

    # ---------- conveniencias ----------

    def find_lpl_riot_mirror(
        self, blue_name: str, red_name: str
    ) -> Optional[str]:
        """Acha o match_id Riot equivalente a uma partida LPL/Tencent.

        A Riot tem os dados de LPL no getSchedule + livefeed, mas nao publica no
        getLive. Esta funcao busca no schedule por um match LPL com os mesmos
        dois times (fuzzy match por nome normalizado). Retorna o match.id da
        Riot se achar, senao None.

        Util quando a Tencent esta no "gap entre mapas" sem expor o draft do
        proximo, mas a Riot ja publicou no livefeed.
        """
        import re as _re
        def norm(s):
            return _re.sub(r"[^a-z0-9]", "", str(s or "").lower())
        bn, rn = norm(blue_name), norm(red_name)
        if not bn or not rn:
            return None

        try:
            sched = self.get_schedule()
        except Exception:
            return None
        events = sched.get("events", []) or []
        candidates = []
        for ev in events:
            # 15/10: qualquer liga (a Tencent tambem lista Worlds/MSI); LPL tem preferencia
            m = ev.get("match") or {}
            teams = m.get("teams") or []
            if len(teams) < 2:
                continue
            t1, t2 = teams[0], teams[1]
            t1n = norm(t1.get("name", "") or t1.get("code", ""))
            t2n = norm(t2.get("name", "") or t2.get("code", ""))
            # checa se ambos os times aparecem (qualquer ordem)
            sides = (t1n, t2n)

            def has(needle: str) -> bool:
                if not needle:
                    return False
                return any(
                    needle == s or (len(needle) >= 3 and (needle in s or s in needle))
                    for s in sides
                )
            if has(bn) and has(rn):
                state = ev.get("state", "?")
                lpl = (ev.get("league") or {}).get("name", "") == "LPL"
                candidates.append((state, not lpl, m.get("id", "")))

        if not candidates:
            return None
        # Prioridade: inProgress > unstarted > completed (mais recente da serie ativa); LPL antes
        priority = {"inProgress": 0, "unstarted": 1, "completed": 2}
        candidates.sort(key=lambda x: (priority.get(x[0], 9), x[1]))
        return candidates[0][2] or None

    def find_upcoming_matchup(
        self, team_a: str, team_b: str, days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """Procura no schedule um confronto entre dois times nos proximos N dias."""
        sched = self.get_schedule()
        events = sched.get("events", [])
        ta, tb = team_a.lower(), team_b.lower()
        for ev in events:
            match = ev.get("match") or {}
            teams = match.get("teams") or []
            names = [
                (t.get("name", "").lower(), t.get("code", "").lower()) for t in teams
            ]
            flat = {x for pair in names for x in pair}
            if ta in flat and tb in flat:
                return ev
        return None
