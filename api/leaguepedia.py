"""Cliente para a Cargo API da Leaguepedia (lol.fandom.com).

Toda a logica de stats historicas (medias do ano, winrate por campeao,
head-to-head, lane matchups) vive aqui. Sem auth.

Tabelas:
  ScoreboardGames (SG): GameId, Tournament, Team1, Team2, Winner, DateTime_UTC,
                       Patch, Gamelength, Gamelength_Number
  ScoreboardPlayers (SP): GameId, Name, Link, Champion, Kills, Deaths, Assists,
                         Role, Side, Team, TeamVs
  Tournaments: Name, League, Year, Region, OverviewPage, DateStart, Date
  TournamentRosters: Team, RosterLinks (separado por ;), Roles
  Teams: Name, Short, Region, OverviewPage
  Players: ID, Team, Role, Country, ResidencyFormer

Limites: 500 linhas/request, sem auth.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests

from .cache import JsonCache

API = "https://lol.fandom.com/api.php"
ROLES = ["Top", "Jungle", "Mid", "Bot", "Support"]
ROLE_SHORT = {"Top": "TOP", "Jungle": "JNG", "Mid": "MID", "Bot": "BOT", "Support": "SUP"}


def _cargo_escape(value: str) -> str:
    text = str(value or "")
    return text.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))


def _player_row_is_win(row: Dict[str, Any]) -> bool:
    winner = str(row.get("W") or "").strip().lower()
    side = str(row.get("Side") or "").strip().lower()
    blue_values = {"1", "blue"}
    red_values = {"2", "red"}
    return (winner in blue_values and side in blue_values) or (
        winner in red_values and side in red_values
    )



class LeaguepediaError(RuntimeError):
    """API retornou erro (ratelimited, invalid_query, etc)."""
    pass


class LeaguepediaClient:
    # Throttle: minimo de delay entre requests para nao estourar rate-limit
    # da Cargo API do Fandom (~10 req/s genuino, mas mais conservador).
    MIN_INTERVAL = 0.35  # segundos

    def __init__(
        self,
        cache: Optional[JsonCache] = None,
        timeout: float = 15.0,
        backoffs: Optional[List[int]] = None,
    ) -> None:
        self.cache = cache
        self.timeout = timeout
        self.backoffs = backoffs if backoffs is not None else [0, 3, 8, 20]
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "LoLPainelBot/0.1 (contato: lolpainel@local)"}
        )
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        """Espera o tempo minimo desde o ultimo request."""
        delta = time.monotonic() - self._last_request_ts
        if delta < self.MIN_INTERVAL:
            time.sleep(self.MIN_INTERVAL - delta)
        self._last_request_ts = time.monotonic()

    # ---------- baixo nivel ----------

    def cargo(
        self,
        tables: str,
        fields: str,
        where: str = "",
        join_on: str = "",
        group_by: str = "",
        order_by: str = "",
        limit: int = 500,
        cache_ttl: int = 86400,
    ) -> List[Dict[str, Any]]:
        params = {
            "action": "cargoquery",
            "format": "json",
            "tables": tables,
            "fields": fields,
            "limit": str(limit),
        }
        if where:
            params["where"] = where
        if join_on:
            params["join_on"] = join_on
        if group_by:
            params["group_by"] = group_by
        if order_by:
            params["order_by"] = order_by

        cache_key = repr(sorted(params.items()))
        if self.cache and cache_ttl > 0:
            cached = self.cache.get("leaguepedia", cache_key)
            if cached is not None:
                return cached

        # Retry com backoff quando ratelimited (Fandom rate-limita agressivo)
        last_err: Optional[Exception] = None
        for wait in self.backoffs:
            if wait:
                time.sleep(wait)
            try:
                resp = self.session.get(API, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                last_err = e
                continue

            # API retornou erro estruturado?
            err = data.get("error")
            if err:
                code = err.get("code", "?")
                info = err.get("info", "")
                if code == "ratelimited":
                    # retry com backoff maior
                    last_err = LeaguepediaError(f"ratelimited: {info}")
                    continue
                # outros erros: nao adianta retry
                raise LeaguepediaError(f"{code}: {info}")

            rows = [row["title"] for row in data.get("cargoquery", [])]
            if self.cache and cache_ttl > 0:
                self.cache.set("leaguepedia", cache_key, rows, ttl_seconds=cache_ttl)
            return rows

        # Esgotou retries
        raise LeaguepediaError(
            f"Cargo API falhou apos {len(self.backoffs)} tentativas: {last_err}"
        )

    # ---------- queries de alto nivel ----------

    def get_team_roster(self, team: str, year: int) -> List[Dict[str, str]]:
        """Roster atual do time no ano (pega o jogador mais recente por role)."""
        rows = self.cargo(
            tables="ScoreboardPlayers=SP, ScoreboardGames=SG",
            join_on="SP.GameId=SG.GameId",
            fields="SP.Link=Player, SP.Role=Role, SP.Team=Team, SG.DateTime_UTC=Date",
            where=(
                f'SP.Team="{team}" '
                f'AND SG.DateTime_UTC >= "{year}-01-01 00:00:00"'
            ),
            order_by="SG.DateTime_UTC DESC",
            limit=500,
        )
        latest_per_role: Dict[str, Dict[str, str]] = {}
        for r in rows:
            role = r.get("Role", "")
            if role not in latest_per_role:
                latest_per_role[role] = {
                    "player": r.get("Player", ""),
                    "role": role,
                    "team": r.get("Team", ""),
                }
        ordered = []
        for role in ROLES:
            if role in latest_per_role:
                ordered.append(latest_per_role[role])
        return ordered

    def get_player_year_stats(self, player_link: str, year: int) -> Dict[str, float]:
        """Media de kills/mortes/assists do jogador no ano (todos os campeoes)."""
        rows = self.cargo(
            tables="ScoreboardPlayers=SP, ScoreboardGames=SG",
            join_on="SP.GameId=SG.GameId",
            fields="SP.Kills=K, SP.Deaths=D, SP.Assists=A, SG.Winner=W, SP.Side=Side",
            where=(
                f'SP.Link="{player_link}" '
                f'AND SG.DateTime_UTC >= "{year}-01-01 00:00:00"'
            ),
            limit=500,
        )
        if not rows:
            return {"games": 0, "kills_avg": 0.0, "deaths_avg": 0.0,
                    "assists_avg": 0.0, "winrate": 0.0}
        n = len(rows)
        k = sum(int(r.get("K") or 0) for r in rows)
        d = sum(int(r.get("D") or 0) for r in rows)
        a = sum(int(r.get("A") or 0) for r in rows)
        wins = sum(1 for r in rows if _player_row_is_win(r))
        return {
            "games": n,
            "kills_avg": round(k / n, 2),
            "deaths_avg": round(d / n, 2),
            "assists_avg": round(a / n, 2),
            "winrate": round(wins / n, 3),
        }

    def get_player_champion_stats(
        self, player_link: str, champion: str, year: int
    ) -> Dict[str, float]:
        rows = self.cargo(
            tables="ScoreboardPlayers=SP, ScoreboardGames=SG",
            join_on="SP.GameId=SG.GameId",
            fields="SP.Kills=K, SP.Deaths=D, SP.Assists=A, SG.Winner=W, SP.Side=Side",
            where=(
                f'SP.Link="{player_link}" '
                f'AND SP.Champion="{champion}" '
                f'AND SG.DateTime_UTC >= "{year}-01-01 00:00:00"'
            ),
            limit=500,
        )
        if not rows:
            return {"games": 0, "kills_avg": 0.0, "deaths_avg": 0.0,
                    "assists_avg": 0.0, "winrate": 0.0}
        n = len(rows)
        k = sum(int(r.get("K") or 0) for r in rows)
        d = sum(int(r.get("D") or 0) for r in rows)
        a = sum(int(r.get("A") or 0) for r in rows)
        wins = sum(1 for r in rows if _player_row_is_win(r))
        return {
            "games": n,
            "kills_avg": round(k / n, 2),
            "deaths_avg": round(d / n, 2),
            "assists_avg": round(a / n, 2),
            "winrate": round(wins / n, 3),
        }

    def get_player_champion_career_stats(
        self, player_link: str, champion: str,
        last_n_years: int = 3,
    ) -> Dict[str, float]:
        """Stats do jogador com um campeao nos ultimos N anos (default 3).

        Antes pegava carreira inteira (incluindo jogos de 2018+), o que
        mostrava amostras estranhas tipo 8/16. Agora limita a 3 anos
        (ano corrente e os 2 anteriores).
        """
        from datetime import datetime
        cur_year = datetime.now().year
        min_year = cur_year - max(0, last_n_years - 1)
        player_q = _cargo_escape(player_link)
        champion_q = _cargo_escape(champion)
        rows = self.cargo(
            tables="ScoreboardPlayers=SP, ScoreboardGames=SG",
            join_on="SP.GameId=SG.GameId",
            fields="SP.Kills=K, SP.Deaths=D, SP.Assists=A, SG.Winner=W, SP.Side=Side",
            where=(
                f'(SP.Link="{player_q}" OR SP.Name="{player_q}") '
                f'AND SP.Champion="{champion_q}" '
                f'AND SG.DateTime_UTC >= "{min_year}-01-01 00:00:00"'
            ),
            order_by="SG.DateTime_UTC DESC",
            limit=500,
            cache_ttl=7 * 86400,
        )
        if not rows:
            return {"games": 0, "wins": 0, "kills_avg": 0.0, "deaths_avg": 0.0,
                    "assists_avg": 0.0, "winrate": 0.0}
        n = len(rows)
        k = sum(int(r.get("K") or 0) for r in rows)
        d = sum(int(r.get("D") or 0) for r in rows)
        a = sum(int(r.get("A") or 0) for r in rows)
        wins = sum(1 for r in rows if _player_row_is_win(r))
        return {
            "games": n,
            "wins": wins,
            "kills_avg": round(k / n, 2),
            "deaths_avg": round(d / n, 2),
            "assists_avg": round(a / n, 2),
            "winrate": round(100 * wins / max(n, 1), 1),
        }

    def get_team_year_stats(self, team: str, year: int) -> Dict[str, float]:
        """Stats agregadas: kills medios totais, duracao media, winrate."""
        rows = self.cargo(
            tables="ScoreboardGames=SG",
            fields=(
                "SG.GameId=GameId, SG.Team1=T1, SG.Team2=T2, SG.Winner=W, "
                "SG.Team1Kills=K1, SG.Team2Kills=K2, SG.Gamelength_Number=Len"
            ),
            where=(
                f'(SG.Team1="{team}" OR SG.Team2="{team}") '
                f'AND SG.DateTime_UTC >= "{year}-01-01 00:00:00"'
            ),
            order_by="SG.DateTime_UTC DESC",
            limit=500,
        )
        if not rows:
            return {"games": 0, "winrate": 0.0, "total_kills_avg": 0.0,
                    "duration_avg_min": 0.0, "over_kills_pct": 0.0,
                    "under_kills_pct": 0.0, "over_time_pct": 0.0,
                    "under_time_pct": 0.0}

        totals: List[int] = []
        durations: List[float] = []
        wins = 0
        for r in rows:
            t1, t2, w = r.get("T1"), r.get("T2"), r.get("W")
            if (t1 == team and w == "1") or (t2 == team and w == "2"):
                wins += 1
            try:
                tk = int(r.get("K1") or 0) + int(r.get("K2") or 0)
                totals.append(tk)
            except ValueError:
                pass
            try:
                durations.append(float(r.get("Len") or 0))
            except ValueError:
                pass
        n = len(rows)

        def _median(xs: List[float]) -> float:
            if not xs:
                return 0.0
            s = sorted(xs)
            mid = len(s) // 2
            return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2

        median_kills = _median([float(x) for x in totals])
        median_time = _median(durations)
        over_k = sum(1 for x in totals if x > median_kills)
        under_k = sum(1 for x in totals if x < median_kills)
        over_t = sum(1 for x in durations if x > median_time)
        under_t = sum(1 for x in durations if x < median_time)

        return {
            "games": n,
            "winrate": round(100 * wins / n, 1),
            "total_kills_avg": round(sum(totals) / max(len(totals), 1), 2),
            "duration_avg_min": round(sum(durations) / max(len(durations), 1), 2),
            "over_kills_pct": round(100 * over_k / max(len(totals), 1), 1),
            "under_kills_pct": round(100 * under_k / max(len(totals), 1), 1),
            "over_time_pct": round(100 * over_t / max(len(durations), 1), 1),
            "under_time_pct": round(100 * under_t / max(len(durations), 1), 1),
        }

    def get_h2h(self, team_a: str, team_b: str, year: int) -> Dict[str, Any]:
        """Head-to-head: winrate de team_a contra team_b no ano."""
        rows = self.cargo(
            tables="ScoreboardGames=SG",
            fields="SG.Team1=T1, SG.Team2=T2, SG.Winner=W, SG.Team1Kills=K1, SG.Team2Kills=K2",
            where=(
                f'((SG.Team1="{team_a}" AND SG.Team2="{team_b}") '
                f'OR (SG.Team1="{team_b}" AND SG.Team2="{team_a}")) '
                f'AND SG.DateTime_UTC >= "{year}-01-01 00:00:00"'
            ),
            limit=500,
        )
        if not rows:
            return {"games": 0, "winrate_a": 0.0, "kills_diff": 0.0,
                    "deaths_diff": 0.0}
        n = len(rows)
        wins_a = 0
        kdiff = 0
        for r in rows:
            t1, t2 = r.get("T1"), r.get("T2")
            w = r.get("W")
            k1 = int(r.get("K1") or 0)
            k2 = int(r.get("K2") or 0)
            if t1 == team_a:
                if w == "1":
                    wins_a += 1
                kdiff += (k1 - k2)
            elif t2 == team_a:
                if w == "2":
                    wins_a += 1
                kdiff += (k2 - k1)
        return {
            "games": n,
            "winrate_a": round(100 * wins_a / n, 1),
            "kills_diff": round(kdiff / n, 2),
            # Mortes do A = kills do B (em jogos contra B)
            "deaths_diff": round(-kdiff / n, 2),
        }
