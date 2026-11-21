"""Fallback Tencent/LPL para partidas ao vivo que nao aparecem no LoL Esports.

A LPL nem sempre publica o live no getLive da Riot/LoL Esports. O site
oficial chines da LPL usa endpoints da Tencent/TJStats; este cliente converte
esses dados para o formato que o painel ja entende.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from api.livefeed import _lf_champion_to_oe
from api.player_resolver import clean_live_name


LPL_TZ = timezone(timedelta(hours=8))
DEFAULT_LPL_AUTH_TOKEN = "7935be4c41d8760a28c05581a7b1f570"
LPL_AUTH_TOKEN = os.getenv("LOL_LPL_AUTH_TOKEN", DEFAULT_LPL_AUTH_TOKEN)

ROLE_MAP = {
    "TOP": "top",
    "JUG": "jng",
    "JNG": "jng",
    "JUN": "jng",
    "MID": "mid",
    "ADC": "bot",
    "BOT": "bot",
    "SUP": "sup",
}


class LplQqClient:
    MATCH_API = "https://apps.game.qq.com/lol/match/apis/searchBMatchInfo_bak.php"
    DETAIL_API = "https://open.tjstats.com/match-auth-app/open/v1/compound/matchDetail"
    TEAM_LIST_URL = "https://lpl.qq.com/web201612/data/LOL_MATCH2_TEAM_LIST.js"

    def __init__(self, timeout: float = 12.0) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 LoLPainelBot/0.1"})
        self._teams: Optional[Dict[str, Dict[str, Any]]] = None

    def live_events(self) -> List[Dict[str, Any]]:
        today = datetime.now(UTC).astimezone(LPL_TZ).date()
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for offset in (-1, 0, 1):
            for row in self._match_rows_for_day(today + timedelta(days=offset)):
                match_id = str(row.get("bMatchId", ""))
                if not match_id or match_id in seen:
                    continue
                seen.add(match_id)
                rows.append(row)

        live_rows = [row for row in rows if _to_int(row.get("MatchStatus")) == 2]
        events: List[Dict[str, Any]] = []
        for row in live_rows:
            try:
                events.append(self._row_to_event(row))
            except Exception:
                continue
        return events

    def extract_draft(self, event_id_or_match_id: str) -> Optional[Dict[str, Any]]:
        match_id = _clean_match_id(event_id_or_match_id)
        detail = self.match_detail(match_id)
        game = _current_game_info(detail)
        if not game:
            return None

        teams = self._team_metadata()
        out: Dict[str, Any] = {
            "blue_bans": [],
            "red_bans": [],
            "blue_picks_draft": [],
            "red_picks_draft": [],
            "patch": str(game.get("patch") or game.get("gameVersion") or ""),
            "game_id": f"lplqq:{match_id}:{_to_int(game.get('bo'), 1)}",
            "match_id": match_id,
        }

        for team_info in game.get("teamInfos", []) or []:
            team_id = str(team_info.get("teamId", ""))
            side = "blue" if _to_int(team_id) == _to_int(game.get("blueTeam")) else "red"
            meta = teams.get(team_id, {})
            team_name = _team_name(meta, team_id)
            team_code = str(meta.get("TeamShortName") or meta.get("TeamName") or team_id)
            out[f"{side}_team_name"] = team_name
            out[f"{side}_team_code"] = team_code
            out[f"{side}_teamid_live"] = team_id

            picks: List[str] = []
            for player in team_info.get("playerInfos", []) or []:
                role = ROLE_MAP.get(str(player.get("playerLocation", "")).upper(), "")
                if not role:
                    continue
                champion = _lf_champion_to_oe(str(
                    player.get("heroNameEn")
                    or player.get("heroTitle")
                    or player.get("heroName")
                    or ""
                ))  # mesmo nome do Oracle que o feed da Riot (ex.: Renata -> Renata Glasc)
                raw_name = str(player.get("playerName", ""))
                player_name = _clean_player_name(raw_name, team_code)
                out[f"{side}_{role}"] = champion
                out[f"{side}_player_{role}"] = str(player.get("playerId", ""))
                out[f"{side}_playername_{role}"] = player_name
                picks.append(champion)
            out[f"{side}_picks_draft"] = picks

        return out

    def match_detail(self, match_id: str) -> Dict[str, Any]:
        response = self.session.get(
            self.DETAIL_API,
            params={"matchId": _clean_match_id(match_id)},
            headers={
                "accept": "application/json",
                "Authorization": LPL_AUTH_TOKEN,
                "Referer": f"https://lpl.qq.com/es/stats.shtml?bmid={_clean_match_id(match_id)}",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("data", {}) or {}

    def _match_rows_for_day(self, day) -> List[Dict[str, Any]]:
        start = f"{day.isoformat()} 00:00:00"
        end = f"{day.isoformat()} 23:59:59"
        response = self.session.get(
            self.MATCH_API,
            params={
                "p8": "",
                "p1": "",
                "p9": start,
                "p10": end,
                "p6": "3",
                "page": "1",
                "pagesize": "50",
                "r1": "retObj",
                "_": str(int(datetime.now(UTC).timestamp() * 1000)),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = _parse_jsonp(response.text, "retObj")
        msg = data.get("msg") or {}
        if isinstance(msg, dict):
            return list(msg.get("result") or [])
        return list(data.get("result") or [])

    def _row_to_event(self, row: Dict[str, Any]) -> Dict[str, Any]:
        match_id = str(row.get("bMatchId", ""))
        detail: Dict[str, Any] = {}
        try:
            detail = self.match_detail(match_id)
        except Exception:
            pass

        teams = self._team_metadata()
        team_a = _team_from_id(str(row.get("TeamA", "")), teams)
        team_b = _team_from_id(str(row.get("TeamB", "")), teams)
        blue = team_a
        red = team_b

        game = _current_game_info(detail)
        if game:
            blue_id = str(game.get("blueTeam", ""))
            if blue_id == team_b.get("id"):
                blue, red = team_b, team_a

        game_number = _to_int(game.get("bo") if game else None, 1)
        return {
            "id": f"lplqq:{match_id}",
            "startTime": str(row.get("MatchDate", "")),
            "state": "inProgress",
            "type": "match",
            "blockName": str(row.get("GameTypeName") or row.get("GameProcName") or "LPL"),
            "league": {"id": "lplqq", "name": "LPL"},
            "match": {
                "id": f"lplqq:{match_id}",
                "teams": [blue, red],
                "games": [
                    {
                        "number": game_number,
                        "id": f"lplqq:{match_id}:{game_number}",
                        "state": "inProgress",
                        "teams": [
                            {"id": blue.get("id", ""), "side": "blue"},
                            {"id": red.get("id", ""), "side": "red"},
                        ],
                    }
                ],
            },
        }

    def _team_metadata(self) -> Dict[str, Dict[str, Any]]:
        if self._teams is not None:
            return self._teams
        response = self.session.get(self.TEAM_LIST_URL, timeout=self.timeout)
        response.raise_for_status()
        data = _parse_jsonp(response.text, "TeamList")
        self._teams = dict((data.get("msg") or {}))
        return self._teams



def _clean_player_name(raw_name: str, team_code: str) -> str:
    cleaned = clean_live_name(raw_name) or str(raw_name or "").strip()
    code = str(team_code or "").strip()
    if code and cleaned.casefold().startswith(code.casefold()) and len(cleaned) > len(code) + 1:
        return cleaned[len(code):].strip()
    return cleaned
def _game_has_draft(game: Dict[str, Any]) -> bool:
    """True se ambos os times tem 5 campeoes definidos nesse mapa."""
    team_infos = game.get("teamInfos", []) or []
    if len(team_infos) < 2:
        return False
    for team in team_infos:
        champs = [
            str(p.get("heroNameEn") or p.get("heroTitle") or p.get("heroName") or "").strip()
            for p in (team.get("playerInfos", []) or [])
        ]
        champs = [c for c in champs if c]
        if len(champs) < 5:
            return False
    return True


def _current_game_info(detail: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Seleciona o mapa ATUAL de uma serie BO.

    Heuristica:
      - Tenta pegar o mapa de MAIOR numero (bo) que ja tem draft completo.
      - Se o ultimo mapa exposto JA TERMINOU (tem matchWin) E a serie ainda
        esta em andamento (top-level matchStatus=1) E o placar mostra mais
        mapas a jogar, ESTAMOS NO GAP entre mapas: retorna None.
        Isso evita mostrar o draft do mapa anterior como se fosse o atual.
    """
    games = list(detail.get("matchInfos", []) or [])
    if not games:
        return None

    # Detecta o "gap entre mapas":
    # - serie em andamento (matchStatus top-level == 1)
    # - todos os mapas expostos ja tem matchWin (todos terminaram)
    # - placar indica que ainda ha mapas a jogar (sum scores >= bo count)
    series_in_progress = _to_int(detail.get("matchStatus")) == 1
    score_sum = _to_int(detail.get("teamAScore")) + _to_int(detail.get("teamBScore"))
    all_have_winner = all(
        _to_int(g.get("matchWin")) > 0 for g in games
    )
    if series_in_progress and all_have_winner and score_sum >= len(games):
        # Estamos no draft do proximo mapa, que a Tencent ainda nao publicou
        return None

    games_sorted = sorted(games, key=lambda g: _to_int(g.get("bo"), 0), reverse=True)
    for game in games_sorted:
        if _game_has_draft(game):
            return game
    # nenhum mapa com draft completo: fallback no matchStatus==1, senao ultimo
    for game in games:
        if _to_int(game.get("matchStatus")) == 1:
            return game
    return games[-1]


def _team_from_id(team_id: str, teams: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    meta = teams.get(str(team_id), {})
    image = str(meta.get("TeamLogo") or "")
    if image.startswith("//"):
        image = f"https:{image}"
    return {
        "id": str(team_id),
        "name": _team_name(meta, team_id),
        "code": str(meta.get("TeamShortName") or meta.get("TeamName") or team_id),
        "image": image,
    }


def _team_name(meta: Dict[str, Any], fallback: str) -> str:
    return str(meta.get("TeamEnName") or meta.get("TeamName") or meta.get("TeamShortName") or fallback)


def _parse_jsonp(text: str, variable: str) -> Dict[str, Any]:
    clean = text.strip()
    prefix = f"var {variable}="
    if clean.startswith(prefix):
        clean = clean[len(prefix):]
    if clean.endswith(";"):
        clean = clean[:-1]
    return json.loads(clean)


def _clean_match_id(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("lplqq:"):
        return text.split(":", 1)[1].split(":", 1)[0]
    return text.split(":", 1)[0]


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default



