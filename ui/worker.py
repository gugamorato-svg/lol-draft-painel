"""QThread wrappers para chamadas HTTP nao travarem a UI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from PyQt6.QtCore import QThread, pyqtSignal

from api.esports import EsportsClient
from api.leaguepedia import LeaguepediaClient, ROLE_SHORT
from api.livefeed import LiveFeedClient
import requests

from api.lplqq import LplQqClient
from api.champion_icons import ChampionIconCache
from api.player_resolver import PlayerResolver, clean_live_name

_DRAFT_ROLES = ("top", "jng", "mid", "bot", "sup")


def _draft_is_complete(draft: Dict[str, Any]) -> bool:
    """True se o draft tem os 10 campeoes (5 por lado) preenchidos.

    Usado para detectar qual mapa de uma serie BO ja teve o draft publicado,
    contornando o lag do campo `state` da Riot.
    """
    if not draft:
        return False
    for side in ("blue", "red"):
        for r in _DRAFT_ROLES:
            champ = draft.get(f"{side}_{r}", "")
            if not champ or str(champ).strip() in ("", "-", "None"):
                return False
    return True


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:180]


def _empty_team_stats(status: str = "sem_dados", error: str = "") -> Dict[str, Any]:
    return {
        "games": 0,
        "total_kills_avg": 0.0,
        "duration_avg_min": 0.0,
        "winrate": 0.0,
        "over_kills_pct": 0.0,
        "under_kills_pct": 0.0,
        "over_time_pct": 0.0,
        "under_time_pct": 0.0,
        "data_status": status,
        "data_error": error,
    }


class MatchupWorker(QThread):
    """Busca dados completos pro confronto Blue x Red e emite o resultado."""

    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        leaguepedia: LeaguepediaClient,
        blue: str,
        red: str,
        year: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.lp = leaguepedia
        self.blue = blue
        self.red = red
        self.year = year

    def run(self) -> None:
        try:
            self.progress.emit("Buscando roster Blue...")
            blue_roster = self.lp.get_team_roster(self.blue, self.year)
            self.progress.emit("Buscando roster Red...")
            red_roster = self.lp.get_team_roster(self.red, self.year)

            self.progress.emit("Buscando stats agregadas Blue...")
            blue_team = self.lp.get_team_year_stats(self.blue, self.year)
            self.progress.emit("Buscando stats agregadas Red...")
            red_team = self.lp.get_team_year_stats(self.red, self.year)

            self.progress.emit("Calculando head-to-head...")
            h2h = self.lp.get_h2h(self.blue, self.red, self.year)

            blue_players = []
            for i, p in enumerate(blue_roster, 1):
                self.progress.emit(f"Stats Blue {i}/{len(blue_roster)}: {p['player']}")
                s = self.lp.get_player_year_stats(p["player"], self.year)
                blue_players.append({**p, **s})

            red_players = []
            for i, p in enumerate(red_roster, 1):
                self.progress.emit(f"Stats Red {i}/{len(red_roster)}: {p['player']}")
                s = self.lp.get_player_year_stats(p["player"], self.year)
                red_players.append({**p, **s})

            # Lane matchups (par por role)
            by_role_blue = {p["role"]: p for p in blue_players}
            by_role_red = {p["role"]: p for p in red_players}
            lane_matchups = []
            for role in ["Top", "Jungle", "Mid", "Bot", "Support"]:
                b = by_role_blue.get(role)
                r = by_role_red.get(role)
                if b and r:
                    lane_matchups.append(
                        {
                            "role": ROLE_SHORT[role],
                            "blue": b["player"],
                            "red": r["player"],
                            "delta_kills": round(
                                b["kills_avg"] - r["kills_avg"], 2
                            ),
                        }
                    )

            result = {
                "blue": {
                    "name": self.blue,
                    "team_stats": blue_team,
                    "players": blue_players,
                },
                "red": {
                    "name": self.red,
                    "team_stats": red_team,
                    "players": red_players,
                },
                "h2h": h2h,
                "lane_matchups": lane_matchups,
            }
            self.finished_ok.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


class LivePicksWorker(QThread):
    """Busca picks/players de uma partida ao vivo (ou recem-completada).

    Fluxo:
      1. get_event_details(event_id) -> games[]
      2. Escolhe game com state='inProgress' ou ultimo 'completed'
      3. livefeed.extract_draft(game_id) -> picks/players por role
      4. (opcional) Tambem busca stats agregadas da Leaguepedia em paralelo
    """

    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        esports: EsportsClient,
        livefeed: LiveFeedClient,
        leaguepedia: LeaguepediaClient,
        event_id: str,
        blue_name: str,
        red_name: str,
        year: int,
        parent=None,
        force_game_id: str = "",
    ) -> None:
        super().__init__(parent)
        self.es = esports
        self.lf = livefeed
        self.lp = leaguepedia
        self.event_id = event_id
        self.blue = blue_name
        self.red = red_name
        self.year = year
        # Quem chama ja sabe qual mapa quer (draft recem-publicado). Pula a
        # sondagem, que exige frames de estatistica (~90s+ de atraso) e atrasava a
        # captura justo no inicio do jogo, logo depois do draft.
        self.force_game_id = str(force_game_id or "")

    def run(self) -> None:
        try:
            if str(self.event_id).startswith("lplqq:"):
                self.progress.emit("Buscando detalhes LPL/Tencent...")
                lpl = LplQqClient()
                try:
                    draft = lpl.extract_draft(self.event_id)
                except requests.RequestException:
                    # 15/10: servidor chines lento/fora -> antes saia o erro cru do requests
                    self.failed.emit(
                        "O servidor da Tencent (LPL) nao respondeu agora.\n"
                        "Tente de novo em alguns segundos. Se a partida tambem aparece na\n"
                        "lista pela Riot (ex.: Worlds), use essa."
                    )
                    return
                if not draft:
                    # A Tencent so atualiza matchInfos depois que cada mapa
                    # ACABA. Entre mapas (durante o draft do proximo), o draft
                    # nao esta disponivel. Avisa com mensagem util.
                    self.failed.emit(
                        "Mapa em draft (LPL/Tencent ainda nao publicou os picks).\n"
                        "Tente novamente daqui ~30s, ou apos o draft fechar."
                    )
                    return
                self.blue = draft.get("blue_team_name") or self.blue
                self.red = draft.get("red_team_name") or self.red
                game_id = draft.get("game_id", self.event_id)
                game_state = "inProgress"
                draft["league"] = "LPL"  # endpoint Tencent so cobre LPL
            else:
                self.progress.emit("Buscando detalhes do evento ao vivo...")
                # fresh=True: ignora cache. Numa serie BO, quando o mapa 1
                # termina e o mapa 2 comeca, o cache mostraria o estado antigo
                # e o draft ficaria preso no mapa 1.
                details = self.es.get_event_details(self.event_id, fresh=True)
                games = (details.get("match") or {}).get("games", [])
                if not games:
                    self.failed.emit("Evento sem games - talvez ainda nao comecou.")
                    return

                # IMPORTANTE: o campo `state` da Riot TEM LAG. Numa serie BO,
                # o mapa 1 ja terminou e o mapa 2 esta em draft, mas a API ainda
                # reporta game 1 = inProgress por varios minutos. Por isso NAO
                # confiamos no state. Sondamos o live feed de cada game, do MAIOR
                # numero pro menor, e pegamos o mapa mais avancado que ja tem um
                # draft real publicado = o mapa atual.
                # NAO confiar no `state` da Riot (tem lag: Mapa 1 fica 'inProgress'
                # minutos apos acabar). Em vez disso, olhar o `gameState` do ULTIMO
                # frame do feed de cada mapa: 'finished' = acabou, 'in_game' = vivo.
                # Escolhe o mapa VIVO de maior numero; se nenhum vivo alcancavel,
                # cai no de maior numero com draft (fallback).
                with_id = sorted(
                    [g for g in games if g.get("id")],
                    key=lambda g: g.get("number", 0), reverse=True,
                )
                target = draft = None
                fallback = None          # (game, draft) do maior mapa finalizado
                unready = []              # numeros de mapa sem draft no feed ainda
                if self.force_game_id:
                    forced = [g for g in with_id if str(g.get("id")) == self.force_game_id]
                    d = self.lf.extract_draft(self.force_game_id) if forced else None
                    if not (d and _draft_is_complete(d)):
                        self.failed.emit("Mapa pedido ainda sem draft no feed.")
                        return
                    target, draft = forced[0], d
                    with_id = []          # nada a sondar
                for g in with_id:
                    gid = g.get("id", "")
                    self.progress.emit(f"Sondando mapa {g.get('number')}...")
                    d = self.lf.extract_draft(gid)
                    if not (d and _draft_is_complete(d)):
                        if g.get("state") != "unneeded":  # serie ja decidida
                            unready.append(g.get("number", 0))
                        continue
                    fr = self.lf._latest_window_frame(gid)
                    gstate = str((fr or {}).get("gameState", "")).lower()
                    if fr is not None and gstate and gstate != "finished":
                        target, draft = g, d      # mapa VIVO de maior numero
                        break
                    if fallback is None:
                        fallback = (g, d)          # mapa finalizado mais alto

                if target is None:
                    # Nenhum mapa vivo alcancavel. Se ha um mapa mais novo iniciado
                    # mas ainda sem feed (transicao/atraso), avisa em vez de mostrar
                    # o mapa anterior finalizado.
                    # O PROXIMO mapa, nao o maior: numa MD5 os mapas 3-5 ainda nao
                    # jogados tambem vem com id e sem feed -> dizia "Mapa 5".
                    fb_number = fallback[0].get("number", 0) if fallback else 0
                    next_map = min((n for n in unready if n > fb_number), default=None)
                    if fallback is not None and next_map is not None:
                        self.failed.emit(
                            f"Mapa {next_map} em transição — o feed ainda não "
                            f"publicou (atraso ~3min). Tente de novo em ~30s."
                        )
                        return
                    if fallback is not None:
                        target, draft = fallback   # so ha mapa finalizado -> mostra
                    else:
                        self.failed.emit(
                            "Nenhum mapa com draft publicado ainda - aguarde o draft."
                        )
                        return

                game_id = target.get("id", "")
                game_state = target.get("state", "?")
                self._sync_real_sides(details, target)
                # Injeta a liga (LCK, LEC, LPL...) para o feature
                # league_champion_wr_diff funcionar em live.
                draft["league"] = (details.get("league") or {}).get("name", "")
                self.progress.emit(
                    f"Mapa {target.get('number')} (game {game_id}) - draft OK"
                )

            self.progress.emit("Buscando stats agregadas Blue (Leaguepedia)...")
            try:
                blue_team_stats = self.lp.get_team_year_stats(self.blue, self.year)
                blue_team_stats.setdefault("data_status", "ok")
            except Exception as exc:
                blue_team_stats = _empty_team_stats("team_stats_error", _short_error(exc))

            self.progress.emit("Buscando stats agregadas Red (Leaguepedia)...")
            try:
                red_team_stats = self.lp.get_team_year_stats(self.red, self.year)
                red_team_stats.setdefault("data_status", "ok")
            except Exception as exc:
                red_team_stats = _empty_team_stats("team_stats_error", _short_error(exc))

            # Roster (precisamos do player_link da Leaguepedia para queries por player).
            # role_full mapeia sigla livefeed -> nome Leaguepedia.
            role_full = {"top": "Top", "jng": "Jungle", "mid": "Mid",
                         "bot": "Bot", "sup": "Support"}
            try:
                self.progress.emit("Buscando roster Blue (Leaguepedia)...")
                blue_roster = self.lp.get_team_roster(self.blue, self.year)
            except Exception:
                blue_roster = []
            try:
                self.progress.emit("Buscando roster Red (Leaguepedia)...")
                red_roster = self.lp.get_team_roster(self.red, self.year)
            except Exception:
                red_roster = []

            # Indexa roster por role (Leaguepedia usa Top/Jungle/Mid/Bot/Support)
            blue_roster_by_role = {p["role"]: p for p in blue_roster}
            red_roster_by_role = {p["role"]: p for p in red_roster}

            resolver = PlayerResolver(self.lp)

            def _player_stats(
                primary_player: str,
                champion: str,
                *,
                team: str,
                role: str,
                fallback_player: str = "",
            ) -> dict:
                """Busca stats usando o player canonico resolvido por contexto."""
                out = {
                    "kills_avg": 0.0,
                    "deaths_avg": 0.0,
                    "assists_avg": 0.0,
                    "games_player": 0,
                    "kills_avg_champ": 0.0,
                    "deaths_avg_champ": 0.0,
                    "assists_avg_champ": 0.0,
                    "games_champ": 0,
                    "wins_champ": 0,
                    "games_champ_year": 0,
                    "wins_champ_year": 0,
                    "resolved_player": "",
                    "display_player": clean_live_name(primary_player),
                    "resolve_reason": "unresolved",
                    "data_status": "unresolved",
                    "data_warnings": [],
                    "status_detail": "",
                }

                def warn(kind: str, detail: str = "") -> None:
                    text = kind if not detail else f"{kind}: {detail}"
                    if text not in out["data_warnings"]:
                        out["data_warnings"].append(text)

                def finalize() -> dict:
                    if not out["resolved_player"]:
                        out["data_status"] = "player_nao_resolvido"
                        warn("player_nao_resolvido", ", ".join(resolution.candidates[:4]))
                    elif out["games_player"] and (not champion or out["games_champ"]):
                        out["data_status"] = "ok"
                    elif out["games_player"] or out["games_champ"]:
                        out["data_status"] = "parcial"
                    else:
                        out["data_status"] = "sem_amostra"
                    out["status_detail"] = " | ".join(out["data_warnings"])
                    return out

                resolution = resolver.resolve(
                    primary_player,
                    team=team,
                    role=role,
                    year=self.year,
                    fallback_name=fallback_player,
                )
                out["resolved_player"] = resolution.canonical
                out["resolved_player_id"] = resolution.player_id
                out["display_player"] = resolution.display
                out["resolve_reason"] = resolution.reason
                if not resolution.canonical:
                    return finalize()

                try:
                    ys = self.lp.get_player_year_stats(resolution.canonical, self.year)
                    if ys.get("games", 0) > 0:
                        out["kills_avg"] = ys.get("kills_avg", 0.0)
                        out["deaths_avg"] = ys.get("deaths_avg", 0.0)
                        out["assists_avg"] = ys.get("assists_avg", 0.0)
                        out["games_player"] = ys.get("games", 0)
                    else:
                        warn("sem_stats_player_ano", str(self.year))
                except Exception as exc:
                    warn("erro_stats_player_ano", _short_error(exc))

                if champion:
                    annual_cs = None
                    try:
                        annual_cs = self.lp.get_player_champion_stats(resolution.canonical, champion, self.year)
                        if annual_cs.get("games", 0) > 0:
                            out["games_champ_year"] = annual_cs.get("games", 0)
                            out["wins_champ_year"] = annual_cs.get("wins", 0)
                        else:
                            warn("sem_wr_champ_ano", f"{champion} {self.year}")
                    except Exception as exc:
                        warn("erro_wr_champ_ano", _short_error(exc))
                        annual_cs = None

                    try:
                        if hasattr(self.lp, "get_player_champion_career_stats"):
                            career_cs = self.lp.get_player_champion_career_stats(resolution.canonical, champion)
                        else:
                            career_cs = annual_cs or {}
                        if career_cs.get("games", 0) > 0:
                            out["kills_avg_champ"] = career_cs.get("kills_avg", 0.0)
                            out["deaths_avg_champ"] = career_cs.get("deaths_avg", 0.0)
                            out["assists_avg_champ"] = career_cs.get("assists_avg", 0.0)
                            out["games_champ"] = career_cs.get("games", 0)
                            out["wins_champ"] = career_cs.get("wins", 0)
                        else:
                            warn("sem_stats_champ_carreira", champion)
                    except Exception as exc:
                        warn("erro_stats_champ_carreira", _short_error(exc))
                else:
                    warn("champion_vazio")
                return finalize()

            icon_cache = ChampionIconCache()

            # WR do campeao no patch DA PARTIDA (ligas migram de patch em datas
            # diferentes, entao vem do feed e nao do "patch atual" global).
            patch_fn = getattr(self.lp, "get_champion_patch_stats", None)

            def _patch_wr(champ: str):
                if not (patch_fn and champ):
                    return None
                try:
                    return patch_fn(champ, draft.get("patch", ""))
                except Exception:
                    return None

            # Mapeia players da live feed para formato esperado pelo UI
            blue_players = []
            red_players = []
            roles_short = ("top", "jng", "mid", "bot", "sup")
            for r in roles_short:
                role_lp = role_full[r]
                blue_champ = draft.get(f"blue_{r}", "")
                red_champ = draft.get(f"red_{r}", "")
                blue_live_raw = draft.get(f"blue_playername_{r}", "-")
                red_live_raw = draft.get(f"red_playername_{r}", "-")
                blue_live_player = clean_live_name(blue_live_raw)
                red_live_player = clean_live_name(red_live_raw)
                blue_link = blue_roster_by_role.get(role_lp, {}).get("player", "")
                red_link = red_roster_by_role.get(role_lp, {}).get("player", "")

                self.progress.emit(f"Stats player Blue {role_lp}...")
                blue_stats = _player_stats(
                    blue_live_raw, blue_champ, team=self.blue, role=role_lp, fallback_player=blue_link
                )
                self.progress.emit(f"Stats player Red {role_lp}...")
                red_stats = _player_stats(
                    red_live_raw, red_champ, team=self.red, role=role_lp, fallback_player=red_link
                )

                bp = {
                    "player":           blue_stats.get("display_player") or blue_live_player or "-",
                    "resolved_player":  blue_stats.get("resolved_player", ""),
                    "resolved_player_id": blue_stats.get("resolved_player_id", ""),
                    "resolve_reason":   blue_stats.get("resolve_reason", ""),
                    "role":             role_lp,
                    "champion_name":    blue_champ,
                    "kills_avg":        blue_stats["kills_avg"],
                    "deaths_avg":       blue_stats["deaths_avg"],
                    "assists_avg":      blue_stats["assists_avg"],
                    "games_player":     blue_stats["games_player"],
                    "kills_avg_champ":  blue_stats["kills_avg_champ"],
                    "deaths_avg_champ": blue_stats["deaths_avg_champ"],
                    "assists_avg_champ": blue_stats["assists_avg_champ"],
                    "games_champ":      blue_stats["games_champ"],
                    "wins_champ":       blue_stats["wins_champ"],
                    "games_champ_year": blue_stats["games_champ_year"],
                    "wins_champ_year":  blue_stats["wins_champ_year"],
                    "champion_icon":    icon_cache.path_for(blue_champ),
                    "patch_wr":         _patch_wr(blue_champ),
                    "data_status":      blue_stats.get("data_status", ""),
                    "data_warnings":    blue_stats.get("data_warnings", []),
                    "status_detail":    blue_stats.get("status_detail", ""),
                }
                rp = {
                    "player":           red_stats.get("display_player") or red_live_player or "-",
                    "resolved_player":  red_stats.get("resolved_player", ""),
                    "resolved_player_id": red_stats.get("resolved_player_id", ""),
                    "resolve_reason":   red_stats.get("resolve_reason", ""),
                    "role":             role_lp,
                    "champion_name":    red_champ,
                    "kills_avg":        red_stats["kills_avg"],
                    "deaths_avg":       red_stats["deaths_avg"],
                    "assists_avg":      red_stats["assists_avg"],
                    "games_player":     red_stats["games_player"],
                    "kills_avg_champ":  red_stats["kills_avg_champ"],
                    "deaths_avg_champ": red_stats["deaths_avg_champ"],
                    "assists_avg_champ": red_stats["assists_avg_champ"],
                    "games_champ":      red_stats["games_champ"],
                    "wins_champ":       red_stats["wins_champ"],
                    "games_champ_year": red_stats["games_champ_year"],
                    "wins_champ_year":  red_stats["wins_champ_year"],
                    "champion_icon":    icon_cache.path_for(red_champ),
                    "patch_wr":         _patch_wr(red_champ),
                    "data_status":      red_stats.get("data_status", ""),
                    "data_warnings":    red_stats.get("data_warnings", []),
                    "status_detail":    red_stats.get("status_detail", ""),
                }
                blue_players.append(bp)
                red_players.append(rp)

            lane_matchups = []
            for bp, rp in zip(blue_players, red_players):
                delta = None
                if bp.get("games_champ", 0) > 0 and rp.get("games_champ", 0) > 0:
                    delta = round(bp["kills_avg_champ"] - rp["kills_avg_champ"], 2)
                lane_matchups.append(
                    {
                        "role": ROLE_SHORT.get(bp["role"], bp["role"]),
                        "blue": bp["player"],
                        "red": rp["player"],
                        "delta_kills": delta,
                    }
                )

            # H2H (do snapshot via predict, mas tambem busca real)
            try:
                h2h = self.lp.get_h2h(self.blue, self.red, self.year)
            except Exception:
                h2h = {"games": 0, "winrate_a": 50.0, "kills_diff": 0.0, "deaths_diff": 0.0}

            # Monta result no mesmo formato do MatchupWorker + extras
            result = {
                "blue": {
                    "name":        self.blue,
                    "team_stats":  blue_team_stats,
                    "players":     blue_players,
                },
                "red": {
                    "name":        self.red,
                    "team_stats":  red_team_stats,
                    "players":     red_players,
                },
                "h2h":           h2h,
                "lane_matchups": lane_matchups,
                "live_draft":    draft,   # <-- NOVO: payload para predict_full
                "live_event_id": self.event_id,
                "live_game_id":  game_id,
                "live_game_state": game_state,
            }
            self.finished_ok.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _sync_real_sides(self, details: Dict[str, Any], target: Dict[str, Any]) -> None:
        """Usa game.teams[].side para corrigir Blue/Red real do live."""
        match = details.get("match") or {}
        teams = match.get("teams") or []
        teams_by_id = {str(team.get("id", "")): team for team in teams}
        blue_id = ""
        red_id = ""
        for team in target.get("teams", []) or []:
            side = str(team.get("side", "")).lower()
            team_id = str(team.get("id", ""))
            if side == "blue":
                blue_id = team_id
            elif side == "red":
                red_id = team_id
        if blue_id and blue_id in teams_by_id:
            self.blue = teams_by_id[blue_id].get("name") or teams_by_id[blue_id].get("code") or self.blue
        if red_id and red_id in teams_by_id:
            self.red = teams_by_id[red_id].get("name") or teams_by_id[red_id].get("code") or self.red


class LiveItemsWorker(QThread):
    """Poll dos ITENS ao vivo de um game_id, resolvendo os icones no thread.

    Emite {"blue_top": [icon_path, ...], ..., "red_sup": [...]} a cada `interval`s.
    Resolve os icones AQUI (nao na UI) para nao travar a interface no 1o download.
    """

    items_ready = pyqtSignal(dict)

    def __init__(self, livefeed, item_icons, game_id: str,
                 roles=("top", "jng", "mid", "bot", "sup"),
                 interval: int = 20, parent=None) -> None:
        super().__init__(parent)
        self.lf = livefeed
        self.icons = item_icons
        self.game_id = game_id
        self.roles = roles
        self.interval = interval
        self._running = True

    def run(self) -> None:
        while self._running:
            try:
                raw = self.lf.extract_items(self.game_id)
                if raw:
                    resolved: Dict[str, Any] = {}
                    for side in ("blue", "red"):
                        for role in self.roles:
                            ids = raw.get(f"{side}_{role}_items", []) or []
                            paths = [self.icons.path_for(i) for i in ids]
                            resolved[f"{side}_{role}"] = [p for p in paths if p]
                            resolved[f"{side}_{role}_level"] = raw.get(f"{side}_{role}_level")
                    if self._running:
                        self.items_ready.emit(resolved)
            except Exception:
                pass
            # dorme em fatias curtas para parar rapido quando stop() e chamado
            for _ in range(max(1, self.interval) * 2):
                if not self._running:
                    break
                self.msleep(500)

    def stop(self) -> None:
        self._running = False


class LiveStateWorker(QThread):
    """Poll do ESTADO do jogo + WR ao vivo (estado + comp).

    Trabalha na perspectiva do FEED (blue do feed), com p_comp fixo = P(feed_blue
    vence) pre-draft. O display no main_window lida com o swap manual de lados.
    Emite (wr, state) a cada `interval`s.
    """

    state_ready = pyqtSignal(dict, dict)

    def __init__(self, livefeed, game_id: str, p_comp_blue: float,
                 interval: int = 20, parent=None) -> None:
        super().__init__(parent)
        self.lf = livefeed
        self.game_id = game_id
        self.p_comp = p_comp_blue
        self.interval = interval
        self._running = True

    def run(self) -> None:
        from models import live_wr
        from data_pipeline.live_logger import append_snapshot
        while self._running:
            try:
                st = self.lf.extract_state(self.game_id)
                if st:
                    wr = live_wr.compute(st, self.p_comp)
                    if wr and self._running:
                        self.state_ready.emit(wr, st)
                        append_snapshot(self.game_id, st, wr, self.p_comp)
            except Exception:
                pass
            for _ in range(max(1, self.interval) * 2):
                if not self._running:
                    break
                self.msleep(500)

    def stop(self) -> None:
        self._running = False


class LiveWorker(QThread):
    """Carrega partidas ao vivo + proximas da Esports API."""

    finished_ok = pyqtSignal(list, list)  # live, upcoming
    failed = pyqtSignal(str)

    def __init__(self, esports: EsportsClient, parent=None) -> None:
        super().__init__(parent)
        self.esports = esports

    def run(self) -> None:
        try:
            live = self.esports.get_live()
            sched = self.esports.get_schedule()
            events = sched.get("events", []) if sched else []
            # mantem so unstarted/upcoming nos proximos eventos
            upcoming = [e for e in events if e.get("state") in ("unstarted", "inProgress")]
            self.finished_ok.emit(live, upcoming[:20])
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")



