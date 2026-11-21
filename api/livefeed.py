"""Cliente para o Live Stats Feed da Riot Esports.

Quando uma partida esta ao vivo, o getEventDetails retorna game IDs.
Esses IDs sao consultados em:
  https://feed.lolesports.com/livestats/v1/window/{gameId}
  https://feed.lolesports.com/livestats/v1/details/{gameId}

O endpoint window retorna snapshots em multiplos de 10 segundos do estado
da partida (gold, kills, towers, dragoes, baroes). O details retorna stats
por jogador (cs, damage, items, runes).

Os timestamps precisam ser arredondados pra baixo no segundo multiplo de 10.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import requests

FEED_BASE = "https://feed.lolesports.com/livestats/v1"


def _floor_to_10s(dt: datetime) -> datetime:
    sec = (dt.second // 10) * 10
    return dt.replace(second=sec, microsecond=0)


class LiveFeedClient:
    def __init__(self, timeout: float = 8.0) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "LoLPainelBot/0.1"})
        self.timeout = timeout
        self._last_good_delay: Optional[int] = None  # atraso do feed que funcionou
        self._item_data = None  # ItemData lazy (custo/completo de item p/ gold efetivo)

    def window(self, game_id: str, starting_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Snapshot do estado do jogo. Se starting_time nao for dado, pega 'agora menos 30s'."""
        if starting_time is None:
            starting_time = datetime.now(timezone.utc) - timedelta(seconds=30)
        ts = _floor_to_10s(starting_time).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = f"{FEED_BASE}/window/{game_id}"
        resp = self.session.get(url, params={"startingTime": ts}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def details(self, game_id: str, starting_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Stats por jogador (CS, dano, itens, runas)."""
        if starting_time is None:
            starting_time = datetime.now(timezone.utc) - timedelta(seconds=30)
        ts = _floor_to_10s(starting_time).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = f"{FEED_BASE}/details/{game_id}"
        resp = self.session.get(url, params={"startingTime": ts}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def extract_draft(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Extrai picks/roles/players da window() sem startingTime.

        Funciona tanto para jogos em andamento quanto recem-completados.
        Retorna None se nao houver dados.

        Estrutura retornada (compativel com temporal_context.build_features):
            {
                "blue_top": "K'Sante", "blue_jng": "Xin Zhao", ...
                "red_top": ..., ...,
                "blue_picks_draft": [...]   # ordem TOP, JNG, MID, BOT, SUP (proxy)
                "red_picks_draft": [...],
                "blue_bans": [],            # livefeed nao expoe bans
                "red_bans": [],
                "blue_player_top": "esportsPlayerId", ...,
                "blue_playername_top": "WECube", ...,
                "blue_teamid_live": "98767991887166787",  # esportsTeamId
                "red_teamid_live":  "...",
                "patch":  "16.9",
            }
        """
        url = f"{FEED_BASE}/window/{game_id}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
        except Exception:
            return None
        if resp.status_code != 200 or not resp.text:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None

        gm = data.get("gameMetadata", {})
        if not gm:
            return None

        out: Dict[str, Any] = {
            "blue_bans":         [],
            "red_bans":          [],
            "blue_picks_draft":  [],
            "red_picks_draft":   [],
            "patch":             _short_patch(gm.get("patchVersion", "")),
        }

        # Mapeia role livefeed -> sigla do nosso modelo
        role_map = {
            "top":     "top",
            "jungle":  "jng",
            "mid":     "mid",
            "bottom":  "bot",
            "support": "sup",
        }

        for side, key in (("blue", "blueTeamMetadata"), ("red", "redTeamMetadata")):
            tm = gm.get(key, {})
            out[f"{side}_teamid_live"] = tm.get("esportsTeamId", "")
            picks_in_order = []
            for p in tm.get("participantMetadata", []):
                role_lf = p.get("role", "").lower()
                role = role_map.get(role_lf, "")
                champ_lf = p.get("championId", "")
                champ_oe = _lf_champion_to_oe(champ_lf)
                pid = p.get("esportsPlayerId", "")
                pname = p.get("summonerName", "")
                if role:
                    out[f"{side}_{role}"]            = champ_oe
                    out[f"{side}_player_{role}"]     = pid
                    out[f"{side}_playername_{role}"] = pname
                picks_in_order.append(champ_oe)
            out[f"{side}_picks_draft"] = picks_in_order

        # Sem startingTime o window devolve os PRIMEIROS frames do jogo -> o 1o frame e
        # o inicio da partida no feed (conferido 22/09: draft visivel ~17s apos o inicio).
        frames = data.get("frames") or []
        out["first_frame_ts"] = frames[0].get("rfc460Timestamp", "") if frames else ""
        return out

    def _latest_details(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Acha o frame de details MAIS RECENTE disponivel.

        O feed de esports tem atraso variavel (ERL chega a ~3-4min). Pedir
        `now-30s` retorna 400 (futuro relativo ao ultimo frame). Sondamos varios
        atrasos ate achar 200 com frames; o 1o que funciona = mais recente.
        Cacheia o atraso que funcionou p/ tentar primeiro na proxima chamada.
        """
        now = datetime.now(timezone.utc)
        delays = [90, 120, 150, 180, 210, 240, 300, 360]
        if self._last_good_delay:
            delays = [self._last_good_delay] + [d for d in delays if d != self._last_good_delay]
        for delay in delays:
            ts = _floor_to_10s(now - timedelta(seconds=delay)).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                r = self.session.get(f"{FEED_BASE}/details/{game_id}",
                                     params={"startingTime": ts}, timeout=self.timeout)
            except Exception:
                continue
            if r.status_code == 200:
                try:
                    j = r.json()
                except ValueError:
                    continue
                if j.get("frames"):
                    self._last_good_delay = delay
                    return j
            # 400 = recente demais (aumenta atraso); 204 = antes do jogo (para)
            if r.status_code == 204:
                break
        return None

    # XP cumulativo p/ ATINGIR cada nivel (indice = nivel 1..18). O window nao
    # expoe xp; derivamos do `level` de cada player (do details) por esta tabela.
    _XP_AT_LEVEL = [0, 0, 280, 660, 1140, 1720, 2400, 3180, 4060, 5040, 6120,
                    7300, 8580, 9960, 11440, 13020, 14700, 16480, 18360]

    def _team_details_stats(self, game_id: str):
        """Do details: xp (via niveis) + gold EFETIVO (itens completos) por lado.

        Retorna dict {'blue':{'xp','eff_gold'}, 'red':{...}} ou None.
        participantId 1-5 = blue, 6-10 = red.
        """
        if self._item_data is None:
            from api.item_data import ItemData
            self._item_data = ItemData()
        det = self._latest_details(game_id)
        frames = (det or {}).get("frames", [])
        if not frames:
            return None
        parts = frames[-1].get("participants", []) or []
        acc = {"blue": {"xp": 0.0, "eff_gold": 0}, "red": {"xp": 0.0, "eff_gold": 0}}
        seen = False
        for i, p in enumerate(parts):
            pid = p.get("participantId", i + 1)
            side = "blue" if 1 <= pid <= 5 else "red" if 6 <= pid <= 10 else None
            if side is None:
                continue
            lvl = max(1, min(18, int(p.get("level", 1) or 1)))
            acc[side]["xp"] += self._XP_AT_LEVEL[lvl]
            acc[side]["eff_gold"] += self._item_data.effective_gold(p.get("items", []) or [])
            seen = True
        return acc if seen else None

    def extract_state(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Estado ATUAL do jogo (window + xp derivado do details).

        Retorna gold/kills/torres/baron/dragoes/xp por lado + game_time, p/ a
        formula de WR ao vivo. None se nao houver frame.
        """
        fr = self._latest_window_frame(game_id)
        if not fr:
            return None
        det = self._team_details_stats(game_id) or {}

        def team(win_side, det_side):
            t = fr.get(win_side, {}) or {}
            drags = t.get("dragons") or []
            ds = det.get(det_side, {})
            return {
                "gold": int(t.get("totalGold", 0) or 0),
                "kills": int(t.get("totalKills", 0) or 0),
                "towers": int(t.get("towers", 0) or 0),
                "barons": int(t.get("barons", 0) or 0),
                "inhibitors": int(t.get("inhibitors", 0) or 0),
                "dragons": list(drags),
                "dragon_count": len(drags),
                "has_soul": len(drags) >= 4,
                "xp": ds.get("xp"),            # None se details indisponivel
                "eff_gold": ds.get("eff_gold"),  # gold em itens completos
            }
        gt = fr.get("gameTime")
        return {
            "blue": team("blueTeam", "blue"),
            "red": team("redTeam", "red"),
            "game_state": fr.get("gameState", ""),
            "game_time_s": gt if isinstance(gt, (int, float)) else None,
            "rfc460": fr.get("rfc460Timestamp", ""),
        }

    def _latest_window_frame(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Sonda o window p/ o frame mais recente disponivel (feed atrasado)."""
        now = datetime.now(timezone.utc)
        delays = [90, 120, 150, 180, 210, 240, 300, 360]
        if self._last_good_delay:
            delays = [self._last_good_delay] + [d for d in delays if d != self._last_good_delay]
        for delay in delays:
            ts = _floor_to_10s(now - timedelta(seconds=delay)).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                r = self.session.get(f"{FEED_BASE}/window/{game_id}",
                                     params={"startingTime": ts}, timeout=self.timeout)
            except Exception:
                continue
            if r.status_code == 200:
                try:
                    frames = r.json().get("frames", [])
                except ValueError:
                    continue
                if frames:
                    self._last_good_delay = delay
                    return frames[-1]
            if r.status_code == 204:
                break
        return None

    def extract_items(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Itens ATUAIS de cada jogador via details(), mapeados por lado+role.

        Retorna {"blue_top_items": [3006, 3153, ...], ..., "red_sup_items": [...]}.
        Correlaciona participantId (metadata do window) com os items (frame do
        details). None se nao houver dado.
        """
        role_map = {"top": "top", "jungle": "jng", "mid": "mid",
                    "bottom": "bot", "support": "sup"}
        # 1) participantId -> (side, role) do metadata do window (raw, SEM
        #    startingTime -> nao 400a; so precisamos do gameMetadata).
        pid_map: Dict[int, tuple] = {}
        order_fallback: Dict[str, list] = {"blue": [], "red": []}
        try:
            wr = self.session.get(f"{FEED_BASE}/window/{game_id}", timeout=self.timeout)
            gm = wr.json().get("gameMetadata", {}) if wr.status_code == 200 else {}
        except Exception:
            return None
        if not gm:
            return None
        for side, key in (("blue", "blueTeamMetadata"), ("red", "redTeamMetadata")):
            for p in gm.get(key, {}).get("participantMetadata", []):
                role = role_map.get(str(p.get("role", "")).lower(), "")
                pid = p.get("participantId")
                if role:
                    order_fallback[side].append(role)
                    if isinstance(pid, int):
                        pid_map[pid] = (side, role)

        # 2) items do frame mais recente disponivel (sonda o atraso do feed)
        det = self._latest_details(game_id)
        frames = (det or {}).get("frames", [])
        if not frames:
            return None
        parts = frames[-1].get("participants", []) or []

        out: Dict[str, Any] = {}
        for idx, part in enumerate(parts):
            pid = part.get("participantId", idx + 1)
            items = part.get("items", []) or []
            sr = pid_map.get(pid)
            if sr is None:
                # fallback posicional: 1-5 = blue, 6-10 = red, na ordem do metadata
                side = "blue" if idx < 5 else "red"
                roles = order_fallback.get(side, [])
                pos = idx if side == "blue" else idx - 5
                role = roles[pos] if pos < len(roles) else ""
                if not role:
                    continue
                sr = (side, role)
            out[f"{sr[0]}_{sr[1]}_items"] = [str(i) for i in items if i]
            lvl = part.get("level")
            if isinstance(lvl, (int, float)) and lvl > 0:
                out[f"{sr[0]}_{sr[1]}_level"] = int(lvl)
        return out or None


# ---------- helpers ----------

def _short_patch(patch_version: str) -> str:
    """Converte '16.9.769.5709' -> '16.9'."""
    parts = (patch_version or "").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else patch_version


_LF_TO_OE_CACHE: Dict[str, str] = {}
_LF_CHAMPION_ALIASES = {
    # Ids internos do Data Dragon que NAO batem com o nome do Oracle nem
    # normalizando (conferido nos 173 campeoes em 13/09/2026 - so esses 3).
    # Sem alias o campeao vira "sem dados" no card E zera as features de
    # campeao do modelo ao vivo.
    "monkeyking": "Wukong",
    "renata": "Renata Glasc",
    "nunu": "Nunu & Willump",
}


def _lf_champion_to_oe(lf_name: str) -> str:
    """Mapeia 'KSante' -> 'K'Sante' usando o snapshot OE.

    A primeira chamada carrega o snapshot e constroi um indice.
    """
    if not lf_name:
        return ""
    if lf_name in _LF_TO_OE_CACHE:
        return _LF_TO_OE_CACHE[lf_name]

    # Constroi indice na 1a chamada
    if not _LF_TO_OE_CACHE:
        try:
            import pickle
            import re
            from pathlib import Path
            snap_path = Path(__file__).parent.parent / "data" / "temporal_state.pkl"
            if snap_path.exists():
                with open(snap_path, "rb") as f:
                    snap = pickle.load(f)
                champs = list(snap.get("champion_pickrate", {}).keys())
                for oe in champs:
                    norm = re.sub(r"[^a-zA-Z]", "", oe).lower()
                    _LF_TO_OE_CACHE[norm] = oe
        except Exception:
            pass

    import re
    norm = re.sub(r"[^a-zA-Z]", "", lf_name).lower()
    mapped = _LF_CHAMPION_ALIASES.get(norm) or _LF_TO_OE_CACHE.get(norm, lf_name)
    _LF_TO_OE_CACHE[lf_name] = mapped
    return mapped
