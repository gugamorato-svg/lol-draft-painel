"""Resolucao robusta de nomes de jogadores vindos do live feed."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ALIASES_PATH = Path(__file__).parent.parent / "data" / "player_aliases.json"
_SPECIAL_TRANSLATION = str.maketrans({
    "Ø": "O", "ø": "o",
    "Æ": "AE", "æ": "ae",
    "Œ": "OE", "œ": "oe",
    "Ð": "D", "ð": "d",
    "Þ": "Th", "þ": "th",
    "Ł": "L", "ł": "l",
    "ß": "ss",
})

ROLE_TO_OE = {
    "Top": "top",
    "Jungle": "jng",
    "Mid": "mid",
    "Bot": "bot",
    "Support": "sup",
    "TOP": "top",
    "JNG": "jng",
    "MID": "mid",
    "BOT": "bot",
    "SUP": "sup",
}


def norm_name(value: Any) -> str:
    text = str(value or "").translate(_SPECIAL_TRANSLATION)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-zA-Z0-9]", "", text).casefold()


def clean_live_name(value: Any) -> str:
    """Remove prefixo separado por espaco, mantendo o nick provavel."""
    text = str(value or "").strip()
    if text.casefold() in {"", "-", "nan", "none", "null"}:
        return ""
    parts = text.split()
    if len(parts) >= 2:
        return parts[-1].strip()
    return text


@dataclass(frozen=True)
class PlayerResolution:
    raw: str
    canonical: str
    display: str
    reason: str
    candidates: List[str]
    player_id: str = ""


class PlayerResolver:
    """Resolve live names into Oracle player names without replacing live players.

    The live feed is the source of truth for who is playing. Roster names are
    only display fallback and are not candidates for stats lookup, otherwise a
    stale roster can replace a substitute/academy player with the regular one.
    """

    def __init__(self, oracle: Any, alias_path: Path = ALIASES_PATH) -> None:
        self.oracle = oracle
        self.aliases = self._load_aliases(alias_path)

    def resolve(
        self,
        raw_name: str,
        *,
        team: str = "",
        role: str = "",
        year: Optional[int] = None,
        fallback_name: str = "",
    ) -> PlayerResolution:
        candidates = self._candidate_names(raw_name, team=team)
        alias = self._resolve_alias(candidates)
        if alias:
            candidates.insert(0, alias)

        exact = self._find_exact(candidates, team=team, role=role, year=year)
        if exact:
            return PlayerResolution(raw_name, exact, exact, "exact/context", candidates,
                                    self._context_id(exact, team=team, role=role, year=year))

        contextual = self._find_contextual(candidates, team=team, role=role, year=year)
        if contextual:
            return PlayerResolution(raw_name, contextual, contextual, "fuzzy/context", candidates,
                                    self._context_id(contextual, team=team, role=role, year=year))

        # Um nome em outro time nao comprova transferencia: pode ser homonimo.
        # Transferencias ja registradas resolvem no contexto acima. Sem esse
        # registro, preserva o nome do feed sem atribuir stats de outro jogador.
        # A busca global so faz sentido quando nenhum time foi informado.
        if not team:
            loose = self._find_loose(candidates)
            if loose:
                return PlayerResolution(raw_name, loose, loose, "loose", candidates)

        display = clean_live_name(raw_name) or clean_live_name(fallback_name) or "-"
        return PlayerResolution(raw_name, "", display, "unresolved", candidates)

    def _context_id(self, name: str, *, team: str, role: str, year: Optional[int]) -> str:
        """Preserva a identidade Oracle confirmada no time/papel/ano consultado."""
        df = self._context_df(team=team, role=role, year=year)
        if "playerid" not in df:
            return ""
        rows = df[df["__playername_norm"] == norm_name(name)]
        ids = {str(v) for v in rows["playerid"].dropna()
               if str(v).lower() not in ("", "nan", "none")}
        return next(iter(ids)) if len(ids) == 1 else ""

    def _load_aliases(self, alias_path: Path) -> Dict[str, str]:
        if not alias_path.exists():
            return {}
        try:
            raw = json.loads(alias_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        aliases: Dict[str, str] = {}
        for key, value in raw.items():
            aliases[norm_name(key)] = str(value)
        return aliases

    def _resolve_alias(self, candidates: Iterable[str]) -> str:
        for candidate in candidates:
            alias = self.aliases.get(norm_name(candidate))
            if alias:
                return alias
        return ""

    def _candidate_names(self, raw_name: str, *, team: str) -> List[str]:
        out: List[str] = []

        def add(value: Any) -> None:
            text = str(value or "").strip()
            if text and text not in out:
                out.append(text)

        add(raw_name)
        add(clean_live_name(raw_name))

        raw_key = norm_name(raw_name)
        for prefix in self._team_keys(team):
            if raw_key.startswith(prefix) and len(raw_key) > len(prefix) + 1:
                add(raw_key[len(prefix):])
        return out

    def _team_keys(self, team: str) -> List[str]:
        words = re.findall(r"[A-Za-z0-9]+", str(team or ""))
        keys = [norm_name(team)]
        if words:
            keys.append(norm_name("".join(word[0] for word in words)))
            keys.extend(norm_name(word) for word in words if len(word) >= 2)
        aliases = {
            # LCK / LEC
            "dpluskia": ["dk", "dplus"],
            "bnkfearx": ["bfx", "fox", "fearx"],
            "g2esports": ["g2"],
            "karminecorp": ["kc"],
            "t1": ["t1"],
            "gendotg": ["geng", "gen"],
            "hanwhalifeesports": ["hle"],
            # LPL - prefixos comuns nos summonerNames
            # (ex: "BLGBin" no time Bilibili Gaming -> precisa do alias blg)
            "bilibiligaming": ["blg"],
            "topesports": ["tes"],
            "weibogaming": ["wbg"],
            "edwardgaming": ["edg"],
            "funplusphoenix": ["fpx"],
            "thundertalkgaming": ["tt", "ttg"],
            "rareatom": ["ra"],
            "invictusgaming": ["ig"],
            "anyoneslegend": ["al"],
            "lgdgaming": ["lgd"],
            "lngesports": ["lng"],
            "beijingjdgesports": ["jdg"],
            "shenzhenninjasinpyjamas": ["nip"],
            "ninjasinpyjamas": ["nip"],
            "xianteamwe": ["we"],
            "teamwe": ["we"],
            "ultraprime": ["up"],
            "rougewarriors": ["rw"],
            # LCKC / academy comuns
            "dpluskiachallengers": ["dkc", "dk"],
            "t1academy": ["t1"],
            "gendotgchallengers": ["gengc", "gen"],
        }
        for key in list(keys):
            keys.extend(aliases.get(key, []))
        return sorted({key for key in keys if key}, key=len, reverse=True)

    def _context_df(self, *, team: str, role: str, year: Optional[int]):
        df = self.oracle.players
        if year:
            df = df[df["__year"] == year]
        role_oe = ROLE_TO_OE.get(str(role).upper(), ROLE_TO_OE.get(str(role).title(), ""))
        if role_oe:
            df = df[df["position"].fillna("").str.lower() == role_oe]
        team_canonical = ""
        try:
            team_canonical = self.oracle._resolve_team(team)
        except Exception:
            team_canonical = ""
        if team_canonical:
            by_team = df[df["teamname"] == team_canonical]
            return by_team if not by_team.empty else df.iloc[0:0]
        if team:
            return df.iloc[0:0]
        return df

    def _find_exact(
        self, candidates: Iterable[str], *, team: str, role: str, year: Optional[int]
    ) -> str:
        df = self._context_df(team=team, role=role, year=year)
        for candidate in candidates:
            key = norm_name(candidate)
            if not key:
                continue
            rows = df[df["__playername_norm"] == key]
            if not rows.empty:
                names = rows["__playername_norm"].unique()
                if len(names) == 1:
                    return str(rows["playername"].value_counts().index[0])
        return ""

    def _find_contextual(
        self, candidates: Iterable[str], *, team: str, role: str, year: Optional[int]
    ) -> str:
        df = self._context_df(team=team, role=role, year=year)
        if df.empty:
            return ""
        for candidate in candidates:
            key = norm_name(candidate)
            if len(key) < 3:
                continue
            rows = df[
                df["__playername_norm"].str.contains(key, regex=False, na=False)
                | df["__playername_norm"].map(
                    lambda value: bool(value) and (value in key or key.endswith(value))
                )
            ]
            if not rows.empty:
                if rows["__playername_norm"].nunique() == 1:
                    return str(rows["playername"].value_counts().index[0])
        return ""

    def _find_loose(self, candidates: Iterable[str]) -> str:
        for candidate in candidates:
            try:
                resolved = self.oracle._resolve_player(candidate)
            except Exception:
                resolved = ""
            if resolved:
                return resolved
        return ""
