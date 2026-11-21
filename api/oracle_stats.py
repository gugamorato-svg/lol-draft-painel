"""Stats agregadas vindas dos CSVs do Oracle's Elixir locais.

Substitui o LeaguepediaClient para queries de stats (roster, kills/jogo,
H2H, champion comfort etc). Vantagens:
  - sem rate limit (Leaguepedia rate-limita feio)
  - 2026 completo (Leaguepedia ainda nao indexa rosters/jogos 2026)
  - 10x mais rapido (tudo em memoria, ~200 MB de pandas DataFrame)

Interface compativel com api.leaguepedia.LeaguepediaClient.

Arquivos esperados em data/oracle_elixir/:
  2024_lol.csv  2025_lol.csv  2026_lol.csv
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from data_pipeline.patches import patch_label, patch_tuple

_DATA = Path(__file__).parent.parent / "data" / "oracle_elixir"

# OE -> Leaguepedia role labels
_ROLE_MAP_OUT = {
    "top": "Top", "jng": "Jungle", "mid": "Mid",
    "bot": "Bot", "sup": "Support",
}
_LP_ROLES = ["Top", "Jungle", "Mid", "Bot", "Support"]
_SPECIAL_TRANSLATION = str.maketrans({
    "Ø": "O", "ø": "o",
    "Æ": "AE", "æ": "ae",
    "Œ": "OE", "œ": "oe",
    "Ð": "D", "ð": "d",
    "Þ": "Th", "þ": "th",
    "Ł": "L", "ł": "l",
    "ß": "ss",
})


def _norm(name: str) -> str:
    """Normaliza nome para match: LØS -> los, paiN -> pain, Gen.G -> geng."""
    if not name:
        return ""
    text = str(name).translate(_SPECIAL_TRANSLATION)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


_CHAMPION_ALIASES = {
    # Riot/Data Dragon id interno; Oracle's Elixir salva o nome exibido.
    # Mesma lista de api/livefeed._LF_CHAMPION_ALIASES.
    "monkeyking": "wukong",
    "renata": "renataglasc",
    "nunu": "nunuwillump",
}


def _champion_key(champion: str) -> str:
    """Chave canonica de campeao para bater Data Dragon/livefeed com Oracle."""
    key = _norm(champion)
    return _CHAMPION_ALIASES.get(key, key)


class OracleStatsClient:
    """Cliente local que le CSVs do Oracle's Elixir.

    Carrega uma vez na inicializacao; queries depois sao O(1) (filtros em DF).
    """

    # ---------- carga ----------

    def __init__(self, years: Optional[Sequence[int]] = None) -> None:
        parts = []
        if years is None:
            discovered = []
            for p in sorted(_DATA.glob("*_lol.csv")):
                try:
                    discovered.append(int(p.name.split("_", 1)[0]))
                except ValueError:
                    continue
            years = discovered or (2024, 2025, 2026)
        self.loaded_years = tuple(int(y) for y in years)
        from data_pipeline.oracle_loader import MODEL_USE_EXTRA, extra_files
        for y in self.loaded_years:
            p = _DATA / f"{y}_lol.csv"
            if p.exists():
                df = pd.read_csv(p, low_memory=False)
                if MODEL_USE_EXTRA:   # 24/09: bases chinesas da Leaguepedia (fora do Oracle)
                    extras = [pd.read_csv(e, low_memory=False) for e in extra_files(y)]
                    if extras:
                        df = pd.concat([df, *extras], ignore_index=True)
                year_col = pd.Series(y, index=df.index, name="__year")
                df = pd.concat([df, year_col], axis=1).copy()
                parts.append(df)
        if not parts:
            raise FileNotFoundError(f"Nenhum CSV em {_DATA}")
        self._career_fallback_cache: Dict[tuple[str, str], Dict[str, float]] = {}
        self._patch_idx = None  # lazy: _patch_index()
        raw = pd.concat(parts, ignore_index=True)
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")

        # Linhas de jogador (participantid 1-10) -- onde estao stats individuais
        players = raw[raw["participantid"].between(1, 10)].copy()
        # Linhas de time (participantid 100/200) -- onde estao picks/bans e result
        teams = raw[raw["participantid"].isin([100, 200])].copy()

        oe_role_to_lp = {"top": "Top", "jng": "Jungle", "jungle": "Jungle", "mid": "Mid",
                         "bot": "Bot", "sup": "Support", "support": "Support"}

        # Adiciona TODAS as colunas auxiliares de uma vez (pd.concat evita fragmentation warning)
        players_extras = pd.DataFrame({
            "__teamname_norm":   players["teamname"].fillna("").map(_norm).values,
            "__playername_norm": players["playername"].fillna("").map(_norm).values,
            "__champion_norm":   players["champion"].fillna("").map(_champion_key).values,
            "__role_lp": (
                players["position"].fillna("").str.lower().map(oe_role_to_lp).fillna("").values
            ),
        }, index=players.index)
        teams_extras = pd.DataFrame({
            "__teamname_norm": teams["teamname"].fillna("").map(_norm).values,
        }, index=teams.index)

        self.players = pd.concat([players, players_extras], axis=1)
        self.teams = pd.concat([teams, teams_extras], axis=1)

    # ---------- name resolution (fuzzy match) ----------

    # Aliases: nomes que a Riot/Tencent live feed usam -> nome OE canonico
    # Necessario pra times onde OE usa nome curto/diferente.
    _TEAM_NAME_ALIASES = {
        # LPL: Riot usa "Beijing JDG Esports" mas OE armazena como "JD Gaming"
        "beijingjdgesports": "JD Gaming",
        "jdgintelesportsclub": "JD Gaming",
        # Variacoes comuns onde a Riot usa o nome longo
        "shenzhenninjasinpyjamas": "Ninjas in Pyjamas",
        "xianteamwe": "Team WE",
        # LCK
        "kthnesports": "KT Rolster",  # fallback se nome alterado
    }

    def _resolve_team(self, name: str) -> str:
        """Retorna o teamname OE canonico para um nome de input.

        Mesmas regras do modelo (models.trained_predictor._resolve_team_id): apelido, exato,
        nucleo do nome (so se unico) e trecho com trava de categoria. 24/09: o app nao achava
        "kt Challengers" (Elo 1500, stats n=0 -> 26% num jogo em que o robo deu 56%) e o
        trecho devolvia o time MAIS FREQUENTE (base podia virar o time principal).
        """
        from models.trained_predictor import (
            _TEAM_ALIASES, _build_team_core_index, _norm_name, _same_tier, _team_core,
        )
        if not name:
            return ""
        key = _norm(name)
        if not key:
            return ""
        # 0. Aliases manuais (live feed names -> OE canonical), os daqui e os do modelo
        alias = self._TEAM_NAME_ALIASES.get(key) or _TEAM_ALIASES.get(_norm_name(name))
        if alias and (self.teams["teamname"] == alias).any():
            return alias
        if not hasattr(self, "_team_norm_map"):
            # nome normalizado -> teamname (o mais frequente quando 2 grafias colidem)
            freq = self.teams["teamname"].value_counts()
            self._team_norm_map: Dict[str, str] = {}
            for t in freq.index:
                self._team_norm_map.setdefault(_norm(t), t)
            self._team_core_map = _build_team_core_index({t: t for t in freq.index})
        # 1. Match exato normalizado
        if key in self._team_norm_map:
            return self._team_norm_map[key]
        # 2. Nucleo do nome sem palavra generica ("FURIA Esports" -> "furia"), so se unico
        core = _team_core(name)
        if core in self._team_core_map:
            return self._team_core_map[core]
        # 3. Trecho (qualquer direcao, min 3): o MAIS PARECIDO, mesma categoria
        best, best_sim = "", 0.0
        for cand, team in self._team_norm_map.items():
            if len(cand) >= 3 and (cand in key or key in cand) and _same_tier(key, cand):
                sim = min(len(cand), len(key)) / max(len(cand), len(key))
                if sim > best_sim:
                    best, best_sim = team, sim
        return best if best_sim >= 0.5 else ""

    def _resolve_player(self, name: str) -> str:
        """Retorna o playername OE canonico."""
        if not name:
            return ""
        key = _norm(name)
        if not key:
            return ""
        matches = self.players[self.players["__playername_norm"] == key]
        if len(matches):
            return matches["playername"].iloc[0]
        # Strip prefix de team (e.g. "WECube" -> "Cube"): tenta os ultimos N chars
        for prefix_len in (2, 3, 4):
            if len(key) > prefix_len + 2:
                stripped = key[prefix_len:]
                matches = self.players[self.players["__playername_norm"] == stripped]
                if len(matches):
                    return matches["playername"].iloc[0]
        # Substring
        if len(key) >= 3:
            sub = self.players[
                self.players["__playername_norm"].str.contains(key, regex=False, na=False)
            ]
            if len(sub):
                return sub["playername"].value_counts().index[0]
        return ""

    # ---------- queries (interface igual a LeaguepediaClient) ----------

    def get_team_roster(self, team: str, year: int) -> List[Dict[str, str]]:
        """Retorna os jogadores titulares mais recentes do time no ano.

        Output: [{'role': 'Top', 'player': 'Kiin', 'team': 'Gen.G'}, ...]
        """
        canonical = self._resolve_team(team)
        if not canonical:
            return []
        df = self.players[
            (self.players["teamname"] == canonical) & (self.players["__year"] == year)
        ]
        if df.empty:
            # Fallback: ano mais recente disponivel
            df = self.players[self.players["teamname"] == canonical]
            if df.empty:
                return []
        # Pega o player mais recente por role
        df = df.sort_values("date")
        out: List[Dict[str, str]] = []
        for role in _LP_ROLES:
            role_rows = df[df["__role_lp"] == role]
            if len(role_rows):
                last = role_rows.iloc[-1]
                out.append({
                    "role":   role,
                    "player": str(last["playername"]),
                    "team":   canonical,
                })
        return out

    def get_team_year_stats(self, team: str, year: int) -> Dict[str, float]:
        """Stats agregadas: games, total_kills_avg, winrate, etc.

        Devolve dict com a mesma estrutura do LeaguepediaClient pra ser
        drop-in replacement na UI.
        """
        canonical = self._resolve_team(team)
        empty = {"games": 0, "winrate": 0.0, "total_kills_avg": 0.0,
                 "duration_avg_min": 0.0, "over_kills_pct": 0.0,
                 "under_kills_pct": 0.0, "over_time_pct": 0.0,
                 "under_time_pct": 0.0}
        if not canonical:
            return empty

        # Pega rows do time-summary (100/200) no ano
        df = self.teams[
            (self.teams["teamname"] == canonical) & (self.teams["__year"] == year)
        ]
        if df.empty:
            return empty

        # Para cada jogo desse time, precisa do total de kills do GAME (blue+red)
        # entao agrupamos por gameid: pega ambos os rows (100 e 200) do mesmo game.
        game_ids = df["gameid"].unique()
        all_team_rows = self.teams[self.teams["gameid"].isin(game_ids)]

        # game-level: 1 row blue (100) + 1 row red (200) por game
        # kills do time = sum dos teamkills nesse game; len em segundos = gamelength
        # winrate calculado em cima do "result" do row do time
        wins = int((df["result"] == 1).sum())
        games_n = len(df)

        # totals por game (somando blue + red)
        per_game = all_team_rows.groupby("gameid").agg(
            total_kills=("teamkills", "sum"),
            length=("gamelength", "first"),
        )
        totals = per_game["total_kills"].astype(float).tolist()
        durations = (per_game["length"].astype(float) / 60.0).tolist()  # min

        # mediana
        def _median(xs):
            if not xs: return 0.0
            s = sorted(xs)
            mid = len(s) // 2
            return s[mid] if len(s) % 2 else (s[mid-1] + s[mid]) / 2

        median_k = _median(totals)
        median_t = _median(durations)
        over_k = sum(1 for x in totals if x > median_k)
        under_k = sum(1 for x in totals if x < median_k)
        over_t = sum(1 for x in durations if x > median_t)
        under_t = sum(1 for x in durations if x < median_t)

        return {
            "games":            games_n,
            "winrate":          round(100 * wins / max(games_n, 1), 1),
            "total_kills_avg":  round(sum(totals) / max(len(totals), 1), 2),
            "duration_avg_min": round(sum(durations) / max(len(durations), 1), 2),
            "over_kills_pct":   round(100 * over_k / max(len(totals), 1), 1),
            "under_kills_pct":  round(100 * under_k / max(len(totals), 1), 1),
            "over_time_pct":    round(100 * over_t / max(len(durations), 1), 1),
            "under_time_pct":   round(100 * under_t / max(len(durations), 1), 1),
        }

    def get_player_year_stats(self, player: str, year: int) -> Dict[str, float]:
        """Stats agregadas do jogador no ano."""
        canonical = self._resolve_player(player)
        empty = {"games": 0, "kills_avg": 0.0, "deaths_avg": 0.0,
                 "assists_avg": 0.0, "kda": 0.0}
        if not canonical:
            return empty
        df = self.players[
            (self.players["playername"] == canonical) & (self.players["__year"] == year)
        ]
        if df.empty:
            return empty
        n = len(df)
        k = df["kills"].astype(float).mean()
        d = df["deaths"].astype(float).mean()
        a = df["assists"].astype(float).mean()
        return {
            "games":       int(n),
            "kills_avg":   round(float(k), 2),
            "deaths_avg":  round(float(d), 2),
            "assists_avg": round(float(a), 2),
            "kda":         round((k + a) / max(d, 1), 2),
        }

    def get_player_champion_stats(
        self, player: str, champion: str, year: int
    ) -> Dict[str, float]:
        """Stats do jogador especificamente nesse champion, somente no ano informado."""
        canonical = self._resolve_player(player)
        empty = {"games": 0, "wins": 0, "kills_avg": 0.0, "deaths_avg": 0.0,
                 "assists_avg": 0.0, "winrate": 0.0}
        champion_key = _champion_key(champion)
        if not canonical or not champion_key:
            return empty
        df = self.players[
            (self.players["playername"] == canonical)
            & (self.players["__year"] == year)
            & (self.players["__champion_norm"] == champion_key)
        ]
        return self._player_champion_stats_from_df(df, empty)

    def get_player_champion_career_stats(
        self, player: str, champion: str
    ) -> Dict[str, float]:
        """Stats do jogador especificamente nesse champion, carreira inteira na base local."""
        canonical = self._resolve_player(player)
        empty = {"games": 0, "wins": 0, "kills_avg": 0.0, "deaths_avg": 0.0,
                 "assists_avg": 0.0, "winrate": 0.0}
        champion_key = _champion_key(champion)
        if not canonical or not champion_key:
            return empty
        df = self.players[
            (self.players["playername"] == canonical)
            & (self.players["__champion_norm"] == champion_key)
        ]
        stats = self._player_champion_stats_from_df(df, empty)
        if stats.get("games", 0) > 0:
            return stats
        return self._leaguepedia_career_fallback(canonical, champion, empty)

    def _leaguepedia_career_fallback(
        self, player: str, champion: str, empty: Dict[str, float]
    ) -> Dict[str, float]:
        """Fallback online quando o historico local nao cobre anos antigos."""
        key = (_norm(player), _champion_key(champion))
        if key in self._career_fallback_cache:
            return self._career_fallback_cache[key]
        try:
            from api.cache import JsonCache
            from api.leaguepedia import LeaguepediaClient

            cache = JsonCache(Path.home() / ".lol_painel" / "cache.sqlite")
            lp = LeaguepediaClient(cache=cache, timeout=6.0, backoffs=[0])
            stats = lp.get_player_champion_career_stats(player, champion)
            self._career_fallback_cache[key] = stats
            return stats
        except Exception:
            self._career_fallback_cache[key] = empty
            return empty

    def _player_champion_stats_from_df(self, df, empty: Dict[str, float]) -> Dict[str, float]:
        if df.empty:
            return empty
        n = len(df)
        wins = int((df["result"] == 1).sum())
        k = df["kills"].astype(float).mean()
        d = df["deaths"].astype(float).mean()
        a = df["assists"].astype(float).mean()
        return {
            "games":       int(n),
            "wins":        wins,
            "kills_avg":   round(float(k), 2),
            "deaths_avg":  round(float(d), 2),
            "assists_avg": round(float(a), 2),
            "winrate":     round(100 * wins / max(n, 1), 1),
        }

    def _patch_index(self):
        """{((major, minor), champ): (games, wins)} + {(major, minor): n_jogos}. Lazy."""
        if self._patch_idx is None:
            p = self.players
            num = pd.to_numeric(p["patch"], errors="coerce")
            df = pd.DataFrame({
                "major": num.floordiv(1),
                "minor": ((num - num.floordiv(1)) * 100).round(),
                "champ": p["__champion_norm"],
                "win": (p["result"] == 1).astype(int),
                "gameid": p["gameid"],
            }).dropna(subset=["major", "minor"])
            df = df[df["champ"] != ""]
            champs = df.groupby(["major", "minor", "champ"])["win"].agg(["size", "sum"])
            by_champ = {((int(ma), int(mi)), c): (int(n), int(w))
                        for (ma, mi, c), (n, w) in champs.iterrows()}
            games = df.groupby(["major", "minor"])["gameid"].nunique()
            by_patch = {(int(ma), int(mi)): int(n) for (ma, mi), n in games.items()}
            self._patch_idx = (by_champ, by_patch)
        return self._patch_idx

    # Patch com menos jogos que isso na base = recem-lancado (ex.: ligas migrando
    # 16.17 -> 16.18): o WR do campeao ali e ruido, entao exibe o patch anterior.
    MIN_PATCH_GAMES = 30

    def get_champion_patch_stats(self, champion: str, patch: Any) -> Dict[str, Any]:
        """WR do campeao no patch - todos os jogos pro da base, qualquer jogador.

        Se o patch pedido nao tem jogos suficientes na base (recem-lancado ou
        dado atrasado), usa o patch anterior mais recente e marca fallback=True.
        `requested_stats` guarda o que houver do patch pedido (p/ tooltip) e
        `prev` o patch anterior ao exibido.
        """
        out = {"games": 0, "wins": 0, "winrate": 0.0, "patch": "",
               "requested": patch_label(patch), "fallback": False, "patch_games": 0,
               "requested_stats": None, "prev": None}
        champion_key = _champion_key(champion)
        by_champ, by_patch = self._patch_index()
        if not champion_key or not by_patch:
            return out

        def stats(pt):
            games, wins = by_champ.get((pt, champion_key), (0, 0))
            return {"patch": f"{pt[0]}.{pt[1]:02d}", "games": games, "wins": wins,
                    "winrate": round(100 * wins / games, 1) if games else 0.0,
                    "patch_games": by_patch[pt]}

        wanted = patch_tuple(patch)
        mature = sorted(pt for pt, n in by_patch.items() if n >= self.MIN_PATCH_GAMES) \
            or sorted(by_patch)
        if wanted in mature:
            used = wanted
        else:
            older = [pt for pt in mature if wanted is None or pt < wanted]
            used = older[-1] if older else mature[-1]
        out.update(stats(used))
        out["fallback"] = used != wanted
        if out["fallback"] and wanted in by_patch:
            out["requested_stats"] = stats(wanted)
        earlier = [pt for pt in mature if pt < used]
        if earlier:
            out["prev"] = stats(earlier[-1])
        return out

    def get_h2h(self, team_a: str, team_b: str, year: int) -> Dict[str, float]:
        """Head-to-head: WR de A contra B + diff de kills/deaths."""
        empty = {"games": 0, "winrate_a": 50.0, "kills_diff": 0.0, "deaths_diff": 0.0}
        ca = self._resolve_team(team_a)
        cb = self._resolve_team(team_b)
        if not ca or not cb:
            return empty
        # team rows: 1 jogo = 2 rows (100 e 200)
        df = self.teams[self.teams["__year"] == year]
        # gameids onde os DOIS times jogaram
        gameids_a = set(df[df["teamname"] == ca]["gameid"])
        gameids_b = set(df[df["teamname"] == cb]["gameid"])
        common = gameids_a & gameids_b
        if not common:
            # fallback sem ano
            df = self.teams
            gameids_a = set(df[df["teamname"] == ca]["gameid"])
            gameids_b = set(df[df["teamname"] == cb]["gameid"])
            common = gameids_a & gameids_b
            if not common:
                return empty
        df_common = df[df["gameid"].isin(common)]
        rows_a = df_common[df_common["teamname"] == ca]
        rows_b = df_common[df_common["teamname"] == cb]
        wins_a = int((rows_a["result"] == 1).sum())
        kd_a = float(rows_a["teamkills"].astype(float).mean())
        kd_b = float(rows_b["teamkills"].astype(float).mean())
        dd_a = float(rows_a["teamdeaths"].astype(float).mean())
        dd_b = float(rows_b["teamdeaths"].astype(float).mean())
        return {
            "games":      len(common),
            "winrate_a":  round(100 * wins_a / len(common), 1),
            "kills_diff": round(kd_a - kd_b, 2),
            "deaths_diff": round(dd_a - dd_b, 2),
        }


# ---------- singleton (carrega CSVs uma vez por processo) ----------

_INSTANCE: Optional[OracleStatsClient] = None


def get_instance() -> OracleStatsClient:
    """Singleton lazy. Carrega os CSVs na primeira chamada."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = OracleStatsClient()
    return _INSTANCE


if __name__ == "__main__":
    import time
    t0 = time.time()
    c = OracleStatsClient()
    print(f"Carga: {time.time()-t0:.2f}s")
    print(f"players rows: {len(c.players)}, teams rows: {len(c.teams)}")
    print()
    for team in ("Gen.G", "T1", "Xi'an Team WE", "Shenzhen NINJAS IN PYJAMAS"):
        for yr in (2025, 2026):
            ts = c.get_team_year_stats(team, yr)
            print(f"{team:<35} {yr}: games={ts['games']}, kills={ts['total_kills_avg']}, wr={ts['winrate']}%")
        print()
    print("Roster Gen.G 2026:")
    for p in c.get_team_roster("Gen.G", 2026):
        print(f"  {p}")
    print()
    print("H2H Gen.G vs T1 2025:", c.get_h2h("Gen.G", "T1", 2025))
    print()
    print("Faker em Azir 2025:", c.get_player_champion_stats("Faker", "Azir", 2025))













