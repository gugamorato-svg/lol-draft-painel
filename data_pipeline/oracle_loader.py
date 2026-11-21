"""Carrega CSVs do Oracle's Elixir e transforma em 1 row/jogo.

Cada CSV bruto tem 12 rows por jogo:
  participantid 1-5   = players do blue
  participantid 6-10  = players do red
  participantid 100   = sumario do blue (picks em ordem de draft)
  participantid 200   = sumario do red

Saida: DataFrame com 1 row por jogo, com colunas como
  blue_picks_draft_order = ["Maokai","Jinx","Leona","Hwei","Gnar"]
  blue_player_champs     = {"top":"Gnar","jng":"Maokai", ...}
  blue_team_id, blue_team_name, blue_won, etc.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


_DATA_DIR = Path(__file__).parent.parent / "data" / "oracle_elixir"

# Ligas tier 1 / tier 2 — filtra ruido das ligas amadoras
TIER1_LEAGUES = {
    "LPL", "LCK", "LEC", "LCS", "LTA N", "LTA S",
    "WLDs", "MSI", "FRC", "LCP",
}
# UNIVERSO DO MODELO (22/09/2026): TODAS as ligas do Oracle.
# Antes era TIER1|TIER2, o que deixava CBLOL, EWC, First Stand, KeSPA Cup, LES, LIT,
# ROL, HLL, EBL e outras de fora — times como Fuego/Only The Family/Team Aqua nem
# existiam para o modelo. A cobertura cresceu ~43% em mapas e a
# perda nos jogos que ja cobriamos e ruido (log loss +0.0021, IC95 [-0.0012,+0.0055]).
# Passar all_leagues=False em load_year/load_years volta ao universo antigo.
MODEL_ALL_LEAGUES = True

# JOGOS "PARCIAIS" (23/09/2026). O Oracle marca como partial todo jogo do LPL (2024 e
# 2025), o LDL 2024 e o MSI 2024 inteiros: faltam so estatisticas de 15 min (ouro, placas,
# torre do meio). Draft, jogadores e resultado estao la. Antes eles eram DESCARTADOS em
# 2024-25 -> o modelo nao tinha nenhum jogo do LPL antes de 2026 (Elo, conforto e forca
# da liga do LPL partiam do zero). O TemporalContext ignora as colunas vazias (NaN).
# Passar accept_partial=False volta ao comportamento antigo (so "complete" antes de 2026).
MODEL_ACCEPT_PARTIAL = True

# JOGOS DE FORA DO ORACLE (24/09/2026): data/oracle_elixir/extra/<ano>_*.csv, no mesmo formato
# longo do Oracle. Hoje: bases chinesas da Leaguepedia (LDL 2025 + circuito 2026), que o
# Oracle nao tem desde jun/2024 (scripts/import_leaguepedia_cn.py). use_extra=False = so Oracle.
MODEL_USE_EXTRA = True

TIER2_LEAGUES = {
    "LCKC", "LFL", "PCS", "PRM", "AL", "LJL",
    "EM", "NACL", "LVP SL", "NLC", "TCL", "RL",
    "LAS", "TSC", "CD",
}


def list_available_years() -> List[int]:
    """Lista anos disponiveis em data/oracle_elixir/."""
    years = []
    for p in _DATA_DIR.glob("*_lol.csv"):
        try:
            years.append(int(p.stem.split("_")[0]))
        except ValueError:
            continue
    return sorted(years)


def extra_files(year: int) -> List[Path]:
    """CSVs de fora do Oracle daquele ano (data/oracle_elixir/extra/<ano>_*.csv)."""
    return sorted((_DATA_DIR / "extra").glob(f"{year}_*.csv"))


def load_year(
    year: int,
    only_tier1: bool = False,
    only_complete: bool = True,
    all_leagues: Optional[bool] = None,
    accept_partial: Optional[bool] = None,
    use_extra: Optional[bool] = None,
) -> pd.DataFrame:
    """Carrega 1 ano em DataFrame ja agregado (1 row/jogo).

    all_leagues: None = usa MODEL_ALL_LEAGUES (padrao do projeto). True/False forca.
    accept_partial: None = usa MODEL_ACCEPT_PARTIAL. Com False, antes de 2026 so entram
    jogos "complete" (comportamento antigo, que tirava o LPL 2024-25 inteiro).
    """
    csv_path = _DATA_DIR / f"{year}_lol.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} nao existe — rode download primeiro")

    raw = pd.read_csv(csv_path, low_memory=False)
    if use_extra is None:
        use_extra = MODEL_USE_EXTRA
    if use_extra:
        extras = [pd.read_csv(p, low_memory=False) for p in extra_files(year)]
        if extras:
            raw = pd.concat([raw, *extras], ignore_index=True)
    if accept_partial is None:
        accept_partial = MODEL_ACCEPT_PARTIAL
    if only_complete and not accept_partial and year < 2026:
        raw = raw[raw["datacompleteness"] == "complete"]

    if all_leagues is None:
        all_leagues = MODEL_ALL_LEAGUES
    if not all_leagues:  # all_leagues=True: sem filtro (CBLOL, EWC, LES, LIT... entram)
        allowed = TIER1_LEAGUES if only_tier1 else TIER1_LEAGUES | TIER2_LEAGUES
        raw = raw[raw["league"].isin(allowed)]

    return _aggregate_to_games(raw)


def load_years(
    years: Iterable[int],
    only_tier1: bool = False,
    only_complete: bool = True,
    all_leagues: Optional[bool] = None,
    accept_partial: Optional[bool] = None,
    use_extra: Optional[bool] = None,
) -> pd.DataFrame:
    """Concatena varios anos em ordem cronologica."""
    parts = [
        load_year(y, only_tier1=only_tier1, only_complete=only_complete,
                  all_leagues=all_leagues, accept_partial=accept_partial, use_extra=use_extra)
        for y in years
    ]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------- internals ----------

_PLAYER_ROLES = ["top", "jng", "mid", "bot", "sup"]
_TEAM_COLS = [
    "gameid", "date", "league", "year", "split", "patch", "game",
    "side", "teamname", "teamid", "result", "gamelength",
    "ban1", "ban2", "ban3", "ban4", "ban5",
    "pick1", "pick2", "pick3", "pick4", "pick5",
    "firstPick",
    # Team-level early/objective columns used by TemporalContext.
    "firstblood", "firstdragon", "dragons", "opp_dragons",
    "firstherald", "heralds", "opp_heralds",
    "void_grubs", "opp_void_grubs",
    "firstbaron", "barons", "opp_barons",
    "firsttower", "towers", "opp_towers",
    "firstmidtower", "firsttothreetowers",
    "turretplates", "opp_turretplates",
    "golddiffat10", "xpdiffat10", "csdiffat10",
    "killsat10", "deathsat10",
    "golddiffat15", "xpdiffat15", "csdiffat15",
    "killsat15", "deathsat15",
]


def _aggregate_to_games(raw: pd.DataFrame) -> pd.DataFrame:
    """Transforma 12 rows/game em 1 row/game."""
    team_rows = raw[raw["participantid"].isin([100, 200])][_TEAM_COLS].copy()
    player_rows = raw[raw["participantid"].between(1, 10)][
        ["gameid", "side", "position", "playername", "playerid", "champion",
         "kills", "deaths", "assists"]
    ].copy()

    # team_rows -> 1 row/jogo (pivot blue/red)
    blue = team_rows[team_rows["side"] == "Blue"].add_prefix("blue_")
    red  = team_rows[team_rows["side"] == "Red"].add_prefix("red_")

    games = blue.merge(
        red,
        left_on="blue_gameid",
        right_on="red_gameid",
        how="inner",
    )

    # Colunas comuns ficam de uma so vez
    common_cols = ["gameid", "date", "league", "year", "split", "patch", "game", "gamelength"]
    for col in common_cols:
        games[col] = games[f"blue_{col}"]
        games = games.drop(columns=[f"blue_{col}", f"red_{col}"])

    # Resultado: 1 = blue venceu
    games["blue_won"] = (games["blue_result"] == 1).astype(int)
    games = games.drop(columns=["blue_result", "red_result", "blue_side", "red_side"])

    # Picks em ordem de draft (lista)
    for side in ("blue", "red"):
        pick_cols = [f"{side}_pick{i}" for i in range(1, 6)]
        ban_cols  = [f"{side}_ban{i}" for i in range(1, 6)]
        games[f"{side}_picks_draft"] = games[pick_cols].values.tolist()
        games[f"{side}_bans"] = games[ban_cols].values.tolist()
        games = games.drop(columns=pick_cols + ban_cols)

    # Player champs por role
    pivot = player_rows.pivot_table(
        index=["gameid", "side"],
        columns="position",
        values="champion",
        aggfunc="first",
    ).reset_index()

    # Players por role (para player_form_diff)
    players_pivot = player_rows.pivot_table(
        index=["gameid", "side"],
        columns="position",
        values="playerid",
        aggfunc="first",
    ).reset_index()

    # Player names por role (para name -> id lookup no app)
    playernames_pivot = player_rows.pivot_table(
        index=["gameid", "side"],
        columns="position",
        values="playername",
        aggfunc="first",
    ).reset_index()

    blue_champs = pivot[pivot["side"] == "Blue"].drop(columns=["side"]).add_prefix("blue_")
    blue_champs = blue_champs.rename(columns={"blue_gameid": "gameid"})
    red_champs  = pivot[pivot["side"] == "Red"].drop(columns=["side"]).add_prefix("red_")
    red_champs  = red_champs.rename(columns={"red_gameid": "gameid"})
    games = games.merge(blue_champs, on="gameid").merge(red_champs, on="gameid")

    blue_players = players_pivot[players_pivot["side"] == "Blue"].drop(columns=["side"])
    blue_players = blue_players.rename(columns={
        r: f"blue_player_{r}" for r in _PLAYER_ROLES
    })
    red_players  = players_pivot[players_pivot["side"] == "Red"].drop(columns=["side"])
    red_players  = red_players.rename(columns={
        r: f"red_player_{r}" for r in _PLAYER_ROLES
    })
    games = games.merge(blue_players, on="gameid").merge(red_players, on="gameid")

    # Player names
    blue_pn = playernames_pivot[playernames_pivot["side"] == "Blue"].drop(columns=["side"])
    blue_pn = blue_pn.rename(columns={r: f"blue_playername_{r}" for r in _PLAYER_ROLES})
    red_pn  = playernames_pivot[playernames_pivot["side"] == "Red"].drop(columns=["side"])
    red_pn  = red_pn.rename(columns={r: f"red_playername_{r}" for r in _PLAYER_ROLES})
    games = games.merge(blue_pn, on="gameid").merge(red_pn, on="gameid")

    # K/D/A por jogador (comfort de performance: KDA no campeao) -> blue_kills_top etc.
    for stat in ("kills", "deaths", "assists"):
        pv = player_rows.pivot_table(index=["gameid", "side"], columns="position",
                                     values=stat, aggfunc="first").reset_index()
        for side, prefix in (("Blue", "blue"), ("Red", "red")):
            part = pv[pv["side"] == side].drop(columns=["side"])
            part = part.rename(columns={r: f"{prefix}_{stat}_{r}" for r in _PLAYER_ROLES})
            games = games.merge(part, on="gameid", how="left")

    # firstPick: 1 = blue picou primeiro
    games["blue_first_pick"] = (games["blue_firstPick"] == 1).astype(int)
    games = games.drop(columns=["blue_firstPick", "red_firstPick"])

    # Normaliza date
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["date"])
    games = games.sort_values("date").reset_index(drop=True)

    return games


if __name__ == "__main__":
    print("Anos disponiveis:", list_available_years())
    df = load_year(2025, only_tier1=False)
    print(f"\n2025: {len(df)} jogos apos agregacao")
    print("\nColunas:", list(df.columns))
    print("\nPrimeira linha:")
    print(df.iloc[0].to_dict())
