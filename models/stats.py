"""Dataclasses representando as estatisticas agregadas que a UI consome."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PlayerStat:
    """Stats de um jogador em uma janela de tempo (ano corrente, por padrao)."""

    name: str
    role: str           # TOP, JNG, MID, BOT, SUP
    team: str
    league: str         # ex: LCK, LEC, LCS, LCP
    # Como jogador (todos os campeoes, no ano): K/D
    player_kills_avg: float = 0.0
    player_deaths_avg: float = 0.0
    # Como campeao do prox jogo (se selecionado): K/D nesse campeao no ano
    champion_name: Optional[str] = None
    champion_kills_avg: float = 0.0
    champion_deaths_avg: float = 0.0
    champion_games: int = 0
    champion_winrate: float = 0.0
    # Media combinada (heuristica simples = media dos dois quando ha jogos do campeao)
    combined_kills_avg: float = 0.0
    combined_deaths_avg: float = 0.0
    games_total: int = 0


@dataclass
class TeamStat:
    """Stats agregadas de um time no ano corrente."""

    name: str
    league: str = ""
    total_kills_avg: float = 0.0
    duration_avg_min: float = 0.0
    over_kills_pct: float = 0.0       # % de jogos acima da mediana de kills
    under_kills_pct: float = 0.0
    over_time_pct: float = 0.0
    under_time_pct: float = 0.0
    games: int = 0
    players: List[PlayerStat] = field(default_factory=list)


@dataclass
class LaneMatchup:
    role: str
    player_blue: str
    player_red: str
    kills_delta: float    # blue.player_kills_avg - red.player_kills_avg


@dataclass
class TeamMatchup:
    """Comparativo Time x Time."""

    blue: TeamStat
    red: TeamStat
    winrate_blue_vs_red: float        # winrate historico entre os dois (Blue side)
    kills_diff: float                 # blue.total_kills_avg - red.total_kills_avg
    deaths_diff: float                # diff de mortes medias
    lane_matchups: List[LaneMatchup] = field(default_factory=list)
