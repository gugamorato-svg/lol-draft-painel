"""Contexto temporal sem leakage para o modelo de WR de LoL.

CRITICO (licao do bug que custou 17pp no projeto Dota):
  Counters e synergies precisam ser ATUALIZADOS a cada match no loop temporal.
  Se ficarem zerados durante todo o treino, o gradiente nao aprende e o
  coeficiente trava em zero - anulando a feature mais importante do modelo.

Uso:
    ctx = TemporalContext()
    for game in games_sorted_by_date:
        features = ctx.build_features(game)   # so usa estado passado
        train_step(features, game.blue_won)
        ctx.update(game)                       # incorpora resultado pro proximo

O state interno acumula:
  - champion_matchups: dict {(own, enemy, role): {"games": n, "wins": w}}
  - synergies:         dict {(champ_a, champ_b): {"games": n, "wins": w}}
  - team_rating:       dict {team_id: elo}
  - team_form:         dict {team_id: deque[bool]} ultimos 10
  - player_form:       dict {player_id: deque[bool]} ultimos 10
  - h2h:               dict {(team_a, team_b): {"games": n, "wins_a": w}}
  - champion_pickrate: dict {champion: count} para ban_score
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Deque, Dict, List, Optional, Tuple

from data_pipeline.patches import oe_patch_key

# ----- constantes (transferidas do Dota; ajustar quando rodar calibracao) -----

SMOOTHING_PRIOR_MATCHUP = 20    # Bayesian prior para counter scores
SMOOTHING_PRIOR_SYNERGY = 30    # synergies precisam de mais jogos
SMOOTHING_PRIOR_H2H     = 10    # H2H entre times converge mais rapido

SCORE_SCALE_MATCHUP  = 10
SCORE_SCALE_SYNERGY  = 10
SCORE_CAP_MATCHUP    = 2.5      # cap de -2.5 a +2.5
SCORE_CAP_SYNERGY    = 2.0

# Oracle e fonte principal. Gol.gg entra so como fallback fraco/capado quando
# a amostra Oracle ainda e pequena, evitando que prior externo domine o modelo.
ORACLE_MIN_MATCHUP_GAMES = 8
ORACLE_MIN_SYNERGY_GAMES = 12
GOLGG_FALLBACK_WEIGHT = 0.50
GOLGG_FALLBACK_MATCHUP_CAP = 18
GOLGG_FALLBACK_SYNERGY_CAP = 24

ELO_K_FACTOR = 24
ELO_INITIAL  = 1500.0

# FORCA DA LIGA (22/09/2026). O Elo e uma piscina quase fechada por liga: dentro do LCK
# os times so jogam entre si, entao o rating diz "quem e melhor no LCK", nao "quao forte
# e o LCK". Medido: em jogo ENTRE ligas o modelo subestimava LPL (+14,6pp) e superestimava
# LCP (-27pp); juntar a forca da liga levou o log loss desses jogos de 0,6292 p/ 0,5757
# (IC95 [-0,082,-0,024]) e o acerto de 65,9% p/ 70,4%.
# Rating em unidades de LOGIT, atualizado so quando duas ligas se enfrentam.
LEAGUE_MIN_GAMES = 8       # jogos p/ uma liga virar a "casa" do time
LEAGUE_REFIT_EVERY = 50    # re-ajusta a forca a cada N jogos entre ligas
LEAGUE_MIN_CROSS = 150     # so comeca a valer depois de N jogos entre ligas
LEAGUE_REG_C = 0.5         # regularizacao do ajuste (liga com poucos jogos encolhe p/ 0)
# Ligas que sao EVENTO: nao definem a casa de ninguem (todo mundo e visitante).
EVENT_LEAGUES = {"WLDs", "MSI", "FST", "EWC", "CCWS", "AC", "WSCI", "AM", "Asia Master",
                 "EM", "KeSPA Cup", "SOOP Cross Region Invitational"}

# LIGAS DE BASE / 2a DIVISAO (24/09/2026, scripts/study_academy_strength.py). Controlado
# pelas chaves da classe (home_decay, league_aliases, league_in_rating, league_parents,
# league_elo); o que esta ligado por padrao e o que pagou no estudo.
# Nome antigo/evento local -> liga de verdade (renomeada nao e outra liga).
LEAGUE_ALIASES = {"LVP SL": "LES", "LTA N": "LCS", "LTA S": "CBLOL", "ESLOL": "ROL",
                  "KeSPA": "LCK", "DCup": "LPL"}
# Liga filha -> liga mae. Sem ranking de base, a forca da filha parte de "mae - degrau"
# (a ordem das ligas principais) e so se afasta disso com jogo entre ligas.
LEAGUE_PARENTS = {
    "LCKC": "LCK", "LAS": "LCKC",
    "LDL": "LPL",
    "NACL": "LCS",
    "CBLOLA": "CBLOL", "CD": "CBLOL", "IGNIS": "CBLOL",
    "LRN": "LLA", "LRS": "LLA",
    "PCS": "LCP", "PCL": "PCS", "VCS": "LCP", "LJL": "LCP", "LCO": "LCP",
    "LFL": "LEC", "LES": "LEC", "PRM": "LEC", "NLC": "LEC", "TCL": "LEC", "EBL": "LEC",
    "HLL": "LEC", "LIT": "LEC", "ROL": "LEC", "LPLOL": "LEC", "GLL": "LEC", "AL": "LEC",
    "UL": "LEC", "HM": "LEC", "RL": "LEC", "EPL": "LEC",
    "LFL2": "LFL", "NL": "LFL", "CDF": "LFL", "NEXO": "LES", "CT": "LES", "IC": "LES",
    "PRMP": "PRM", "NLC Aurora Open": "NLC", "TSC": "TCL", "EBLPA": "EBL", "GLLPA": "GLL",
    "USP": "UL", "HC": "HM", "HW": "HM",
}
LEAGUE_GAP_SCALE = 3.0     # coluna do degrau x3 = regularizacao 9x mais fraca nele
ELO_PER_LOGIT = 400.0 / math.log(10.0)

ELO_CLAMP    = 250.0            # rating_diff dividido por 250 -> range ~+/-1.5

FORM_WINDOW  = 10               # janela movel para form
PLAYER_FORM_INITIAL = 0.5

# Meta recente dos campeoes. Decay por jogo: meia-vida de ~346 jogos,
# o bastante para seguir patch/meta sem apagar totalmente ligas pequenas.
CHAMPION_META_DECAY = 0.998
CHAMPION_META_WR_PRIOR = 12.0
CHAMPION_META_WR_WEIGHT = 0.30

PLAYER_CHAMPION_SAMPLE_LOG_CAP = 50.0

# COMFORT DE PERFORMANCE (23/09/2026): KDA (k+a)/(d+1) do jogador no campeao ACIMA do normal
# daquele campeao na mesma funcao, media encolhida p/ 0 com PRIOR jogos virtuais. No estudo
# de 2026 ajudou o modelo draft+comfort de forma consistente
# nos 2 semestres. DPM, ouro@15 e "boost WR x performance" nao ajudaram. So entra no
# draft+comfort (model_coefficients_zero.csv), nao no modelo completo (nao testado la).
COMFORT_KDA_PRIOR = 5.0
COMFORT_KDA_SCALE = 2.5
TEAM_CHAMPION_PRIOR = 10
PATCH_CHAMPION_PRIOR = 8
LEAGUE_CHAMPION_PRIOR = 12   # ligas menores tem amostra menor; prior mais conservador
TEAM_SIDE_PRIOR = 20
TEAM_SERIES_PRIOR = 15

# --- pre_match_block (descoberta do Dota) ---
# Junta as features de "forca pre-jogo" (rating, h2h, forma) num bloco unico.
# O mercado/odd JA precifica esse contexto -> capamos para nao dominar o draft.
# Pesos internos = proporcao da importancia aprendida (ver model_coefficients).
# Mantemos pre_match_RAW (sem cap) no output; o cap e aplicado no treino,
# permitindo varrer varios caps sem re-rodar o loop temporal.
PREMATCH_WEIGHTS = {
    "player_form_diff": 1.04,
    "rating_diff":      0.87,
    "recent_form_diff": -0.26,   # coef negativo (mean reversion)
    "h2h_diff":         0.14,
}


_ROLES = ("top", "jng", "mid", "bot", "sup")


# ----- helpers -----

def _norm(s: Optional[str]) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return str(s).strip()


def _h2h_key(team_a: str, team_b: str) -> Tuple[str, str, bool]:
    """Retorna chave ordenada e flag indicando se trocou."""
    if team_a <= team_b:
        return (team_a, team_b, False)
    return (team_b, team_a, True)


def _synergy_key(c1: str, c2: str) -> Tuple[str, str]:
    return (c1, c2) if c1 <= c2 else (c2, c1)


def _smoothed(wins: float, games: int, prior: int, scale: float, cap: float) -> float:
    """Bayesian smoothed signal. games=0 -> 0.0."""
    if games <= 0:
        return 0.0
    centered = wins - 0.5 * games
    return max(-cap, min(cap, centered / (games + prior) * scale))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _num(value: Any) -> Optional[float]:
    """float, ou None quando a coluna esta vazia. Jogo "partial" do Oracle (LPL 2024-25)
    nao tem os numeros de 15 min: vazio NAO pode virar 0 (primeiro dragao = 0 seria
    "nunca pega o primeiro dragao")."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


# perfil de objetivos: estatistica -> (coluna, coluna do adversario p/ diferenca)
_OBJECTIVE_STATS = {
    "firstdragon": ("firstdragon", None),
    "dragondiff": ("dragons", "opp_dragons"),
    "firstherald": ("firstherald", None),
    "heralddiff": ("heralds", "opp_heralds"),
    "voidgrubdiff": ("void_grubs", "opp_void_grubs"),
    "firstbaron": ("firstbaron", None),
    "barondiff": ("barons", "opp_barons"),
    "firsttower": ("firsttower", None),
    "towerdiff": ("towers", "opp_towers"),
    "firstmidtower": ("firstmidtower", None),
    "firsttothreetowers": ("firsttothreetowers", None),
    "platediff": ("turretplates", "opp_turretplates"),
}


def _game_number(game: Dict[str, Any]) -> int:
    try:
        n = int(float(game.get("game", game.get("game_number", 1)) or 1))
    except (TypeError, ValueError):
        n = 1
    return max(1, min(7, n))


# ----- state -----

@dataclass
class TemporalContext:
    champion_matchups: Dict[Tuple[str, str, str], Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"games": 0, "wins": 0.0})
    )
    synergies: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"games": 0, "wins": 0.0})
    )
    prior_champion_matchups: Dict[Tuple[str, str, str], Dict[str, float]] = field(
        default_factory=dict
    )
    prior_synergies: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=dict
    )
    @classmethod
    def with_prior(cls, use_golgg: bool = True) -> "TemporalContext":
        """Cria contexto Oracle-first com Gol.gg separado como fallback.

        O prior nao e somado ao estado Oracle. Ele fica em dicionarios separados
        e so e consultado quando a amostra Oracle do par ainda e pequena.
        """
        ctx = cls()
        if not use_golgg:
            return ctx
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from data_pipeline.golgg_prior import (
                load_matchups_prior, load_synergies_prior,
            )
            mu = load_matchups_prior()
            sy = load_synergies_prior()
            ctx.prior_champion_matchups = {key: dict(rec) for key, rec in mu.items()}
            ctx.prior_synergies = {key: dict(rec) for key, rec in sy.items()}
            print(f"[OK] Prior gol.gg carregado como fallback: {len(mu)} matchups, {len(sy)} synergies")
        except Exception as exc:
            print(f"[!] Falha ao carregar prior gol.gg: {exc}")
        return ctx
    team_rating: Dict[str, float] = field(
        default_factory=lambda: defaultdict(lambda: ELO_INITIAL)
    )
    league_rating: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    team_league_games: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    cross_league_games: List[Tuple[str, str, int]] = field(default_factory=list)
    team_form: Dict[str, Deque[int]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    )
    player_form: Dict[str, Deque[int]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    )
    h2h: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"games": 0, "wins_a": 0.0})
    )
    champion_pickrate: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    champion_recent_meta: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"games": 0.0, "wins": 0.0})
    )
    # (player_id, champion) -> {games, wins} para player_champion_comfort
    player_champion: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"games": 0, "wins": 0.0})
    )
    team_early_profile: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {
            "games": 0.0, "gold15": 0.0, "xp15": 0.0, "cs15": 0.0,
            "killdiff15": 0.0, "firstblood": 0.0,
        })
    )
    team_objective_profile: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {
            "games": 0.0, "firstdragon": 0.0, "dragondiff": 0.0,
            "firstherald": 0.0, "heralddiff": 0.0, "voidgrubdiff": 0.0,
            "firstbaron": 0.0, "barondiff": 0.0,
            "firsttower": 0.0, "towerdiff": 0.0,
            "firstmidtower": 0.0, "firsttothreetowers": 0.0, "platediff": 0.0,
        })
    )
    team_side: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"games": 0.0, "wins": 0.0})
    )
    team_series: Dict[Tuple[str, int], Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"games": 0.0, "wins": 0.0})
    )
    team_champion: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"games": 0.0, "wins": 0.0})
    )
    patch_champion: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"games": 0.0, "wins": 0.0})
    )
    # (league, champion) -> {games, wins} -- captura diferenca regional de
    # forca/meta. Lucian na LEC pode ter WR 40% e na LAS WR 60%.
    league_champion: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"games": 0.0, "wins": 0.0})
    )
    # Total de jogos por player (independente de champion). Usado para feature
    # de TAXA de comfort: games_on_champ / total_player_games. Liga-neutra.
    player_total_games: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # (player_id, champion) -> {n, rel}: soma do KDA acima do normal do campeao na funcao
    player_champion_kda: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"n": 0.0, "rel": 0.0})
    )
    # (champion, role) -> {n, sum}: KDA medio do campeao naquela funcao (a referencia)
    champion_role_kda: Dict[Tuple[str, str], Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: {"n": 0.0, "sum": 0.0})
    )
    n_games_seen: int = 0

    # Chaves das ligas de base (sem anotacao = atributo de classe, fora do snapshot).
    # Estudo 24/09 (study_academy_strength.py, log loss 2025+ walk-forward): LIGADAS as que
    # pagaram -- liga de casa recente + nomes antigos (-0,0010) e forca da liga dentro do
    # rating_diff (-0,0012; eventos de base -0,012). DESLIGADAS as que nao pagaram: forca
    # hierarquica mae/filha (piorou eventos +0,010) e Elo entre ligas no update (empate).
    home_decay = 0.97                    # liga de casa = a recente, nao a da vida toda
    league_aliases = LEAGUE_ALIASES
    league_in_rating = True              # rating_diff = Elo + forca da liga de casa
    league_parents = None                # LEAGUE_PARENTS (forca da liga hierarquica)
    league_elo = False                   # Elo entre ligas espera o resultado c/ a forca da liga

    # ---------- feature extraction (read-only no state) ----------

    def build_features(self, game: Dict[str, Any]) -> Dict[str, float]:
        """Extrai dict de features para 1 jogo, usando apenas estado passado.

        game e um dict com as colunas geradas pelo oracle_loader:
          blue_top/jng/mid/bot/sup, red_top/jng/mid/bot/sup,
          blue_picks_draft (ordem), red_picks_draft (ordem),
          blue_bans, red_bans,
          blue_player_top/jng/.../sup, red_player_...,
          blue_teamid, red_teamid.
        """
        blue_champs = [_norm(game[f"blue_{r}"]) for r in _ROLES]
        red_champs  = [_norm(game[f"red_{r}"])  for r in _ROLES]
        blue_picks_draft = [_norm(c) for c in (game.get("blue_picks_draft") or [])]
        red_picks_draft  = [_norm(c) for c in (game.get("red_picks_draft")  or [])]
        blue_bans = [_norm(c) for c in (game.get("blue_bans") or [])]
        red_bans  = [_norm(c) for c in (game.get("red_bans") or [])]

        blue_team = _norm(game.get("blue_teamid"))
        red_team  = _norm(game.get("red_teamid"))
        blue_players = [_norm(game.get(f"blue_player_{r}")) for r in _ROLES]
        red_players  = [_norm(game.get(f"red_player_{r}"))  for r in _ROLES]
        patch = oe_patch_key(game.get("patch"))  # live "16.9"/"16.10" -> chave do Oracle
        game_no = _game_number(game)

        # 1. lane_matchup_diff: matchup por role
        lane_signal = 0.0
        for r in _ROLES:
            bc = _norm(game[f"blue_{r}"])
            rc = _norm(game[f"red_{r}"])
            lane_signal += self._matchup_score(bc, rc, r)
        lane_matchup_diff = lane_signal / 10.0

        # 2. flex_matchup_diff (todos os pares cross-team)
        flex_signal = 0.0
        for bc in blue_champs:
            for rc in red_champs:
                flex_signal += self._matchup_score(bc, rc, "")
        # 5 inimigos por champ -> normaliza por 5*5
        flex_matchup_diff = flex_signal / 25.0 / 10.0

        # 3. counter_diff: soma counters cross-time, normalizado.
        # _matchup_score(rc, bc) == -_matchup_score(bc, rc) por simetria do
        # _smoothed, entao red_counter = -blue_counter. Calculamos so um lado.
        # 25 pares (5x5), cap individual de +-2 -> sinal vai de +-50, dividimos
        # por 25/2 = 12.5 (equivalente a (blue-red)/5/5/2 da formula antiga).
        counter_signal = sum(
            self._matchup_score(bc, rc, "")
            for bc in blue_champs for rc in red_champs
        )
        counter_diff = counter_signal / 12.5

        # 4. last_pick_counter_diff: counter so do ultimo pick
        # blue_picks_draft[-1] e o ultimo champ que o blue picou
        last_pick_diff = 0.0
        if blue_picks_draft and red_picks_draft:
            blue_last = blue_picks_draft[-1]
            red_last  = red_picks_draft[-1]
            blue_lp_signal = sum(
                self._matchup_score(blue_last, rc, "") for rc in red_champs
            )
            red_lp_signal = sum(
                self._matchup_score(red_last, bc, "") for bc in blue_champs
            )
            last_pick_diff = (blue_lp_signal - red_lp_signal) / 5.0 / 5.0

        # 5. synergy_diff: pares mesmo time
        blue_syn = self._team_synergy(blue_champs)
        red_syn  = self._team_synergy(red_champs)
        synergy_diff = (blue_syn - red_syn) / 5.0

        # 6. rating_diff (Elo)
        blue_rating = self.team_rating[blue_team]
        red_rating  = self.team_rating[red_team]
        rating_raw = blue_rating - red_rating
        if self.league_in_rating:   # Elo e por liga: soma a forca da liga de casa (em pts)
            rating_raw += self.league_elo_offset(blue_team, red_team)
        rating_clamped = max(-ELO_CLAMP * 1.5, min(ELO_CLAMP * 1.5, rating_raw))
        rating_diff = rating_clamped / ELO_CLAMP

        # 6b. league_strength_diff: forca da LIGA de casa de cada time. Vale 0 quando
        # os dois jogam na mesma liga (90% dos jogos); so pesa em evento internacional.
        league_strength_diff = (self._league_strength(blue_team)
                                - self._league_strength(red_team))

        # 7. ban_score_diff: quanto cada time tirou do adversario
        # heuristica: se um champ banido tem alta pickrate, ban "vale mais"
        blue_ban_score = sum(self._ban_value(b) for b in blue_bans)
        red_ban_score  = sum(self._ban_value(b) for b in red_bans)
        ban_score_diff = (blue_ban_score - red_ban_score) / 10.0

        # 8. player_form_diff: soma form-WR dos 5 jogadores
        blue_pf = sum(self._player_form_wr(p) for p in blue_players)
        red_pf  = sum(self._player_form_wr(p) for p in red_players)
        player_form_diff = (blue_pf - red_pf) / 5.0

        # 9. h2h_diff: WR direto smoothed
        h2h_signal = self._h2h_score(blue_team, red_team)
        h2h_diff = h2h_signal

        # 10. recent_form_diff: WR janela movel
        blue_tf = self._team_form_wr(blue_team)
        red_tf  = self._team_form_wr(red_team)
        recent_form_diff = (blue_tf - red_tf)

        # 11. champion_meta_diff: forca recente do pick no pro play.
        # Usa so jogos anteriores por causa do loop temporal.
        champion_meta_diff = (
            self._team_meta_score(blue_champs) - self._team_meta_score(red_champs)
        )

        # 12. side_advantage: blue side tem ~52% WR estrutural em pro LoL
        # Como sempre estamos predizendo P(blue wins), basta o intercept
        # capturar isso. Aqui mantemos como feature constante = 1 pro
        # modelo isolar o efeito do lado.
        side_advantage = 1.0

        # 12. first_pick_diff: blue picou primeiro?
        # +1 se blue picou primeiro, -1 se red picou primeiro
        first_pick_diff = 0.0
        if "blue_first_pick" in game:
            try:
                fp = int(game["blue_first_pick"])
                first_pick_diff = 2 * fp - 1
            except (ValueError, TypeError):
                pass

        # 13. player_champion_comfort_diff:
        # WR de cada player no champion que vai jogar (Bayesian smoothed)
        blue_comfort = 0.0
        red_comfort  = 0.0
        for r, p, c in zip(_ROLES, blue_players, blue_champs):
            blue_comfort += self._player_champion_score(p, c)
        for r, p, c in zip(_ROLES, red_players, red_champs):
            red_comfort  += self._player_champion_score(p, c)
        player_champion_diff = (blue_comfort - red_comfort) / 5.0

        blue_sample = sum(
            self._player_champion_sample_volume(p, c)
            for p, c in zip(blue_players, blue_champs)
        ) / 5.0
        red_sample = sum(
            self._player_champion_sample_volume(p, c)
            for p, c in zip(red_players, red_champs)
        ) / 5.0
        sample_size_confidence_diff = blue_sample - red_sample

        # Taxa de comfort (liga-neutra): % do tempo que o player joga aquele champ
        blue_rate = sum(
            self._player_champion_rate_score(p, c)
            for p, c in zip(blue_players, blue_champs)
        ) / 5.0
        red_rate = sum(
            self._player_champion_rate_score(p, c)
            for p, c in zip(red_players, red_champs)
        ) / 5.0
        player_champion_rate_diff = blue_rate - red_rate

        # KDA do jogador no campeao acima do normal do campeao (comfort de performance)
        comfort_kda_diff = (
            sum(self._comfort_kda_score(p, c) for p, c in zip(blue_players, blue_champs))
            - sum(self._comfort_kda_score(p, c) for p, c in zip(red_players, red_champs))
        ) / 5.0

        team_patch_meta_diff = (
            self._team_champion_meta_score(blue_team, blue_champs)
            - self._team_champion_meta_score(red_team, red_champs)
        )
        jungle_mid_synergy_diff = (
            self._pair_synergy_score(blue_champs[1], blue_champs[2])
            - self._pair_synergy_score(red_champs[1], red_champs[2])
        )
        bot_2v2_synergy_diff = (
            self._pair_synergy_score(blue_champs[3], blue_champs[4])
            - self._pair_synergy_score(red_champs[3], red_champs[4])
        )
        early_game_team_diff = self._team_early_score(blue_team) - self._team_early_score(red_team)
        objective_control_diff = self._team_objective_score(blue_team) - self._team_objective_score(red_team)
        side_specific_team_wr_diff = (
            self._team_side_score(blue_team, "blue")
            - self._team_side_score(red_team, "red")
        )
        series_context_diff = (
            self._team_series_score(blue_team, game_no)
            - self._team_series_score(red_team, game_no)
        )
        patch_champion_wr_diff = (
            self._patch_champion_wr_score(patch, blue_champs)
            - self._patch_champion_wr_score(patch, red_champs)
        )
        # Liga-especifico: champions tem WR diferentes entre regioes
        league = _norm(game.get("league") or game.get("league_name"))
        league_champion_wr_diff = (
            self._league_champion_wr_score(league, blue_champs)
            - self._league_champion_wr_score(league, red_champs)
        )

        # pre_match_block (RAW, sem cap): combina forca pre-jogo em logit units.
        # O cap moderado e aplicado no treino/predicao (ver PREMATCH_CAP).
        pre_match_raw = (
            PREMATCH_WEIGHTS["player_form_diff"] * player_form_diff
            + PREMATCH_WEIGHTS["rating_diff"]      * rating_diff
            + PREMATCH_WEIGHTS["recent_form_diff"] * recent_form_diff
            + PREMATCH_WEIGHTS["h2h_diff"]         * h2h_diff
        )

        return {
            "lane_matchup_diff":   lane_matchup_diff,
            "flex_matchup_diff":   flex_matchup_diff,
            "counter_diff":        counter_diff,
            "last_pick_counter_diff": last_pick_diff,
            "synergy_diff":        synergy_diff,
            "rating_diff":         rating_diff,
            "ban_score_diff":      ban_score_diff,
            "player_form_diff":    player_form_diff,
            "h2h_diff":            h2h_diff,
            "recent_form_diff":    recent_form_diff,
            "champion_meta_diff":  champion_meta_diff,
            "side_advantage":      side_advantage,
            "first_pick_diff":     first_pick_diff,
            "player_champion_diff": player_champion_diff,
            "sample_size_confidence_diff": sample_size_confidence_diff,
            "team_patch_meta_diff": team_patch_meta_diff,
            "jungle_mid_synergy_diff": jungle_mid_synergy_diff,
            "bot_2v2_synergy_diff": bot_2v2_synergy_diff,
            "early_game_team_diff": early_game_team_diff,
            "objective_control_diff": objective_control_diff,
            "side_specific_team_wr_diff": side_specific_team_wr_diff,
            "series_context_diff": series_context_diff,
            "patch_champion_wr_diff": patch_champion_wr_diff,
            "league_champion_wr_diff": league_champion_wr_diff,
            "league_strength_diff": league_strength_diff,
            "player_champion_rate_diff": player_champion_rate_diff,
            "comfort_kda_diff": comfort_kda_diff,
            "pre_match_raw": pre_match_raw,
        }

    # ---------- score helpers ----------

    def _score_oracle_first(
        self,
        oracle_rec: Optional[Dict[str, float]],
        prior_rec: Optional[Dict[str, float]],
        *,
        min_oracle_games: int,
        prior_games_cap: int,
        smoothing_prior: int,
        score_scale: float,
        score_cap: float,
    ) -> Optional[float]:
        """Score Oracle-first com Gol.gg apenas como fallback/smoothing.

        Regras:
          1. Se Oracle tem amostra suficiente, usa Oracle puro.
          2. Se Oracle tem pouca amostra, usa um pedaco capado do prior Gol.gg
             so para completar a amostra ate o minimo.
          3. Se Oracle nao tem nada, usa Gol.gg fraco e capado.
        """
        oracle_games = float((oracle_rec or {}).get("games", 0) or 0)
        oracle_wins = float((oracle_rec or {}).get("wins", 0.0) or 0.0)
        if oracle_games >= min_oracle_games:
            return _smoothed(oracle_wins, oracle_games, smoothing_prior, score_scale, score_cap)

        prior_games = float((prior_rec or {}).get("games", 0) or 0)
        prior_wins = float((prior_rec or {}).get("wins", 0.0) or 0.0)
        if prior_games > 0:
            prior_wr = prior_wins / prior_games
            if oracle_games > 0:
                missing_games = max(0.0, min_oracle_games - oracle_games)
                fallback_games = min(prior_games, prior_games_cap, missing_games) * GOLGG_FALLBACK_WEIGHT
                games = oracle_games + fallback_games
                wins = oracle_wins + prior_wr * fallback_games
            else:
                games = min(prior_games, prior_games_cap) * GOLGG_FALLBACK_WEIGHT
                wins = prior_wr * games
            if games > 0:
                return _smoothed(wins, games, smoothing_prior, score_scale, score_cap)

        if oracle_games > 0:
            return _smoothed(oracle_wins, oracle_games, smoothing_prior, score_scale, score_cap)
        return None

    def _matchup_score(self, own: str, opp: str, role: str) -> float:
        """Score Bayesian smoothed do matchup own vs opp (Oracle-first)."""
        if not own or not opp:
            return 0.0
        # Tenta com role primeiro; fallback sem role.
        for key in [(own, opp, role), (own, opp, "")]:
            score = self._score_oracle_first(
                self.champion_matchups.get(key),
                self.prior_champion_matchups.get(key),
                min_oracle_games=ORACLE_MIN_MATCHUP_GAMES,
                prior_games_cap=GOLGG_FALLBACK_MATCHUP_CAP,
                smoothing_prior=SMOOTHING_PRIOR_MATCHUP,
                score_scale=SCORE_SCALE_MATCHUP,
                score_cap=SCORE_CAP_MATCHUP,
            )
            if score is not None:
                return score
        # Tenta inverso (signal negativo) para snapshots antigos/parciais.
        for key in [(opp, own, role), (opp, own, "")]:
            score = self._score_oracle_first(
                self.champion_matchups.get(key),
                self.prior_champion_matchups.get(key),
                min_oracle_games=ORACLE_MIN_MATCHUP_GAMES,
                prior_games_cap=GOLGG_FALLBACK_MATCHUP_CAP,
                smoothing_prior=SMOOTHING_PRIOR_MATCHUP,
                score_scale=SCORE_SCALE_MATCHUP,
                score_cap=SCORE_CAP_MATCHUP,
            )
            if score is not None:
                return -score
        return 0.0

    def _team_synergy(self, champs: List[str]) -> float:
        total = 0.0
        for a, b in combinations(champs, 2):
            if not a or not b:
                continue
            key = _synergy_key(a, b)
            score = self._score_oracle_first(
                self.synergies.get(key),
                self.prior_synergies.get(key),
                min_oracle_games=ORACLE_MIN_SYNERGY_GAMES,
                prior_games_cap=GOLGG_FALLBACK_SYNERGY_CAP,
                smoothing_prior=SMOOTHING_PRIOR_SYNERGY,
                score_scale=SCORE_SCALE_SYNERGY,
                score_cap=SCORE_CAP_SYNERGY,
            )
            if score is not None:
                total += score
        return total

    def _pair_synergy_score(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        key = _synergy_key(a, b)
        score = self._score_oracle_first(
            self.synergies.get(key),
            self.prior_synergies.get(key),
            min_oracle_games=ORACLE_MIN_SYNERGY_GAMES,
            prior_games_cap=GOLGG_FALLBACK_SYNERGY_CAP,
            smoothing_prior=SMOOTHING_PRIOR_SYNERGY,
            score_scale=SCORE_SCALE_SYNERGY,
            score_cap=SCORE_CAP_SYNERGY,
        )
        return score or 0.0

    def _player_champion_score(self, player_id: str, champion: str) -> float:
        """WR Bayesian smoothed do player no champion. Range [-2, +2].

        Aplica cap de 50 jogos para equalizar amostras entre ligas:
        players de LCK/LPL jogam muito mais que players de LEC/CBLOL,
        e antes do cap o smoothing puxava o score do player de liga
        pequena mais forte pra 50%, mesmo com WR identico.
        """
        if not player_id or not champion:
            return 0.0
        rec = self.player_champion.get((player_id, champion))
        if not rec or rec["games"] <= 0:
            return 0.0
        games = rec["games"]
        wins  = rec["wins"]
        # Fix A: cap em 50 jogos. Player de LCK com 200 jogos eh tratado
        # como 50. Player de LEC com 80 jogos tambem cap em 50. Mesma WR
        # = mesmo score, independente de liga.
        if games > 50.0:
            wins = wins * (50.0 / games)
            games = 50.0
        return _smoothed(wins, games, 10, 10.0, 2.0)

    def _player_champion_rate_score(self, player_id: str, champion: str) -> float:
        """Taxa de comfort do player no champion: games_on_champ / total_games.

        Liga-neutra: Faker que joga Azir 25% do tempo recebe o mesmo sinal
        que Caps que joga Azir 25% do tempo, mesmo Faker tendo 4x mais
        jogos absolutos. Smooth toward 0 com prior=30 jogos virtuais.

        Centrado em 10% (limiar tipico de comfort), range ~[-0.5, +4.5].
        Cap em +3 pra nao explodir com one-tricks (60-80% rate).
        """
        if not player_id or not champion:
            return 0.0
        rec = self.player_champion.get((player_id, champion))
        if not rec or rec["games"] <= 0:
            return 0.0
        total = self.player_total_games.get(player_id, 0)
        if total < 10:   # cold start: player muito novo, sinal nao confiavel
            return 0.0
        games_on_champ = rec["games"]
        # Smooth com prior virtual de 30 jogos a 10% (taxa "tipica")
        smoothed_rate = (games_on_champ + 30 * 0.10) / (total + 30)
        # Centra em 10% e escala. 30% rate -> +1.0, 60% rate -> +2.5
        signal = (smoothed_rate - 0.10) * 5.0
        return max(-1.0, min(3.0, signal))

    def _comfort_kda_score(self, player_id: str, champion: str) -> float:
        """KDA do jogador no campeao acima do normal do campeao, encolhido p/ 0 (em ~desvios)."""
        if not player_id or not champion:
            return 0.0
        rec = self.player_champion_kda.get((player_id, champion))
        if not rec or rec["n"] <= 0:
            return 0.0
        return rec["rel"] / (rec["n"] + COMFORT_KDA_PRIOR) / COMFORT_KDA_SCALE

    def _player_champion_sample_volume(self, player_id: str, champion: str) -> float:
        if not player_id or not champion:
            return 0.0
        rec = self.player_champion.get((player_id, champion))
        games = float((rec or {}).get("games", 0.0) or 0.0)
        return min(1.0, math.log1p(games) / math.log1p(PLAYER_CHAMPION_SAMPLE_LOG_CAP))

    def _team_champion_meta_score(self, team_id: str, champs: List[str]) -> float:
        if not team_id:
            return 0.0
        scores = []
        for champ in champs:
            if not champ:
                continue
            rec = self.team_champion.get((team_id, champ))
            games = float((rec or {}).get("games", 0.0) or 0.0)
            wins = float((rec or {}).get("wins", 0.0) or 0.0)
            if games <= 0:
                scores.append(0.0)
                continue
            scores.append(_smoothed(wins, games, TEAM_CHAMPION_PRIOR, 5.0, 1.5))
        return sum(scores) / max(len(scores), 1) if scores else 0.0

    def _patch_champion_wr_score(self, patch: str, champs: List[str]) -> float:
        if not patch:
            return 0.0
        scores = []
        for champ in champs:
            if not champ:
                continue
            rec = self.patch_champion.get((patch, champ))
            games = float((rec or {}).get("games", 0.0) or 0.0)
            wins = float((rec or {}).get("wins", 0.0) or 0.0)
            if games <= 0:
                scores.append(0.0)
                continue
            scores.append(_smoothed(wins, games, PATCH_CHAMPION_PRIOR, 5.0, 1.25))
        return sum(scores) / max(len(scores), 1) if scores else 0.0

    def _refit_league_ratings(self) -> None:
        """Bradley-Terry em TODOS os jogos entre ligas ja vistos (nunca os futuros)."""
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            return
        jogos = self.cross_league_games
        pais = self.league_parents or {}

        def cadeia(liga: str) -> List[str]:          # liga, mae, avo
            out = [liga]
            while out[-1] in pais and len(out) < 4:
                out.append(pais[out[-1]])
            return out

        # hierarquico: forca(filha) = forca(mae) + degrau + desvio proprio (encolhido p/ 0);
        # filha sem jogo entre ligas ainda ganha "mae + degrau" (entra no conjunto de ligas)
        jogadas = {l for a, b, _ in jogos for l in (a, b)}
        if len(jogadas) < 2:
            return
        ligas = sorted({x for l in jogadas | set(pais) for x in cadeia(l)})
        idx = {l: i for i, l in enumerate(ligas)}
        degrau = len(ligas)
        X = np.zeros((len(jogos), len(ligas) + (1 if pais else 0)))
        y = np.empty(len(jogos), dtype=int)
        for k, (a, b, venceu) in enumerate(jogos):
            for liga, s in ((a, 1.0), (b, -1.0)):
                ch = cadeia(liga)
                for x in ch:
                    X[k, idx[x]] += s
                if pais:
                    X[k, degrau] += s * (len(ch) - 1) * LEAGUE_GAP_SCALE
            y[k] = venceu
        if y.min() == y.max():
            return
        mdl = LogisticRegression(C=LEAGUE_REG_C, max_iter=2000).fit(X, y)
        coef = mdl.coef_[0]
        self.league_rating.clear()
        for l in ligas:
            ch = cadeia(l)
            r = sum(coef[idx[x]] for x in ch)
            if pais:
                r += coef[degrau] * (len(ch) - 1) * LEAGUE_GAP_SCALE
            self.league_rating[l] = float(r)

    def league_elo_offset(self, blue_team: str, red_team: str) -> float:
        """Forca da liga de casa (azul - vermelho) em pontos de Elo; 0 na mesma liga."""
        lb, lr = self.home_league(blue_team), self.home_league(red_team)
        if not lb or not lr or lb == lr:
            return 0.0
        return (self.league_rating.get(lb, 0.0) - self.league_rating.get(lr, 0.0)) * ELO_PER_LOGIT

    def home_league(self, team_id: str) -> str:
        """Liga onde o time mais jogou (fora eventos). '' se ainda nao deu LEAGUE_MIN_GAMES."""
        contagem = self.team_league_games.get(team_id)
        if not contagem:
            return ""
        liga, n = max(contagem.items(), key=lambda kv: kv[1])
        return liga if n >= LEAGUE_MIN_GAMES else ""

    def _league_strength(self, team_id: str) -> float:
        liga = self.home_league(team_id)
        return self.league_rating.get(liga, 0.0) if liga else 0.0

    def _league_champion_wr_score(self, league: str, champs: List[str]) -> float:
        """WR media dos champs nessa liga especifica (Bayesian smoothed).

        Captura diferenca regional: Lucian na LEC pode ter 43% WR,
        mas na LAS 60%. Sem essa feature o modelo so via media global.
        """
        if not league:
            return 0.0
        scores = []
        for champ in champs:
            if not champ:
                continue
            rec = self.league_champion.get((league, champ))
            games = float((rec or {}).get("games", 0.0) or 0.0)
            wins = float((rec or {}).get("wins", 0.0) or 0.0)
            if games <= 0:
                scores.append(0.0)
                continue
            scores.append(_smoothed(wins, games, LEAGUE_CHAMPION_PRIOR, 5.0, 1.25))
        return sum(scores) / max(len(scores), 1) if scores else 0.0

    def _team_early_score(self, team_id: str) -> float:
        rec = self.team_early_profile.get(team_id)
        games = float((rec or {}).get("games", 0.0) or 0.0)
        if games <= 0:
            return 0.0
        gold = (rec.get("gold15", 0.0) / games) / 2500.0
        xp = (rec.get("xp15", 0.0) / games) / 2000.0
        cs = (rec.get("cs15", 0.0) / games) / 45.0
        killdiff = (rec.get("killdiff15", 0.0) / games) / 5.0
        fb = (rec.get("firstblood", 0.0) / games) - 0.5
        return max(-2.0, min(2.0, 0.42 * gold + 0.22 * xp + 0.16 * cs + 0.14 * killdiff + 0.06 * fb))

    def _team_objective_score(self, team_id: str) -> float:
        rec = self.team_objective_profile.get(team_id)
        games = float((rec or {}).get("games", 0.0) or 0.0)
        if games <= 0:
            return 0.0

        def media(stat: str) -> Optional[float]:
            # cada estatistica tem o proprio contador (coluna vazia nao conta);
            # snapshot antigo sem contador usa o total de jogos
            n = float(rec.get("n_" + stat, games) or 0.0)
            return rec.get(stat, 0.0) / n if n > 0 else None

        def primeiro(stat: str) -> float:
            m = media(stat)
            return 0.0 if m is None else m - 0.5

        def dif(stat: str, escala: float) -> float:
            m = media(stat)
            return 0.0 if m is None else m / escala

        fd = primeiro("firstdragon")
        drag = dif("dragondiff", 3.0)
        fh = primeiro("firstherald")
        herald = dif("heralddiff", 2.0)
        grubs = dif("voidgrubdiff", 6.0)
        fb = primeiro("firstbaron")
        baron = dif("barondiff", 1.0)
        ft = primeiro("firsttower")
        tower = dif("towerdiff", 6.0)
        mid = primeiro("firstmidtower")
        three = primeiro("firsttothreetowers")
        plate = dif("platediff", 8.0)
        score = (0.18 * fd + 0.18 * drag + 0.10 * fh + 0.08 * herald + 0.06 * grubs
                 + 0.12 * fb + 0.10 * baron + 0.10 * ft + 0.08 * tower
                 + 0.05 * mid + 0.05 * three + 0.04 * plate)
        return max(-2.0, min(2.0, score))

    def _team_side_score(self, team_id: str, side: str) -> float:
        if not team_id:
            return 0.0
        rec = self.team_side.get((team_id, side))
        games = float((rec or {}).get("games", 0.0) or 0.0)
        wins = float((rec or {}).get("wins", 0.0) or 0.0)
        return _smoothed(wins, games, TEAM_SIDE_PRIOR, 5.0, 1.5)

    def _team_series_score(self, team_id: str, game_no: int) -> float:
        if not team_id:
            return 0.0
        rec = self.team_series.get((team_id, game_no))
        games = float((rec or {}).get("games", 0.0) or 0.0)
        wins = float((rec or {}).get("wins", 0.0) or 0.0)
        return _smoothed(wins, games, TEAM_SERIES_PRIOR, 5.0, 1.5)

    def _team_meta_score(self, champs: List[str]) -> float:
        valid = [c for c in champs if c]
        if not valid:
            return 0.0
        return sum(self._champion_meta_score(c) for c in valid) / len(valid)

    def _champion_meta_score(self, champ: str) -> float:
        """Forca recente do champion no pro play.

        Mede APENAS WR (forca real), nao popularidade. Versao anterior misturava
        pick_score (popularidade, sempre positivo) com wr_signal reduzido a 0.30:
        o modelo aprendia coef negativo porque "champ popular = time fraco
        forcando seguro" (colinearidade, nao causal).

        Formula nova: volume * wr_signal, range [-1, +1] simetrico.
        - volume = quanto confiavel e a amostra (log(games) ate 40)
        - wr_signal = quao acima/abaixo de 50% o champ vence
        """
        if not champ:
            return 0.0
        rec = self.champion_recent_meta.get(champ)
        games = float((rec or {}).get("games", 0.0) or 0.0)
        if games <= 0:
            return 0.0
        wins = float((rec or {}).get("wins", 0.0) or 0.0)
        wr = (wins + CHAMPION_META_WR_PRIOR * 0.5) / (games + CHAMPION_META_WR_PRIOR)
        volume = min(1.0, math.log1p(games) / math.log1p(40.0))
        wr_signal = (wr - 0.5) * 2.0
        return volume * wr_signal

    def _decay_champion_meta(self) -> None:
        if not self.champion_recent_meta:
            return
        stale = []
        for champ, rec in self.champion_recent_meta.items():
            rec["games"] *= CHAMPION_META_DECAY
            rec["wins"] *= CHAMPION_META_DECAY
            if rec["games"] < 0.01:
                stale.append(champ)
        for champ in stale:
            self.champion_recent_meta.pop(champ, None)

    def _ban_value(self, champ: str) -> float:
        """Valor heuristico do ban - log da pickrate."""
        if not champ:
            return 0.0
        n = self.champion_pickrate.get(champ, 0)
        if n <= 0:
            return 0.0
        # log normalizado: champ raro -> ~0, champ super pickado -> ~3
        return math.log1p(n / 50.0)

    def _player_form_wr(self, player_id: str) -> float:
        if not player_id:
            return PLAYER_FORM_INITIAL
        dq = self.player_form.get(player_id)
        if not dq or len(dq) == 0:
            return PLAYER_FORM_INITIAL
        return sum(dq) / len(dq)

    def _team_form_wr(self, team_id: str) -> float:
        if not team_id:
            return 0.5
        dq = self.team_form.get(team_id)
        if not dq or len(dq) == 0:
            return 0.5
        return sum(dq) / len(dq)

    def _h2h_score(self, team_a: str, team_b: str) -> float:
        """WR de A vs B (smoothed). Sinal positivo = A favorecido."""
        if not team_a or not team_b:
            return 0.0
        a, b, swapped = _h2h_key(team_a, team_b)
        key = (a, b)
        rec = self.h2h.get(key)
        if not rec or rec["games"] <= 0:
            return 0.0
        wins_a = rec["wins_a"] if not swapped else (rec["games"] - rec["wins_a"])
        return _smoothed(
            wins_a, rec["games"],
            SMOOTHING_PRIOR_H2H, 10.0, 2.0,
        )

    # ---------- state update (chamar APOS build_features) ----------

    def update(self, game: Dict[str, Any]) -> None:
        """Incorpora o resultado do jogo no state para os proximos."""
        blue_champs = [_norm(game[f"blue_{r}"]) for r in _ROLES]
        red_champs  = [_norm(game[f"red_{r}"])  for r in _ROLES]
        blue_won = int(game["blue_won"])
        blue_team = _norm(game.get("blue_teamid"))
        red_team  = _norm(game.get("red_teamid"))
        blue_players = [_norm(game.get(f"blue_player_{r}")) for r in _ROLES]
        red_players  = [_norm(game.get(f"red_player_{r}"))  for r in _ROLES]
        patch = oe_patch_key(game.get("patch"))  # live "16.9"/"16.10" -> chave do Oracle
        game_no = _game_number(game)

        self._decay_champion_meta()

        # 1. Atualiza champion_matchups (com e sem role)
        for r, bc, rc in zip(_ROLES, blue_champs, red_champs):
            if not bc or not rc:
                continue
            for role_key in (r, ""):
                rec = self.champion_matchups[(bc, rc, role_key)]
                rec["games"] += 1
                rec["wins"] += blue_won
                rec_rev = self.champion_matchups[(rc, bc, role_key)]
                rec_rev["games"] += 1
                rec_rev["wins"] += (1 - blue_won)

        # 1b. Atualiza matchups cross-role (todos vs todos) sem role
        for bc in blue_champs:
            for rc in red_champs:
                if not bc or not rc:
                    continue
                # ja contado se for same-role; este e o agregado overall
                # so adiciona se for cross-role para evitar double-count com same-role
                # Atualizamos sem role
                pass  # Mantemos so same-role para nao duplicar

        # 2. Synergies (pares mesmo time)
        for champs_team, won in [(blue_champs, blue_won), (red_champs, 1 - blue_won)]:
            for a, b in combinations(champs_team, 2):
                if not a or not b:
                    continue
                key = _synergy_key(a, b)
                rec = self.synergies[key]
                rec["games"] += 1
                rec["wins"] += won

        # 3. Pickrate (para ban_score)
        for c in blue_champs + red_champs:
            if c:
                self.champion_pickrate[c] += 1

        for c, won in ([(c, blue_won) for c in blue_champs] +
                       [(c, 1 - blue_won) for c in red_champs]):
            if c:
                rec = self.champion_recent_meta[c]
                rec["games"] += 1.0
                rec["wins"] += float(won)

        # 3b. Player x champion (para player_champion_diff)
        for p, c in zip(blue_players, blue_champs):
            if p and c:
                rec = self.player_champion[(p, c)]
                rec["games"] += 1
                rec["wins"] += blue_won
        for p, c in zip(red_players, red_champs):
            if p and c:
                rec = self.player_champion[(p, c)]
                rec["games"] += 1
                rec["wins"] += (1 - blue_won)

        # 3c. Total de jogos do player (para player_champion_rate_diff).
        # Liga-neutra: mede % do tempo que o player joga aquele champ.
        for p in blue_players + red_players:
            if p:
                self.player_total_games[p] += 1

        # 3d. KDA do jogador no campeao relativo ao normal do campeao naquela funcao
        #     (a referencia e a media ANTES deste jogo; jogo sem K/D/A e pulado)
        for side, players, champs in (("blue", blue_players, blue_champs),
                                      ("red", red_players, red_champs)):
            for r, p, c in zip(_ROLES, players, champs):
                if not p or not c:
                    continue
                k, d, a = (_num(game.get(f"{side}_{s}_{r}")) for s in ("kills", "deaths", "assists"))
                if k is None and d is None and a is None:
                    continue
                kda = ((k or 0.0) + (a or 0.0)) / ((d or 0.0) + 1.0)
                ck = self.champion_role_kda[(c, r)]
                base = ck["sum"] / ck["n"] if ck["n"] else kda
                rec = self.player_champion_kda[(p, c)]
                rec["n"] += 1.0
                rec["rel"] += kda - base
                ck["n"] += 1.0
                ck["sum"] += kda

        if blue_team:
            for c in blue_champs:
                if c:
                    rec = self.team_champion[(blue_team, c)]
                    rec["games"] += 1.0
                    rec["wins"] += float(blue_won)
        if red_team:
            for c in red_champs:
                if c:
                    rec = self.team_champion[(red_team, c)]
                    rec["games"] += 1.0
                    rec["wins"] += float(1 - blue_won)
        if patch:
            for c in blue_champs:
                if c:
                    rec = self.patch_champion[(patch, c)]
                    rec["games"] += 1.0
                    rec["wins"] += float(blue_won)
            for c in red_champs:
                if c:
                    rec = self.patch_champion[(patch, c)]
                    rec["games"] += 1.0
                    rec["wins"] += float(1 - blue_won)

        # 3a2. de que liga e cada time (p/ o rating de liga). Evento nao conta.
        liga_atual = _norm(game.get("league") or game.get("league_name"))
        if self.league_aliases:
            liga_atual = self.league_aliases.get(liga_atual, liga_atual)
        if liga_atual and liga_atual not in EVENT_LEAGUES:
            for t in (blue_team, red_team):
                if t:
                    cont = self.team_league_games[t]
                    if self.home_decay:           # time promovido/rebaixado muda de casa
                        for k in cont:
                            cont[k] *= self.home_decay
                    cont[liga_atual] += 1.0

        # 3b. league_champion: WR de cada champ na liga atual
        league = _norm(game.get("league") or game.get("league_name"))
        if league:
            for c in blue_champs:
                if c:
                    rec = self.league_champion[(league, c)]
                    rec["games"] += 1.0
                    rec["wins"] += float(blue_won)
            for c in red_champs:
                if c:
                    rec = self.league_champion[(league, c)]
                    rec["games"] += 1.0
                    rec["wins"] += float(1 - blue_won)

        # 4. Elo
        if blue_team and red_team:
            blue_r = self.team_rating[blue_team]
            red_r  = self.team_rating[red_team]
            # entre ligas: perder p/ time de liga mais forte e o esperado, nao derruba o Elo
            off = self.league_elo_offset(blue_team, red_team) if self.league_elo else 0.0
            expected_blue = 1.0 / (1.0 + 10 ** ((red_r - blue_r - off) / 400))
            delta = ELO_K_FACTOR * (blue_won - expected_blue)
            self.team_rating[blue_team] = blue_r + delta
            self.team_rating[red_team]  = red_r - delta

            # 4b. Forca da LIGA: guarda o jogo entre ligas e re-ajusta TODAS juntas de
            # tempos em tempos (Bradley-Terry). Passo incremental tipo Elo nao serve:
            # com ~10 jogos diretos por par a escala nao converge (a LFL chegou a ficar
            # acima da LCK). O ajuste conjunto usa a rede inteira de confrontos.
            lb, lr = self.home_league(blue_team), self.home_league(red_team)
            if lb and lr and lb != lr:
                self.cross_league_games.append((lb, lr, int(blue_won)))
                n = len(self.cross_league_games)
                if n >= LEAGUE_MIN_CROSS and n % LEAGUE_REFIT_EVERY == 0:
                    self._refit_league_ratings()

            # 5. Team form
            self.team_form[blue_team].append(blue_won)
            self.team_form[red_team].append(1 - blue_won)

            # 6. H2H
            a, b, swapped = _h2h_key(blue_team, red_team)
            rec = self.h2h[(a, b)]
            rec["games"] += 1
            # wins_a = wins do time A (o "primeiro" alfabeticamente)
            if not swapped:
                # blue == A
                rec["wins_a"] += blue_won
            else:
                # red == A
                rec["wins_a"] += (1 - blue_won)

            side_rec = self.team_side[(blue_team, "blue")]
            side_rec["games"] += 1.0
            side_rec["wins"] += float(blue_won)
            side_rec = self.team_side[(red_team, "red")]
            side_rec["games"] += 1.0
            side_rec["wins"] += float(1 - blue_won)

            series_rec = self.team_series[(blue_team, game_no)]
            series_rec["games"] += 1.0
            series_rec["wins"] += float(blue_won)
            series_rec = self.team_series[(red_team, game_no)]
            series_rec["games"] += 1.0
            series_rec["wins"] += float(1 - blue_won)

            self._update_team_profiles(game, "blue", blue_team)
            self._update_team_profiles(game, "red", red_team)

        # 7. Player form
        for p in blue_players:
            if p:
                self.player_form[p].append(blue_won)
        for p in red_players:
            if p:
                self.player_form[p].append(1 - blue_won)

        self.n_games_seen += 1

    def _update_team_profiles(self, game: Dict[str, Any], side: str, team_id: str) -> None:
        if not team_id:
            return
        # early game so conta jogo que TEM os numeros de 15 min (partial nao tem)
        gold15 = _num(game.get(f"{side}_golddiffat15"))
        if gold15 is not None:
            early = self.team_early_profile[team_id]
            early["games"] += 1.0
            early["gold15"] += gold15
            early["xp15"] += _to_float(game.get(f"{side}_xpdiffat15"))
            early["cs15"] += _to_float(game.get(f"{side}_csdiffat15"))
            early["killdiff15"] += (
                _to_float(game.get(f"{side}_killsat15"))
                - _to_float(game.get(f"{side}_deathsat15"))
            )
            early["firstblood"] += _to_float(game.get(f"{side}_firstblood"))

        obj = self.team_objective_profile[team_id]
        obj["games"] += 1.0
        for stat, (col, opp_col) in _OBJECTIVE_STATS.items():
            v = _num(game.get(f"{side}_{col}"))
            if v is None:
                continue
            if opp_col:
                o = _num(game.get(f"{side}_{opp_col}"))
                if o is None:
                    continue
                v -= o
            obj[stat] = obj.get(stat, 0.0) + v
            obj["n_" + stat] = obj.get("n_" + stat, 0.0) + 1.0

    # ---------- diagnostics ----------

    def stats(self) -> Dict[str, int]:
        return {
            "games_seen":     self.n_games_seen,
            "matchup_pairs":  len(self.champion_matchups),
            "synergy_pairs":  len(self.synergies),
            "golgg_matchup_fallback_pairs": len(self.prior_champion_matchups),
            "golgg_synergy_fallback_pairs": len(self.prior_synergies),
            "teams_tracked":  len(self.team_rating),
            "players_tracked": len(self.player_form),
            "h2h_pairs":      len(self.h2h),
            "champs_picked":  len(self.champion_pickrate),
            "champs_recent_meta": len(self.champion_recent_meta),
            "team_early_profiles": len(self.team_early_profile),
            "team_objective_profiles": len(self.team_objective_profile),
            "team_side_profiles": len(self.team_side),
            "team_series_profiles": len(self.team_series),
            "team_champion_pairs": len(self.team_champion),
            "patch_champion_pairs": len(self.patch_champion),
            "league_champion_pairs": len(self.league_champion),
            "leagues_rated": len(self.league_rating),
            "cross_league_games": len(self.cross_league_games),
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_pipeline.oracle_loader import load_year

    print("Carregando 2024 + 2025...")
    df_2024 = load_year(2024)
    df_2025 = load_year(2025)
    import pandas as pd
    df = pd.concat([df_2024, df_2025]).sort_values("date").reset_index(drop=True)
    print(f"  total: {len(df)} jogos")

    ctx = TemporalContext()
    nonzero_counts = {"counter_diff": 0, "synergy_diff": 0, "h2h_diff": 0,
                      "rating_diff": 0, "lane_matchup_diff": 0}
    total = 0
    for _, game in df.iterrows():
        feats = ctx.build_features(game.to_dict())
        for k in nonzero_counts:
            if abs(feats[k]) > 1e-9:
                nonzero_counts[k] += 1
        ctx.update(game.to_dict())
        total += 1

    print(f"\nState final: {ctx.stats()}")
    print(f"\nFeatures != 0 em {total} jogos:")
    for k, c in nonzero_counts.items():
        pct = 100 * c / total
        flag = " <- ARMADILHA" if pct < 50 else ""
        print(f"  {k:<22} {c:>6} ({pct:.1f}%){flag}")










