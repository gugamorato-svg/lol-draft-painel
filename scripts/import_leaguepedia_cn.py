"""Traz da Leaguepedia os jogos das bases chinesas que o Oracle's Elixir NAO tem.

Por que (24/09/2026): o Oracle tem o LDL so ate jun/2024. Os times de base chineses (BLG
Junior, EDG Youth, WBG Youth, TES Challenger, LGD Young) apareciam no modelo so com dado de
2024 + eventos internacionais -> jogadores "sem experiencia" no campeao, perfil velho. O
modelo deu 21% p/ EDG Youth contra FUEGO e a EDG Youth venceu.

O que vem (Leaguepedia, tabelas ScoreboardGames/ScoreboardPlayers via Special:CargoExport,
que nao cai no limite da api.php):
  LDL 2025 (Split 1, Split 1 Cup, Split 2 Round 1/2, Split 3, Split 3 Cup)   ~900 jogos
  2026 Asia Masters Elite Invitational 1 e 2 (o circuito das bases chinesas) ~176 jogos
  WSCI 2026 China Qualifier                                                   24 jogos
Tudo entra como liga "LDL" (e o circuito domestico das bases chinesas). Jogo que o Oracle
ja tem (mesmo dia e mesmos times) fica de fora.

Formato: o MESMO CSV "longo" do Oracle (12 linhas por jogo: 10 jogadores + 2 times),
datacompleteness="partial" (sem ouro/xp aos 15 -- o TemporalContext pula o que falta).
Time e jogador que o Oracle conhece ganham o id do Oracle; os outros, "lp:team:..." e
"lp:player:...". Patch = o do Oracle naquela data (a Leaguepedia escreve 26.1 p/ 16.10).

Saida: data/oracle_elixir/extra/<ano>_leaguepedia_cn.csv (o oracle_loader junta sozinho)

Uso: python scripts/import_leaguepedia_cn.py
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

ORACLE = ROOT / "data" / "oracle_elixir"
EXTRA = ORACLE / "extra"
EXPORT = "https://lol.fandom.com/wiki/Special:CargoExport"
# Circuitos (campo League da wiki) que viram a liga "LDL" do modelo. Os torneios sao
# descobertos a cada rodada: edicao nova (ex.: Asia Masters Elite 3) entra sozinha.
CIRCUITOS = ("LoL Development League", "Asia Masters Elite Invitational Tournament",
             "WSCI-LPL Regional Qualifier")
DESDE = "2025-01-01"          # antes disso o Oracle ja tem o LDL (ate jun/2024)
ROLE = {"Top": "top", "Jungle": "jng", "Mid": "mid", "Bot": "bot", "Support": "sup"}
PID = {"top": 1, "jng": 2, "mid": 3, "bot": 4, "sup": 5}
# nome da Leaguepedia -> nome do Oracle (quando nem o nome normalizado bate)
TEAM_ALIASES = {"edwardgamingyouthteam": "EDG Youth Team"}
# jogador so herda o id do Oracle se ja jogou competicao chinesa la (auditoria 24/09: 9 de
# 108 casavam com homonimo de outra regiao -- "Relax" da WE Academy virava um jogador da LRN)
LIGAS_CN = {"LDL", "LPL", "DCup", "Asia Master", "AM", "WSCI"}


def norm(s: Any) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).replace("&amp;", "&")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def export(**kw) -> List[Dict[str, Any]]:
    kw.setdefault("format", "json")
    kw.setdefault("limit", "5000")
    for espera in (0, 20, 60, 180):
        time.sleep(espera or 2)
        r = requests.get(EXPORT, params=kw, timeout=90,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        if r.status_code == 200:
            return r.json()
    r.raise_for_status()
    return []


def torneios() -> Dict[str, str]:
    """Nome do torneio -> liga do modelo, descoberto na wiki."""
    ligas = " OR ".join(f"League='{c}'" for c in CIRCUITOS)
    t = export(tables="Tournaments", fields="Name,DateStart,League",
               where=f"DateStart >= '{DESDE}' AND ({ligas})", order_by="DateStart")
    return {x["Name"].replace("&amp;", "&"): "LDL" for x in t}


def baixar(nomes_t: Dict[str, str]) -> tuple[list, list]:
    jogos, linhas = [], []
    for t in nomes_t:
        g = export(tables="ScoreboardGames", where=f"Tournament='{t}'",
                   fields="GameId,MatchId,Tournament,Team1,Team2,Winner,DateTime_UTC,Team1Picks,"
                          "Team2Picks,Team1Bans,Team2Bans,Gamelength_Number,Team1Dragons,"
                          "Team2Dragons,Team1Barons,Team2Barons,Team1Towers,Team2Towers,"
                          "N_GameInMatch")
        # ScoreboardPlayers nao tem Tournament: junta com ScoreboardGames pelo GameId
        p = export(tables="ScoreboardPlayers=SP,ScoreboardGames=SG", join_on="SP.GameId=SG.GameId",
                   where=f"SG.Tournament='{t}'",
                   fields="SP.GameId=GameId,SP.Name=Name,SP.Link=Link,SP.Team=Team,SP.Champion=Champion,"
                          "SP.Kills=Kills,SP.Deaths=Deaths,SP.Assists=Assists,SP.Role=Role,SP.Side=Side")
        print(f"    {t}: {len(g)} jogos, {len(p)} linhas de jogador")
        jogos += g
        linhas += p
    return jogos, linhas


class Nomes:
    """Nome da Leaguepedia -> id do Oracle (time e jogador), com o que o Oracle ja tem."""

    def __init__(self, oracle: pd.DataFrame) -> None:
        t = oracle[oracle["participantid"].isin([100, 200])]
        self.time, self.nome_time = {}, {}
        for nome, tid in t.sort_values("date")[["teamname", "teamid"]].dropna().itertuples(index=False):
            self.time[norm(nome)] = tid                      # o mais recente vence
            self.nome_time[tid] = nome                       # nome do Oracle p/ esse id
        p = oracle[oracle["participantid"].between(1, 10)].dropna(subset=["playerid"])
        cn = set(p.loc[p["league"].isin(LIGAS_CN), "playerid"])
        p = p[p["playerid"].isin(cn)]
        self.jog: Dict[str, Dict[str, tuple]] = defaultdict(dict)
        self.nome_jog = dict(zip(p["playerid"], p["playername"]))
        for nome, pid, time_, data in p[["playername", "playerid", "teamname", "date"]].itertuples(index=False):
            ant = self.jog[norm(nome)].get(pid)
            if ant is None or str(data) > ant[1]:
                self.jog[norm(nome)][pid] = (norm(time_), str(data))
        self.stats = defaultdict(int)

    def team_id(self, nome: str) -> str:
        k = norm(nome)
        k = norm(TEAM_ALIASES.get(k, nome))
        if k in self.time:
            self.stats["time_oracle"] += 1
            return self.time[k]
        self.stats["time_novo"] += 1
        return f"lp:team:{norm(nome)}"

    def player_id(self, nome: str, time_: str, link: str = "") -> str:
        cands = self.jog.get(norm(nome), {})
        if not cands:
            self.stats["jogador_novo"] += 1
            return f"lp:player:{norm(link or nome)}"   # Link = pagina da wiki (unica)
        self.stats["jogador_oracle"] += 1
        if len(cands) == 1:
            return next(iter(cands))
        # mesmo nome, ids diferentes: o que jogou no mesmo time; senao o mais recente
        mesmo = [pid for pid, (t, _) in cands.items() if t and (t in norm(time_) or norm(time_) in t)]
        if len(mesmo) == 1:
            return mesmo[0]
        return max(cands.items(), key=lambda kv: kv[1][1])[0]


def patch_por_data(oracle: pd.DataFrame) -> pd.Series:
    """Data -> patch mais usado no Oracle naquele dia (formato do Oracle)."""
    t = oracle[oracle["participantid"] == 100][["date", "patch"]].dropna()
    t = t.assign(dia=pd.to_datetime(t["date"]).dt.normalize())
    return t.groupby("dia")["patch"].agg(lambda s: s.mode().iloc[0]).sort_index()


def montar(jogos, linhas, nomes: Nomes, patches: pd.Series, ligas: Dict[str, str]) -> pd.DataFrame:
    por_jogo = defaultdict(list)
    for x in linhas:
        por_jogo[x["GameId"]].append(x)
    rows, pulados = [], defaultdict(int)
    for g in jogos:
        ps = por_jogo.get(g["GameId"], [])
        if len(ps) != 10 or g.get("Winner") not in (1, 2, "1", "2"):
            pulados["sem 10 jogadores/vencedor"] += 1
            continue
        data = pd.Timestamp(g["DateTime UTC"])
        i = patches.index.searchsorted(data.normalize(), side="right") - 1
        patch = patches.iloc[max(i, 0)]
        azul_venceu = int(g["Winner"]) == 1
        gid = "LP_" + hashlib.md5(g["GameId"].encode("utf-8")).hexdigest()[:16]   # estavel
        base = {"gameid": gid, "datacompleteness": "partial", "url": "leaguepedia",
                "league": ligas[g["Tournament"]], "year": data.year, "split": g["Tournament"],
                "playoffs": 0, "date": data.strftime("%Y-%m-%d %H:%M:%S"),
                "game": int(g.get("N GameInMatch") or 1), "patch": patch,
                "gamelength": round(float(g.get("Gamelength Number") or 0) * 60)}
        times = {}
        for lado, n, venceu in (("Blue", 1, azul_venceu), ("Red", 2, not azul_venceu)):
            nome = str(g[f"Team{n}"]).replace("&amp;", "&")
            tid = nomes.team_id(nome)
            nome = nomes.nome_time.get(tid, nome)   # mesmo nome do Oracle (o app filtra por nome)
            times[n] = (nome, tid)
            o = 2 if n == 1 else 1
            picks = list(g.get(f"Team{n}Picks") or []) + [None] * 5
            bans = list(g.get(f"Team{n}Bans") or []) + [None] * 5
            row = {**base, "participantid": 100 if n == 1 else 200, "side": lado,
                   "position": "team", "teamname": nome, "teamid": tid, "result": int(venceu),
                   "firstPick": int(n == 1),
                   "dragons": g.get(f"Team{n}Dragons"), "opp_dragons": g.get(f"Team{o}Dragons"),
                   "barons": g.get(f"Team{n}Barons"), "opp_barons": g.get(f"Team{o}Barons"),
                   "towers": g.get(f"Team{n}Towers"), "opp_towers": g.get(f"Team{o}Towers")}
            for k in range(5):
                row[f"pick{k + 1}"], row[f"ban{k + 1}"] = picks[k], bans[k]
            rows.append(row)
        for p in ps:
            papel = ROLE.get(p.get("Role"))
            n = int(p.get("Side") or 0)
            if not papel or n not in (1, 2):
                continue
            nome_t, tid = times[n]
            pid = nomes.player_id(p["Name"], nome_t, p.get("Link", ""))
            rows.append({**base, "participantid": PID[papel] + (0 if n == 1 else 5),
                         "side": "Blue" if n == 1 else "Red", "position": papel,
                         "playername": nomes.nome_jog.get(pid, p["Name"]), "playerid": pid,
                         "teamname": nome_t, "teamid": tid, "champion": p["Champion"],
                         "result": int(azul_venceu if n == 1 else not azul_venceu),
                         "kills": p.get("Kills"), "deaths": p.get("Deaths"), "assists": p.get("Assists")})
    if pulados:
        print(f"[~] pulados: {dict(pulados)}")
    return pd.DataFrame(rows)


def ja_no_oracle(novo: pd.DataFrame, oracle: pd.DataFrame) -> set:
    """gameids novos cujo (dia, par de times) o Oracle ja tem -> duplicado, fica de fora."""
    def chaves(df):
        t = df[df["participantid"].isin([100, 200])]
        dia = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
        par = t.assign(dia=dia, k=t["teamname"].map(norm)).groupby("gameid").agg(
            dia=("dia", "first"), times=("k", lambda s: "|".join(sorted(s))))
        return par
    o, n = chaves(oracle), chaves(novo)
    ok = set(zip(o["dia"], o["times"]))
    return {gid for gid, r in n.iterrows() if (r["dia"], r["times"]) in ok}


def main() -> int:
    print("[*] Oracle (nomes, ids, patches)...")
    cols = ["gameid", "date", "league", "participantid", "playername", "playerid", "teamname",
            "teamid", "patch"]
    oracle = pd.concat([pd.read_csv(p, usecols=cols, low_memory=False)
                        for p in sorted(ORACLE.glob("*_lol.csv"))], ignore_index=True)
    nomes, patches = Nomes(oracle), patch_por_data(oracle)
    print("[*] Leaguepedia (CargoExport)...")
    ligas = torneios()
    print(f"    {len(ligas)} torneios: {', '.join(ligas)}")
    jogos, linhas = baixar(ligas)
    df = montar(jogos, linhas, nomes, patches, ligas)
    dup = ja_no_oracle(df, oracle)
    df = df[~df["gameid"].isin(dup)]
    n_jogos = df["gameid"].nunique()
    print(f"[*] {n_jogos} jogos novos ({len(dup)} ja estavam no Oracle) | ids: {dict(nomes.stats)}")
    EXTRA.mkdir(parents=True, exist_ok=True)
    for ano, parte in df.groupby("year"):
        out = EXTRA / f"{ano}_leaguepedia_cn.csv"
        if out.exists():   # queda da wiki no meio nao pode trocar arquivo bom por um menor
            antes = pd.read_csv(out, usecols=["gameid"])["gameid"].nunique()
            if parte["gameid"].nunique() < antes:
                print(f"[!] {out.name}: {parte['gameid'].nunique()} jogos < {antes} atuais, mantido")
                continue
        parte.to_csv(out, index=False)
        print(f"[OK] {out.relative_to(ROOT)}: {parte['gameid'].nunique()} jogos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
