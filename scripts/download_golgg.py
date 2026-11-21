"""Baixa dados de champion matchups e synergies do gol.gg premium.

Usa as APIs JSON internas do gol.gg:
  POST /premium/directmatchup.php   -> direct matchups
  POST /premium/indirectmatchup.php -> indirect matchups
  POST /premium/goodwith.php        -> good with (synergies)

Autentica via CDP (Chrome DevTools Protocol) para suportar v20/App-Bound Encryption.

Uso basico:
    python scripts/download_golgg.py

Opcoes:
    --season S16         temporada (padrao S16)
    --split  Spring      split (padrao Spring)
    --browser brave      brave | chrome | edge (padrao brave)
    --limit  10          limita a N campeoes (util para testar)
    --delay  0.5         segundos entre requests (padrao 0.5)
    --resume             retoma de onde parou
    --cookie "name=val"  cookie manual (--cookie "rememberme=xxx")

Saida:
    data/champion_matchups.csv   -- Direct + Indirect matchups
    data/champion_synergies.csv  -- Good with (synergias)
    data/champion_ids.csv        -- mapa nome -> id interno gol.gg
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_URL = "https://gol.gg"
DATA_DIR = Path(__file__).parent.parent / "data"


# ---------- auth via CDP ----------

def make_session(browser: str, cookie_string: str = ""):
    """Cria requests.Session autenticada para gol.gg."""
    import requests

    sess = requests.Session()
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    })

    if cookie_string:
        for part in cookie_string.split(";"):
            part = part.strip()
            if "=" in part:
                name, _, value = part.partition("=")
                sess.cookies.set(name.strip(), value.strip(), domain="gol.gg")
        print("[OK] Cookies carregados via --cookie")
        return sess

    print(f"[*] Extraindo cookies via CDP ({browser})...")
    cdp_cookies = _get_cookies_via_cdp(browser)
    if cdp_cookies:
        for name, value in cdp_cookies.items():
            sess.cookies.set(name, value, domain="gol.gg")
        print(f"[OK] {len(cdp_cookies)} cookies gol.gg via CDP")
        return sess

    _print_manual_cookie_instructions()
    return sess


def _get_cookies_via_cdp(browser: str) -> dict:
    """Lanca o browser em modo debug e extrai cookies gol.gg."""
    import requests

    home = Path(os.environ.get("USERPROFILE", Path.home()))
    exe_candidates = {
        "brave": [
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
            str(home / "AppData/Local/BraveSoftware/Brave-Browser/Application/brave.exe"),
        ],
        "chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
        "edge": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ],
    }
    user_data_dirs = {
        "brave":  str(home / "AppData/Local/BraveSoftware/Brave-Browser/User Data"),
        "chrome": str(home / "AppData/Local/Google/Chrome/User Data"),
        "edge":   str(home / "AppData/Local/Microsoft/Edge/User Data"),
    }

    exe = next(
        (p for p in exe_candidates.get(browser, []) if Path(p).exists()), None
    )
    if not exe:
        print(f"  [!] Executavel do {browser} nao encontrado")
        return {}

    user_data = user_data_dirs.get(browser, "")
    debug_port = 9222

    already_running = False
    with socket.socket() as s:
        already_running = s.connect_ex(("127.0.0.1", debug_port)) == 0

    proc = None
    if not already_running:
        print(f"  [*] Iniciando {browser} com porta de debug {debug_port}...")
        proc = subprocess.Popen(
            [
                exe,
                f"--remote-debugging-port={debug_port}",
                f"--remote-allow-origins=http://localhost:{debug_port}",
                f"--user-data-dir={user_data}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-extensions",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started = False
        for _ in range(20):
            time.sleep(0.5)
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", debug_port)) == 0:
                    started = True
                    break
        if not started:
            print("  [!] Browser nao respondeu na porta de debug")
            if proc:
                proc.terminate()
            return {}
        print(f"  [OK] {browser} iniciado (modo debug)")
        time.sleep(1.0)

    try:
        r = requests.get(f"http://localhost:{debug_port}/json/list", timeout=5)
        targets = r.json()
        target = next((t for t in targets if t.get("type") == "page"), None)
        if not target:
            r2 = requests.put(
                f"http://localhost:{debug_port}/json/new?about:blank", timeout=5
            )
            target = r2.json()

        ws_url = target.get("webSocketDebuggerUrl", "")
        if not ws_url:
            print("  [!] Nao obteve WebSocket URL do CDP")
            return {}

        try:
            import websocket as _ws  # noqa: F401
        except ImportError:
            print("  [*] Instalando websocket-client...")
            subprocess.run(
                ["pip", "install", "websocket-client", "-q"],
                check=True,
                stdout=subprocess.DEVNULL,
            )

        import websocket as ws_lib

        ws_conn = ws_lib.create_connection(ws_url, timeout=10)
        ws_conn.send(json.dumps({
            "id": 1,
            "method": "Network.getAllCookies",
            "params": {},
        }))
        resp = json.loads(ws_conn.recv())
        ws_conn.close()

        all_cookies = resp.get("result", {}).get("cookies", [])
        golgg_cookies = {
            c["name"]: c["value"]
            for c in all_cookies
            if "gol.gg" in c.get("domain", "")
        }
        if golgg_cookies:
            print(f"  [OK] Cookies encontrados: {list(golgg_cookies.keys())}")
        else:
            print("  [!] Nenhum cookie gol.gg -- voce esta logado no gol.gg no Brave?")
        return golgg_cookies

    except Exception as exc:
        print(f"  [!] CDP falhou: {exc}")
        return {}
    finally:
        if proc is not None:
            print(f"  [*] Encerrando {browser}...")
            proc.terminate()


def _print_manual_cookie_instructions() -> None:
    print()
    print("-" * 55)
    print("Como obter o cookie manualmente:")
    print("  1. Abra o Brave e acesse gol.gg logado")
    print("  2. Pressione F12 -> Application -> Cookies -> gol.gg")
    print("  3. Copie o valor de 'rememberme'")
    print("  4. Rode: --cookie \"rememberme=VALOR\"")
    print("-" * 55)
    print()


def check_logged_in(sess) -> bool:
    """Verifica se a sessao tem acesso premium."""
    import requests
    url = f"{BASE_URL}/premium/esports/"
    try:
        r = sess.get(url, timeout=10, allow_redirects=True)
        if r.status_code != 200 or "login" in r.url:
            print("[ERR] Nao autenticado (redirecionado para login).")
            _print_manual_cookie_instructions()
            return False
        body = r.text.lower()
        if 'name="password"' in body or 'name="email"' in body:
            print("[ERR] Pagina retornou formulario de login.")
            _print_manual_cookie_instructions()
            return False
        return True
    except requests.RequestException as exc:
        print(f"[!] Erro ao verificar login: {exc}")
        return False


# ---------- champion list ----------

def get_champion_ids(sess, season: str, split: str) -> List[Dict]:
    """Extrai mapa {id, name} da lista de campeoes via selectize JS."""
    url = f"{BASE_URL}/champion/list/season-{season}/split-{split}/tournament-ALL/"
    r = sess.get(url, timeout=15)
    r.raise_for_status()

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, "lxml")

    champions: Dict[int, str] = {}
    for script in soup.find_all("script"):
        text = script.string or ""
        if "class:'champion'" not in text:
            continue
        matches = re.findall(
            r"class:\s*'champion'\s*,\s*value:\s*'c_(\d+)'\s*,\s*name:\s*'([^']+)'",
            text,
        )
        for cid_str, name in matches:
            cid = int(cid_str)
            if name and cid not in champions:
                champions[cid] = name

    result = [{"id": k, "name": v} for k, v in sorted(champions.items())]
    print(f"[OK] {len(result)} campeoes encontrados")
    return result


# ---------- JSON API calls ----------

def _fetch_matchup_json(
    sess,
    endpoint: str,
    champ_id: int,
    season: str,
    split: str,
) -> Optional[List[Dict]]:
    """Chama um dos 3 endpoints PHP e retorna lista de dicts ou None."""
    url = f"{BASE_URL}/premium/{endpoint}.php"
    params = {
        "id": champ_id,
        "tr": "%",
        "season": season,
        "split": split,
        "patch": "%",
        "role": "%",
        "pickfilter": "ALL",
    }
    headers = {
        "X-Requested-With": "XMLHttpRequest",
        "Referer": (
            f"{BASE_URL}/premium/champion-synergy/{champ_id}/"
            f"season-{season}/split-{split}/tournament-ALL/"
        ),
    }
    try:
        r = sess.post(url, data=params, headers=headers, timeout=20)
    except Exception as exc:
        print(f"  [!] rede: {exc}")
        return None

    if r.status_code == 302 or "login" in r.url:
        print("  [ERR] Redirecionado para login -- sessao expirou")
        return None
    if r.status_code != 200:
        print(f"  [!] HTTP {r.status_code}")
        return None

    text = r.text.strip()
    if not text or text == "[]":
        return []
    try:
        return r.json()
    except Exception:
        return None


def _json_to_rows(
    records: List[Dict],
    own_name: str,
    matchup_type: str,
) -> List[Dict]:
    """Converte registros JSON em linhas para o CSV."""
    rows = []
    for rec in records:
        try:
            games = int(rec.get("nbgames") or 0)
            wins  = int(rec.get("wins") or 0)
            if games <= 0:
                continue
            rows.append({
                "own_champion": own_name,
                "opp_champion": rec.get("champion_name", ""),
                "matchup_type": matchup_type,
                "role":         rec.get("role", ""),
                "games":        games,
                "winrate_pct":  round(wins / games * 100, 1),
                "csd15":        float(rec.get("csd") or 0),
                "gd15":         float(rec.get("gd") or 0),
                "xpd15":        float(rec.get("xpd") or 0),
                "kda":          float(rec.get("kda") or 0),
            })
        except (ValueError, TypeError):
            continue
    return rows


def fetch_champion_data(
    sess,
    champ_id: int,
    own_name: str,
    season: str,
    split: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Baixa Direct, Indirect e Good With para um campeo.

    Retorna (matchup_rows, synergy_rows, n_calls) onde:
      matchup_rows = direct + indirect combinados
      synergy_rows = good_with
    """
    direct   = _fetch_matchup_json(sess, "directmatchup",   champ_id, season, split)
    indirect = _fetch_matchup_json(sess, "indirectmatchup", champ_id, season, split)
    good_with = _fetch_matchup_json(sess, "goodwith",       champ_id, season, split)

    matchup_rows = []
    if direct is not None:
        matchup_rows += _json_to_rows(direct,    own_name, "direct")
    if indirect is not None:
        matchup_rows += _json_to_rows(indirect,  own_name, "indirect")

    synergy_rows = []
    if good_with is not None:
        synergy_rows += _json_to_rows(good_with, own_name, "good_with")

    return matchup_rows, synergy_rows, (
        len(direct or []), len(indirect or []), len(good_with or [])
    )


# ---------- I/O ----------

_MATCHUP_FIELDS = [
    "own_champion", "opp_champion", "matchup_type",
    "role", "games", "winrate_pct", "csd15", "gd15", "xpd15", "kda",
]


def append_csv(rows: List[Dict], path: Path, write_header: bool) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_MATCHUP_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Baixa matchups/synergies do gol.gg premium")
    ap.add_argument("--season",  default="S16")
    ap.add_argument("--split",   default="Spring")
    ap.add_argument("--browser", default="brave",
                    choices=["brave", "chrome", "edge"])
    ap.add_argument("--limit",   type=int, default=0,
                    help="Limita a N campeoes (0 = todos)")
    ap.add_argument("--delay",   type=float, default=0.5,
                    help="Segundos entre campeoes (padrao 0.5)")
    ap.add_argument("--resume",  action="store_true",
                    help="Retoma download, pulando campeoes ja salvos")
    ap.add_argument("--cookie",  default="",
                    help='Cookie manual: "rememberme=xxx"')
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Carregando sessao do {args.browser}...")
    sess = make_session(args.browser, cookie_string=args.cookie)

    if not check_logged_in(sess):
        raise SystemExit(1)
    print("[OK] Sessao premium valida\n")

    print(f"[*] Buscando lista de campeoes (season={args.season}, split={args.split})...")
    champions = get_champion_ids(sess, args.season, args.split)
    if not champions:
        print("[ERR] Nao foi possivel obter lista de campeoes.")
        raise SystemExit(1)

    ids_path = DATA_DIR / "champion_ids.csv"
    with open(ids_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name"])
        w.writeheader()
        w.writerows(champions)
    print(f"  -> IDs salvos em {ids_path.name}\n")

    if args.limit:
        champions = champions[: args.limit]
        print(f"  (limitado a {args.limit} campeoes)\n")

    matchups_path  = DATA_DIR / "champion_matchups.csv"
    synergies_path = DATA_DIR / "champion_synergies.csv"

    done_names: set = set()
    if args.resume and matchups_path.exists():
        with open(matchups_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_names.add(row.get("own_champion", ""))
        print(f"  -> Resumindo: {len(done_names)} campeoes ja baixados\n")

    first_write = not (args.resume and matchups_path.exists())
    total_matchups = 0
    total_synergies = 0
    errors = 0

    for i, champ in enumerate(champions, 1):
        name = champ["name"]
        cid  = champ["id"]
        prefix = f"[{i:>3}/{len(champions)}] {name:<20} (id={cid})"

        if args.resume and name in done_names:
            print(f"{prefix} -> pulado")
            continue

        print(f"{prefix} ...", end=" ", flush=True)
        matchup_rows, synergy_rows, counts = fetch_champion_data(
            sess, cid, name, args.season, args.split
        )
        d_count, i_count, g_count = counts

        if not matchup_rows and not synergy_rows:
            print(f"-- sem dados (D={d_count} I={i_count} G={g_count})")
            errors += 1
            time.sleep(args.delay)
            continue

        append_csv(matchup_rows,  matchups_path,  write_header=first_write)
        append_csv(synergy_rows,  synergies_path, write_header=first_write)
        first_write = False

        total_matchups  += len(matchup_rows)
        total_synergies += len(synergy_rows)
        print(f"[OK] D={d_count} I={i_count} G={g_count}")

        time.sleep(args.delay)

    print(f"\n{'='*55}")
    print("[OK] Concluido!")
    print(f"   Matchups:  {total_matchups:>5} linhas -> {matchups_path.name}")
    print(f"   Synergies: {total_synergies:>5} linhas -> {synergies_path.name}")
    print(f"   Erros/sem dados: {errors}")
    if errors > 10:
        print("   [!] Muitos erros -- verifique a sessao (--limit 3)")


if __name__ == "__main__":
    main()
