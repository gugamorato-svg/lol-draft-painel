"""Atualiza os CSVs do Oracle's Elixir (dados de partidas) automaticamente.

O Oracle's Elixir publica 1 CSV por ano numa pasta publica do Google Drive,
atualizada 1x por dia. Esta e a fonte que alimenta o MODELO (treino/predicao)
via data_pipeline/oracle_loader.py — e ela NAO se atualiza sozinha: os arquivos
em data/oracle_elixir/{ano}_lol.csv sao estaticos ate serem rebaixados.

Este script fecha esse gap:
  1. Lista a pasta do Drive (resolve o file id do ano na hora — nao hardcoda id,
     entao sobrevive a re-uploads do Oracle).
  2. Baixa o(s) ano(s) alvo para um arquivo temporario. Tenta ANONIMO primeiro;
     se o Google bloquear por quota ("Too many users..."), cai pro modo
     AUTENTICADO com cookie do Google extraido do Brave via CDP (a quota
     autenticada e separada da anonima). Mesmo padrao do download_golgg.py.
  3. Valida o download (header esperado + contagem de linhas) ANTES de substituir.
     Um download truncado por queda de rede NAO pode sobrescrever um CSV bom.
  4. Substitui de forma atomica (os.replace) so se passar na validacao.

Uso basico (baixa o ano corrente, anonimo com fallback autenticado):
    python scripts/update_oracle.py

Opcoes:
    --years 2025 2026     anos a baixar (padrao: ano corrente)
    --folder-url URL      pasta do Drive (padrao: pasta oficial do Oracle)
    --browser brave       navegador p/ cookie autenticado (brave|chrome|edge)
    --auth-only           pula o modo anonimo, ja vai direto no autenticado
    --no-auth             so anonimo (nao tenta cookie do navegador)
    --force               sobrescreve mesmo se o novo CSV tiver menos linhas
    --quiet               menos output do gdown

Para o modo autenticado funcionar, esteja LOGADO no Google no navegador
escolhido (--browser). O script abre o navegador em modo debug, le o cookie
e fecha — nao guarda o cookie em disco.

Requer: gdown, requests (pip install gdown requests).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

# Pasta publica oficial do Oracle's Elixir (1 CSV por ano, atualizado diariamente).
# Se algum dia mudar, pegue a nova em https://oracleselixir.com/tools/downloads
# e passe via --folder-url.
DEFAULT_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
)

DATA_DIR = Path(__file__).parent.parent / "data" / "oracle_elixir"

# Colunas que TEM que existir no header — se faltarem, o download veio corrompido
# ou o formato mudou, e nao devemos substituir o arquivo bom.
_REQUIRED_COLS = {"gameid", "participantid", "champion", "league", "date"}

# Guarda-corpo: um mesmo ano so cresce ao longo do tempo. Se o novo arquivo tiver
# menos que esta fracao das linhas do atual, provavelmente foi um download
# truncado — recusa a menos que --force.
_MIN_ROW_RATIO = 0.9

# Sinais de que o Google devolveu a pagina de quota/HTML em vez do CSV.
_QUOTA_MARKERS = ("quota exceeded", "too many users")


# ---------- cookies do Google via CDP (espelha download_golgg.py) ----------

def _get_google_cookies_via_cdp(browser: str) -> dict:
    """Abre o navegador em modo debug e extrai cookies *.google.com.

    Reaproveita o mesmo mecanismo do scripts/download_golgg.py: como o Chrome/
    Brave v20 cifra os cookies (App-Bound Encryption), a unica forma confiavel
    de le-los e via CDP (Chrome DevTools Protocol) com o navegador rodando.
    """
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

    exe = next((p for p in exe_candidates.get(browser, []) if Path(p).exists()), None)
    if not exe:
        print(f"  [!] Executavel do {browser} nao encontrado")
        return {}

    user_data = user_data_dirs.get(browser, "")
    debug_port = 9222

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
                check=True, stdout=subprocess.DEVNULL,
            )
        import websocket as ws_lib

        ws_conn = ws_lib.create_connection(ws_url, timeout=10)
        ws_conn.send(json.dumps(
            {"id": 1, "method": "Network.getAllCookies", "params": {}}
        ))
        resp = json.loads(ws_conn.recv())
        ws_conn.close()

        all_cookies = resp.get("result", {}).get("cookies", [])
        google_cookies = {
            c["name"]: c["value"]
            for c in all_cookies
            if "google.com" in c.get("domain", "")
        }
        if google_cookies:
            print(f"  [OK] {len(google_cookies)} cookies google.com via CDP")
        else:
            print("  [!] Nenhum cookie google.com — voce esta logado no Google?")
        return google_cookies
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] CDP falhou: {exc}")
        return {}
    finally:
        if proc is not None:
            print(f"  [*] Encerrando {browser}...")
            proc.terminate()


def _list_folder(folder_url: str) -> List[object]:
    """Lista os arquivos da pasta do Drive sem baixar (resolve ids na hora)."""
    import gdown

    files = gdown.download_folder(
        folder_url, skip_download=True, quiet=True, use_cookies=False
    )
    if not files:
        raise RuntimeError(
            "Pasta do Drive vazia ou inacessivel. Verifique --folder-url."
        )
    return files


def _resolve_year_ids(files: List[object]) -> Dict[int, str]:
    """Mapeia {ano: file_id} a partir dos nomes '{ano}_LoL_esports_match_data...'."""
    out: Dict[int, str] = {}
    for f in files:
        name = Path(f.path).name
        head = name.split("_", 1)[0]
        try:
            year = int(head)
        except ValueError:
            continue
        out[year] = f.id
    return out


class QuotaExceeded(RuntimeError):
    """Google devolveu a pagina de quota em vez do arquivo (acesso anonimo)."""


def _tmp_path(dest: Path) -> Path:
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    return tmp


def _looks_like_quota_html(path: Path) -> bool:
    """True se o 'download' foi na verdade a pagina HTML de quota do Drive."""
    try:
        head = path.read_bytes()[:2048].decode("utf-8", "replace").lower()
    except OSError:
        return False
    if "<!doctype html" not in head and "<html" not in head:
        return False
    return any(m in head for m in _QUOTA_MARKERS)


def _download_anonymous(file_id: str, dest: Path, quiet: bool) -> Path:
    """Baixa via gdown anonimo. Levanta QuotaExceeded se o Google bloquear."""
    import gdown

    tmp = _tmp_path(dest)
    try:
        got = gdown.download(id=file_id, output=str(tmp), quiet=quiet)
    except Exception as exc:  # gdown levanta FileURLRetrievalError na quota
        if any(m in str(exc).lower() for m in _QUOTA_MARKERS):
            raise QuotaExceeded(str(exc).splitlines()[0]) from exc
        raise
    if (not got or not tmp.exists()) or _looks_like_quota_html(tmp):
        tmp.unlink(missing_ok=True)
        raise QuotaExceeded("gdown retornou pagina de quota/HTML")
    return tmp


def _download_authenticated(
    file_id: str, dest: Path, cookies: dict, quiet: bool
) -> Path:
    """Baixa via drive.usercontent.google.com com cookies do Google logado.

    A quota autenticada e separada da anonima, entao isso costuma funcionar
    quando o anonimo bate em 'Too many users'. Trata o formulario de
    confirmacao de 'arquivo grande' que o Drive devolve.
    """
    import requests

    sess = requests.Session()
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    })
    for name, value in cookies.items():
        sess.cookies.set(name, value, domain=".google.com")

    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "confirm": "t"}

    r = sess.get(url, params=params, stream=True, timeout=60)
    ctype = r.headers.get("content-type", "")

    # Se veio HTML, ou e a pagina de quota ou o formulario de confirmacao.
    if "text/html" in ctype:
        body = r.content.decode("utf-8", "replace")
        low = body.lower()
        if any(m in low for m in _QUOTA_MARKERS):
            raise QuotaExceeded("quota bloqueada mesmo autenticado")
        # Extrai os hidden inputs do formulario de confirmacao e re-submete.
        action = re.search(r'action="([^"]+)"', body)
        hidden = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', body))
        if not action or not hidden:
            raise RuntimeError("HTML inesperado do Drive (sem formulario)")
        r = sess.get(action.group(1).replace("&amp;", "&"),
                     params=hidden, stream=True, timeout=60)
        if "text/html" in r.headers.get("content-type", ""):
            raise RuntimeError("Drive insistiu em HTML apos confirmacao")

    tmp = _tmp_path(dest)
    total = 0
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
                total += len(chunk)
    if not quiet:
        print(f"  [OK] autenticado: {total/1e6:.1f} MB baixados")
    if _looks_like_quota_html(tmp):
        tmp.unlink(missing_ok=True)
        raise QuotaExceeded("pagina de quota mesmo autenticado")
    return tmp


def _count_and_validate(csv_path: Path) -> int:
    """Confere o header e conta as linhas de dados. Levanta se invalido."""
    if csv_path.stat().st_size < 1_000_000:  # CSVs de ano tem dezenas de MB
        raise RuntimeError(
            f"arquivo suspeito de tao pequeno ({csv_path.stat().st_size} bytes)"
        )
    with open(csv_path, encoding="utf-8") as fh:
        header = fh.readline()
        cols = {c.strip() for c in header.split(",")}
        missing = _REQUIRED_COLS - cols
        if missing:
            raise RuntimeError(f"header sem colunas obrigatorias: {sorted(missing)}")
        rows = sum(1 for _ in fh)  # linhas de dados (header ja consumido)
    if rows <= 0:
        raise RuntimeError("CSV sem linhas de dados")
    return rows


def _existing_row_count(csv_path: Path) -> Optional[int]:
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, encoding="utf-8") as fh:
            fh.readline()
            return sum(1 for _ in fh)
    except OSError:
        return None


def update_year(
    year: int,
    file_id: str,
    *,
    force: bool = False,
    quiet: bool = False,
    auth_only: bool = False,
    get_cookies=None,
) -> bool:
    """Baixa, valida e substitui data/oracle_elixir/{year}_lol.csv. Retorna True se trocou.

    `get_cookies` e um callable sem args que devolve o dict de cookies do Google
    (buscado sob demanda e cacheado pelo chamador). Se None, so tenta anonimo.
    """
    dest = DATA_DIR / f"{year}_lol.csv"
    old_rows = _existing_row_count(dest)

    print(f"[*] {year}: baixando (file id {file_id})...")
    tmp: Optional[Path] = None

    if not auth_only:
        try:
            tmp = _download_anonymous(file_id, dest, quiet=quiet)
        except QuotaExceeded as exc:
            print(f"  [!] anonimo bloqueado: {exc}")

    if tmp is None:
        if get_cookies is None:
            print(f"[ERR] {year}: quota anonima e --no-auth ligado — pulado.")
            return False
        cookies = get_cookies()
        if not cookies:
            print(f"[ERR] {year}: sem cookies do Google — nao da p/ autenticar.")
            return False
        print(f"  [*] {year}: tentando modo autenticado...")
        try:
            tmp = _download_authenticated(file_id, dest, cookies, quiet=quiet)
        except QuotaExceeded as exc:
            print(f"[ERR] {year}: {exc} — mantido o arquivo atual.")
            return False

    try:
        new_rows = _count_and_validate(tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"[ERR] {year}: download invalido — mantido o arquivo atual. ({exc})")
        return False

    if old_rows is not None and new_rows < old_rows * _MIN_ROW_RATIO and not force:
        tmp.unlink(missing_ok=True)
        print(
            f"[ERR] {year}: novo CSV tem {new_rows} linhas vs {old_rows} atuais "
            f"(< {_MIN_ROW_RATIO:.0%}). Suspeita de truncamento — mantido o atual. "
            f"Use --force para sobrescrever mesmo assim."
        )
        return False

    os.replace(tmp, dest)  # atomico no mesmo volume
    delta = "" if old_rows is None else f" (era {old_rows}, +{new_rows - old_rows})"
    print(f"[OK]  {year}: {new_rows} linhas{delta} -> {dest.name}")
    return True


def main() -> None:
    current_year = _dt.date.today().year
    ap = argparse.ArgumentParser(
        description="Baixa/atualiza os CSVs anuais do Oracle's Elixir."
    )
    ap.add_argument(
        "--years", type=int, nargs="+", default=[current_year],
        help=f"Anos a baixar (padrao: {current_year})",
    )
    ap.add_argument("--folder-url", default=DEFAULT_FOLDER_URL,
                    help="Pasta do Drive do Oracle's Elixir")
    ap.add_argument("--browser", default="brave",
                    choices=["brave", "chrome", "edge"],
                    help="Navegador p/ cookie autenticado (padrao brave)")
    ap.add_argument("--auth-only", action="store_true",
                    help="Pula o anonimo, ja vai no autenticado")
    ap.add_argument("--no-auth", action="store_true",
                    help="So anonimo, nunca usa cookie do navegador")
    ap.add_argument("--force", action="store_true",
                    help="Sobrescreve mesmo com menos linhas que o atual")
    ap.add_argument("--quiet", action="store_true",
                    help="Menos output do gdown")
    args = ap.parse_args()

    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[ERR] gdown nao instalado. Rode: pip install gdown")
        raise SystemExit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] Listando pasta do Oracle's Elixir no Drive...")
    try:
        files = _list_folder(args.folder_url)
        year_ids = _resolve_year_ids(files)
    except Exception as exc:
        print(f"[ERR] nao consegui listar a pasta: {exc}")
        raise SystemExit(1)

    if not year_ids:
        print("[ERR] nenhum arquivo '{ano}_LoL_esports_match_data...' na pasta.")
        raise SystemExit(1)

    print(f"    anos disponiveis no Drive: {sorted(year_ids)}\n")

    # Cookies do Google buscados no maximo 1x (abre/fecha o navegador uma vez).
    _cookie_cache: Dict[str, dict] = {}

    def get_cookies() -> dict:
        if "c" not in _cookie_cache:
            print(f"[*] Extraindo cookies do Google via {args.browser} (CDP)...")
            _cookie_cache["c"] = _get_google_cookies_via_cdp(args.browser)
        return _cookie_cache["c"]

    cookie_provider = None if args.no_auth else get_cookies

    changed = 0
    failed = 0
    for year in args.years:
        fid = year_ids.get(year)
        if not fid:
            print(f"[ERR] {year}: nao existe na pasta do Drive — pulado.")
            failed += 1
            continue
        try:
            if update_year(
                year, fid,
                force=args.force, quiet=args.quiet,
                auth_only=args.auth_only, get_cookies=cookie_provider,
            ):
                changed += 1
            else:
                failed += 1  # nao atualizou (quota/validacao) — conta como falha
        except Exception as exc:
            print(f"[ERR] {year}: {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"[OK] Atualizados: {changed} | Falhas/pulados: {failed}")
    if failed and not changed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
