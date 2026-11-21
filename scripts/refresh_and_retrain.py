"""Atualiza o dado do Oracle e RETREINA os modelos - so se vier dado novo.

Roda na tarefa agendada OracleElixir_DailyRefresh (a cada 3h). Fluxo:
  1. Tenta baixar o 2026 (anonimo; SEM Brave -> nao trava a tarefa). Falhar
     por quota do Drive e ok: seguimos com o dado local.
  2. Compara a data-max do CSV local com o MARCADOR data/last_trained_data_date.txt
     (data-max do ultimo retreino 100% bem-sucedido). Se o local e mais novo,
     retreina: team_only (snapshot) -> prematch (principal) -> pure_draft ->
     live_wr (este so se o principal deu certo; valida no holdout antes de salvar).
  3. So grava o marcador se TODOS deram certo -> falha parcial retenta no proximo run.
Saidas: 0 = em dia/retreinado | 1 = download falhou mas modelos em dia |
        3 = retreino parcial | 4 = CSV ilegivel.

O caminho AUTENTICADO (Brave CDP) foi removido: Brave/Chrome recentes bloqueiam
remote-debugging no perfil real, entao ele so travava a tarefa. Sobra o anonimo,
que abre em janelas - por isso rodamos com frequencia.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
PY = sys.executable
CSV_2026 = ROOT / "data" / "oracle_elixir" / "2026_lol.csv"
LOG = ROOT / "logs" / "refresh_and_retrain.log"
MARKER = ROOT / "data" / "last_trained_data_date.txt"  # data-max do ultimo retreino 100% OK


def _log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _max_date(path: Path):
    try:
        d = pd.read_csv(path, usecols=["date"], low_memory=False)
        return pd.to_datetime(d["date"], errors="coerce").max()
    except Exception:
        return None


# 15/10 (dono): a tarefa roda com pythonw.exe (sem janela); os passos usam o python.exe da
# mesma instalacao com CREATE_NO_WINDOW -> nada aparece na tela a cada 3 h.
_PY_FILHO = Path(PY).with_name("python.exe") if Path(PY).name.lower() == "pythonw.exe" else Path(PY)
_SEM_JANELA = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run(args, label):
    _log(f"  -> {label}...")
    r = subprocess.run([str(_PY_FILHO)] + args, cwd=str(ROOT), capture_output=True, text=True,
                       creationflags=_SEM_JANELA)
    if r.returncode != 0:
        _log(f"     [ERRO rc={r.returncode}] {(r.stdout or '')[-300:]} {(r.stderr or '')[-300:]}")
    return r.returncode


def _read_marker():
    try:
        return pd.Timestamp(MARKER.read_text(encoding="utf-8").strip())
    except Exception:
        return None  # sem marcador = nunca treinado com sucesso -> retreina


def main() -> int:
    _log("==== refresh_and_retrain ====")

    # 1) download (falhar e ok: ainda retreinamos dado LOCAL que nunca foi treinado)
    rc_dl = _run(["scripts/update_oracle.py", "--years", "2026", "--no-auth", "--quiet"],
                 "download 2026 (anonimo)")
    if rc_dl != 0:
        _log("[quota bloqueada] download falhou - usa o dado local.")

    # 2) retreina se o dado local e mais novo que o ULTIMO TREINO BEM-SUCEDIDO.
    #    (antes comparava so antes/depois do download: se o retreino falhasse,
    #    nunca mais tentava -> foi assim que o dado de 03/08 ficou sem treinar)
    local = _max_date(CSV_2026)
    trained = _read_marker()
    _log(f"dado local ate {local} | ultimo treino OK com dado ate {trained}")
    if local is None:
        _log("[ERRO] nao consegui ler o CSV 2026.")
        return 4
    if trained is not None and local <= trained:
        _log("modelos ja estao em dia com o dado local - nada a retreinar.")
        return 0 if rc_dl == 0 else 1
    after = local

    _log("dado NAO TREINADO detectado -> retreinando modelos...")
    rc_team = _run(["scripts/train_team_only.py", "--prior"], "train_team_only (snapshot+team)")
    rc_main = _run(["scripts/train_prematch_sweep.py", "--prior", "--save-cap", "1.6"], "train_prematch (principal)")
    rc_draft = _run(["scripts/train_pure_draft.py"], "train_pure_draft (champs_only)")
    # live_wr depende dos coefs de comp do principal -> so roda se ele deu certo.
    # O proprio script valida no holdout e NAO sobrescreve se vier ruim.
    if rc_main == 0:
        rc_live = _run(["scripts/train_live_wr.py"], "train_live_wr (WR ao vivo)")
    else:
        rc_live = -1
        _log("  [pulado] train_live_wr: modelo principal falhou.")
    status = {"team": rc_team, "principal": rc_main, "draft": rc_draft, "live_wr": rc_live}
    ok = all(v == 0 for v in status.values())
    if ok:
        # so marca como treinado se TUDO deu certo -> falha parcial retenta no proximo run
        MARKER.write_text(str(after), encoding="utf-8")
    _log(f"[{'OK' if ok else 'PARCIAL - retenta no proximo run'}] retreino com dado ate {after} | rc={status}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
