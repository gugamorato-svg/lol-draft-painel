"""Pesos IMPLANTADOS: ajuste final + metadados (13/10/2026, auditoria).

Antes os 3 treinadores do retreino de 3 em 3 h (train_team_only, train_prematch_sweep,
train_pure_draft) ajustavam nos primeiros 80% dos jogos (corte em 19-26/04/2026), avaliavam nos
20% finais e SALVAVAM esse ajuste -- os pesos de producao nunca viam os ultimos ~6 mil jogos e
divergiam do backtest mensal (que usa todo o passado de cada mes). Agora: a avaliacao continua
80/20 temporal (so p/ medir), e o que vai p/ producao e o MESMO modelo (mesmas features, C=1)
re-ajustado com todo o passado. Cada arquivo de pesos ganha uma entrada em data/model_meta.json:
quando treinou, ate que data de jogo, onde foi o corte da avaliacao, as metricas e o sha256.
Medido em 13/10: os pesos ate abril x todo o passado mudam a prob em mediana 0,7pp (jun-set:
+58,1u x +59,7u) -> e correcao de pratica, nao de lucro.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
META_PATH = DATA_DIR / "model_meta.json"
UPSTREAM_SOURCES = (
    "models/trained_predictor.py", "models/predictor.py", "models/model_meta.py",
    "data_pipeline/temporal_context.py", "data_pipeline/oracle_loader.py",
    "data_pipeline/training_timeline.py", "data/player_aliases.json",
    "data_pipeline/golgg_prior.py", "scripts/train_pure_draft.py",
    "scripts/train_prematch_sweep.py", "scripts/train_team_only.py",
    "api/player_resolver.py", "api/livefeed.py", "api/oracle_stats.py",
    "api/esports.py", "api/cache.py", "api/leaguepedia.py", "api/lplqq.py",
    "ui/worker.py", "data_pipeline/patches.py",
    "scripts/refresh_and_retrain.py",
)


def upstream_source_sha256(root: Optional[Path] = None) -> str:
    """Identifica a politica de features/treino, separada dos pesos atualizados.

    Falha se algum fonte estiver ausente; nao permite tratar codigo desconhecido
    como a mesma politica do estudo prospectivo.
    """
    root = Path(root) if root is not None else DATA_DIR.parent
    h = hashlib.sha256()
    for name in UPSTREAM_SOURCES:
        content = (root / name).read_bytes().replace(b"\r\n", b"\n")
        h.update(name.encode("utf-8") + b"\0")
        h.update(len(content).to_bytes(8, "big"))
        h.update(content)
    return h.hexdigest()


def ajuste_final(X, y, C: float = 1.0):
    """O modelo de producao: mesma regressao da avaliacao, com TODO o passado."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=1000).fit(X, y)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_meta(path: Path = META_PATH) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def registrar(arquivo: str, datas: Iterable, split_at: int, avaliacao: Dict[str, float],
              path: Path = META_PATH, data_dir: Path = DATA_DIR, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Grava os metadados do arquivo de pesos `arquivo` (ja salvo em data_dir)."""
    import pandas as pd
    d = pd.to_datetime(pd.Series(list(datas)), errors="coerce")
    entrada = {
        "treinado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pesos": "ajuste final com todo o passado (avaliacao 80/20 temporal so p/ medir)",
        "jogos": int(len(d)),
        "treino_ate": str(d.max())[:19],
        "avaliacao": {"corte": str(d.iloc[split_at])[:19], "jogos": int(len(d) - split_at),
                      **{k: round(float(v), 4) for k, v in avaliacao.items()}},
        "sha256": sha256(data_dir / arquivo)[:12],
        **(extra or {}),
    }
    meta = load_meta(path)
    meta[arquivo] = entrada
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return entrada
