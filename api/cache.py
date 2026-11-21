"""Cache simples em SQLite para evitar bombardear as APIs.

Chave = (namespace, key_string). Valor = JSON. TTL configuravel por entrada.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional


class JsonCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        namespace TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        expires_at REAL NOT NULL,
                        PRIMARY KEY (namespace, key)
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def get(self, namespace: str, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value, expires_at FROM cache WHERE namespace=? AND key=?",
                    (namespace, key),
                ).fetchone()
            finally:
                conn.close()
        if not row:
            return None
        value, expires_at = row
        if expires_at < now:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    def set(self, namespace: str, key: str, value: Any, ttl_seconds: int) -> None:
        expires_at = time.time() + ttl_seconds
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (namespace, key, value, expires_at) "
                    "VALUES (?, ?, ?, ?)",
                    (namespace, key, json.dumps(value), expires_at),
                )
                conn.commit()
            finally:
                conn.close()

    def clear(self, namespace: Optional[str] = None) -> None:
        with self._lock:
            conn = self._connect()
            try:
                if namespace:
                    conn.execute("DELETE FROM cache WHERE namespace=?", (namespace,))
                else:
                    conn.execute("DELETE FROM cache")
                conn.commit()
            finally:
                conn.close()

    def close(self) -> None:
        return None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "JsonCache":
        return self

    def __exit__(self, *_) -> None:
        self.close()
