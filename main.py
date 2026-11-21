"""LoL Painel - entry point."""

from __future__ import annotations

import atexit
import ctypes
import sys
from ctypes import wintypes

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


_MUTEX_NAME = "Local\\LoLPainelSingleInstance_v1"
_ERROR_ALREADY_EXISTS = 183


class SingleInstanceGuard:
    """Windows named mutex guard so the desktop shortcut opens only one app."""

    def __init__(self, name: str) -> None:
        self.handle: int | None = None
        self.already_running = False

        if sys.platform != "win32":
            return

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        self._kernel32 = kernel32
        self.handle = kernel32.CreateMutexW(None, False, name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

        self.already_running = ctypes.get_last_error() == _ERROR_ALREADY_EXISTS
        atexit.register(self.release)

    def release(self) -> None:
        if getattr(self, "handle", None):
            self._kernel32.CloseHandle(self.handle)
            self.handle = None


def _notify_already_open() -> None:
    message = "O LoL Painel ja esta aberto. Feche a janela atual antes de abrir outra."
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(None, message, "LoL Painel", 0x40)
    else:
        print(message)


def main() -> int:
    guard = SingleInstanceGuard(_MUTEX_NAME)
    if guard.already_running:
        guard.release()
        _notify_already_open()
        return 0

    app = QApplication(sys.argv)
    app.aboutToQuit.connect(guard.release)
    app._single_instance_guard = guard  # keep the mutex handle alive

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())