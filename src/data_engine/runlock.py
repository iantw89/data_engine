"""Cross-process run lock for a workspace: ``<workspace>/logs/run.lock`` = JSON ``{pid, started, command}``.

The CLI takes the lock around a round; the web server reads :func:`holder` to know whether a
run is in progress. A lock whose process is gone is stale and is taken over by the next
``acquire``. (On Windows ``os.kill(pid, 0)`` would *terminate* the process, so liveness is
checked with psutil, or OpenProcess/GetExitCodeProcess when psutil is unavailable.)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def lock_path(workspace) -> Path:
    """Accept a Workspace, or the workspace directory itself."""
    if isinstance(workspace, (str, Path)):
        return Path(workspace) / "logs" / "run.lock"
    return workspace.path("logs", "run.lock")


def pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists
        try:
            code = wintypes.DWORD()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def holder(workspace) -> dict | None:
    """The live holder ``{pid, started, command}``, or None when no run is in progress (stale locks count as free)."""
    data = _read(lock_path(workspace))
    if data is None or not pid_alive(data.get("pid")):
        return None
    return data


def acquire(workspace, command: str = "") -> bool:
    """Take the lock; False if another live process (or this one) holds it."""
    path = lock_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "started": time.time(), "command": command or " ".join(sys.argv)})
    for _ in range(3):
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            current = _read(path)
            if current is not None and pid_alive(current.get("pid")):
                return False
            try:  # stale (dead pid) or unreadable: take over; losing the race to another taker is fine
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return False
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        return True
    return False


def release(workspace) -> None:
    """Remove the lock if this process holds it (or it is stale); never remove another live holder's lock."""
    path = lock_path(workspace)
    current = _read(path)
    if current is not None and current.get("pid") != os.getpid() and pid_alive(current.get("pid")):
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
