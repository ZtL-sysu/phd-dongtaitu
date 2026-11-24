"""Conservative RUN-owned Python job scanner with read-only Toolhelp fallback.

Native metadata is used only to classify an otherwise unnamed process. It does
not supply or infer command lines or working directories.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any


class InventoryError(RuntimeError):
    pass


def _toolhelp_snapshot() -> dict[int, dict[str, Any]]:
    if os.name != "nt":
        raise InventoryError("native process-name fallback requires Windows Toolhelp32")
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel.Process32FirstW.restype = wintypes.BOOL
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel.Process32NextW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL

    ctypes.set_last_error(0)
    handle = kernel.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid or handle is None:
        raise ctypes.WinError(ctypes.get_last_error())
    rows: dict[int, dict[str, Any]] = {}
    try:
        row = PROCESSENTRY32W()
        row.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ctypes.set_last_error(0)
        ok = kernel.Process32FirstW(handle, ctypes.byref(row))
        if not ok:
            error = ctypes.get_last_error()
            if error != 18:  # ERROR_NO_MORE_FILES; an empty/unreadable list is not usable.
                raise ctypes.WinError(error)
            raise InventoryError("Toolhelp32 returned an empty process snapshot")
        while ok:
            pid = int(row.th32ProcessID)
            name = str(row.szExeFile)
            if pid < 0 or not name or pid in rows:
                raise InventoryError("Toolhelp32 process snapshot has an invalid or duplicate row")
            rows[pid] = {"name": name, "parent_pid": int(row.th32ParentProcessID)}
            ctypes.set_last_error(0)
            ok = kernel.Process32NextW(handle, ctypes.byref(row))
        error = ctypes.get_last_error()
        if error != 18:  # ERROR_NO_MORE_FILES is the only valid terminal condition.
            raise ctypes.WinError(error)
    finally:
        if not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    if not rows:
        raise InventoryError("Toolhelp32 returned an empty process snapshot")
    return rows


def owned_processes(root: Path, *, process_iter=None, current_pid=None, ancestor_pids=None,
                    native_snapshot=None) -> list[dict[str, Any]]:
    """Find RUN-owned Python jobs; ambiguous ownership/classification fails closed."""
    import psutil

    root = Path(root).resolve()
    current_pid = os.getpid() if current_pid is None else current_pid
    if ancestor_pids is None:
        try:
            ancestors = {p.pid for p in psutil.Process(current_pid).parents()}
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            ancestors = set()
    else:
        ancestors = set(ancestor_pids)
    if process_iter is None:
        process_iter = psutil.process_iter(["pid", "cmdline", "cwd", "exe", "name"])

    script_names = {
        "baseline_campaign.py", "bound_phase_campaign.py", "frozen_robustness_phase.py",
        "profile_training_forward_components.py", "count_loss_training_controller_v2.py",
        "count_loss_training_controller_v3.py", "count_loss_training_controller_v4.py",
        "count_loss_training_controller_v5.py", "count_loss_training_adapter_v1.py",
    }
    snapshot = None

    def native_name(pid: int) -> str:
        nonlocal snapshot
        if snapshot is None:
            try:
                snapshot = (native_snapshot or _toolhelp_snapshot)()
            except Exception as exc:
                raise InventoryError(f"cannot obtain Toolhelp32 process-name fallback: {exc}") from exc
            if not isinstance(snapshot, dict) or not snapshot:
                raise InventoryError("Toolhelp32 process-name fallback is empty or malformed")
        record = snapshot.get(pid)
        name = record.get("name") if isinstance(record, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise InventoryError(f"Toolhelp32 has no classifiable executable name for PID {pid}")
        return name

    def is_python(exe, argv, name=""):
        candidates = [str(exe or ""), str(name or "")]
        if argv:
            candidates.append(str(argv[0]))
        return any(Path(x.replace("\\", "/")).name.casefold().startswith(("python", "pypy")) for x in candidates if x)

    def under_root(path, cwd):
        if not path:
            return False
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(cwd or root) / candidate
        try:
            return candidate.resolve(strict=False).is_relative_to(root)
        except (OSError, ValueError):
            return False

    def python_target(argv):
        if not argv or len(argv) < 2:
            return None
        no_value = {"-b", "-bb", "-B", "-d", "-E", "-h", "-i", "-I", "-O", "-OO",
                    "-q", "-s", "-S", "-u", "-v", "-vv", "-V", "-x"}
        i = 1
        while i < len(argv):
            arg = str(argv[i])
            if arg in {"-m", "-c"}:
                if i + 1 >= len(argv):
                    return None
                return ("module" if arg == "-m" else "command", str(argv[i + 1]))
            if arg in {"-X", "-W", "--check-hash-based-pycs"}:
                if i + 1 >= len(argv):
                    return None
                i += 2
                continue
            if arg.startswith(("-X", "-W")) or arg in no_value:
                i += 1
                continue
            if arg == "--":
                return ("script", str(argv[i + 1])) if i + 1 < len(argv) else None
            if arg.startswith("-"):
                return None
            return ("script", arg)
        return None

    def job_identity(argv, cwd):
        target = python_target(argv)
        if target is None or target[0] == "command":
            return False
        kind, value = target
        if kind == "module":
            module = value.casefold()
            return module in {"edsgn.train", "edsgn.calibrate", "edsgn.evaluate", "edsgn.profile"} or (
                module.startswith("edsgn.") and any(t in module for t in ("train", "calibrat", "evaluat", "profile")))
        script = value.replace("\\", "/")
        base = Path(script).name.casefold()
        if base in script_names or (base.startswith("count_loss_training_controller_v") and base.endswith(".py")) or (
                base.startswith("count_loss_training_adapter_v") and base.endswith(".py")):
            return True
        if "resource_probe" in base or (base.startswith("frozen_") and base.endswith(".py")):
            return True
        return script.casefold().endswith(".py") and under_root(script, cwd) and any(
            token in base for token in ("train", "calibrat", "evaluat", "profile", "robust", "resource"))

    active = []
    for proc in process_iter:
        pid = None
        try:
            pid = int(proc.pid)
            if pid == current_pid or pid in ancestors:
                continue
            info = proc.info
            info = info if isinstance(info, dict) else {}
            raw_argv = info.get("cmdline")
            argv = [str(x) for x in raw_argv] if isinstance(raw_argv, (list, tuple)) else []
            cwd = info.get("cwd") or ""
            exe = info.get("exe") or ""
            name = info.get("name") or ""
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, OSError, TypeError, ValueError) as exc:
            try:
                name = proc.name() or ""
            except psutil.NoSuchProcess:
                continue
            except (psutil.AccessDenied, OSError):
                name = ""
            if not name and pid is not None:
                name = native_name(pid)
            if not name or not name.casefold().startswith(("python", "pypy")):
                continue
            raise InventoryError(f"cannot inspect process {pid}; refusing launch") from exc
        if not name:
            try:
                name = proc.name() or ""
            except psutil.NoSuchProcess:
                continue
            except (psutil.AccessDenied, OSError):
                name = ""
        if not name and pid is not None:
            name = native_name(pid)
        python_process = is_python(exe, argv, name)
        if not python_process:
            if not (exe or name or (argv and argv[0])):
                raise InventoryError(f"cannot classify process {pid}; refusing launch")
            continue
        cwd_known = bool(cwd)
        argv_known = bool(argv)
        owned = (cwd_known and under_root(cwd, root)) or any(under_root(arg, cwd) for arg in argv[1:])
        if not owned and (not cwd_known or not argv_known):
            raise InventoryError(f"cannot establish ownership of Python process {pid}; refusing launch")
        if not owned:
            continue
        target = python_target(argv)
        if target is None:
            raise InventoryError(f"RUN-owned Python process {pid} has no inspectable argv; refusing launch")
        if target[0] == "command":
            raise InventoryError(f"RUN-owned Python -c process {pid} has no inspectable script target; refusing launch")
        if job_identity(argv, cwd):
            active.append({"pid": pid})
    return active
