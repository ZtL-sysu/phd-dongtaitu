"""Read-only root diagnostic; does not authorize any launch."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys
import psutil

ROOT = Path(__file__).resolve().parents[2]

class Entry(ctypes.Structure):
    _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
                ('th32ProcessID', wintypes.DWORD), ('th32DefaultHeapID', ctypes.c_size_t),
                ('th32ModuleID', wintypes.DWORD), ('cntThreads', wintypes.DWORD),
                ('th32ParentProcessID', wintypes.DWORD), ('pcPriClassBase', wintypes.LONG),
                ('dwFlags', wintypes.DWORD), ('szExeFile', wintypes.WCHAR * 260)]

def snapshot():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32FirstW.restype = wintypes.BOOL
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32NextW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    rows = {}
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(Entry)
        ok = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while ok:
            rows[int(entry.th32ProcessID)] = {
                'name': entry.szExeFile, 'parent_pid': int(entry.th32ParentProcessID)}
            ok = kernel.Process32NextW(handle, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)
    return rows

def main():
    sys.path[:0] = [str(ROOT/'edsgn/scripts'), str(ROOT/'edsgn/staging'), str(ROOT/'edsgn/src')]
    import count_loss_training_controller_v4 as controller
    rows = snapshot()
    focused = list(psutil.process_iter(['pid', 'cmdline', 'cwd', 'exe', 'name']))
    focused = [p for p in focused if p.pid == 33368]
    report = {
        'scope': 'metadata_only_native_snapshot_and_exact_live_pid_scan_no_launch',
        'native_snapshot_selected': {str(pid): rows.get(pid) for pid in (188, 33368)},
        'filtered_live_scan': controller._owned_processes(ROOT, process_iter=focused, ancestor_pids=[]),
        'global_scan_or_launch_authorized': False}
    output = ROOT/'logs/count-controller-v4-root-live-process-scan-v1.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
