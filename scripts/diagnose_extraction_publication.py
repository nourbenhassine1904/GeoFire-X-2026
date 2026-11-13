"""Read-only Windows publication diagnostics. Never renames or unlocks files."""
import argparse
import ctypes
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import uuid

from geofirex.corpus import json_new, sha256_file, verify_member


def windows_locks(paths):
    if os.name != 'nt':
        return dict(status='UNSUPPORTED_PLATFORM')

    class Process(ctypes.Structure):
        _fields_ = [('pid', w.DWORD), ('started', w.FILETIME)]

    class Info(ctypes.Structure):
        _fields_ = [('process', Process), ('name', w.WCHAR * 256),
                    ('service', w.WCHAR * 64), ('type', w.UINT),
                    ('status', w.ULONG), ('session', w.DWORD), ('restartable', w.BOOL)]

    rm = ctypes.WinDLL('rstrtmgr', use_last_error=True)
    rm.RmStartSession.argtypes = [ctypes.POINTER(w.DWORD), w.DWORD, w.LPWSTR]
    rm.RmRegisterResources.argtypes = [w.DWORD, w.UINT, ctypes.POINTER(w.LPCWSTR),
                                      w.UINT, ctypes.c_void_p, w.UINT, ctypes.c_void_p]
    rm.RmGetList.argtypes = [w.DWORD, ctypes.POINTER(w.UINT), ctypes.POINTER(w.UINT),
                            ctypes.POINTER(Info), ctypes.POINTER(w.DWORD)]
    rm.RmEndSession.argtypes = [w.DWORD]
    handle, key = w.DWORD(), ctypes.create_unicode_buffer(33)
    code = rm.RmStartSession(ctypes.byref(handle), 0, key)
    if code:
        return dict(status='ERROR', operation='RmStartSession', winerror=code)
    try:
        names = (w.LPCWSTR * len(paths))(*map(str, paths))
        code = rm.RmRegisterResources(handle, len(paths), names, 0, None, 0, None)
        if code:
            return dict(status='ERROR', operation='RmRegisterResources', winerror=code)
        needed, count, reasons = w.UINT(), w.UINT(), w.DWORD()
        for _ in range(4):
            rows = (Info * count.value)() if count.value else None
            code = rm.RmGetList(handle, ctypes.byref(needed), ctypes.byref(count), rows,
                                ctypes.byref(reasons))
            if code == 234:  # ERROR_MORE_DATA: the process list can change.
                count.value = needed.value
                continue
            if code:
                return dict(status='ERROR', operation='RmGetList', winerror=code)
            return dict(status='QUERIED', processes=[dict(pid=rows[i].process.pid,
                         name=rows[i].name, service=rows[i].service) for i in range(count.value)])
        return dict(status='PROCESS_LIST_UNSTABLE')
    finally:
        rm.RmEndSession(handle)


def delete_access_probe(path):
    """Request DELETE access, immediately close; no delete disposition is set."""
    if os.name != 'nt':
        return dict(status='UNSUPPORTED_PLATFORM')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p,
                                  w.DWORD, w.DWORD, w.HANDLE]
    kernel.CreateFileW.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handle = kernel.CreateFileW(str(path), 0x10000, 7, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        code = ctypes.get_last_error()
        return dict(status='DENIED', winerror=code, message=ctypes.FormatError(code))
    kernel.CloseHandle(handle)
    return dict(status='ACCESS_GRANTED_NO_MUTATION')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default='D:/GeoFire-X/WildfireSpreadTS/rasters')
    parser.add_argument('--entry', default='2018/fire_21890115/2018-07-09.tif')
    parser.add_argument('--part', default='2018-07-09.tif.459a753a50264d4f9c9f6704007da800.part')
    parser.add_argument('--index', default='data/manifests/real/corpus_zip_index_v1.json')
    parser.add_argument('--report')
    args = parser.parse_args()
    root = Path(args.destination).resolve()
    entries = json.loads(Path(args.index).read_text(encoding='utf-8'))['entries']
    entry = next(e for e in entries if e['path'] == args.entry)
    output = root / entry['path']
    part = output.with_name(args.part)
    if not part.resolve().is_relative_to(root) or not output.resolve().is_relative_to(root):
        raise ValueError('Path escapes destination')
    before = part.stat()
    result = dict(output=str(output), output_exists=output.exists(), part=str(part),
                  bytes=before.st_size, expected_bytes=entry['geotiff_bytes'],
                  expected_crc32=entry['crc32'], crc_valid=verify_member(part, entry),
                  sha256=sha256_file(part), attributes=getattr(before, 'st_file_attributes', None),
                  restart_manager=windows_locks([part]), delete_access=delete_access_probe(part),
                  note='No rename/delete/shutdown/restart was requested; transient or kernel filter locks may be absent from this snapshot.')
    after = part.stat()
    result['unchanged_during_diagnosis'] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    report = Path(args.report or f'reports/extraction_publication_diagnostic-{uuid.uuid4().hex}.json')
    json_new(report, result)
    print(json.dumps(dict(report=str(report), **result), indent=2))


if __name__ == '__main__':
    main()
