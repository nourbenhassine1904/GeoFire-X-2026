"""Read existing monitor and OS I/O counters only. No network or mutation."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path

campaign=Path(__file__).resolve().parent
timeline=campaign/'lot-001-upload-monitor-04/memory.jsonl'
with timeline.open('rb') as file:
    file.seek(0,2);file.seek(max(0,file.tell()-65536))
    rows=file.read().splitlines()
    sample=json.loads(rows[-1])
result=dict(elapsed_seconds=round(sample['seconds']),available_gib=round(sample['system']['available']/1024**3,2),
    process_tree_mib=round(sample['tree']['rss_bytes']/1024**2,1))
class Counters(ctypes.Structure):
    _fields_=[(name,ctypes.c_ulonglong) for name in ('ReadOperationCount','WriteOperationCount','OtherOperationCount',
        'ReadTransferCount','WriteTransferCount','OtherTransferCount')]
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
k.GetProcessIoCounters.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters)];k.GetProcessIoCounters.restype=wintypes.BOOL
k.CloseHandle.argtypes=[wintypes.HANDLE]
processes=sample['tree']['processes']
if processes:
    handle=k.OpenProcess(0x1000,False,processes[-1]['pid'])
    if handle:
        try:
            counters=Counters()
            if k.GetProcessIoCounters(handle,ctypes.byref(counters)):
                estimate=max(0,min(4829918684,counters.ReadTransferCount-4829918684))
                result.update(approximate_payload_read_percent=round(100*estimate/4829918684,1),
                    warning='Local read estimate after whole-ZIP hashing; NOT acknowledged HTTP upload bytes')
        finally:k.CloseHandle(handle)
finished=campaign/'lot-001-create-attempt-04/result.json'
result['publication_status']=json.loads(finished.read_text())['status'] if finished.exists() else 'IN_PROGRESS'
print(json.dumps(result))
