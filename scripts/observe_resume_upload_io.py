"""Read only owned publisher I/O counters; this is not server acknowledgement."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path

base=Path('reports/kaggle_remaining_20261108_v1')
execution=base/'resume-01/execution'
plan=json.loads((base/'plan.json').read_text())
class Counters(ctypes.Structure):
    _fields_=[(name,ctypes.c_ulonglong) for name in ('ReadOperationCount','WriteOperationCount','OtherOperationCount',
        'ReadTransferCount','WriteTransferCount','OtherTransferCount')]
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
k.GetProcessIoCounters.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters)];k.GetProcessIoCounters.restype=wintypes.BOOL
k.CloseHandle.argtypes=[wintypes.HANDLE]
for spec in plan['lots']:
    monitor=execution/spec['tag']/'publish-monitor'
    if not (monitor/'memory.jsonl').exists() or (monitor/'supervisor.json').exists():continue
    with (monitor/'memory.jsonl').open('rb') as stream:
        stream.seek(0,2);stream.seek(max(0,stream.tell()-16384));lines=stream.read().splitlines()
    sample=json.loads(lines[-1])
    receipt=json.loads((Path(spec['staging'])/'staging-receipt.json').read_text())
    processes=sample['tree']['processes']
    if not processes:continue
    handle=k.OpenProcess(0x1000,False,processes[-1]['pid'])
    if not handle:continue
    try:
        counters=Counters()
        if k.GetProcessIoCounters(handle,ctypes.byref(counters)):
            estimate=max(0,min(receipt['payload_bytes'],counters.ReadTransferCount-receipt['payload_bytes']))
            print(json.dumps(dict(lot=spec['number'],elapsed_seconds=round(sample['seconds']),
                approximate_upload_source_read_percent=round(100*estimate/receipt['payload_bytes'],1),
                acknowledged_upload_bytes=None,warning='Local read estimate after ZIP hashing; not HTTP acknowledgement')))
    finally:k.CloseHandle(handle)
