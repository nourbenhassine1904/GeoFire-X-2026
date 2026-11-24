"""Bounded real compact loader parity; no GPU, forward or optimizer step."""
from pathlib import Path
import importlib.util
import torch
from geofirex import v2_execution as v2

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    torch.set_num_threads(2)
    result=v2.preflight(ROOT)
    result['host_memory_total_available']=v2.host_memory()
    v2.save(ROOT/v2.REPORT/'cpu_preflight.json',result)
    spec=importlib.util.spec_from_file_location('parity',ROOT/'scripts/verify_protocol_v2_cpu.py')
    parity=importlib.util.module_from_spec(spec); spec.loader.exec_module(parity)
    parity.cpu_parity(ROOT/'reports/mission_20261111/compact_full',
        ROOT/'reports/protocol_v2_20261118/index',ROOT/'reports/gpu_acceleration_20261111_v1/index',ROOT/v2.REPORT)
    print('CPU identity and real loader parity PASS; no forward/optimizer/GPU/2021')
