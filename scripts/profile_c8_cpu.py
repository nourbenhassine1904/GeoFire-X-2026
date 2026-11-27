"""Tiny synthetic CPU compute profile. No actual dataset pixels or GPU query."""
import json
from pathlib import Path
import statistics
import time
import torch
from geofirex.benchmark_models import StackedUNet
from geofirex.v2_execution import host_memory
from geofirex.protocol_v2 import save

if __name__=='__main__':
    memory=host_memory()
    if memory[1]<512*1024**2:raise RuntimeError('Not enough RAM for even this bounded fixture')
    torch.set_num_threads(2);torch.manual_seed(42)
    model=StackedUNet(8,3).eval();x=torch.zeros(16,3,8,64,64)
    times=[]
    with torch.inference_mode():
        for _ in range(2):model(x)
        for _ in range(5):
            t=time.perf_counter();model(x);times.append(time.perf_counter()-t)
    result=dict(scope='synthetic compute only, no real inference or throughput qualification',
        batches=5,warmup_batches=2,batch_shape=list(x.shape),threads=2,
        batch_seconds=times,median_batch_seconds=statistics.median(times),
        validation_batches=1931,extrapolated_compute_seconds=1931*statistics.median(times),
        excludes=['ZIP loading','prediction writing','SHA validation','AP sorting','event calibration','transfers'],
        torch=str(torch.__version__),host_ram_total_available=memory,gpu_used=False,
        training_executed=False,real_inference_executed=False,test_pixels_opened=0)
    save(Path('reports/c8_propagation_20261121_v1/cpu_synthetic_profile.json'),result)
    print(json.dumps(result,indent=2))
