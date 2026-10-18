"""GPU performance-monitoring counters (NVML GPM) of one GPU, sampled on the host clock (diagnostic).

    .venv-engine/bin/python tools/gpm_sampler.py --gpu 2 --out gpm.jsonl [--period 0.5]

Every period it takes a GPM sample and writes the metrics over the interval since the previous one: SM utilization
(the fraction of time SMs had at least one warp, averaged over SMs), SM occupancy (resident warps over the maximum),
DRAM bandwidth utilization, tensor-pipe, FP16 and FP32 utilization (percent).  Each row carries the interval's start
and end on ``time.perf_counter`` (CLOCK_MONOTONIC, the clock of the engine's stage trace).  These are device-wide
counters: they cover every process on the GPU (``tools/engine_profile.py`` records the process inventory).  Stops on
SIGTERM.
"""

from __future__ import annotations

import argparse
import json
import signal
import time

METRICS = {"sm_util": "NVML_GPM_METRIC_SM_UTIL", "sm_occupancy": "NVML_GPM_METRIC_SM_OCCUPANCY", "dram_bw_util": "NVML_GPM_METRIC_DRAM_BW_UTIL",
           "tensor_util": "NVML_GPM_METRIC_ANY_TENSOR_UTIL", "fp16_util": "NVML_GPM_METRIC_FP16_UTIL", "fp32_util": "NVML_GPM_METRIC_FP32_UTIL",
           "graphics_util": "NVML_GPM_METRIC_GRAPHICS_UTIL"}


def main() -> int:
    import pynvml as n

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--period", type=float, default=0.5)
    a = ap.parse_args()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    n.nvmlInit()
    h = n.nvmlDeviceGetHandleByIndex(a.gpu)
    if not n.nvmlGpmQueryDeviceSupport(h).isSupportedDevice:
        raise SystemExit("GPM not supported on this GPU")
    ids = {k: getattr(n, v) for k, v in METRICS.items()}
    prev, t_prev = n.nvmlGpmSampleAlloc(), None
    cur = n.nvmlGpmSampleAlloc()
    n.nvmlGpmSampleGet(h, prev)
    t_prev = time.perf_counter()
    with open(a.out, "a") as f:
        while not stop:
            time.sleep(a.period)
            n.nvmlGpmSampleGet(h, cur)
            t = time.perf_counter()
            mg = n.c_nvmlGpmMetricsGet_t()
            mg.version = n.NVML_GPM_METRICS_GET_VERSION
            mg.numMetrics = len(ids)
            mg.sample1, mg.sample2 = prev, cur
            for i, mid in enumerate(ids.values()):
                mg.metrics[i].metricId = mid
            n.nvmlGpmMetricsGet(mg)
            row = {"t0": round(t_prev, 6), "t1": round(t, 6)}
            for i, k in enumerate(ids):
                row[k] = None if mg.metrics[i].nvmlReturn != 0 else round(mg.metrics[i].value, 3)
            f.write(json.dumps(row) + "\n")
            f.flush()
            prev, cur, t_prev = cur, prev, t
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
