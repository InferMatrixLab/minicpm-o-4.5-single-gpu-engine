"""Fresh-process regression of the feature processes' memory over the robustness scenario sequence (no server).

    python tools/spike/feature_scenario_rss.py OUT.json [--sustain-jobs 12000] [--processes 4]

The worker's own pool (``ProcessPoolExecutor`` with ``_feat_init`` and ``_feat_warm``, as ``_worker_main`` builds it),
then the feature jobs of the robustness scenarios in their order: mixed (16 jobs, the default voice), cancel (8 warm-up
rows), EOS boundary (192 tuning rows), fallback (one 48 s voice), overload (128 tuning rows at once), and the sustained
load (the 32 evaluation rows cycled, 32 jobs in flight).  Every input is loaded before the first sample, so the series
measures the processes, not this script.  Per PID: VmRSS, RssAnon and RssFile after each phase and every 500 sustain
jobs.  The reference is the first sample at or after ``ref_jobs`` sustain jobs (the jobs the four processes run in the
soak's first 60 s: 32,455 requests / 1800 s * 60 s, about 1,080); the envelope is +-1% of the reference for every
later sample, per PID and in total (the robustness criterion).
"""

from __future__ import annotations

import argparse
import base64
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

MODEL = "/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"


def _mem(pid: int) -> dict | None:
    try:
        d = {}
        for line in open(f"/proc/{pid}/status"):
            k = line.split(":")[0]
            if k in ("VmRSS", "RssAnon", "RssFile"):
                d[k] = int(line.split()[1])
        return d
    except OSError:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sustain-jobs", type=int, default=12000)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--ref-jobs", type=int, default=1080)
    a = ap.parse_args(argv)

    import engine.runtime.t2w_worker as W
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from robustness import long_reference_wav

    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    voice = open(f"{MODEL}/assets/HT_ref_audio.wav", "rb").read()
    phases = [
        ("mixed", [voice] * 16, 16),
        ("cancel", [r.ref_audio_bytes() for r in canonical_requests(rows, sets.warmup(8))], 8),
        ("eos_boundary", [r.ref_audio_bytes() for r in canonical_requests(rows, sets.tuning(32)[:192])], 32),
        ("fallback", [base64.b64decode(long_reference_wav(48.0).split(",", 1)[1])], 1),
        ("overload", [r.ref_audio_bytes() for r in canonical_requests(rows, sets.tuning(32)[:128])], 128),
    ]
    sustain = [r.ref_audio_bytes() for r in canonical_requests(rows, sets.eval(32))]

    for k, v in (("MALLOC_ARENA_MAX", "2"), ("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "0"), ("MALLOC_MMAP_THRESHOLD_", "131072")):
        os.environ.setdefault(k, v)  # as the worker sets them for its feature processes
    if os.environ.get("FEATURE_ALLOCATOR", "jemalloc") == "jemalloc":  # the worker's default (FEATURE_ALLOCATOR=glibc: without)
        os.environ["LD_PRELOAD"] = W.JEMALLOC
    pool = ProcessPoolExecutor(a.processes, mp_context=mp.get_context("spawn"), initializer=W._feat_init, initargs=(MODEL, 8, True))
    for f in [pool.submit(W._feat_warm, voice) for _ in range(a.processes)]:
        f.result()
    pids = sorted(pool._processes)
    samples = []

    def sample(label: str, jobs: int = 0) -> None:
        samples.append({"label": label, "sustain_jobs": jobs, "t": time.monotonic(), "mem": {str(p): _mem(p) for p in pids}})
        tot = sum((m or {}).get("VmRSS", 0) for m in samples[-1]["mem"].values()) / 1024
        print(f"{label:>14} {jobs:6d} total {tot:8.1f} MB  anon/file per PID",
              [(round(m["RssAnon"] / 1024), round(m["RssFile"] / 1024)) for m in samples[-1]["mem"].values() if m], flush=True)

    sample("warm")
    for name, refs, inflight in phases:
        for k in range(0, len(refs), inflight):
            wait([pool.submit(W._feat_job, r) for r in refs[k:k + inflight]])
        sample(name)
    done, futs = 0, set()
    while done < a.sustain_jobs:
        while len(futs) < 32 and done + len(futs) < a.sustain_jobs:
            futs.add(pool.submit(W._feat_job, sustain[(done + len(futs)) % len(sustain)]))
        fin, futs = wait(futs, return_when="FIRST_COMPLETED")
        for f in fin:
            f.result()
        before = done
        done += len(fin)
        if done // 500 != before // 500:
            sample("sustain", done)
    sample("end", done)
    alive = sorted(pool._processes)
    pool.shutdown()

    ref = next(s for s in samples if s["label"] in ("sustain", "end") and s["sustain_jobs"] >= a.ref_jobs)
    later = [s for s in samples if s["t"] > ref["t"]]

    def worst(key: str) -> dict:
        out = {}
        for p in [str(x) for x in pids] + ["total"]:
            def val(s):
                return sum(m[key] for m in s["mem"].values()) if p == "total" else s["mem"][p][key]
            r0 = val(ref)
            out[p] = max((abs(val(s) - r0) / r0 for s in later), default=0.0)
        return out

    res = {"processes": a.processes, "pids": pids, "pids_at_end": alive, "ref_jobs": a.ref_jobs, "reference_sample": ref["label"],
           "reference_sustain_jobs": ref["sustain_jobs"], "max_rel_dev_vmrss": worst("VmRSS"), "max_rel_dev_anon": worst("RssAnon"),
           "samples": samples}
    res["within_1pct"] = all(v <= 0.01 for v in res["max_rel_dev_vmrss"].values()) and alive == pids
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=1))
    return 0 if res["within_1pct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
