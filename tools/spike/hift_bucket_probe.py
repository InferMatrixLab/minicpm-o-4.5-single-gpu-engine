"""HiFT graph row buckets (GPU, diagnostic): time of a whole ``HiFTGraphs.__call__`` (noise, copies, replays) for 4-32
rows with the current buckets (1/2/4/8/16, larger calls sliced) and with extended ones, per mel shape (16 ramp, 50
first, 58 steady), and the GPU memory each captured graph adds.

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/hift_bucket_probe.py [--tf32]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
EXTENDED = (1, 2, 4, 8, 12, 16, 20, 24, 32)


@torch.inference_mode()
def main():
    from engine.models.token2wav import SOURCE_CACHE, Token2Wav
    from engine.runtime.hift_graph import FRAMES, ROWS, HiFTGraphs

    ap = argparse.ArgumentParser()
    ap.add_argument("--tf32", action="store_true")
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=True)
    hift = t2w.hift
    sets = {"current": ROWS, "extended": EXTENDED}
    graphs = {}
    for name, rows in sets.items():
        g = HiFTGraphs(hift, rows=rows, final_frames={}, decode_tf32=a.tf32)
        mem = {}
        for F in FRAMES:
            for r in rows:
                torch.cuda.synchronize()
                m0 = torch.cuda.memory_reserved()
                g._capture(r, F)
                torch.cuda.synchronize()
                mem[(r, F)] = (torch.cuda.memory_reserved() - m0) / 2**20
        graphs[name] = g
        print(f"{name} buckets {rows}: graph pool growth MiB per (rows, frames):", {f"{r}x{F}": round(v) for (r, F), v in mem.items() if v}, flush=True)
    gen = torch.Generator(device="cuda").manual_seed(0)
    for F, src in FRAMES.items():
        for n in (4, 8, 12, 16, 17, 20, 24, 28, 32):
            mel = torch.randn(n, hift.conv_pre.in_channels, F, device="cuda", generator=gen) - 4
            cache = torch.randn(n, 1, src, device="cuda", generator=gen) * 0.01
            seeds = list(range(n))
            res = {}
            for name, g in graphs.items():
                for _ in range(3):
                    g(mel, cache, seeds)
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(10):
                    out = g(mel, cache, seeds)
                torch.cuda.synchronize()
                res[name] = ((time.perf_counter() - t) * 100, out)
            (tc, oc), (te, oe) = res["current"], res["extended"]
            rel = float((oe[0] - oc[0]).norm() / oc[0].norm())
            print(f"F={F:2d} rows={n:2d}: current {tc:6.2f} ms ({tc / n:.3f}/row)  extended {te:6.2f} ms ({te / n:.3f}/row)  "
                  f"speedup {tc / te:.2f}  waveform rel-L2 extended vs current {rel:.1e}", flush=True)
    del SOURCE_CACHE


if __name__ == "__main__":
    main()
