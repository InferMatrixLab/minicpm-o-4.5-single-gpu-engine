"""Batched Token2Wav window time, eager vs CUDA-graph estimator, FP32 and FP16 flow (GPU, diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/t2w_batch_profile.py [out.json]

B streams with distinct seed-tts reference voices decode steady windows together; reports ms per batched window call
(flow + HiFT), and graph-vs-eager waveform agreement.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")


def main():
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(1)[:32])
    out = {}
    for f16 in (False, True):
        t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=f16)
        prompts = [t2w.prepare(cr.ref_audio_bytes()) for cr in reqs]
        eager, graph = BatchedToken2Wav(t2w), BatchedToken2Wav(t2w, use_graphs=True)
        gen = np.random.default_rng(0)
        toks = [[int(x) for x in gen.integers(0, 6561, 25 * 6)] for _ in reqs]
        res = {}
        for B in (1, 8, 32):
            per = {}
            for name, bt in (("eager", eager), ("graph", graph)):
                sts = [t2w.new_stream(p, noise_seed=i) for i, p in enumerate(prompts[:B])]
                chs = [StreamChunker() for _ in range(B)]
                wins = [ch.push(t) for ch, t in zip(chs, toks[:B])]
                times, wavs = [], [[] for _ in range(B)]
                for k in range(len(wins[0])):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    o = bt.decode([(sts[i], wins[i][k].tokens, False) for i in range(B)])
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t) * 1000)
                    for i in range(B):
                        wavs[i].append(o[i])
                per[name] = {"ms_per_call": times, "steady_ms": float(np.median(times[2:])), "wav": [np.concatenate(w) for w in wavs]}
            agree = max(float(np.linalg.norm(a - b) / np.linalg.norm(b)) for a, b in zip(per["graph"]["wav"], per["eager"]["wav"]))
            res[B] = {"eager_steady_ms": per["eager"]["steady_ms"], "graph_steady_ms": per["graph"]["steady_ms"],
                      "graph_first_calls_ms": per["graph"]["ms_per_call"][:2], "graph_vs_eager_wav_rel_l2_max": agree,
                      "audio_s_per_s_graph": B * 1.0 / (per["graph"]["steady_ms"] / 1000)}
            print("fp16" if f16 else "fp32", B, json.dumps(res[B]), flush=True)
        out["fp16" if f16 else "fp32"] = res
        del t2w, eager, graph, prompts
        torch.cuda.empty_cache()
    Path(sys.argv[1] if len(sys.argv) > 1 else "results/l20x/spike/t2w_batch_profile.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
