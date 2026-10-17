"""SIM equivalence worker (locked evaluation environment, cwd = the pinned sglang-omni checkout):

    <eval-env>/python evals/speech/sim_equivalence_worker.py <job.json> <out.json>

For every scorer batch of the job's rows (sorted by id, ``sim_batch`` rows: references then outputs, exactly as
``scorer_worker.run_sim`` forms them), the speaker embeddings and scores of the pinned attention and of the
query-blocked one (``evals.speech.wavlm_blocked``), compared bit for bit.  A batch the pinned attention cannot hold
in memory is recorded as such and scored by the blocked attention only.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.nn.functional as F
    from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
    from s3prl.upstream.wavlm import modules

    import evals.speech.wavlm_blocked as wb

    scorer = WavLMSpeakerSimilarity(finetune_checkpoint=job["wavlm_finetune"], wavlm_base=job["wavlm_base"], device=job["device"])
    pinned = modules.MultiheadAttention.forward
    order = sorted(job["rows"], key=lambda r: r["id"])
    batch, out = int(job["sim_batch"]), []
    for i in range(0, len(order), batch):
        part = order[i:i + batch]
        refs, wavs = [r["ref_wav"] for r in part], [r["wav"] for r in part]
        rec = {"batch": i // batch, "ids": [r["id"] for r in part]}
        res = {}
        for name, fwd in (("pinned", pinned), ("blocked", wb._forward)):
            modules.MultiheadAttention.forward = fwd
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            try:
                emb = scorer.embed(refs + wavs)
            except torch.OutOfMemoryError as e:
                rec[f"{name}_oom"] = str(e).splitlines()[0][:200]
                torch.cuda.empty_cache()
                continue
            torch.cuda.synchronize()
            rec[f"{name}_s"] = time.perf_counter() - t
            rec[f"{name}_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
            res[name] = emb
            rec[f"{name}_scores"] = (F.cosine_similarity(emb[:len(part)], emb[len(part):], dim=-1).cpu() * 100.0).tolist()
            del emb
            torch.cuda.empty_cache()
        if "pinned" in res and "blocked" in res:
            a, b = res["pinned"], res["blocked"]
            rec["bitwise_equal"] = bool(torch.equal(a, b))
            rec["max_abs_embedding_diff"] = float((a - b).abs().max())
            rec["max_abs_score_diff"] = max(abs(x - y) for x, y in zip(rec["pinned_scores"], rec["blocked_scores"]))
        out.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("batch", "bitwise_equal", "max_abs_score_diff", "pinned_oom", "blocked_peak_gib")}), flush=True)
    modules.MultiheadAttention.forward = pinned
    Path(sys.argv[2]).write_text(json.dumps({"query_block": wb.QUERY_BLOCK, "version": wb.VERSION, "torch": torch.__version__,
                                             "device": torch.cuda.get_device_name(), "batches": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
