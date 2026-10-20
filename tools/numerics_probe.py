"""Where does the engine Thinker's bf16 deviation from HF eager come from?  (diagnostic, GPU)

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/numerics_probe.py results/engine_parity/thinker_ref_bf16.pt [N]

Variants, each teacher-forced over the HF tokens and compared per token (cosine of last-layer hidden states):
  ref_sdpa   - engine reference port (torch SDPA attention, HF-style norm/RoPE rounding)
  ref_eager  - same, attention as HF eager: bf16 QK^T, fp32 softmax cast to bf16, bf16 PV
  paged_hf   - paged FlashInfer attention with HF-style norm/RoPE rounding
  paged_fused- paged FlashInfer attention with FlashInfer fused RMSNorm/RoPE
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH


def eager_attention(q, k, v, attn_mask=None, enable_gqa=True):
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    w = torch.matmul(q, k.transpose(2, 3)) * (q.shape[-1] ** -0.5)
    w = w.masked_fill(~attn_mask, float("-inf")) if attn_mask is not None else w
    w = torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(w, v)


@torch.inference_mode()
def main():
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV

    ref = torch.load(sys.argv[1])[: int(sys.argv[2]) if len(sys.argv) > 2 else 8]
    th = Thinker.load(MODEL, device="cuda")
    c = th.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    out = {}
    for name in ("ref_sdpa", "ref_eager", "paged_hf", "paged_fused"):
        cos_all = []
        if name.startswith("paged"):
            kv = PagedKV(spec, 2048)
            run = DecoderRunner(th.w, spec, kv, numerics="hf" if name == "paged_hf" else "fused")
        for i, r in enumerate(ref):
            toks = r["tokens"].cuda()
            if name.startswith("paged"):
                h = run.forward(th.embed(toks), kv.layout([i], [toks.numel()]))
                kv.release(i)
            else:
                orig = F.scaled_dot_product_attention
                if name == "ref_eager":
                    F.scaled_dot_product_attention = eager_attention
                try:
                    cache = th.new_cache(1, toks.numel() + 2)
                    h = th.forward(th.embed(toks)[None], cache, torch.tensor([0], device="cuda"), all_logits=True)[0][0]
                finally:
                    F.scaled_dot_product_attention = orig
            cos_all.append(F.cosine_similarity(h.float(), r["hidden"].cuda().float(), dim=-1).cpu())
        cat = torch.cat(cos_all)
        worst = [(i, int(x.argmin()), float(x.min())) for i, x in enumerate(cos_all)]
        out[name] = {"min": float(cat.min()), "p01": float(cat.quantile(0.01)), "mean": float(cat.mean()), "worst_per_prompt": worst}
        print(name, json.dumps({k: v for k, v in out[name].items() if k != "worst_per_prompt"}), flush=True)
    Path("results/engine_parity/numerics_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
