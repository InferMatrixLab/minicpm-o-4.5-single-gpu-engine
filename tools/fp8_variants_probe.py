"""AC-3.7 probe: FP8 Thinker variants (which projections in FP8) against the BF16 engine Thinker (GPU, diagnostic)."""
import sys, json, torch
sys.path.insert(0, "/workspace/cgzheng/cgzheng/qwen3_omni")
from pathlib import Path
from engine.models.thinker import Thinker
from engine.runtime.fp8 import quantize_layers
from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV, fuse_weights
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
th = Thinker.load(MODEL, device="cuda"); c = th.cfg
spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
kv = PagedKV(spec, 4096); layers = fuse_weights(th.w, spec)
ref = torch.load("results/engine_parity/thinker_ref_bf16.pt")[:24]
base = DecoderRunner(th.w, spec, kv, layers=layers)
def variant(names, skip_layers=()):
    q = quantize_layers(layers, names)
    return [L if i in skip_layers else q[i] for i, L in enumerate(layers)] if skip_layers else q
n = c.num_layers
variants = {"all": variant(("qkv", "o", "gate_up", "down")), "no_down": variant(("qkv", "o", "gate_up")),
            "attn_only": variant(("qkv", "o")), "mlp_only": variant(("gate_up", "down")),
            "all_skip_first_last2": variant(("qkv", "o", "gate_up", "down"), skip_layers=(0, 1, n - 2, n - 1)),
            "no_down_skip_first_last2": variant(("qkv", "o", "gate_up"), skip_layers=(0, 1, n - 2, n - 1))}
with torch.inference_mode():
    for name, L in variants.items():
        run = DecoderRunner(th.w, spec, kv, layers=L)
        cos, ag, tot = [], 0, 0
        for i, r in enumerate(ref):
            t = r["tokens"].cuda(); outs = []
            for j, rr in enumerate((base, run)):
                h = rr.forward(th.embed(t), kv.layout([2 * i + j], [t.numel()])); kv.release(2 * i + j); outs.append(h)
            cos.append(torch.nn.functional.cosine_similarity(outs[0].float(), outs[1].float(), dim=-1).cpu())
            tp = [torch.nn.functional.linear(h, th.w["lm_head.weight"]).argmax(-1) for h in outs]
            ag += int((tp[0] == tp[1]).sum()); tot += t.numel()
        cat = torch.cat(cos)
        print(name, json.dumps({"top1": round(ag / tot, 4), "cos_min": round(float(cat.min()), 4), "cos_p01": round(float(cat.quantile(0.01)), 4), "cos_mean": round(float(cat.mean()), 5)}), flush=True)
        del run, L
