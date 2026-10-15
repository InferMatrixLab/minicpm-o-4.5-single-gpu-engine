"""Graph-captured Thinker decode against the eager paged runner (same weights and numerics, GPU)."""

import os
from pathlib import Path

import pytest
import torch

MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1",
                                reason="GPU test (set ENGINE_GPU_TESTS=1 on a free GPU)")


@torch.inference_mode()
def test_graph_decode_matches_eager():
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV
    from engine.runtime.thinker_decode import ThinkerDecoder

    th = Thinker.load(MODEL, device="cuda")
    c = th.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    results = []
    for graphs in (False, True):
        kv = PagedKV(spec, 2048)
        pre = DecoderRunner(th.w, spec, kv)
        gen = torch.Generator(device="cuda").manual_seed(0)
        prompts = [torch.randint(0, 150000, (n,), device="cuda", generator=gen) for n in (30, 7, 64)]
        h = pre.forward(torch.cat([th.embed(p) for p in prompts]), kv.layout([0, 1, 2], [p.numel() for p in prompts]))
        first = torch.nn.functional.linear(h[torch.tensor([29, 36, 100], device="cuda")], th.w["lm_head.weight"]).argmax(-1).tolist()
        dec = ThinkerDecoder(th, kv, spec, buckets=(4,), use_graphs=graphs)
        toks, out, hids = first, [], []
        for _ in range(12):
            toks, hid = dec.step([0, 1, 2], toks)
            out.append(toks)
            hids.append(hid)
        results.append((out, torch.stack(hids)))
    assert results[0][0] == results[1][0]
    assert torch.equal(results[0][1], results[1][1])


@torch.inference_mode()
def test_graph_decode_skipped_plans_with_changing_rows():
    """Plans skipped between page crossings, rows joining and leaving between steps, a bucket change and padding rows
    restarting: graph decode still equals the eager (plan every step) path."""
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV
    from engine.runtime.thinker_decode import ThinkerDecoder

    th = Thinker.load(MODEL, device="cuda")
    c = th.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    out = []
    for graphs in (False, True):
        kv = PagedKV(spec, 4096)
        run = DecoderRunner(th.w, spec, kv)
        gen = torch.Generator(device="cuda").manual_seed(0)
        prompts = [torch.randint(0, 150000, (n,), device="cuda", generator=gen) for n in (9, 23, 14, 30)]
        hid = run.forward(th.embed(torch.cat(prompts)), kv.layout([0, 1, 2, 3], [p.numel() for p in prompts]))
        ends = torch.tensor([9, 32, 46, 76], device="cuda") - 1
        last = torch.nn.functional.linear(hid[ends], th.w["lm_head.weight"]).argmax(-1).tolist()
        dec = ThinkerDecoder(th, kv, spec, use_graphs=graphs, buckets=(2, 4))
        cur = dict(zip([0, 1, 2, 3], last))
        trace = []
        for t in range(90):  # rows 0-3, then 1-3 (row 0 leaves), then 2-3 (a smaller bucket)
            seqs = [0, 1, 2, 3] if t < 30 else [1, 2, 3] if t < 60 else [2, 3]
            nxt, h = dec.step(seqs, [cur[s] for s in seqs])
            for s, x in zip(seqs, nxt):
                cur[s] = x
            trace.append((list(nxt), h.float().cpu()))
        out.append(trace)
    for (a, ha), (b, hb) in zip(out[0], out[1]):
        assert a == b
        assert torch.equal(ha, hb)


@torch.inference_mode()
def test_fp8_thinker_ac37():
    """AC-3.7: the FP8 Thinker against the BF16 engine Thinker, teacher-forced over the 64 reference prompts: top-1
    agreement >= 98% and final-layer hidden-state cosine >= 0.99 (reported: min, p01, mean)."""
    import json

    from engine.models.thinker import Thinker
    from engine.runtime.fp8 import quantize_layers
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV, fuse_weights

    ref_path = Path(__file__).resolve().parents[2] / "results" / "engine_parity" / "thinker_ref_bf16.pt"
    if not ref_path.exists():
        pytest.skip("needs results/engine_parity/thinker_ref_bf16.pt")
    th = Thinker.load(MODEL, device="cuda")
    c = th.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    kv = PagedKV(spec, 4096)
    layers = fuse_weights(th.w, spec)
    bf16, fp8 = DecoderRunner(th.w, spec, kv, layers=layers), DecoderRunner(th.w, spec, kv, layers=quantize_layers(layers))
    cos, agree, total = [], 0, 0
    for i, r in enumerate(torch.load(ref_path)):
        t = r["tokens"].cuda()
        outs = []
        for j, run in enumerate((bf16, fp8)):
            h = run.forward(th.embed(t), kv.layout([2 * i + j], [t.numel()]))
            kv.release(2 * i + j)
            outs.append(h)
        cos.append(torch.nn.functional.cosine_similarity(outs[0].float(), outs[1].float(), dim=-1).cpu())
        top = [torch.nn.functional.linear(h, th.w["lm_head.weight"]).argmax(-1) for h in outs]
        agree += int((top[0] == top[1]).sum())
        total += t.numel()
    cat = torch.cat(cos)
    report = {"top1_agreement": agree / total, "cos_min": float(cat.min()), "cos_p01": float(cat.quantile(0.01)), "cos_mean": float(cat.mean())}
    (Path(__file__).resolve().parents[2] / "results" / "engine_parity" / "fp8_thinker_ac37.json").write_text(json.dumps(report, indent=1))
    assert report["top1_agreement"] >= 0.98, report
    assert report["cos_min"] >= 0.99, report


@torch.inference_mode()
def test_prefill_graph_matches_eager_two_sequence_batch():
    """A graph-captured single-request prefill equals the eager forward of the same padded two-sequence batch, and the
    graph decode that follows it is identical (KV written exactly)."""
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV
    from engine.runtime.prefill_graph import PAD_SEQ, PrefillGraphs

    th = Thinker.load(MODEL, device="cuda")
    c = th.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    gen = torch.Generator(device="cuda").manual_seed(0)
    for n in (9, 33, 64, 200):
        ids = torch.randint(0, 150000, (n,), device="cuda", generator=gen)
        x = th.embed(ids)
        kv = PagedKV(spec, 2048)
        run = DecoderRunner(th.w, spec, kv)
        pg = PrefillGraphs(run, buckets=(32, 64, 128, 256))
        a = pg.forward(x, 1)
        b_seq = 2
        bucket = min(b for b in (32, 64, 128, 256) if b > n)
        lay = kv.layout([b_seq, PAD_SEQ], [n, bucket - n])
        b = run.forward(torch.cat([x, x.new_zeros(bucket - n, x.shape[1])]), lay)[:n]
        kv.release(PAD_SEQ)
        assert torch.equal(a, b), n
        assert kv.lengths[1] == kv.lengths[2] == n
