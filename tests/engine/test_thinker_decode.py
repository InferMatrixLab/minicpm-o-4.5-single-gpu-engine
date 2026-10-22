"""Graph-captured Thinker decode against the eager paged runner (same weights and numerics, GPU)."""

import os
from pathlib import Path

import pytest
import torch

from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH
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


@pytest.mark.parametrize("hf_kernels", [False, True], ids=["torch_ops", "prefill_hf_kernels"])
@torch.inference_mode()
def test_prefill_graph_matches_eager_two_sequence_batch(hf_kernels):
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
        run = DecoderRunner(th.w, spec, kv, hf_kernels=hf_kernels)
        pg = PrefillGraphs(run, buckets=(32, 64, 128, 256))
        a = pg.forward(x, 1)
        b_seq = 2
        bucket = min(b for b in (32, 64, 128, 256) if b > n)
        lay = kv.layout([b_seq, PAD_SEQ], [n, bucket - n])
        b = run.forward(torch.cat([x, x.new_zeros(bucket - n, x.shape[1])]), lay)[:n]
        kv.release(PAD_SEQ)
        assert torch.equal(a, b), n
        assert kv.lengths[1] == kv.lengths[2] == n


@torch.inference_mode()
def test_fused_kernels_match_torch_step():
    """The fused reference-rounding kernels against the torch elementwise step: same greedy tokens over 24 steps and
    hidden states within bf16 noise (rare one-ulp norm differences, from a norm's fp32 sum order, accumulate)."""
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV
    from engine.runtime.thinker_decode import ThinkerDecoder

    th = Thinker.load(MODEL, device="cuda")
    c = th.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    runs = []
    for fused in (False, True):
        kv = PagedKV(spec, 2048)
        pre = DecoderRunner(th.w, spec, kv)

        gen = torch.Generator(device="cuda").manual_seed(1)
        prompts = [torch.randint(0, 150000, (n,), device="cuda", generator=gen) for n in (17, 40)]
        h = pre.forward(torch.cat([th.embed(p) for p in prompts]), kv.layout([0, 1], [p.numel() for p in prompts]))
        toks = torch.nn.functional.linear(h[torch.tensor([16, 56], device="cuda")], th.w["lm_head.weight"]).argmax(-1).tolist()
        dec = ThinkerDecoder(th, kv, spec, buckets=(2,), fused=fused)
        out, hids = [], []
        for _ in range(24):
            toks, hid = dec.step([0, 1], toks)
            out.append(toks)
            hids.append(hid)
        runs.append((out, torch.stack(hids).float()))

    assert runs[0][0] == runs[1][0]
    a, b = runs[0][1], runs[1][1]
    cos = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1), dim=-1)
    assert float(cos.min()) > 0.999, float(cos.min())  # one-ulp norm differences accumulate through the KV (0.9998 at 24 steps)


@torch.inference_mode()
def test_more_rows_than_buckets_and_pad_release():
    """More rows than the largest bucket decode as slices; releasing the scratch padding rows when idle returns every
    page they held (no pages remain once the live rows are released too)."""
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV
    from engine.runtime.thinker_decode import ThinkerDecoder

    th = Thinker.load(MODEL, device="cuda")
    c = th.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    kv = PagedKV(spec, 2048)
    pre = DecoderRunner(th.w, spec, kv)
    gen = torch.Generator(device="cuda").manual_seed(2)
    prompts = [torch.randint(0, 150000, (5 + i,), device="cuda", generator=gen) for i in range(6)]
    h = pre.forward(torch.cat([th.embed(p) for p in prompts]), kv.layout(list(range(6)), [p.numel() for p in prompts]))
    ends = torch.cumsum(torch.tensor([p.numel() for p in prompts], device="cuda"), 0) - 1
    first = torch.nn.functional.linear(h[ends], th.w["lm_head.weight"]).argmax(-1).tolist()
    dec = ThinkerDecoder(th, kv, spec, buckets=(1, 2, 4))
    toks, _ = dec.step(list(range(6)), first)  # 6 rows > bucket 4: slices of 4 and 2
    assert len(toks) == 6
    dec.release_pads()
    for s in range(6):
        kv.release(s)
    assert kv.used_pages == 0


@pytest.mark.parametrize("hf_kernels", [False, True], ids=["torch_ops", "prefill_hf_kernels"])
@torch.inference_mode()
def test_thinker_chunk_prefill_graphs_match_eager(hf_kernels):
    """``thinker_chunk_prefill_graphs``: multi-row Thinker prompt prefills (fresh prompts and continued chunks of 9-700
    tokens, 1-4 rows) through ``ChunkPrefillGraphs``.  As for the single-request prefill graphs, a replay equals the
    eager forward of the same padded batch (the rows plus the scratch sequence) bit for bit, hidden states and appended
    K/V; against the unpadded eager call only GEMM tiling over the padded rows differs, and the greedy next tokens must
    agree wherever the eager top-1/top-2 margin exceeds 0.25 (random-token logits are flat; bf16 resolves 0.125-0.25
    near the top, AC-3.1's divergence rule).  The padding effect is printed beside the batching control (the same rows one at a time)."""
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV
    from engine.runtime.prefill_graph import PAD_SEQ, ChunkPrefillGraphs

    th = Thinker.load(MODEL, device="cuda")
    c = th.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    kvs = [PagedKV(spec, 160, 64, device="cuda") for _ in range(4)]  # graphs, eager padded, eager together, eager one at a time
    runs = [DecoderRunner(th.w, spec, kv, hf_kernels=hf_kernels) for kv in kvs]
    buckets = (64, 128, 256, 512, 1024)
    graphs = ChunkPrefillGraphs(runs[0], rows=(1, 2, 3, 4), buckets=buckets, lane="thinker", pad_pages=16)
    gen = torch.Generator(device="cuda").manual_seed(0)
    emb = lambda n: th.embed(torch.randint(0, 150000, (n,), device="cuda", generator=gen))  # noqa: E731
    calls = [([1], [40]), ([2, 3], [130, 61]), ([4, 5, 6, 7], [90, 120, 9, 300]), ([1, 2], [200, 33]), ([3], [700]), ([5, 6, 7], [64, 64, 64])]
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())  # noqa: E731
    pad_effect = control = 0.0
    head = th.w["lm_head.weight"]
    flips = []  # top-1/top-2 margins where the padded and unpadded greedy tokens differ
    for seqs, ns in calls:
        x, total = emb(sum(ns)), sum(ns)
        hg, last = graphs.forward(x, seqs, ns)
        hg = hg.clone()
        B = next(b for b in buckets if b > total)
        lay = kvs[1].layout(list(seqs) + [PAD_SEQ], list(ns) + [B - total])
        hp = runs[1].forward(torch.cat([x, x.new_zeros(B - total, x.shape[1])]), lay)[:total]
        kvs[1].release(PAD_SEQ)
        assert torch.equal(hg, hp), (seqs, ns)  # the replay is the eager padded batch
        lay = kvs[2].layout(seqs, ns)
        he = runs[2].forward(x, lay)
        assert torch.equal(last, lay.last_token_index)
        parts, a = [], 0
        for s, n in zip(seqs, ns):
            parts.append(runs[3].forward(x[a:a + n], kvs[3].layout([s], [n])))
            a += n
        ho = torch.cat(parts)
        pad_effect, control = max(pad_effect, rel(hg, he)), max(control, rel(ho, he))
        lg, le = torch.nn.functional.linear(hg[last], head).float(), torch.nn.functional.linear(he[last], head).float()
        top2 = le.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]  # random-token logits are flat: GEMM tiling flips only near-ties (bf16 resolves 0.125-0.25 there)
        flip = lg.argmax(-1) != le.argmax(-1)
        flips += [round(float(m), 3) for m in margin[flip]]
        assert not (flip & (margin > 0.25)).any(), (seqs, ns, margin[flip])
    assert graphs.counts["graph"] == len(calls)
    for seq in range(1, 8):
        n = kvs[0].lengths[seq]
        assert kvs[1].lengths[seq] == n
        for i in range(len(kvs[0].k)):
            ga, pa = (kv.k[i][kv.pages[seq][:-(-n // kv.page_size)]].flatten(0, 1)[:n] for kv in kvs[:2])
            gv, pv = (kv.v[i][kv.pages[seq][:-(-n // kv.page_size)]].flatten(0, 1)[:n] for kv in kvs[:2])
            assert torch.equal(ga, pa) and torch.equal(gv, pv), (seq, i)
    print(f"thinker chunk prefill graphs: bitwise equal to the eager padded batch; padding effect on hidden {pad_effect:.2e}, "
          f"batching control {control:.2e}; greedy flips at margins {flips}")
