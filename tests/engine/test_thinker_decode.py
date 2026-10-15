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
