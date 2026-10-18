"""Graph K-frame Talker decode against the eager per-frame path (same weights, same sampler, GPU)."""

import os
from pathlib import Path

import pytest
import torch

MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1",
                                reason="GPU test (set ENGINE_GPU_TESTS=1 on a free GPU)")


@pytest.mark.parametrize("page_size", [16, 128])
@torch.inference_mode()
def test_graph_k_step_matches_eager(page_size):
    """Re-planning only at page crossings (graph-side last-page increments otherwise) equals the eager path; 40 frames
    cross 16-token pages several times and stay inside one 128-token page."""
    from engine.models.talker import Talker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV
    from engine.runtime.talker_decode import TalkerDecoder
    from engine.talker_sampler import PAD

    ta = Talker.load(MODEL, device="cuda")
    c = ta.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, False)
    results = []
    for graphs in (False, True):
        kv = PagedKV(spec, 2048, page_size)
        pre = DecoderRunner(ta.w, spec, kv)
        gen = torch.Generator(device="cuda").manual_seed(0)
        conds = [ta.condition(torch.randint(0, 150000, (n,), device="cuda", generator=gen), torch.randn(n, 4096, device="cuda", generator=gen))
                 for n in (12, 30, 7)]
        hid = pre.forward(torch.cat(conds), kv.layout([0, 1, 2], [x.shape[0] for x in conds]))
        logits = torch.nn.functional.linear(hid[torch.tensor([11, 41, 48], device="cuda")], ta.head).float()
        first = logits.argmax(-1)
        dec = TalkerDecoder(ta, kv, spec, use_graphs=graphs, buckets=(4,))
        st = {"codes": first, "seeds": torch.tensor([1, 2, 3], device="cuda"), "steps": torch.ones(3, dtype=torch.int64, device="cuda"),
              "history": torch.full((3, 16), PAD, dtype=torch.int64, device="cuda"), "live": torch.ones(3, dtype=torch.bool, device="cuda")}
        st["history"][:, -1] = first
        results.append(torch.cat([dec.decode_k([0, 1, 2], st, 8) for _ in range(5)]))
    assert torch.equal(results[0], results[1])


@pytest.mark.parametrize("page_size", [16, 128])
@pytest.mark.parametrize("k", [4, 8])
@torch.inference_mode()
def test_whole_k_graph_matches_frame_graphs(page_size, k):
    """Whole-K capture (``whole_k``: every frame of a block in one graph, each frame planned through its own wrapper
    before the replay) against the per-frame graph and the eager path: identical codes, sampler state and KV for blocks
    that cross pages, rows that reach EOS inside a block at different frames (max_new_tokens forces it), a padded
    bucket, and a row cancelled between blocks."""
    from engine.models.talker import Talker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV
    from engine.runtime.talker_decode import TalkerDecoder
    from engine.talker_sampler import PAD, TalkerSampling

    ta = Talker.load(MODEL, device="cuda")
    c = ta.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, False)
    cfg = TalkerSampling(min_new_tokens=0, max_new_tokens=27)  # rows starting at different steps hit EOS mid-block
    results = []
    for mode in ("eager", "frames", "whole_k"):
        kv = PagedKV(spec, 2048, page_size)
        pre = DecoderRunner(ta.w, spec, kv)
        gen = torch.Generator(device="cuda").manual_seed(0)
        conds = [ta.condition(torch.randint(0, 150000, (n,), device="cuda", generator=gen), torch.randn(n, 4096, device="cuda", generator=gen))
                 for n in (12, 30, 7)]
        hid = pre.forward(torch.cat(conds), kv.layout([0, 1, 2], [x.shape[0] for x in conds]))
        logits = torch.nn.functional.linear(hid[torch.tensor([11, 41, 48], device="cuda")], ta.head).float()
        first = logits.argmax(-1)
        dec = TalkerDecoder(ta, kv, spec, cfg, use_graphs=mode != "eager", buckets=(4,), whole_k=mode == "whole_k")
        if mode != "eager":
            dec.capture_all(ks=(k,))
        st = {"codes": first, "seeds": torch.tensor([1, 2, 3], device="cuda"), "steps": torch.tensor([1, 5, 9], device="cuda"),
              "history": torch.full((3, 16), PAD, dtype=torch.int64, device="cuda"), "live": torch.ones(3, dtype=torch.bool, device="cuda")}
        st["history"][:, -1] = first
        out = [dec.decode_k([0, 1, 2], st, k) for _ in range(2)]
        sub = {name: v[[0, 2]].clone() for name, v in st.items()}  # row 1 cancelled between blocks
        out += [torch.nn.functional.pad(dec.decode_k([0, 2], sub, k), (0, 1), value=-7) for _ in range(4)]
        kvs = [torch.cat([kv.k[i][kv.pages[s][: -(-kv.lengths[s] // page_size)]].flatten(0, 1)[: kv.lengths[s]] for s in (0, 2)]) for i in (0, 19)]
        results.append((torch.cat(out), {name: v.clone() for name, v in sub.items()}, kvs))
    codes = [r[0] for r in results]
    assert (codes[0][:, :2] != PAD).any() and (codes[0] == PAD).any()  # both live frames and EOS-frozen rows occur
    for r in results[1:]:
        assert torch.equal(r[0], results[0][0])
        assert all(torch.equal(r[1][name], results[0][1][name]) for name in r[1])
        assert all(torch.equal(a, b) for a, b in zip(r[2], results[0][2]))
