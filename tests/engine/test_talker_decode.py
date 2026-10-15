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
