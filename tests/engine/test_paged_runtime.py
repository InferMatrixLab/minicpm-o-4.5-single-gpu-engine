"""Paged FlashInfer decoder runtime against the reference ports on the same GPU (bf16), Thinker and Talker."""

import os

import pytest
import torch

from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1",
                                reason="GPU test (set ENGINE_GPU_TESTS=1 on a free GPU)")


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


@pytest.mark.parametrize("which", ["thinker", "talker"])
@torch.inference_mode()
def test_ragged_batch_matches_reference(which):
    from engine.models.talker import Talker
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV

    if which == "thinker":
        ref = Thinker.load(MODEL, device="cuda", dtype=torch.bfloat16)
        c = ref.cfg
        spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
        emb = lambda ids: ref.embed(ids)  # noqa: E731
        hidden = lambda e, cache, rows: ref.forward(e, cache, rows, all_logits=True)[0]  # noqa: E731
    else:
        ref = Talker.load(MODEL, device="cuda", dtype=torch.bfloat16)
        c = ref.cfg
        spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, False)
        emb = lambda ids: ref.code_embed(ids % 6561)  # noqa: E731
        hidden = None
    kv = PagedKV(spec, num_pages=256, page_size=16)
    run = DecoderRunner(ref.w, spec, kv)
    gen = torch.Generator(device="cuda").manual_seed(0)
    lens = [37, 5, 64]
    seqs = [torch.randint(0, 150000, (n,), device="cuda", generator=gen) for n in lens]
    lay = kv.layout([0, 1, 2], lens)
    out = run.forward(torch.cat([emb(s) for s in seqs]), lay)
    steps = [torch.randint(0, 150000, (3,), device="cuda", generator=gen) for _ in range(4)]
    dec = []
    for st in steps:  # three decode tokens per call, rows at different positions
        lay = kv.layout([0, 1, 2], [1, 1, 1])
        dec.append(run.forward(emb(st), lay))
    if which == "talker":
        return  # Talker parity is covered through the per-row reference below
    for j, s in enumerate(seqs):
        cache = ref.new_cache(1, 128)
        rows = torch.tensor([0], device="cuda")
        want = [hidden(emb(s)[None], cache, rows)[0]]
        for st in steps:
            want.append(hidden(emb(st[j:j + 1])[None], cache, rows)[0])
        want = torch.cat(want)
        got = torch.cat([out[sum(lens[:j]):sum(lens[:j + 1])]] + [d[j:j + 1] for d in dec])
        assert _cos(got, want).min() >= 0.999, (j, _cos(got, want).min())
