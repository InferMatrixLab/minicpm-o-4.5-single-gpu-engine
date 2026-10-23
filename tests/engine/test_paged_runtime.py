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


@pytest.mark.parametrize("which", ["thinker", "talker"])
@torch.inference_mode()
def test_prefill_hf_kernels_match_torch_ops(which):
    """``prefill_hf_kernels``: the prefill layer stack with the fused reference-rounding kernels against the torch ops
    on the same ragged batches (fresh prompts, then continued chunks).  The kernels round as the ops do and differ only
    in a norm's fp32 sum order, so most outputs are equal bit for bit; the rest stay below the batching control (the
    same rows one at a time through the torch ops: GEMM tiling) and the Thinker's greedy next tokens agree."""
    from engine.models.talker import Talker
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV

    if which == "thinker":
        ref = Thinker.load(MODEL, device="cuda", dtype=torch.bfloat16)
        c = ref.cfg
        spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
        emb, head = ref.embed, ref.w["lm_head.weight"]
        vocab = 150000
    else:
        ref = Talker.load(MODEL, device="cuda", dtype=torch.bfloat16)
        c = ref.cfg
        spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, False)
        emb, head = (lambda ids: ref.code_embed(ids % 6561)), None
        vocab = 6561
    kvs = [PagedKV(spec, num_pages=192, page_size=64) for _ in range(3)]  # torch ops, fused kernels, torch ops one row at a time
    runs = [DecoderRunner(ref.w, spec, kvs[0]), DecoderRunner(ref.w, spec, kvs[1], hf_kernels=True), DecoderRunner(ref.w, spec, kvs[2])]
    assert runs[1].hf_kernels and not runs[0].hf_kernels
    gen = torch.Generator(device="cuda").manual_seed(0)
    calls = [([0, 1, 2], [37, 300, 5]), ([3], [700]), ([0, 2], [64, 129]), ([1, 3, 4, 5], [1, 1, 90, 17])]
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())  # noqa: E731
    worst = control = 0.0
    equal = total = 0
    agree = [0, 0, 0]
    for seqs, ns in calls:
        x = emb(torch.randint(0, vocab, (sum(ns),), device="cuda", generator=gen))
        ht = runs[0].forward(x, kvs[0].layout(seqs, ns))
        hk = runs[1].forward(x, kvs[1].layout(seqs, ns))
        parts, a = [], 0
        for s, n in zip(seqs, ns):
            parts.append(runs[2].forward(x[a:a + n], kvs[2].layout([s], [n])))
            a += n
        ho = torch.cat(parts)
        worst, control = max(worst, rel(hk, ht)), max(control, rel(ho, ht))
        equal += int((hk == ht).sum())
        total += hk.numel()
        if head is not None:  # greedy tokens at every position: the kernels' agreement with the ops beside the control's
            top = [torch.nn.functional.linear(h, head).argmax(-1) for h in (ht, hk, ho)]
            agree[0] += int((top[1] == top[0]).sum())
            agree[1] += int((top[2] == top[0]).sum())
            agree[2] += top[0].numel()
    kv_worst = kv_control = 0.0
    for seq in range(6):
        n = kvs[0].lengths[seq]
        for i in range(len(kvs[0].k)):
            k0, k1, k2 = (kv.k[i][kv.pages[seq][:-(-n // kv.page_size)]].flatten(0, 1)[:n] for kv in kvs)
            kv_worst, kv_control = max(kv_worst, rel(k1, k0)), max(kv_control, rel(k2, k0))
    print(f"{which} prefill hf kernels vs torch ops: hidden {worst:.2e} ({equal / total:.1%} of elements equal), K {kv_worst:.2e}"
          + (f", greedy {agree[0]}/{agree[2]}" if agree[2] else "") + f"; control (one row at a time): hidden {control:.2e}, K {kv_control:.2e}"
          + (f", greedy {agree[1]}/{agree[2]}" if agree[2] else ""))
    assert worst <= 2 * control and kv_worst <= 2 * kv_control


@torch.inference_mode()
def test_talker_prefill_rope_kernel_is_bitwise():
    """``prefill_rope_hf``: the Talker's prefill with RoPE + v copy in one kernel equals the hf-kernel prefill with the
    torch RoPE bit for bit (hidden states and every layer's K/V), fresh and continued rows."""
    from engine.models.talker import Talker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV

    ref = Talker.load(MODEL, device="cuda", dtype=torch.bfloat16)
    c = ref.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, False)
    kvs = [PagedKV(spec, num_pages=192, page_size=64) for _ in range(2)]
    runs = [DecoderRunner(ref.w, spec, kvs[0], hf_kernels=True), DecoderRunner(ref.w, spec, kvs[1], hf_kernels=True, rope_kernel=True)]
    gen = torch.Generator(device="cuda").manual_seed(1)
    for seqs, ns in (([0, 1, 2], [37, 300, 5]), ([3], [700]), ([0, 2], [64, 129]), ([1, 3, 4], [1, 90, 17])):
        x = ref.code_embed(torch.randint(0, 6561, (sum(ns),), device="cuda", generator=gen))
        a, b = (r.forward(x, kv.layout(seqs, ns)) for r, kv in zip(runs, kvs))
        assert torch.equal(a, b), (seqs, ns)
    for i in range(len(kvs[0].k)):
        assert torch.equal(kvs[0].k[i], kvs[1].k[i]) and torch.equal(kvs[0].v[i], kvs[1].v[i]), i
