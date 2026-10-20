"""Talker chunk-prefill CUDA graphs (``ChunkPrefillGraphs``, ``talker_prefill_graphs``) against the eager prefill (GPU).

Real Talker weights; four sequences with prior contexts of 1-300 tokens (inside a page and across page boundaries),
then chunk prefills of 1-4 rows of 9-100 code-embedding tokens, appended to every sequence several times.  Two paged
KV pools receive identical histories; one appends by the eager ``DecoderRunner.forward`` of the same rows, the other by
the graphs.  Compared: the final-norm hidden states of every new token and every layer's appended K/V.  The graphs pad
each call to a token bucket with a scratch sequence, so GEMM tiling may differ from the eager call; the control is the
difference the engine already has between prefilling the same rows together and one at a time.  The pools diverge as
the calls repeat (each attends to its own history), for the control as for the graphs.  The bound: the graph
difference of hidden states, K/V and next-frame logits is at most the batching control (plus 1e-3).  Measured: most
calls are bitwise equal to the eager call; the rest differ like the control.

The first frames of one call are sampled in one ``step`` over all rows: rows are independent (per-row counter RNG,
row-wise warpers), so tokens and states must equal the per-row calls bitwise.
"""

import os

import pytest
import torch

from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1",
                                reason="GPU test (set ENGINE_GPU_TESTS=1 on a free GPU)")
PRIORS = {1: 1, 2: 5, 3: 130, 4: 300}
CALLS = [([1], [12]), ([2, 3], [13, 11]), ([1, 2, 3, 4], [12, 14, 13, 9]), ([4], [60]), ([2, 3], [100, 20]), ([3, 1, 4], [10, 12, 11]),
         ([1, 2, 3, 4], [13, 13, 13, 13])]


def _talker():
    from engine.models.talker import Talker
    from engine.runtime.paged import DecoderSpec

    talker = Talker.load(MODEL, device="cuda")
    t = talker.cfg
    return talker, DecoderSpec(t.num_layers, t.hidden_size, t.num_heads, t.num_kv_heads, t.head_dim, t.rope_theta, t.rms_eps, False)


def _kv_of(kv, seq):
    """Every layer's K and V of ``seq`` in order, [layers, tokens, ...]."""
    n = kv.lengths[seq]
    pages = kv.pages[seq][:-(-n // kv.page_size)]
    k = torch.stack([kv.k[i][pages].flatten(0, 1)[:n] for i in range(len(kv.k))])
    v = torch.stack([kv.v[i][pages].flatten(0, 1)[:n] for i in range(len(kv.v))])
    return k, v


@torch.inference_mode()
def test_chunk_prefill_graphs_match_eager():
    from engine.runtime.paged import DecoderRunner, PagedKV
    from engine.runtime.prefill_graph import ChunkPrefillGraphs

    talker, spec = _talker()
    kvs = [PagedKV(spec, 64, 128, device="cuda") for _ in range(3)]  # eager together, graphs, eager one row at a time
    runs = [DecoderRunner(talker.w, spec, kv) for kv in kvs]
    graphs = ChunkPrefillGraphs(runs[1])
    gen = torch.Generator(device="cuda").manual_seed(0)
    emb = lambda n: talker.code_embed(torch.randint(0, 6561, (n,), device="cuda", generator=gen))  # noqa: E731
    for seq, n in PRIORS.items():
        x = emb(n)
        for run, kv in zip(runs, kvs):
            run.forward(x, kv.layout([seq], [n]))
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())  # noqa: E731
    worst, control, lg_worst, lg_control = 0.0, 0.0, 0.0, 0.0
    for _ in range(2):
        for seqs, ns in CALLS:
            x = emb(sum(ns))
            lay = kvs[0].layout(seqs, ns)
            he = runs[0].forward(x, lay)
            hg, last = graphs.forward(x, seqs, ns)
            assert torch.equal(last, lay.last_token_index)
            parts, a = [], 0
            for s, n in zip(seqs, ns):  # control: the same rows one at a time
                parts.append(runs[2].forward(x[a:a + n], kvs[2].layout([s], [n])))
                a += n
            ho = torch.cat(parts)
            worst = max(worst, rel(hg, he))
            control = max(control, rel(ho, he))
            le, lg, lo = (torch.nn.functional.linear(h[last], talker.head).float() for h in (he, hg, ho))
            lg_worst, lg_control = max(lg_worst, rel(lg, le)), max(lg_control, rel(lo, le))
    assert graphs.counts["graph"] == 2 * len(CALLS)
    kv_worst, kv_control = 0.0, 0.0
    for seq in PRIORS:
        (ke, ve), (kg, vg), (ko, vo) = (_kv_of(kv, seq) for kv in kvs)
        assert ke.shape == kg.shape
        kv_worst = max(kv_worst, rel(kg, ke), rel(vg, ve))
        kv_control = max(kv_control, rel(ko, ke), rel(vo, ve))
    print(f"chunk prefill graphs vs eager: hidden max rel L2 {worst:.2e}, K/V {kv_worst:.2e}, logits {lg_worst:.2e}; "
          f"control (rows together vs one at a time): hidden {control:.2e}, K/V {kv_control:.2e}, logits {lg_control:.2e}")
    assert worst <= control + 1e-3 and kv_worst <= kv_control + 1e-3 and lg_worst <= lg_control + 1e-3


@torch.inference_mode()
def test_batched_first_frame_sampling_is_exact():
    from engine.talker_sampler import PAD, DecodeState, TalkerSampling, step

    cfg = TalkerSampling(min_new_tokens=0, max_new_tokens=1 << 30)
    gen = torch.Generator(device="cuda").manual_seed(1)
    n, V, eos = 7, 6562, 6561
    for trial in range(20):
        logits = torch.randn(n, V, device="cuda", generator=gen) * 4
        seeds = torch.randint(0, 1 << 31, (n,), device="cuda", generator=gen)
        steps = torch.randint(0, 50, (n,), device="cuda", generator=gen) if trial % 2 else torch.zeros(n, dtype=torch.int64, device="cuda")
        hist = torch.where(torch.rand(n, cfg.window, device="cuda", generator=gen) < 0.5,
                           torch.randint(0, V, (n, cfg.window), device="cuda", generator=gen), torch.full((n, cfg.window), PAD, device="cuda"))
        st = DecodeState(seeds.clone(), steps.clone(), hist.clone(), torch.zeros(n, dtype=torch.bool, device="cuda"), torch.ones(n, dtype=torch.bool, device="cuda"))
        tb = step(st, logits, eos, cfg)
        for i in range(n):
            s1 = DecodeState(seeds[i:i + 1].clone(), steps[i:i + 1].clone(), hist[i:i + 1].clone(), torch.zeros(1, dtype=torch.bool, device="cuda"),
                             torch.ones(1, dtype=torch.bool, device="cuda"))
            t1 = step(s1, logits[i:i + 1], eos, cfg)
            assert int(t1[0]) == int(tb[i])
            assert torch.equal(s1.steps[0], st.steps[i]) and torch.equal(s1.history[0], st.history[i])
