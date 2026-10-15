"""Layout (a): one process, custom paged FlashInfer Thinker and Talker sharing one allocator, Token2Wav in-process."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.spike.common import MODEL_DIR, SpikeResult, Workload, timed, token2wav_window_ms  # noqa: E402


def run(w: Workload, out: Path) -> SpikeResult:
    from engine.models.talker import Talker
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV

    th, ta = Thinker.load(MODEL_DIR, device="cuda"), Talker.load(MODEL_DIR, device="cuda")
    c, t = th.cfg, ta.cfg
    ts = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    as_ = DecoderSpec(t.num_layers, t.hidden_size, t.num_heads, t.num_kv_heads, t.head_dim, t.rope_theta, t.rms_eps, False)
    pages = 4096
    tkv, akv = PagedKV(ts, pages), PagedKV(as_, pages)
    tr, ar = DecoderRunner(th.w, ts, tkv), DecoderRunner(ta.w, as_, akv)
    r = SpikeResult("a: custom FlashInfer, one process", vars(w))
    r.kv_budget_note = (f"one explicit page pool per model sized by the engine ({pages} pages x 16 tokens each; "
                        f"Thinker {tkv.k.numel() * 2 * 2 / 2**30:.1f} GiB, Talker {akv.k.numel() * 2 * 2 / 2**30:.2f} GiB); no framework reservation")
    gen = torch.Generator(device="cuda").manual_seed(0)
    seq = [0]

    def prefill():
        ids = [seq[0] + i for i in range(w.batch)]
        seq[0] += w.batch
        lay = tkv.layout(ids, [w.prompt] * w.batch)
        tr.forward(th.embed(torch.randint(0, 150000, (w.batch * w.prompt,), device="cuda", generator=gen)), lay)
        for s in ids:
            tkv.release(s)

    r.thinker_prefill_ms = timed(prefill, w.iters)
    dec = list(range(10_000, 10_000 + w.batch))
    tr.forward(th.embed(torch.randint(0, 150000, (w.batch * w.prompt,), device="cuda", generator=gen)), tkv.layout(dec, [w.prompt] * w.batch))
    r.thinker_decode_step_ms = timed(lambda: tr.forward(th.embed(torch.randint(0, 150000, (w.batch,), device="cuda", generator=gen)),
                                                        tkv.layout(dec, [1] * w.batch)), w.iters)
    hid = [torch.randn(w.reply, 4096, device="cuda", dtype=torch.bfloat16) for _ in range(w.batch)]
    ids = torch.randint(0, 150000, (w.reply,), device="cuda")
    r.handoff_ms = timed(lambda: [ta.condition(ids, h) for h in hid], w.iters)  # in-process: the hidden states never move
    tal = list(range(20_000, 20_000 + w.batch))
    conds = torch.cat([ta.condition(ids, h) for h in hid])

    def talker_prefill():
        s = [seq[0] + 50_000 + i for i in range(w.batch)]
        seq[0] += w.batch
        ar.forward(conds, akv.layout(s, [w.reply + 2] * w.batch))
        for x in s:
            akv.release(x)

    r.talker_prefill_ms = timed(talker_prefill, w.iters)
    ar.forward(conds, akv.layout(tal, [w.reply + 2] * w.batch))
    r.talker_decode_step_ms = timed(lambda: torch.nn.functional.linear(
        ar.forward(ta.code_embed(torch.randint(0, 6561, (w.batch,), device="cuda")), akv.layout(tal, [1] * w.batch)), ta.head), w.iters)
    r.token2wav_window_ms = token2wav_window_ms(w)
    r.notes.append("thinker/talker step times above are eager; graph-captured times below")
    from engine.runtime.talker_decode import TalkerDecoder
    from engine.runtime.thinker_decode import ThinkerDecoder
    from engine.talker_sampler import PAD

    td = ThinkerDecoder(th, tkv, ts, buckets=(w.batch,))
    toks = [1] * w.batch
    r.notes.append({"thinker_graph_decode_step_ms": timed(lambda: td.step(dec, toks), w.iters)})
    adec = TalkerDecoder(ta, akv, as_, buckets=(w.batch,))
    st = {"codes": torch.zeros(w.batch, dtype=torch.int64, device="cuda"), "seeds": torch.arange(w.batch, device="cuda"),
          "steps": torch.ones(w.batch, dtype=torch.int64, device="cuda"), "history": torch.full((w.batch, 16), PAD, dtype=torch.int64, device="cuda"),
          "live": torch.ones(w.batch, dtype=torch.bool, device="cuda")}
    k = 8
    r.notes.append({"talker_graph_frame_ms": timed(lambda: adec.decode_k(tal, st, k), max(3, w.iters // 4)) / k})
    graph_path = r.thinker_prefill_ms + w.reply * r.notes[-2]["thinker_graph_decode_step_ms"] + r.handoff_ms + r.talker_prefill_ms + \
        28 * r.notes[-1]["talker_graph_frame_ms"] + r.token2wav_window_ms
    r.notes.append({"first_audio_critical_path_graphs_ms": graph_path})
    return r.finish(w)


if __name__ == "__main__":
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    run(Workload(batch=b), Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"results/l20x/spike/a_b{b}.json")).save(
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"results/l20x/spike/a_b{b}.json"))
