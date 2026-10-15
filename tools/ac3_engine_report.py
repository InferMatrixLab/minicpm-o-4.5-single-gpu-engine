"""AC-3.1/3.2/3.4 metrics of the engine against the HF references (GPU, engine env; diagnostic report).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/ac3_engine_report.py OUT.json
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")


@torch.inference_mode()
def main():
    from engine.models.talker import Talker
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV

    out = {}
    th = Thinker.load(MODEL, device="cuda")
    c = th.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    kv = PagedKV(spec, 4096)
    run = DecoderRunner(th.w, spec, kv)
    tref = torch.load("results/engine_parity/thinker_ref_bf16.pt")
    cos, agree, total, identical, divergences = [], 0, 0, 0, []
    for i, r in enumerate(tref):
        t = r["tokens"].cuda()
        h = run.forward(th.embed(t), kv.layout([i], [t.numel()]))
        kv.release(i)
        cos.append(F.cosine_similarity(h.float(), r["hidden"].cuda().float(), dim=-1).cpu())
        agree += int((F.linear(h, th.w["lm_head.weight"]).argmax(-1).cpu() == r["top1"]).sum())
        total += t.numel()
        p, seq = r["prompt_len"], 10_000 + i
        hh = run.forward(th.embed(t[:p]), kv.layout([seq], [p]))
        gen, ref_gen = [], r["tokens"][p:].tolist()
        nxt = int(F.linear(hh[-1:], th.w["lm_head.weight"]).argmax())
        while len(gen) < len(ref_gen):
            gen.append(nxt)
            if nxt in (151645, 151643):
                break
            hh = run.forward(th.embed(torch.tensor([nxt], device="cuda")), kv.layout([seq], [1]))
            nxt = int(F.linear(hh[-1:], th.w["lm_head.weight"]).argmax())
        kv.release(seq)
        if gen == ref_gen:
            identical += 1
        else:
            j = next((k for k in range(min(len(gen), len(ref_gen))) if gen[k] != ref_gen[k]), min(len(gen), len(ref_gen)))
            divergences.append({"prompt": i, "position": j, "hf_margin": float(r["margin"][p + j - 1])})
    cc = torch.cat(cos)
    out["thinker"] = {"hidden_cos": {"min": float(cc.min()), "p01": float(cc.quantile(0.01)), "mean": float(cc.mean())}, "top1_agreement": agree / total,
                      "greedy_identical": identical / len(tref), "divergences": divergences}
    print("thinker", json.dumps({k: v for k, v in out["thinker"].items() if k != "divergences"}), flush=True)
    del run, kv
    ta = Talker.load(MODEL, device="cuda")
    aref = [r for r in torch.load("results/engine_parity/talker_ref_bf16.pt") if r["codes"] is not None]
    a_agree = a_total = 0
    rels = []
    for r in aref:
        cache = ta.new_cache(1, r["condition"].shape[0] + r["codes"].numel() + 4)
        row = torch.tensor([0], device="cuda")
        lg = [ta.forward(r["condition"].cuda().to(ta.dtype)[None], cache, row)]
        for code in r["codes"][:-1].cuda():
            lg.append(ta.forward(ta.code_embed(code.view(1, 1)), cache, row))
        lg = torch.cat(lg).cpu()
        n = min(lg.shape[0], r["logits"].shape[0])
        a_agree += int((lg[:n].argmax(-1) == r["logits"][:n].argmax(-1)).sum())
        a_total += n
        rels.append(float((lg[:n] - r["logits"][:n]).norm() / r["logits"][:n].norm()))
    out["talker"] = {"top1_agreement": a_agree / a_total, "logit_rel_l2_mean": float(np.mean(rels))}
    print("talker", json.dumps(out["talker"]), flush=True)
    from stepaudio2 import Token2wav

    from engine.models.token2wav import Token2Wav
    from engine.token2wav_chunker import StreamChunker

    import librosa
    import soundfile as sf

    wav16, _ = librosa.load(str(MODEL / "assets" / "HT_ref_audio.wav"), sr=16000, mono=True)
    tokens = [int(x) for x in np.random.default_rng(0).integers(0, 6561, 137)]
    ch = StreamChunker()
    windows = [(w.tokens, w.last) for w in ch.push(tokens)] + [(lambda w: (w.tokens, w.last))(ch.finish())]

    def ref_run():
        torch.manual_seed(7)
        ref = Token2wav(str(MODEL / "assets" / "token2wav"))
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, wav16, 16000)
            ref.cache = None
            ref.stream_cache, ref.hift_cache_dict = ref.set_stream_cache(f.name)
        return np.concatenate([ref.stream(t, prompt_wav=None, last_chunk=l, return_waveform=True)[0] for t, l in windows])

    a, b = ref_run(), ref_run()
    ours = Token2Wav(MODEL / "assets" / "token2wav", noise_seed=7)
    st = ours.new_stream(ours.prepare(MODEL / "assets" / "HT_ref_audio.wav"))
    e = np.concatenate([ours.decode(st, t, l) for t, l in windows])
    rl = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))  # noqa: E731
    out["token2wav"] = {"ref_vs_ref_rel_l2": rl(b, a), "engine_vs_ref_rel_l2": rl(e, a), "samples": int(len(a))}
    print("token2wav", json.dumps(out["token2wav"]), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
