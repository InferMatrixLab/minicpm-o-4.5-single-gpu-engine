"""Token2Wav per-window cost split (flow encoder, DiT Euler solve, HiFT) at batch 1, fp32 and fp16 flow (GPU, diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/t2w_profile.py [out.json]

Times the steady-state windows of one stream (after the attention caches reach their trimmed length), each component
synchronised separately, plus the DiT estimator on stacked equal-length batches (2B rows: B requests times CFG) to show
how far batching amortises it.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH


def ms(fn, n=5):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000 / n, out


@torch.inference_mode()
def profile(float16: bool) -> dict:
    from engine.models.token2wav import Token2Wav
    from engine.token2wav_chunker import StreamChunker

    t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=float16)
    prompt = t2w.prepare(MODEL / "assets" / "HT_ref_audio.wav")
    st = t2w.new_stream(prompt)
    ch = StreamChunker()
    toks = [int(x) for x in np.random.default_rng(0).integers(0, 6561, 25 * 8)]
    wins = ch.push(toks)
    for w in wins[:5]:
        t2w.decode(st, w.tokens, w.last)
    w = wins[5]
    flow, dt = t2w.flow, torch.float16 if float16 else torch.float32
    tok = torch.tensor([w.tokens], dtype=torch.int32, device="cuda")
    c = st.flow_cache
    res = {"float16": float16, "estimator_att_len": int(c["estimator_att_cache"].shape[4]), "prompt_mel": int(prompt.mels.shape[1])}
    with torch.amp.autocast("cuda", dtype=dt):
        res["window_total_ms"], _ = ms(lambda: t2w.decode(_clone(st), w.tokens, False))  # includes the per-window cache copy
        spk = flow.spk_embed_affine_layer(torch.nn.functional.normalize(prompt.spk, dim=1))

        def enc():
            x = flow.input_embedding(tok)
            h, _, _ = flow.encoder.forward_chunk(xs=x, last_chunk=False, cnn_cache=c["conformer_cnn_cache"], att_cache=c["conformer_att_cache"])
            return flow.encoder_proj(h)

        res["encoder_ms"], h = ms(enc)

        def dit():
            return flow.decoder.forward_chunk(mu=h.transpose(1, 2).contiguous(), spks=spk, cond=torch.zeros_like(h).transpose(1, 2).contiguous(),
                                              n_timesteps=10, temperature=1.0, cnn_cache=c["estimator_cnn_cache"], att_cache=c["estimator_att_cache"])

        res["dit_solve_ms"], (mel, _, _) = ms(dit)
        hc = st.hift_cache
        melc = torch.concat([hc["mel"], mel], dim=2)
        res["hift_ms"], _ = ms(lambda: t2w.hift(melc, hc["source"]))
        res["mel_frames"] = int(mel.shape[2])
        # one estimator call on B stacked equal-length requests (x 2 for CFG)
        est = flow.decoder.estimator
        att = c["estimator_att_cache"][0]  # [depth, 2, nh, T, 2c]
        cnn = c["estimator_cnn_cache"][0]
        t_in = torch.zeros(2, device="cuda", dtype=dt)
        mu = h.transpose(1, 2).contiguous()
        x = torch.randn_like(mu)
        scale = {}
        for b in (1, 8, 32):
            xb, mb = x.repeat(2 * b, 1, 1), mu.repeat(2 * b, 1, 1)
            sb, tb = spk.repeat(2 * b, 1), t_in.repeat(b)
            ab, cb = att.repeat(1, b, 1, 1, 1).contiguous(), cnn.repeat(1, b, 1, 1).contiguous()
            est.cnn_cache_buffer = torch.zeros(cb.shape, dtype=cb.dtype, device="cuda")
            est.att_cache_buffer = torch.zeros(ab.shape[:3] + (ab.shape[3] + xb.shape[2],) + ab.shape[4:], dtype=ab.dtype, device="cuda")
            step_ms, _ = ms(lambda: est.forward_chunk(x=xb, mu=mb, t=tb, spks=sb, cond=torch.zeros_like(mb), cnn_cache=cb, att_cache=ab))
            scale[b] = {"estimator_step_ms": step_ms, "per_request_window_ms": step_ms * 10 / b}
        res["estimator_batch_scaling"] = scale
    return res


def _clone(st):
    from engine.models.token2wav import StreamState, clone_tree

    return StreamState(st.prompt, clone_tree(st.flow_cache), clone_tree(st.hift_cache), st.chunks)


def main():
    out = {"fp32": profile(False), "fp16": profile(True)}
    for k, v in out.items():
        print(k, json.dumps(v), flush=True)
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/l20x/spike/t2w_profile.json")
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
