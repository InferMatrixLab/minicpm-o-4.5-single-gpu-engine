"""Cost split of Token2Wav.prepare (the per-request reference-voice setup) on one GPU (diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/prepare_profile.py [N]
"""

import io
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH


def main():
    import librosa
    import s3tokenizer
    import soundfile as sf
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import SILENCE_TOKEN, Token2Wav, strict_fp32

    t2w = Token2Wav(MODEL / "assets" / "token2wav")
    ref = t2w.ref
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(1)[: int(sys.argv[1]) if len(sys.argv) > 1 else 6])
    acc = {}

    def tick(name, t):
        torch.cuda.synchronize()
        now = time.perf_counter()
        acc.setdefault(name, []).append((now - t) * 1000)
        return now

    for i, cr in enumerate(reqs):
        with torch.inference_mode(), strict_fp32():
            t = time.perf_counter()
            wav16, _ = librosa.load(io.BytesIO(cr.ref_audio_bytes()), sr=16000, mono=True)
            f = tempfile.NamedTemporaryFile(suffix=".wav")
            sf.write(f.name, wav16, 16000)
            t = tick("librosa_load_resample_write", t)
            audio = s3tokenizer.load_audio(f.name, sr=16000)
            mels = s3tokenizer.log_mel_spectrogram(audio)
            mels, mels_lens = s3tokenizer.padding([mels])
            t = tick("s3_logmel", t)
            toks, toks_len = ref.audio_tokenizer.quantize(mels.cuda(), mels_lens.cuda())
            t = tick("s3_quantize_gpu", t)
            feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            t = tick("campplus_fbank", t)
            spk = torch.tensor(ref.spk_model.run(None, {ref.spk_model.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()})[0], device="cuda")
            t = tick("campplus_onnx_cpu", t)
            a24, sr = torchaudio.load(f.name, backend="soundfile")
            a24 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)(a24.mean(dim=0, keepdim=True))
            pm = mel_spectrogram(a24).transpose(1, 2).squeeze(0)[None].cuda()
            pm = torch.nn.functional.pad(pm, (0, 0, 0, toks.shape[1] * t2w.flow.up_rate - pm.shape[1]), mode="replicate")
            t = tick("prompt_mel_24k", t)
            pad = torch.full((1, 3), SILENCE_TOKEN, device="cuda", dtype=toks.dtype)
            t2w.flow.setup_cache(torch.cat([toks, pad], dim=1), pm, spk, n_timesteps=10)
            t = tick("flow_setup_cache", t)
            acc.setdefault("prompt_tokens", []).append(int(toks.shape[1]))
    for k, v in acc.items():
        vv = v[1:] if len(v) > 1 else v
        print(f"{k:32s} mean {sum(vv) / len(vv):8.1f}  (first {v[0]:.1f})")


if __name__ == "__main__":
    main()
