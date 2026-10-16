"""RSS of one Token2Wav feature process over repeated real voices, per component (GPU for the S3 part, diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/feature_memory_profile.py <component> [jobs]

component: full | campplus | s3 | frontend.  Each run is a fresh process: 20 warm jobs, then ``jobs`` measured jobs
over 200 evaluation-set voices; prints RSS every 50 jobs (MB) and the delta after warm-up.
"""

import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
MODEL = "/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"


def main():
    comp, jobs = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    import librosa
    import s3tokenizer
    import soundfile as sf
    import torch
    import torchaudio.compliance.kaldi as kaldi

    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.runtime import t2w_worker as W

    W._feat_init(MODEL, 4, comp in ("full", "s3"))
    rows = load_seed_tts_rows()
    refs = [cr.ref_audio_bytes() for cr in canonical_requests(rows, make_id_sets(rows).eval(32)[:200])]

    def job(ref):
        if comp == "full":
            return W._feat_job(ref)
        wav16, _ = librosa.load(io.BytesIO(ref), sr=16000, mono=True)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f, torch.inference_mode():
            sf.write(f.name, wav16, 16000)
            audio = s3tokenizer.load_audio(f.name, sr=16000)
            if comp == "campplus":
                feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
                feat = feat - feat.mean(dim=0, keepdim=True)
                W._FEAT.spk_model.run(None, {W._FEAT.spk_model.get_inputs()[0].name: feat.unsqueeze(0).numpy()})
            elif comp == "s3":
                mels, lens = s3tokenizer.padding([s3tokenizer.log_mel_spectrogram(audio)])
                W._FEAT.audio_tokenizer.quantize(mels.cuda(), lens.cuda())
            elif comp == "frontend":
                import torchaudio
                from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

                a, sr = torchaudio.load(f.name, backend="soundfile")
                W._RESAMPLE.setdefault(sr, torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000))
                mel_spectrogram(W._RESAMPLE[sr](a.mean(0, keepdim=True)))
        W._trim_heap()

    if os.environ.get("WARM") == "1":
        import time as _t

        t = _t.time()
        W._warm_shape_caches()
        print(f"shape warm-up {(_t.time() - t):.1f} s", flush=True)
    for r in refs[:20]:
        job(r)
    r0 = W._rss_kib(os.getpid())
    for i in range(jobs):
        job(refs[i % len(refs)])
        if (i + 1) % 50 == 0:
            r = W._rss_kib(os.getpid())
            print(f"{comp} {i + 1} rss_mb {r / 1024:.0f} delta_mb {(r - r0) / 1024:+.1f}", flush=True)


if __name__ == "__main__":
    main()
