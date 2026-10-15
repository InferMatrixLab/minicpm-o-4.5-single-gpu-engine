"""32 deterministic mixed multimodal samples for prompt-assembly and encoder parity.

Images: the checkpoint's example images at several sizes and crops (different slice grids); audio: example
recordings cut to 1-40 s (single and multi-segment Whisper chunks); video: 1 fps frames of a Daily-Omni clip with its
audio track (the frames-plus-audio form).  ``build`` writes plain arrays so both environments read identical inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
DAILY = sorted(Path("/workspace/models/hub/datasets--liarliar--Daily-Omni/snapshots").glob("*/Videos"))


def _images():
    from PIL import Image

    out = []
    for name in ("fossil.png", "highway.png"):
        im = Image.open(MODEL / "assets" / name).convert("RGB")
        out.append(np.asarray(im))
        for size in ((224, 224), (448, 1344), (1600, 900), (333, 777)):
            out.append(np.asarray(im.resize(size)))
        w, h = im.size
        out.append(np.asarray(im.crop((0, 0, w // 2, h // 3))))
    return out  # 12


def _audio():
    import librosa

    files = sorted((MODEL / "assets" / "audio_cases").glob("*.wav")) + [MODEL / "assets" / "HT_ref_audio.wav", MODEL / "assets" / "bajie.wav"]
    out = []
    for i, f in enumerate(files):
        w, _ = librosa.load(str(f), sr=16000, mono=True)
        for secs in ((1.0, 7.5, 31.0)[i % 3], 40.0)[: 1 + (i % 2)]:
            out.append(w[: int(secs * 16000)].astype(np.float32))
    return out


def _video(n_frames=8):
    import av

    clip = sorted((DAILY[0] / sorted(p.name for p in DAILY[0].iterdir())[0]).glob("*_video.mp4"))[0]
    frames = []
    with av.open(str(clip)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i % 30 == 0:
                frames.append(fr.to_ndarray(format="rgb24"))
            if len(frames) >= n_frames:
                break
    import librosa

    wav, _ = librosa.load(str(clip).replace("_video.mp4", "_audio.wav"), sr=16000, mono=True)
    return frames, wav[: 16000 * n_frames].astype(np.float32)


def build(out: Path) -> list[dict]:
    """Each sample: {"parts": [("image", array) | ("audio", array) | ("text", str)], "tts": bool}."""
    imgs, auds = _images(), _audio()
    frames, vwav = _video()
    s = []
    for i in range(10):
        s.append({"parts": [("image", imgs[i]), ("text", "What is shown in the image?")], "tts": False})
    for i in range(10):
        s.append({"parts": [("audio", auds[i % len(auds)]), ("text", "What is said in the recording?")], "tts": False})
    for i in range(6):
        s.append({"parts": [("image", imgs[(i + 3) % len(imgs)]), ("audio", auds[(i + 5) % len(auds)]), ("text", "Relate the picture to the audio.")],
                  "tts": False})
    for n in (1, 2, 4, 8, 3, 6):  # frames interleaved with their audio second, the Daily-Omni form
        parts = []
        for j in range(n):
            parts += [("image", frames[j]), ("audio", vwav[j * 16000:(j + 1) * 16000])]
        s.append({"parts": parts + [("text", "What happens in the video?")], "tts": False})
    assert len(s) == 32
    import pickle

    out.write_bytes(pickle.dumps(s))
    return s


def to_msgs(sample: dict) -> list[dict]:
    from PIL import Image

    return [{"role": "user", "content": [Image.fromarray(v) if k == "image" else v for k, v in sample["parts"]]}]
