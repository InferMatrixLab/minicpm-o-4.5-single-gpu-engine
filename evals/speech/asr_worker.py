"""Offline fixed-batch ASR worker: runs in the dedicated ASR environment.

    <asr-env>/python evals/speech/asr_worker.py <job.json> <out.json>

Job: ``rows`` [{id, wav}], ``model`` ("qwen3-asr" | "whisper"), ``model_path``,
``batch_size``.  Rows are transcribed in a stable order (sorted by id) in
microbatches of exactly ``batch_size``; the final batch is padded by repeating
its last item and the padded outputs are discarded.  Decoding is greedy, the
model runs in BF16, and the language is forced to English.  The output records
every executed batch's size and the effective settings.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ASR_SR = 16000


def _load_16k(path: str) -> np.ndarray:
    import librosa

    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != ASR_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=ASR_SR, res_type="soxr_hq")
    return x.astype(np.float32)


def fixed_batches(ids: list[str], b: int) -> list[list[str]]:
    """Exactly ``b`` items per batch; the last batch repeats its final item."""
    if b < 1:
        raise ValueError("batch size must be >= 1")
    out = []
    for i in range(0, len(ids), b):
        part = ids[i : i + b]
        out.append(part + [part[-1]] * (b - len(part)))
    return out


def run_qwen3(rows: dict, model_path: str, b: int) -> tuple[dict, dict]:
    from qwen_asr import Qwen3ASRModel

    m = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=b, max_new_tokens=512)
    gc = m.model.generation_config
    gc.do_sample = False
    gc.num_beams = 1
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    order = sorted(rows)
    texts, sizes = {}, []
    for batch in fixed_batches(order, b):
        res = m.transcribe(audio=[(_load_16k(rows[i]), ASR_SR) for i in batch], language="English")
        sizes.append(len(batch))
        for i, r in zip(batch, res):
            texts.setdefault(i, r.text)  # padded duplicates are discarded
    settings = {"backend": "qwen_asr transformers", "dtype": str(m.model.dtype), "device": str(m.model.device), "do_sample": gc.do_sample,
                "num_beams": gc.num_beams, "language": "English", "max_new_tokens": 512, "resample": "librosa soxr_hq to 16 kHz"}
    return texts, {"batch_sizes": sizes, "settings": settings}


def run_whisper(rows: dict, model_path: str, b: int) -> tuple[dict, dict]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    order = sorted(rows)
    texts, sizes = {}, []
    for batch in fixed_batches(order, b):
        feats = proc([_load_16k(rows[i]) for i in batch], sampling_rate=ASR_SR, return_tensors="pt").input_features.to("cuda:0", torch.bfloat16)
        with torch.inference_mode():
            ids = model.generate(feats, language="en", task="transcribe", do_sample=False, num_beams=1, max_new_tokens=440)
        dec = proc.batch_decode(ids, skip_special_tokens=True)
        sizes.append(len(batch))
        for i, t in zip(batch, dec):
            texts.setdefault(i, t.strip())
    settings = {"backend": "transformers whisper", "dtype": str(model.dtype), "device": "cuda:0", "do_sample": False, "num_beams": 1,
                "language": "en", "task": "transcribe", "resample": "librosa soxr_hq to 16 kHz"}
    return texts, {"batch_sizes": sizes, "settings": settings}


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    rows = {r["id"]: r["wav"] for r in job["rows"]}
    t0 = time.time()
    fn = run_qwen3 if job["model"] == "qwen3-asr" else run_whisper
    texts, ev = fn(rows, job["model_path"], int(job["batch_size"]))
    missing = sorted(set(rows) - set(texts))
    if missing:
        raise SystemExit(f"{len(missing)} rows not transcribed")
    ev["batch_size"] = int(job["batch_size"])
    ev["n_rows"] = len(rows)
    ev["n_batches"] = len(ev["batch_sizes"])
    ev["padded_items"] = sum(ev["batch_sizes"]) - len(rows)
    ev["elapsed_s"] = time.time() - t0
    ev["torch"] = torch.__version__
    Path(sys.argv[2]).write_text(json.dumps({"texts": texts, "evidence": ev}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
