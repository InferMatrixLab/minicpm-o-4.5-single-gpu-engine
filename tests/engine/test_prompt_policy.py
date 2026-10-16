"""Served media prompts follow the request's processor policy and are never silently truncated (B18).

The native Daily-Omni client sends ``mm_processor_kwargs = {max_slice_nums: 1, use_image_id: False}``; the server keeps
and validates it, the backend passes it to the checkpoint processor, and a prompt that does not fit the engine's
context is an HTTP 400 instead of a processor-side cut at 8192 tokens.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.server import BadRequest, parse_request  # noqa: E402

NATIVE_POLICY = {"max_slice_nums": 1, "use_image_id": False}


def _body(**kw):
    img = "data:image/png;base64," + __import__("base64").b64encode((Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/"
                                                                         "503e754207c94da6bb26850b4469f367c9ea3582/assets/fossil.png")).read_bytes()).decode()
    return {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}}, {"type": "text", "text": "What is it?"}]}],
            "modalities": ["text"], **kw}


def test_parser_keeps_and_validates_the_processor_policy():
    assert parse_request(_body(mm_processor_kwargs=NATIVE_POLICY)).processor_kwargs == NATIVE_POLICY
    assert parse_request(_body()).processor_kwargs == {}
    for bad in ({"max_slice_nums": 0}, {"max_slice_nums": "1"}, {"use_image_id": "no"}, {"fps": 2}, "x"):
        with pytest.raises(BadRequest):
            parse_request(_body(mm_processor_kwargs=bad))


class _Frontend:
    def __init__(self, n_tokens):
        self.n, self.calls = n_tokens, []

    def build_prompt(self, msgs, **kw):
        self.calls.append(kw)
        return {"input_ids": torch.zeros(1, self.n, dtype=torch.long)}

    def embed(self, inputs):
        return torch.zeros(inputs["input_ids"].shape[1], 4)


def _backend(n_tokens, max_len=8192):
    import threading

    from engine.runtime.backend import EngineBackend

    b = EngineBackend.__new__(EngineBackend)
    b.frontend, b._lock, b.tdec = _Frontend(n_tokens), threading.Lock(), SimpleNamespace(max_len=max_len)
    return b


def test_backend_passes_the_policy_and_rejects_prompts_beyond_the_context():
    from engine.runtime.backend import EngineBackend

    b = _backend(300)
    ids, emb = EngineBackend._prompt(b, parse_request(_body(mm_processor_kwargs=NATIVE_POLICY)))
    kw = b.frontend.calls[-1]
    assert kw["max_slice_nums"] == 1 and kw["use_image_id"] is False and kw["max_inp_length"] is None  # no processor-side cut
    assert len(ids) == 300
    long = _backend(8100)
    with pytest.raises(BadRequest, match="exceeds the engine's context"):
        EngineBackend._prompt(long, parse_request(_body(max_tokens=256)))


@pytest.mark.skipif(not (ROOT / "results" / "engine_parity" / "mm_samples.pkl").exists(), reason="multimodal fixtures not generated")
def test_real_frontend_keeps_long_prompts_whole():
    """A prompt longer than 8192 tokens keeps its final question and consistent image bounds (the processor would have cut
    it at 8192 with its default ``max_length``)."""
    os.environ.setdefault("HF_MODULES_CACHE", "/workspace/models/cgzheng/hf_modules")
    from PIL import Image

    from engine.models.multimodal import MultimodalFrontend
    from engine.testing.mm_samples import MODEL

    f = MultimodalFrontend(MODEL, device="cpu", dtype=torch.float32)
    img = Image.open(Path(MODEL) / "assets" / "fossil.png").convert("RGB")
    question = "Finally, which animal is shown?"
    msgs = [{"role": "user", "content": [img, "word " * 9000, question]}]
    whole = f.build_prompt(msgs, max_slice_nums=1, use_image_id=False, max_inp_length=None)
    ids = whole["input_ids"][0].tolist()
    assert len(ids) > 8192
    tail = f.tokenizer.decode(ids[-40:])
    assert question in tail
    cut = f.build_prompt(msgs, max_slice_nums=1, use_image_id=False)  # the processor default
    assert cut["input_ids"].shape[1] == 8192 and question not in f.tokenizer.decode(cut["input_ids"][0].tolist()[-40:])
    bounds = whole["image_bound"][0]
    assert len(bounds) == 1 and int(bounds[0][1]) < len(ids)  # one image (max_slice_nums=1), inside the prompt
