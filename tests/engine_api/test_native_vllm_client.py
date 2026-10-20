"""The unchanged vllm-omni native client (``vllm bench serve --omni --backend openai-chat-omni``) against the engine API."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench import paths  # noqa: E402

VLLM = paths.VLLM_OMNI_REPO
MODEL = str(paths.MODEL_PATH)
DATA = str(paths.SEED_TTS_EVAL)
pytestmark = pytest.mark.skipif(not (VLLM / ".venv/bin/vllm").exists(), reason="vllm-omni environment not available")

SERVER = """
import sys
sys.path.insert(0, {root!r}); sys.path.insert(0, {tests!r})
from aiohttp import web
from test_server import Scripted
from engine.server import EngineServer
web.run_app(EngineServer(Scripted(chunks=4, delay=0.02)).app, host="127.0.0.1", port={port}, print=None)
"""


def test_native_client_parses_engine_stream(tmp_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = subprocess.Popen([sys.executable, "-c", SERVER.format(root=str(ROOT), tests=str(ROOT / "tests" / "engine_api"), port=port)])
    try:
        time.sleep(3)
        out = subprocess.run([str(VLLM / ".venv/bin/vllm"), "bench", "serve", "--omni", "--host", "127.0.0.1", "--port", str(port),
                              "--model", "openbmb/MiniCPM-o-4_5", "--tokenizer", MODEL, "--trust-remote-code", "--backend", "openai-chat-omni",
                              "--endpoint", "/v1/chat/completions", "--dataset-name", "seed-tts", "--dataset-path", DATA, "--seed-tts-locale", "en",
                              "--disable-shuffle", "--no-oversample", "--num-prompts", "4", "--max-concurrency", "2", "--request-rate", "inf",
                              "--extra-body", '{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}',
                              "--percentile-metrics", "ttft,e2el,audio_ttfp,audio_rtf,audio_duration"],
                             cwd=str(VLLM), capture_output=True, text=True, timeout=600,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"})
        text = out.stdout + out.stderr
        assert "Successful requests:                     4" in text and "Failed requests:                         0" in text, text[-2000:]
        assert "Mean AUDIO_DURATION (s):                 2.00" in text
    finally:
        srv.terminate()
        srv.wait(10)
