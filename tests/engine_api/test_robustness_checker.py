"""The AC-6 robustness checker (tools/robustness.py) against the real EngineServer with scripted backends (CPU).

Faults it must reject: an SSE error followed by [DONE], overlapping audio, a reply without audio, a cancellation the
engine does not drain, and frame counts that never reach a window boundary.  A clean backend passes.
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
from aiohttp.test_utils import TestServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tools.robustness as rb  # noqa: E402
from bench.adapters import get_adapter  # noqa: E402
from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets  # noqa: E402
from engine.server import AudioChunk, Done, EngineServer, TextDelta  # noqa: E402

SR = 24000


class Backend:
    """mode: ok | error_after_audio | overlap | no_audio | stuck (never drains); frames: audio_frames per reply."""

    def __init__(self, mode="ok", frames=None):
        self.mode, self.frames, self.n, self.live = mode, frames or [33], 0, 0

    def stats(self):
        return {"admitted": 1 if self.mode == "stuck" else self.live, "live": self.live, "thinker_kv_pages_used": 0, "talker_kv_pages_used": 0,
                "first_chunk": 8, "t2w_requests": 0, "t2w_cache_slots_used": 0, "t2w_dit_eager_calls": self.n}

    async def generate(self, req):
        self.n += 1
        self.live += 1
        frames = self.frames[(self.n - 1) % len(self.frames)]  # this request's frame count, fixed at its start
        try:
            yield TextDelta("Hello.")
            x = (0.2 * np.sin(np.arange(SR // 4) * 0.05)).astype(np.float32)
            if self.mode != "no_audio":
                yield AudioChunk(x, 0)
                await asyncio.sleep(0.01)
                if self.mode == "error_after_audio":
                    raise RuntimeError("engine error: boom")
                yield AudioChunk(x, len(x) // 2 if self.mode == "overlap" else len(x))
            yield Done("stop", {"prompt_tokens": 5, "completion_tokens": 2, "audio_frames": frames})
        finally:
            self.live -= 1


def _run(backend, fn):
    async def go():
        srv = TestServer(EngineServer(backend).app)
        await srv.start_server()
        try:
            return await fn(f"http://127.0.0.1:{srv.port}")
        finally:
            await srv.close()
    return asyncio.run(go())


def _reqs(n):
    rows = load_seed_tts_rows()
    return canonical_requests(rows, make_id_sets(rows).tuning(32)[:n])


def test_mixed_passes_clean_and_rejects_faults():
    a = get_adapter("engine")
    assert _run(Backend("ok"), lambda u: rb.scenario_mixed(u, a, n=4))["passed"]
    for mode in ("error_after_audio", "overlap", "no_audio"):
        res = _run(Backend(mode), lambda u: rb.scenario_mixed(u, a, n=4))
        assert not res["passed"], (mode, res)
        assert res["invalid_reasons"], mode


def test_cancel_requires_observed_cancellations_and_a_drained_engine():
    a = get_adapter("engine")
    payloads = [a.build_payload(r) for r in _reqs(4)]
    ok = _run(Backend("ok"), lambda u: rb.scenario_cancel(u, a, payloads))
    assert ok["passed"] and ok["cancelled_observed"] >= 2, ok
    stuck = _run(Backend("stuck"), lambda u: rb.scenario_cancel(u, a, payloads))
    assert not stuck["passed"] and stuck["idle_problems"], stuck


def test_eos_boundary_coverage_comes_from_observed_frame_counts():
    a = get_adapter("engine")
    reqs = _reqs(6)
    covered = _run(Backend("ok", frames=[32, 33, 34, 58, 40, 41]), lambda u: rb.scenario_eos_boundary(u, a, reqs, 8, concurrency=3))
    assert covered["passed"], covered  # (f - 8) % 25 hits 24, 0, 1
    missing = _run(Backend("ok", frames=[40, 41, 42]), lambda u: rb.scenario_eos_boundary(u, a, reqs, 8, concurrency=3))
    assert not missing["passed"], missing
    faulty = _run(Backend("error_after_audio", frames=[32, 33, 34]), lambda u: rb.scenario_eos_boundary(u, a, reqs, 8, concurrency=3))
    assert not faulty["passed"]


def test_overload_counts_only_valid_or_503():
    a = get_adapter("engine")
    res = _run(Backend("error_after_audio"), lambda u: rb.scenario_overload(u, a, _reqs(3)))
    assert not res["passed"] and res["other_failures"]


def test_served_identity_matches_the_harness_resolution():
    """``/info`` (what the server loaded) and ``bench.cli.resolve_engine_deploy`` (what the harness declares) agree, and the
    checker's identity check fails for another config."""
    from bench.cli import resolve_engine_deploy
    from engine.server import served_identity

    cfg = str(ROOT / "configs" / "engine" / "track_s_fast.json")
    served = served_identity(cfg)
    want = resolve_engine_deploy(cfg)
    assert served["config_sha256"] == want["config_sha256"] and served["engine_tree_sha256"] == want["engine_tree_sha256"]
    assert rb.identity_check(served, cfg)["matches"]
    assert not rb.identity_check(served, str(ROOT / "configs" / "engine" / "track_s_proc.json"))["matches"]
