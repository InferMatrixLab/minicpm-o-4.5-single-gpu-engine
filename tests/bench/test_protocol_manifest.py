import asyncio
import json

import pytest
from aiohttp import web
from helpers import engine_audio, engine_text, sse, tone

from bench.adapters import EngineAdapter
from bench.client import make_session
from bench.manifest import build_manifest, validate_manifest
from bench.protocol import closed_loop, plan_sweep, CellPlan, evaluate_run, run_cell
from bench.workload import CanonicalRequest, SeedTTSRow, load_seed_tts_rows, make_id_sets, n_eval, seed_tts_root

SR = 24000


class FakeEngine:
    """Streams 0.5 s of audio in two chunks after a fixed delay; tracks in-flight count."""

    def __init__(self, delay=0.05, fail_ids=()):
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.order: list[str] = []
        self.fail_ids = set(fail_ids)

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        rid = body["messages"][0]["content"]
        self.order.append(rid)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write(engine_text("hi"))
            await asyncio.sleep(self.delay)
            if any(f in rid for f in self.fail_ids):
                await resp.write(sse({"error": {"message": "injected"}}))
            await resp.write(engine_audio(tone(SR // 4, 300), 0))
            await asyncio.sleep(self.delay / 2)
            await resp.write(engine_audio(tone(SR // 4, 400), SR // 4))
            await resp.write(sse("[DONE]"))
            return resp
        finally:
            self.in_flight -= 1


async def _serve(fake: FakeEngine):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", fake.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _reqs(n, prefix="row"):
    out = []
    for i in range(n):
        row = SeedTTSRow(row_id=f"{prefix}{i:03d}", ref_text="r", ref_wav_path="", target_text=f"{prefix}{i:03d}")
        out.append(CanonicalRequest(row=row, seed=i))
    return out


class TextOnlyEngineAdapter(EngineAdapter):
    """Puts the row id as the whole user content so the fake server can see dispatch order."""

    def build_payload(self, req):
        return {"messages": [{"role": "user", "content": req.row.target_text}], "stream": True}


@pytest.mark.asyncio
async def test_closed_loop_order_and_concurrency():
    fake = FakeEngine(delay=0.03)
    runner, url = await _serve(fake)
    try:
        reqs = _reqs(24)
        async with make_session() as s:
            traces = await closed_loop(s, url, TextOnlyEngineAdapter(), reqs, concurrency=4)
        assert [t.request_id for t in traces] == [r.request_id for r in reqs]
        assert fake.order == [r.row.target_text for r in reqs]  # fixed ID order
        assert fake.max_in_flight == 4
        run = evaluate_run("run1", 4, traces)
        assert run.metrics.valid
        assert run.metrics.total_audio_s == pytest.approx(24 * 0.5)
        # closed loop with no think time: dispatches happen as slots free
        starts = sorted(t.t_dispatch for t in traces)
        ends = sorted(t.t_complete for t in traces)
        assert starts[4] >= ends[0] - 1e-3
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_run_cell_discards_run0_and_flags_invalid(tmp_path):
    fake = FakeEngine(delay=0.01, fail_ids=("ev005",))
    runner, url = await _serve(fake)
    try:
        warm = _reqs(4, "wu")
        ev = _reqs(8, "ev")
        results = await run_cell(url, TextOnlyEngineAdapter(), warm, ev, concurrency=2, k_measured=2, out_dir=tmp_path, log=lambda s: None)
        assert [r.label for r in results] == ["run0", "run1", "run2"]
        assert [r.measured for r in results] == [False, True, True]
        assert all(not r.metrics.valid for r in results)  # one injected error per run
        for k in range(3):
            assert (tmp_path / f"run{k}" / "traces.jsonl").exists()
            s = json.loads((tmp_path / f"run{k}" / "summary.json").read_text())
            assert s["n_failed"] == 1
        with pytest.raises(ValueError):
            await run_cell(url, TextOnlyEngineAdapter(), ev[:2], ev, concurrency=2, k_measured=1, out_dir=tmp_path / "x", log=lambda s: None)
    finally:
        await runner.cleanup()


def test_plan_sweep_randomized_reproducibly():
    cells = [CellPlan(e, "cfg", c) for e in ("vllm-omni", "sglang-omni") for c in (1, 2, 4, 8, 16, 32)]
    a = plan_sweep(cells, seed=7)
    b = plan_sweep(cells, seed=7)
    assert a == b and sorted(map(str, a)) == sorted(map(str, cells)) and a != cells


def test_seed_tts_id_sets_disjoint():
    if not (seed_tts_root() / "en" / "meta.lst").exists():
        pytest.skip("seed-tts-eval not available")
    rows = load_seed_tts_rows()
    assert len(rows) == 1088
    sets = make_id_sets(rows)
    w, t, e = set(sets.warmup_pool), set(sets.tuning_pool), set(sets.eval_pool)
    assert not (w & t) and not (w & e) and not (t & e)
    assert len(w) + len(t) + len(e) == 1088
    for c in (1, 2, 4, 8, 16, 32):
        assert len(sets.eval(c)) == n_eval(c) == max(320, 20 * c)
        assert len(sets.warmup(c)) == 2 * c
        assert len(set(sets.eval(c))) == len(sets.eval(c))


def test_manifest_validation(tmp_path):
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    m = build_manifest(
        repos={"qwen3_omni": root},
        env_pythons={"bench": sys.executable},
        out_dir=tmp_path,
        gpu_index=1,
        deploy={"engine": "fake", "config": "x", "resolved": {"k": 1}},
        sampling={"talker": {"temperature": 0.8}},
        datasets=[{"repo": "zhaochenyang20/seed-tts-eval", "revision": "8f5e1aa", "ids": {"eval": ["a"]}}],
        run_order_seed=7,
    )
    assert validate_manifest(m) == [], validate_manifest(m)
    bad = dict(m)
    del bad["hardware"]
    assert "missing hardware" in validate_manifest(bad)
    bad = json.loads(json.dumps(m))
    bad["model"]["revision"] = "main"
    bad["datasets"][0]["ids"] = {}
    errs = validate_manifest(bad)
    assert any("model revision" in e for e in errs) and any("id lists" in e for e in errs)
