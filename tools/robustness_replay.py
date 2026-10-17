"""Rebuild a robustness run's whole verdict from its saved artifacts (written by tools/robustness.py).

    python tools/robustness_replay.py DIR      # exit 0 when the rebuilt verdict equals the report; writes DIR/replay.json

From ``robustness.json``, ``requests.jsonl`` and ``wire.jsonl.gz`` alone (both read as streams, in their common write
order):

* the artifacts' hashes equal the ones the report recorded, and the checker's sources are the ones it recorded;
* every request and cancellation record has exactly its wire record (same scenario, request id, dispatch and completion
  times), there are no duplicate or extra records, and replaying the wire bytes through the harness client
  (``bench.client.replay``) and validity rules (``bench.metrics.request_metrics``) gives the stored metrics -- for a
  cancelling client, its prefix up to the disconnect and its cancellation problems (``robustness.cancel_problems``);
* the raw observations (counter readings before/after cancel, fallback and the long-prefill scenario, the warmed
  baseline, every sustain sample and the endpoint, identity, GPU, geometry, workload, the frozen PID inventory) feed
  the same verdict functions the live run used (``robustness.verdict_*``): every scenario's result, identity/GPU
  equality, the reference and memory envelope, allocation/capacity restoration, cancellation/fallback deltas, the
  continuity rule over the requests' own times and the in-flight timeline, phase order, the acceptance geometry and
  the overall Boolean must equal the report's;
* an acceptance report must hold every scenario.

A run written before these observations were persisted (round 7 and earlier) is reported with the observations it lacks;
nothing is inferred for them.
"""

from __future__ import annotations

import base64
import gzip
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench.adapters import get_adapter  # noqa: E402
from bench.client import replay  # noqa: E402
from bench.metrics import RequestMetrics, request_metrics  # noqa: E402

TOL = 1e-6


def _diff(a, b, path: str = "", out: list | None = None) -> list:
    """Differences between two JSON values (floats within ``TOL``)."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: {'missing in rebuild' if k not in a else 'not in report'}")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items rebuilt, {len(b)} reported")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool)
                and (a == b or (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= TOL * max(1.0, abs(b))))):
            out.append(f"{path}: rebuilt {a!r} != report {b!r}")
    elif a != b:
        out.append(f"{path}: rebuilt {str(a)[:120]!r} != report {str(b)[:120]!r}")
    return out


def _lines(p: Path, gz: bool = False):
    op = gzip.open if gz else open
    with op(p, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _slim(r: dict) -> dict:
    """What the verdicts read of a request record (its trace's usage only): keeps 30,000 sustain records small."""
    return {**{k: v for k, v in r.items() if k != "trace"}, "trace": {"usage": (r.get("trace") or {}).get("usage") or {}}}


def _check_request(adapter, r: dict, w: dict, problems: list) -> None:
    import robustness as rb

    key = (r["scenario"], r["request_id"])
    for f in ("t_dispatch", "t_complete"):
        if r.get(f) != w.get(f):
            problems.append(f"{key}: request {f} {r.get(f)!r} != wire {w.get(f)!r}")
    stored = RequestMetrics(**r["metrics"])
    chunks = [(w["t_dispatch"] + dt, base64.b64decode(b)) for dt, b in w["chunks"]]
    if r["record"] == "cancel":
        tr = replay(adapter, r["request_id"], w["t_dispatch"], chunks, t_end=r.get("t_disconnect"), streaming=True)
        tr.http_status = r["http_status"]
        stored_error = (r.get("trace") or {}).get("error")
        if not tr.error and (stored_error or r["http_status"] != 200):  # a client-side failure is not in the wire bytes: it can only add problems
            tr.error = stored_error or "non-200"
        got = {"cancel_problems": rb.cancel_problems(tr), "unique_samples": tr.audio.unique_samples}
        for f, v in got.items():
            if r.get(f) != v:
                problems.append(f"{key}: replayed {f} {v!r} != stored {r.get(f)!r}")
        return
    if r["http_status"] != 200:
        if stored.ok:
            problems.append(f"{key}: HTTP {r['http_status']} stored as valid")
        return
    tr = replay(adapter, r["request_id"], w["t_dispatch"], chunks, t_end=w["t_complete"], streaming=True)
    m = request_metrics(tr)
    for f in ("ok", "invalid_reasons"):
        if getattr(m, f) != getattr(stored, f):
            problems.append(f"{key}: replayed {f} {getattr(m, f)!r} != stored {getattr(stored, f)!r}")
    for f in ("duration_s", "stall_s", "ttfa_s"):
        a, b = getattr(m, f, None), getattr(stored, f, None)
        if (a is None) != (b is None) or (a is not None and abs(a - b) > TOL):
            problems.append(f"{key}: replayed {f} {a!r} != stored {b!r}")
    if (tr.usage or {}) != ((r.get("trace") or {}).get("usage") or {}):
        problems.append(f"{key}: replayed usage differs from the stored trace")


def verify(out_dir: str | Path) -> dict:
    import robustness as rb

    d = Path(out_dir)
    problems: list[str] = []
    try:
        rep = json.loads((d / "robustness.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        return {"ok": False, "problems": [f"unreadable report: {e}"]}
    art = rep.get("artifacts") or {}
    for f in ("requests.jsonl", "wire.jsonl.gz"):
        if rb._sha_file(d / f) != art.get(f):
            problems.append(f"{f} differs from the recorded hash (or is missing)")
    if rep.get("checker") and rb.checker_identity() != rep["checker"]:
        problems.append("the checker sources differ from the ones that produced the report (replay under the recorded revision)")
    if not (d / "requests.jsonl").is_file() or not (d / "wire.jsonl.gz").is_file():
        return {"ok": False, "problems": problems + ["artifacts missing"]}

    adapter = get_adapter("engine")
    by: dict[str, list] = {}
    snaps: dict[tuple, list] = {}
    seen: set = set()
    n_rec = 0
    wires = _lines(d / "wire.jsonl.gz", gz=True)
    try:
        for r in _lines(d / "requests.jsonl"):
            n_rec += 1
            kind = r.get("record")
            if kind == "snapshot":
                snaps.setdefault((r["scenario"], r["label"]), []).append(r["m"])
                continue
            if kind not in ("request", "cancel"):
                problems.append(f"unknown record {str(r)[:80]}")
                continue
            key = (r["scenario"], r["request_id"])
            if key in seen:
                problems.append(f"{key}: duplicate record")
            seen.add(key)
            w = next(wires, None)
            if w is None or (w.get("scenario"), w.get("request_id")) != key:
                problems.append(f"{key}: no wire record" + ("" if w is None else f" (next wire record is {(w.get('scenario'), w.get('request_id'))})"))
                if w is None:
                    wires = iter(())
                else:  # resynchronise: the rest would all be misaligned
                    return {"ok": False, "problems": problems}
            else:
                _check_request(adapter, r, w, problems)
            by.setdefault(r["scenario"], []).append(_slim(r))
        extra = sum(1 for _ in wires)
        if extra:
            problems.append(f"{extra} wire records without a request record")
    except (OSError, json.JSONDecodeError, EOFError) as e:
        return {"ok": False, "problems": problems + [f"unreadable artifacts: {e}"]}
    if art.get("records") is not None and n_rec != art["records"]:
        problems.append(f"{n_rec} records, the report recorded {art['records']}")

    def one(scen, label):
        xs = snaps.get((scen, label)) or []
        if len(xs) != 1:
            problems.append(f"observation {scen}/{label}: {len(xs)} recorded, 1 expected")
            return None
        return xs[0]

    rebuilt: dict = {"mode": rep.get("mode")}
    ident, gpu, wl = one("run", "identity"), one("run", "gpu"), one("run", "workload")
    if ident is None or gpu is None or wl is None:
        return {"ok": False, "problems": problems + ["the run's raw observations are not in the artifacts (written before they were persisted)"]}
    rebuilt["identity"] = {"matches": rb.identity_matches(ident["served"], ident["expected"])}
    gp = rb.gpu_problems(gpu["served_uuid"], gpu["nvidia_smi"], gpu["declared_uuid"])
    rebuilt["gpu"] = {"problems": gp, "matches": not gp}
    if wl.get("mode") != rep.get("mode") or wl.get("workload") != rep.get("workload"):
        problems.append("the workload declaration or mode differs from the report")
    ex_snap = one("run", "expect")
    expect = rb.Expect(**{"feature_processes": ex_snap["feature_processes"], "pids": frozenset(ex_snap["pids"]) if ex_snap.get("pids") else None}) \
        if ex_snap else rb.Expect()
    cfg = wl.get("config")
    if cfg:
        try:
            want = rb.expect_from_config(cfg).feature_processes
        except Exception:  # the config is not available here: the recorded count stands
            want = None
        if want is not None and want != expect.feature_processes:
            problems.append(f"feature-process count {expect.feature_processes} != the config's {want}")
    baseline = one("run", "baseline")
    rebuilt["baseline_problems"] = rb.idle_problems(baseline, now=baseline.get("t"), expect=expect) if baseline else ["no baseline"]

    phases = [p["name"] for p in snaps.get(("run", "phase"), [])]
    want_order = [s for s in rb.SCENARIOS if s in phases]
    if phases != want_order:
        problems.append(f"phase order {phases} != {want_order}")
    times = [(p["start"], p["end"]) for p in snaps.get(("run", "phase"), [])]
    if any(a[1] > b[0] or a[0] > a[1] for a, b in zip(times, times[1:])):
        problems.append("phases overlap or run backwards")

    if "mixed" in phases:
        rebuilt["mixed"] = rb.verdict_mixed(by.get("mixed", []))
    if "cancel" in phases:
        sent = len(wl["workload"]["cancel"]["row_ids"]) // 2  # the first half of the declared rows cancel
        b, a = one("cancel", "before"), one("cancel", "after")
        rebuilt["cancel"] = rb.verdict_cancel([r for r in by.get("cancel", []) if r["record"] == "request"],
                                              [r for r in by.get("cancel", []) if r["record"] == "cancel"], sent, b or {}, a or {}, baseline, expect)
    if "eos_boundary" in phases:
        fc = one("eos_boundary", "first_chunk")
        rebuilt["eos_boundary"] = rb.verdict_eos_boundary(by.get("eos_boundary", []), (fc or {}).get("first_chunk"))
    if "fallback" in phases:
        rebuilt["fallback"] = rb.verdict_fallback(by.get("fallback", []), one("fallback", "before") or {}, one("fallback", "after") or {})
    if "overload" in phases:
        rebuilt["overload"] = rb.verdict_overload(by.get("overload", []))
    if "sustain" in phases:
        geo, end = one("sustain", "geometry"), one("sustain", "endpoint")
        if geo is not None and end is not None:
            rows = (wl["workload"].get("sustain") or {}).get("row_ids")
            if geo.get("row_ids") != rows:
                problems.append("sustain: the geometry's row sequence differs from the workload declaration")
            rebuilt["sustain"] = rb.verdict_sustain(by.get("sustain", []), snaps.get(("sustain", "sample"), []), end, geo, baseline, expect)
    if "long_prefill" in phases:
        rebuilt["long_prefill"] = rb.verdict_long_prefill(by.get("long_prefill", []), by.get("long_prefill_streams", []),
                                                          one("long_prefill", "before") or {}, one("long_prefill", "after") or {}, baseline, expect)
    for k in set(rb.SCENARIOS) & set(rep) - set(phases):
        problems.append(f"{k} in the report without its phase observations")
    ov = rb.verdict_overall(rebuilt)
    rebuilt.update(ov)
    for k in [s for s in rb.SCENARIOS if s in rebuilt]:
        problems += _diff(rebuilt[k], rep.get(k), k)
    problems += _diff(rebuilt["baseline_problems"], rep.get("baseline_problems"), "baseline_problems")
    for k in ("identity", "gpu"):
        if rebuilt[k]["matches"] != (rep.get(k) or {}).get("matches"):
            problems.append(f"{k}.matches: rebuilt {rebuilt[k]['matches']} != report {(rep.get(k) or {}).get('matches')}")
    problems += _diff(ov["passed"], rep.get("passed"), "passed")
    problems += _diff(ov["problems"], rep.get("problems", []), "problems")
    if ov["overall"] != rep.get("overall"):
        problems.append(f"overall: rebuilt {ov['overall']} != report {rep.get('overall')}")
    return {"ok": not problems, "problems": problems, "rebuilt": {k: rebuilt[k] for k in rebuilt if k in rb.SCENARIOS or k in ("passed", "overall", "problems")}}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    res = verify(argv[0])
    (Path(argv[0]) / "replay.json").write_text(json.dumps(res, indent=1, default=str))  # the whole result
    rebuilt = res["rebuilt"]
    print(json.dumps({"ok": res["ok"], "problems": res["problems"], "overall": rebuilt.get("overall"), "passed": rebuilt.get("passed"),
                      "written": str(Path(argv[0]) / "replay.json")}, indent=1, default=str))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
