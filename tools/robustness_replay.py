"""Rebuild a robustness run's request-derived results from its saved artifacts (written by tools/robustness.py).

    python tools/robustness_replay.py DIR      # exit 0 when every rebuilt value matches the report

Checks, from ``robustness.json``, ``requests.jsonl`` and ``wire.jsonl.gz`` alone:
* the artifacts' hashes equal the ones the report recorded, and the checker's sources are the ones it recorded;
* every request record has its wire record; replaying the wire bytes through the harness client
  (``bench.client.replay``) and validity rules (``bench.metrics.request_metrics``) gives the stored metrics;
* the per-scenario results are rebuilt from the records: request counts and validity, cancellations with audio,
  EOS-boundary residues, overload statuses, and for the sustained load the request count, invalid count, largest stall,
  row cycles, clients and their refill gaps, and the request-derived part of the verdict;
* the sustain workload is the declared row sequence: request ``seq`` i is row i mod n of cycle i div n, and every seq
  from 0 to the request count appears once.
"""

from __future__ import annotations

import base64
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench.adapters import get_adapter  # noqa: E402
from bench.client import replay  # noqa: E402
from bench.metrics import RequestMetrics, request_metrics  # noqa: E402

TOL = 1e-6


def _jsonl(p: Path, gz: bool = False) -> list:
    op = gzip.open if gz else open
    with op(p, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify(out_dir: str | Path) -> dict:
    import robustness as rb

    d = Path(out_dir)
    rep = json.loads((d / "robustness.json").read_text())
    problems: list[str] = []
    art = rep.get("artifacts") or {}
    for f in ("requests.jsonl", "wire.jsonl.gz"):
        if rb._sha_file(d / f) != art.get(f):
            problems.append(f"{f} differs from the recorded hash (or is missing)")
    if rep.get("checker") and rb.checker_identity() != rep["checker"]:
        problems.append("the checker sources differ from the ones that produced the report (replay under the recorded revision)")
    try:
        recs = _jsonl(d / "requests.jsonl")
        wires = {(w["scenario"], w["request_id"]): w for w in _jsonl(d / "wire.jsonl.gz", gz=True)}
    except (OSError, json.JSONDecodeError, EOFError) as e:
        return {"ok": False, "problems": problems + [f"unreadable artifacts: {e}"]}
    if art.get("records") is not None and len(recs) != art["records"]:
        problems.append(f"{len(recs)} records, the report recorded {art['records']}")
    adapter = get_adapter("engine")
    by: dict[str, list] = {}
    obs: dict[str, list] = {}
    for r in recs:
        if r.get("record") == "observation":
            obs.setdefault(r["scenario"], []).append(r)
            continue
        if r.get("record") != "request":
            problems.append(f"unknown record {str(r)[:80]}")
            continue
        key = (r["scenario"], r["request_id"])
        stored = RequestMetrics(**r["metrics"])
        if r["http_status"] == 200:
            w = wires.get(key)
            if w is None:
                problems.append(f"{key}: no wire record")
                continue
            tr = replay(adapter, r["request_id"], w["t_dispatch"], [(w["t_dispatch"] + dt, base64.b64decode(b)) for dt, b in w["chunks"]],
                        t_end=w["t_complete"], streaming=True)
            m = request_metrics(tr)
            for f in ("ok", "invalid_reasons"):
                if getattr(m, f) != getattr(stored, f):
                    problems.append(f"{key}: replayed {f} {getattr(m, f)!r} != stored {getattr(stored, f)!r}")
            for f in ("duration_s", "stall_s", "ttfa_s"):
                a, b = getattr(m, f, None), getattr(stored, f, None)
                if (a is None) != (b is None) or (a is not None and abs(a - b) > TOL):
                    problems.append(f"{key}: replayed {f} {a!r} != stored {b!r}")
            if (tr.usage or {}).get("audio_frames") != (r["trace"].get("usage") or {}).get("audio_frames"):
                problems.append(f"{key}: replayed usage differs from the stored trace")
        elif stored.ok:
            problems.append(f"{key}: HTTP {r['http_status']} stored as valid")
        by.setdefault(r["scenario"], []).append((r, stored))

    rebuilt: dict = {}

    def summ(items):
        ms = [m for _, m in items]
        bad = [m for m in ms if not m.ok]
        return {"n": len(ms), "ok": len(ms) - len(bad), "invalid_reasons": sorted({x for m in bad for x in m.invalid_reasons})}

    def same(name: str, field: str, got) -> None:
        want = (rep.get(name) or {}).get(field)
        if isinstance(want, float) or isinstance(got, float):
            if want is None or got is None or abs(want - got) > TOL:
                problems.append(f"{name}.{field}: rebuilt {got!r} != report {want!r}")
        elif want != got:
            problems.append(f"{name}.{field}: rebuilt {got!r} != report {want!r}")

    if "mixed" in rep:
        s = summ(by.get("mixed", []))
        rebuilt["mixed"] = s
        for k in ("n", "ok", "invalid_reasons"):
            same("mixed", k, s[k])
    if "cancel" in rep:
        comp = summ(by.get("cancel", []))
        k_obs = [o for o in obs.get("cancel", []) if o.get("kind") == "cancel_client"]
        rebuilt["cancel"] = {"companions": comp, "cancelled_sent": len(k_obs), "cancelled_with_audio": sum(o["audio_events"] > 0 for o in k_obs)}
        same("cancel", "companions", comp)
        same("cancel", "cancelled_sent", len(k_obs))
        same("cancel", "cancelled_with_audio", rebuilt["cancel"]["cancelled_with_audio"])
    if "eos_boundary" in rep:
        items = by.get("eos_boundary", [])
        s = summ(items)
        frames = [(r["trace"].get("usage") or {}).get("audio_frames") for r, _ in items]
        covered, _ = rb.boundary_coverage(frames, rep["eos_boundary"]["first_chunk"])
        rebuilt["eos_boundary"] = {**s, "boundary_residues_covered": covered}
        for k in ("n", "ok"):
            same("eos_boundary", k, s[k])
        same("eos_boundary", "boundary_residues_covered", covered)
    if "overload" in rep:
        items = by.get("overload", [])
        rebuilt["overload"] = {"n": len(items), "ok": sum(m.ok for _, m in items), "rejected_503": sum(r["http_status"] == 503 for r, _ in items)}
        for k, v in rebuilt["overload"].items():
            same("overload", k, v)
    if "sustain" in rep:
        sus = rep["sustain"]
        items = by.get("sustain", [])
        ms = [m for _, m in items]
        done = [(r["client"], m, r["t_dispatch"], r["t_complete"]) for r, m in items]
        gaps = rb.refill_gaps(done, sus["deadline"])
        rows = (rep.get("workload") or {}).get("sustain", {}).get("row_ids") or []
        if rows:  # request seq i is row i mod len(rows) of cycle i div len(rows); seqs 0 .. n-1 all present
            bad = [r["request_id"] for r, _ in items if r.get("seq") is None or r["row"] != rows[r["seq"] % len(rows)] or r["cycle"] != r["seq"] // len(rows)]
            if bad:
                problems.append(f"sustain: {len(bad)} records outside the declared row sequence (e.g. {bad[:3]})")
            if sorted(r.get("seq") for r, _ in items if r.get("seq") is not None) != list(range(len(items))):
                problems.append("sustain: the request sequence has gaps or duplicates")
        rebuilt["sustain"] = {"requests": len(ms), "invalid": sum(not m.ok for m in ms), "stall_max_s": max((m.stall_s for m in ms if m.ok), default=None),
                              "row_cycles": len(ms) / max(1, len(rows)) if rows else None, **gaps}
        for k in ("requests", "invalid", "stall_max_s", "clients", "clients_busy_to_deadline", "max_refill_gap_s"):
            same("sustain", k, rebuilt["sustain"][k])
        request_ok = (len(ms) > 0 and rebuilt["sustain"]["invalid"] == 0 and (rebuilt["sustain"]["stall_max_s"] or 0) <= 2.0
                      and gaps["clients"] == sus["concurrency"] and gaps["clients_busy_to_deadline"] == sus["concurrency"])
        rebuilt["sustain"]["request_part_passed"] = request_ok
        if sus.get("passed") and not request_ok:
            problems.append("sustain passed in the report but its request-derived conditions fail on the records")
    return {"ok": not problems, "problems": problems, "rebuilt": rebuilt}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    res = verify(argv[0])
    print(json.dumps({k: v for k, v in res.items()}, indent=1)[:20000])
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
