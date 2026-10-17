"""Row-by-row comparison of two scorings of the same generations (e.g. before and after an evaluator change).

    python tools/rescore_compare.py OLD.json NEW.json [OLD2.json NEW2.json ...] --out OUT.json

Each argument pair names two speech quality reports (``evals.speech.suite``); their row sidecars
(``<report>.rows.jsonl``) are joined on (seed_index, row_id).  Every scored field is compared exactly; rows present in
only one report are listed.  The pooled summaries are compared too.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SKIP = ()  # every field is compared


def _rows(report: Path) -> dict:
    side = report.with_suffix(".rows.jsonl")
    return {(r["seed_index"], r["row_id"]): r for r in (json.loads(x) for x in side.read_text().splitlines() if x.strip())}


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def compare(old: Path, new: Path) -> dict:
    a, b = _rows(old), _rows(new)
    fields: dict[str, list] = {}
    for k in sorted(set(a) & set(b)):
        for f in sorted(set(a[k]) | set(b[k])):
            if f in SKIP:
                continue
            if not _same(a[k].get(f), b[k].get(f)):
                fields.setdefault(f, []).append({"row": list(k), "old": a[k].get(f), "new": b[k].get(f)})
    so, sn = json.loads(old.read_text())["summary"]["pooled"], json.loads(new.read_text())["summary"]["pooled"]
    return {"old": str(old), "new": str(new), "rows_old": len(a), "rows_new": len(b), "rows_compared": len(set(a) & set(b)),
            "only_old": sorted(map(list, set(a) - set(b)))[:20], "only_new": sorted(map(list, set(b) - set(a)))[:20],
            "differing_fields": {f: {"rows": len(v), "examples": v[:5]} for f, v in fields.items()},
            "summary_differences": {k: {"old": so.get(k), "new": sn.get(k)} for k in sorted(set(so) | set(sn)) if not _same(so.get(k), sn.get(k))}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.reports) % 2:
        raise SystemExit("give OLD NEW pairs")
    res = [compare(Path(o), Path(n)) for o, n in zip(a.reports[::2], a.reports[1::2])]
    Path(a.out).write_text(json.dumps(res, indent=1))
    for r in res:
        print(f"{Path(r['new']).parent.name}/{Path(r['new']).name}: {r['rows_compared']} rows, only old {len(r['only_old'])}, only new {len(r['only_new'])}, "
              f"differing {({f: v['rows'] for f, v in r['differing_fields'].items()})}, summary {sorted(r['summary_differences'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
