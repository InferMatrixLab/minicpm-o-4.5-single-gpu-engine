"""Copy the reportable part of measured cells into ``evidence/``: specs, summaries, per-request records and server logs.

    python tools/collect_evidence.py --out evidence/20261019-l20x CELL_DIR [CELL_DIR ...]

Per cell: ``experiment.json``, ``result.json`` (verification, manifest, run summaries), the environment locks, every
run's ``summary.json`` and ``requests.jsonl`` (gzipped), the warmup summaries and the server logs (gzipped).  Audio,
wire dumps and traces stay with the raw cell.  ``manifest.json`` lists every copied file with its SHA-256 and the source cell it came from.
"""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy(src: Path, dst: Path, compress: bool) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        dst = dst.with_name(dst.name + ".gz")
        with src.open("rb") as fi, gzip.GzipFile(dst, "wb", mtime=0) as fo:
            shutil.copyfileobj(fi, fo)
    else:
        shutil.copyfile(src, dst)
    return dst


def collect(cell: Path, out: Path, logs: bool = True) -> list[dict]:
    exp = json.loads((cell / "experiment.json").read_text())["experiment"]
    name = f"{exp['candidate']['candidate_id'].replace(':', '_')}_c{exp['concurrency']}"
    dst, files = out / name, []
    picks = [(cell / "experiment.json", False), (cell / "result.json", False)] + [(p, False) for p in sorted(cell.glob("env-*.lock"))]
    for run in sorted(p for p in cell.iterdir() if p.is_dir() and p.name.startswith("run")):
        picks += [(run / "summary.json", False), (run / "requests.jsonl", True)]
    picks += [(p, True) for p in sorted(cell.glob("server_*.log"))] if logs else []
    for src, compress in picks:
        if not src.exists():
            continue
        d = copy(src, dst / src.relative_to(cell), compress)
        files.append({"path": str(d.relative_to(out)), "sha256": sha256(d), "source": str(src), "source_sha256": sha256(src)})
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-logs", action="store_true", help="leave the server logs with the raw cells")
    ap.add_argument("cells", nargs="+", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = [f for c in a.cells for f in collect(c, a.out, logs=not a.no_logs)]
    (a.out / "manifest.json").write_text(json.dumps({"schema": "evidence/v1", "cells": [str(c) for c in a.cells], "files": files}, indent=1))
    print(f"{len(files)} files from {len(a.cells)} cells -> {a.out}")


if __name__ == "__main__":
    main()
