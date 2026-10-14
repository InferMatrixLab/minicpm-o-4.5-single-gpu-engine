"""Pin evaluation assets: HF snapshot revisions and SHA-256 of scorer checkpoints.

    python evals/lock_assets.py > evals/assets.lock.json
"""

import hashlib
import json
import os
from pathlib import Path

HUB = Path(os.environ.get("HF_HUB_CACHE", "/workspace/models/hub"))
SGL_CACHE = Path(os.environ.get("SGLANG_OMNI_CACHE", Path.home() / ".cache" / "sglang-omni"))

REPOS = {
    "model": ["openbmb/MiniCPM-o-4_5", "Qwen/Qwen3-ASR-1.7B", "openai/whisper-large-v3", "balacoon/utmos"],
    "dataset": [
        "zhaochenyang20/seed-tts-eval",
        "zhaochenyang20/seed-tts-eval-arrow",
        "MMMU/MMMU",
        "zhaochenyang20/mmmu-ci-50",
        "ddwang2000/MMSU",
        "zhaochenyang20/mmsu-ci-2000",
        "liarliar/Daily-Omni",
    ],
}
FILES = [
    SGL_CACHE / "speaker_sim" / "wavlm_large.pt",
    SGL_CACHE / "speaker_sim" / "wavlm_large_finetune.pth",
    SGL_CACHE / "utmos" / "utmos.jit",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshots(kind: str, repo: str) -> dict:
    d = HUB / f"{kind}s--{repo.replace('/', '--')}"
    refs = {p.name: p.read_text().strip() for p in (d / "refs").glob("*")} if (d / "refs").is_dir() else {}
    snaps = sorted(p.name for p in (d / "snapshots").glob("*")) if (d / "snapshots").is_dir() else []
    return {"refs": refs, "snapshots": snaps, "present": bool(snaps)}


def main() -> None:
    out = {"hub": str(HUB), "repos": {}, "files": {}}
    for kind, repos in REPOS.items():
        for r in repos:
            out["repos"][r] = {"kind": kind, **snapshots(kind, r)}
    for f in FILES:
        out["files"][str(f)] = {"sha256": sha256(f) if f.exists() else None, "bytes": f.stat().st_size if f.exists() else None}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
