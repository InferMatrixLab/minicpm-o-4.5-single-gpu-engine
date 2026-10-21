# Reproducing the benchmark

The harness (`bench/`) drives the engine, vLLM-Omni and SGLang-Omni with the same requests and computes every metric
from client timestamps. A measured cell is self-describing: its `experiment.json` freezes the spec, the source and
environment fingerprints and the server launch, and `bench.verify` re-checks the cell from its artifacts.

## 1. Environments and checkouts

```bash
# harness
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python --no-deps -r bench/bench.lock
# engine (see the README)
uv venv --python 3.12 .venv-engine && uv pip install --python .venv-engine/bin/python --no-deps \
  --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 -r engine/env/engine.lock

# baselines next to this repository (or point OMNI_VLLM_REPO / OMNI_SGLANG_REPO at them)
git clone https://github.com/vllm-project/vllm-omni.git ../vllm-omni && git -C ../vllm-omni checkout 2ec5f265c69520cdd5e5f0a19db485d38885843e
git clone https://github.com/sgl-project/sglang-omni.git ../sglang-omni && git -C ../sglang-omni checkout 22d4ace88f9547777b819a69d7691204bb9ce847
```

Each baseline gets its own `.venv` inside its checkout, installed from `evals/vllm-omni-env.lock` and the SGLang-Omni
lock recorded in each baseline cell's `env-sglang-omni.lock`. The speech evaluators (`evals/`) need two more
environments (`evals/eval-env.lock`, `evals/asr-env.lock`); their locations are `OMNI_EVAL_ENV` and `OMNI_ASR_ENV`. The
locks name this host's editable checkouts on their `-e` lines, and `evals/assets.lock.json` records this host's cache
paths. Regenerate the asset lock on a new host with `python evals/lock_assets.py > evals/assets.lock.json`.

Download the pinned assets into the Hugging Face cache:

```bash
hf download openbmb/MiniCPM-o-4_5 --revision 503e754207c94da6bb26850b4469f367c9ea3582
hf download zhaochenyang20/seed-tts-eval --repo-type dataset --revision 8f5e1aa2a35d42f42e940074c1983358b9491f89
```

Machine-specific locations go in `scripts/env.local.sh` (untracked). `scripts/env.sh` lists every variable.

## 2. One engine cell

```bash
.venv/bin/python -m bench.cli cell --adapter engine --config track_s_fast_r16 --mps on --env ENGINE_MPS=1 \
  --gpu 0 --port 18932 --concurrency 32 --k 5 --out results/engine/c32
```

This starts a fresh server for every run (`--cache-policy restart`): a warmup, a discarded run 0 and five measured runs
over the evaluation requests (320, or 20 × C when larger). The cell refuses to start if the served identity (`/info`:
config digest, source tree digest, interpreter and package digest, GPU UUID) differs from the declaration.

## 3. Sweeps

The recorded specs of the published cells are in `sweeps/`:

- `sweeps/engine/r11c_gpu2.json`, `r11c_gpu3.json`: the engine at C = 1/4/16 and 2/8/32;
- `sweeps/campaign/eval_c*.json`: the baseline winners per concurrency;
- `sweeps/campaign/{tune,qualify,fullq}_*.json`: the baseline tuning, class-R qualification and full-set quality sweeps.

They name the GPU index and, for vLLM-Omni, the absolute path of the deploy overlay on the recording host; edit both
for another host. Run one with:

```bash
.venv/bin/python -m bench.cli sweep --spec sweeps/engine/r11c_gpu2.json --out results/engine_sweeps/r11c_gpu2
```

A sweep resumes completed cells only when their fingerprints still match.

## 4. Baseline campaign

`evals/campaign.py` reproduces the baseline selection: class-R qualification on the tuning requests, per-C tuning,
winner selection (`select`), the evaluation sweeps (`stage-specs`) and the frozen envelope (`envelope`). See its
docstring for the command sequence. Tuning and evaluation use disjoint request pools, so the envelope is not selected
on the requests it is measured on.

## 5. Comparison and report

```bash
.venv/bin/python tools/engine_vs_envelope.py results/engine_sweeps/r11c_gpu2 results/l20x/campaign/envelope.json
.venv/bin/python tools/collect_evidence.py --out evidence/<date>-<gpu> <cell> [<cell> ...]
.venv/bin/python tools/report_results.py evidence/<date>-<gpu>
```

`engine_vs_envelope.py` verifies both cells from their artifacts and writes point ratios and one-sided 95% bootstrap
lower bounds. `collect_evidence.py` copies the reportable files with their SHA-256. `report_results.py` writes
`docs/current-results.json`, the comparison figure and the README tables.

## 6. Quality, robustness and profiling

- Speech quality: `python -m evals.speech.suite score --cell <cell> --scope final ...` and
  `python -m evals.speech.qref gate` against the frozen speech reference (1088 rows × 3 seeds at C=1 and C=32).
- Multimodal accuracy: `python -m evals.multimodal.suite run --adapter engine ...`.
- Robustness: `scripts/robustness_run.sh 0 <gpu> . track_s_fast <port> results/robustness/<name>` (C=32, 30 minutes,
  cancellations, overload, long prompts and the memory rule; `tools/robustness_replay.py` rebuilds the verdict).
- Attribution: `tools/engine_profile.py` (stage spans, CUPTI kernel windows, GPM counters; see its docstring).

## Tests

```bash
.venv/bin/python -m pytest tests/bench tests/evals          # harness and evaluators, CPU
.venv-engine/bin/python -m pytest tests/engine_api         # engine API, scheduler, worker loop, attribution, CPU
ENGINE_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0 .venv-engine/bin/python -m pytest tests/engine   # parity and graph tests: a free GPU and the checkpoint
```

The HF parity references are produced in the model card's environment (`engine/env/hfref.lock`) with `tools/hfref/`.
