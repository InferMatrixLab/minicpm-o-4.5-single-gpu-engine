"""Layout (b): vLLM's runner embedded for the Thinker (run in the vllm-omni environment, which has vllm).

The Thinker weights are exported once as a plain Qwen3 checkpoint (``llm.*`` -> ``model.*``) so stock vLLM can load them.
Measured: offline prefill and per-step decode latency at batch B with vLLM's own scheduler and CUDA graphs, the KV
budget vLLM reserves (``gpu_memory_utilization``), and whether per-token last-layer hidden states (the Talker's
condition) are reachable from the public API.
"""

import json
import os
import sys
import time
from pathlib import Path

MODEL_DIR = Path(os.environ.get("MODEL_PATH") or os.path.join(os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub"), "models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"))


def export(dst: Path) -> Path:
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    if (dst / "config.json").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    c = json.loads((MODEL_DIR / "config.json").read_text())
    keep = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "rope_theta",
            "rms_norm_eps", "vocab_size", "max_position_embeddings", "hidden_act")
    cfg = {k: c[k] for k in keep} | {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "tie_word_embeddings": False,
                                     "torch_dtype": "bfloat16", "attention_bias": False}
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    for f in sorted({v for k, v in index.items() if k.startswith("llm.")}):
        out = {}
        with safe_open(str(MODEL_DIR / f), "pt") as fh:
            for k in fh.keys():
                if k.startswith("llm."):
                    out[k[4:]] = fh.get_tensor(k)
        save_file(out, str(dst / f.replace("model-", "thinker-")))
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"):
        if (MODEL_DIR / name).exists():
            shutil.copy(MODEL_DIR / name, dst / name)
    return dst


def run(batch: int, out: Path, export_dir: Path) -> dict:
    from vllm import LLM, SamplingParams

    llm = LLM(model=str(export(export_dir)), gpu_memory_utilization=0.5, max_model_len=4096, enforce_eager=False, max_num_seqs=64)
    prompts = [{"prompt_token_ids": [100 + (i * 7 + j) % 50000 for j in range(60)]} for i in range(batch)]
    sp1 = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
    spr = SamplingParams(temperature=0, max_tokens=31, ignore_eos=True)
    llm.generate(prompts, sp1, use_tqdm=False)
    t0 = time.perf_counter()
    llm.generate(prompts, sp1, use_tqdm=False)
    pre = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    llm.generate(prompts, spr, use_tqdm=False)
    full = (time.perf_counter() - t0) * 1e3
    res = {"layout": "b: vLLM runner for the Thinker", "batch": batch, "thinker_prefill_ms": pre, "thinker_decode_step_ms": (full - pre) / 30,
           "kv_budget_note": "vLLM reserves gpu_memory_utilization of the device for its own weights+KV at start; the Talker, Token2Wav and"
                             " their graphs must fit in the remainder, sized before any request is seen",
           "hidden_states_reachable": False,
           "notes": ["the public generate API returns tokens/logprobs only; per-token last-layer hidden states need a model-runner hook"
                     " (vllm-omni's custom runner does this), i.e. ownership of vLLM's forward context",
                     "vLLM's scheduler/engine loop owns batching and graph capture; the Talker/Token2Wav would run in a second loop beside it"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    root = Path(__file__).resolve().parents[2]
    run(b, root / "results" / "l20x" / "spike" / f"b_b{b}.json", Path(os.environ.get("THINKER_EXPORT_DIR", root / "artifacts" / "thinker_qwen3_export")))
