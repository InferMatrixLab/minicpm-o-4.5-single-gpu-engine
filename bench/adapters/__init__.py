"""Engine adapters: map one canonical request to each engine's native API and parse its responses."""

from bench.adapters.base import Adapter
from bench.adapters.engine import EngineAdapter
from bench.adapters.sglang_omni import SglangOmniAdapter
from bench.adapters.vllm_omni import VllmOmniAdapter

ADAPTERS: dict[str, type[Adapter]] = {
    "engine": EngineAdapter,
    "vllm-omni": VllmOmniAdapter,
    "sglang-omni": SglangOmniAdapter,
}


def get_adapter(name: str, **kwargs) -> Adapter:
    try:
        cls = ADAPTERS[name]
    except KeyError as e:
        raise ValueError(f"unknown adapter {name!r}; choose from {sorted(ADAPTERS)}") from e
    return cls(**kwargs)


__all__ = ["ADAPTERS", "Adapter", "EngineAdapter", "SglangOmniAdapter", "VllmOmniAdapter", "get_adapter"]
