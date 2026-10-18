"""Host-to-device copies that never wait for the GPU.

``torch.tensor(data, device="cuda")`` copies from pageable memory and then synchronizes the stream: the host waits for
every kernel queued before it.  ``to_device`` queues a non-blocking copy instead (CUDA stages pageable host memory at
the call, so the source may be freed at once) and the values are the same.  FlashInfer's ``plan`` reads its index
pointers and last-page lengths on the host: given device tensors it copies them back (one synchronization each), given
host tensors (``host``) it does not.
"""

from __future__ import annotations

import torch


def host(data, dtype=torch.int32) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype)


def to_device(data, dtype=torch.int32, device="cuda") -> torch.Tensor:
    return torch.tensor(data, dtype=dtype).to(device, non_blocking=True)


def plan(wrapper, *args, **kwargs) -> None:
    """``wrapper.plan(*args, **kwargs)`` for a FlashInfer wrapper, ordered against its previous plan.

    ``plan`` writes the plan into the wrapper's page-locked host buffer and uploads it with an asynchronous copy on the
    current stream.  With host inputs nothing synchronizes before that write, so a second plan of the same wrapper could
    overwrite the buffer while the first upload is still queued (the replay that follows it would read the second plan:
    an illegal memory access under load).  Each plan therefore waits for its predecessor's upload, an event recorded
    right after it, and not for the work queued behind it."""
    done = getattr(wrapper, "_engine_plan_done", None)
    if done is not None:
        done.synchronize()
    wrapper.plan(*args, **kwargs)
    ev = torch.cuda.Event()
    ev.record()
    wrapper._engine_plan_done = ev
