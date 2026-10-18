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
