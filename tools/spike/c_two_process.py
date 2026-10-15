"""Layout (c), the diagnostic control: Thinker and Talker in two processes, reply hidden states handed over by CUDA IPC.

Only the handoff differs from (a) (both processes run the same kernels), so this measures what the process boundary
adds: the IPC handle export/import and the cross-process wake-up per finished reply.
"""

import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.spike.common import SpikeResult, Workload  # noqa: E402


def _talker_side(q_in, q_out):
    torch.cuda.set_device(0)
    while True:
        item = q_in.get()
        if item is None:
            return
        t_sent, tensors = item
        s = sum(float(x[0, 0]) for x in tensors)  # touch the shared memory on the consumer's device
        torch.cuda.synchronize()
        q_out.put((t_sent, time.perf_counter(), s))


def run(w: Workload) -> SpikeResult:
    ctx = mp.get_context("spawn")
    q_in, q_out = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=_talker_side, args=(q_in, q_out))
    p.start()
    lat = []
    for i in range(w.iters + 2):
        hid = [torch.randn(w.reply, 4096, device="cuda", dtype=torch.bfloat16) for _ in range(w.batch)]
        torch.cuda.synchronize()
        q_in.put((time.perf_counter(), hid))
        t_sent, t_recv, _ = q_out.get()
        if i >= 2:
            lat.append((t_recv - t_sent) * 1e3)
        del hid
    q_in.put(None)
    p.join()
    lat.sort()
    r = SpikeResult("c: two processes, CUDA IPC handoff", vars(w))
    r.handoff_ms = lat[len(lat) // 2]
    r.kv_budget_note = "each process owns its pool; memory split must be fixed at launch (no sharing of free pages)"
    r.notes.append("kernel timings are those of layout (a); only the handoff and the process boundary are measured here")
    return r


if __name__ == "__main__":
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = Path(f"results/l20x/spike/c_b{b}.json")
    r = run(Workload(batch=b))
    r.peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30
    r.save(out)
