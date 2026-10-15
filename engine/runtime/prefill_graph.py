"""CUDA graphs of a single-request Thinker prefill (``DecoderRunner.body``), bucketed by token count.

A prompt of ``n`` tokens runs in the bucket ``B >= n`` as a two-sequence ragged batch: the request (``n`` tokens at
positions 0..n-1) and a scratch sequence holding the ``B - n`` padding tokens (its own pages, released after the
call).  The FlashInfer prefill wrapper runs in CUDA-graph mode on fixed buffers; its plan is host work done before each
replay.  The request's KV and hidden states are exactly what the eager forward produces for the same two-sequence
batch; versus a batch of the request alone, only GEMM tiling over the padded rows can differ.
"""

from __future__ import annotations

import torch

from engine.runtime.paged import DecoderRunner

PAD_SEQ = (1 << 42) + 7
BUCKETS = (32, 64, 128, 256, 512)


class _Bucket:
    def __init__(self, run: DecoderRunner, n: int, workspace: torch.Tensor):
        import flashinfer

        dev, s = run.kv.k.device, run.spec
        self.n = n
        max_pages = 2 * (n // run.kv.page_size + 2)
        self.qo = torch.zeros(3, dtype=torch.int32, device=dev)
        self.kv_indptr = torch.zeros(3, dtype=torch.int32, device=dev)
        self.kv_indices = torch.zeros(max_pages, dtype=torch.int32, device=dev)
        self.last = torch.zeros(2, dtype=torch.int32, device=dev)
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", use_cuda_graph=True, qo_indptr_buf=self.qo,
                                                                     paged_kv_indptr_buf=self.kv_indptr, paged_kv_indices_buf=self.kv_indices,
                                                                     paged_kv_last_page_len_buf=self.last)
        self.x = torch.zeros(n, s.hidden_size, dtype=run.kv.k.dtype, device=dev)
        self.positions = torch.zeros(n, dtype=torch.int32, device=dev)
        self.batch_idx = torch.zeros(n, dtype=torch.int32, device=dev)
        self.out = None
        self.graph = None


class PrefillGraphs:
    def __init__(self, run: DecoderRunner, buckets=BUCKETS, workspace_mb: int = 256):
        self.run, self.kv = run, run.kv
        self.workspace = torch.empty(workspace_mb << 20, dtype=torch.uint8, device=run.kv.k.device)
        self.buckets = {b: _Bucket(run, b, self.workspace) for b in buckets}
        self.pool = torch.cuda.graph_pool_handle()

    def supports(self, n: int) -> bool:
        return n < max(self.buckets)  # the scratch sequence always holds at least one token

    def _plan(self, bk: _Bucket, seq: int, n: int) -> None:
        lay = self.kv.layout([seq, PAD_SEQ], [n, bk.n - n])
        bk.qo.copy_(lay.qo_indptr)
        bk.kv_indptr.copy_(lay.kv_indptr)
        bk.kv_indices[:lay.kv_indices.numel()].copy_(lay.kv_indices)
        bk.last.copy_(lay.kv_last_page_len)
        bk.positions.copy_(lay.positions)
        bk.batch_idx.copy_(lay.batch_indices)
        s = self.run.spec
        bk.wrapper.plan(lay.qo_indptr, lay.kv_indptr, lay.kv_indices, lay.kv_last_page_len, s.num_heads, s.num_kv_heads, s.head_dim,
                        self.kv.page_size, causal=True, q_data_type=self.kv.k.dtype, kv_data_type=self.kv.k.dtype)

    def _body(self, bk: _Bucket) -> torch.Tensor:
        return self.run.body(bk.x, bk.positions, bk.batch_idx, bk.kv_indices, bk.kv_indptr, bk.last, bk.wrapper)

    @torch.inference_mode()
    def _capture(self, bk: _Bucket) -> None:
        scratch = PAD_SEQ + 1
        self._plan(bk, scratch, 1)
        self._body(bk)  # warm-up
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
            bk.out = self._body(bk)
        torch.cuda.synchronize()
        bk.graph = g
        self.kv.release(scratch)
        self.kv.release(PAD_SEQ)

    def capture_all(self) -> None:
        for bk in self.buckets.values():
            if bk.graph is None:
                self._capture(bk)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, seq: int) -> torch.Tensor:
        """Prefill ``x`` [n, hidden] as new sequence ``seq`` (positions from 0); returns its final-norm hidden states."""
        n = x.shape[0]
        bk = self.buckets[min(b for b in self.buckets if b > n)]
        if bk.graph is None:
            self._capture(bk)
        self._plan(bk, seq, n)
        bk.x[:n].copy_(x)
        bk.x[n:].zero_()
        bk.graph.replay()
        out = bk.out[:n].clone()
        self.kv.release(PAD_SEQ)
        return out
