"""Batched Talker frame decode: paged FlashInfer decode attention and the HF-order sampler inside CUDA graphs.

One graph per batch bucket covers a whole frame for every row: code embedding,
20 decoder layers (FlashInfer paged decode attention), final norm, the
weight-normalised codec head and the sampler (temperature, 16-frame penalty,
top-p/top-k with min_keep 3, min/max-length EOS rules, counter RNG keyed by
(seed, frame)).  The sampled code is written back into the graph's input buffer,
so ``decode_k`` replays K frames with only a host-side ``plan`` (page-table
update) between replays and a single device-to-host copy at the end.

Rows that finished inside a K block keep running masked: they emit ``PAD`` and
their state is frozen, but their KV length still advances, which keeps the
host's page bookkeeping exact without reading the device.  Padding rows of a
bucket use a scratch sequence and never touch live rows.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.runtime.paged import DecoderSpec, PagedKV, fuse_weights
from engine.talker_sampler import PAD, TalkerSampling, draw, repetition_penalty, top_k_filter, top_p_filter

PAD_SEQ_BASE = 1 << 40  # scratch sequence ids for bucket padding rows


def processed_logits_static(logits, history, steps, eos, cfg: TalkerSampling):
    """``talker_sampler.processed_logits`` without data-dependent host branches (graph-safe)."""
    x = logits.float() / cfg.temperature
    y = top_k_filter(top_p_filter(repetition_penalty(x, history, cfg.repetition_penalty), cfg.top_p, cfg.min_keep), cfg.top_k, cfg.min_keep)
    x = torch.where((steps > 0)[:, None], y, x)
    col = torch.arange(x.size(-1), device=x.device)[None, :] == eos
    x = x.masked_fill(col & (steps < cfg.min_new_tokens)[:, None], float("-inf"))
    forced = (steps >= cfg.max_new_tokens - 1)[:, None]
    return torch.where(forced & ~col, torch.full_like(x, float("-inf")), torch.where(forced & col, torch.zeros_like(x), x))


class _Bucket:
    def __init__(self, dec: "TalkerDecoder", b: int):
        import flashinfer

        dev, s = dec.device, dec.spec
        self.b = b
        max_pages = b * (dec.max_len // dec.kv.page_size + 2)
        self.indptr = torch.zeros(b + 1, dtype=torch.int32, device=dev)
        self.indices = torch.zeros(max_pages, dtype=torch.int32, device=dev)
        self.last = torch.zeros(b, dtype=torch.int32, device=dev)
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(dec.workspace, "NHD", use_cuda_graph=dec.use_graphs,
                                                                    paged_kv_indptr_buffer=self.indptr, paged_kv_indices_buffer=self.indices,
                                                                    paged_kv_last_page_len_buffer=self.last)
        self.codes = torch.zeros(b, dtype=torch.int64, device=dev)
        self.positions = torch.zeros(b, dtype=torch.int32, device=dev)
        self.batch_idx = torch.arange(b, dtype=torch.int32, device=dev)
        self.seeds = torch.zeros(b, dtype=torch.int64, device=dev)
        self.steps = torch.zeros(b, dtype=torch.int64, device=dev)
        self.history = torch.full((b, dec.cfg.window), PAD, dtype=torch.int64, device=dev)
        self.live = torch.zeros(b, dtype=torch.bool, device=dev)
        self.out = torch.full((b,), PAD, dtype=torch.int64, device=dev)
        self.graph = None
        self.spec = s


class TalkerDecoder:
    def __init__(self, talker, kv: PagedKV, spec: DecoderSpec, cfg: TalkerSampling = TalkerSampling(), buckets=(1, 2, 4, 8, 16, 32, 64),
                 max_len: int = 4096, use_graphs: bool = True):
        self.talker, self.kv, self.spec, self.cfg = talker, kv, spec, cfg
        self.layers = fuse_weights(talker.w, spec)
        self.final_norm = talker.w[f"{spec.prefix}norm.weight"]
        self.head = talker.head
        self.emb = talker.w["emb_code.0.weight"]
        self.eos = talker.cfg.eos
        self.device = kv.k.device
        self.max_len, self.use_graphs = max_len, use_graphs
        self.workspace = torch.empty(128 << 20, dtype=torch.uint8, device=self.device)
        self.buckets = {b: _Bucket(self, b) for b in buckets}
        self._pad_next = PAD_SEQ_BASE

    # one frame for the rows of a bucket, all inputs read from the bucket's static buffers
    def _frame(self, bk: _Bucket) -> None:
        import flashinfer as fi

        s = self.spec
        nq, nk = s.num_heads * s.head_dim, s.num_kv_heads * s.head_dim
        x = F.embedding(bk.codes.clamp(min=0), self.emb)
        for i, L in enumerate(self.layers):
            h = fi.rmsnorm(x, L["ln1"], s.rms_eps)
            qkv = F.linear(h, L["qkv"])
            q = qkv[:, :nq].reshape(-1, s.num_heads, s.head_dim).contiguous()
            k = qkv[:, nq:nq + nk].reshape(-1, s.num_kv_heads, s.head_dim).contiguous()
            v = qkv[:, nq + nk:].reshape(-1, s.num_kv_heads, s.head_dim).contiguous()
            fi.apply_rope_pos_ids_inplace(q, k, bk.positions, rope_theta=s.rope_theta)
            fi.append_paged_kv_cache(k, v, bk.batch_idx, bk.positions, (self.kv.k[i], self.kv.v[i]), bk.indices, bk.indptr, bk.last)
            a = bk.wrapper.run(q, (self.kv.k[i], self.kv.v[i]))
            x = x + F.linear(a.reshape(-1, nq), L["o"])
            h = fi.rmsnorm(x, L["ln2"], s.rms_eps)
            g, u = F.linear(h, L["gate_up"]).chunk(2, dim=-1)
            x = x + F.linear(F.silu(g) * u, L["down"])
        logits = F.linear(fi.rmsnorm(x, self.final_norm, s.rms_eps), self.head)
        tok = draw(processed_logits_static(logits, bk.history, bk.steps, self.eos, self.cfg), bk.seeds, bk.steps)
        tok = torch.where(bk.live, tok, torch.full_like(tok, PAD))
        bk.history.copy_(torch.where(bk.live[:, None], torch.cat([bk.history[:, 1:], tok[:, None]], dim=1), bk.history))
        bk.steps.add_(bk.live.to(torch.int64))
        bk.live.logical_and_(tok != self.eos)
        bk.out.copy_(tok)
        bk.codes.copy_(tok)
        bk.positions.add_(1)

    def _plan(self, bk: _Bucket, seqs: list[int]) -> None:
        lay = self.kv.layout(seqs, [1] * len(seqs), device=self.device)
        n = lay.kv_indices.numel()
        bk.indptr.copy_(lay.kv_indptr)
        bk.indices[:n].copy_(lay.kv_indices)
        bk.last.copy_(lay.kv_last_page_len)
        s = self.spec
        bk.wrapper.plan(lay.kv_indptr, lay.kv_indices, lay.kv_last_page_len, s.num_heads, s.num_kv_heads, s.head_dim, self.kv.page_size,
                        q_data_type=self.kv.k.dtype, kv_data_type=self.kv.k.dtype)

    def _capture(self, bk: _Bucket) -> None:
        """Warm up and capture the bucket's frame graph on scratch rows only (the capture itself does not execute)."""
        pads = [self._pad_next + i for i in range(bk.b)]
        for p in pads:
            self.kv.reserve(p, 0)
        bk.positions.zero_()
        bk.live.zero_()
        bk.steps.zero_()
        bk.history.fill_(PAD)
        bk.codes.zero_()
        self._plan(bk, pads)
        self._frame(bk)
        torch.cuda.synchronize()
        bk.positions.zero_()
        bk.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bk.graph, capture_error_mode="thread_local"):  # other threads' GPU work cannot invalidate it
            self._frame(bk)
        torch.cuda.synchronize()
        for p in pads:
            self.kv.release(p)

    @torch.inference_mode()
    def capture_all(self) -> None:
        """Capture every bucket now (at startup, before other threads use the GPU)."""
        if self.use_graphs:
            for bk in self.buckets.values():
                if bk.graph is None:
                    self._capture(bk)

    def bucket_for(self, n: int) -> int:
        return min(b for b in self.buckets if b >= n)

    @torch.inference_mode()
    def decode_k(self, seqs: list[int], state: dict, k: int) -> torch.Tensor:
        """K frames for ``seqs``. ``state`` holds per-row tensors ``codes, seeds, steps, history, live`` (updated in place).

        Returns [K, len(seqs)] codes (``PAD`` after a row's EOS), copied to the host once.
        """
        n = len(seqs)
        bk = self.buckets[self.bucket_for(n)]
        if self.use_graphs and bk.graph is None:
            self._capture(bk)
        pads = [self._pad_next + i for i in range(bk.b - n)]
        rows = list(seqs) + pads
        for p in pads:
            self.kv.reserve(p, 0)
        bk.codes[:n].copy_(state["codes"])
        bk.codes[n:].zero_()
        bk.positions[:n].copy_(torch.tensor([self.kv.lengths[s] for s in seqs], dtype=torch.int32))
        bk.positions[n:].copy_(torch.tensor([self.kv.lengths.get(p, 0) for p in pads], dtype=torch.int32))
        for name in ("seeds", "steps", "history", "live"):
            getattr(bk, name)[:n].copy_(state[name])
        bk.live[n:] = False
        outs = []
        for _ in range(k):
            self._plan(bk, rows)
            if self.use_graphs:
                bk.graph.replay()
            else:
                self._frame(bk)
            outs.append(bk.out[:n].clone())
        for p in pads:
            self.kv.release(p)
        for name in ("codes", "steps", "history", "live"):
            state[name].copy_(getattr(bk, name)[:n])
        return torch.stack(outs).cpu()
