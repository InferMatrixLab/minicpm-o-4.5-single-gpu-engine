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

from engine.runtime.hostcopy import plan as fi_plan
from engine.runtime.paged import DecoderSpec, PagedKV, capture_stream, fuse_weights
from engine.talker_sampler import PAD, TalkerSampling, draw, repetition_penalty, top_p_top_k_filter

PAD_SEQ_BASE = 1 << 40  # scratch sequence ids for bucket padding rows


def processed_logits_static(logits, history, steps, eos, cfg: TalkerSampling):
    """``talker_sampler.processed_logits`` without data-dependent host branches (graph-safe)."""
    x = logits.float() / cfg.temperature
    y = top_p_top_k_filter(repetition_penalty(x, history, cfg.repetition_penalty), cfg.top_p, cfg.top_k, cfg.min_keep)
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
        self.max_pages = max_pages
        # whole-K capture: per frame of a block its own attention wrapper and layout buffers (planned before the
        # replay), a fixed [K, b] output, and one graph per K
        self.steps_attn: list[tuple] = []
        self.out_k = None
        self.kgraphs: dict[tuple[int, tuple], torch.cuda.CUDAGraph] = {}  # (K, frames with a page crossing) -> graph

    def step_attn(self, dec: "TalkerDecoder", j: int) -> tuple:
        """(wrapper, indptr, indices, last) of frame ``j`` of a block (made on first use)."""
        import flashinfer

        while len(self.steps_attn) <= j:
            dev = dec.device
            indptr = torch.zeros(self.b + 1, dtype=torch.int32, device=dev)
            indices = torch.zeros(self.max_pages, dtype=torch.int32, device=dev)
            last = torch.zeros(self.b, dtype=torch.int32, device=dev)
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(dec.workspace, "NHD", use_cuda_graph=True, paged_kv_indptr_buffer=indptr,
                                                              paged_kv_indices_buffer=indices, paged_kv_last_page_len_buffer=last)
            self.steps_attn.append((w, indptr, indices, last))
        return self.steps_attn[j]


class TalkerDecoder:
    def __init__(self, talker, kv: PagedKV, spec: DecoderSpec, cfg: TalkerSampling = TalkerSampling(), buckets=(1, 2, 4, 8, 16, 32, 64),
                 max_len: int = 4096, use_graphs: bool = True, whole_k: bool = False):
        """``whole_k``: one CUDA graph per (bucket, K) holds all K frames of a block (each frame attends through its own
        wrapper, planned before the replay) and writes a fixed [K, rows] output; otherwise one frame graph replayed K
        times."""
        self.talker, self.kv, self.spec, self.cfg = talker, kv, spec, cfg
        self.layers = fuse_weights(talker.w, spec)
        self.final_norm = talker.w[f"{spec.prefix}norm.weight"]
        self.head = talker.head
        self.emb = talker.w["emb_code.0.weight"]
        self.eos = talker.cfg.eos
        self.device = kv.k.device
        self.max_len, self.use_graphs, self.whole_k = max_len, use_graphs, whole_k
        self.workspace = torch.empty(128 << 20, dtype=torch.uint8, device=self.device)
        self.buckets = {b: _Bucket(self, b) for b in buckets}
        self._pad_next = PAD_SEQ_BASE

    # one frame for the rows of a bucket, all inputs read from the bucket's static buffers
    def _frame(self, bk: _Bucket, attn: tuple | None = None, out: torch.Tensor | None = None, advance_last: bool | None = None) -> None:
        """``attn``: (wrapper, indptr, indices, last) of a whole-K frame (default: the bucket's own); ``out``: where the
        sampled codes go (default ``bk.out``); ``advance_last``: carry the layout to the next frame by the last-page
        increment (default: for the bucket's own buffers)."""
        import flashinfer as fi

        wrapper, indptr, indices, last = attn or (bk.wrapper, bk.indptr, bk.indices, bk.last)
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
            fi.append_paged_kv_cache(k, v, bk.batch_idx, bk.positions, (self.kv.k[i], self.kv.v[i]), indices, indptr, last)
            a = wrapper.run(q, (self.kv.k[i], self.kv.v[i]))
            x = torch.addmm(x, a.reshape(-1, nq), L["o"].t())  # residual add in the GEMM epilogue
            h = fi.rmsnorm(x, L["ln2"], s.rms_eps)
            x = torch.addmm(x, fi.silu_and_mul(F.linear(h, L["gate_up"])), L["down"].t())
        logits = F.linear(fi.rmsnorm(x, self.final_norm, s.rms_eps), self.head)
        tok = draw(processed_logits_static(logits, bk.history, bk.steps, self.eos, self.cfg), bk.seeds, bk.steps)
        tok = torch.where(bk.live, tok, torch.full_like(tok, PAD))
        bk.history.copy_(torch.where(bk.live[:, None], torch.cat([bk.history[:, 1:], tok[:, None]], dim=1), bk.history))
        bk.steps.add_(bk.live.to(torch.int64))
        bk.live.logical_and_(tok != self.eos)
        (bk.out if out is None else out).copy_(tok)
        bk.codes.copy_(tok)
        bk.positions.add_(1)
        if advance_last if advance_last is not None else attn is None:
            last.add_(1)  # the next frame's last-page fill; the host re-plans whenever a row starts a new page

    def _plan(self, bk: _Bucket, seqs: list[int]) -> None:
        lay = self.kv.layout(seqs, [1] * len(seqs), device=self.device)
        n = lay.kv_indices.numel()
        bk.indptr.copy_(lay.kv_indptr)
        bk.indices[:n].copy_(lay.kv_indices)
        bk.last.copy_(lay.kv_last_page_len)
        s = self.spec
        fi_plan(bk.wrapper, lay.kv_indptr_h, lay.kv_indices, lay.kv_last_page_len_h, s.num_heads, s.num_kv_heads, s.head_dim, self.kv.page_size,
                        q_data_type=self.kv.k.dtype, kv_data_type=self.kv.k.dtype)

    def _plan_lengths(self, bk: _Bucket, rows: list[int], lengths: list[int], attn: tuple | None = None) -> None:
        """Plan for KV lengths ``lengths`` (after this frame's append) over the rows' already reserved pages
        (``attn``: a whole-K frame's wrapper and buffers instead of the bucket's)."""
        wrapper, b_indptr, b_indices, b_last = attn or (bk.wrapper, bk.indptr, bk.indices, bk.last)
        ps = self.kv.page_size
        npages = [-(-L // ps) for L in lengths]
        indices = [p for r, n in zip(rows, npages) for p in self.kv.pages[r][:n]]
        indptr = torch.tensor([0] + list(torch.tensor(npages).cumsum(0).tolist()), dtype=torch.int32)
        last = torch.tensor([L - (n - 1) * ps for L, n in zip(lengths, npages)], dtype=torch.int32)
        idx = torch.tensor(indices, dtype=torch.int32)
        b_indptr.copy_(indptr, non_blocking=True)  # queued, no wait for the GPU (engine.runtime.hostcopy)
        b_indices[:idx.numel()].copy_(idx, non_blocking=True)
        b_last.copy_(last, non_blocking=True)
        s = self.spec
        # the plan reads the index pointers and last-page lengths on the host, the indices on the device
        fi_plan(wrapper, indptr, idx.to(self.device, non_blocking=True), last, s.num_heads, s.num_kv_heads, s.head_dim, ps,
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
        with torch.cuda.graph(bk.graph, stream=capture_stream("talker"), capture_error_mode="thread_local"):  # other threads' GPU work cannot invalidate it
            self._frame(bk)
        torch.cuda.synchronize()
        for p in pads:
            self.kv.release(p)

    @staticmethod
    def _plan_steps(k: int, crossings: tuple) -> list[int]:
        """Frame j of a block attends through the wrapper of the last planned frame at or before it: frame 0 and every
        frame where some row starts a new page are planned (as the per-frame path re-plans), the others carry the layout
        forward by the in-graph last-page increment."""
        w, out = 0, []
        for j in range(k):
            if j in crossings:
                w = j
            out.append(w)
        return out

    def _capture_k(self, bk: _Bucket, k: int, crossings: tuple = ()) -> None:
        """Warm up and capture the bucket's whole-K graph for one page-crossing pattern on scratch rows: K frames, frame
        ``j`` through the wrapper of its last planned frame, codes into ``out_k[j]``."""
        pads = [self._pad_next + i for i in range(bk.b)]
        for p in pads:
            self.kv.reserve(p, k)
        steps = self._plan_steps(k, crossings)
        attn = [bk.step_attn(self, j) for j in range(k)]
        if bk.out_k is None or bk.out_k.shape[0] < k:
            bk.out_k = torch.full((max(k, 4), bk.b), PAD, dtype=torch.int64, device=self.device)

        def plan_all():
            for j in sorted(set(steps)):
                self._plan_lengths(bk, pads, [j + 1] * bk.b, attn[j])

        def frames():
            for j in range(k):
                self._frame(bk, attn[steps[j]], bk.out_k[j], advance_last=j + 1 < k and steps[j + 1] == steps[j])

        plan_all()
        bk.live.zero_()
        bk.steps.zero_()
        bk.history.fill_(PAD)
        bk.codes.zero_()
        bk.positions.zero_()
        frames()  # warm-up
        torch.cuda.synchronize()
        bk.positions.zero_()
        plan_all()  # the warm-up advanced the carried last-page fills
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=capture_stream("talker"), capture_error_mode="thread_local"):
            frames()
        torch.cuda.synchronize()
        bk.kgraphs[(k, crossings)] = g
        for p in pads:
            self.kv.release(p)

    @torch.inference_mode()
    def capture_all(self, ks=()) -> None:
        """Capture every bucket now (at startup, before other threads use the GPU)."""
        if self.use_graphs:
            for bk in self.buckets.values():
                if bk.graph is None:
                    self._capture(bk)
                if self.whole_k:  # every page-crossing pattern of a block (at startup, not while serving)
                    from itertools import combinations

                    for k in ks:
                        for m in range(k):
                            for crossings in combinations(range(1, k), m):
                                if (k, crossings) not in bk.kgraphs:
                                    self._capture_k(bk, k, crossings)

    def bucket_for(self, n: int) -> int:
        return min(b for b in self.buckets if b >= n)

    @torch.inference_mode()
    def decode_k(self, seqs: list[int], state: dict, k: int) -> torch.Tensor:
        """K frames for ``seqs`` (see ``_decode_k``); more rows than the largest bucket run as consecutive slices."""
        top = max(self.buckets)
        if len(seqs) <= top:
            return self._decode_k(seqs, state, k)
        outs = []
        for a in range(0, len(seqs), top):
            part = {name: v[a:a + top] for name, v in state.items()}  # views: updated in place
            outs.append(self._decode_k(seqs[a:a + top], part, k))
        return torch.cat(outs, dim=1)

    def _decode_k(self, seqs: list[int], state: dict, k: int) -> torch.Tensor:
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
        bk.positions[:n].copy_(torch.tensor([self.kv.lengths[s] for s in seqs], dtype=torch.int32), non_blocking=True)
        bk.positions[n:].copy_(torch.tensor([self.kv.lengths.get(p, 0) for p in pads], dtype=torch.int32), non_blocking=True)
        for name in ("seeds", "steps", "history", "live"):
            getattr(bk, name)[:n].copy_(state[name])
        bk.live[n:] = False
        start = [self.kv.lengths[r] for r in rows]
        for r in rows:
            self.kv.reserve(r, k)  # pages for all K frames up front
        ps = self.kv.page_size
        if self.use_graphs and self.whole_k:  # the planned frames' layouts up front, then one replay of the whole block
            crossings = tuple(j for j in range(1, k) if any((L + j) % ps == 0 for L in start))
            if (k, crossings) not in bk.kgraphs:
                self._capture_k(bk, k, crossings)
            for j in sorted(set(self._plan_steps(k, crossings))):
                self._plan_lengths(bk, rows, [L + j + 1 for L in start], bk.step_attn(self, j))
            bk.kgraphs[(k, crossings)].replay()
            for r, L in zip(rows, start):
                self.kv.lengths[r] = L + k
            for p in pads:
                self.kv.release(p)
            for name in ("codes", "steps", "history", "live"):
                state[name].copy_(getattr(bk, name)[:n])
            return bk.out_k[:k, :n].cpu()
        outs = []
        for j in range(k):
            # re-plan only when the layout's page structure changes (a row starts a new page) or on the first frame;
            # otherwise the graph's own last-page / position increments carry the layout forward
            # (eager mode re-plans every frame: a non-graph wrapper keeps its own copy of the layout buffers)
            if j == 0 or not self.use_graphs or any((L + j) % ps == 0 for L in start):
                self._plan_lengths(bk, rows, [L + j + 1 for L in start])
            if self.use_graphs:
                bk.graph.replay()
            else:
                self._frame(bk)
            outs.append(bk.out[:n].clone())
        for r, L in zip(rows, start):
            self.kv.lengths[r] = L + k
        for p in pads:
            self.kv.release(p)
        for name in ("codes", "steps", "history", "live"):
            state[name].copy_(getattr(bk, name)[:n])
        return torch.stack(outs).cpu()
