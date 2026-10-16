"""Graph-captured greedy Thinker decode: one token per row per replay, the final-norm hidden state kept for the Talker.

Same layer math as ``engine.runtime.paged.DecoderRunner`` (reference rounding for norms and RoPE), with FlashInfer's
paged decode attention in CUDA-graph mode.  Per bucket the static buffers hold the fed token, its position, the
page table, the greedy next token and the hidden state of the fed token (``emb_text(id) + projector(h)`` uses it).
Bucket padding rows decode a scratch sequence and never touch live rows.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.runtime.fp8 import linear
from engine.runtime.paged import DecoderSpec, PagedKV, fuse_weights, hf_rms_norm, hf_rope, hf_rope_tables

PAD_SEQ_BASE = 1 << 41


class _Bucket:
    def __init__(self, dec: "ThinkerDecoder", b: int):
        import flashinfer

        dev = dec.device
        self.b = b
        self.indptr = torch.zeros(b + 1, dtype=torch.int32, device=dev)
        self.indices = torch.zeros(b * (dec.max_len // dec.kv.page_size + 2), dtype=torch.int32, device=dev)
        self.last = torch.zeros(b, dtype=torch.int32, device=dev)
        self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(dec.workspace, "NHD", use_cuda_graph=dec.use_graphs, use_tensor_cores=True,
                                                                    paged_kv_indptr_buffer=self.indptr, paged_kv_indices_buffer=self.indices,
                                                                    paged_kv_last_page_len_buffer=self.last)
        self.tokens = torch.zeros(b, dtype=torch.int64, device=dev)
        self.positions = torch.zeros(b, dtype=torch.int32, device=dev)
        self.batch_idx = torch.arange(b, dtype=torch.int32, device=dev)
        self.next = torch.zeros(b, dtype=torch.int64, device=dev)
        self.hidden = torch.zeros(b, dec.spec.hidden_size, dtype=dec.kv.k.dtype, device=dev)
        self.graph = None
        self.rows: list[int] | None = None  # rows of the last plan (None: re-plan)
        self.pads = [PAD_SEQ_BASE + 1000 * b + i for i in range(b)]  # persistent scratch rows of this bucket


class ThinkerDecoder:
    def __init__(self, thinker, kv: PagedKV, spec: DecoderSpec, buckets=(1, 2, 4, 8, 16, 32, 64), max_len: int = 8192, use_graphs: bool = True,
                 layers: list[dict] | None = None, fused: bool = True):
        """``fused``: the reference-rounding Triton kernels of ``engine.runtime.kernels`` (about 11 kernels per layer
        instead of about 50; the same roundings, a norm's fp32 sum order aside) in place of the torch elementwise ops."""
        self.spec, self.kv = spec, kv
        self.fused = fused and kv.k.dtype == torch.bfloat16
        self.layers = layers if layers is not None else fuse_weights(thinker.w, spec)
        self.final_norm = thinker.w["model.norm.weight"]
        self.emb, self.head = thinker.w["model.embed_tokens.weight"], thinker.w["lm_head.weight"]
        self.device, self.max_len, self.use_graphs = kv.k.device, max_len, use_graphs
        self.workspace = torch.empty(256 << 20, dtype=torch.uint8, device=self.device)
        self.buckets = {b: _Bucket(self, b) for b in buckets}

    def _step(self, bk: _Bucket) -> None:
        if self.fused:
            self._step_fused(bk)
            return
        import flashinfer as fi

        s = self.spec
        nq, nk = s.num_heads * s.head_dim, s.num_kv_heads * s.head_dim
        x = F.embedding(bk.tokens, self.emb)
        tables = hf_rope_tables(bk.positions, s.head_dim, s.rope_theta, x.dtype)  # once per step, not per layer
        for i, L in enumerate(self.layers):
            h = hf_rms_norm(x, L["ln1"], s.rms_eps)
            qkv = linear(h, L["qkv"])
            q = hf_rms_norm(qkv[:, :nq].reshape(-1, s.head_dim), L["q_norm"], s.rms_eps).view(-1, s.num_heads, s.head_dim)
            k = hf_rms_norm(qkv[:, nq:nq + nk].reshape(-1, s.head_dim), L["k_norm"], s.rms_eps).view(-1, s.num_kv_heads, s.head_dim)
            v = qkv[:, nq + nk:].reshape(-1, s.num_kv_heads, s.head_dim).contiguous()
            q, k = hf_rope(q, k, bk.positions, s.rope_theta, tables)
            q, k = q.contiguous(), k.contiguous()
            fi.append_paged_kv_cache(k, v, bk.batch_idx, bk.positions, (self.kv.k[i], self.kv.v[i]), bk.indices, bk.indptr, bk.last)
            a = bk.wrapper.run(q, (self.kv.k[i], self.kv.v[i]))
            x = x + linear(a.reshape(-1, nq), L["o"])
            h = hf_rms_norm(x, L["ln2"], s.rms_eps)
            g, u = linear(h, L["gate_up"]).chunk(2, dim=-1)
            x = x + linear(F.silu(g) * u, L["down"])
        hid = hf_rms_norm(x, self.final_norm, s.rms_eps)
        bk.hidden.copy_(hid)
        bk.next.copy_(F.linear(hid, self.head).argmax(-1))
        bk.last.add_(1)  # the next step's last-page fill when the host does not re-plan (same rows, no new page)

    def _step_fused(self, bk: _Bucket) -> None:
        """``_step`` with the residual add fused into the next norm, q/k norm + RoPE in one kernel, and SiLU-and-multiply."""
        import flashinfer as fi

        from engine.runtime.kernels import add_rms_norm_hf, qk_norm_rope_hf, rms_norm_hf, silu_mul_hf

        s = self.spec
        nq = s.num_heads * s.head_dim
        x = F.embedding(bk.tokens, self.emb)
        cos, sin = hf_rope_tables(bk.positions, s.head_dim, s.rope_theta, x.dtype)
        h = rms_norm_hf(x, self.layers[0]["ln1"], s.rms_eps)
        last = len(self.layers) - 1
        for i, L in enumerate(self.layers):
            q, k, v = qk_norm_rope_hf(linear(h, L["qkv"]), L["q_norm"], L["k_norm"], cos, sin, s.num_heads, s.num_kv_heads, s.head_dim,
                                      s.rms_eps)
            fi.append_paged_kv_cache(k, v, bk.batch_idx, bk.positions, (self.kv.k[i], self.kv.v[i]), bk.indices, bk.indptr, bk.last)
            a = bk.wrapper.run(q, (self.kv.k[i], self.kv.v[i]))
            x, h = add_rms_norm_hf(x, linear(a.reshape(-1, nq), L["o"]), L["ln2"], s.rms_eps)
            d = linear(silu_mul_hf(linear(h, L["gate_up"])), L["down"])
            x, h = add_rms_norm_hf(x, d, self.final_norm if i == last else self.layers[i + 1]["ln1"], s.rms_eps)
        bk.hidden.copy_(h)
        bk.next.copy_(F.linear(h, self.head).argmax(-1))
        bk.last.add_(1)

    def _plan(self, bk: _Bucket, seqs: list[int]) -> None:
        lay = self.kv.layout(seqs, [1] * len(seqs), device=self.device)
        bk.indptr.copy_(lay.kv_indptr)
        bk.indices[: lay.kv_indices.numel()].copy_(lay.kv_indices)
        bk.last.copy_(lay.kv_last_page_len)
        s = self.spec
        bk.wrapper.plan(lay.kv_indptr, lay.kv_indices, lay.kv_last_page_len, s.num_heads, s.num_kv_heads, s.head_dim, self.kv.page_size,
                        q_data_type=self.kv.k.dtype, kv_data_type=self.kv.k.dtype)

    def _capture(self, bk: _Bucket) -> None:
        pads = [PAD_SEQ_BASE + i for i in range(bk.b)]
        for p in pads:
            self.kv.reserve(p, 0)
        bk.tokens.zero_()
        bk.positions.zero_()
        self._plan(bk, pads)
        self._step(bk)
        torch.cuda.synchronize()
        bk.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bk.graph, capture_error_mode="thread_local"):  # other threads' GPU work cannot invalidate it
            self._step(bk)
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

    PAD_RESET = 64  # scratch rows restart (and the bucket re-plans) after this many steps: they hold at most 4 pages

    @torch.inference_mode()
    def step(self, seqs: list[int], tokens: list[int]) -> tuple[list[int], torch.Tensor]:
        """Feed ``tokens[i]`` to sequence ``seqs[i]``; returns (greedy next tokens, hidden states of the fed tokens [n, H]).

        The FlashInfer plan is redone only when the rows change or a row starts a new KV page (graph mode); otherwise the
        graph's own last-page increment carries the layout forward.  Padding rows are persistent scratch sequences.
        """
        n = len(seqs)
        bk = self.buckets[min(b for b in self.buckets if b >= n)]
        if self.use_graphs and bk.graph is None:
            self._capture(bk)
        pads = bk.pads[: bk.b - n]
        for p in pads:
            if self.kv.lengths.get(p, 0) >= self.PAD_RESET:
                self.kv.release(p)
                bk.rows = None
            self.kv.reserve(p, 0)
        rows = list(seqs) + pads
        pos = [self.kv.lengths[r] for r in rows]
        bk.tokens[:n].copy_(torch.tensor(tokens, dtype=torch.int64))
        bk.tokens[n:].zero_()
        bk.positions.copy_(torch.tensor(pos, dtype=torch.int32))
        ps = self.kv.page_size
        if not self.use_graphs or bk.rows != rows or any(L % ps == 0 for L in pos):
            self._plan(bk, rows)
            bk.rows = rows
        else:
            for r in rows:  # the same bookkeeping as kv.layout(rows, [1] * len(rows))
                self.kv.reserve(r, 1)
                self.kv.lengths[r] += 1
        if self.use_graphs:
            bk.graph.replay()
        else:
            self._step(bk)
        for other in self.buckets.values():  # another bucket may have planned rows these pads share
            if other is not bk:
                other.rows = None
        return bk.next[:n].tolist(), bk.hidden[:n].clone()
