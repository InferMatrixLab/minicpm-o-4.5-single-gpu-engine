"""Fast Token2Wav DiT solve: packed ragged rows and in-place attention caches in a page-size-1 slot pool.

The reference DiT (``stepaudio2.cosyvoice2.flow.decoder_dit``) prepends each window's keys to a per-request cache tensor
(``[new, cache]``) and trims it (``token2wav_batch._trim``: the first ``P`` entries plus the last 100, ``P`` = the
prompt's mel frames) after every window, per Euler step and block: a batched call spends most of its time copying
caches.  The DiT has no positional encoding, so attention depends only on the *set* of cached keys.  Here every cached
frame of a CFG row is one slot of a shared pool (one key/value tensor per (step, block), FlashInfer's paged layout
with page size 1); a window's new keys are written into fresh slots, attention reads each row's slot list, and the
reference's tensor order is kept as a host-side list of slot ids so trimming evicts exactly the reference's frames.
Conv caches (two frames per row, step and block) live in a row-slot pool, gathered and scattered inside the solve.

Everything else follows the reference block (``DiTBlock.forward_chunk`` as ``BatchedToken2Wav._estimator_temb`` runs
it under FP16 autocast): norms and modulation in fp32, GEMMs in fp16, caches stored in fp16
(``engine.runtime.dit_kernels``).  The timestep embedding is the same for every row of a step, so the adaLN
modulations are computed once per step at construction.  Causal convolutions (kernel 3, two cached frames) are a
gather plus one GEMM over the packed tokens.  Rows are ragged: steady windows, first windows, final flushes and prompt
setups share one code path.

Token layout of a call: ``[cond rows | empty rows | pad row | uncond rows | empty rows | pad row]``; each half holds
``Nb / 2`` tokens (the pad row takes the rest), so the CFG halves sit at fixed offsets and a call can replay a CUDA
graph captured for its ``(rows, tokens)`` bucket.  The attention plan is made on the host before each replay.  A call
takes the smallest row bucket with a token bucket for its tokens; ``setup_tokens`` raises every row bucket's token
top by that many, so prompt setups can ride a decode's solve (``BatchedToken2Wav.decode(setups=...)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from engine.runtime.dit_kernels import conv_gather, ln_mish, ln_mod, qk_norm_scatter
from engine.runtime.hostcopy import host, plan as fi_plan, to_device

TRIM_SUFFIX = 100
ROW_BUCKETS = (2, 4, 8, 16, 32, 64)  # request rows (2 x requests), excluding the two pad rows
MAX_GRAPH_T = 56  # frames per row of steady, first and final windows
SETUP_TOKENS = 4096  # small row buckets also cover prompt setups (whole prompts, both CFG halves) up to this many tokens
SETUP_ROWS = 8


def bucket_top(rb: int, extra: int = 0) -> int:
    """Most tokens (both halves) a row bucket's graphs take (``extra``: fused prompt-setup tokens, ``FastDiT.setup_tokens``)."""
    return max(rb * MAX_GRAPH_T + extra, SETUP_TOKENS if rb <= SETUP_ROWS else 0)


TOP = max(bucket_top(rb) for rb in ROW_BUCKETS)


def token_buckets(rb: int, extra: int = 0) -> list[int]:
    """Token buckets (both halves) of a row bucket: multiples of 256 (128 below 512) up to ``bucket_top(rb, extra)``."""
    top = bucket_top(rb, extra)
    out = [b for b in list(range(128, 512, 128)) + list(range(512, top, 256)) if b < top]
    return out + [top]


class SlotPool:
    """Key/value slots for every (Euler step, block): ``k[s * depth + b]`` is [n_slots, 1, heads, head_dim]."""

    def __init__(self, n_slots: int, steps: int, depth: int, heads: int, head_dim: int, dtype=torch.float16, device="cuda"):
        self.n_slots, self.steps, self.depth = n_slots, steps, depth
        self.k = torch.zeros(steps * depth, n_slots, 1, heads, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.free = list(range(n_slots - 1, -1, -1))

    def alloc(self, n: int) -> list[int]:
        if n > len(self.free):
            raise RuntimeError(f"Token2Wav cache pool exhausted ({n} slots wanted, {len(self.free)} free of {self.n_slots})")
        out = self.free[len(self.free) - n:][::-1] if n else []
        del self.free[len(self.free) - n:]
        return out

    def release(self, ids: list[int]) -> None:
        self.free.extend(ids)

    def in_use(self) -> int:
        return self.n_slots - len(self.free)


@dataclass
class FastRow:
    """One CFG row's caches: key slots in the reference tensor order and its conv-cache row slot."""
    order: list[int] = field(default_factory=list)
    cnn: int = -1


@dataclass
class FastCache:
    """A request's estimator caches (conditional and unconditional rows) and its trim length (prompt mel frames)."""
    rows: tuple[FastRow, FastRow]
    keep: int = 0

    @property
    def length(self) -> int:
        return len(self.rows[0].order)


class FastDiT:
    def __init__(self, cfm, steps: int, n_slots: int = 32768, max_streams: int = 96, device="cuda", use_graphs: bool = False,
                 setup_tokens: int = 0):
        """``setup_tokens``: tokens (both halves) every row bucket's graphs take beyond its windows, for prompt setups
        in a decode's solve (0: setups ride only the small row buckets' ``SETUP_TOKENS``)."""
        est = cfm.estimator
        self.cfm, self.est, self.steps = cfm, est, steps
        self.depth = len(est.blocks)
        b0 = est.blocks[0]
        self.heads, self.hd = b0.attn.num_heads, b0.attn.head_dim
        self.hidden = est.in_proj.out_features
        self.mel = est.out_channels
        self.dtype = next(est.parameters()).dtype
        assert self.dtype == torch.float16, "the fast DiT runs the FP16 flow"
        self.device, self.use_graphs = device, use_graphs
        self.setup_tokens = int(setup_tokens)
        self.top = max(bucket_top(rb, self.setup_tokens) for rb in ROW_BUCKETS)
        self.scratch = self.top  # slots reserved for pad-row tokens (never attended by real rows)
        self.pool = SlotPool(n_slots + self.scratch, steps, self.depth, self.heads, self.hd, self.dtype, device)
        self.pad_slots = self.pool.alloc(self.scratch)
        n_rs = 2 * max_streams + ROW_BUCKETS[-1] + 2
        self.cnn = torch.zeros(steps, self.depth, n_rs, 2, 2, self.hidden, dtype=self.dtype, device=device)  # [.., rowslot, conv, frame, C]
        self.cnn_free = list(range(n_rs - 1, -1, -1))
        self.cnn_scratch = [self.cnn_free.pop() for _ in range(ROW_BUCKETS[-1] + 2)]  # read / write slots of empty and pad rows
        self.cfg_rate = float(cfm.inference_cfg_rate)
        self.blocks = []
        for blk in est.blocks:
            a = blk.attn
            c1, c2 = blk.conv.block[1], blk.conv.block[6]
            assert a.q_norm.eps == a.k_norm.eps
            self.blocks.append({
                "qkv": (torch.cat([a.to_q.weight, a.to_k.weight, a.to_v.weight]).contiguous(), torch.cat([a.to_q.bias, a.to_k.bias, a.to_v.bias]).contiguous()),
                "qn": (a.q_norm.weight, a.q_norm.bias, a.q_norm.eps), "kn": (a.k_norm.weight, a.k_norm.bias, a.k_norm.eps),
                "proj": (a.proj.weight, a.proj.bias),
                "c1": (c1.weight.permute(0, 2, 1).reshape(c1.out_channels, -1).contiguous(), c1.bias),  # [out, k * in] (frame-major)
                "c2": (c2.weight.permute(0, 2, 1).reshape(c2.out_channels, -1).contiguous(), c2.bias),
                "cln": (blk.conv.block[3].weight, blk.conv.block[3].bias, blk.conv.block[3].eps),
                "fc1": (blk.mlp.fc1.weight, blk.mlp.fc1.bias), "fc2": (blk.mlp.fc2.weight, blk.mlp.fc2.bias),
                "eps": (blk.norm1.eps, blk.norm2.eps, blk.norm3.eps)})
        self.final = (est.final_layer.linear.weight, est.final_layer.linear.bias, est.final_layer.norm_final.eps)
        self.in_proj = (est.in_proj.weight, est.in_proj.bias)
        self._modulations()
        import flashinfer

        self.workspace = torch.empty(256 << 20, dtype=torch.uint8, device=device)
        self.attn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "NHD")  # eager calls (setups, large shapes)
        self.graphs: dict[tuple[int, int], tuple] = {}
        self.graph_pool = torch.cuda.graph_pool_handle() if use_graphs else None
        self._gw: dict[int, object] = {}  # row bucket -> CUDA-graph attention wrapper
        self._gbuf: dict[str, torch.Tensor] = {}
        self.counts = {"graph": 0, "eager": 0}

    @torch.inference_mode()
    def _modulations(self) -> None:
        """The solver's t recurrence (as ``EulerGraphs._tembs``) and every step's adaLN outputs (one row: they are the
        same for every row).  ``1 + scale`` is kept in fp16, as ``modulate`` computes it."""
        t_span = torch.linspace(0, 1, self.steps + 1, device=self.device, dtype=self.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t, dt = t_span[0].unsqueeze(0), t_span[1] - t_span[0]
        self.dts, self.mods, self.final_mods = [], [], []
        with torch.amp.autocast("cuda", dtype=torch.float16):
            for step in range(1, self.steps + 1):
                temb = self.est.t_embedder(t).unsqueeze(1)  # [1, 1, hidden]
                per = []
                for blk in self.est.blocks:
                    sh_a, sc_a, g_a, sh_m, sc_m, g_m, sh_c, sc_c, g_c = blk.adaLN_modulation(temb).reshape(9, -1)
                    per.append({"msa": (sh_a, 1 + sc_a, g_a), "mlp": (sh_m, 1 + sc_m, g_m), "conv": (sh_c, 1 + sc_c, g_c)})
                self.mods.append(per)
                sh, sc = self.est.final_layer.adaLN_modulation(temb).reshape(2, -1)
                self.final_mods.append((sh, 1 + sc))
                self.dts.append(dt.clone())
                t = t + dt
                if step < self.steps:
                    dt = t_span[step + 1] - t

    # --- caches -------------------------------------------------------------------------------------------------------
    def new_cache(self) -> FastCache:
        if len(self.cnn_free) < 2:
            raise RuntimeError("Token2Wav conv-cache row slots exhausted")
        rows = (FastRow(cnn=self.cnn_free.pop()), FastRow(cnn=self.cnn_free.pop()))
        for r in rows:
            self.cnn[:, :, r.cnn].zero_()
        return FastCache(rows)

    def release(self, c: FastCache) -> None:
        for r in c.rows:
            self.pool.release(r.order)
            r.order = []
            if r.cnn >= 0:
                self.cnn_free.append(r.cnn)
                r.cnn = -1

    def clone(self, c: FastCache) -> FastCache:
        """An independent copy (new slots holding the same keys, values and conv caches)."""
        out = self.new_cache()
        out.keep = c.keep
        for src, dst in zip(c.rows, out.rows):
            self.cnn[:, :, dst.cnn] = self.cnn[:, :, src.cnn]
            dst.order = self.pool.alloc(len(src.order))
            if src.order:
                s = torch.tensor(src.order, device=self.device)
                d = torch.tensor(dst.order, device=self.device)
                self.pool.k[:, d] = self.pool.k[:, s]
                self.pool.v[:, d] = self.pool.v[:, s]
        return out

    # --- planning -----------------------------------------------------------------------------------------------------
    def _bucket(self, n_rows: int, n_tok: int, max_t: int):
        """The smallest row bucket holding ``n_rows`` with a token bucket for ``n_tok``, and that token bucket."""
        if not self.use_graphs:
            return None
        for rb in ROW_BUCKETS:
            if rb >= n_rows:
                nb = next((b for b in token_buckets(rb, self.setup_tokens) if b >= n_tok), None)
                if nb is not None:
                    return rb, nb
        return None

    def _plan(self, caches: list[FastCache], Ts: list[int], keep: list[bool]):
        """Host plan of one call: new slots, the token layout (padded to a bucket when graphs apply), attention lists,
        conv gather metadata and the conv-cache row slots to read and write."""
        n = len(caches)
        R = 2 * n
        bucket = self._bucket(R, 2 * sum(Ts), max(Ts))
        rb, nb = bucket if bucket is not None else (R, 2 * sum(Ts))
        half_rows, half_tok = rb // 2 + 1, nb // 2  # each half: its request rows, empty rows, one pad row
        new = [self.pool.alloc(t) for t in Ts + Ts]
        rows = [c.rows[0] for c in caches] + [c.rows[1] for c in caches]
        pad_t = half_tok - sum(Ts)
        layout = []  # (tokens, kv slot list, new slots, cnn read slot, cnn write slot) per row in call order
        scr = iter(self.cnn_scratch)
        for hh in range(2):
            for i in range(n):
                r, nw = rows[hh * n + i], new[hh * n + i]
                layout.append((Ts[i], r.order + nw, nw, r.cnn, r.cnn if keep[i] else next(scr)))
            for _ in range(rb // 2 - n):
                s = next(scr)
                layout.append((0, [], [], s, s))
            ps = self.pad_slots[hh * half_tok: hh * half_tok + pad_t]
            s = next(scr)
            layout.append((pad_t, ps, ps, s, s))
        assert len(layout) == 2 * half_rows
        qo, kv, idx, slots, tok_row, tok_pos, starts, lens, rd, wr = [0], [0], [], [], [], [], [], [], [], []
        for r, (t, kvl, nw, a, b) in enumerate(layout):
            starts.append(qo[-1])
            lens.append(t)
            qo.append(qo[-1] + t)
            kv.append(kv[-1] + len(kvl))
            idx += kvl
            slots += nw
            tok_row += [r] * t
            tok_pos += list(range(t))
            rd.append(a)
            wr.append(b)
        dev = self.device
        it = lambda xs, dt=torch.int32: to_device(xs, dt, dev)  # noqa: E731  (queued, no synchronization)
        # the attention plan reads the index pointers and last-page lengths on the host (engine.runtime.hostcopy)
        plan = {"bucket": bucket, "rows": 2 * half_rows, "tokens": nb, "half": half_tok, "qo": host(qo), "kv": host(kv), "kv_idx": it(idx),
                "last": host([1] * (2 * half_rows)),
                "slots": it(slots, torch.int64), "meta": {"tok_row": it(tok_row), "tok_pos": it(tok_pos), "row_start": it(starts), "row_len": it(lens)},
                "rd": it(rd, torch.int64), "wr": it(wr, torch.int64), "new": new, "rows_obj": rows}
        return plan

    # --- the solve body (graph-capturable) ----------------------------------------------------------------------------
    def _body(self, buf: dict, attn) -> None:
        """Ten Euler steps from ``buf``: x0 [Nb/2, mel], static [Nb, 3 mel], slots, meta, rd / wr (conv row slots);
        writes ``buf['out']`` [Nb/2, mel] and the conv caches of the write slots."""
        H, half = self.hidden, buf["x0"].shape[0]
        N = 2 * half
        cin = self.cnn.index_select(2, buf["rd"])  # [steps, depth, R, 2, 2, C]
        cout = torch.empty_like(cin)
        q = torch.empty(N, self.heads, self.hd, dtype=self.dtype, device=self.device)
        im = torch.empty(N, 3 * H, dtype=self.dtype, device=self.device)
        x = buf["x0"]
        for step in range(self.steps):
            h = F.linear(torch.cat([torch.cat([x, x]), buf["static"]], dim=1), *self.in_proj)
            for b, blk in enumerate(self.blocks):
                mod, sb = self.mods[step][b], step * self.depth + b
                eps1, eps2, eps3 = blk["eps"]
                sh, sc1, g = mod["msa"]
                qkv = F.linear(ln_mod(h, sh, sc1, eps1), *blk["qkv"])
                kc, vc = self.pool.k[sb], self.pool.v[sb]
                qk_norm_scatter(qkv, blk["qn"], blk["kn"], q, kc, vc, buf["slots"], self.heads, self.hd)
                h = torch.addcmul(h, g, F.linear(attn.run(q, (kc, vc)).view(N, H), *blk["proj"]))
                sh, sc1, g = mod["conv"]
                a = ln_mod(h, sh, sc1, eps3)
                y = F.linear(conv_gather(a, cin[step, b, :, 0], cout[step, b, :, 0], buf["meta"], im), *blk["c1"])
                y = ln_mish(y, blk["cln"])
                h = torch.addcmul(h, g, F.linear(conv_gather(y, cin[step, b, :, 1], cout[step, b, :, 1], buf["meta"], im), *blk["c2"]))
                sh, sc1, g = mod["mlp"]
                a = ln_mod(h, sh, sc1, eps2)
                h = torch.addcmul(h, g, F.linear(F.gelu(F.linear(a, *blk["fc1"]), approximate="tanh"), *blk["fc2"]))
            sh, sc1 = self.final_mods[step]
            dphi = F.linear(ln_mod(h, sh, sc1, self.final[2]), self.final[0], self.final[1])
            x = x + self.dts[step] * ((1.0 + self.cfg_rate) * dphi[:half] - self.cfg_rate * dphi[half:])
        self.cnn.index_copy_(2, buf["wr"], cout)
        buf["out"].copy_(x)

    # --- graphs -------------------------------------------------------------------------------------------------------
    def _graph_wrapper(self, rb: int):
        if rb not in self._gw:
            import flashinfer

            R = rb + 2
            if "kv_idx" not in self._gbuf:
                self._gbuf["kv_idx"] = torch.zeros(self._kv_capacity(), dtype=torch.int32, device=self.device)
            ws = torch.empty(256 << 20, dtype=torch.uint8, device=self.device)
            self._gw[rb] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                ws, "NHD", use_cuda_graph=True, qo_indptr_buf=torch.zeros(R + 1, dtype=torch.int32, device=self.device),
                paged_kv_indptr_buf=torch.zeros(R + 1, dtype=torch.int32, device=self.device), paged_kv_indices_buf=self._gbuf["kv_idx"],
                paged_kv_last_page_len_buf=torch.ones(R, dtype=torch.int32, device=self.device))
            # graph mode fixes its limits at the first plan: make it the largest (tokens and cached keys) this bucket
            # will see; later plans must fit within it
            nb, cap = bucket_top(rb, self.setup_tokens), self._kv_capacity()
            qo = torch.tensor([0] * (rb // 2 + 1) + [nb // 2] * (rb // 2 + 1) + [nb], dtype=torch.int32, device=self.device)
            kv = torch.tensor([0] * (rb // 2 + 1) + [cap // 2] * (rb // 2 + 1) + [cap], dtype=torch.int32, device=self.device)
            idx = torch.tensor(self.pad_slots, dtype=torch.int32, device=self.device).repeat(cap // len(self.pad_slots) + 1)[:cap]
            fi_plan(self._gw[rb], qo, kv, idx, torch.ones(R, dtype=torch.int32, device=self.device), self.heads, self.heads, self.hd, 1,
                              causal=False, q_data_type=self.dtype, kv_data_type=self.dtype)
        return self._gw[rb]

    def _kv_capacity(self) -> int:
        return 2 * ROW_BUCKETS[-1] * 1400 + self.top

    def _static_bufs(self, rb: int, nb: int) -> dict:
        R, half = rb + 2, nb // 2
        top = self.top
        if "x0" not in self._gbuf:
            g = self._gbuf
            g["x0"] = torch.zeros(top // 2, self.mel, dtype=self.dtype, device=self.device)
            g["static"] = torch.zeros(top, 3 * self.mel, dtype=self.dtype, device=self.device)
            g["out"] = torch.zeros(top // 2, self.mel, dtype=self.dtype, device=self.device)
            g["slots"] = torch.zeros(top, dtype=torch.int64, device=self.device)
            for k in ("tok_row", "tok_pos"):
                g[k] = torch.zeros(top, dtype=torch.int32, device=self.device)
            for k in ("row_start", "row_len"):
                g[k] = torch.zeros(ROW_BUCKETS[-1] + 2, dtype=torch.int32, device=self.device)
            for k in ("rd", "wr"):
                g[k] = torch.zeros(ROW_BUCKETS[-1] + 2, dtype=torch.int64, device=self.device)
        g = self._gbuf
        return {"x0": g["x0"][:half], "static": g["static"][:nb], "out": g["out"][:half], "slots": g["slots"][:nb],
                "meta": {"tok_row": g["tok_row"][:nb], "tok_pos": g["tok_pos"][:nb], "row_start": g["row_start"][:R], "row_len": g["row_len"][:R]},
                "rd": g["rd"][:R], "wr": g["wr"][:R]}

    @torch.inference_mode()
    def capture(self, rb: int, nb: int) -> None:
        """Capture the solve for a (row, token) bucket on scratch rows (pad slots, scratch conv slots)."""
        if (rb, nb) in self.graphs:
            return
        buf = self._static_bufs(rb, nb)
        w = self._graph_wrapper(rb)
        R, half = rb + 2, nb // 2
        # a valid plan for capture: each half's tokens all in its pad row, attending to the reserved pad slots
        qo = [0] * (rb // 2 + 1) + [half] * (rb // 2 + 1) + [nb]
        dev = self.device
        qo_t = torch.tensor(qo, dtype=torch.int32, device=dev)
        idx = torch.tensor(self.pad_slots[:nb], dtype=torch.int32, device=dev)
        fi_plan(w, qo_t, qo_t, idx, torch.ones(R, dtype=torch.int32, device=dev), self.heads, self.heads, self.hd, 1, causal=False,
               q_data_type=self.dtype, kv_data_type=self.dtype)
        buf["slots"].copy_(idx.long())
        rows = [r for r in range(R) for _ in range(qo[r + 1] - qo[r])]
        buf["meta"]["tok_row"].copy_(torch.tensor(rows, dtype=torch.int32))
        buf["meta"]["tok_pos"].copy_(torch.cat([torch.arange(half), torch.arange(half)]).int())
        buf["meta"]["row_start"].copy_(qo_t[:-1])
        buf["meta"]["row_len"].copy_(qo_t[1:] - qo_t[:-1])
        sc = torch.tensor(self.cnn_scratch[:R], dtype=torch.int64, device=dev)
        buf["rd"].copy_(sc)
        buf["wr"].copy_(sc)
        self._body(buf, w)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.graph_pool, capture_error_mode="thread_local"):
            self._body(buf, w)
        torch.cuda.synchronize()
        self.graphs[(rb, nb)] = (g, buf, w)

    def capture_all(self, row_buckets=ROW_BUCKETS) -> int:
        for rb in row_buckets:
            for nb in token_buckets(rb, self.setup_tokens):
                self.capture(rb, nb)
        return len(self.graphs)

    # --- solve --------------------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def solve(self, caches: list[FastCache], mu: list[torch.Tensor], spks: torch.Tensor, cond: list[torch.Tensor] | None,
              keep: list[bool]) -> list[torch.Tensor]:
        """The CFG Euler solve of one window per request (``CausalConditionalCFM.forward_chunk``, cosine schedule).

        ``mu[i]`` [mel, T_i] (fp16), ``spks`` [n, mel] (after the affine layer), ``cond[i]`` [mel, T_i] or None (zeros;
        ``cond`` None: zeros for every request).
        Requests with ``keep[i]`` keep the window's keys and conv caches (then trimmed as the reference does); the
        others (final flushes) leave their caches untouched.  Returns each request's mel [mel, T_i] in fp32.
        """
        n = len(caches)
        Ts = [int(m.shape[1]) for m in mu]
        lens = [c.length for c in caches]
        p = self._plan(caches, Ts, keep)
        half, nb = p["half"], p["tokens"]
        pad_t = half - sum(Ts)
        mel = self.mel
        x0 = torch.cat([self.cfm.rand_noise[0, :, o:o + t].t() for o, t in zip(lens, Ts)]
                       + [torch.zeros(pad_t, mel, device=self.device, dtype=self.cfm.rand_noise.dtype)]).to(self.dtype)
        mu_t = torch.cat([m.t() for m in mu]).to(self.dtype)
        spk_t = torch.cat([spks[i:i + 1].expand(t, -1) for i, t in enumerate(Ts)]).to(self.dtype)
        cond_t = torch.cat([c.t() if c is not None else mu_t.new_zeros(t, mel) for c, t in zip(cond, Ts)]).to(self.dtype) \
            if cond is not None else torch.zeros_like(mu_t)
        static = torch.zeros(nb, 3 * mel, dtype=self.dtype, device=self.device)
        static[:sum(Ts)] = torch.cat([mu_t, spk_t, cond_t], dim=1)
        if p["bucket"] is not None and p["kv_idx"].numel() > self._kv_capacity():
            p["bucket"] = None  # long prompts: more cached keys than the graph plan buffer holds
        if p["bucket"] is not None:
            self.capture(*p["bucket"])
            g, buf, w = self.graphs[p["bucket"]]
            fi_plan(w, p["qo"], p["kv"], p["kv_idx"], p["last"], self.heads, self.heads, self.hd, 1, causal=False, q_data_type=self.dtype,
                   kv_data_type=self.dtype)
            buf["x0"].copy_(x0)
            buf["static"].copy_(static)
            buf["slots"].copy_(p["slots"])
            for k, v in p["meta"].items():
                buf["meta"][k].copy_(v)
            buf["rd"].copy_(p["rd"])
            buf["wr"].copy_(p["wr"])
            g.replay()
            x = buf["out"]
            self.counts["graph"] += 1
        else:
            fi_plan(self.attn, p["qo"], p["kv"], p["kv_idx"], p["last"], self.heads, self.heads, self.hd, 1, causal=False, q_data_type=self.dtype,
                           kv_data_type=self.dtype)
            buf = {"x0": x0, "static": static, "out": torch.empty_like(x0), "slots": p["slots"], "meta": p["meta"], "rd": p["rd"], "wr": p["wr"]}
            self._body(buf, self.attn)
            x = buf["out"]
            self.counts["eager"] += 1
        rows, new = p["rows_obj"], p["new"]
        for i, c in enumerate(caches):
            for r, nw in ((rows[i], new[i]), (rows[n + i], new[n + i])):
                if keep[i]:
                    order = nw + r.order  # [new, cache], as the reference concatenates
                    if len(order) > c.keep + TRIM_SUFFIX:
                        self.pool.release(order[c.keep:len(order) - TRIM_SUFFIX])
                        order = order[:c.keep] + order[len(order) - TRIM_SUFFIX:]
                    r.order = order
                else:
                    self.pool.release(nw)
        out, o = [], 0
        for t in Ts:
            out.append(x[o:o + t].t().float())
            o += t
        return out
