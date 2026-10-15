"""CUDA graphs of the Token2Wav DiT estimator call (all blocks) at bucketed shapes.

One graph per ``(rows, T, L)``: ``rows`` = 2 x batch (CFG rows, padded to a bucket), ``T`` = frames of a steady
window, ``L`` = the cache length bucket.  All graphs share one input arena and one output arena (contiguous prefix views)
and one memory pool, so memory is that of the largest shape, not the sum; inputs are copied in before each replay and the
outputs must be consumed (copied) before the next one.  Padding rows attend to their own new frames only, so they never produce NaNs; padded keys are masked.

The eager path (``DiT.forward_chunk`` with a key mask, ``BatchedToken2Wav._estimator``) is the reference; graph replay
runs the same kernels.
"""

from __future__ import annotations

import torch

ROW_BUCKETS = (2, 4, 8, 16, 32, 64)
# Cache-length buckets: graphs captured in sequence do not reuse each other's pool memory (~1.7-4 GB each at 64 rows),
# so few buckets; attention over padded keys costs little next to the block GEMMs.
L_BUCKETS = (512, 1024)
STEADY_T = (50,)  # mel frames of a 25-token window (exact: its caches are kept)
FINAL_T = (56,)  # final windows (at most 27 tokens -> 54 frames), padded: their caches are never read
SETUP_T = (256, 512, 768, 1024)  # prompt setups (no cache), padded: conv caches gathered at the real end
MAX_L = L_BUCKETS[-1]
ROW_LIMIT = {**{t: ROW_BUCKETS[-1] for t in STEADY_T}, **{t: 16 for t in FINAL_T}, **{t: 4 for t in SETUP_T}}


def bucket(n: int, buckets) -> int:
    for b in buckets:
        if b >= n:
            return b
    raise ValueError(f"{n} exceeds the largest bucket {buckets[-1]}")


def l_bucket(n: int) -> int:
    return 0 if n == 0 else bucket(n, L_BUCKETS)


class EstimatorGraphs:
    """``eager_fn(est, x, mu, temb, spks, cond, cnn, att, mask)`` takes the timestep embedding (computed outside the graph:
    ``TimestepEmbedder`` builds its frequency table on the host)."""

    def __init__(self, est, eager_fn, dtype, autocast_dtype, graph_t=STEADY_T + FINAL_T + SETUP_T, max_rows: int = ROW_BUCKETS[-1],
                 max_l: int = MAX_L):
        self.est, self.eager_fn, self.dtype, self.autocast_dtype = est, eager_fn, dtype, autocast_dtype
        self.graphs: dict[tuple[int, int, int], tuple] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.graph_t, self.max_rows, self.max_l = tuple(graph_t), max_rows, max_l
        self._arena: dict[str, torch.Tensor] = {}
        self.counts = {"graph": 0, "eager": 0, "eager_reasons": {}}
        p = next(est.parameters())
        self.depth, self.nh = len(est.blocks), est.blocks[0].attn.num_heads
        self.dk2 = 2 * est.blocks[0].attn.head_dim
        self.cnn_c = est.blocks[0].conv.in_channels + est.blocks[0].conv.out_channels
        self.mel = est.in_proj.in_features // 4
        self.hidden = est.in_proj.out_features
        self.device = p.device

    def _view(self, name: str, shape: tuple, dtype) -> torch.Tensor:
        """A contiguous view of the shared input arena (every graph's static inputs alias it; replays are sequential and
        inputs are written right before each replay)."""
        n = 1
        for d in shape:
            n *= d
        buf = self._arena.get(name)
        if buf is None or buf.numel() < n:
            raise RuntimeError(f"arena {name} too small for {shape}")
        return buf[:n].view(shape)

    def _shapes(self):
        for T in self.graph_t:
            for rows in ROW_BUCKETS:
                if rows > min(self.max_rows, ROW_LIMIT.get(T, self.max_rows)):
                    break
                for L in ((0,) if T in SETUP_T else L_BUCKETS):
                    if L <= self.max_l:
                        yield rows, T, L

    def _alloc_arena(self) -> None:
        sizes = {}
        for rows, T, L in self._shapes():
            need = {"x": rows * self.mel * T, "mu": rows * self.mel * T, "t": rows * self.hidden, "spks": rows * self.mel,
                    "cond": rows * self.mel * T, "cnn": self.depth * rows * self.cnn_c * 2, "att": self.depth * rows * self.nh * max(L, 1) * self.dk2,
                    "o_dphi": rows * self.mel * T, "o_cnn": self.depth * rows * self.cnn_c * 2, "o_att": self.depth * rows * self.nh * (T + L) * self.dk2}
            for k, n in need.items():
                sizes[k] = max(sizes.get(k, 0), n)
            sizes["mask"] = max(sizes.get("mask", 0), rows * T * (T + L))
            sizes["tr"] = max(sizes.get("tr", 0), rows)
        for k, n in sizes.items():
            dt = torch.bool if k == "mask" else torch.int64 if k == "tr" else self.dtype
            self._arena[k] = torch.zeros(n, dtype=dt, device=self.device)

    def _static(self, rows: int, T: int, L: int) -> dict:
        if not self._arena:
            self._alloc_arena()
        v = lambda name, *shape, dt=None: self._view(name, shape, dt or self.dtype)  # noqa: E731
        mask = v("mask", rows, T, T + L, dt=torch.bool)
        mask.zero_()
        mask[:, :, :T] = True
        tr = v("tr", rows, dt=torch.int64)
        tr.fill_(T)
        return {"x": v("x", rows, self.mel, T), "mu": v("mu", rows, self.mel, T), "t": v("t", rows, 1, self.hidden), "spks": v("spks", rows, self.mel),
                "cond": v("cond", rows, self.mel, T), "cnn": v("cnn", self.depth, rows, self.cnn_c, 2),
                "att": v("att", self.depth, rows, self.nh, L, self.dk2), "mask": mask, "tr": tr}

    def supports(self, n_rows: int, T: int, L: int) -> bool:
        if T not in self.graph_t or n_rows > min(self.max_rows, ROW_LIMIT.get(T, self.max_rows)):
            return False
        return L == 0 if T in SETUP_T else L <= self.max_l

    @torch.inference_mode()
    def capture_all(self) -> int:
        """Capture every (rows bucket, steady T, L bucket) graph now; returns the count."""
        for rows, T, L in self._shapes():
            self.capture(rows, T, L)
        return len(self.graphs)

    @torch.inference_mode()
    def _capture(self, key):
        rows, T, L = key
        s = self._static(rows, T, L)
        args = (s["x"], s["mu"], s["t"], s["spks"], s["cond"], s["cnn"], s["att"], s["mask"], s["tr"])
        v = lambda name, *shape: self._view(name, shape, self.dtype)  # noqa: E731
        out = (v("o_dphi", rows, self.mel, T), v("o_cnn", self.depth, rows, self.cnn_c, 2), v("o_att", self.depth, rows, self.nh, T + L, self.dk2))
        with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
            self.eager_fn(self.est, *args, out=out)  # warm-up (lazy init, cuBLAS handles, compilation) outside the capture
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                self.eager_fn(self.est, *args, out=out)  # results land in the shared output arena
        torch.cuda.synchronize()
        self.graphs[key] = (g, s, out)
        return self.graphs[key]

    def capture(self, rows: int, T: int, L: int) -> None:
        key = (rows, T, L)
        if key not in self.graphs:
            self._capture(key)

    @torch.inference_mode()
    def __call__(self, x, mu, t, spks, cond, cnn, att, mask, tr):
        """Same contract as the eager estimator call; ``x`` etc. have ``n_rows <= rows bucket`` rows and cache length L."""
        n, T = x.shape[0], x.shape[2]
        L = att.shape[3]
        with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
            t = self.est.t_embedder(t).unsqueeze(1)
        if not self.supports(n, T, L):
            self.counts["eager"] += 1
            r = f"rows={n} T={T} L={L}"
            self.counts["eager_reasons"][r] = self.counts["eager_reasons"].get(r, 0) + 1
            with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                return self.eager_fn(self.est, x, mu, t, spks, cond, cnn, att, mask, tr)
        key = (bucket(n, ROW_BUCKETS), T, l_bucket(L))
        if key not in self.graphs:
            self._capture(key)
        self.counts["graph"] += 1
        g, s, out = self.graphs[key]
        rows, _, Lb = key
        s["x"][:n].copy_(x)
        s["mu"][:n].copy_(mu)
        s["t"][:n].copy_(t)
        s["spks"][:n].copy_(spks)
        s["cond"][:n].copy_(cond)
        s["cnn"][:, :n].copy_(cnn)
        s["att"][:, :n, :, :L].copy_(att)
        if Lb > L:
            s["att"][:, :n, :, L:].zero_()
        s["tr"][:n].copy_(tr)
        s["tr"][n:] = T
        s["mask"][:n, :, :T].copy_(mask[:, :, :T])  # new frames, then the cache at offset T (padded to the bucket)
        s["mask"][:n, :, T:T + L].copy_(mask[:, :, T:])
        s["mask"][:n, :, T + L:] = False
        if rows > n:  # padding rows: their own new frames only
            s["mask"][n:] = False
            s["mask"][n:, :, :T] = True
        g.replay()
        dphi, new_cnn, new_att = out
        return dphi[:n], new_cnn[:, :n], new_att[:, :n, :, :T + L]  # new frames then the real cache (padding after)


class EncoderGraphs:
    """CUDA graphs of the flow encoder's chunk forward (``token2wav_batch._encoder_core``) for steady windows.

    Keys ``(rows, L1)``: rows padded to a bucket, the first-stack cache length padded (on the left, masked) to a bucket;
    the window is 28 tokens (25 plus 3 lookahead), not final.  Inputs and outputs share arenas across graphs.
    """

    ROWS = (1, 2, 4, 8, 16, 32)
    L1 = (128, 256, 384, 512)
    TOKENS = 28

    def __init__(self, enc, core_fn, autocast_dtype):
        self.enc, self.core_fn, self.autocast_dtype = enc, core_fn, autocast_dtype
        self.graphs: dict[tuple[int, int], tuple] = {}
        self.pool = torch.cuda.graph_pool_handle()
        p = next(enc.parameters())
        self.device = p.device
        self.dtype = p.dtype  # caches are stored in the module dtype
        self.d1, self.d2 = len(enc.encoders), len(enc.up_encoders)
        a = enc.encoders[0].self_attn
        self.nh, self.dk2, self.c = a.h, 2 * a.d_k, enc._output_size
        self.s, self.look = enc.up_layer.stride, enc.pre_lookahead_layer.pre_lookahead_len
        self.cnn_t = 2 + 2 * self.s
        self._arena: dict[str, torch.Tensor] = {}

    def l_bucket(self, n: int) -> int:
        return bucket(n, self.L1)

    def supports(self, rows: int, tokens: int, off1: int, last: bool) -> bool:
        return not last and tokens == self.TOKENS and rows <= self.ROWS[-1] and off1 <= self.L1[-1]

    def _dims(self, rows, L1):
        T1 = self.TOKENS - self.look
        T2, L2 = T1 * self.s, L1 * self.s
        return {"xs": ((rows, self.TOKENS, self.c), None), "cnn": ((rows, self.c, self.cnn_t), torch.float32),
                "A1": ((self.d1, rows, self.nh, L1, self.dk2), self.dtype), "m1": ((rows, T1, L1 + T1), torch.bool),
                "A2": ((self.d2, rows, self.nh, L2, self.dk2), self.dtype), "m2": ((rows, T2, L2 + T2), torch.bool),
                "o_xs": ((rows, T2, self.c), None), "o_new1": ((self.d1, rows, self.nh, L1 + T1, self.dk2), None),
                "o_new2": ((self.d2, rows, self.nh, L2 + T2, self.dk2), None), "o_cnn": ((rows, self.c, self.cnn_t), torch.float32)}

    def _view(self, name, shape, dtype):
        n = 1
        for d in shape:
            n *= d
        buf = self._arena.get((name, dtype))
        if buf is None or buf.numel() < n:
            raise RuntimeError(f"encoder arena {name} too small")
        return buf[:n].view(shape)

    def _alloc(self, dtypes: dict) -> None:
        sizes = {}
        for rows in self.ROWS:
            for L1 in self.L1:
                for name, (shape, _) in self._dims(rows, L1).items():
                    n = 1
                    for d in shape:
                        n *= d
                    key = (name, dtypes[name])
                    sizes[key] = max(sizes.get(key, 0), n)
        for (name, dt), n in sizes.items():
            self._arena[(name, dt)] = torch.zeros(n, dtype=dt, device=self.device)

    @torch.inference_mode()
    def _capture(self, rows: int, L1: int, dtypes: dict):
        if not self._arena:
            self._alloc(dtypes)
        st = {name: self._view(name, shape, dtypes[name]) for name, (shape, _) in self._dims(rows, L1).items()}
        st["m1"].fill_(True)
        st["m2"].fill_(True)
        args = (st["xs"], st["cnn"], st["A1"], st["m1"], st["A2"], st["m2"], False)
        out = (st["o_xs"], st["o_new1"], st["o_new2"], st["o_cnn"])
        with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
            self.core_fn(self.enc, *args, out=out)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                self.core_fn(self.enc, *args, out=out)
        torch.cuda.synchronize()
        self.graphs[(rows, L1)] = (g, st, out)

    @torch.inference_mode()
    def capture_all(self, xs_dtype, cnn_dtype, att_dtype) -> int:
        """Capture every key with dummy inputs of the dtypes real calls use."""
        T1 = self.TOKENS - self.look
        for rows in self.ROWS:
            for L1 in self.L1:
                if (rows, L1) in self.graphs:
                    continue
                z = lambda shape, dt: torch.zeros(shape, dtype=dt, device=self.device)  # noqa: E731
                self(z((rows, self.TOKENS, self.c), xs_dtype), z((rows, self.c, self.cnn_t), cnn_dtype),
                     z((self.d1, rows, self.nh, L1, self.dk2), att_dtype), torch.ones(rows, T1, L1 + T1, dtype=torch.bool, device=self.device),
                     z((self.d2, rows, self.nh, L1 * self.s, self.dk2), att_dtype),
                     torch.ones(rows, T1 * self.s, (L1 + T1) * self.s, dtype=torch.bool, device=self.device), False)
        return len(self.graphs)

    @torch.inference_mode()
    def __call__(self, xs, cnns, A1, mask1, A2, mask2, last):
        n, L1 = xs.shape[0], A1.shape[3]
        rows = bucket(n, self.ROWS)
        if (rows, L1) not in self.graphs:
            # output dtypes follow the eager run (autocast): learn them once from an eager call
            with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
                ref = self.core_fn(self.enc, xs, cnns, A1, mask1, A2, mask2, last)
            dtypes = {"xs": xs.dtype, "cnn": cnns.dtype, "A1": A1.dtype, "m1": torch.bool, "A2": A2.dtype, "m2": torch.bool,
                      "o_xs": ref[0].dtype, "o_new1": ref[1].dtype, "o_new2": ref[2].dtype, "o_cnn": ref[3].dtype}
            self._capture(rows, L1, dtypes)
            return ref
        g, st, out = self.graphs[(rows, L1)]
        st["xs"][:n].copy_(xs)
        st["cnn"][:n].copy_(cnns)
        st["A1"][:, :n].copy_(A1)
        st["A2"][:, :n].copy_(A2)
        st["m1"][:n].copy_(mask1)
        st["m2"][:n].copy_(mask2)
        if rows > n:
            st["m1"][n:] = True
            st["m2"][n:] = True
        g.replay()
        return out[0][:n], out[1][:, :n], out[2][:, :n], out[3][:n]
