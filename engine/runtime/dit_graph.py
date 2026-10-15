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
