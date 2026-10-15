"""CUDA graphs of the Token2Wav DiT estimator call (all blocks) at bucketed shapes.

One graph per ``(rows, T, L)``: ``rows`` = 2 x batch (CFG rows, padded to a bucket), ``T`` = frames of the window
(exact for windows whose caches are kept, padded for final windows), ``L`` = the cache length bucket.  Inputs are copied
into the graph's static buffers; the outputs live in the graphs' shared memory pool and must be consumed (copied) before
the next replay.  Padding rows attend to their own new frames only, so they never produce NaNs; padded keys are masked.

The eager path (``DiT.forward_chunk`` with a key mask, ``BatchedToken2Wav._estimator``) is the reference; graph replay
runs the same kernels.
"""

from __future__ import annotations

import torch

ROW_BUCKETS = (2, 4, 8, 16, 32, 64)
L_STEP = 128


def bucket(n: int, buckets) -> int:
    for b in buckets:
        if b >= n:
            return b
    raise ValueError(f"{n} exceeds the largest bucket {buckets[-1]}")


def l_bucket(n: int) -> int:
    return max(L_STEP, -(-n // L_STEP) * L_STEP)


class EstimatorGraphs:
    """``eager_fn(est, x, mu, temb, spks, cond, cnn, att, mask)`` takes the timestep embedding (computed outside the graph:
    ``TimestepEmbedder`` builds its frequency table on the host)."""

    def __init__(self, est, eager_fn, dtype, autocast_dtype, max_graphs: int = 128):
        self.est, self.eager_fn, self.dtype, self.autocast_dtype = est, eager_fn, dtype, autocast_dtype
        self.graphs: dict[tuple[int, int, int], tuple] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.max_graphs = max_graphs
        p = next(est.parameters())
        self.depth, self.nh = len(est.blocks), est.blocks[0].attn.num_heads
        self.dk2 = 2 * est.blocks[0].attn.head_dim
        self.cnn_c = est.blocks[0].conv.in_channels + est.blocks[0].conv.out_channels
        self.mel = est.in_proj.in_features // 4
        self.hidden = est.in_proj.out_features
        self.device = p.device

    def _static(self, rows: int, T: int, L: int) -> dict:
        z = lambda *s, dt=None: torch.zeros(*s, dtype=dt or self.dtype, device=self.device)  # noqa: E731
        mask = torch.zeros(rows, T, T + L, dtype=torch.bool, device=self.device)
        mask[:, :, :T] = True
        return {"x": z(rows, self.mel, T), "mu": z(rows, self.mel, T), "t": z(rows, 1, self.hidden), "spks": z(rows, self.mel), "cond": z(rows, self.mel, T),
                "cnn": z(self.depth, rows, self.cnn_c, 2), "att": z(self.depth, rows, self.nh, L, self.dk2), "mask": mask}

    def _capture(self, key):
        rows, T, L = key
        s = self._static(rows, T, L)
        args = (s["x"], s["mu"], s["t"], s["spks"], s["cond"], s["cnn"], s["att"], s["mask"])
        with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
            self.eager_fn(self.est, *args)  # warm-up (lazy init, cuBLAS handles) outside the capture
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                out = self.eager_fn(self.est, *args)
        torch.cuda.synchronize()
        self.graphs[key] = (g, s, out)
        return self.graphs[key]

    def capture(self, rows: int, T: int, L: int) -> None:
        key = (rows, T, L)
        if key not in self.graphs:
            self._capture(key)

    def __call__(self, x, mu, t, spks, cond, cnn, att, mask):
        """Same contract as the eager estimator call; ``x`` etc. have ``n_rows <= rows bucket`` rows and cache length L."""
        n, T = x.shape[0], x.shape[2]
        L = att.shape[3]
        with torch.amp.autocast("cuda", dtype=self.autocast_dtype):
            t = self.est.t_embedder(t).unsqueeze(1)
        key = (bucket(n, ROW_BUCKETS), T, l_bucket(L))
        if key not in self.graphs:
            if len(self.graphs) >= self.max_graphs:
                return self.eager_fn(self.est, x, mu, t, spks, cond, cnn, att, mask)
            self._capture(key)
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
        s["mask"][:n, :, :T + L].copy_(mask)
        s["mask"][:n, :, T + L:] = False
        if rows > n:  # padding rows: their own new frames only
            s["mask"][n:] = False
            s["mask"][n:, :, :T] = True
        g.replay()
        dphi, new_cnn, new_att = out
        return dphi[:n], new_cnn[:, :n], new_att[:, :n, :, :T + L]
