"""CUDA graphs of the HiFT vocoder (``hift_forward_noise``) for the window shapes of a stream.

Keys ``(rows, mel frames)``: 50 frames for a stream's first window (no mel / source cache) and 58 for every later
25-token window (8 cached mel frames, 3840 cached source samples).  The per-row source noise is drawn outside the graph
from each request's chunk seed (``hift_noise``) and copied in; final windows (other lengths) run eagerly.  Outputs are
copied out before the next replay.  Runs in strict FP32 (AC-3.4).
"""

from __future__ import annotations

import torch

from engine.models.token2wav import SOURCE_CACHE, hift_forward_noise, hift_noise, hift_samples, strict_fp32

ROWS = (1, 2, 4, 8, 16)
FRAMES = {50: 0, 58: SOURCE_CACHE}  # mel frames -> cached source samples


class HiFTGraphs:
    def __init__(self, hift, rows=ROWS, frames=FRAMES):
        self.hift, self.rows, self.frames = hift, tuple(rows), dict(frames)
        self.graphs: dict[tuple[int, int], tuple] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.dim = hift.m_source.l_sin_gen.dim
        self.mels = hift.conv_pre.in_channels
        self.device = next(hift.parameters()).device
        self.counts = {"graph": 0, "eager": 0}

    @torch.inference_mode()
    def _capture(self, rows: int, F: int):
        n = hift_samples(self.hift, F)
        z = lambda *s: torch.zeros(*s, device=self.device)  # noqa: E731
        st = {"mel": z(rows, self.mels, F), "src": z(rows, 1, self.frames[F]), "rand": z(rows, self.dim), "noise": z(rows, n, self.dim)}
        with strict_fp32():
            hift_forward_noise(self.hift, st["mel"], st["src"], st["rand"], st["noise"])  # warm-up
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                out = hift_forward_noise(self.hift, st["mel"], st["src"], st["rand"], st["noise"])
        torch.cuda.synchronize()
        self.graphs[(rows, F)] = (g, st, out)

    def capture_all(self) -> int:
        for F in self.frames:
            for r in self.rows:
                if (r, F) not in self.graphs:
                    self._capture(r, F)
        return len(self.graphs)

    def supports(self, F: int, src_len: int) -> bool:
        return self.frames.get(F, -1) == src_len

    @torch.inference_mode()
    def __call__(self, mel: torch.Tensor, cache_source: torch.Tensor, seeds: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Same contract as ``hift_forward``; rows beyond the largest bucket are run in slices."""
        F, n = mel.shape[2], mel.shape[0]
        rand_ini, noise = hift_noise(self.hift, seeds, hift_samples(self.hift, F), mel.device)
        if not self.supports(F, cache_source.shape[2]):
            self.counts["eager"] += 1
            with strict_fp32():
                return hift_forward_noise(self.hift, mel, cache_source, rand_ini, noise)
        speech, source = [], []
        for a in range(0, n, self.rows[-1]):
            b = min(n, a + self.rows[-1])
            k = next(r for r in self.rows if r >= b - a)
            if (k, F) not in self.graphs:
                self._capture(k, F)
            g, st, out = self.graphs[(k, F)]
            st["mel"][:b - a].copy_(mel[a:b])
            st["src"][:b - a].copy_(cache_source[a:b])
            st["rand"][:b - a].copy_(rand_ini[a:b])
            st["noise"][:b - a].copy_(noise[a:b])
            g.replay()
            self.counts["graph"] += 1
            speech.append(out[0][:b - a].clone())
            source.append(out[1][:b - a].clone())
        return torch.cat(speech), torch.cat(source)
