"""Batched Token2Wav: one flow encoder pass, one Euler solve and one HiFT pass for many streams' windows.

Same per-stream semantics as ``engine.models.token2wav.Token2Wav.decode`` (the HF ``stepaudio2`` stream path), with
every request keeping its own caches.  Requests have different reference prompts, so their cache lengths differ:

* **Flow encoder** (two conformer stacks with ESPnet relative positions).  The relative position of query ``i`` and key
  ``j`` is ``(L - T + i) - j`` for ``L`` keys and ``T`` queries, so a cache *left*-padded to a common length keeps every
  real query/key distance; padded keys are masked.  The two stacks' caches are padded separately (the first stack's
  cache is the first half of the packed cache, as in ``UpsampleConformerEncoderV2.forward_chunk``).
* **DiT estimator** (no positional encoding).  Its blocks concatenate ``[new, cache]``, so caches are *right*-padded
  and masked; each request's new cache is then the contiguous prefix of its rows.  Conditional and unconditional (CFG)
  rows are ``[req_1 .. req_n, req_1 .. req_n]``, as ``x.repeat(2, 1, 1)`` lays them out for one request.
* **HiFT** runs on windows with equal mel length (8 cached frames plus the window, or the first window alone); each
  row's source noise comes from its own per-chunk seed (``hift_forward``).

Windows are grouped by token count and the final-chunk flag, so one call can mix steady windows, first windows and
final flushes.  The flow's fixed noise is read at each request's own offset (its estimator cache length).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from engine.models.token2wav import MEL_CACHE, SOURCE_CACHE, StreamState, Token2Wav, chunk_noise_seed, fade_in_out, hift_forward, strict_fp32
from engine.runtime.hostcopy import to_device
from engine.runtime.stage_trace import span

TRIM_SUFFIX = 100


def _trim(cache: torch.Tensor, keep: int, dim: int) -> torch.Tensor:
    if cache.shape[dim] > keep + TRIM_SUFFIX:
        return torch.cat([cache.narrow(dim, 0, keep), cache.narrow(dim, cache.shape[dim] - TRIM_SUFFIX, TRIM_SUFFIX)], dim=dim)
    return cache


def _encoder_core(enc, xs, cnns, A1, mask1, A2, mask2, last: bool, valid=None, out=None):
    """The flow encoder's chunk forward after the token embedding, on padded caches (graph-capturable): embed, pre-lookahead
    conv, the first conformer stack, the 2x upsampling, the second stack and the final norm.  Returns
    ``(xs, new first-stack att caches, new second-stack att caches, new cnn cache)`` (into ``out`` when given).

    ``valid`` [rows] lets final windows ride in a non-final call: frames from a row's ``valid`` token on are zeroed after
    the embedding, which is exactly the reference's zero lookahead padding for the frames before it (the pre-lookahead
    conv sees zeros either way, later convs are causal, relative positions are unchanged, and the caller masks the
    padded frames as keys); their caches are garbage, and a final window's caches are never read."""
    L1, L2 = A1.shape[3], A2.shape[3]
    xs, _, _ = enc.embed(xs, None)
    if valid is not None:  # rows of final windows: zero frames after their tokens (the reference pads zero lookahead)
        xs = xs * (torch.arange(xs.shape[1], device=xs.device)[None, :] < valid[:, None]).to(xs.dtype)[..., None]
    if last:
        xs = F.pad(xs, (0, 0, 0, enc.pre_lookahead_layer.pre_lookahead_len))
    xs, new_cnn1 = enc.pre_lookahead_layer.forward_chunk(xs, cache=cnns[:, :, :2])
    pos = enc.embed.position_encoding(offset=None, size=L1 + xs.shape[1])
    new1 = []
    for idx, layer in enumerate(enc.encoders):
        xs, _, c, _ = layer(xs, mask1, pos, att_cache=A1[idx])
        new1.append(c)
    xs = xs.transpose(1, 2).contiguous()
    xs, _, new_cnn2 = enc.up_layer.forward_chunk(xs, None, cache=cnns[:, :, 2:])
    xs = xs.transpose(1, 2).contiguous()
    xs, _, _ = enc.up_embed(xs, None)
    pos = enc.embed.position_encoding(offset=None, size=L2 + xs.shape[1])
    new2 = []
    for idx, layer in enumerate(enc.up_encoders):
        xs, _, c, _ = layer(xs, mask2, pos, att_cache=A2[idx])
        new2.append(c)
    if enc.normalize_before:
        xs = enc.after_norm(xs)
    res = (xs, torch.stack(new1), torch.stack(new2), torch.cat([new_cnn1, new_cnn2], dim=2))
    if out is None:
        return res
    for o, r in zip(out, res):
        o.copy_(r)
    return out


def _conv_chunk(conv, x: torch.Tensor, cache: torch.Tensor, tr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``CausalConv1d.forward_chunk`` whose new cache is the 2 frames ending at each row's real length ``tr``."""
    full = torch.cat([cache, x], dim=2)  # [b, c, 2 + T]
    k = conv.causal_padding[0]
    idx = (tr[:, None] + torch.arange(k, device=x.device)[None, :])[:, None, :].expand(-1, full.shape[1], -1)
    return torch.nn.Conv1d.forward(conv, full), torch.gather(full, 2, idx)


def _conv_block_chunk(cb, x: torch.Tensor, cnn_cache: torch.Tensor, tr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``CausalConvBlock.forward_chunk`` with real-end caches (``cb.block``: T, conv, T, LN, Mish, T, conv, T)."""
    c1, c2 = cnn_cache.split((cb.in_channels, cb.out_channels), dim=1)
    h = cb.block[0](x)
    h, n1 = _conv_chunk(cb.block[1], h, c1, tr)
    h = cb.block[2:6](h)
    h, n2 = _conv_chunk(cb.block[6], h, c2, tr)
    return cb.block[7](h), torch.cat((n1, n2), dim=1)


def _block_chunk(block, x, c, cnn_cache, att_cache, mask, tr):
    """``DiTBlock.forward_chunk`` with real-end conv caches (identical when ``tr`` is the frame count)."""
    from stepaudio2.cosyvoice2.flow.decoder_dit import modulate

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_conv, scale_conv, gate_conv = block.adaLN_modulation(c).chunk(9, dim=-1)
    x_att, new_att = block.attn.forward_chunk(modulate(block.norm1(x), shift_msa, scale_msa), att_cache, mask)
    x = x + gate_msa * x_att
    x_conv, new_cnn = _conv_block_chunk(block.conv, modulate(block.norm3(x), shift_conv, scale_conv), cnn_cache, tr)
    x = x + gate_conv * x_conv
    x = x + gate_mlp * block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))
    return x, new_cnn, new_att


class PendingAudio:
    """One decode call's samples on their way to the host: a non-blocking copy into a pinned host tensor on the compute
    stream, then an event.  The device source is stream-ordered before any later work that could reuse its memory, and
    the caching host allocator reuses the pinned block only after the copy's event, so the call's owner only has to keep
    this object until ``result`` (a request closed meanwhile simply drops its samples)."""

    def __init__(self, flat: torch.Tensor | None, spans: list):
        self.spans = spans
        self.host = None
        if flat is not None:
            self.host = torch.empty(flat.numel(), dtype=flat.dtype, pin_memory=True)
            self.host.copy_(flat, non_blocking=True)
        self.event = torch.cuda.Event()
        self.event.record()

    def ready(self) -> bool:
        return self.event.query()

    def result(self) -> list[np.ndarray | None]:
        self.event.synchronize()
        flat = self.host.numpy() if self.host is not None else np.zeros(0, np.float32)
        out = [flat[a:b].copy() if sp is not None else None for sp in self.spans for a, b in [sp or (0, 0)]]
        self.host = None  # the pinned block goes back to the allocator
        return out


@dataclass
class PreSetup:
    """A prompt setup whose flow encoder already ran (``BatchedToken2Wav.setup_encode``), not solved yet: its features,
    stream state (fast-DiT caches allocated) and solve rows.  ``decode(setups=...)`` and ``setup_many`` take it in place
    of the features; one that is never used goes back with ``release_presetup``."""
    feats: tuple
    st: StreamState
    mu: torch.Tensor
    spk: torch.Tensor
    cond: torch.Tensor


class BatchedToken2Wav:
    def __init__(self, t2w: Token2Wav, use_graphs: bool = False, compile_estimator: bool = False, euler_graphs: bool = False,
                 fast_dit: bool = False, fast_slots: int = 49152, max_streams: int = 96, hift_decode_tf32: bool = False,
                 batch_setup_encoder: bool = False, hift_rows: tuple | None = None, merge_solve: bool = False, fuse_setup_frames: int = 0,
                 hift_lean: bool = False, dit_fused_residual: bool = False, dit_fused_conv: bool = False,
                 hift_fused_bias: bool = False, encoder_batched_io: bool = False):
        """``fast_dit``: the estimator solve of ``engine.runtime.dit_fast`` (slot-pool caches, packed ragged rows, CUDA
        graphs with ``use_graphs``) in place of the reference-module solve; FP16 flow only.  Streams must then be made with
        ``new_stream`` and released with ``release_stream``.  ``hift_decode_tf32``: HiFT's upsampling stages
        in TF32 (``token2wav.decoder_tf32``; the f0 predictor, the source and the input/output convolutions stay strict FP32).
        ``batch_setup_encoder``: the prompt setups of one ``setup_many`` call share one padded flow-encoder batch.
        ``hift_rows``: the HiFT graph row buckets (default ``hift_graph.ROWS``; larger calls run in slices of the largest).
        ``merge_solve`` (fast DiT): a decode with several window groups runs each group's flow encoder and then one packed
        solve over all of them (``_flow_merged``) instead of one solve per group.
        ``fuse_setup_frames`` (fast DiT): ``decode(setups=...)`` may carry prompt setups of up to this many mel frames in
        all into the decode's solve (the DiT graphs take ``2 * fuse_setup_frames`` more tokens per row bucket).
        ``hift_lean``: HiFT's Snake activations fused and its weight norms folded (``engine.runtime.hift_lean``, the same
        vocoder within cuDNN's run-to-run spread), before its graphs are made; ``hift_fused_bias``: and its ResBlocks'
        biases, residuals and activations fused (``make_lean(fused_bias=True)``, the same ResBlock outputs bit for bit).
        ``dit_fused_residual`` (fast DiT): its gated residual adds fused with the norms after them
        (``FastDiT(fused_residual=True)``); ``dit_fused_conv``: and the convolutions' im2col written by the norms before
        them (``FastDiT(fused_conv=True)``).  ``encoder_batched_io``: the flow encoder's caches gathered and written back for
        all of a call's streams in one kernel each (``engine.runtime.encoder_io``, the same caches bit for bit)."""
        self.t2w, self.flow, self.hift = t2w, t2w.flow, t2w.hift
        if hift_lean:
            from engine.runtime.hift_lean import make_lean

            make_lean(self.hift, fused_bias=hift_fused_bias)
        assert hift_lean or not hift_fused_bias, "hift_fused_bias is a hift_lean option"
        self.batch_setup_encoder = batch_setup_encoder
        self.encoder_batched_io = encoder_batched_io
        self.hift_rows = tuple(hift_rows) if hift_rows else None
        self.fast = None
        if fast_dit:
            from engine.runtime.dit_fast import FastDiT

            assert t2w.float16, "the fast DiT runs the FP16 flow"
            self.fast = FastDiT(t2w.flow.decoder, t2w.n_timesteps, n_slots=fast_slots, max_streams=max_streams, use_graphs=use_graphs,
                                setup_tokens=2 * fuse_setup_frames, fused_residual=dit_fused_residual,
                                fused_conv=dit_fused_conv)
        assert not fuse_setup_frames or fast_dit, "fused setups are a fast-DiT option"
        self.fuse_setup_frames = int(fuse_setup_frames)
        self.dtype = torch.float16 if t2w.float16 else torch.float32
        self.merge_finals = True
        self.merge_solve = merge_solve
        self.max_flow_streams = 32  # the staged per-step solve keeps a 32-stream call's transient memory small
        self.profile: dict | None = None  # stage -> [calls, seconds, items] when set
        self.trace = None  # diagnostic stage trace (engine.runtime.stage_trace): inner spans, no synchronization
        self.hift_graphs = self.enc_graphs = None
        self.hift_decode_tf32 = hift_decode_tf32
        if use_graphs:
            from engine.runtime.dit_graph import EncoderGraphs
            from engine.runtime.hift_graph import HiFTGraphs

            self.hift_graphs = HiFTGraphs(self.hift, decode_tf32=hift_decode_tf32, **({"rows": self.hift_rows} if self.hift_rows else {}))
            self.enc_graphs = EncoderGraphs(self.flow.encoder, _encoder_core, self.dtype)
        self.dtype = torch.float16 if t2w.float16 else torch.float32
        self.graphs = self.euler = None
        if use_graphs and self.fast is None:
            from engine.runtime.dit_graph import EstimatorGraphs, EulerGraphs

            est = self.flow.decoder.estimator
            fn = BatchedToken2Wav._estimator_temb
            if compile_estimator:  # inductor fuses the adaLN / norm / activation chains (dynamic shapes: one compile)
                fn = torch.compile(fn, dynamic=True)
            self.graphs = EstimatorGraphs(est, fn, next(est.parameters()).dtype, self.dtype, fallback_fn=BatchedToken2Wav._estimator_temb)
            if euler_graphs:
                self.euler = EulerGraphs(self.flow.decoder, fn, next(est.parameters()).dtype, self.dtype, t2w.n_timesteps)

    @torch.inference_mode()
    def decode(self, items: list[tuple[StreamState, list[int], bool]], async_result: bool = False, setups: list[tuple] | None = None):
        """``items[i] = (stream, window tokens, last)`` -> float32 samples per item, exactly as ``Token2Wav.decode`` each.

        ``async_result``: return a ``PendingAudio`` instead, without waiting for the GPU (the samples come through a
        pinned buffer and an event; every stream's caches are already advanced on the host).
        ``setups`` (``fuse_setup_frames``): prompt features to set up in the same call, their rows in the decode's packed
        solve (at most ``max_flow_streams`` requests in all); returns ``(samples, prompts)``, the prompts as
        ``setup_many`` returns them."""
        from engine.token2wav_chunker import CHUNK, PRE_LOOKAHEAD

        if setups:
            assert self.fuse_setup_frames and len(items) + len(setups) <= self.max_flow_streams, (len(items), len(setups))
        groups = defaultdict(list)
        for i, (_, toks, last) in enumerate(items):
            # final windows of at most 25 tokens join the steady 28-token group (zero-padded, see _encoder_core)
            key = (CHUNK + PRE_LOOKAHEAD, False) if last and len(toks) <= CHUNK and self.merge_finals else (len(toks), bool(last))
            groups[key].append(i)
        mels: list[torch.Tensor | None] = [None] * len(items)
        with strict_fp32():
            with torch.amp.autocast("cuda", dtype=self.dtype):
                if setups or (self.merge_solve and self.fast is not None and len(groups) > 1):
                    chunks = [(last, idx_all[a:a + self.max_flow_streams]) for (_, last), idx_all in groups.items()
                              for a in range(0, len(idx_all), self.max_flow_streams)]
                    parts = [([items[i][0] for i in idx], [items[i][1] for i in idx], last, [bool(items[i][2]) for i in idx]) for last, idx in chunks]
                    rows = self._setup_rows(setups) if setups else None
                    for (_, idx), ms in zip(chunks, self._prof("flow", len(items), self._flow_merged, parts, rows)):
                        for i, m in zip(idx, ms):
                            mels[i] = m
                    prompts = [self.t2w.prompt(f, st.flow_cache) for f, st in zip(rows[4], rows[0])] if setups else None
                else:
                    prompts = None
                    self._flow_groups(items, groups, mels)
            # HiFT outside the flow's autocast, as the per-group path runs it: its eager calls stay FP32 / TF32
            out = self._prof("vocode", len(items), self._vocode, items, mels, async_result)
        return (out, prompts) if setups else out

    def _flow_groups(self, items, groups, mels) -> None:
        """One flow call per group (at most ``max_flow_streams`` requests each), mels written into ``mels``."""
        for (_, last), idx_all in groups.items():
            # at most max_flow_streams per flow call: the padded caches of a call are staged for all Euler steps
            for a in range(0, len(idx_all), self.max_flow_streams):
                idx = idx_all[a:a + self.max_flow_streams]
                for i, m in zip(idx, self._prof("flow", len(idx), self._flow, [items[i][0] for i in idx], [items[i][1] for i in idx], last,
                                                [bool(items[i][2]) for i in idx])):
                    mels[i] = m

    def new_stream(self, prompt, noise_seed: int = 0, adopt: bool = False):
        """``Token2Wav.new_stream``; with the fast DiT the stream gets its own copy of the prompt's estimator caches, or
        takes them over (``adopt``: the prompt is used once, as the engine's worker does)."""
        if self.fast is None:
            return self.t2w.new_stream(prompt, noise_seed=noise_seed)
        fc = prompt.flow_cache
        st = self.t2w.new_stream(prompt.__class__(**{**prompt.__dict__, "flow_cache": {k: v for k, v in fc.items() if k != "fast"}}),
                                 noise_seed=noise_seed)
        st.prompt = prompt
        st.flow_cache["fast"] = fc["fast"] if adopt else self.fast.clone(fc["fast"])
        if adopt:
            fc["fast"] = None
        return st

    def release_stream(self, st) -> None:
        """Return a finished or abandoned stream's estimator cache slots (fast DiT; a no-op otherwise)."""
        if self.fast is not None and st is not None and st.flow_cache.get("fast") is not None:
            self.fast.release(st.flow_cache["fast"])
            st.flow_cache["fast"] = None

    def release_prompt(self, prompt) -> None:
        if self.fast is not None and prompt is not None and prompt.flow_cache.get("fast") is not None:
            self.fast.release(prompt.flow_cache["fast"])
            prompt.flow_cache["fast"] = None

    @torch.inference_mode()
    def capture_graphs(self, prompt) -> dict:
        """Capture every Token2Wav graph now (startup); ``prompt`` (any prepared one) supplies the cache dtypes."""
        n = {}
        if self.fast is not None and self.fast.use_graphs:
            n["dit"] = self.fast.capture_all()
        if self.graphs is not None:
            n["estimator"] = self.graphs.capture_all()
        if self.euler is not None:
            n["euler"] = self.euler.capture_all()
        if self.hift_graphs is not None:
            n["hift"] = self.hift_graphs.capture_all()
        if self.enc_graphs is not None:
            emb_dtype = self.flow.input_embedding.weight.dtype
            with strict_fp32():
                n["encoder"] = self.enc_graphs.capture_all(emb_dtype, prompt.flow_cache["conformer_cnn_cache"].dtype,
                                                           prompt.flow_cache["conformer_att_cache"].dtype)
        return n

    def _prof(self, name, n, fn, *a):
        if self.profile is None:
            if self.trace is not None:
                with self.trace.span(name, n):
                    return fn(*a)
            return fn(*a)
        import time

        torch.cuda.current_stream().synchronize()
        t = time.perf_counter()
        r = fn(*a)
        torch.cuda.current_stream().synchronize()
        acc = self.profile.setdefault(name, [0, 0.0, 0])
        acc[0] += 1
        acc[1] += time.perf_counter() - t
        acc[2] += n
        return r

    # --- flow ---------------------------------------------------------------------------------------------------------
    def _flow(self, sts: list[StreamState], windows: list[list[int]], last: bool, lasts: list[bool] | None = None) -> list[torch.Tensor]:
        """One group: windows of equal token count and final flag, or steady windows with merged final windows
        (``lasts[i]`` true: zero-padded to the group's token count; their real frames are 2 x their tokens)."""
        enc = self._flow_encode(sts, windows, last, lasts)
        if self.fast is not None:
            mels = self._prof("solve", len(sts), self.fast.solve, [st.flow_cache["fast"] for st in sts], enc["mu"], enc["spk"], None, enc["keep"])
        else:
            mels = self._prof("solve", len(sts), self._solve, sts, enc["h"].transpose(1, 2).contiguous(), enc["spk"], None, not last, enc["t_real"])
        return self._flow_finish(sts, mels, enc["t_real"])

    def _flow_merged(self, parts: list[tuple[list[StreamState], list[list[int]], bool, list[bool]]], setup_rows: tuple | None = None) -> list[list[torch.Tensor]]:
        """Fast DiT only: every group's flow encoder (one call per group, as ``_flow``), then the groups' windows in one
        packed ragged solve of at most ``max_flow_streams`` requests per call instead of one solve per group.
        ``setup_rows`` (``_setup_rows``): prompt setups whose rows lead the first solve (prompt mel as condition)."""
        encs = [self._flow_encode(*part) for part in parts]
        flat = [(st.flow_cache["fast"], mu, enc["spk"][i:i + 1], k, None) for (sts, *_), enc in zip(parts, encs)
                for i, (st, mu, k) in enumerate(zip(sts, enc["mu"], enc["keep"]))]
        n_set = 0
        if setup_rows is not None:
            sts, mus, spks, conds = setup_rows[:4]
            n_set = len(sts)
            flat = [(st.flow_cache["fast"], mu, spks[i:i + 1], True, c) for i, (st, mu, c) in enumerate(zip(sts, mus, conds))] + flat
        mels = []
        for a in range(0, len(flat), self.max_flow_streams):
            sl = flat[a:a + self.max_flow_streams]
            fused = any(x[4] is not None for x in sl)
            mels += self._prof("solve", len(sl), self.fast.solve, [x[0] for x in sl], [x[1] for x in sl],
                               torch.cat([x[2] for x in sl]), [x[4] for x in sl] if fused else None, [x[3] for x in sl])
        mels = mels[n_set:]
        out, o = [], 0
        for (sts, *_), enc in zip(parts, encs):
            out.append(self._flow_finish(sts, mels[o:o + len(sts)], enc["t_real"]))
            o += len(sts)
        return out

    def _flow_encode(self, sts: list[StreamState], windows: list[list[int]], last: bool, lasts: list[bool] | None = None) -> dict:
        """A group's flow encoder: the solve inputs (``mu`` per request for the fast DiT, ``h`` for the reference solve)."""
        from engine.token2wav_chunker import CHUNK, PRE_LOOKAHEAD

        flow = self.flow
        n = len(sts)
        lasts = lasts or [last] * n
        merged = any(lasts) and not last
        L = CHUNK + PRE_LOOKAHEAD if merged else max(len(w) for w in windows)  # merged finals ride in a 28-token call
        tok = to_device([w + [0] * (L - len(w)) for w in windows], torch.int32)  # queued: no wait for the GPU (engine.runtime.hostcopy)
        vl = [len(w) for w in windows] if merged else None
        valid = to_device(vl, torch.int64) if merged else None
        spk = flow.spk_embed_affine_layer(F.normalize(torch.cat([st.prompt.spk for st in sts]), dim=1))
        h = flow.encoder_proj(self._prof("encoder", n, self._encoder, sts, flow.input_embedding(tok), last, valid, vl))
        t_real = [2 * len(w) if fin else h.shape[1] for w, fin in zip(windows, lasts)] if merged else None
        assert all(len(w) - (0 if fin else PRE_LOOKAHEAD) <= h.shape[1] // 2 for w, fin in zip(windows, lasts))
        enc = {"h": h, "spk": spk, "t_real": t_real, "keep": [not fin for fin in lasts]}
        if self.fast is not None:
            mu = h.transpose(1, 2)
            Ts = t_real if t_real is not None else [h.shape[1]] * n
            enc["mu"] = [mu[i, :, :Ts[i]] for i in range(n)]
        return enc

    def _flow_finish(self, sts: list[StreamState], mels, t_real: list[int] | None) -> list[torch.Tensor]:
        out = []
        for i, st in enumerate(sts):
            keep = st.prompt.mels.shape[1]
            c = st.flow_cache
            if self.fast is None:
                c["estimator_att_cache"] = _trim(c["estimator_att_cache"], keep, 4)
            c["conformer_att_cache"] = _trim(c["conformer_att_cache"], keep, 3)
            if self.fast is not None:
                out.append(mels[i][None])
            else:
                out.append(mels[i:i + 1, :, :t_real[i]] if t_real is not None else mels[i:i + 1])
        return out

    def _encoder(self, sts: list[StreamState], xs: torch.Tensor, last: bool, valid: torch.Tensor | None = None, vl: list[int] | None = None) -> torch.Tensor:
        """``UpsampleConformerEncoderV2.forward_chunk`` over a batch with left-padded caches; updates each stream's caches.

        The cache length is padded to a graph bucket for steady windows (``EncoderGraphs``), else to the batch maximum.
        """
        enc = self.flow.encoder
        n, dev = len(sts), xs.device
        atts = [st.flow_cache["conformer_att_cache"] for st in sts]  # [depth1 + depth2, 1, nh, 2 * t1, 2 dk]
        cnns = torch.cat([st.flow_cache["conformer_cnn_cache"] for st in sts])  # [n, c, 2 + 2 * stride]
        d1, d2, s = len(enc.encoders), len(enc.up_encoders), enc.up_layer.stride
        off1 = [a.shape[3] // 2 for a in atts]
        graphs = self.enc_graphs if self.enc_graphs is not None and self.enc_graphs.supports(n, xs.shape[1], max(off1), last) else None
        L1 = graphs.l_bucket(max(off1)) if graphs is not None else max(off1)
        T1 = xs.shape[1] if last else xs.shape[1] - enc.pre_lookahead_layer.pre_lookahead_len
        T2, L2 = T1 * s, L1 * s
        nh, dk2 = atts[0].shape[2], atts[0].shape[4]
        if valid is not None and vl is None:
            vl = valid.tolist()
        if self.encoder_batched_io:
            A1, A2, mask1, mask2 = self._encoder_in(atts, off1, n, L1, T1, s, vl, xs.shape[1], graphs)
            core = graphs if graphs is not None else (lambda *a: _encoder_core(enc, *a))
            xs, new1, new2, new_cnn = core(xs, cnns, A1, mask1, A2, mask2, last, valid)
            return self._encoder_out(sts, xs, new1, new2, new_cnn, off1, T1, L1, graphs is not None)
        A1 = atts[0].new_zeros((d1, n, nh, L1, dk2))
        A2 = atts[0].new_zeros((d2, n, nh, L2, dk2))
        mask1 = torch.ones(n, T1, L1 + T1, dtype=torch.bool, device=dev)
        mask2 = torch.ones(n, T2, L2 + T2, dtype=torch.bool, device=dev)
        for i, a in enumerate(atts):
            A1[:, i, :, L1 - off1[i]:] = a[:d1, 0, :, :off1[i]]
            mask1[i, :, :L1 - off1[i]] = False
            A2[:, i, :, L2 - off1[i] * s:] = a[d1:, 0]
            mask2[i, :, :L2 - off1[i] * s] = False
            if vl is not None and vl[i] < xs.shape[1]:  # a merged final window: its frames after its tokens are padding
                mask1[i, :, L1 + vl[i]:] = False
                mask2[i, :, L2 + vl[i] * s:] = False
        core = graphs if graphs is not None else (lambda *a: _encoder_core(enc, *a))
        xs, new1, new2, new_cnn = core(xs, cnns, A1, mask1, A2, mask2, last, valid)
        for i, st in enumerate(sts):
            c1 = new1[:, i:i + 1, :, L1 - off1[i]:]
            c2 = new2[:, i:i + 1, :, L2 - off1[i] * s:]
            st.flow_cache["conformer_att_cache"] = torch.cat([c1.repeat(1, 1, 1, 2, 1), c2], dim=0)
            st.flow_cache["conformer_cnn_cache"] = new_cnn[i:i + 1].clone()
        return xs.clone() if graphs is not None else xs

    def _encoder_in(self, atts, off1, n, L1, T1, s, vl, tok, graphs):
        """``encoder_batched_io``: the padded caches in one gather (into the graph's input arena when it is captured) and
        the key masks from the rows' cache lengths (and merged finals' token counts) in a few vector ops."""
        from engine.runtime.encoder_io import gather_caches

        d1, d2 = len(self.flow.encoder.encoders), len(self.flow.encoder.up_encoders)
        nh, dk2, dev = atts[0].shape[2], atts[0].shape[4], atts[0].device
        arena = graphs.arena_inputs(n, L1, tok) if graphs is not None else None
        if arena is None:
            A1f, A2f = (torch.empty((d, n, nh, L, dk2), dtype=atts[0].dtype, device=dev) for d, L in ((d1, L1), (d2, L1 * s)))
        else:
            A1f, A2f = arena
        gather_caches(atts, off1, A1f, A2f)
        lo = L1 - to_device(off1, torch.int64, dev)
        k1, k2 = torch.arange(L1 + T1, device=dev)[None, None, :], torch.arange((L1 + T1) * s, device=dev)[None, None, :]
        mask1, mask2 = k1 >= lo[:, None, None], k2 >= s * lo[:, None, None]
        if vl is not None:  # merged final windows: frames after their tokens are padding (rows with all tokens are unaffected)
            hi = L1 + to_device(vl, torch.int64, dev)
            mask1, mask2 = mask1 & (k1 < hi[:, None, None]), mask2 & (k2 < s * hi[:, None, None])
        mask1, mask2 = mask1.expand(n, T1, -1).contiguous(), mask2.expand(n, T1 * s, -1).contiguous()
        return A1f[:, :n], A2f[:, :n], mask1, mask2

    def _encoder_out(self, sts, xs, new1, new2, new_cnn, off1, T1, L1, graphed: bool) -> torch.Tensor:
        """``encoder_batched_io``: every stream's new attention cache in one kernel, already trimmed as ``_flow_finish``
        trims it, and the conv caches as views of one copy."""
        from engine.runtime.encoder_io import scatter_caches

        atts = scatter_caches(new1, new2, off1, [st.prompt.mels.shape[1] for st in sts], T1, L1, TRIM_SUFFIX)
        cnn = new_cnn.clone() if graphed else new_cnn
        for i, st in enumerate(sts):
            st.flow_cache["conformer_att_cache"] = atts[i]
            st.flow_cache["conformer_cnn_cache"] = cnn[i:i + 1]
        return xs.clone() if graphed else xs

    @torch.inference_mode()
    def setup(self, feats: tuple):
        """``Token2Wav.setup`` (reference ``Flow.setup_cache``) on this module's solver (see ``setup_many``)."""
        return self.setup_many([feats])[0]

    def _setup_rows(self, setups: list) -> tuple:
        """Fast DiT: the setups' flow encoder and empty caches, not solved: (streams, mu per prompt, spks, cond per prompt,
        features per prompt).  ``setups``: features, or ``PreSetup`` whose encoder already ran (the others' run now)."""
        pre = self._presetups(setups)
        return [p.st for p in pre], [p.mu for p in pre], torch.cat([p.spk for p in pre]), [p.cond for p in pre], [p.feats for p in pre]

    def _presetups(self, setups: list) -> list[PreSetup]:
        raw = [f for f in setups if not isinstance(f, PreSetup)]
        enc = iter(self.setup_encode(raw) if raw else ())
        return [f if isinstance(f, PreSetup) else next(enc) for f in setups]

    @torch.inference_mode()
    def setup_encode(self, feats_list: list[tuple]) -> list[PreSetup]:
        """Fast DiT: the setups' flow encoder (one batch, as ``setup_many``) and empty caches, ahead of their solve."""
        sts, mus, spks, conds = self.setup_many(feats_list, rows_only=True)
        return [PreSetup(f, st, mu, spks[i:i + 1], c) for i, (f, st, mu, c) in enumerate(zip(feats_list, sts, mus, conds))]

    def release_presetup(self, p: PreSetup) -> None:
        if self.fast is not None and p.st.flow_cache.get("fast") is not None:
            self.fast.release(p.st.flow_cache["fast"])
            p.st.flow_cache["fast"] = None

    @torch.inference_mode()
    def setup_many(self, feats_list: list[tuple], rows_only: bool = False) -> list:
        """Prompt setups of several requests: each prompt (plus 3 silence tokens) through the flow encoder (one padded
        batch with ``batch_setup_encoder``, else one prompt at a time), then one Euler solve over all prompts' frames
        (each with its prompt mel as condition and empty caches; frames right-padded to the longest / a graph bucket,
        padding masked, conv caches gathered at each prompt's real end).  Entries may be ``PreSetup`` (fast DiT: encoder
        already run); then every prompt's solve runs here, the others' encoder first."""
        if any(isinstance(f, PreSetup) for f in feats_list):
            assert self.fast is not None and not rows_only
            pre = self._presetups(feats_list)
            with strict_fp32(), torch.amp.autocast("cuda", dtype=self.dtype), span(self.trace, "setup.solve", len(pre)):
                self.fast.solve([p.st.flow_cache["fast"] for p in pre], [p.mu for p in pre], torch.cat([p.spk for p in pre]), [p.cond for p in pre],
                                [True] * len(pre))
            return [self.t2w.prompt(p.feats, p.st.flow_cache) for p in pre]
        flow = self.flow
        est = flow.decoder.estimator
        steps, depth = self.t2w.n_timesteps, len(est.blocks)
        blk = est.blocks[0]
        dt = next(est.parameters()).dtype
        sts, hs, conds, spks, lens = [], [], [], [], []
        with strict_fp32(), torch.amp.autocast("cuda", dtype=self.dtype):
            with span(self.trace, "setup.encoder", len(feats_list)):
                encs = self._setup_encoder_batched(feats_list) if self.batch_setup_encoder else [self._setup_encoder_one(f) for f in feats_list]
            for (_, _, spk, mels, _), (h, cnn, att) in zip(feats_list, encs):
                assert h.shape[1] == mels.shape[1], (h.shape, mels.shape)
                cache = {"conformer_cnn_cache": cnn, "conformer_att_cache": att}
                if self.fast is None:
                    cache["estimator_cnn_cache"] = torch.zeros(steps, depth, 2, blk.conv.in_channels + blk.conv.out_channels, 2, device="cuda", dtype=dt)
                    cache["estimator_att_cache"] = torch.zeros(steps, depth, 2, blk.attn.num_heads, 0, 2 * blk.attn.head_dim, device="cuda", dtype=dt)
                sts.append(StreamState(prompt=None, flow_cache=cache, hift_cache={}))
                hs.append(h.transpose(1, 2))
                conds.append(mels.transpose(1, 2))
                spks.append(flow.spk_embed_affine_layer(F.normalize(spk, dim=1)))
                lens.append(h.shape[1])
            if self.fast is not None:
                caches = [self.fast.new_cache() for _ in sts]
                for c, st, L in zip(caches, sts, lens):
                    c.keep = L  # the trim length: the prompt's mel frames
                    st.flow_cache["fast"] = c
                if rows_only:
                    return sts, [x[0] for x in hs], torch.cat(spks), [x[0] for x in conds]
                with span(self.trace, "setup.solve", len(sts)):
                    self.fast.solve(caches, [x[0] for x in hs], torch.cat(spks), [x[0] for x in conds], [True] * len(sts))
                return [self.t2w.prompt(f, st.flow_cache) for f, st in zip(feats_list, sts)]
            T = max(lens)
            padded = lambda xs: torch.cat([F.pad(x, (0, T - x.shape[2])) for x in xs])  # noqa: E731
            self._solve(sts, padded(hs).contiguous(), torch.cat(spks), cond=padded(conds).contiguous(), keep_cache=True, t_real=lens)
        return [self.t2w.prompt(f, st.flow_cache) for f, st in zip(feats_list, sts)]

    def _setup_encoder_one(self, feats: tuple) -> tuple:
        """One prompt (plus 3 silence tokens) through the reference ``forward_chunk`` with empty caches: (encoder_proj
        output [1, 2n, C], conv caches [1, c, 2 + 2 * stride], attention caches [d1 + d2, 1, nh, 2n, 2 dk])."""
        from engine.models.token2wav import SILENCE_TOKEN

        flow = self.flow
        tokens = feats[0]
        pad = torch.full((1, 3), SILENCE_TOKEN, device=tokens.device, dtype=tokens.dtype)
        h, cnn, att = flow.encoder.forward_chunk(xs=flow.input_embedding(torch.cat([tokens, pad], dim=1)), last_chunk=False, cnn_cache=None, att_cache=None)
        return flow.encoder_proj(h), cnn.clone(), att.clone()  # the encoder's caches are scratch it reuses (BL-20261015-flow-shared-caches)

    def _setup_encoder_batched(self, feats_list: list[tuple]) -> list[tuple]:
        """``_setup_encoder_one`` of every prompt in one right-padded batch.  Each prompt's real positions see exactly its
        own inputs: the pre-lookahead conv reads 3 tokens ahead, which for every real frame are its own (the 3 silence
        tokens end each prompt); the later convs are causal; the conformer layers have no conv module, and attention keys
        past each prompt's frames are masked.  Conv caches are gathered at each prompt's real end (the reference takes
        the last frames of its unpadded sequence) and attention caches cut to its real length."""
        from engine.models.token2wav import SILENCE_TOKEN

        flow, enc = self.flow, self.flow.encoder
        pl, up = enc.pre_lookahead_layer, enc.up_layer
        look, s = pl.pre_lookahead_len, up.stride
        dev = feats_list[0][0].device
        toks = [torch.cat([f[0], torch.full((1, look), SILENCE_TOKEN, device=dev, dtype=f[0].dtype)], dim=1) for f in feats_list]
        T = max(t.shape[1] for t in toks)
        ns = [t.shape[1] - look for t in toks]  # each prompt's frames after the pre-lookahead conv
        n = to_device(ns, torch.int64, dev)
        B, T1 = len(toks), T - look
        xs, _, _ = enc.embed(flow.input_embedding(torch.cat([F.pad(t, (0, T - t.shape[1])) for t in toks])), None)
        # pre-lookahead layer (PreLookaheadLayer.forward_chunk with an empty cache)
        o = F.leaky_relu(pl.conv1(xs.transpose(1, 2).contiguous()))  # [B, c, T1]
        cnn1 = torch.gather(o, 2, (n[:, None] - 2 + torch.arange(2, device=dev)[None, :])[:, None, :].expand(-1, o.shape[1], -1))
        o = pl.conv2(torch.cat([o.new_zeros(B, o.shape[1], 2), o], dim=2)).transpose(1, 2).contiguous() + xs[:, :-look]
        keys1 = torch.arange(T1, device=dev)[None, :] < n[:, None]
        mask1 = keys1[:, None, :].expand(B, T1, T1)
        pos = enc.embed.position_encoding(offset=None, size=T1)
        new1 = []
        for layer in enc.encoders:
            o, _, c, _ = layer(o, mask1, pos, att_cache=None)
            new1.append(c)
        # upsampling (Upsample1D.forward_chunk with an empty cache: zeros, then the upsampled frames)
        u = torch.cat([o.new_zeros(B, o.shape[2], 2 * s), F.interpolate(o.transpose(1, 2).contiguous(), scale_factor=up.scale_factor, mode="nearest")], dim=2)
        cnn2 = torch.gather(u, 2, (s * n[:, None] + torch.arange(2 * s, device=dev)[None, :])[:, None, :].expand(-1, u.shape[1], -1))
        o, _, _ = enc.up_embed(up.conv(u).transpose(1, 2).contiguous(), None)
        T2 = s * T1
        keys2 = torch.arange(T2, device=dev)[None, :] < s * n[:, None]
        mask2 = keys2[:, None, :].expand(B, T2, T2)
        pos = enc.embed.position_encoding(offset=None, size=T2)
        new2 = []
        for layer in enc.up_encoders:
            o, _, c, _ = layer(o, mask2, pos, att_cache=None)
            new2.append(c)
        if enc.normalize_before:
            o = enc.after_norm(o)
        h = flow.encoder_proj(o)
        new1, new2 = torch.stack(new1), torch.stack(new2)
        out = []
        for i, ni in enumerate(ns):
            att = torch.cat([new1[:, i:i + 1, :, :ni].repeat(1, 1, 1, 2, 1), new2[:, i:i + 1, :, :s * ni]], dim=0)
            out.append((h[i:i + 1, :s * ni].contiguous(), torch.cat([cnn1[i:i + 1], cnn2[i:i + 1]], dim=2).contiguous(), att.contiguous()))
        return out

    def _t_pad(self, T: int, keep_cache: bool, empty_cache: bool) -> int:
        """Frames the solve runs at: the graph bucket for T when padding is exact for what is kept, else T."""
        if self.graphs is None:
            return T
        from engine.runtime.dit_graph import FINAL_T, SETUP_T

        if T in self.graphs.graph_t:
            return T
        if not keep_cache:  # final windows: their caches are never read again
            return next((b for b in FINAL_T if b >= T), T)
        if empty_cache:  # prompt setup: conv caches are gathered at the real end, attention keys masked
            return next((b for b in SETUP_T if b >= T), T)
        return T

    def _solve(self, sts: list[StreamState], mu: torch.Tensor, spks: torch.Tensor, cond: torch.Tensor | None = None,
               keep_cache: bool = True, t_real: list[int] | None = None) -> torch.Tensor:
        """``CausalConditionalCFM.forward_chunk`` (cosine schedule, CFG) over a batch with right-padded estimator caches.

        All rows have ``T = mu.shape[2]`` real frames; the solve may run at a padded frame count (see ``_t_pad``): padded
        frames are masked as keys, never feed real frames (causal convs, per-frame norms), and are dropped.
        """
        cfm = self.flow.decoder
        est = cfm.estimator
        n, T, dev = len(sts), mu.shape[2], mu.device
        tl = list(t_real) if t_real is not None else [T] * n  # real frames per request (<= T; the rest is padding)
        atts = [st.flow_cache["estimator_att_cache"] for st in sts]  # [steps, depth, 2, nh, L, 2 dk]
        cnns = [st.flow_cache["estimator_cnn_cache"] for st in sts]  # [steps, depth, 2, c, 2]
        lens = [a.shape[4] for a in atts]
        L = max(lens)
        Tp = self._t_pad(T, keep_cache, L == 0)
        steps, depth = self.t2w.n_timesteps, atts[0].shape[1]
        assert all(a.shape[0] == steps for a in atts) and all(c.shape[0] == steps for c in cnns)
        if (self.euler is not None and keep_cache and cond is None and t_real is None and L > 0 and Tp == T
                and self.euler.supports(2 * n, T, L)):
            return self._solve_euler_graph(sts, mu, spks, atts, cnns, lens, T, L)
        if (self.graphs is not None and keep_cache and cond is None and t_real is None and L > 0 and Tp == T
                and self.graphs.supports(2 * n, T, L)):
            return self._solve_staged(sts, mu, spks, atts, cnns, lens, T, L)
        A = atts[0].new_zeros((steps, depth, 2 * n) + atts[0].shape[3:4] + (L, atts[0].shape[5]))
        C = cnns[0].new_empty((steps, depth, 2 * n) + cnns[0].shape[3:])
        mask = torch.zeros(2 * n, Tp, Tp + L, dtype=torch.bool, device=dev)
        for i in range(n):
            for half in range(2):
                r = i + half * n
                A[:, :, r, :, :lens[i]] = atts[i][:, :, half]
                C[:, :, r] = cnns[i][:, :, half]
                mask[r, :, :tl[i]] = True
                mask[r, :, Tp:Tp + lens[i]] = True
        tr = torch.tensor(tl + tl, dtype=torch.int64, device=dev)
        padT = (lambda z: F.pad(z, (0, Tp - T))) if Tp > T else (lambda z: z)
        x = padT(torch.stack([F.pad(cfm.rand_noise[0, :, o:o + t], (0, T - t)) for o, t in zip(lens, tl)]).to(mu.dtype))
        t_span = torch.linspace(0, 1, steps + 1, device=dev, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t, dt = t_span[0].unsqueeze(0), t_span[1] - t_span[0]
        mu_in = padT(torch.cat([mu, torch.zeros_like(mu)]))
        spks_in = torch.cat([spks, torch.zeros_like(spks)])
        cond_in = torch.zeros_like(mu_in) if cond is None else padT(torch.cat([cond, torch.zeros_like(cond)]))
        new_att = A.new_empty((steps, depth, 2 * n) + A.shape[3:4] + (Tp + L, A.shape[5])) if keep_cache else None
        new_cnn = torch.empty_like(C) if keep_cache else None
        if self.graphs is not None:
            call = self.graphs
        else:
            call = lambda x, mu, t, *a: self._estimator_temb(est, x, mu, est.t_embedder(t).unsqueeze(1), *a)  # noqa: E731
        for step in range(1, steps + 1):
            if self.graphs is not None:  # later steps reuse the window's mu / spks / cond / mask / tr already in place
                dphi, nc, na = call(x.repeat(2, 1, 1), mu_in, t.repeat(2 * n), spks_in, cond_in, C[step - 1], A[step - 1], mask, tr, same=step > 1)
            else:
                dphi, nc, na = call(x.repeat(2, 1, 1), mu_in, t.repeat(2 * n), spks_in, cond_in, C[step - 1], A[step - 1], mask, tr)
            if keep_cache:
                new_att[step - 1].copy_(na)  # graph outputs are overwritten by the next replay
                new_cnn[step - 1].copy_(nc)
            d, u = dphi.chunk(2, dim=0)
            x = x + dt * ((1.0 + cfm.inference_cfg_rate) * d - cfm.inference_cfg_rate * u)
            t = t + dt
            if step < steps:
                dt = t_span[step + 1] - t
        if keep_cache:
            for i, st in enumerate(sts):
                rows = []
                for r in (i, n + i):  # this request's keys: its real new frames, then its own cache
                    na = new_att[:, :, r]
                    rows.append(torch.cat([na[:, :, :, :tl[i]], na[:, :, :, Tp:Tp + lens[i]]], dim=3) if Tp > tl[i] else na[:, :, :, :T + lens[i]])
                st.flow_cache["estimator_att_cache"] = torch.stack(rows, dim=2)
                st.flow_cache["estimator_cnn_cache"] = torch.stack([new_cnn[:, :, i], new_cnn[:, :, n + i]], dim=2)
        return x[:, :, :T].float()

    def _solve_staged(self, sts, mu, spks, atts, cnns, lens, T: int, L: int) -> torch.Tensor:
        """The per-step graph solve with each step's caches written straight into the graph inputs and its new caches
        straight out into each request's cache (no all-steps staging tensors: at 32 streams those are ~18 GB)."""
        cfm = self.flow.decoder
        n, dev = len(sts), mu.device
        steps = self.t2w.n_timesteps
        g_att, g_cnn = self.graphs.stage(2 * n, T, L)
        mask = torch.zeros(2 * n, T, T + L, dtype=torch.bool, device=dev)
        for i in range(n):
            for half in range(2):
                mask[i + half * n, :, :T + lens[i]] = True
        tr = torch.full((2 * n,), T, dtype=torch.int64, device=dev)
        x = torch.stack([cfm.rand_noise[0, :, o:o + T] for o in lens]).to(mu.dtype)
        t_span = torch.linspace(0, 1, steps + 1, device=dev, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t, dt = t_span[0].unsqueeze(0), t_span[1] - t_span[0]
        mu_in = torch.cat([mu, torch.zeros_like(mu)])
        spks_in = torch.cat([spks, torch.zeros_like(spks)])
        cond_in = torch.zeros_like(mu_in)
        new_att = [atts[i].new_empty(atts[i].shape[:4] + (T + lens[i], atts[i].shape[5])) for i in range(n)]
        new_cnn = [torch.empty_like(c) for c in cnns]
        for step in range(1, steps + 1):
            for i in range(n):
                for half in range(2):
                    r = i + half * n
                    g_att[:, r, :, :lens[i]].copy_(atts[i][step - 1, :, half])
                    g_cnn[:, r].copy_(cnns[i][step - 1, :, half])
            dphi, nc, na = self.graphs(x.repeat(2, 1, 1), mu_in, t.repeat(2 * n), spks_in, cond_in, None, None, mask, tr, same=step > 1,
                                       staged_len=L)
            for i in range(n):
                for half in range(2):
                    r = i + half * n
                    new_att[i][step - 1, :, half].copy_(na[:, r, :, :T + lens[i]])
                    new_cnn[i][step - 1, :, half].copy_(nc[:, r])
            d, u = dphi.chunk(2, dim=0)
            x = x + dt * ((1.0 + cfm.inference_cfg_rate) * d - cfm.inference_cfg_rate * u)
            t = t + dt
            if step < steps:
                dt = t_span[step + 1] - t
        for i, st in enumerate(sts):
            st.flow_cache["estimator_att_cache"] = new_att[i]
            st.flow_cache["estimator_cnn_cache"] = new_cnn[i]
        return x.float()

    def _solve_euler_graph(self, sts, mu, spks, atts, cnns, lens, T: int, L: int) -> torch.Tensor:
        """A steady window's whole Euler solve in one graph replay (``EulerGraphs``): caches go straight into and come
        straight out of the graph's arenas."""
        eg = self.euler
        n = len(sts)
        key = eg.key(2 * n, T, L)
        eg.capture(*key)
        g, w, _ = eg.graphs[key]
        R = key[0]
        H = R // 2  # graph rows: [cond rows 0..H-1 | uncond rows H..R-1] (x.repeat(2, 1, 1) of H request rows)
        cfm = self.flow.decoder
        w["mask"].zero_()
        w["mask"][:, :, :T] = True  # padding rows: their own new frames only
        w["tr"].fill_(T)  # the arena is shared by every key: this window's frame count, every call
        w["cond"].zero_()
        w["mu"][H:].zero_()
        w["spks"][H:].zero_()
        w["mu"][:n].copy_(mu)
        w["spks"][:n].copy_(spks)
        for i in range(n):
            w["x0"][i].copy_(cfm.rand_noise[0, :, lens[i]:lens[i] + T])
            for half in range(2):
                r = i + half * H
                w["att"][:, :, r, :, :lens[i]] = atts[i][:, :, half]
                w["cnn"][:, :, r] = cnns[i][:, :, half]
                w["mask"][r, :, T:T + lens[i]] = True
        g.replay()
        x = w["o_x"][:n].clone()
        for i, st in enumerate(sts):
            st.flow_cache["estimator_att_cache"] = torch.stack([w["o_att"][:, :, i, :, :T + lens[i]], w["o_att"][:, :, H + i, :, :T + lens[i]]], dim=2)
            st.flow_cache["estimator_cnn_cache"] = torch.stack([w["o_cnn"][:, :, i], w["o_cnn"][:, :, H + i]], dim=2)
        return x.float()

    @staticmethod
    def _estimator(est, x, mu, t, spks, cond, cnn_cache, att_cache, mask):
        """``DiT.forward_chunk`` (eager path) with a key mask for padded caches; returns (dphi, new cnn, new att)."""
        tr = torch.full((x.shape[0],), x.shape[2], dtype=torch.int64, device=x.device)
        return BatchedToken2Wav._estimator_temb(est, x, mu, est.t_embedder(t).unsqueeze(1), spks, cond, cnn_cache, att_cache, mask, tr)

    @staticmethod
    def _estimator_temb(est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out=None):
        """The estimator after the timestep embedding (``t``: [rows, 1, hidden]); this part is graph-captured.

        ``tr`` [rows]: real frames per row (the rest is padding); each causal conv's new cache is taken at the row's real
        end.  ``out = (dphi, new_cnn [depth, ...], new_att [depth, ...])`` buffers receive the results block by block (cast to
        their dtype, as the reference stores its caches); without ``out`` they are allocated in the cache dtype.
        """
        from einops import pack, repeat

        T = x.shape[2]
        if out is None:
            out = (torch.empty_like(x), torch.empty_like(cnn_cache),
                   att_cache.new_empty(att_cache.shape[:3] + (T + att_cache.shape[3], att_cache.shape[4])))
        x = pack([x, mu], "b * t")[0]
        x = pack([x, repeat(spks, "b c -> b c t", t=x.shape[-1])], "b * t")[0]
        x = pack([x, cond], "b * t")[0]
        x = est.in_proj(x.transpose(1, 2))
        for b, block in enumerate(est.blocks):
            x, c, a = _block_chunk(block, x, t, cnn_cache[b], att_cache[b], mask, tr)
            out[1][b].copy_(c)
            out[2][b].copy_(a)
        out[0].copy_(est.final_layer(x, t).transpose(1, 2))
        return out

    # --- vocoder ------------------------------------------------------------------------------------------------------
    def _vocode(self, items, mels, async_result: bool = False):
        """HiFT per group of equal mel / source-cache length, then the reference cross-fade, cache update and trimming on
        the whole group, and one device-to-host copy for the call."""
        groups = defaultdict(list)
        full = []
        for i, (st, _, _) in enumerate(items):
            m = torch.concat([st.hift_cache["mel"], mels[i]], dim=2)
            full.append(m)
            groups[(m.shape[2], st.hift_cache["source"].shape[2])].append(i)
        pieces: list[tuple[int, torch.Tensor]] = []
        voc = self.hift_graphs if self.hift_graphs is not None else (lambda *a: hift_forward(self.hift, *a, decode_tf32=self.hift_decode_tf32))
        for (_, src_len), idx in groups.items():
            M = torch.cat([full[i] for i in idx])
            speech, source = voc(M, torch.cat([items[i][0].hift_cache["source"] for i in idx]),
                                 [chunk_noise_seed(items[i][0].noise_seed, items[i][0].chunks) for i in idx])
            first = src_len == 0  # a stream's first window has no source / speech cache yet
            if not first:
                speech = fade_in_out(speech, torch.cat([items[i][0].hift_cache["speech"] for i in idx]), self.t2w.window)
            mel_c, src_c, sp_c = M[..., -MEL_CACHE:].clone(), source[:, :, -SOURCE_CACHE:].clone(), speech[:, -SOURCE_CACHE:].clone()
            for j, i in enumerate(idx):
                st, _, last = items[i]
                st.hift_cache = dict(mel=mel_c[j:j + 1], source=src_c[j:j + 1], speech=sp_c[j:j + 1])
                st.chunks += 1
                sp = speech[j]
                if not last:
                    sp = torch.cat([torch.zeros(SOURCE_CACHE, device=sp.device, dtype=sp.dtype), sp[:-SOURCE_CACHE]]) if first else sp[:-SOURCE_CACHE]
                pieces.append((i, sp))
        spans: list[tuple[int, int] | None] = [None] * len(items)
        o = 0
        for i, p in pieces:
            spans[i] = (o, o + p.numel())
            o += p.numel()
        if async_result:
            return PendingAudio(torch.cat([p for _, p in pieces]).float() if pieces else None, spans)
        flat = torch.cat([p for _, p in pieces]).float().cpu().numpy() if pieces else np.zeros(0, np.float32)
        return [flat[a:b] if sp is not None else None for sp in spans for a, b in [sp or (0, 0)]]
