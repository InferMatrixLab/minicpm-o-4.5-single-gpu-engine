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

import numpy as np
import torch
import torch.nn.functional as F

from engine.models.token2wav import MEL_CACHE, SOURCE_CACHE, StreamState, Token2Wav, chunk_noise_seed, fade_in_out, hift_forward, strict_fp32

TRIM_SUFFIX = 100


def _trim(cache: torch.Tensor, keep: int, dim: int) -> torch.Tensor:
    if cache.shape[dim] > keep + TRIM_SUFFIX:
        return torch.cat([cache.narrow(dim, 0, keep), cache.narrow(dim, cache.shape[dim] - TRIM_SUFFIX, TRIM_SUFFIX)], dim=dim)
    return cache


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


class BatchedToken2Wav:
    def __init__(self, t2w: Token2Wav, use_graphs: bool = False, compile_estimator: bool = False):
        self.t2w, self.flow, self.hift = t2w, t2w.flow, t2w.hift
        self.profile: dict | None = None  # stage -> [calls, seconds, items] when set
        self.dtype = torch.float16 if t2w.float16 else torch.float32
        self.graphs = None
        if use_graphs:
            from engine.runtime.dit_graph import EstimatorGraphs

            est = self.flow.decoder.estimator
            fn = BatchedToken2Wav._estimator_temb
            if compile_estimator:  # inductor fuses the adaLN / norm / activation chains (dynamic shapes: one compile)
                fn = torch.compile(fn, dynamic=True)
            self.graphs = EstimatorGraphs(est, fn, next(est.parameters()).dtype, self.dtype)

    @torch.inference_mode()
    def decode(self, items: list[tuple[StreamState, list[int], bool]]) -> list[np.ndarray]:
        """``items[i] = (stream, window tokens, last)`` -> float32 samples per item, exactly as ``Token2Wav.decode`` each."""
        groups = defaultdict(list)
        for i, (_, toks, last) in enumerate(items):
            groups[(len(toks), bool(last))].append(i)
        mels: list[torch.Tensor | None] = [None] * len(items)
        with strict_fp32():
            with torch.amp.autocast("cuda", dtype=self.dtype):
                for (_, last), idx in groups.items():
                    for i, m in zip(idx, self._prof("flow", len(idx), self._flow, [items[i][0] for i in idx], [items[i][1] for i in idx], last)):
                        mels[i] = m
            return self._prof("vocode", len(items), self._vocode, items, mels)

    def _prof(self, name, n, fn, *a):
        if self.profile is None:
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
    def _flow(self, sts: list[StreamState], windows: list[list[int]], last: bool) -> list[torch.Tensor]:
        flow = self.flow
        n = len(sts)
        tok = torch.tensor(windows, dtype=torch.int32, device="cuda")
        spk = flow.spk_embed_affine_layer(F.normalize(torch.cat([st.prompt.spk for st in sts]), dim=1))
        h = flow.encoder_proj(self._prof("encoder", n, self._encoder, sts, flow.input_embedding(tok), last))
        mel = self._prof("solve", n, self._solve, sts, h.transpose(1, 2).contiguous(), spk, None, not last)
        out = []
        for i, st in enumerate(sts):
            keep = st.prompt.mels.shape[1]
            c = st.flow_cache
            c["estimator_att_cache"] = _trim(c["estimator_att_cache"], keep, 4)
            c["conformer_att_cache"] = _trim(c["conformer_att_cache"], keep, 3)
            out.append(mel[i:i + 1])
        return out

    def _encoder(self, sts: list[StreamState], xs: torch.Tensor, last: bool) -> torch.Tensor:
        """``UpsampleConformerEncoderV2.forward_chunk`` over a batch with left-padded caches; updates each stream's caches."""
        enc = self.flow.encoder
        n, dev = len(sts), xs.device
        atts = [st.flow_cache["conformer_att_cache"] for st in sts]  # [depth1 + depth2, 1, nh, 2 * t1, 2 dk]
        cnns = torch.cat([st.flow_cache["conformer_cnn_cache"] for st in sts])  # [n, c, 2 + 2 * stride]
        d1 = len(enc.encoders)
        off1 = [a.shape[3] // 2 for a in atts]
        L1 = max(off1)
        xs, _, _ = enc.embed(xs, None)
        if last:
            xs = F.pad(xs, (0, 0, 0, enc.pre_lookahead_layer.pre_lookahead_len))
        xs, new_cnn1 = enc.pre_lookahead_layer.forward_chunk(xs, cache=cnns[:, :, :2])
        T1 = xs.shape[1]
        A1 = atts[0].new_zeros((d1, n) + atts[0].shape[2:3] + (L1, atts[0].shape[4]))
        mask1 = torch.ones(n, T1, L1 + T1, dtype=torch.bool, device=dev)
        for i, a in enumerate(atts):
            A1[:, i, :, L1 - off1[i]:] = a[:d1, 0, :, :off1[i]]
            mask1[i, :, :L1 - off1[i]] = False
        pos = enc.embed.position_encoding(offset=None, size=L1 + T1)
        new1 = []
        for idx, layer in enumerate(enc.encoders):
            xs, _, c, _ = layer(xs, mask1, pos, att_cache=A1[idx])
            new1.append(c)
        new1 = torch.stack(new1)  # [d1, n, nh, L1 + T1, 2 dk]
        xs = xs.transpose(1, 2).contiguous()
        xs, _, new_cnn2 = enc.up_layer.forward_chunk(xs, None, cache=cnns[:, :, 2:])
        xs = xs.transpose(1, 2).contiguous()
        xs, _, _ = enc.up_embed(xs, None)
        T2, s = xs.shape[1], enc.up_layer.stride
        L2 = L1 * s
        A2 = atts[0].new_zeros((len(enc.up_encoders), n) + atts[0].shape[2:3] + (L2, atts[0].shape[4]))
        mask2 = torch.ones(n, T2, L2 + T2, dtype=torch.bool, device=dev)
        for i, a in enumerate(atts):
            o2 = off1[i] * s
            A2[:, i, :, L2 - o2:] = a[d1:, 0]
            mask2[i, :, :L2 - o2] = False
        pos = enc.embed.position_encoding(offset=None, size=L2 + T2)
        new2 = []
        for idx, layer in enumerate(enc.up_encoders):
            xs, _, c, _ = layer(xs, mask2, pos, att_cache=A2[idx])
            new2.append(c)
        new2 = torch.stack(new2)
        if enc.normalize_before:
            xs = enc.after_norm(xs)
        new_cnn = torch.cat([new_cnn1, new_cnn2], dim=2)
        for i, st in enumerate(sts):
            c1 = new1[:, i:i + 1, :, L1 - off1[i]:]
            c2 = new2[:, i:i + 1, :, L2 - off1[i] * s:]
            st.flow_cache["conformer_att_cache"] = torch.cat([c1.repeat(1, 1, 1, 2, 1), c2], dim=0)
            st.flow_cache["conformer_cnn_cache"] = new_cnn[i:i + 1].clone()
        return xs

    @torch.inference_mode()
    def setup(self, feats: tuple):
        """``Token2Wav.setup`` (reference ``Flow.setup_cache``) on this module's solver: the prompt tokens plus 3 silence
        tokens through the flow encoder, then the Euler solve over the prompt frames with the prompt mel as condition and
        empty caches (frames padded to a graph bucket; padding is masked and the conv caches come from the real end)."""
        tokens, _, spk, mels, _ = feats
        flow = self.flow
        from engine.models.token2wav import SILENCE_TOKEN

        pad = torch.full((1, 3), SILENCE_TOKEN, device=tokens.device, dtype=tokens.dtype)
        with strict_fp32(), torch.amp.autocast("cuda", dtype=self.dtype):
            spks = flow.spk_embed_affine_layer(F.normalize(spk, dim=1))
            h, cnn, att = flow.encoder.forward_chunk(xs=flow.input_embedding(torch.cat([tokens, pad], dim=1)), last_chunk=False, cnn_cache=None,
                                                    att_cache=None)
            h = flow.encoder_proj(h)
            assert h.shape[1] == mels.shape[1], (h.shape, mels.shape)
            est = flow.decoder.estimator
            steps, depth = self.t2w.n_timesteps, len(est.blocks)
            blk = est.blocks[0]
            dt = next(est.parameters()).dtype
            st = StreamState(prompt=None, flow_cache={
                "conformer_cnn_cache": cnn.clone(), "conformer_att_cache": att.clone(),
                "estimator_cnn_cache": torch.zeros(steps, depth, 2, blk.conv.in_channels + blk.conv.out_channels, 2, device="cuda", dtype=dt),
                "estimator_att_cache": torch.zeros(steps, depth, 2, blk.attn.num_heads, 0, 2 * blk.attn.head_dim, device="cuda", dtype=dt)},
                hift_cache={})
            self._solve([st], h.transpose(1, 2).contiguous(), spks, cond=mels.transpose(1, 2).contiguous(), keep_cache=True)
        return self.t2w.prompt(feats, st.flow_cache)

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
               keep_cache: bool = True) -> torch.Tensor:
        """``CausalConditionalCFM.forward_chunk`` (cosine schedule, CFG) over a batch with right-padded estimator caches.

        All rows have ``T = mu.shape[2]`` real frames; the solve may run at a padded frame count (see ``_t_pad``): padded
        frames are masked as keys, never feed real frames (causal convs, per-frame norms), and are dropped.
        """
        cfm = self.flow.decoder
        est = cfm.estimator
        n, T, dev = len(sts), mu.shape[2], mu.device
        atts = [st.flow_cache["estimator_att_cache"] for st in sts]  # [steps, depth, 2, nh, L, 2 dk]
        cnns = [st.flow_cache["estimator_cnn_cache"] for st in sts]  # [steps, depth, 2, c, 2]
        lens = [a.shape[4] for a in atts]
        L = max(lens)
        Tp = self._t_pad(T, keep_cache, L == 0)
        steps, depth = self.t2w.n_timesteps, atts[0].shape[1]
        assert all(a.shape[0] == steps for a in atts) and all(c.shape[0] == steps for c in cnns)
        A = atts[0].new_zeros((steps, depth, 2 * n) + atts[0].shape[3:4] + (L, atts[0].shape[5]))
        C = cnns[0].new_empty((steps, depth, 2 * n) + cnns[0].shape[3:])
        mask = torch.zeros(2 * n, Tp, Tp + L, dtype=torch.bool, device=dev)
        for i in range(n):
            for half in range(2):
                r = i + half * n
                A[:, :, r, :, :lens[i]] = atts[i][:, :, half]
                C[:, :, r] = cnns[i][:, :, half]
                mask[r, :, :T] = True
                mask[r, :, Tp:Tp + lens[i]] = True
        tr = torch.full((2 * n,), T, dtype=torch.int64, device=dev)
        padT = (lambda z: F.pad(z, (0, Tp - T))) if Tp > T else (lambda z: z)
        x = padT(torch.stack([cfm.rand_noise[0, :, o:o + T] for o in lens]).to(mu.dtype))
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
                for r in (i, n + i):  # this request's keys: its T new frames, then its own cache
                    na = new_att[:, :, r]
                    rows.append(torch.cat([na[:, :, :, :T], na[:, :, :, Tp:Tp + lens[i]]], dim=3) if Tp > T else na[:, :, :, :T + lens[i]])
                st.flow_cache["estimator_att_cache"] = torch.stack(rows, dim=2)
                st.flow_cache["estimator_cnn_cache"] = torch.stack([new_cnn[:, :, i], new_cnn[:, :, n + i]], dim=2)
        return x[:, :, :T].float()

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

        rows, T = x.shape[0], x.shape[2]
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
    def _vocode(self, items, mels) -> list[np.ndarray]:
        out: list[np.ndarray | None] = [None] * len(items)
        groups = defaultdict(list)
        full = []
        for i, (st, _, _) in enumerate(items):
            m = torch.concat([st.hift_cache["mel"], mels[i]], dim=2)
            full.append(m)
            groups[(m.shape[2], st.hift_cache["source"].shape[2])].append(i)
        for _, idx in groups.items():
            speech, source = hift_forward(self.hift, torch.cat([full[i] for i in idx]), torch.cat([items[i][0].hift_cache["source"] for i in idx]),
                                          [chunk_noise_seed(items[i][0].noise_seed, items[i][0].chunks) for i in idx])
            for j, i in enumerate(idx):
                st, _, last = items[i]
                sp, h = speech[j:j + 1], st.hift_cache
                first = h["speech"].shape[-1] == 0
                if not first:
                    sp = fade_in_out(sp, h["speech"], self.t2w.window)
                st.hift_cache = dict(mel=full[i][..., -MEL_CACHE:].clone(), source=source[j:j + 1, :, -SOURCE_CACHE:].clone(),
                                     speech=sp[:, -SOURCE_CACHE:].clone())
                if not last:
                    sp = torch.cat([torch.zeros(1, SOURCE_CACHE, device=sp.device), sp[:, :-SOURCE_CACHE]], dim=1) if first else sp[:, :-SOURCE_CACHE]
                st.chunks += 1
                out[i] = sp.float().cpu().numpy()[0]
        return out
