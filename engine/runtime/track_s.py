"""Track S: incremental Talker conditioning while the Thinker is still generating (reference mode, one request).

Port of ``MiniCPMO.streaming_generate`` (``modeling_minicpmo.py:2031-2300``), ``ChunkPrefillChunkGenerate``
and ``TTSStreamingGenerator.generate_with_buffer`` (``utils.py``) with ``attention_type = full_attention``:

* Thinker chunks: the first generates 11 tokens and passes the 10 that have hidden states; every later chunk
  generates 10 and passes the previous chunk's last token plus its first 9; the final chunk passes everything up
  to (not including) the terminator (``<|tts_eos|>``, ``<|im_end|>``).
* Talker per chunk: ``emb_text(ids) + normalize(projector_semantic(h))``, plus ``text_eos`` on the final chunk,
  plus ``audio_bos`` on every chunk, prefilled after everything before it (one KV cache for the whole
  utterance, positions continue).  Frames are sampled until EOS (no minimum length on this path, at most 500 per
  chunk); the frame-0 rule applies to the first frame of the utterance only and the 16-frame penalty window spans
  chunks and holds real frames only (the reference never adds a chunk's EOS to ``all_generated_tokens``).  Codec tokens accumulate in a 25-token buffer that carries across chunks; the remainder is emitted as
  the last chunk when the final text chunk's EOS arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from engine.talker_sampler import DecodeState, TalkerSampling, step

THINKER_CHUNK = 10


def history_from(frames: list[int], window: int, device) -> torch.Tensor:
    """The sampler's penalty window [1, window] holding the last ``window`` real frames (PAD before the first)."""
    from engine.talker_sampler import PAD

    tail = frames[-window:]
    return torch.tensor([[PAD] * (window - len(tail)) + tail], dtype=torch.int64, device=device)
TALKER_MAX_PER_CHUNK = 500
AUDIO_CHUNK = 25


def thinker_chunks(thinker, prompt_embeds: torch.Tensor, terminators: set[int], max_new_tokens: int = 256):
    """Greedy Thinker in the streaming chunk scheme; yields (token ids, hidden states [n, H], text_finished).

    Each chunk passes exactly the tokens fed to the model during that chunk, i.e. the ones whose hidden states
    exist: the first chunk's 10 (of 11 generated), then the previous chunk's last token plus 9, and on the final
    chunk everything fed up to the terminator.
    """
    import math

    dev = prompt_embeds.device
    cache = thinker.new_cache(1, prompt_embeds.shape[0] + max_new_tokens + 16)
    row = torch.tensor([0], device=dev)
    gen, hid, passed = [], [], 0
    for chunk in range(math.ceil(max_new_tokens / THINKER_CHUNK)):
        size = THINKER_CHUNK + (1 if chunk == 0 else 0)
        finished = False
        for i in range(size):
            if chunk == 0 and i == 0:
                _, logits = thinker.forward(prompt_embeds[None], cache, row)
            else:
                h, logits = thinker.forward(thinker.embed(torch.tensor([[gen[-1]]], device=dev)), cache, row)
                hid.append(h[0, -1])
            tok = int(logits.argmax(-1))
            if tok in terminators:
                finished = True
                break
            gen.append(tok)
        last_chunk = finished or chunk == math.ceil(max_new_tokens / THINKER_CHUNK) - 1
        ids, h = gen[passed:len(hid)], hid[passed:]
        passed = len(hid)
        yield ids, (torch.stack(h) if h else torch.zeros(0, prompt_embeds.shape[-1], device=dev, dtype=prompt_embeds.dtype)), last_chunk
        if last_chunk:
            return


@dataclass
class TalkerStream:
    talker: object
    seed: int
    cfg: TalkerSampling = field(default_factory=lambda: TalkerSampling(min_new_tokens=0, max_new_tokens=1 << 30))
    max_len: int = 4096

    def __post_init__(self):
        self.cache = self.talker.new_cache(1, self.max_len)
        self.row = torch.tensor([0], device=self.cache.k.device)
        self.state = DecodeState.new([self.seed], self.cfg, device=self.cache.k.device)
        self.buffer: list[int] = []
        self.generated: list[int] = []  # every non-EOS codec frame of the utterance
        self.done = False

    def feed(self, ids: list[int], hidden: torch.Tensor, text_finished: bool):
        """One text chunk -> (codec tokens, is_last) chunks of 25 (the last may be shorter), as the reference yields them."""
        t = self.talker
        dev = self.cache.k.device
        cond = t.condition(torch.tensor(ids, dtype=torch.long, device=dev), hidden) if ids else t.condition(
            torch.zeros(0, dtype=torch.long, device=dev), hidden)
        if not text_finished:
            cond = torch.cat([cond[:-2], cond[-1:]], dim=0)  # text_eos only on the final chunk; audio_bos always
        logits = t.forward(cond[None], self.cache, self.row)
        out = []
        for _ in range(TALKER_MAX_PER_CHUNK):
            tok = int(step(self.state, logits, t.cfg.eos, self.cfg)[0])
            if tok == t.cfg.eos:
                self.state.finished[:] = False  # EOS ends this text chunk only; the utterance continues with the next
                # the reference never adds EOS to all_generated_tokens: the penalty window holds real frames only
                self.state.history = history_from(self.generated, self.cfg.window, self.state.history.device)
                if text_finished:
                    out.append((list(self.buffer), True))
                    self.buffer = []
                    self.done = True
                return out
            self.buffer.append(tok)
            self.generated.append(tok)
            if len(self.buffer) >= AUDIO_CHUNK:
                out.append((self.buffer[:AUDIO_CHUNK], False))
                self.buffer = self.buffer[AUDIO_CHUNK:]
            logits = t.forward(t.code_embed(torch.tensor([[tok]], device=dev)), self.cache, self.row)
        return out
