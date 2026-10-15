"""Track S Thinker chunk boundaries against the reference yield rules (modeling_minicpmo.py:2205-2260), scripted model."""

import pytest
import torch

from engine.runtime.track_s import thinker_chunks

TERM = 999


class ScriptedThinker:
    """Greedy outputs follow ``script``; the hidden state of a fed token is its one-hot (so alignment is checkable)."""

    def __init__(self, script):
        self.script, self.i, self.dtype = list(script), 0, torch.float32

    def new_cache(self, b, n):
        return None

    def embed(self, ids):
        return torch.nn.functional.one_hot(ids, 1000).float()

    def forward(self, emb, cache, rows):
        tok = self.script[self.i] if self.i < len(self.script) else TERM
        self.i += 1
        logits = torch.full((1, 1000), -1e9)
        logits[0, tok] = 0.0
        return emb, logits


def reference(script, max_new_tokens=256):
    """The reference loop: chunk_generate (11 first / 10 later) and the yield rules of streaming_generate."""
    out, generated_ids, it, chunk_idx = [], [], iter(script), 0
    num_chunks = (max_new_tokens + 9) // 10
    for chunk_idx in range(num_chunks):
        first = chunk_idx == 0
        size = 10 + first
        toks, finished = [], False
        for _ in range(size):
            nxt = next(it, TERM)
            if nxt == TERM:
                finished = True
                break
            toks.append(nxt)
        if first:
            y = toks if finished else toks[:-1]
        elif finished:
            y = generated_ids[-1:] + toks
        else:
            y = generated_ids[-1:] + toks[:-1]
        out.append((y, finished))
        generated_ids += toks
        if finished:
            break
    return out


@pytest.mark.parametrize("n", [0, 1, 9, 10, 11, 12, 20, 21, 22, 37, 60])
def test_chunks_match_reference(n):
    script = [100 + i for i in range(n)] + [TERM]
    ours = [(ids, h, fin) for ids, h, fin in thinker_chunks(ScriptedThinker(script), torch.zeros(5, 1000), {TERM})]
    ref = reference(script)
    assert [(ids, fin) for ids, _, fin in ours] == ref
    for ids, h, _ in ours:  # every passed token carries the hidden state of feeding exactly that token
        assert h.shape[0] == len(ids) and all(int(h[j].argmax()) == t for j, t in enumerate(ids))


class ScriptedTalker:
    """Codec logits that force a scripted frame sequence (EOS = cfg.eos); records the penalty window seen at each frame."""

    def __init__(self, script, eos=6561):
        import types

        self.script, self.i = list(script), 0
        self.cfg = types.SimpleNamespace(eos=eos)
        self.vocab = eos + 1

    def new_cache(self, b, n):
        import types

        return types.SimpleNamespace(k=torch.zeros(1))

    def condition(self, ids, hidden):
        return torch.zeros(ids.numel() + 2, 4)

    def code_embed(self, codes):
        return torch.zeros(1, 1, 4)

    def forward(self, emb, cache, row):
        tok = self.script[self.i] if self.i < len(self.script) else self.cfg.eos
        self.i += 1
        logits = torch.full((1, self.vocab), -1e9)
        logits[0, tok] = 0.0
        return logits


def test_talker_stream_history_excludes_chunk_eos():
    """After a chunk's EOS the penalty window holds the real frames only (HF never adds EOS to all_generated_tokens)."""
    from engine.runtime.track_s import TalkerStream
    from engine.talker_sampler import PAD

    eos = 6561
    ta = ScriptedTalker([5, 6, eos, 7, eos, 8, 9, eos], eos=eos)
    ts = TalkerStream(ta, seed=1)
    ts.feed([1, 2], torch.zeros(2, 4), False)
    assert ts.state.history[0, -2:].tolist() == [5, 6] and eos not in ts.state.history[0].tolist()
    ts.feed([3], torch.zeros(1, 4), False)
    assert ts.state.history[0, -3:].tolist() == [5, 6, 7] and eos not in ts.state.history[0].tolist()
    out = ts.feed([], torch.zeros(0, 4), True)
    assert ts.generated == [5, 6, 7, 8, 9] and ts.done
    assert ts.state.history[0].tolist() == [PAD] * 11 + [5, 6, 7, 8, 9]
    assert [t for chunk, _ in out for t in chunk] == [5, 6, 7, 8, 9]
