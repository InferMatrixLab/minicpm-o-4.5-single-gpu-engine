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
