"""Token2Wav stream chunker against the HF stream driver (modeling_minicpmo.py:2356-2410)."""

import pytest

from engine import token2wav_chunker as tc


def hf_generator(tokens, chunk=25):
    """TTSStreamingGenerator yields (utils.py:845-869): 25-token chunks, then the remainder (possibly empty) as the last."""
    i = 0
    while len(tokens) - i >= chunk:
        yield tokens[i:i + chunk], False
        i += chunk
    yield tokens[i:], True


def hf_driver(tokens, silence=True, pre_lookahead=3):
    buffer = [4218] * 3 if silence else []
    out = []
    for chunk, last in hf_generator(tokens):
        buffer += list(chunk)
        if len(buffer) >= 25 + pre_lookahead:
            out.append((buffer[: 25 + pre_lookahead], last))
            buffer = buffer[25:]
    if len(buffer) > 0:
        out.append((buffer, True))
    return out


def ours(tokens, step=1):
    c = tc.StreamChunker()
    out = []
    for i in range(0, len(tokens), step):
        out += [(w.tokens, w.last) for w in c.push(tokens[i:i + step])]
    w = c.finish()
    return out + ([(w.tokens, w.last)] if w else [])


@pytest.mark.parametrize("n", [0, 1, 24, 25, 26, 49, 50, 51, 137, 500])
@pytest.mark.parametrize("step", [1, 7, 25])
def test_windows_match_the_reference_driver(n, step):
    tokens = [(i * 37) % 4000 for i in range(n)]
    assert ours(tokens, step) == hf_driver(tokens)


def test_window_geometry():
    w = ours(list(range(100)))
    assert w[0][0][:3] == [4218] * 3 and all(len(x) == 28 for x, last in w[:-1]) and w[-1][1]
    assert all(a[0][-3:] == b[0][:3] for a, b in zip(w, w[1:]))  # the 3 lookahead tokens are carried


@pytest.mark.parametrize("mutation", [{"silence": False}, {"pre_lookahead": 0}])
def test_missing_silence_prefix_or_lookahead_is_detected(mutation):
    tokens = list(range(80))
    assert ours(tokens) != hf_driver(tokens, **mutation)


def test_flush_once():
    c = tc.StreamChunker()
    c.push([1, 2])
    assert c.finish().last
    with pytest.raises(RuntimeError):
        c.push([3])


def test_first_chunk_ramp_windows():
    """first_chunk=8: the first window is 3 silence + 5 tokens + 3 lookahead, later windows are the reference 25 + 3,
    consecutive windows overlap by the 3 lookahead tokens, and every token is decoded exactly once."""
    from engine.token2wav_chunker import CHUNK, PRE_LOOKAHEAD, SILENCE_PREFIX, SILENCE_TOKEN, StreamChunker

    toks = list(range(1, 81))
    ch = StreamChunker(first_chunk=8)
    ws = []
    for t in toks:
        ws += ch.push([t])
    ws.append(ch.finish())
    assert ws[0].tokens == [SILENCE_TOKEN] * SILENCE_PREFIX + toks[:8]
    assert all(len(w.tokens) == CHUNK + PRE_LOOKAHEAD for w in ws[1:-1])
    for a, b in zip(ws, ws[1:]):
        assert a.tokens[-PRE_LOOKAHEAD:] == b.tokens[:PRE_LOOKAHEAD]
    decoded = [w.tokens[:-PRE_LOOKAHEAD] for w in ws[:-1]] + [ws[-1].tokens]
    flat = [t for d in decoded for t in d]
    assert flat == [SILENCE_TOKEN] * SILENCE_PREFIX + toks


def test_first_chunk_ramp_ready_after_first_chunk_frames():
    from engine.token2wav_chunker import StreamChunker

    ch = StreamChunker(first_chunk=8)
    assert [len(ch.push([i])) for i in range(1, 9)] == [0] * 7 + [1]
