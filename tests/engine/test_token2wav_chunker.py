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
