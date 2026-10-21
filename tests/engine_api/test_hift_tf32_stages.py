"""``t2w_hift_decode_tf32`` values: True / False / a list of stages or ``"<stage>:<part>"`` entries (CPU)."""

import pytest

from engine.models.token2wav import PARTS, tf32_stages


def test_tf32_stage_selection():
    every = {(i, q) for i in range(3) for q in PARTS}
    assert tf32_stages(True, 3) == every
    assert tf32_stages(False, 3) == tf32_stages(None, 3) == tf32_stages([], 3) == frozenset()
    assert tf32_stages([1, 2, "0:src", "0:res"], 3) == every - {(0, "ups")}  # the B24 repair
    assert tf32_stages(["2"], 3) == {(2, q) for q in PARTS}
    for bad in ([3], ["0:conv"], [-1]):
        with pytest.raises(AssertionError):
            tf32_stages(bad, 3)
