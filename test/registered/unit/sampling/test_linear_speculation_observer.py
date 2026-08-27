from types import SimpleNamespace

import pytest

from sglang.srt.sampling.sampling_observer import (
    get_active_linear_speculation_observer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _LinearObserver:
    def __init__(self, *, active=True):
        self.active = active

    def is_linear_speculation_active(self, batch):
        return self.active

    def before_linear_speculation(self, target_logits, batch, vocab_mask):
        raise AssertionError("capability lookup must not begin observation")


def test_get_active_linear_speculation_observer_returns_active_observer():
    observer = _LinearObserver()
    batch = SimpleNamespace()

    assert get_active_linear_speculation_observer(observer, batch) is observer


def test_active_linear_speculation_observer_is_absent_without_observer():
    assert get_active_linear_speculation_observer(None, SimpleNamespace()) is None


def test_linear_speculation_rejects_an_incompatible_observer():
    class SamplingObserverWithoutLinearSpeculation:
        pass

    with pytest.raises(TypeError, match="requires an observer"):
        get_active_linear_speculation_observer(
            SamplingObserverWithoutLinearSpeculation(), SimpleNamespace()
        )


def test_active_linear_speculation_observer_is_absent_when_inactive():
    observer = _LinearObserver(active=False)

    assert get_active_linear_speculation_observer(observer, SimpleNamespace()) is None
