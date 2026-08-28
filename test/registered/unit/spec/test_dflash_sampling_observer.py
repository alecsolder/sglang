from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.dflash_worker_v2 import (
    _commit_accept,
    _finalize_dflash_linear_observation,
    _prepare_dflash_acceptance_with_observer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_dflash_declares_sampling_observer_support():
    runner = object.__new__(ModelRunner)
    runner.server_args = SimpleNamespace(dllm_algorithm=None)
    runner.spec_algorithm = SimpleNamespace(
        is_none=lambda: False,
        is_dflash=lambda: True,
    )

    assert runner.supports_sampling_observer()


def test_dflash_rejects_observer_without_linear_speculation_extension():
    runner = object.__new__(ModelRunner)
    runner._sampling_observer = None
    runner.server_args = SimpleNamespace(dllm_algorithm=None)
    runner.spec_algorithm = SimpleNamespace(
        is_none=lambda: False,
        is_dflash=lambda: True,
    )
    observer = SimpleNamespace()

    with pytest.raises(ValueError, match="requires a linear-speculation"):
        runner.sampling_observer = observer


class _Observation:
    def __init__(self, events, output):
        self.events = events
        self.output = output

    def finalize(self, *, accept_tokens, num_accept_tokens):
        self.events.append(("finalize", accept_tokens, num_accept_tokens))
        return self.output


class _Observer:
    def __init__(self, events, output):
        self.events = events
        self.observation = _Observation(events, output)

    def is_linear_speculation_active(self, batch):
        return True

    def before_linear_speculation(self, target_logits, batch, vocab_mask):
        self.events.append(("capture", target_logits.clone(), batch, vocab_mask))
        return self.observation


class _GrammarMask:
    def __init__(self, events):
        self.events = events
        self.vocab_mask = torch.ones(3, 1, dtype=torch.int32)

    def apply(self, logits):
        self.events.append("mask")
        logits.add_(1)


def test_dflash_observer_wraps_grammar_and_acceptance():
    events = []
    auxiliary_output = object()
    batch = SimpleNamespace()
    grammar_mask = _GrammarMask(events)
    logits_output = SimpleNamespace(
        next_token_logits=torch.zeros(3, 5),
        auxiliary_device_output=object(),
    )

    observation = _prepare_dflash_acceptance_with_observer(
        logits_output=logits_output,
        sampling_observer=_Observer(events, auxiliary_output),
        batch=batch,
        grammar_mask=grammar_mask,
        batch_size=1,
        block_size=3,
    )

    assert observation is not None
    assert events[0][0] == "capture"
    assert events[0][1].shape == (1, 3, 5)
    assert events[0][1].eq(0).all()
    assert events[0][2] is batch
    assert events[0][3] is grammar_mask.vocab_mask
    assert events[1] == "mask"
    assert logits_output.next_token_logits.eq(1).all()
    assert logits_output.auxiliary_device_output is None

    accept_tokens, num_accept_tokens = _commit_accept(
        torch.tensor([[3, 4, 5]]),
        torch.tensor([1]),
        torch.tensor([7]),
    )
    events.append("accept")
    _finalize_dflash_linear_observation(
        logits_output=logits_output,
        observation=observation,
        accept_tokens=accept_tokens,
        num_accept_tokens=num_accept_tokens,
    )

    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "capture",
        "mask",
        "accept",
        "finalize",
    ]
    assert events[3][1] is accept_tokens
    assert events[3][2] is num_accept_tokens
    assert accept_tokens.tolist() == [[4, 7, 0]]
    assert num_accept_tokens.tolist() == [2]
    assert num_accept_tokens.dtype == torch.int32
    assert logits_output.auxiliary_device_output is auxiliary_output


def test_dflash_without_observer_clears_reused_output_without_viewing_logits():
    class NoViewTensor:
        def view(self, *args):
            raise AssertionError("no-observer path must not view logits")

    logits_output = SimpleNamespace(
        next_token_logits=NoViewTensor(),
        auxiliary_device_output=object(),
    )

    observation = _prepare_dflash_acceptance_with_observer(
        logits_output=logits_output,
        sampling_observer=None,
        batch=SimpleNamespace(),
        grammar_mask=None,
        batch_size=1,
        block_size=3,
    )

    assert observation is None
    assert logits_output.auxiliary_device_output is None
