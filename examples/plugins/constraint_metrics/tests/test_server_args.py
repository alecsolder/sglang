import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang_constraint_metrics.config import MAX_TOP_K_ENV, deployment_max_top_k
from sglang_constraint_metrics.integration import (
    _constraint_metrics_linear_speculation_width,
    _max_constraint_metrics_rows,
    alloc_constraint_metrics_memory_pool,
    init_constraint_metrics_model_runner,
    validate_constraint_metrics_server_args,
)
from sglang_constraint_metrics.observer import ConstraintMetricsObserver


def _args(**overrides):
    values = {
        "grammar_backend": "llguidance",
        "max_running_requests": 64,
        "skip_tokenizer_init": False,
        "reasoning_parser": None,
        "enable_strict_thinking": False,
        "disaggregation_mode": "null",
        "pp_size": 1,
        "tokenizer_worker_num": 1,
        "detokenizer_worker_num": 1,
        "use_ray": False,
        "device": "cuda",
        "dllm_algorithm": None,
        "speculative_algorithm": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _validate(args, *, cuda_available=True):
    with (
        patch(
            "sglang_constraint_metrics.integration._resolved_server_args",
            return_value=args,
        ),
        patch("sglang.srt.utils.is_cuda", return_value=cuda_available),
    ):
        validate_constraint_metrics_server_args(None, args)


@pytest.mark.parametrize("grammar_backend", ["llguidance", "xgrammar"])
@pytest.mark.parametrize("speculative_algorithm", [None, "DFLASH"])
def test_constraint_metrics_accepts_supported_server_config(
    grammar_backend, speculative_algorithm
):
    _validate(
        _args(
            grammar_backend=grammar_backend,
            speculative_algorithm=speculative_algorithm,
        )
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"disaggregation_mode": "decode"}, "PD disaggregation"),
        ({"pp_size": 2}, "pipeline parallelism"),
        ({"skip_tokenizer_init": True}, "tokenizer initialization"),
        ({"max_running_requests": None}, "requires --max-running-requests"),
        ({"tokenizer_worker_num": 2}, "one tokenizer and detokenizer"),
        ({"detokenizer_worker_num": 2}, "one tokenizer and detokenizer"),
        ({"use_ray": True}, "does not support --use-ray"),
        ({"device": "cpu"}, "requires NVIDIA CUDA"),
        ({"dllm_algorithm": "low_confidence"}, "diffusion LLM"),
        ({"speculative_algorithm": "EAGLE"}, "has no constraint-metrics adapter"),
        ({"grammar_backend": "outlines"}, "requires --grammar-backend"),
        ({"grammar_backend": "none"}, "requires --grammar-backend"),
        ({"reasoning_parser": "deepseek-r1"}, "does not support"),
        ({"enable_strict_thinking": True}, "does not support"),
    ],
)
def test_constraint_metrics_rejects_unsupported_server_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        _validate(_args(**overrides))


def test_constraint_metrics_rejects_rust_server():
    with (
        patch(
            "sglang.srt.environ.envs.SGLANG_RUST_SERVER.get",
            return_value=True,
        ),
        pytest.raises(ValueError, match="SGLANG_RUST_SERVER"),
    ):
        _validate(_args())


def test_constraint_metrics_rejects_rocm_cuda_device_name():
    with pytest.raises(ValueError, match="requires NVIDIA CUDA"):
        _validate(_args(device="cuda"), cuda_available=False)


def test_constraint_metrics_rejects_dflash_without_observer_api(monkeypatch):
    monkeypatch.delattr(ModelRunner, "requires_linear_speculation_observer")

    with pytest.raises(RuntimeError, match="linear-speculation observer API"):
        _validate(_args(speculative_algorithm="DFLASH"))


def test_constraint_metrics_accepts_adapter_sampling_path():
    raw_args = _args(speculative_algorithm="raw")
    resolved_args = _args(speculative_algorithm="CUSTOM")
    validator = Mock()
    raw_args._validate_constraint_metrics_server_args = validator

    with (
        patch(
            "sglang_constraint_metrics.integration._resolved_server_args",
            return_value=resolved_args,
        ),
        patch("sglang.srt.utils.is_cuda", return_value=True),
    ):
        validate_constraint_metrics_server_args(None, raw_args)

    validator.assert_called_once_with(resolved_args)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [(None, 8), ("0", 0), ("32", 32)],
)
def test_constraint_metrics_top_k_environment(raw_value, expected):
    env = {} if raw_value is None else {MAX_TOP_K_ENV: raw_value}
    with patch.dict(os.environ, env, clear=True):
        deployment_max_top_k.cache_clear()
        assert deployment_max_top_k() == expected
    deployment_max_top_k.cache_clear()


@pytest.mark.parametrize("raw_value", ["invalid", "-1", "33"])
def test_constraint_metrics_rejects_invalid_top_k_environment(raw_value):
    with (
        patch.dict(os.environ, {MAX_TOP_K_ENV: raw_value}),
        pytest.raises(ValueError, match=MAX_TOP_K_ENV),
    ):
        deployment_max_top_k.cache_clear()
        deployment_max_top_k()
    deployment_max_top_k.cache_clear()


def test_constraint_metrics_rejects_another_sampling_observer():
    runner = object.__new__(ModelRunner)
    runner._sampling_observer = object()
    runner.is_draft_worker = False
    runner.ps = SimpleNamespace(attn_tp_rank=0, attn_cp_rank=0)

    with pytest.raises(RuntimeError, match="cannot share"):
        init_constraint_metrics_model_runner(None, runner)


@pytest.mark.parametrize("disable_overlap_schedule", [False, True])
def test_constraint_metrics_prepares_workspace_before_kv_pool(
    disable_overlap_schedule,
):
    runner = object.__new__(ModelRunner)
    runner._sampling_observer = Mock(spec=ConstraintMetricsObserver)
    runner._constraint_metrics_workspace_ready = False
    runner.model_config = SimpleNamespace(vocab_size=202_048)
    runner.server_args = SimpleNamespace(
        max_running_requests=1_632,
        disable_overlap_schedule=disable_overlap_schedule,
    )
    runner.ps = SimpleNamespace(attn_dp_size=1)
    runner.decode_num_tokens_per_req = lambda: 1
    runner.spec_algorithm = SimpleNamespace(is_dflash=lambda: False)
    runner.device = "cuda:2"
    events = []
    runner.forward_stream = Mock()
    runner.forward_stream.synchronize.side_effect = lambda: events.append("sync")
    current_stream = Mock()
    current_stream.synchronize.side_effect = lambda: events.append("sync")
    retention_reserve = Mock()
    retention_reserve.numel.return_value = 64
    runner._sampling_observer.prepare_runtime_workspace.side_effect = lambda **_: (
        events.append("prepare") or 123
    )

    with (
        patch.dict(os.environ, {MAX_TOP_K_ENV: "16"}),
        patch(
            "sglang_constraint_metrics.integration.torch.cuda.stream",
            return_value=nullcontext(),
        ) as use_stream,
        patch(
            "sglang_constraint_metrics.integration.torch.cuda.current_stream",
            return_value=current_stream,
        ) as get_current_stream,
        patch(
            "sglang_constraint_metrics.integration.reserve_device_archive_headroom",
            side_effect=lambda *_: events.append("reserve") or retention_reserve,
        ) as reserve_retention,
    ):
        deployment_max_top_k.cache_clear()
        alloc_constraint_metrics_memory_pool(
            lambda *_: events.append("pool"),
            runner,
            object(),
        )
    deployment_max_top_k.cache_clear()

    assert events == ["prepare", "reserve", "sync", "pool"]
    reserve_retention.assert_called_once_with("cuda:2")
    expected_stream = (
        current_stream if disable_overlap_schedule else runner.forward_stream
    )
    use_stream.assert_called_once_with(expected_stream)
    if disable_overlap_schedule:
        get_current_stream.assert_called_once_with("cuda:2")
    else:
        get_current_stream.assert_not_called()
    workspace_args = (
        runner._sampling_observer.prepare_runtime_workspace.call_args.kwargs
    )
    assert workspace_args["max_top_k"] == 16
    assert workspace_args["linear_speculation_width"] is None


def test_constraint_metrics_uses_dflash_decode_width_for_linear_workspace():
    runner = object.__new__(ModelRunner)
    runner.spec_algorithm = SimpleNamespace(is_dflash=lambda: True)
    runner.decode_num_tokens_per_req = lambda: 16

    assert _constraint_metrics_linear_speculation_width(runner) == 16


def test_constraint_metrics_workspace_shape_can_be_supplied_by_an_adapter():
    runner = object.__new__(ModelRunner)
    runner._constraint_metrics_max_rows = lambda: 72
    runner._constraint_metrics_linear_speculation_width = lambda: 9

    assert _max_constraint_metrics_rows(runner) == 72
    assert _constraint_metrics_linear_speculation_width(runner) == 9


def test_constraint_metrics_adapter_can_defer_to_generic_workspace_shape():
    runner = object.__new__(ModelRunner)
    runner._constraint_metrics_max_rows = lambda: None
    runner._constraint_metrics_linear_speculation_width = lambda: None
    runner.server_args = SimpleNamespace(max_running_requests=64)
    runner.ps = SimpleNamespace(attn_dp_size=2)
    runner.spec_algorithm = SimpleNamespace(is_dflash=lambda: True)
    runner.decode_num_tokens_per_req = lambda: 5

    assert _max_constraint_metrics_rows(runner) == 160
    assert _constraint_metrics_linear_speculation_width(runner) == 5


def test_constraint_metrics_workspace_covers_full_decode_batch():
    runner = object.__new__(ModelRunner)
    runner.server_args = SimpleNamespace(max_running_requests=64)
    runner.ps = SimpleNamespace(attn_dp_size=2)
    runner.decode_num_tokens_per_req = lambda: 5

    assert _max_constraint_metrics_rows(runner) == 160


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
