"""Explicitly enabled constraint-metrics server integration."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Iterable, Optional

import fastapi
import torch

from sglang.srt.managers.io_struct import AbortReq, EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang_constraint_metrics.config import (
    CONSTRAINT_METRICS_STREAM_ENABLED_KEY,
    CONSTRAINT_METRICS_STREAM_KEY,
    CONSTRAINT_METRICS_TRANSPORT_KEY,
    config_from_sampling_custom_params,
    deployment_max_top_k,
    lower_constraint_metrics_sampling_params,
)
from sglang_constraint_metrics.observer import (
    ConstraintMetricsObserver,
    constraint_metrics_state_for_req,
    discard_constraint_metrics_state_for_req,
    prepare_streamed_constraint_metrics,
    release_streamed_constraint_metrics,
    snapshot_constraint_metrics,
)
from sglang_constraint_metrics.records import reserve_device_archive_headroom

logger = logging.getLogger(__name__)

_CUSTOMIZED_INFO_KEYS = frozenset(
    {
        CONSTRAINT_METRICS_STREAM_ENABLED_KEY,
        CONSTRAINT_METRICS_STREAM_KEY,
        CONSTRAINT_METRICS_TRANSPORT_KEY,
    }
)


def _resolved_server_args(server_args: Any) -> Any:
    from sglang.srt.arg_groups import overrides

    resolver = getattr(overrides, "resolving_view", None)
    if resolver is None:
        resolver = overrides.resolved_view
    return resolver(server_args)


def validate_constraint_metrics_server_args(
    _result: None,
    server_args: Any,
) -> None:
    """Validate the selected plugin against the resolved server configuration."""
    from sglang.srt.environ import envs
    from sglang.srt.utils import is_cuda

    deployment_max_top_k()
    cfg = _resolved_server_args(server_args)
    if cfg.grammar_backend not in {"llguidance", "xgrammar"}:
        raise ValueError(
            "constraint_metrics requires --grammar-backend llguidance or xgrammar"
        )
    if cfg.skip_tokenizer_init:
        raise ValueError("constraint_metrics requires tokenizer initialization")
    if cfg.max_running_requests is None:
        raise ValueError(
            "constraint_metrics requires --max-running-requests so its CUDA "
            "workspace is bounded before KV-cache sizing"
        )
    if cfg.reasoning_parser or cfg.enable_strict_thinking:
        raise ValueError(
            "constraint_metrics does not support "
            "--reasoning-parser or --enable-strict-thinking"
        )
    if cfg.disaggregation_mode != "null":
        raise ValueError("constraint_metrics does not support PD disaggregation")
    if cfg.pp_size > 1:
        raise ValueError("constraint_metrics does not support pipeline parallelism")
    if cfg.tokenizer_worker_num > 1 or cfg.detokenizer_worker_num > 1:
        raise ValueError(
            "constraint_metrics requires one tokenizer and detokenizer worker"
        )
    if envs.SGLANG_RUST_SERVER.get():
        raise ValueError("constraint_metrics does not support SGLANG_RUST_SERVER")
    if cfg.use_ray:
        raise ValueError("constraint_metrics does not support --use-ray")
    if not is_cuda() or cfg.device != "cuda":
        raise ValueError("constraint_metrics currently requires NVIDIA CUDA")

    if cfg.dllm_algorithm is not None:
        raise ValueError("constraint_metrics does not support diffusion LLM decoding")

    if cfg.speculative_algorithm == "DFLASH":
        try:
            from sglang.srt.sampling import sampling_observer
        except ImportError as error:
            raise RuntimeError(
                "constraint_metrics with DFLASH requires SGLang's "
                "linear-speculation observer API"
            ) from error
        if not hasattr(
            sampling_observer, "LinearSpeculationObservation"
        ) or not hasattr(ModelRunner, "requires_linear_speculation_observer"):
            raise RuntimeError(
                "constraint_metrics with DFLASH requires SGLang's "
                "linear-speculation observer API"
            )

    adapter_validator = getattr(
        server_args,
        "_validate_constraint_metrics_server_args",
        None,
    )
    if adapter_validator is not None:
        adapter_validator(cfg)
    elif cfg.speculative_algorithm not in (None, "DFLASH"):
        raise ValueError(
            "constraint_metrics supports normal decoding and DFLASH; "
            "the configured sampling path has no constraint-metrics adapter"
        )


class _ConstraintMetricsAbortAwareSender(SenderWrapper):
    def __init__(self, sender: SenderWrapper) -> None:
        super().__init__(sender.socket)
        self._sender = sender

    def send_output(self, output: Any, recv_obj: Any = None) -> None:
        if isinstance(output, AbortReq) and isinstance(recv_obj, Req):
            discard_constraint_metrics_state_for_req(recv_obj)
        self._sender.send_output(output, recv_obj)


class _ConstraintMetricsSchedulerOutputStreamerMixin:
    __slots__ = ()

    has_additional_customized_info = True

    def __post_init__(self) -> None:
        self._constraint_metrics_active_for_output = False
        super().__post_init__()

    def _stream_output_generation(
        self,
        reqs: list[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
        is_idle_batch: bool = False,
    ) -> None:
        self._constraint_metrics_active_for_output = any(
            constraint_metrics_state_for_req(req) is not None for req in reqs
        )
        try:
            if self._constraint_metrics_active_for_output:
                prepare_streamed_constraint_metrics(reqs)
            super()._stream_output_generation(
                reqs,
                return_logprob,
                skip_req,
                is_idle_batch,
            )
        finally:
            self._constraint_metrics_active_for_output = False

    def build_additional_customized_info(self, reqs: list[Req]) -> dict[str, list]:
        if not self._constraint_metrics_active_for_output:
            return super().build_additional_customized_info(reqs)
        additional_info = (
            super().build_additional_customized_info(reqs)
            if self._base_output_streamer_wants_customized_info()
            else {}
        )
        snapshots = snapshot_constraint_metrics(reqs)
        release_streamed_constraint_metrics(reqs)
        if all(snapshot is None for snapshot in snapshots):
            return additional_info
        if CONSTRAINT_METRICS_TRANSPORT_KEY in additional_info:
            raise ValueError(
                f"duplicate customized_info key: {CONSTRAINT_METRICS_TRANSPORT_KEY}"
            )
        additional_info[CONSTRAINT_METRICS_TRANSPORT_KEY] = [
            [] if snapshot is None else [snapshot] for snapshot in snapshots
        ]
        return additional_info

    def should_build_additional_customized_info(self) -> bool:
        return (
            self._constraint_metrics_active_for_output
            or self._base_output_streamer_wants_customized_info()
        )

    def _base_output_streamer_wants_customized_info(self) -> bool:
        return (
            super().has_additional_customized_info
            and super().should_build_additional_customized_info()
        )


class ConstraintMetricsSchedulerOutputStreamer(
    _ConstraintMetricsSchedulerOutputStreamerMixin,
    SchedulerOutputStreamer,
):
    __slots__ = ("_constraint_metrics_active_for_output",)


def _compose_constraint_metrics_output_streamer(
    base_output_streamer_class: type[SchedulerOutputStreamer],
) -> type[SchedulerOutputStreamer]:
    if not issubclass(base_output_streamer_class, SchedulerOutputStreamer):
        raise TypeError(
            "constraint metrics require a SchedulerOutputStreamer subclass, got "
            f"{base_output_streamer_class!r}"
        )
    if issubclass(
        base_output_streamer_class,
        _ConstraintMetricsSchedulerOutputStreamerMixin,
    ):
        return base_output_streamer_class
    if base_output_streamer_class is SchedulerOutputStreamer:
        return ConstraintMetricsSchedulerOutputStreamer
    return type(
        f"ConstraintMetrics{base_output_streamer_class.__name__}",
        (_ConstraintMetricsSchedulerOutputStreamerMixin, base_output_streamer_class),
        {
            "__module__": __name__,
            "__slots__": ("_constraint_metrics_active_for_output",),
        },
    )


def _max_constraint_metrics_rows(model_runner: ModelRunner) -> int:
    provider = getattr(model_runner, "_constraint_metrics_max_rows", None)
    if provider is not None:
        provided_max_rows = provider()
        if provided_max_rows is not None:
            return provided_max_rows
    max_running_requests = _resolved_server_args(
        model_runner.server_args
    ).max_running_requests
    if max_running_requests is None:
        raise RuntimeError(
            "constraint metrics require an explicit max_running_requests"
        )
    max_batch_size = max(
        1,
        max_running_requests // model_runner.ps.attn_dp_size,
    )
    return max_batch_size * model_runner.decode_num_tokens_per_req()


def _constraint_metrics_linear_speculation_width(
    model_runner: ModelRunner,
) -> Optional[int]:
    provider = getattr(
        model_runner,
        "_constraint_metrics_linear_speculation_width",
        None,
    )
    if provider is not None:
        provided_width = provider()
        if provided_width is not None:
            return provided_width
    if model_runner.spec_algorithm.is_dflash():
        return model_runner.decode_num_tokens_per_req()
    return None


def init_constraint_metrics_model_runner(
    _result: None,
    model_runner: ModelRunner,
    *_args,
    **_kwargs,
) -> None:
    model_runner._constraint_metrics_workspace_ready = False
    if (
        not model_runner.is_draft_worker
        and model_runner.ps.attn_tp_rank == 0
        and model_runner.ps.attn_cp_rank == 0
    ):
        existing_observer = model_runner.sampling_observer
        if existing_observer is not None:
            raise RuntimeError(
                "constraint metrics cannot share ModelRunner.sampling_observer "
                f"with {type(existing_observer).__name__}"
            )
        model_runner.sampling_observer = ConstraintMetricsObserver(
            model_config=model_runner.model_config,
        )


def alloc_constraint_metrics_memory_pool(
    base_alloc_memory_pool,
    model_runner: ModelRunner,
    memory_pool_config=None,
):
    retention_reserve = None
    if (
        isinstance(model_runner.sampling_observer, ConstraintMetricsObserver)
        and not model_runner._constraint_metrics_workspace_ready
    ):
        max_rows = _max_constraint_metrics_rows(model_runner)
        cfg = _resolved_server_args(model_runner.server_args)
        workspace_stream = (
            torch.cuda.current_stream(model_runner.device)
            if cfg.disable_overlap_schedule
            else model_runner.forward_stream
        )
        with torch.cuda.stream(workspace_stream):
            reserved_bytes = model_runner.sampling_observer.prepare_runtime_workspace(
                max_rows=max_rows,
                max_top_k=deployment_max_top_k(),
                vocab_size=model_runner.model_config.vocab_size,
                device=model_runner.device,
                linear_speculation_width=_constraint_metrics_linear_speculation_width(
                    model_runner
                ),
            )
            retention_reserve = reserve_device_archive_headroom(model_runner.device)
            reserved_bytes += retention_reserve.numel()
        workspace_stream.synchronize()
        model_runner._constraint_metrics_workspace_ready = True
        logger.info(
            "Reserved %.2f GiB for constraint-metrics capture before KV sizing "
            "(%d rows).",
            reserved_bytes / (1 << 30),
            max_rows,
        )
    try:
        return base_alloc_memory_pool(model_runner, memory_pool_config)
    finally:
        del retention_reserve


def init_constraint_metrics_ipc_channels(
    _result: None,
    scheduler: Scheduler,
    _port_args,
) -> None:
    scheduler.ipc_channels = replace(
        scheduler.ipc_channels,
        send_to_tokenizer=_ConstraintMetricsAbortAwareSender(
            scheduler.ipc_channels.send_to_tokenizer
        ),
    )


def get_constraint_metrics_output_streamer_class(
    base_output_streamer_class: type[SchedulerOutputStreamer],
    scheduler: Scheduler,
) -> type[SchedulerOutputStreamer]:
    return _compose_constraint_metrics_output_streamer(base_output_streamer_class)


def _lower_request(obj: GenerateReqInput | EmbeddingReqInput) -> bool:
    if isinstance(obj, EmbeddingReqInput):
        return False
    sampling_params = obj.sampling_params
    if sampling_params is None:
        return False
    if isinstance(sampling_params, dict):
        lowered = lower_constraint_metrics_sampling_params(sampling_params)
        if lowered is sampling_params:
            return False
        obj.sampling_params = lowered
        return True

    lowered_sampling_params = None
    for index, params in enumerate(sampling_params):
        if params is None:
            continue
        lowered = lower_constraint_metrics_sampling_params(params)
        if lowered is params:
            continue
        if lowered_sampling_params is None:
            lowered_sampling_params = list(sampling_params)
        lowered_sampling_params[index] = lowered
    if lowered_sampling_params is None:
        return False
    obj.sampling_params = lowered_sampling_params
    return True


def _validate_constraint_metrics_request(
    obj: GenerateReqInput | EmbeddingReqInput,
) -> None:
    if isinstance(obj, EmbeddingReqInput):
        return
    if obj.sampling_params is None:
        return

    sampling_params_by_item = (
        [obj.sampling_params]
        if isinstance(obj.sampling_params, dict)
        else obj.sampling_params
    )
    for index, sampling_params in enumerate(sampling_params_by_item):
        sampling_params = sampling_params or {}
        config = config_from_sampling_custom_params(
            sampling_params.get("custom_params") or {}
        )
        if config is None:
            continue
        if not any(
            sampling_params.get(name)
            for name in ("ebnf", "regex", "json_schema", "structural_tag")
        ):
            raise ValueError("sampling_params.constraint_metrics requires a grammar")
        if config.stream_all_token_metrics and not obj.stream:
            raise ValueError(
                "sampling_params.constraint_metrics.stream_all_token_metrics "
                "requires stream=true"
            )
        if obj.input_embeds is not None:
            raise ValueError(
                "constraint metrics do not support input_embeds retraction"
            )
        if config.emits_tables and config.top_k > deployment_max_top_k():
            raise ValueError(
                f"sampling_params.constraint_metrics.top_k={config.top_k} exceeds "
                "this deployment's SGLANG_CONSTRAINT_METRICS_MAX_TOP_K="
                f"{deployment_max_top_k()}"
            )
        custom_logit_processor = obj.custom_logit_processor
        if isinstance(custom_logit_processor, list):
            if isinstance(obj.sampling_params, dict):
                custom_logit_processor = next(
                    (
                        processor
                        for processor in custom_logit_processor
                        if processor is not None
                    ),
                    None,
                )
            else:
                custom_logit_processor = (
                    custom_logit_processor[index]
                    if index < len(custom_logit_processor)
                    else None
                )
        if custom_logit_processor is not None:
            raise ValueError(
                "constraint metrics do not support custom_logit_processor ordering"
            )


def update_constraint_metrics_request_meta_info(
    base_update_request_meta_info,
    tokenizer_manager: TokenizerManager,
    meta_info: dict,
    state: ReqState,
    customized_info: dict,
    index: int,
    finish_reason: Optional[dict],
) -> None:
    if customized_info.keys().isdisjoint(_CUSTOMIZED_INFO_KEYS):
        return base_update_request_meta_info(
            tokenizer_manager,
            meta_info,
            state,
            customized_info,
            index,
            finish_reason,
        )

    from sglang_constraint_metrics.observer import (
        ConstraintMetricsResponse,
        ConstraintMetricsStreamRecord,
    )

    other_info = {
        key: values
        for key, values in customized_info.items()
        if key not in _CUSTOMIZED_INFO_KEYS
    }
    if other_info:
        base_update_request_meta_info(
            tokenizer_manager,
            meta_info,
            state,
            other_info,
            index,
            finish_reason,
        )

    if CONSTRAINT_METRICS_STREAM_ENABLED_KEY in customized_info:
        enabled_values = customized_info[CONSTRAINT_METRICS_STREAM_ENABLED_KEY][index]
        if not isinstance(enabled_values, list):
            raise TypeError("invalid streamed constraint metrics marker")
        if enabled_values and not (
            all(value is True for value in enabled_values)
            or all(value is None for value in enabled_values)
        ):
            raise TypeError("invalid streamed constraint metrics marker")
        stream_enabled = bool(enabled_values) and enabled_values[0] is True
        if CONSTRAINT_METRICS_STREAM_KEY not in customized_info:
            raise TypeError("streamed constraint metrics payload is missing")
        stream_records = customized_info[CONSTRAINT_METRICS_STREAM_KEY][index]
        if not isinstance(stream_records, list):
            raise TypeError("invalid streamed constraint metrics payload")
        if stream_enabled:
            materialized_records = []
            for record in stream_records:
                if record is None or isinstance(record, dict):
                    materialized_records.append(record)
                elif isinstance(record, ConstraintMetricsStreamRecord):
                    materialized_records.append(record.materialize())
                else:
                    raise TypeError("invalid streamed constraint metrics record")
            if tokenizer_manager.incremental_streaming_output:
                meta_info[CONSTRAINT_METRICS_STREAM_KEY] = materialized_records
            else:
                tokenizer_manager._accumulate_request_meta_info(
                    meta_info,
                    state,
                    CONSTRAINT_METRICS_STREAM_KEY,
                    materialized_records,
                )
    elif CONSTRAINT_METRICS_STREAM_KEY in customized_info:
        raise TypeError("streamed constraint metrics marker is missing")

    if CONSTRAINT_METRICS_TRANSPORT_KEY not in customized_info:
        return
    values = customized_info[CONSTRAINT_METRICS_TRANSPORT_KEY][index]
    if not values:
        return
    if finish_reason is None:
        raise ValueError("constraint metrics were emitted before request completion")
    if len(values) != 1 or not isinstance(values[0], ConstraintMetricsResponse):
        raise TypeError("invalid constraint metrics response payload")
    meta_info["constraint_metrics"] = values[0].materialize_constraint_metrics()


async def generate_constraint_metrics_request(
    base_generate_request,
    tokenizer_manager: TokenizerManager,
    obj: GenerateReqInput | EmbeddingReqInput,
    request: Optional[fastapi.Request] = None,
):
    _lower_request(obj)
    _validate_constraint_metrics_request(obj)
    async for response in base_generate_request(tokenizer_manager, obj, request):
        yield response


def coalesce_constraint_metrics_streaming_chunks(
    base_coalesce_streaming_chunks,
    tokenizer_manager: TokenizerManager,
    out_list: list,
    rid: str,
    customized_info_keys: Optional[Iterable[str]] = None,
) -> dict:
    keys = set(customized_info_keys or ())
    if any(
        CONSTRAINT_METRICS_STREAM_KEY in (chunk.get("meta_info") or {})
        for chunk in out_list
    ):
        keys.add(CONSTRAINT_METRICS_STREAM_KEY)
    return base_coalesce_streaming_chunks(
        tokenizer_manager,
        out_list,
        rid,
        keys,
    )


async def wait_one_constraint_metrics_response(
    base_wait_one_response,
    tokenizer_manager: TokenizerManager,
    obj,
    request=None,
):
    stream_offset = 0
    async for out in base_wait_one_response(tokenizer_manager, obj, request):
        meta_info = out.get("meta_info") or {}
        if (
            CONSTRAINT_METRICS_STREAM_KEY in meta_info
            and not tokenizer_manager.incremental_streaming_output
        ):
            records = meta_info[CONSTRAINT_METRICS_STREAM_KEY]
            if len(records) < stream_offset:
                raise ValueError("constraint metrics stream history shrank")
            meta_info[CONSTRAINT_METRICS_STREAM_KEY] = records[stream_offset:]
            stream_offset = len(records)
        yield out
