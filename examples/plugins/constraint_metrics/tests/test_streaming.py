import sys
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
    _GenerationStreamAccumulator,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_constraint_metrics.config import (
    CONSTRAINT_METRICS_STREAM_ENABLED_KEY,
    CONSTRAINT_METRICS_STREAM_KEY,
    CONSTRAINT_METRICS_TRANSPORT_KEY,
    ConstraintMetricsConfig,
)
from sglang_constraint_metrics.integration import (
    ConstraintMetricsSchedulerOutputStreamer,
    get_constraint_metrics_output_streamer_class,
)
from sglang_constraint_metrics.observer import ConstraintMetricsState


def _record(output_index: int, *, triggered: bool) -> dict:
    return {
        "output_index": output_index,
        "token_id": 100 + output_index,
        "grammar_active": True,
        "constraint_pressure_nats": 0.1 if not triggered else 1.5,
        "triggered": triggered,
    }


class _Req:
    def __init__(
        self,
        output_ids: list[int],
        records: list[dict | None],
        *,
        finished: bool = False,
        stream_interval: int = 1,
    ):
        self.rid = "request"
        self.beam_group = None
        self.http_worker_ipc = None
        self._finished = finished
        self.finished_reason = (
            SimpleNamespace(to_json=lambda: {"type": "stop"}) if finished else None
        )
        self.finished_output = False
        self.finished_len = None
        self.stream = True
        self.sampling_params = SimpleNamespace(
            stream_interval=stream_interval,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )
        self.output_ids = output_ids
        self.output_ids_through_stop = output_ids
        self.send_token_offset = 0
        self.send_output_token_logprobs_offset = 0
        self.send_decode_id_offset = 0
        self.decoded_text = ""
        self.origin_input_ids = []
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.retraction_count = 0
        self.time_stats = None
        self.weight_version_events = []
        self.return_hidden_states = False
        self.return_routed_experts = False
        self.return_indexer_topk = False
        self.return_sampling_mask = False
        self.mm_image_tokens = 0
        self.mm_audio_tokens = 0
        self.mm_video_tokens = 0
        self.multimodal_inputs = None
        self.customized_info = {
            CONSTRAINT_METRICS_STREAM_KEY: records,
            CONSTRAINT_METRICS_STREAM_ENABLED_KEY: [True] * len(records),
        }

    def finished(self):
        return self._finished

    def init_incremental_detokenize(self):
        return self.output_ids_through_stop, 0

    def check_match_stop_str_prefix(self):
        return False


def _accumulator() -> _GenerationStreamAccumulator:
    version_kwargs = {}
    if any(
        field.name == "current_weight_version"
        for field in fields(_GenerationStreamAccumulator)
    ):
        version_kwargs["current_weight_version"] = None
    return _GenerationStreamAccumulator(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        return_sampling_mask=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        default_stream_interval=1,
        default_force_stream_interval=1,
        get_cached_tokens_details=lambda req: None,
        rust_server_mode=False,
        **version_kwargs,
    )


class _Streamer(ConstraintMetricsSchedulerOutputStreamer):
    def get_cached_tokens_details(self, req):
        return None


def _streamer(outputs) -> _Streamer:
    return _Streamer(
        send_to_detokenizer=SimpleNamespace(send_output=outputs.append),
        tree_cache=None,
        ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
        server_args=SimpleNamespace(
            stream_interval=1,
            enable_request_time_stats_logging=False,
        ),
        is_generation=True,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_hicache_storage=lambda: False,
    )


def test_triggered_stream_interval_keeps_records_aligned_with_output_ids():
    clean = _record(0, triggered=False)
    triggered = _record(2, triggered=True)
    req = _Req([10], [clean], stream_interval=2)

    first = _accumulator()
    first.accept(req=req)
    first_payload = first.to_payload(dp_rank=0, is_idle_batch=False)
    first_info = unwrap_from_pickle(first_payload.customized_info)
    assert first_payload.output_ids == [[10]]
    assert first_info[CONSTRAINT_METRICS_STREAM_KEY] == [[clean]]

    req.output_ids.extend([11, 12])
    req.customized_info[CONSTRAINT_METRICS_STREAM_KEY].extend(
        [_record(1, triggered=False), triggered]
    )
    second = _accumulator()
    second.accept(req=req)
    second_payload = second.to_payload(dp_rank=0, is_idle_batch=False)
    second_info = unwrap_from_pickle(second_payload.customized_info)
    assert second_payload.output_ids == [[11, 12]]
    assert second_info[CONSTRAINT_METRICS_STREAM_KEY] == [
        [_record(1, triggered=False), triggered]
    ]


def test_stream_preserves_null_only_for_uncaptured_positions():
    records = [_record(0, triggered=False), None, _record(2, triggered=True)]
    req = _Req([10, 11, 12], records)

    accumulator = _accumulator()
    accumulator.accept(req=req)
    payload = accumulator.to_payload(dp_rank=0, is_idle_batch=False)
    info = unwrap_from_pickle(payload.customized_info)

    assert payload.output_ids == [[10, 11, 12]]
    assert info[CONSTRAINT_METRICS_STREAM_KEY] == [records]
    assert info[CONSTRAINT_METRICS_STREAM_KEY][0][1] is None
    assert all(
        record is not None
        for index, record in enumerate(info[CONSTRAINT_METRICS_STREAM_KEY][0])
        if index != 1
    )


def test_mixed_batch_marks_request_off_stream_values():
    enabled = _Req([10], [_record(0, triggered=False)])
    request_off = _Req([20], [])
    request_off.customized_info = None

    accumulator = _accumulator()
    accumulator.accept(req=enabled)
    accumulator.accept(req=request_off)
    payload = accumulator.to_payload(dp_rank=0, is_idle_batch=False)
    info = unwrap_from_pickle(payload.customized_info)

    assert info[CONSTRAINT_METRICS_STREAM_ENABLED_KEY] == [[True], [None]]
    assert info[CONSTRAINT_METRICS_STREAM_KEY] == [
        [_record(0, triggered=False)],
        [None],
    ]


def test_terminal_response_coexists_with_stream_records_and_is_emitted_once():
    outputs = []
    streamer = _streamer(outputs)
    record = _record(0, triggered=True)
    req = _Req([10], [record], finished=True)
    req._constraint_metrics_state = ConstraintMetricsState(
        config=ConstraintMetricsConfig("triggered", 8),
        streaming=True,
    )
    terminal = SimpleNamespace(kind="terminal_constraint_metrics")

    with (
        patch(
            "sglang.srt.managers.scheduler_components.output_streamer.get_serving",
            return_value=SimpleNamespace(stream_interval=1, weight_version=None),
        ),
        patch(
            "sglang.srt.managers.scheduler_components.output_streamer.get_observability",
            return_value=SimpleNamespace(enable_request_time_stats_logging=False),
        ),
        patch(
            "sglang_constraint_metrics.integration.snapshot_constraint_metrics",
            side_effect=lambda reqs: [terminal for _ in reqs],
        ),
    ):
        streamer._stream_output_generation([req], False)
        streamer._stream_output_generation([req], False)

    assert len(outputs) == 1
    info = unwrap_from_pickle(outputs[0].customized_info)
    assert info[CONSTRAINT_METRICS_STREAM_KEY] == [[record]]
    assert info[CONSTRAINT_METRICS_TRANSPORT_KEY] == [[terminal]]
    assert req.customized_info[CONSTRAINT_METRICS_STREAM_KEY] == [None]
    assert req.customized_info[CONSTRAINT_METRICS_STREAM_ENABLED_KEY] == [None]


def test_request_off_emits_no_constraint_metrics_output():
    outputs = []
    streamer = _streamer(outputs)
    req = _Req([10], [], finished=True)
    req.customized_info = None

    with (
        patch(
            "sglang.srt.managers.scheduler_components.output_streamer.get_serving",
            return_value=SimpleNamespace(stream_interval=1, weight_version=None),
        ),
        patch(
            "sglang.srt.managers.scheduler_components.output_streamer.get_observability",
            return_value=SimpleNamespace(enable_request_time_stats_logging=False),
        ),
        patch(
            "sglang_constraint_metrics.integration.prepare_streamed_constraint_metrics"
        ) as prepare,
        patch(
            "sglang_constraint_metrics.integration.snapshot_constraint_metrics"
        ) as snapshot,
        patch(
            "sglang_constraint_metrics.integration.release_streamed_constraint_metrics"
        ) as release,
    ):
        streamer._stream_output_generation([req], False)

    assert len(outputs) == 1
    assert outputs[0].customized_info is None
    prepare.assert_not_called()
    snapshot.assert_not_called()
    release.assert_not_called()


def test_constraint_metrics_selects_output_streamer():
    scheduler = object.__new__(Scheduler)
    assert (
        get_constraint_metrics_output_streamer_class(
            SchedulerOutputStreamer,
            scheduler,
        )
        is ConstraintMetricsSchedulerOutputStreamer
    )


def test_constraint_metrics_composes_with_existing_output_streamer():
    class ExistingOutputStreamer(SchedulerOutputStreamer):
        has_additional_customized_info = True

        def build_additional_customized_info(self, reqs):
            return {"existing": [["value"] for _ in reqs]}

    scheduler = object.__new__(Scheduler)
    composed_class = get_constraint_metrics_output_streamer_class(
        ExistingOutputStreamer,
        scheduler,
    )
    streamer = object.__new__(composed_class)
    streamer._constraint_metrics_active_for_output = True
    terminal = object()

    with (
        patch(
            "sglang_constraint_metrics.integration.snapshot_constraint_metrics",
            return_value=[terminal],
        ),
        patch(
            "sglang_constraint_metrics.integration.release_streamed_constraint_metrics"
        ),
    ):
        additional_info = streamer.build_additional_customized_info([object()])

    assert issubclass(composed_class, ExistingOutputStreamer)
    assert additional_info == {
        "existing": [["value"]],
        CONSTRAINT_METRICS_TRANSPORT_KEY: [[terminal]],
    }

    streamer._constraint_metrics_active_for_output = False
    assert streamer.should_build_additional_customized_info()
    assert streamer.build_additional_customized_info([object()]) == {
        "existing": [["value"]]
    }


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
