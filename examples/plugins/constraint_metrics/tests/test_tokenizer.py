import asyncio
import os
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang_constraint_metrics import layout
from sglang_constraint_metrics.config import (
    CONSTRAINT_METRICS_STREAM_ENABLED_KEY,
    CONSTRAINT_METRICS_STREAM_KEY,
    CONSTRAINT_METRICS_TRANSPORT_KEY,
    CUSTOM_PARAMS_KEY,
    MAX_TOP_K_ENV,
    ConstraintMetricsConfig,
    deployment_max_top_k,
)
from sglang_constraint_metrics.integration import (
    _lower_request,
    _validate_constraint_metrics_request,
    coalesce_constraint_metrics_streaming_chunks,
    generate_constraint_metrics_request,
    update_constraint_metrics_request_meta_info,
    wait_one_constraint_metrics_response,
)
from sglang_constraint_metrics.observer import (
    ConstraintMetricsResponse,
    ConstraintMetricsStreamRecord,
)
from sglang_constraint_metrics.records import ConstraintMetricsPackedRows


def _manager():
    manager = object.__new__(TokenizerManager)
    manager.server_args = SimpleNamespace()
    manager.incremental_streaming_output = False
    return manager


def _update_request_meta_info(
    manager,
    customized_info,
    *,
    index=0,
    finish_reason=None,
):
    meta_info = {}
    state = SimpleNamespace(customized_info_accumulated={})
    update_constraint_metrics_request_meta_info(
        TokenizerManager.update_request_meta_info,
        manager,
        meta_info,
        state,
        customized_info,
        index,
        finish_reason,
    )
    return meta_info, state


def _request(
    mode="forensic",
    *,
    sampling_params=None,
    top_k=8,
    stream=False,
    stream_all_token_metrics=False,
):
    sampling_params = dict(sampling_params or {})
    sampling_params.setdefault("regex", r"[a-z]+")
    config = {"mode": mode}
    if mode == "forensic":
        config["top_k"] = top_k
    if stream_all_token_metrics:
        config["stream_all_token_metrics"] = True
    sampling_params["constraint_metrics"] = config
    request = GenerateReqInput(
        text="x",
        sampling_params=sampling_params,
        stream=stream,
    )
    _lower_request(request)
    request.normalize_batch_and_arguments()
    return request


def _metrics_payload(request):
    return request.sampling_params["custom_params"][CUSTOM_PARAMS_KEY]


def _collect_wait_outputs(manager, outputs):
    async def base_wait_one_response(self, obj, request=None):
        for output in outputs:
            yield output

    async def collect():
        request = SimpleNamespace(
            mm_stats=None,
            disagg_prefill_media_stats=None,
        )
        return [
            output
            async for output in wait_one_constraint_metrics_response(
                TokenizerManager._wait_one_response,
                manager,
                request,
            )
        ]

    with patch.object(
        TokenizerManager,
        "_wait_one_response",
        base_wait_one_response,
    ):
        return asyncio.run(collect())


def test_request_lowers_msgpack_safe_metrics_config():
    request = _request(
        sampling_params={"custom_params": {"caller": "value"}},
        top_k=20,
    )

    assert request.sampling_params["custom_params"]["caller"] == "value"
    assert _metrics_payload(request) == {
        "mode": "forensic",
        "top_k": 20,
    }
    transported = SamplingParams(**request.sampling_params)
    assert transported.custom_params == request.sampling_params["custom_params"]


def test_generate_request_lowers_base_engine_request():
    manager = _manager()
    request = GenerateReqInput(
        text="x",
        sampling_params={
            "regex": r"[a-z]+",
            "constraint_metrics": {"mode": "summary"},
        },
    )

    async def base_generate_request(self, obj, request=None):
        yield obj.sampling_params

    async def collect():
        return [
            response
            async for response in generate_constraint_metrics_request(
                base_generate_request,
                manager,
                request,
                request=None,
            )
        ]

    outputs = asyncio.run(collect())

    assert outputs[0]["custom_params"][CUSTOM_PARAMS_KEY] == {"mode": "summary"}
    assert "constraint_metrics" not in outputs[0]


def test_generate_request_preserves_inactive_batch_sampling_params():
    manager = _manager()
    sampling_params = [{"temperature": 0.5}, {"max_new_tokens": 8}]
    request = GenerateReqInput(text=["a", "b"], sampling_params=sampling_params)

    async def base_generate_request(self, obj, request=None):
        yield obj.sampling_params is sampling_params

    async def collect():
        return [
            response
            async for response in generate_constraint_metrics_request(
                base_generate_request,
                manager,
                request,
                request=None,
            )
        ]

    assert asyncio.run(collect()) == [True]


def test_request_rejects_reserved_transport_key():
    request = GenerateReqInput(
        text="x",
        sampling_params={
            "custom_params": {CUSTOM_PARAMS_KEY: {"mode": "forensic"}},
        },
    )

    with pytest.raises(ValueError, match="reserves"):
        _lower_request(request)


def test_generate_request_rejects_reserved_transport_key_at_ingress():
    manager = _manager()
    request = GenerateReqInput(
        text="x",
        sampling_params={
            "custom_params": {CUSTOM_PARAMS_KEY: {"mode": "summary"}},
        },
    )

    async def base_generate_request(_obj, _request=None):
        if False:
            yield None

    async def collect():
        return [
            output
            async for output in generate_constraint_metrics_request(
                base_generate_request,
                manager,
                request,
            )
        ]

    with pytest.raises(ValueError, match="reserves"):
        asyncio.run(collect())


def test_base_engine_request_requires_grammar():
    request = GenerateReqInput(
        text="x",
        sampling_params={"constraint_metrics": {"mode": "summary"}},
    )
    _lower_request(request)
    request.normalize_batch_and_arguments()

    with pytest.raises(ValueError, match="requires a grammar"):
        _validate_constraint_metrics_request(request)


@pytest.mark.parametrize(
    ("grammar_key", "grammar"),
    [
        ("ebnf", ""),
        ("regex", ""),
        ("json_schema", {}),
        ("structural_tag", ""),
    ],
)
def test_base_engine_request_rejects_empty_grammar(grammar_key, grammar):
    request = GenerateReqInput(
        text="x",
        sampling_params={
            grammar_key: grammar,
            "constraint_metrics": {"mode": "summary"},
        },
    )
    _lower_request(request)
    request.normalize_batch_and_arguments()

    with pytest.raises(ValueError, match="requires a grammar"):
        _validate_constraint_metrics_request(request)


def test_tokenizer_enforces_table_top_k_policy():
    with (
        patch.dict(os.environ, {MAX_TOP_K_ENV: "8"}),
        pytest.raises(ValueError, match=MAX_TOP_K_ENV),
    ):
        deployment_max_top_k.cache_clear()
        _validate_constraint_metrics_request(
            _request(mode="forensic", top_k=20),
        )
    deployment_max_top_k.cache_clear()


@pytest.mark.parametrize("mode", ["triggered", "forensic"])
def test_tokenizer_accepts_dense_metric_streaming(mode):
    request = _request(
        mode=mode,
        stream=True,
        stream_all_token_metrics=True,
    )

    _validate_constraint_metrics_request(request)
    assert _metrics_payload(request)["stream_all_token_metrics"] is True


@pytest.mark.parametrize("mode", ["triggered", "forensic"])
def test_dense_metric_streaming_requires_streaming_response(mode):
    request = _request(mode=mode, stream_all_token_metrics=True)

    with pytest.raises(ValueError, match="requires stream=true"):
        _validate_constraint_metrics_request(request)


def test_tokenizer_accepts_mixed_native_batch_within_policy():
    request = GenerateReqInput(
        text=["summary", "forensic"],
        sampling_params=[
            {
                "regex": r"[a-z]+",
                "constraint_metrics": {"mode": "summary"},
            },
            {
                "max_new_tokens": 64,
                "regex": r"[a-z]+",
                "constraint_metrics": {"mode": "forensic", "top_k": 8},
            },
        ],
    )
    _lower_request(request)
    request.normalize_batch_and_arguments()

    _validate_constraint_metrics_request(request)


def test_tokenizer_rejects_custom_logit_processor_ordering():
    request = _request()
    request.custom_logit_processor = "processor"

    with pytest.raises(ValueError, match="custom_logit_processor ordering"):
        _validate_constraint_metrics_request(request)


def test_tokenizer_rejects_processor_in_shared_sampling_params_batch():
    request = GenerateReqInput(
        text=["first", "second"],
        sampling_params={
            "regex": r"[a-z]+",
            "constraint_metrics": {"mode": "summary"},
        },
        custom_logit_processor=[None, "processor"],
    )
    _lower_request(request)

    with pytest.raises(ValueError, match="custom_logit_processor ordering"):
        _validate_constraint_metrics_request(request)


def test_tokenizer_rejects_input_embeds_retraction_semantics():
    request = GenerateReqInput(
        input_embeds=[[0.1, 0.2]],
        sampling_params={
            "regex": r"[a-z]+",
            "constraint_metrics": {"mode": "summary"},
        },
    )
    _lower_request(request)
    request.normalize_batch_and_arguments()

    with pytest.raises(ValueError, match="input_embeds retraction"):
        _validate_constraint_metrics_request(request)


def test_request_validation_runs_before_base_generation():
    manager = _manager()
    request = GenerateReqInput(
        text="x",
        sampling_params={
            "regex": r"[a-z]+",
            "constraint_metrics": {"mode": "forensic", "top_k": 20},
        },
    )
    base_called = False

    async def base_generate_request(_manager, _obj, _request=None):
        nonlocal base_called
        base_called = True
        if False:
            yield None

    async def collect():
        return [
            output
            async for output in generate_constraint_metrics_request(
                base_generate_request,
                manager,
                request,
            )
        ]

    with (
        patch.dict(os.environ, {MAX_TOP_K_ENV: "8"}),
        pytest.raises(ValueError, match=MAX_TOP_K_ENV),
    ):
        deployment_max_top_k.cache_clear()
        asyncio.run(collect())
    deployment_max_top_k.cache_clear()

    assert not base_called


def test_terminal_constraint_metrics_are_extracted_from_customized_info():
    manager = _manager()
    response = ConstraintMetricsResponse(
        config=ConstraintMetricsConfig("summary", 0),
        record_slices=[],
        validation_token_ids=[],
        input_token_count=1,
        output_token_count=0,
        summary={"output_token_count": 0},
        captured_prefix_length=0,
    )

    with patch.object(
        ConstraintMetricsResponse,
        "materialize_constraint_metrics",
        return_value={"status": "complete"},
    ):
        meta_info, state = _update_request_meta_info(
            manager,
            {CONSTRAINT_METRICS_TRANSPORT_KEY: [[response]]},
            finish_reason={"type": "length"},
        )

    assert meta_info == {"constraint_metrics": {"status": "complete"}}
    assert state.customized_info_accumulated == {}


def test_constraint_metrics_transport_rejects_intermediate_output():
    manager = _manager()
    response = object.__new__(ConstraintMetricsResponse)

    with pytest.raises(ValueError, match="before request completion"):
        _update_request_meta_info(
            manager,
            {CONSTRAINT_METRICS_TRANSPORT_KEY: [[response]]},
        )


def test_unrelated_customized_info_uses_base_accumulation():
    manager = _manager()

    meta_info, state = _update_request_meta_info(
        manager,
        {"custom_metric": [[1, 2]]},
    )

    assert meta_info == {"custom_metric": [1, 2]}
    assert state.customized_info_accumulated == {"custom_metric": [1, 2]}


def test_mixed_constraint_and_unrelated_info_delegates_unrelated_keys():
    manager = _manager()
    response = ConstraintMetricsResponse(
        config=ConstraintMetricsConfig("summary", 0),
        record_slices=[],
        validation_token_ids=[],
        input_token_count=1,
        output_token_count=0,
        summary={"output_token_count": 0},
        captured_prefix_length=0,
    )
    base_update = Mock()
    meta_info = {}
    state = SimpleNamespace(customized_info_accumulated={})

    with patch.object(
        ConstraintMetricsResponse,
        "materialize_constraint_metrics",
        return_value={"status": "complete"},
    ):
        update_constraint_metrics_request_meta_info(
            base_update,
            manager,
            meta_info,
            state,
            {
                "other_plugin": [["value"]],
                CONSTRAINT_METRICS_TRANSPORT_KEY: [[response]],
            },
            0,
            {"type": "length"},
        )

    base_update.assert_called_once_with(
        manager,
        meta_info,
        state,
        {"other_plugin": [["value"]]},
        0,
        {"type": "length"},
    )
    assert meta_info["constraint_metrics"] == {"status": "complete"}


def test_incremental_stream_records_materialize_after_transport():
    manager = _manager()
    manager.incremental_streaming_output = True
    ints = [0] * layout.KERNEL_ROW_INT_COUNT
    ints[layout.KERNEL_INT_TOKEN_ID] = 42
    ints[layout.KERNEL_INT_EFFECTIVE_LEGAL_TOKEN_COUNT] = 1
    ints[layout.KERNEL_INT_EMITTED_ALLOWED] = 1
    rows = ConstraintMetricsPackedRows(
        floats=[[0.0] * layout.KERNEL_ROW_FLOAT_COUNT],
        ints=[ints],
        grammar_active=[True],
        output_indices=[0],
    )
    record = ConstraintMetricsStreamRecord(
        rows=rows,
        row_index=0,
        triggered=False,
    )
    record = pickle.loads(pickle.dumps(record))

    meta_info, state = _update_request_meta_info(
        manager,
        {
            CONSTRAINT_METRICS_STREAM_ENABLED_KEY: [[True]],
            CONSTRAINT_METRICS_STREAM_KEY: [[record]],
        },
    )

    materialized = meta_info[CONSTRAINT_METRICS_STREAM_KEY][0]
    assert materialized["output_index"] == 0
    assert materialized["token_id"] == 42
    assert materialized["triggered"] is False
    assert state.customized_info_accumulated == {}


@pytest.mark.parametrize("incremental", [False, True])
def test_mixed_batch_suppresses_stream_field_for_request_without_metrics(incremental):
    manager = _manager()
    manager.incremental_streaming_output = incremental
    record = {"output_index": 0, "triggered": False}
    customized_info = {
        CONSTRAINT_METRICS_STREAM_ENABLED_KEY: [[True], [None]],
        CONSTRAINT_METRICS_STREAM_KEY: [[record], [None]],
    }

    enabled_meta_info, enabled_state = _update_request_meta_info(
        manager, customized_info
    )
    off_meta_info, off_state = _update_request_meta_info(
        manager, customized_info, index=1
    )

    assert enabled_meta_info == {CONSTRAINT_METRICS_STREAM_KEY: [record]}
    assert enabled_state.customized_info_accumulated == (
        {} if incremental else {CONSTRAINT_METRICS_STREAM_KEY: [record]}
    )
    assert off_meta_info == {}
    assert off_state.customized_info_accumulated == {}


def test_incremental_stream_record_coalescing_preserves_every_chunk():
    manager = _manager()
    chunks = [
        {
            "text": "a",
            "output_ids": [10],
            "meta_info": {CONSTRAINT_METRICS_STREAM_KEY: [{"output_index": 0}]},
        },
        {
            "text": "b",
            "output_ids": [11],
            "meta_info": {CONSTRAINT_METRICS_STREAM_KEY: [{"output_index": 1}]},
        },
    ]

    output = coalesce_constraint_metrics_streaming_chunks(
        TokenizerManager._coalesce_streaming_chunks,
        manager,
        chunks,
        "request",
    )

    assert output["text"] == "ab"
    assert output["output_ids"] == [10, 11]
    assert output["meta_info"][CONSTRAINT_METRICS_STREAM_KEY] == [
        {"output_index": 0},
        {"output_index": 1},
    ]


def test_cumulative_output_emits_constraint_metric_deltas_per_yield():
    manager = _manager()
    manager.server_args.model_path = "/model"
    rows = [{"output_index": index} for index in range(4)]
    outputs = [
        {
            "output_ids": [10],
            "meta_info": {
                "finish_reason": None,
                CONSTRAINT_METRICS_STREAM_KEY: rows[:1],
            },
        },
        {
            "output_ids": [10],
            "meta_info": {
                "finish_reason": None,
                CONSTRAINT_METRICS_STREAM_KEY: rows[:1],
            },
        },
        {
            "output_ids": [10, 11, 12],
            "meta_info": {
                "finish_reason": None,
                CONSTRAINT_METRICS_STREAM_KEY: rows[:3],
            },
        },
        {
            "output_ids": [10, 11, 12, 13],
            "meta_info": {
                "finish_reason": {"type": "length"},
                CONSTRAINT_METRICS_STREAM_KEY: rows,
                "constraint_metrics": {"status": "complete"},
            },
        },
    ]

    result = _collect_wait_outputs(manager, outputs)

    assert [
        output["meta_info"][CONSTRAINT_METRICS_STREAM_KEY] for output in result
    ] == [rows[:1], [], rows[1:3], rows[3:]]
    assert [output["output_ids"] for output in result] == [
        [10],
        [10],
        [10, 11, 12],
        [10, 11, 12, 13],
    ]
    assert result[-1]["meta_info"]["constraint_metrics"] == {"status": "complete"}


def test_cumulative_output_rejects_shrinking_constraint_metric_history():
    manager = _manager()
    manager.server_args.model_path = "/model"
    outputs = [
        {
            "meta_info": {
                "finish_reason": None,
                CONSTRAINT_METRICS_STREAM_KEY: [
                    {"output_index": 0},
                    {"output_index": 1},
                ],
            }
        },
        {
            "meta_info": {
                "finish_reason": None,
                CONSTRAINT_METRICS_STREAM_KEY: [{"output_index": 0}],
            }
        },
    ]

    with pytest.raises(ValueError, match="stream history shrank"):
        _collect_wait_outputs(manager, outputs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
