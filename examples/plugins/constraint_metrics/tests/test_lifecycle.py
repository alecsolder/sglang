import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
from sglang_constraint_metrics import observer
from sglang_constraint_metrics.config import ConstraintMetricsConfig
from sglang_constraint_metrics.integration import (
    _ConstraintMetricsAbortAwareSender,
)
from sglang_constraint_metrics.observer import (
    bind_constraint_metrics_state,
)


def test_direct_abort_releases_request_state_and_payload():
    payload = Mock()
    record_slice = Mock(output_index_start=0, count=1, payload=payload)
    req = object.__new__(Req)
    req.sampling_params = SimpleNamespace(
        json_schema={},
        regex=None,
        ebnf=None,
        structural_tag=None,
    )
    req.stream = True
    req.customized_info = None
    state = bind_constraint_metrics_state(
        req,
        config=ConstraintMetricsConfig("forensic", 0, stream_all_token_metrics=True),
    )
    assert state is not None
    state.record_slices.append(record_slice)
    state.captured_prefix_length = record_slice.count
    payload.acquire_committed_slice()
    stream_records = [object()]
    stream_enabled = [True]
    req.customized_info = {
        observer.CONSTRAINT_METRICS_STREAM_KEY: stream_records,
        observer.CONSTRAINT_METRICS_STREAM_ENABLED_KEY: stream_enabled,
    }
    delegate = Mock(spec=SenderWrapper)
    delegate.socket = None
    sender = _ConstraintMetricsAbortAwareSender(delegate)
    abort = AbortReq(rid="request")
    sender.send_output(abort, req)

    assert observer.constraint_metrics_state_for_req(req) is None
    assert state.record_slices == []
    assert stream_records == []
    assert stream_enabled == []
    payload.release_committed_slice.assert_called_once_with()
    normal_output = object()
    sender.send_output(normal_output, req)

    assert delegate.send_output.call_args_list == [
        call(abort, req),
        call(normal_output, req),
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
