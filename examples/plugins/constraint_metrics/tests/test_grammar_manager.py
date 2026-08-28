import sys
from types import SimpleNamespace

import pytest

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang_constraint_metrics.config import (
    CUSTOM_PARAMS_KEY,
    config_from_sampling_custom_params,
)
from sglang_constraint_metrics.grammar_manager import (
    _bind_request_grammar,
    bind_ready_constraint_metrics_grammars,
    process_constraint_metrics_req_with_grammar,
)
from sglang_constraint_metrics.observer import (
    bind_constraint_metrics_state,
    constraint_metrics_state_for_req,
)


class _Grammar(BaseGrammarObject):
    def __init__(self):
        super().__init__()


class _Request:
    pass


def _request(grammar):
    request = _Request()
    request.grammar = grammar
    request.origin_input_ids = [1]
    request.sampling_params = SimpleNamespace(
        custom_params={
            CUSTOM_PARAMS_KEY: {
                "mode": "summary",
            }
        },
        json_schema=None,
        regex=None,
        ebnf='root ::= "ok"',
        structural_tag=None,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        min_p=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        logit_bias=None,
    )
    return request


def _manager():
    return object.__new__(GrammarManager)


def _xgrammar():
    return XGrammarGrammar(
        matcher=SimpleNamespace(is_terminated=lambda: False),
        vocab_size=32,
        ctx=None,
        override_stop_tokens=None,
    )


def test_cached_grammar_is_bound_once_when_request_enters_scheduler():
    grammar = _Grammar()
    req = _request(grammar)
    manager = _manager()

    queued = process_constraint_metrics_req_with_grammar(
        lambda _manager, _req: False,
        manager,
        req,
    )

    assert not queued
    assert grammar.constraint_metrics_config.mode == "summary"
    assert (
        constraint_metrics_state_for_req(req).config
        is grammar.constraint_metrics_config
    )


def test_compiled_grammar_is_bound_when_future_becomes_ready():
    grammar = _Grammar()
    req = _request(grammar)
    manager = _manager()
    bind_constraint_metrics_state(
        req,
        config=config_from_sampling_custom_params(req.sampling_params.custom_params),
    )

    requests = bind_ready_constraint_metrics_grammars([req], manager)

    assert requests == [req]
    assert grammar.constraint_metrics_config.mode == "summary"


def test_xgrammar_request_is_bound_when_it_enters_scheduler():
    grammar = _xgrammar()
    req = _request(grammar)

    queued = process_constraint_metrics_req_with_grammar(
        lambda _manager, _req: False,
        _manager(),
        req,
    )

    assert not queued
    assert grammar.constraint_metrics_config.mode == "summary"


def test_metrics_request_fails_loudly_with_wrong_backend_object():
    req = _request(SimpleNamespace())
    bind_constraint_metrics_state(
        req,
        config=config_from_sampling_custom_params(req.sampling_params.custom_params),
    )

    with pytest.raises(TypeError, match="SGLang grammar object"):
        _bind_request_grammar(req)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
