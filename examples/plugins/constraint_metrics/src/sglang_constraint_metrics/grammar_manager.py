"""Grammar-manager integration for request-scoped constraint metrics."""

from __future__ import annotations

from concurrent.futures import Future

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarObject,
    InvalidGrammarObject,
)
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.managers.schedule_batch import Req
from sglang_constraint_metrics.config import config_from_sampling_custom_params
from sglang_constraint_metrics.observer import (
    bind_constraint_metrics_state,
    constraint_metrics_state_for_req,
)


def _bind_request_grammar(req: Req) -> None:
    state = constraint_metrics_state_for_req(req)
    grammar = req.grammar
    if state is None or grammar is None:
        return
    if isinstance(grammar, (Future, InvalidGrammarObject)):
        return
    if not isinstance(grammar, BaseGrammarObject):
        raise TypeError("constraint metrics require an SGLang grammar object")
    if getattr(grammar, "constraint_metrics_config", None) is not None:
        raise RuntimeError("constraint metrics were already bound to this grammar")
    grammar.constraint_metrics_config = state.config


def process_constraint_metrics_req_with_grammar(
    base_process_req_with_grammar,
    grammar_manager: GrammarManager,
    req: Req,
) -> bool:
    config = config_from_sampling_custom_params(req.sampling_params.custom_params)
    bind_constraint_metrics_state(req, config=config)
    queued = base_process_req_with_grammar(grammar_manager, req)
    if not queued:
        _bind_request_grammar(req)
    return queued


def bind_ready_constraint_metrics_grammars(
    requests: list[Req],
    grammar_manager: GrammarManager,
) -> list[Req]:
    for req in requests:
        _bind_request_grammar(req)
    return requests
