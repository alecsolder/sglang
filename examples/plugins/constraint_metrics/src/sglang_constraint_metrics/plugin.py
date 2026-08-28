"""SGLang plugin entry point."""

from __future__ import annotations

import logging
import os
from importlib import import_module
from typing import Any

PLUGIN_NAME = "constraint_metrics"

_METHOD_HOOKS = (
    (
        "sglang.srt.server_args.ServerArgs.check_server_args",
        "validate_selected_constraint_metrics",
        "AFTER",
    ),
    (
        "sglang.srt.model_executor.model_runner.ModelRunner.__init__",
        "init_constraint_metrics_model_runner",
        "AFTER",
    ),
    (
        "sglang.srt.model_executor.model_runner.ModelRunner.alloc_memory_pool",
        "alloc_constraint_metrics_memory_pool",
        "AROUND",
    ),
    (
        "sglang.srt.managers.scheduler.Scheduler.init_ipc_channels",
        "init_constraint_metrics_ipc_channels",
        "AFTER",
    ),
    (
        "sglang.srt.managers.scheduler.Scheduler.get_output_streamer_class",
        "get_constraint_metrics_output_streamer_class",
        "AFTER",
    ),
    (
        "sglang.srt.constrained.grammar_manager.GrammarManager.process_req_with_grammar",
        "process_constraint_metrics_req_with_grammar",
        "AROUND",
    ),
    (
        "sglang.srt.constrained.grammar_manager.GrammarManager.get_ready_grammar_requests",
        "bind_ready_constraint_metrics_grammars",
        "AFTER",
    ),
    (
        "sglang.srt.managers.tokenizer_manager.TokenizerManager.generate_request",
        "generate_constraint_metrics_request",
        "AROUND",
    ),
    (
        "sglang.srt.managers.tokenizer_manager.TokenizerManager.update_request_meta_info",
        "update_constraint_metrics_request_meta_info",
        "AROUND",
    ),
    (
        "sglang.srt.managers.tokenizer_manager.TokenizerManager._coalesce_streaming_chunks",
        "coalesce_constraint_metrics_streaming_chunks",
        "AROUND",
    ),
    (
        "sglang.srt.managers.tokenizer_manager.TokenizerManager._wait_one_response",
        "wait_one_constraint_metrics_response",
        "AROUND",
    ),
)

logger = logging.getLogger(__name__)
_registered = False


def verify_runtime_hooks() -> None:
    """Fail startup when explicit activation was only partially applied."""
    from sglang.srt.plugins.hook_registry import HookRegistry

    mismatches = [
        target for target, _, _ in _METHOD_HOOKS if target not in HookRegistry._patched
    ]
    if mismatches:
        raise RuntimeError(
            "constraint_metrics plugin hooks were not fully applied: "
            + ", ".join(mismatches)
        )


def validate_selected_constraint_metrics(
    _result: None,
    server_args: Any,
) -> None:
    """Fail startup after SGLang has resolved and checked its server args."""
    verify_runtime_hooks()
    from sglang_constraint_metrics.integration import (
        validate_constraint_metrics_server_args,
    )

    validate_constraint_metrics_server_args(None, server_args)


def _is_explicitly_selected() -> bool:
    selected = os.getenv("SGLANG_PLUGINS", "")
    return PLUGIN_NAME in {name.strip() for name in selected.split(",") if name.strip()}


def register() -> None:
    """Register hooks only when this plugin is explicitly selected."""
    global _registered
    if _registered:
        return
    if not _is_explicitly_selected():
        logger.info(
            "The constraint-metrics package is installed but inactive; add "
            "constraint_metrics to SGLANG_PLUGINS to load it."
        )
        return

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    validation_target, _, validation_hook_type = _METHOD_HOOKS[0]
    HookRegistry.register(
        validation_target,
        validate_selected_constraint_metrics,
        HookType[validation_hook_type],
    )

    hook_modules = {
        "process_constraint_metrics_req_with_grammar": import_module(
            "sglang_constraint_metrics.grammar_manager"
        ),
        "bind_ready_constraint_metrics_grammars": import_module(
            "sglang_constraint_metrics.grammar_manager"
        ),
    }
    integration = import_module("sglang_constraint_metrics.integration")
    for target, hook_name, hook_type in _METHOD_HOOKS[1:]:
        hook_module = hook_modules.get(hook_name, integration)
        HookRegistry.register(
            target,
            getattr(hook_module, hook_name),
            HookType[hook_type],
        )
    _registered = True
