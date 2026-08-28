"""Request configuration for constraint metrics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from typing import Any, Optional

MODE_SUMMARY = "summary"
MODE_FORENSIC = "forensic"
MODE_TRIGGERED = "triggered"
MODES = (MODE_SUMMARY, MODE_TRIGGERED, MODE_FORENSIC)

DEFAULT_TOP_K = 8
MAX_TOP_K = 32
TOP_K_COMPUTE_BUCKETS = (0, 8, 16, 32)
MAX_TOP_K_ENV = "SGLANG_CONSTRAINT_METRICS_MAX_TOP_K"
SAMPLING_PARAMS_KEY = "constraint_metrics"

CONSTRAINT_METRICS_STREAM_KEY = "constraint_metrics_tokens"
CONSTRAINT_METRICS_STREAM_ENABLED_KEY = "_constraint_metrics_stream_enabled"
CONSTRAINT_METRICS_TRANSPORT_KEY = "__constraint_metrics_response"

# Reserved SamplingParams.custom_params key. Values below it must remain
# msgpack-safe until ScheduleBatch replaces transport objects with local Req refs.
CUSTOM_PARAMS_KEY = "__constraint_metrics"
_CONFIG_FIELDS = {"mode", "top_k", "stream_all_token_metrics"}


@cache
def deployment_max_top_k() -> int:
    """Return the process-wide forensic top-k limit for this plugin."""
    raw_value = os.getenv(MAX_TOP_K_ENV)
    if raw_value is None:
        return DEFAULT_TOP_K
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{MAX_TOP_K_ENV} must be an integer, got {raw_value!r}"
        ) from error
    if not 0 <= value <= MAX_TOP_K:
        raise ValueError(f"{MAX_TOP_K_ENV} must be in [0, {MAX_TOP_K}], got {value}")
    return value


def normalize_mode(
    mode: Optional[str], *, path: str = f"sampling_params.{SAMPLING_PARAMS_KEY}"
) -> str:
    if mode is None:
        raise ValueError(f"{path}.mode is required")
    if not isinstance(mode, str):
        raise ValueError(f"{path}.mode must be a string, got {mode!r}")
    normalized = mode.strip().lower()
    if normalized not in MODES:
        raise ValueError(f"{path}.mode must be one of {list(MODES)}, got {mode!r}")
    return normalized


def normalize_stream_all_token_metrics(
    value: Any, *, path: str = f"sampling_params.{SAMPLING_PARAMS_KEY}"
) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(
            f"{path}.stream_all_token_metrics must be a boolean, got {value!r}"
        )
    return value


def normalize_top_k(
    top_k: Optional[int], *, path: str = f"sampling_params.{SAMPLING_PARAMS_KEY}"
) -> int:
    if top_k is None:
        return DEFAULT_TOP_K
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f"{path}.top_k must be an integer, got {top_k!r}")
    if top_k < 0:
        raise ValueError(f"{path}.top_k must be non-negative, got {top_k}")
    if top_k > MAX_TOP_K:
        raise ValueError(f"{path}.top_k must not exceed {MAX_TOP_K}, got {top_k}")
    return top_k


@dataclass(frozen=True, slots=True)
class ConstraintMetricsConfig:
    mode: str
    top_k: int
    stream_all_token_metrics: bool = False

    @property
    def emits_tables(self) -> bool:
        return self.mode == MODE_FORENSIC

    @property
    def kernel_top_k(self) -> int:
        return self.top_k if self.emits_tables else 0

    def to_custom_params(self) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": self.mode}
        if self.mode == MODE_FORENSIC:
            result["top_k"] = self.top_k
        if self.stream_all_token_metrics:
            result["stream_all_token_metrics"] = True
        return result

    @classmethod
    def _from_mapping(cls, value: Any, *, path: str) -> ConstraintMetricsConfig:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        unknown_fields = sorted(value.keys() - _CONFIG_FIELDS)
        if unknown_fields:
            raise ValueError(f"Unknown {path} field(s): {unknown_fields}")
        mode = normalize_mode(value.get("mode"), path=path)
        if mode != MODE_FORENSIC and "top_k" in value:
            raise ValueError(f"{path}.top_k is only valid in {MODE_FORENSIC!r} mode")
        stream_all_token_metrics = normalize_stream_all_token_metrics(
            value.get("stream_all_token_metrics"), path=path
        )
        if stream_all_token_metrics and mode not in (MODE_TRIGGERED, MODE_FORENSIC):
            raise ValueError(
                f"{path}.stream_all_token_metrics "
                f"requires {MODE_TRIGGERED!r} or {MODE_FORENSIC!r} mode"
            )
        return cls(
            mode=mode,
            top_k=(
                normalize_top_k(value.get("top_k"), path=path)
                if mode == MODE_FORENSIC
                else 0
            ),
            stream_all_token_metrics=stream_all_token_metrics,
        )

    @classmethod
    def from_request(cls, value: Any) -> ConstraintMetricsConfig:
        return cls._from_mapping(
            value,
            path=f"sampling_params.{SAMPLING_PARAMS_KEY}",
        )

    @classmethod
    def from_custom_params(cls, value: Any) -> ConstraintMetricsConfig:
        return cls._from_mapping(
            value,
            path=f"sampling_params.custom_params.{CUSTOM_PARAMS_KEY}",
        )


def config_from_sampling_custom_params(value: Any) -> Optional[ConstraintMetricsConfig]:
    if not isinstance(value, dict):
        return None
    if CUSTOM_PARAMS_KEY not in value:
        return None
    return ConstraintMetricsConfig.from_custom_params(value[CUSTOM_PARAMS_KEY])


def lower_constraint_metrics_sampling_params(value: dict[str, Any]) -> dict[str, Any]:
    """Lower the public request block into the private scheduler transport."""
    custom_params = value.get("custom_params")
    if isinstance(custom_params, dict) and CUSTOM_PARAMS_KEY in custom_params:
        raise ValueError(f"sampling_params.custom_params reserves {CUSTOM_PARAMS_KEY}")

    if SAMPLING_PARAMS_KEY not in value:
        return value

    result = dict(value)
    request_config = result.pop(SAMPLING_PARAMS_KEY)
    config = ConstraintMetricsConfig.from_request(request_config)
    if custom_params is not None and not isinstance(custom_params, dict):
        raise ValueError("sampling_params.custom_params must be an object")

    lowered_custom_params = dict(custom_params or {})
    lowered_custom_params[CUSTOM_PARAMS_KEY] = config.to_custom_params()
    result["custom_params"] = lowered_custom_params
    return result
