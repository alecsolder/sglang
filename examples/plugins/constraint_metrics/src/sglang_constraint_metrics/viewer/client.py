"""Small stdlib HTTP client for the live viewer workflow."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .data import (
    ConstraintMetrics,
    SchemaError,
    parse_response_obj,
    terminal_safe_text,
)


def _base_url(url: str) -> str:
    normalized = url.rstrip("/")
    for suffix in ("/generate", "/v1/detokenize", "/detokenize"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _post_json(url: str, body: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:1000]
        raise SchemaError(f"{url} returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise SchemaError(f"could not reach {url}: {error.reason}") from error
    except OSError as error:
        raise SchemaError(f"request to {url} failed: {error}") from error
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise SchemaError(f"{url} returned invalid JSON: {error}") from error


def build_request_body(request: dict[str, Any], top_k: int) -> dict[str, Any]:
    """Return a single non-streaming forensic request without mutating input."""

    if top_k < 0 or top_k > 32:
        raise SchemaError("--top-k must be between 0 and 32")
    body = dict(request)
    if isinstance(body.get("text"), list):
        raise SchemaError("the viewer accepts one request; request.text is batched")
    input_ids = body.get("input_ids")
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        raise SchemaError(
            "the viewer accepts one request; request.input_ids is batched"
        )
    input_embeds = body.get("input_embeds")
    if isinstance(input_embeds, list) and any(
        isinstance(sequence, list)
        and any(isinstance(value, list) for value in sequence)
        for sequence in input_embeds
    ):
        raise SchemaError(
            "the viewer accepts one request; request.input_embeds is batched"
        )
    raw_sampling_params = body.get("sampling_params", {})
    if not isinstance(raw_sampling_params, dict):
        raise SchemaError("request.sampling_params must be an object")
    sampling_params = dict(raw_sampling_params)
    n = sampling_params.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n != 1:
        raise SchemaError("the viewer accepts one request; sampling_params.n must be 1")
    sampling_params["constraint_metrics"] = {
        "mode": "forensic",
        "top_k": top_k,
    }
    body["sampling_params"] = sampling_params
    body["stream"] = False
    return body


def load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text())
    except FileNotFoundError as error:
        raise SchemaError(f"JSON file not found: {source}") from error
    except json.JSONDecodeError as error:
        raise SchemaError(f"invalid JSON in {source}: {error}") from error


def load_token_labels(path: str | Path) -> dict[int, str]:
    """Load a JSON object mapping token IDs to their decoded text."""

    value = load_json(path)
    if not isinstance(value, dict):
        raise SchemaError("token labels JSON must contain one object")
    labels = {}
    for raw_token_id, label in value.items():
        try:
            token_id = int(raw_token_id)
        except (TypeError, ValueError) as error:
            raise SchemaError(
                f"token label key {raw_token_id!r} is not an integer"
            ) from error
        if token_id < 0 or str(token_id) != raw_token_id:
            raise SchemaError(f"token label key {raw_token_id!r} is not canonical")
        if not isinstance(label, str):
            raise SchemaError(f"token label {raw_token_id!r} must be a string")
        labels[token_id] = label
    return labels


def save_token_labels(path: str | Path, labels: Mapping[int, str]) -> None:
    """Save decoded token text in the offline viewer's portable format."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {str(token_id): label for token_id, label in labels.items()}, indent=2
        )
        + "\n"
    )


def detokenize(
    url: str, token_ids: Iterable[int], *, timeout: float = 30.0
) -> dict[int, str]:
    ids = sorted(set(token_ids))
    if not ids:
        return {}
    endpoint = f"{_base_url(url)}/v1/detokenize"
    response = _post_json(
        endpoint,
        {"tokens": [[token_id] for token_id in ids], "skip_special_tokens": False},
        timeout,
    )
    obj = response if isinstance(response, dict) else {}
    texts = obj.get("text")
    if not isinstance(texts, list) or len(texts) != len(ids):
        raise SchemaError(
            f"{endpoint} returned an invalid text array for {len(ids)} token ids"
        )
    if not all(isinstance(value, str) for value in texts):
        raise SchemaError(f"{endpoint} returned a non-string token label")
    return dict(zip(ids, texts, strict=True))


def submit_request(
    url: str,
    request: dict[str, Any],
    *,
    top_k: int = 8,
    timeout: float = 120.0,
    save: str | Path | None = None,
) -> ConstraintMetrics:
    body = build_request_body(request, top_k)
    base_url = _base_url(url)
    response = _post_json(f"{base_url}/generate", body, timeout)
    if save is not None:
        Path(save).write_text(json.dumps(response, indent=2) + "\n")
    metrics = parse_response_obj(response, source=str(save or url), endpoint=base_url)
    try:
        labels = detokenize(base_url, metrics.token_ids(), timeout=min(timeout, 30.0))
    except SchemaError as error:
        warnings.warn(
            "token labels unavailable; showing token IDs: "
            f"{terminal_safe_text(str(error))}",
            RuntimeWarning,
            stacklevel=2,
        )
        return metrics
    return metrics.with_token_labels(labels)
