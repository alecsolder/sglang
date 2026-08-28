"""Send one low-pressure Muse-Glimmer request to an enabled SGLang server."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Any

from transformers import AutoTokenizer

HERE = pathlib.Path(__file__).parent
DEFAULT_MODEL = "meta-models/Muse-Glimmer-30B"
DEFAULT_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
MAX_EXPECTED_PRESSURE_NATS = 1e-2


def _post(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _request(model: str, revision: str) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer clearly and concisely."},
            {
                "role": "user",
                "content": "Reply with one sentence introducing yourself.",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_strength="low",
    )
    return {
        "input_ids": tokenizer.encode(prompt, add_special_tokens=True),
        "sampling_params": {
            "ebnf": (HERE / "chat_turn.lark").read_text(),
            "max_new_tokens": 256,
            "temperature": 0,
            "skip_special_tokens": False,
            "constraint_metrics": {"mode": "forensic", "top_k": 8},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--save", type=pathlib.Path)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    status, body = _post(args.url, _request(args.model, args.revision), args.timeout)
    if args.save:
        args.save.write_text(json.dumps(body, indent=2) + "\n")

    metrics = (body.get("meta_info") or {}).get("constraint_metrics") or {}
    summary = metrics.get("summary") or {}
    print(json.dumps(summary, indent=2))

    if status != 200:
        print(f"request failed with HTTP {status}: {body}", file=sys.stderr)
        return 1
    if metrics.get("status") != "complete":
        print(f"capture did not complete: {metrics}", file=sys.stderr)
        return 1
    if metrics.get("warnings"):
        print(f"capture returned warnings: {metrics['warnings']}", file=sys.stderr)
        return 1
    if summary.get("argmax_masked_count") != 0:
        print("the grammar masked at least one model argmax", file=sys.stderr)
        return 1
    max_pressure = summary.get("max_constraint_pressure_nats")
    if not isinstance(max_pressure, (int, float)) or not math.isfinite(max_pressure):
        print("the response has no finite peak-pressure metric", file=sys.stderr)
        return 1
    if max_pressure >= MAX_EXPECTED_PRESSURE_NATS:
        print(
            f"peak pressure {max_pressure:.3e} exceeded the calibration limit "
            f"of {MAX_EXPECTED_PRESSURE_NATS:.3e}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
