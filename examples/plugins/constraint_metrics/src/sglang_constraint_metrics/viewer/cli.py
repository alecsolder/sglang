"""Command line entry point for the optional Textual viewer."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .client import (
    detokenize,
    load_json,
    load_token_labels,
    save_token_labels,
    submit_request,
)
from .data import SchemaError, parse_response_obj, terminal_safe_text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sglang-constraint-metrics-view",
        description="Inspect one SGLang constraint-metrics response.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--response", help="saved /generate response JSON")
    source.add_argument("--request", help="request JSON to submit to --url")
    parser.add_argument(
        "--url",
        help="SGLang base URL (required live; optional offline for token labels)",
    )
    parser.add_argument(
        "--token-labels",
        help="saved token-ID-to-text JSON for a fully offline response",
    )
    parser.add_argument("--top-k", type=int, default=8, help="live candidate width")
    parser.add_argument("--save", help="save the exact live /generate response")
    parser.add_argument(
        "--save-token-labels",
        help="save decoded token labels for later offline viewing",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.request and not args.url:
        parser.error("--request requires --url")
    if args.token_labels and not args.response:
        parser.error("--token-labels requires --response")
    if args.token_labels and args.url:
        parser.error("choose --token-labels or --url for offline token labels")
    if args.response and args.save:
        parser.error("--save is only valid with --request")
    if args.save_token_labels and not args.url:
        parser.error("--save-token-labels requires --url")
    try:
        import textual  # noqa: F401

        from .app import ConstraintMetricsApp
    except ModuleNotFoundError as error:
        if (error.name or "").split(".", 1)[0] in {"rich", "textual"}:
            parser.exit(
                2,
                "viewer dependency missing; install "
                "sglang-constraint-metrics[viewer]\n",
            )
        raise

    try:
        if args.request:
            request = load_json(args.request)
            if not isinstance(request, dict):
                raise SchemaError("request JSON must contain one object")
            metrics = submit_request(
                args.url,
                request,
                top_k=args.top_k,
                timeout=args.timeout,
                save=args.save,
            )
        else:
            metrics = parse_response_obj(
                load_json(args.response),
                token_labels=(
                    load_token_labels(args.token_labels) if args.token_labels else None
                ),
                source=args.response,
                endpoint=args.url,
            )
            if args.url:
                metrics = metrics.with_token_labels(
                    detokenize(
                        args.url,
                        metrics.token_ids(),
                        timeout=min(args.timeout, 30.0),
                    )
                )
        if args.save_token_labels:
            save_token_labels(args.save_token_labels, metrics.token_labels)
    except SchemaError as error:
        parser.exit(2, f"error: {terminal_safe_text(str(error))}\n")

    ConstraintMetricsApp(metrics).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
