"""Replay request files and save complete responses for offline inspection."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from sglang_constraint_metrics.viewer.client import (
    build_request_body,
    detokenize,
    save_token_labels,
)
from sglang_constraint_metrics.viewer.data import (
    SchemaError,
    parse_response_obj,
    terminal_safe_text,
)

HERE = pathlib.Path(__file__).parent
DEFAULT_MODEL = "meta-models/Muse-Glimmer-30B"
DEFAULT_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _model_info(
    opener: urllib.request.OpenerDirector, url: str, *, timeout: float = 5
) -> dict[str, Any]:
    endpoint = f"{url.rstrip('/')}/model_info"
    try:
        with opener.open(endpoint, timeout=timeout) as response:
            raw = response.read()
    except OSError as error:
        raise SchemaError(f"could not reach {endpoint}: {error}") from error
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SchemaError(f"{endpoint} returned invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise SchemaError(f"{endpoint} must return one object")
    return value


def _wait_until_ready(
    opener: urllib.request.OpenerDirector,
    url: str,
    server: subprocess.Popen[bytes],
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"SGLang exited with status {server.returncode}")
        try:
            return _model_info(opener, url)
        except SchemaError:
            pass
        time.sleep(2)
    raise TimeoutError(f"SGLang did not become ready within {timeout:g} seconds")


def _check_model(info: dict[str, Any], expected: str) -> None:
    actual = info.get("model_path")
    if actual != expected:
        raise SchemaError(
            f"server model is {actual!r}, expected {expected!r}; pass the matching "
            "--model or use --skip-model-check"
        )


def _ensure_port_available(host: str, port: int) -> None:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    except OSError as error:
        raise SchemaError(f"cannot start SGLang on {host}:{port}: {error}") from error


def _server_url(host: str, port: int) -> str:
    url_host = f"[{host}]" if ":" in host else host
    return f"http://{url_host}:{port}"


def _start_server(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    plugins = [value for value in env.get("SGLANG_PLUGINS", "").split(",") if value]
    if "constraint_metrics" not in plugins:
        plugins.append("constraint_metrics")
    env["SGLANG_PLUGINS"] = ",".join(plugins)
    env.setdefault("SGLANG_CONSTRAINT_METRICS_MAX_TOP_K", "8")
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--revision",
        args.revision,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--grammar-backend",
        "llguidance",
        "--max-running-requests",
        "8",
        "--mem-fraction-static",
        "0.75",
        "--disable-cuda-graph",
    ]
    return subprocess.Popen(command, env=env, start_new_session=True)


def _stop_server(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def _read_request(path: pathlib.Path) -> dict[str, Any]:
    try:
        body = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise SchemaError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(body, dict):
        raise SchemaError(f"{path} must contain one request object")
    return body


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def _post_request(
    opener: urllib.request.OpenerDirector,
    url: str,
    payload: bytes,
    timeout: float,
) -> tuple[int, bytes]:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _validate_roots(args: argparse.Namespace) -> None:
    roots = [
        args.requests.resolve(),
        args.submitted_requests.resolve(),
        args.responses.resolve(),
        args.token_labels.resolve(),
    ]
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if (
                left == right
                or left.is_relative_to(right)
                or right.is_relative_to(left)
            ):
                raise SchemaError(
                    f"input and output roots must be separate: {left}, {right}"
                )


def _capture(args: argparse.Namespace, url: str) -> int:
    _validate_roots(args)
    request_paths = sorted(args.requests.rglob("*.json"))
    if not request_paths:
        raise FileNotFoundError(f"no JSON requests found under {args.requests}")
    opener = _opener()
    failed = 0
    for request_path in request_paths:
        relative = request_path.relative_to(args.requests)
        submitted_path = args.submitted_requests / relative
        response_path = args.responses / relative
        label_path = args.token_labels / relative
        for stale_path in (submitted_path, response_path, label_path):
            stale_path.unlink(missing_ok=True)
        try:
            body = build_request_body(_read_request(request_path), args.top_k)
            payload = _json_bytes(body)
            submitted_path.parent.mkdir(parents=True, exist_ok=True)
            submitted_path.write_bytes(payload)
            status, raw_response = _post_request(
                opener, url, payload, args.request_timeout
            )
            response_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.write_bytes(raw_response)
            try:
                response_body: Any = json.loads(raw_response)
            except json.JSONDecodeError as error:
                raise SchemaError(
                    f"{response_path} contains invalid JSON: {error}"
                ) from error
            if status != 200:
                raise RuntimeError(f"server returned HTTP {status}")
            metrics = parse_response_obj(
                response_body, source=str(response_path), endpoint=url
            )
            labels = detokenize(url, metrics.token_ids(), timeout=30)
            save_token_labels(label_path, labels)
            summary = metrics.summary
            print(
                f"{relative}: tokens={summary.output_token_count} "
                f"masked={summary.argmax_masked_count} "
                f"max_pressure={summary.max_constraint_pressure_nats:.6g}"
            )
        except (OSError, RuntimeError, SchemaError) as error:
            failed += 1
            print(
                f"{relative}: error: {terminal_safe_text(str(error))}",
                file=sys.stderr,
            )
    return failed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="reuse an existing SGLang server")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--requests", type=pathlib.Path, default=HERE / "requests")
    parser.add_argument(
        "--submitted-requests",
        type=pathlib.Path,
        default=HERE / "submitted_requests",
        help="save the exact payloads sent after the forensic overlay",
    )
    parser.add_argument("--responses", type=pathlib.Path, default=HERE / "responses")
    parser.add_argument(
        "--token-labels", type=pathlib.Path, default=HERE / "token_labels"
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=600)
    args = parser.parse_args(argv)

    try:
        if args.url:
            if not args.skip_model_check:
                _check_model(_model_info(_opener(), args.url), args.model)
            return int(_capture(args, args.url) != 0)

        _ensure_port_available(args.host, args.port)
        server = _start_server(args)
        url = _server_url(args.host, args.port)
        try:
            info = _wait_until_ready(_opener(), url, server, args.startup_timeout)
            _check_model(info, args.model)
            return int(_capture(args, url) != 0)
        finally:
            _stop_server(server)
    except (OSError, RuntimeError, SchemaError) as error:
        parser.exit(2, f"error: {terminal_safe_text(str(error))}\n")


if __name__ == "__main__":
    raise SystemExit(main())
