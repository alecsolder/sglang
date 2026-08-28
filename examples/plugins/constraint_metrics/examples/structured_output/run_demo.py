"""Replay request files and save complete responses for offline inspection."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from sglang_constraint_metrics.viewer.client import detokenize, save_token_labels
from sglang_constraint_metrics.viewer.data import SchemaError, parse_response_obj

HERE = pathlib.Path(__file__).parent
DEFAULT_MODEL = "meta-models/Muse-Glimmer-30B"
DEFAULT_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _wait_until_ready(
    opener: urllib.request.OpenerDirector,
    url: str,
    server: subprocess.Popen[bytes],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    endpoint = f"{url.rstrip('/')}/model_info"
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"SGLang exited with status {server.returncode}")
        try:
            with opener.open(endpoint, timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise TimeoutError(f"SGLang did not become ready within {timeout:g} seconds")


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


def _read_request(path: pathlib.Path) -> bytes:
    raw = path.read_bytes()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SchemaError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(body, dict):
        raise SchemaError(f"{path} must contain one request object")
    if body.get("stream") is not False:
        raise SchemaError(f"{path} must explicitly set stream to false")
    return raw


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


def _capture(args: argparse.Namespace, url: str) -> None:
    request_paths = sorted(args.requests.rglob("*.json"))
    if not request_paths:
        raise FileNotFoundError(f"no JSON requests found under {args.requests}")
    opener = _opener()
    for request_path in request_paths:
        relative = request_path.relative_to(args.requests)
        response_path = args.responses / relative
        label_path = args.token_labels / relative
        status, raw_response = _post_request(
            opener, url, _read_request(request_path), args.request_timeout
        )
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_bytes(raw_response)
        try:
            body: Any = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise SchemaError(
                f"{response_path} contains invalid JSON: {error}"
            ) from error
        if status != 200:
            raise RuntimeError(f"{relative} returned HTTP {status}: {body}")
        metrics = parse_response_obj(body, source=str(response_path), endpoint=url)
        labels = detokenize(url, metrics.token_ids(), timeout=30)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        save_token_labels(label_path, labels)
        summary = metrics.summary
        print(
            f"{relative}: tokens={summary.output_token_count} "
            f"masked={summary.argmax_masked_count} "
            f"max_pressure={summary.max_constraint_pressure_nats:.6g}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="reuse an existing SGLang server")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--requests", type=pathlib.Path, default=HERE / "requests")
    parser.add_argument("--responses", type=pathlib.Path, default=HERE / "responses")
    parser.add_argument(
        "--token-labels", type=pathlib.Path, default=HERE / "token_labels"
    )
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=600)
    args = parser.parse_args()

    if args.url:
        _capture(args, args.url)
        return 0

    server = _start_server(args)
    url = f"http://{args.host}:{args.port}"
    try:
        _wait_until_ready(_opener(), url, server, args.startup_timeout)
        _capture(args, url)
    finally:
        _stop_server(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
