import argparse
import json
from pathlib import Path

import pytest
from viewer_fixture import forensic_response

from examples.structured_output import run_demo
from sglang_constraint_metrics.viewer.client import load_token_labels
from sglang_constraint_metrics.viewer.data import SchemaError


def _args(tmp_path: Path, requests: Path) -> argparse.Namespace:
    return argparse.Namespace(
        requests=requests,
        submitted_requests=tmp_path / "submitted",
        responses=tmp_path / "responses",
        token_labels=tmp_path / "labels",
        top_k=5,
        request_timeout=10,
    )


def test_capture_discovers_request_files_and_preserves_relative_paths(
    monkeypatch, tmp_path
):
    requests = tmp_path / "requests"
    expected_paths = set()
    for index in range(6):
        relative = Path(f"shard-{index % 2}") / f"request-{index}.json"
        path = requests / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "text": f"prompt {index}",
                    "sampling_params": {"temperature": index / 10},
                    "stream": True,
                }
            )
        )
        expected_paths.add(relative)

    posted = []

    def post(_opener, _url, payload, _timeout):
        body = json.loads(payload)
        posted.append(body)
        response = forensic_response()
        response["text"] = body["text"]
        return 200, json.dumps(response, separators=(",", ":")).encode()

    monkeypatch.setattr(run_demo, "_opener", object)
    monkeypatch.setattr(run_demo, "_post_request", post)
    monkeypatch.setattr(
        run_demo,
        "detokenize",
        lambda _url, token_ids, timeout: {
            token_id: f"token-{token_id}" for token_id in token_ids
        },
    )
    args = _args(tmp_path, requests)

    assert run_demo._capture(args, "http://server") == 0
    assert len(posted) == len(expected_paths)
    assert {
        path.relative_to(args.submitted_requests)
        for path in args.submitted_requests.rglob("*.json")
    } == expected_paths
    assert {
        path.relative_to(args.responses) for path in args.responses.rglob("*.json")
    } == expected_paths
    assert {
        path.relative_to(args.token_labels)
        for path in args.token_labels.rglob("*.json")
    } == expected_paths

    for relative in expected_paths:
        source = json.loads((requests / relative).read_text())
        submitted = json.loads((args.submitted_requests / relative).read_text())
        response = json.loads((args.responses / relative).read_text())
        labels = load_token_labels(args.token_labels / relative)
        assert submitted["text"] == source["text"]
        assert (
            submitted["sampling_params"]["temperature"]
            == source["sampling_params"]["temperature"]
        )
        assert submitted["sampling_params"]["constraint_metrics"] == {
            "mode": "forensic",
            "top_k": 5,
        }
        assert submitted["stream"] is False
        assert response["text"] == source["text"]
        assert labels[10] == "token-10"


def test_capture_continues_after_a_bad_request(monkeypatch, tmp_path):
    requests = tmp_path / "requests"
    requests.mkdir()
    (requests / "bad.json").write_text("not JSON")
    (requests / "good.json").write_text(json.dumps({"text": "still captured"}))

    monkeypatch.setattr(run_demo, "_opener", object)
    monkeypatch.setattr(
        run_demo,
        "_post_request",
        lambda *_args: (200, json.dumps(forensic_response()).encode()),
    )
    monkeypatch.setattr(run_demo, "detokenize", lambda *_args, **_kwargs: {})
    args = _args(tmp_path, requests)

    assert run_demo._capture(args, "http://server") == 1
    assert (args.responses / "good.json").is_file()
    assert not (args.responses / "bad.json").exists()


def test_capture_rejects_overlapping_input_and_output_roots(tmp_path):
    requests = tmp_path / "requests"
    args = _args(tmp_path, requests)
    args.responses = requests / "responses"

    with pytest.raises(SchemaError, match="roots must be separate"):
        run_demo._capture(args, "http://server")
