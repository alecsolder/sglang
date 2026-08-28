import argparse
import json
import re
from pathlib import Path

import pytest
from viewer_fixture import forensic_response

from examples.structured_output import run_demo
from sglang_constraint_metrics.viewer.client import (
    build_request_body,
    load_token_labels,
)
from sglang_constraint_metrics.viewer.data import SchemaError, parse_response_obj

DEMO_DIR = Path(run_demo.__file__).parent
DEMO_GRAMMAR = DEMO_DIR / "chat_turn_json.lark"
CALIBRATION_GRAMMAR = DEMO_DIR.parent / "muse_glimmer" / "chat_turn.lark"


def _args(tmp_path: Path, requests: Path) -> argparse.Namespace:
    return argparse.Namespace(
        requests=requests,
        submitted_requests=tmp_path / "submitted",
        responses=tmp_path / "responses",
        token_labels=tmp_path / "labels",
        top_k=5,
        request_timeout=10,
    )


def _within_bounds(actual, bounds):
    return bounds.get("min", actual) <= actual <= bounds.get("max", actual)


def _matches_token(metrics, token, expected):
    return (
        expected.get("output_index", token.output_index) == token.output_index
        and expected.get("argmax_masked", token.argmax_masked) is token.argmax_masked
        and _within_bounds(
            token.constraint_pressure_nats,
            expected.get("constraint_pressure_nats", {}),
        )
        and expected.get("emitted_label", metrics.token_label(token.token_id))
        == metrics.token_label(token.token_id)
        and expected.get(
            "unconstrained_top1_label",
            metrics.token_label(token.unconstrained_top1_token_id),
        )
        == metrics.token_label(token.unconstrained_top1_token_id)
        and (
            "top_masked_label" not in expected
            or any(
                metrics.token_label(candidate.token_id) == expected["top_masked_label"]
                for candidate in token.top_masked
            )
        )
    )


def _assert_expectation(metrics, expectation):
    assert expectation["version"] == 1
    for name, bounds in expectation.get("summary", {}).items():
        assert _within_bounds(getattr(metrics.summary, name), bounds)
    for expected in expectation.get("any_token", []):
        assert any(_matches_token(metrics, token, expected) for token in metrics.tokens)
    for expected in expectation.get("every_token_labeled", []):
        label = expected["emitted_label"]
        labeled = [
            token
            for token in metrics.tokens
            if metrics.token_label(token.token_id) == label
        ]
        assert labeled, f"no emitted token is labeled {label!r}"
        assert all(_matches_token(metrics, token, expected) for token in labeled)


def _grammar_rules(text):
    """Return the single-line rule bodies of a lark grammar, comments stripped."""

    rules = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([a-z][\w-]*):\s*(.*?)\s*(?://.*)?", line)
        if match:
            rules[match.group(1)] = match.group(2)
    return rules


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
    (requests / "shard-0" / "ignored.expect.json").write_text(
        json.dumps({"version": 1})
    )

    posted = {}

    def post(_opener, _url, payload, _timeout):
        body = json.loads(payload)
        posted[body["text"]] = payload
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
        response_path = args.responses / relative
        response = json.loads(response_path.read_text())
        labels = load_token_labels(args.token_labels / relative)
        assert (args.submitted_requests / relative).read_bytes() == posted[
            source["text"]
        ]
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
        assert response_path.read_bytes().endswith(b"\n")
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
    monkeypatch.setattr(
        run_demo,
        "detokenize",
        lambda _url, token_ids, timeout: {
            token_id: f"token-{token_id}" for token_id in token_ids
        },
    )
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


def test_demo_grammar_only_replaces_the_calibration_answer_body():
    calibration = _grammar_rules(CALIBRATION_GRAMMAR.read_text())
    demo = _grammar_rules(DEMO_GRAMMAR.read_text())
    shared = (calibration.keys() & demo.keys()) - {"answer_turn"}

    assert "start" in shared and "to_user_header" in shared
    assert {name: demo[name] for name in shared} == {
        name: calibration[name] for name in shared
    }
    assert calibration["answer_turn"] == "to_user_header content_nonempty eot"
    assert demo["answer_turn"] == "to_user_header json_answer eot"


@pytest.mark.parametrize(
    "request_path",
    run_demo._request_paths(DEMO_DIR / "requests"),
    ids=lambda path: str(path.relative_to(DEMO_DIR / "requests")),
)
def test_committed_request_posts_the_full_turn_grammar(request_path):
    sampling_params = json.loads(request_path.read_text())["sampling_params"]

    assert sampling_params["ebnf"] == DEMO_GRAMMAR.read_text()
    assert "json_schema" not in sampling_params


@pytest.mark.parametrize(
    "request_path",
    run_demo._request_paths(DEMO_DIR / "requests"),
    ids=lambda path: str(path.relative_to(DEMO_DIR / "requests")),
)
def test_committed_request_generates_from_the_recipient_marker(request_path):
    input_ids = json.loads(request_path.read_text())["input_ids"]

    # <|start|>assistant, with no prefilled recipient: the grammar chooses it.
    assert input_ids[-2:] == [200022, 140680]


@pytest.mark.parametrize(
    "request_path",
    run_demo._request_paths(DEMO_DIR / "requests"),
    ids=lambda path: str(path.relative_to(DEMO_DIR / "requests")),
)
def test_committed_demo_fixture_matches_expectation(request_path):
    relative = request_path.relative_to(DEMO_DIR / "requests")
    submitted_path = DEMO_DIR / "submitted_requests" / relative
    response_path = DEMO_DIR / "responses" / relative
    label_path = DEMO_DIR / "token_labels" / relative
    expectation_path = run_demo._expectation_path(request_path)
    if not response_path.is_file():
        pytest.skip(
            f"{relative} has not been captured; "
            "see examples/structured_output/README.md"
        )

    assert submitted_path.read_bytes() == run_demo._json_bytes(
        build_request_body(json.loads(request_path.read_text()), 8)
    )
    metrics = parse_response_obj(
        json.loads(response_path.read_text()), source=str(response_path)
    )
    labels = load_token_labels(label_path)
    assert metrics.status == "complete"
    assert metrics.mode == "forensic"
    assert not metrics.warnings
    assert metrics.token_ids() <= labels.keys()
    _assert_expectation(
        metrics.with_token_labels(labels), json.loads(expectation_path.read_text())
    )
