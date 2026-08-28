import copy
import json

import pytest
from viewer_fixture import forensic_response

from sglang_constraint_metrics.viewer import client
from sglang_constraint_metrics.viewer.data import SchemaError


def test_build_request_body_isolated_overlay():
    request = {
        "text": "hello",
        "stream": True,
        "sampling_params": {
            "temperature": 0.2,
            "constraint_metrics": {"mode": "summary"},
        },
    }
    original = copy.deepcopy(request)

    body = client.build_request_body(request, 4)

    assert request == original
    assert body["stream"] is False
    assert body["sampling_params"] == {
        "temperature": 0.2,
        "constraint_metrics": {"mode": "forensic", "top_k": 4},
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", ["a", "b"]),
        ("input_ids", [[1], [2]]),
        ("input_embeds", [[[0.1, 0.2]], [[0.3, 0.4]]]),
    ],
)
def test_build_request_rejects_batched_prompt(field, value):
    with pytest.raises(SchemaError, match="accepts one request"):
        client.build_request_body({field: value}, 4)


@pytest.mark.parametrize("n", [0, 2, True, "1"])
def test_build_request_rejects_multiple_or_invalid_samples(n):
    with pytest.raises(SchemaError, match="sampling_params.n must be 1"):
        client.build_request_body({"text": "hello", "sampling_params": {"n": n}}, 4)


def test_submit_requests_forensic_then_detokenizes_unique_ids(monkeypatch, tmp_path):
    calls = []
    response = forensic_response()

    def post(url, body, timeout):
        calls.append((url, body, timeout))
        if url.endswith("/generate"):
            return response
        ids = [row[0] for row in body["tokens"]]
        return {"text": [f"token-{token_id}" for token_id in ids]}

    monkeypatch.setattr(client, "_post_json", post)
    save = tmp_path / "response.json"
    metrics = client.submit_request(
        "http://localhost:30000/generate",
        {"text": "hello", "sampling_params": {}},
        top_k=2,
        timeout=10,
        save=save,
    )

    assert calls[0][0] == "http://localhost:30000/generate"
    assert calls[0][1]["sampling_params"]["constraint_metrics"] == {
        "mode": "forensic",
        "top_k": 2,
    }
    assert calls[1][0] == "http://localhost:30000/v1/detokenize"
    assert calls[1][1]["tokens"] == [
        [10],
        [11],
        [12],
        [21],
        [30],
        [31],
        [32],
        [33],
    ]
    assert metrics.token_label(31) == "token-31"
    assert json.loads(save.read_text()) == response


@pytest.mark.parametrize("top_k", [-1, 33])
def test_build_request_rejects_invalid_top_k(top_k):
    with pytest.raises(SchemaError, match="between 0 and 32"):
        client.build_request_body({}, top_k)


def test_detokenize_rejects_misaligned_response(monkeypatch):
    monkeypatch.setattr(client, "_post_json", lambda *_args: {"text": ["one"]})

    with pytest.raises(SchemaError, match="invalid text array"):
        client.detokenize("http://localhost:30000", [1, 2])


def test_load_token_labels(tmp_path):
    path = tmp_path / "token_labels.json"
    path.write_text('{"10": "alpha", "11": " beta"}')

    assert client.load_token_labels(path) == {10: "alpha", 11: " beta"}


@pytest.mark.parametrize(
    "value,match",
    [
        ('["alpha"]', "must contain one object"),
        ('{"01": "alpha"}', "is not canonical"),
        ('{"10": 10}', "must be a string"),
    ],
)
def test_load_token_labels_rejects_invalid_maps(tmp_path, value, match):
    path = tmp_path / "token_labels.json"
    path.write_text(value)

    with pytest.raises(SchemaError, match=match):
        client.load_token_labels(path)


def test_save_token_labels_round_trips(tmp_path):
    path = tmp_path / "token_labels.json"

    client.save_token_labels(path, {10: "alpha", 11: " beta"})

    assert client.load_token_labels(path) == {10: "alpha", 11: " beta"}


def test_submit_falls_back_to_token_ids_when_detokenize_fails(monkeypatch):
    def post(url, _body, _timeout):
        if url.endswith("/generate"):
            return forensic_response()
        raise SchemaError("detokenizer unavailable")

    monkeypatch.setattr(client, "_post_json", post)
    with pytest.warns(RuntimeWarning, match="showing token IDs"):
        metrics = client.submit_request("http://localhost:30000", {"text": "hi"})

    assert metrics.token_labels == {}
