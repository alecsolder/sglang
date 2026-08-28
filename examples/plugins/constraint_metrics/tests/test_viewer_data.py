import copy
import math

import pytest
from viewer_fixture import forensic_response

from sglang_constraint_metrics.viewer.data import SchemaError, parse_response_obj


def test_parse_current_forensic_columns_and_derived_probabilities():
    metrics = parse_response_obj(
        forensic_response(), token_labels={10: "alpha", 11: " beta"}
    )

    assert metrics.initial_token_position == 1
    assert metrics.token_label(11) == " beta"
    assert metrics.token_label(12) == "<12>"
    assert metrics.output_text == "alpha beta gamma"
    assert metrics.token_ids() == {10, 11, 12, 21, 30, 31, 32, 33}
    assert metrics.summary.top_pressure_events[0].emitted_token_id == 11

    selected = metrics.tokens[1]
    assert selected.spec_source == "accepted_draft"
    assert selected.high_pressure
    assert selected.legal_mass == pytest.approx(math.exp(-1.5))
    assert selected.masked_mass == pytest.approx(1.0 - math.exp(-1.5))
    assert selected.emitted_model_probability == pytest.approx(0.1)
    assert selected.emitted_conditioned_probability == pytest.approx(
        0.1 / math.exp(-1.5)
    )
    assert [candidate.token_id for candidate in selected.top_masked] == [31, 32]
    assert selected.top_legal_after_conditioning[0].conditioned_probability == 0.45
    assert selected.top_legal_after_conditioning[0].model_probability == pytest.approx(
        0.45 * math.exp(-1.5)
    )


def test_parse_accepts_bare_metrics_and_one_item_response_list():
    response = forensic_response()
    expected = parse_response_obj(response)

    assert parse_response_obj([response]).tokens == expected.tokens
    assert (
        parse_response_obj(response["meta_info"]["constraint_metrics"]).tokens
        == expected.tokens
    )


def test_parse_rejects_misaligned_column():
    response = copy.deepcopy(forensic_response())
    del response["meta_info"]["constraint_metrics"]["tokens"]["entropy_nats"][-1]

    with pytest.raises(SchemaError, match="entropy_nats has 2 values; expected 3"):
        parse_response_obj(response)


def test_parse_rejects_bad_fixed_stride_table():
    response = copy.deepcopy(forensic_response())
    response["meta_info"]["constraint_metrics"]["tokens"]["top_masked"][
        "token_id"
    ].pop()

    with pytest.raises(SchemaError, match="top_masked.token_id has 5 values"):
        parse_response_obj(response)


@pytest.mark.parametrize("mode,tokens", [("summary", []), ("triggered", None)])
def test_parse_modes_without_terminal_token_rows(mode, tokens):
    response = forensic_response()
    metrics = response["meta_info"]["constraint_metrics"]
    metrics["mode"] = mode
    metrics["tokens"] = tokens
    if mode == "triggered":
        metrics["triggered_token_count"] = 2

    parsed = parse_response_obj(response)
    assert parsed.tokens == ()
    assert parsed.triggered_token_count == (2 if mode == "triggered" else None)


def test_conditioned_probability_uses_log_space_before_exponentiating():
    response = forensic_response()
    tokens = response["meta_info"]["constraint_metrics"]["tokens"]
    tokens["constraint_pressure_nats"][0] = 800.0
    tokens["emitted"]["model_log_prob"][0] = -800.0

    assert parse_response_obj(response).tokens[0].emitted_conditioned_probability == 1.0


def test_parse_rejects_batch_instead_of_silently_selecting_first():
    with pytest.raises(SchemaError, match="one response at a time"):
        parse_response_obj([forensic_response(), forensic_response()])


def test_parse_rejects_numeric_overflow_as_schema_error():
    response = forensic_response()
    response["meta_info"]["constraint_metrics"]["tokens"]["constraint_pressure_nats"][
        0
    ] = (10**10000)

    with pytest.raises(SchemaError, match="must be a finite number"):
        parse_response_obj(response)


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("constraint_pressure_nats", -1000.0, "must be non-negative"),
        ("entropy_nats", -1.0, "must be non-negative"),
    ],
)
def test_parse_rejects_negative_nonnegative_metrics(column, value, match):
    response = forensic_response()
    response["meta_info"]["constraint_metrics"]["tokens"][column][0] = value

    with pytest.raises(SchemaError, match=match):
        parse_response_obj(response)


def test_parse_rejects_invalid_probability_and_log_probability():
    response = forensic_response()
    tokens = response["meta_info"]["constraint_metrics"]["tokens"]
    tokens["unconstrained_top1"]["prob"][0] = 1.1
    with pytest.raises(SchemaError, match="must be between 0 and 1"):
        parse_response_obj(response)

    response = forensic_response()
    response["meta_info"]["constraint_metrics"]["tokens"]["emitted"]["model_log_prob"][
        0
    ] = 0.1
    with pytest.raises(SchemaError, match="must be non-positive"):
        parse_response_obj(response)


def test_parse_rejects_conditioned_probability_above_one():
    response = forensic_response()
    tokens = response["meta_info"]["constraint_metrics"]["tokens"]
    tokens["constraint_pressure_nats"][0] = 10.0
    tokens["emitted"]["model_log_prob"][0] = -1.0

    with pytest.raises(SchemaError, match="is inconsistent with"):
        parse_response_obj(response)


@pytest.mark.parametrize(
    "path",
    [
        ("summary", "empty_support_count"),
        ("summary", "effective_legal_token_count_histogram", 0),
        ("tokens", "effective_legal_token_count", 0),
    ],
)
def test_parse_rejects_negative_counts(path):
    response = forensic_response()
    value = response["meta_info"]["constraint_metrics"]
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = -1

    with pytest.raises(SchemaError, match="must be non-negative"):
        parse_response_obj(response)


def test_parse_rejects_mismatched_summary_output_count():
    response = forensic_response()
    response["meta_info"]["constraint_metrics"]["summary"]["output_token_count"] = 2

    with pytest.raises(SchemaError, match="does not match"):
        parse_response_obj(response)


def test_parse_accepts_partially_dropped_summary_prefix():
    response = forensic_response()
    metrics = response["meta_info"]["constraint_metrics"]
    metrics["status"] = "partially_dropped"
    metrics["warnings"] = ["constraint metrics are missing 1 committed token record(s)"]
    metrics["mode"] = "summary"
    metrics["summary"]["output_token_count"] = 2
    metrics["tokens"] = []

    parsed = parse_response_obj(response)

    assert parsed.status == "partially_dropped"
    assert parsed.summary.output_token_count == 2
    assert parsed.output_token_count == 3
