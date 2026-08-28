import copy

import pytest
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Static
from viewer_fixture import forensic_response

from sglang_constraint_metrics.viewer.app import (
    ConstraintMetricsApp,
    InspectionScreen,
    SummaryScreen,
)
from sglang_constraint_metrics.viewer.data import parse_response_obj, terminal_safe_text


def make_app():
    labels = {
        10: "alpha",
        11: " beta",
        12: " gamma",
        21: " blocked",
        30: "x",
        31: "y",
        32: "z",
        33: "delta",
    }
    return ConstraintMetricsApp(
        parse_response_obj(forensic_response(), token_labels=labels)
    )


@pytest.mark.asyncio
async def test_starts_at_peak_and_keeps_candidate_panes_in_sync():
    app = make_app()
    async with app.run_test(size=(150, 44)) as pilot:
        await pilot.pause()
        assert app.cursor == 1
        assert app.current.output_index == 1
        assert app.query_one("#masked", DataTable).row_count == 2
        assert app.query_one("#legal", DataTable).row_count == 1

        await pilot.press("down")
        assert app.cursor == 2
        assert app.query_one("#masked", DataTable).row_count == 0
        assert app.query_one("#legal", DataTable).row_count == 1


@pytest.mark.asyncio
async def test_high_pressure_navigation_and_summary():
    app = make_app()
    async with app.run_test(size=(150, 44)) as pilot:
        await pilot.pause()
        await pilot.press("home", "n")
        assert app.cursor == 1
        await pilot.press("s")
        await pilot.pause()
        assert isinstance(app.screen, SummaryScreen)
        await pilot.press("escape")


@pytest.mark.asyncio
async def test_core_panes_remain_visible_in_standard_terminal():
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.query_one("#masked", DataTable).size.height > 0
        assert app.query_one("#legal", DataTable).size.height > 0
        assert "alpha beta gamma" in app._output_text().plain


@pytest.mark.asyncio
async def test_token_labels_are_literal_text_not_rich_markup():
    response = forensic_response()
    metrics = parse_response_obj(
        response,
        token_labels={10: "[bold red]literal[/]", 30: "[green]candidate[/]"},
    )
    app = ConstraintMetricsApp(metrics)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.press("home")
        timeline_token = app.query_one("#timeline", DataTable).get_row_at(0)[1]
        candidate_token = app.query_one("#masked", DataTable).get_row_at(0)[0]
        assert isinstance(timeline_token, Text)
        assert timeline_token.plain == "[bold red]literal[/]"
        assert isinstance(candidate_token, Text)
        assert candidate_token.plain == "[green]candidate[/]"


@pytest.mark.asyncio
async def test_summary_mode_does_not_describe_output_as_missing_rows():
    response = copy.deepcopy(forensic_response())
    payload = response["meta_info"]["constraint_metrics"]
    payload["mode"] = "summary"
    payload["tokens"] = []
    app = ConstraintMetricsApp(parse_response_obj(response))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        summary = app._summary_text().plain
        assert "captured rows 0" in summary
        assert "output tokens 3" in summary
        assert not app.query_one("#tables").display


def test_terminal_control_characters_are_rendered_inert():
    value = "before\x1b]52;c;clipboard\x07\x9b31m\rafter\nnext"
    rendered = terminal_safe_text(value, multiline=True)

    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "\x9b" not in rendered
    assert "\r" not in rendered
    assert "\\x1b" in rendered
    assert "\\x07" in rendered
    assert rendered.endswith("after\nnext")


@pytest.mark.asyncio
async def test_untrusted_response_text_is_sanitized_before_rendering():
    response = copy.deepcopy(forensic_response())
    payload = response["meta_info"]["constraint_metrics"]
    response["text"] = "output\x1b]52;c;clipboard\x07\nnext"
    payload["warnings"] = ["warning\x1b[31m"]
    payload["tokens"]["spec_source"][1] = "draft\x9b31m"
    metrics = parse_response_obj(
        response,
        token_labels={11: "token\x1b]0;title\x07"},
        source="file\rname",
    )
    app = ConstraintMetricsApp(metrics)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        visible = (
            f"{app.sub_title}\n{app._summary_text().plain}\n"
            f"{app._output_text().plain}\n{app._detail_text().plain}"
        )
        assert not any(character in visible for character in "\x1b\x07\x9b\r")
        assert "\\x1b" in visible
        assert "\\x07" in visible


@pytest.mark.asyncio
async def test_full_token_inspection_is_scrollable_at_80_by_24():
    response = copy.deepcopy(forensic_response())
    response["text"] = "\n".join(f"output line {index}" for index in range(40))
    app = ConstraintMetricsApp(parse_response_obj(response))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, InspectionScreen)
        content = app.screen.query_one(Static).render().plain
        assert "TOP MASKED" in content
        assert "TOP LEGAL AFTER CONDITIONING" in content
        assert "output line 39" in content
        scroll = app.screen.query_one("#modal-box", VerticalScroll)
        assert scroll.max_scroll_y > 0
        await pilot.press("pagedown")
        await pilot.pause()
        assert scroll.scroll_y > 0
