"""Textual UI for one parsed constraint-metrics response."""

from __future__ import annotations

import math
from typing import ClassVar

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Static

from .data import (
    INF_PRESSURE_SENTINEL,
    ConstraintMetrics,
    TokenMetric,
    terminal_safe_text,
)


def _number(value: float) -> str:
    if value >= INF_PRESSURE_SENTINEL:
        return "∞"
    magnitude = abs(value)
    if magnitude != 0.0 and magnitude < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _probability(value: float | None) -> str:
    return "—" if value is None else _number(value)


def _token_text(metrics: ConstraintMetrics, token_id: int, width: int = 22) -> str:
    visible = terminal_safe_text(metrics.token_label(token_id))
    if len(visible) > width:
        visible = visible[: width - 1] + "…"
    return visible


def _pressure_bar(value: float, maximum: float, width: int = 8) -> str:
    if value >= INF_PRESSURE_SENTINEL:
        return "█" * width
    if value <= 0.0 or maximum <= 0.0:
        return ""
    scale = math.log1p(value) / math.log1p(maximum)
    return "█" * max(1, round(width * scale))


_SCROLLABLE_MODAL_BINDINGS = [
    Binding("escape,q", "dismiss", "Close"),
    Binding("down,j", "scroll_down", "Scroll down", priority=True),
    Binding("up,k", "scroll_up", "Scroll up", priority=True),
    Binding("pagedown", "page_down", "Page down", show=False, priority=True),
    Binding("pageup", "page_up", "Page up", show=False, priority=True),
]


class _ScrollableModal(ModalScreen):
    BINDINGS: ClassVar = _SCROLLABLE_MODAL_BINDINGS

    def action_scroll_down(self) -> None:
        self.query_one("#modal-box", VerticalScroll).scroll_down(animate=False)

    def action_scroll_up(self) -> None:
        self.query_one("#modal-box", VerticalScroll).scroll_up(animate=False)

    def action_page_down(self) -> None:
        self.query_one("#modal-box", VerticalScroll).scroll_page_down(animate=False)

    def action_page_up(self) -> None:
        self.query_one("#modal-box", VerticalScroll).scroll_page_up(animate=False)


class SummaryScreen(_ScrollableModal):
    BINDINGS: ClassVar = _SCROLLABLE_MODAL_BINDINGS + [Binding("s", "dismiss", "Close")]

    def __init__(self, metrics: ConstraintMetrics) -> None:
        super().__init__()
        self.metrics = metrics

    def compose(self) -> ComposeResult:
        summary = self.metrics.summary
        histogram = "\n".join(
            f"  {escape(terminal_safe_text(label)):>8} legal tokens  {count:>7}"
            for label, count in summary.effective_legal_token_count_histogram
            if count
        )
        events = "\n".join(
            "  "
            f"#{event.output_index:<5} "
            f"pressure {_number(event.constraint_pressure_nats):>9}  "
            f"token {escape(_token_text(self.metrics, event.emitted_token_id, 18))} "
            f"({event.emitted_token_id})  legal {event.effective_legal_token_count}"
            for event in summary.top_pressure_events
        )
        triggered = (
            f"\ntriggered token count          {self.metrics.triggered_token_count}"
            if self.metrics.triggered_token_count is not None
            else ""
        )
        body = (
            "[b]Request summary[/b]\n\n"
            f"status                         {self.metrics.status}\n"
            f"mode                           {self.metrics.mode}\n"
            f"input / output tokens          {self.metrics.input_token_count} / "
            f"{self.metrics.output_token_count}\n"
            f"grammar-active / finite rows   {summary.grammar_active_token_count} / "
            f"{summary.finite_active_token_count}\n"
            f"finite-active cumulative       {_number(summary.cumulative_constraint_pressure_nats)} nats\n"
            f"finite-active mean             {_number(summary.mean_constraint_pressure_nats)} nats\n"
            f"finite-active peak             {_number(summary.max_constraint_pressure_nats)} nats "
            f"at output {summary.max_pressure_output_index}\n"
            f"argmax masked                  {summary.argmax_masked_count}\n"
            f"mean / max top-1 gap           "
            f"{_number(summary.mean_top1_logit_gap_to_top2_nats)} / "
            f"{_number(summary.max_top1_logit_gap_to_top2_nats)} nats\n"
            f"high-pressure + low-entropy    {summary.high_pressure_low_entropy_count}\n"
            f"longest high-pressure run      {summary.longest_high_pressure_run}\n"
            f"empty / EOS-only support       {summary.empty_support_count} / "
            f"{summary.eos_only_support_count}\n"
            f"non-finite rows                {summary.has_non_finite_count}\n\n"
            "[b]Effective legal-token histogram[/b]\n"
            f"{histogram or '  no captured rows'}\n\n"
            "[b]Top finite-pressure events[/b]\n"
            f"{events or '  no finite pressure events'}"
            f"{triggered}"
        )
        with VerticalScroll(id="modal-box"):
            yield Static(body)


class HelpScreen(_ScrollableModal):
    BINDINGS: ClassVar = _SCROLLABLE_MODAL_BINDINGS + [
        Binding("question_mark", "dismiss", "Close")
    ]

    def compose(self) -> ComposeResult:
        yield Static(
            "[b]Constraint metrics[/b]\n\n"
            "pressure = -ln(Z), where Z is the model probability mass already "
            "allowed by the grammar. High pressure means the constraint changed "
            "the distribution strongly; it is not an output-quality verdict.\n\n"
            "masked mass = 1 - Z. The emitted raw probability is measured before "
            "grammar conditioning; q is its probability after conditioning.\n\n"
            "Top masked lists high-probability tokens removed by the grammar. "
            "Top legal after conditioning lists legal candidates by q.\n\n"
            "Summary pressure aggregates cover finite grammar-active rows. Empty "
            "support is shown as infinite in the token timeline and counted separately.\n\n"
            "Navigation: j/k or arrows move one token; n/p jump between rows with "
            "pressure ≥ 1 nat; Home/End jump to the ends; s opens the full summary.",
            id="modal-box",
        )


class InspectionScreen(_ScrollableModal):
    BINDINGS: ClassVar = _SCROLLABLE_MODAL_BINDINGS + [
        Binding("enter,i", "dismiss", "Close")
    ]

    def __init__(self, content: Text) -> None:
        super().__init__()
        self.content = content

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="modal-box"):
            yield Static(self.content)


class ConstraintMetricsApp(App):
    """Interactive token timeline with synchronized distribution details."""

    CSS = """
    Screen { background: #0f111a; color: #dfe3ee; }
    #summary { height: 4; padding: 0 1; background: #141824; border-bottom: solid #2c3450; }
    #body { height: 1fr; }
    #timeline { width: 52%; border: round #4c3f78; }
    #right { width: 1fr; }
    #output { height: 5; padding: 0 1; border: round #2c5870; overflow-y: auto; }
    #detail { height: 7; padding: 0 1; border: round #4c3f78; }
    #tables { height: 1fr; }
    .candidate-pane { width: 1fr; border: round #2c3450; }
    .table-title { height: 1; padding: 0 1; background: #141824; }
    .candidate-table { height: 1fr; }
    #definitions { height: 2; padding: 0 1; color: #9aa4bd; }
    ModalScreen { align: center middle; background: #0f111a 75%; }
    #modal-box { width: 78%; height: 80%; padding: 1 2; border: round #a78bfa; background: #141824; }
    """

    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit"),
        Binding("down,j", "move(1)", "next token", priority=True),
        Binding("up,k", "move(-1)", "previous token", priority=True),
        Binding("right,n", "next_high", "next high pressure", priority=True),
        Binding("left,p", "previous_high", "previous high pressure", priority=True),
        Binding("home,g", "first", "first", show=False, priority=True),
        Binding("end,G", "last", "last", show=False, priority=True),
        Binding("enter,i", "inspect", "full detail", priority=True),
        Binding("s", "summary", "summary"),
        Binding("question_mark,f1", "help", "explain"),
    ]

    def __init__(self, metrics: ConstraintMetrics) -> None:
        super().__init__()
        self.metrics = metrics
        self.cursor = metrics.initial_token_position or 0

    def compose(self) -> ComposeResult:
        yield Static(id="summary")
        with Horizontal(id="body"):
            yield DataTable(id="timeline")
            with Vertical(id="right"):
                yield Static(id="output")
                yield Static(id="detail")
                with Horizontal(id="tables"):
                    with Vertical(classes="candidate-pane"):
                        yield Static("Top masked (raw model p)", classes="table-title")
                        yield DataTable(id="masked", classes="candidate-table")
                    with Vertical(classes="candidate-pane"):
                        yield Static(
                            "Top legal after conditioning (q / raw p)",
                            classes="table-title",
                        )
                        yield DataTable(id="legal", classes="candidate-table")
        yield Static(
            "pressure = -ln(legal mass) · high pressure ≥ 1 nat · ? explains metrics",
            id="definitions",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "SGLang constraint metrics"
        self.sub_title = terminal_safe_text(self.metrics.source or "")
        timeline = self.query_one("#timeline", DataTable)
        timeline.cursor_type = "row"
        timeline.zebra_stripes = True
        timeline.add_columns(
            "#", "token", "id", "pressure", "masked", "legal", "source"
        )
        maximum = max(
            (
                token.constraint_pressure_nats
                for token in self.metrics.tokens
                if token.constraint_pressure_nats < INF_PRESSURE_SENTINEL
            ),
            default=0.0,
        )
        for position, token in enumerate(self.metrics.tokens):
            marker = "!" if token.high_pressure else "*" if token.argmax_masked else ""
            pressure = _number(token.constraint_pressure_nats)
            bar = _pressure_bar(token.constraint_pressure_nats, maximum)
            timeline.add_row(
                str(token.output_index),
                Text(_token_text(self.metrics, token.token_id)),
                str(token.token_id),
                f"{marker}{pressure} {bar}",
                _number(token.masked_mass),
                str(token.effective_legal_token_count),
                terminal_safe_text(token.spec_source or "—"),
                key=str(position),
            )
        for table_id, columns in (
            ("#masked", ("token", "id", "raw p")),
            ("#legal", ("token", "id", "q", "raw p")),
        ):
            table = self.query_one(table_id, DataTable)
            table.cursor_type = "none"
            table.zebra_stripes = True
            table.add_columns(*columns)
        if self.metrics.mode != "forensic":
            self.query_one("#tables").display = False
        if self.metrics.tokens:
            timeline.move_cursor(row=self.cursor)
            timeline.focus()
        self._refresh_all()

    @property
    def current(self) -> TokenMetric | None:
        if not self.metrics.tokens:
            return None
        return self.metrics.tokens[self.cursor]

    def _refresh_all(self) -> None:
        self.query_one("#summary", Static).update(self._summary_text())
        self.query_one("#output", Static).update(self._output_text())
        self.query_one("#detail", Static).update(self._detail_text())
        self._refresh_tables()

    def _summary_text(self) -> Text:
        summary = self.metrics.summary
        text = Text()
        text.append("CONSTRAINT METRICS  ", style="bold cyan")
        text.append(f"{self.metrics.mode} / {self.metrics.status}\n", style="bold")
        text.append(
            f"captured rows {len(self.metrics.tokens)}  "
            f"output tokens {self.metrics.output_token_count}  "
            f"finite cumulative {_number(summary.cumulative_constraint_pressure_nats)} nats  "
            f"finite peak {_number(summary.max_constraint_pressure_nats)} @ {summary.max_pressure_output_index}  "
            f"argmax masked {summary.argmax_masked_count}"
        )
        if self.metrics.warnings:
            text.append(
                "\n" + " · ".join(map(terminal_safe_text, self.metrics.warnings)),
                style="bold yellow",
            )
        elif self.metrics.triggered_token_count is not None:
            text.append(
                f"\ntriggered rows {self.metrics.triggered_token_count}", style="dim"
            )
        return text

    def _output_text(self) -> Text:
        text = Text("TOKEN CONTEXT\n", style="bold cyan")
        if not self.metrics.tokens:
            text.append("No token records in this response mode.", style="dim")
            if self.metrics.output_text is not None:
                text.append("\nAUTHORITATIVE OUTPUT\n", style="bold cyan")
                text.append(
                    terminal_safe_text(self.metrics.output_text, multiline=True)
                )
            return text
        start = max(0, self.cursor - 16)
        end = min(len(self.metrics.tokens), self.cursor + 17)
        if start:
            text.append("…", style="dim")
        for position in range(start, end):
            token = self.metrics.tokens[position]
            label = terminal_safe_text(self.metrics.token_label(token.token_id))
            style = "reverse bold cyan" if position == self.cursor else ""
            if position != self.cursor and token.high_pressure:
                style = "bold yellow"
            text.append(label, style=style)
        if end < len(self.metrics.tokens):
            text.append("…", style="dim")
        current = self.metrics.tokens[self.cursor]
        text.append(
            f"\nselected output {current.output_index}: "
            f"{_token_text(self.metrics, current.token_id, 48)} ({current.token_id})",
            style="bold",
        )
        if self.metrics.output_text is not None:
            text.append("\nAUTHORITATIVE OUTPUT\n", style="bold cyan")
            text.append(terminal_safe_text(self.metrics.output_text, multiline=True))
        return text

    def _detail_text(self) -> Text:
        token = self.current
        if token is None:
            return Text("TOKEN DETAIL\nNo captured token rows.", style="dim")
        flags = ", ".join(token.flags) or "none"
        source = terminal_safe_text(token.spec_source or "normal")
        conditioned = _probability(token.emitted_conditioned_probability)
        text = Text("TOKEN DETAIL", style="bold cyan")
        text.append(
            f"  output {token.output_index}  grammar "
            f"{'active' if token.grammar_active else 'inactive'}  source {source}\n"
        )
        text.append(
            f"pressure {_number(token.constraint_pressure_nats)} nats  "
            f"legal mass {_number(token.legal_mass)}  masked mass {_number(token.masked_mass)}  "
            f"entropy {_number(token.entropy_nats)} nats\n"
        )
        text.append(
            f"emitted {_token_text(self.metrics, token.token_id)} ({token.token_id})  "
            f"raw p {_number(token.emitted_model_probability)}  q {conditioned}  "
            f"rank {token.emitted_rank if token.emitted_rank is not None else '—'}  "
            f"allowed {token.emitted_allowed}\n"
        )
        text.append(
            f"unconstrained top-1 "
            f"{_token_text(self.metrics, token.unconstrained_top1_token_id)} "
            f"({token.unconstrained_top1_token_id})  "
            f"p {_number(token.unconstrained_top1_probability)}  "
            f"gap {_number(token.top1_logit_gap_to_top2_nats)} nats\n"
        )
        text.append(
            f"effective legal tokens {token.effective_legal_token_count}  "
            f"argmax masked {token.argmax_masked}  flags {flags}"
        )
        return text

    def _refresh_tables(self) -> None:
        masked = self.query_one("#masked", DataTable)
        legal = self.query_one("#legal", DataTable)
        masked.clear(columns=False)
        legal.clear(columns=False)
        token = self.current
        if token is None:
            return
        for candidate in token.top_masked:
            masked.add_row(
                Text(_token_text(self.metrics, candidate.token_id)),
                str(candidate.token_id),
                _number(candidate.model_probability),
            )
        for candidate in token.top_legal_after_conditioning:
            legal.add_row(
                Text(_token_text(self.metrics, candidate.token_id)),
                str(candidate.token_id),
                _probability(candidate.conditioned_probability),
                _number(candidate.model_probability),
            )

    def _set_cursor(self, position: int) -> None:
        if not self.metrics.tokens:
            return
        position = max(0, min(position, len(self.metrics.tokens) - 1))
        if position == self.cursor:
            return
        self.cursor = position
        self.query_one("#timeline", DataTable).move_cursor(row=position)
        self._refresh_all()

    def action_move(self, delta: int) -> None:
        self._set_cursor(self.cursor + delta)

    def action_first(self) -> None:
        self._set_cursor(0)

    def action_last(self) -> None:
        self._set_cursor(len(self.metrics.tokens) - 1)

    def action_next_high(self) -> None:
        positions = [
            position
            for position, token in enumerate(self.metrics.tokens)
            if position > self.cursor and token.high_pressure
        ]
        if positions:
            self._set_cursor(positions[0])
        else:
            self.notify("no later high-pressure row", timeout=2)

    def action_previous_high(self) -> None:
        positions = [
            position
            for position, token in enumerate(self.metrics.tokens)
            if position < self.cursor and token.high_pressure
        ]
        if positions:
            self._set_cursor(positions[-1])
        else:
            self.notify("no earlier high-pressure row", timeout=2)

    def action_summary(self) -> None:
        self.push_screen(SummaryScreen(self.metrics))

    def action_inspect(self) -> None:
        token = self.current
        if token is None:
            self.notify("no captured token row to inspect", timeout=2)
            return
        content = Text("FULL TOKEN INSPECTION\n\n", style="bold cyan")
        content.append_text(self._output_text())
        content.append("\n\n")
        content.append_text(self._detail_text())
        content.append("\n\nTOP MASKED (raw model p)\n", style="bold cyan")
        if token.top_masked:
            for candidate in token.top_masked:
                content.append(
                    f"{_token_text(self.metrics, candidate.token_id, 36):<36} "
                    f"id {candidate.token_id:<8} p {_number(candidate.model_probability)}\n"
                )
        else:
            content.append("none\n", style="dim")
        content.append(
            "\nTOP LEGAL AFTER CONDITIONING (q / raw p)\n", style="bold cyan"
        )
        if token.top_legal_after_conditioning:
            for candidate in token.top_legal_after_conditioning:
                content.append(
                    f"{_token_text(self.metrics, candidate.token_id, 30):<30} "
                    f"id {candidate.token_id:<8} "
                    f"q {_probability(candidate.conditioned_probability):<10} "
                    f"raw p {_number(candidate.model_probability)}\n"
                )
        else:
            content.append("none\n", style="dim")
        self.push_screen(InspectionScreen(content))

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "timeline" and event.cursor_row != self.cursor:
            self.cursor = event.cursor_row
            self._refresh_all()
