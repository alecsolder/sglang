import builtins

import pytest
from viewer_fixture import forensic_response

from sglang_constraint_metrics.viewer import cli
from sglang_constraint_metrics.viewer.app import ConstraintMetricsApp


def test_cli_explains_missing_optional_viewer_dependency(monkeypatch, capsys):
    original_import = builtins.__import__

    def missing_textual(name, *args, **kwargs):
        if name == "textual":
            raise ModuleNotFoundError("No module named 'textual'", name="textual")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_textual)
    with pytest.raises(SystemExit) as error:
        cli.main(["--response", "unused.json"])

    assert error.value.code == 2
    assert "sglang-constraint-metrics[viewer]" in capsys.readouterr().err


def test_cli_loads_offline_response_and_runs_app(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "load_json", lambda _path: forensic_response())
    monkeypatch.setattr(
        ConstraintMetricsApp, "run", lambda self: seen.append(self.metrics)
    )

    assert cli.main(["--response", "response.json"]) == 0
    assert seen[0].mode == "forensic"
    assert len(seen[0].tokens) == 3


def test_cli_reports_schema_errors_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_json", lambda _path: {})

    with pytest.raises(SystemExit) as error:
        cli.main(["--response", "response.json"])

    assert error.value.code == 2
    assert "response.meta_info is missing" in capsys.readouterr().err


def test_cli_sanitizes_control_characters_in_errors(monkeypatch, capsys):
    def fail(_path):
        raise cli.SchemaError("server said \x1b]52;c;clipboard\x07")

    monkeypatch.setattr(cli, "load_json", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(["--response", "response.json"])

    stderr = capsys.readouterr().err
    assert error.value.code == 2
    assert "\x1b" not in stderr
    assert "\x07" not in stderr
    assert "\\x1b" in stderr
    assert "\\x07" in stderr
