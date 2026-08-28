import os
import subprocess
import sys


def _run_in_fresh_process(
    source: str, *, selected: bool
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if selected:
        env["SGLANG_PLUGINS"] = "constraint_metrics"
    else:
        env.pop("SGLANG_PLUGINS", None)
    return subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        env=env,
        text=True,
        timeout=120,
    )


def test_installed_entry_point_is_inert_until_explicitly_selected():
    result = _run_in_fresh_process(
        """
import importlib.metadata
import sys

entry_points = [
    entry_point
    for entry_point in importlib.metadata.entry_points(group="sglang.srt.plugins")
    if entry_point.name == "constraint_metrics"
]
assert len(entry_points) == 1
entry_points[0].load()()
assert "sglang_constraint_metrics.integration" not in sys.modules
print("inactive-ok")
""",
        selected=False,
    )

    assert result.stdout.strip() == "inactive-ok"


def test_selected_entry_point_applies_and_verifies_every_runtime_hook():
    result = _run_in_fresh_process(
        """
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import ServerArgs

original_server_args = ServerArgs
original_engine_server_args = Engine.server_args_class

load_plugins()

from sglang_constraint_metrics.plugin import verify_runtime_hooks

verify_runtime_hooks()
from sglang.srt.server_args import ServerArgs as LoadedServerArgs
assert LoadedServerArgs is original_server_args
assert Engine.server_args_class is original_engine_server_args
print("active-ok")
""",
        selected=True,
    )

    assert result.stdout.strip() == "active-ok"


def test_partial_registration_leaves_startup_guard_in_place():
    result = _run_in_fresh_process(
        """
import sglang_constraint_metrics.plugin as plugin
from sglang.srt.plugins import load_plugins
from sglang.srt.plugins.hook_registry import HookRegistry

real_import_module = plugin.import_module

def fail_implementation_import(name):
    if name == "sglang_constraint_metrics.integration":
        raise ImportError("simulated incompatible implementation")
    return real_import_module(name)

plugin.import_module = fail_implementation_import
load_plugins()
assert "sglang.srt.server_args.ServerArgs.check_server_args" in HookRegistry._patched
try:
    plugin.verify_runtime_hooks()
except RuntimeError as error:
    assert "hooks were not fully applied" in str(error)
else:
    raise AssertionError("partial plugin registration was not detected")
print("partial-registration-guard-ok")
""",
        selected=True,
    )

    assert result.stdout.strip() == "partial-registration-guard-ok"
