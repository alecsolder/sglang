"""Render captured cases as scrolling TUI recordings in `media/`.

The cursor walks every token from the first to the last. Frames are emitted on
every step around a token the grammar contests and on every third step through
the long uncontested runs, which keeps the files small without hiding any row:
the table still scrolls past all of them.

Requires `cairosvg` and `ffmpeg` in addition to the viewer extra.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import subprocess
import sys
from collections.abc import Sequence

from sglang_constraint_metrics.viewer.app import ConstraintMetricsApp
from sglang_constraint_metrics.viewer.client import load_token_labels
from sglang_constraint_metrics.viewer.data import (
    ConstraintMetrics,
    SchemaError,
    parse_response_obj,
)

HERE = pathlib.Path(__file__).parent
TERMINAL_SIZE = (144, 38)
SVG_SCALE = 0.62
GIF_WIDTH = 760
FPS = 10
STRIDE = 3  # steps to skip while nothing is contested
NEAR = 3  # steps around a contested token that stay frame by frame
HOLD_FRAMES = 8  # extra frames where the grammar actually bites


def _load(name: str, responses: pathlib.Path, labels: pathlib.Path):
    return parse_response_obj(
        json.loads((responses / f"{name}.json").read_text()),
        token_labels=load_token_labels(labels / f"{name}.json"),
        source=name,
    )


def _contested(token) -> bool:
    return token.argmax_masked or token.constraint_pressure_nats >= 1.0


async def _frames(metrics: ConstraintMetrics, frame_dir: pathlib.Path) -> int:
    import cairosvg

    tokens = metrics.tokens
    last = len(tokens) - 1
    hot = {
        position
        for index, token in enumerate(tokens)
        if _contested(token)
        for position in range(index - NEAR, index + NEAR + 1)
    }

    frame_dir.mkdir(parents=True, exist_ok=True)
    for stale in frame_dir.glob("*.png"):
        stale.unlink()

    app = ConstraintMetricsApp(metrics)
    written = 0
    async with app.run_test(size=TERMINAL_SIZE) as pilot:
        await pilot.press("g")  # start at the first token
        await pilot.pause()
        for position, token in enumerate(tokens):
            edge = position in (0, last)
            if edge or position in hot or position % STRIDE == 0:
                png = cairosvg.svg2png(
                    bytestring=app.export_screenshot().encode(), scale=SVG_SCALE
                )
                for _ in range(1 + (HOLD_FRAMES if edge or _contested(token) else 0)):
                    (frame_dir / f"f{written:05d}.png").write_bytes(png)
                    written += 1
            await pilot.press("j")
            await pilot.pause()
    return written


def _encode(frame_dir: pathlib.Path, gif: pathlib.Path) -> None:
    palette = frame_dir / "palette.png"
    scale = f"scale={GIF_WIDTH}:-2:flags=lanczos"
    common = ["-framerate", str(FPS), "-i", str(frame_dir / "f%05d.png")]
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            *common,
            "-vf",
            f"{scale},palettegen=max_colors=32:stats_mode=diff",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            *common,
            "-i",
            str(palette),
            "-lavfi",
            f"{scale}[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle",
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
    )
    palette.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="default: every captured case")
    parser.add_argument("--responses", type=pathlib.Path, default=HERE / "responses")
    parser.add_argument(
        "--token-labels", type=pathlib.Path, default=HERE / "token_labels"
    )
    parser.add_argument("--media", type=pathlib.Path, default=HERE / "media")
    parser.add_argument(
        "--frames",
        type=pathlib.Path,
        help="keep intermediate PNG frames here instead of a temporary directory",
    )
    args = parser.parse_args(argv)

    cases = args.cases or sorted(path.stem for path in args.responses.glob("*.json"))
    if not cases:
        parser.exit(2, f"error: no captured responses under {args.responses}\n")
    args.media.mkdir(parents=True, exist_ok=True)

    for case in cases:
        try:
            metrics = _load(case, args.responses, args.token_labels)
        except (OSError, SchemaError) as error:
            print(f"{case}: error: {error}", file=sys.stderr)
            return 1
        frame_dir = (args.frames or args.media / ".frames") / case
        count = asyncio.run(_frames(metrics, frame_dir))
        gif = args.media / f"{case}.gif"
        _encode(frame_dir, gif)
        if args.frames is None:
            for frame in frame_dir.glob("*.png"):
                frame.unlink()
            frame_dir.rmdir()
            frame_dir.parent.rmdir()
        print(f"{case}: {count} frames -> {gif} ({gif.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
