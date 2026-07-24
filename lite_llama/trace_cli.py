"""CLI entry point for live Lite Llama inference tracing.

Examples:
    python -m lite_llama.trace_cli run --trace-level layer -- \
        --checkpoints_dir my_weight/Qwen3-32B --port 8213

    python -m lite_llama.trace_cli attach http://127.0.0.1:8213
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Sequence


TRACE_LEVELS = ("request", "scheduler", "layer")


def _strip_separator(arguments: Sequence[str]) -> list[str]:
    values = list(arguments)
    if values and values[0] == "--":
        return values[1:]
    return values


def _option_value(
    arguments: Sequence[str], option: str, default: str
) -> str:
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return str(arguments[index + 1])
        if argument.startswith(f"{option}="):
            return argument.split("=", 1)[1]
    return default


def build_server_command(
    *,
    trace_level: str,
    trace_output: str | None,
    trace_buffer_events: int,
    server_args: Sequence[str],
) -> list[str]:
    """Build the existing server CLI while keeping one implementation path."""

    arguments = _strip_separator(server_args)
    if "--checkpoints_dir" not in arguments and not any(
        argument.startswith("--checkpoints_dir=")
        for argument in arguments
    ):
        raise ValueError(
            "server arguments must include --checkpoints_dir after `--`"
        )
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "server.py"),
        "--trace",
        "--trace_level",
        trace_level,
        "--trace_buffer_events",
        str(int(trace_buffer_events)),
    ]
    if trace_output:
        command.extend(("--trace_output", trace_output))
    if (
        trace_level == "layer"
        and "--compiled_model" not in arguments
        and "--no_compiled_model" not in arguments
    ):
        command.append("--no_compiled_model")
    command.extend(arguments)
    return command


def _viewer_url(server_args: Sequence[str]) -> str:
    arguments = _strip_separator(server_args)
    host = _option_value(arguments, "--host", "127.0.0.1")
    port = _option_value(arguments, "--port", "8000")
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{port}/debug/trace"


def _wait_until_ready(url: str, timeout: float = 180.0) -> bool:
    health_url = url.rsplit("/debug/trace", 1)[0] + "/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    return False


def run_trace(args: argparse.Namespace) -> int:
    command = build_server_command(
        trace_level=args.trace_level,
        trace_output=args.trace_output,
        trace_buffer_events=args.trace_buffer_events,
        server_args=args.server_args,
    )
    viewer_url = _viewer_url(args.server_args)
    process = subprocess.Popen(command)
    try:
        if not args.no_open:
            if _wait_until_ready(viewer_url):
                webbrowser.open(viewer_url)
            else:
                print(
                    f"Trace server did not become ready at {viewer_url}",
                    file=sys.stderr,
                )
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            return process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or attach to the Lite Llama inference trace viewer."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Start server.py with tracing enabled.",
    )
    run.add_argument(
        "--trace-level",
        choices=TRACE_LEVELS,
        default="layer",
    )
    run.add_argument("--trace-output", default=None)
    run.add_argument("--trace-buffer-events", type=int, default=50_000)
    run.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the local web browser after the server is ready.",
    )
    run.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to server.py after `--`.",
    )
    run.set_defaults(handler=run_trace)

    attach = subparsers.add_parser(
        "attach",
        help="Open the trace viewer of an already-running server.",
    )
    attach.add_argument("server_url")
    attach.set_defaults(
        handler=lambda values: int(
            not webbrowser.open(
                values.server_url.rstrip("/") + "/debug/trace"
            )
        )
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ValueError as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
