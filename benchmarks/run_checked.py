#!/usr/bin/env python3
"""Run one benchmark with atomic outcome, logs, timeout, and tree RSS.

This utility classifies execution only; a zero exit code does not certify the
physics in a child-produced result.  Limits apply only to the new process group.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import psutil


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _terminate_owned_group(process: subprocess.Popen, grace: float = 2.0) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - Windows CI is not currently used
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--rss-gib", type=float, default=0.0)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--expected-output", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    limits = (args.timeout, args.rss_gib, args.interval)
    if (
        not command
        or not all(math.isfinite(value) for value in limits)
        or min(args.timeout, args.rss_gib) < 0.0
        or args.interval <= 0.0
    ):
        parser.error("need a command, nonnegative limits, and positive interval")

    record = args.record.resolve()
    record.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = record.with_suffix(".stdout.log")
    stderr_path = record.with_suffix(".stderr.log")
    trace_path = record.with_suffix(".resources.jsonl")
    expected = [path.resolve() for path in args.expected_output]
    started = time.monotonic()
    process = None
    peak = 0
    reason = "not_started"
    returncode = None
    payload = {
        "schema": "vmex.checked-run/1",
        "command": command,
        "cwd": str(Path.cwd()),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "resource_trace": str(trace_path),
        "timeout_seconds": args.timeout,
        "rss_limit_bytes": int(args.rss_gib * 2**30),
        "expected_outputs": [str(path) for path in expected],
        "scope": "execution outcome only; exit zero does not certify physics",
    }
    _atomic_json(record, {**payload, "status": "running", "complete": False})
    try:
        with (
            stdout_path.open("wb") as stdout,
            stderr_path.open("wb") as stderr,
            trace_path.open("w", encoding="utf-8") as trace,
        ):
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                start_new_session=os.name == "posix",
            )
            tracked = psutil.Process(process.pid)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                try:
                    members = [tracked, *tracked.children(recursive=True)]
                    rss = sum(member.memory_info().rss for member in members if member.is_running())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    rss = 0
                peak = max(peak, rss)
                trace.write(json.dumps({"elapsed_seconds": elapsed, "tree_rss_bytes": rss}) + "\n")
                trace.flush()
                if args.timeout and elapsed > args.timeout:
                    reason = "timeout"
                    _terminate_owned_group(process)
                    break
                if args.rss_gib and rss > args.rss_gib * 2**30:
                    reason = "memory_limit"
                    _terminate_owned_group(process)
                    break
                time.sleep(args.interval)
            returncode = process.wait()
            if reason == "not_started":
                reason = "completed" if returncode == 0 else "child_failed"
    except KeyboardInterrupt:
        reason = "interrupted"
        if process is not None:
            _terminate_owned_group(process)
            returncode = process.poll()
    except Exception as error:  # pragma: no cover - defensive outcome capture
        reason = "runner_error"
        payload["error"] = f"{type(error).__name__}: {error}"
        if process is not None:
            _terminate_owned_group(process)
            returncode = process.poll()
    finally:
        outputs = {str(path): path.is_file() for path in expected}
        payload.update(
            status=reason,
            complete=reason == "completed" and all(outputs.values()),
            returncode=returncode,
            child_signal=(-returncode if returncode is not None and returncode < 0 else None),
            elapsed_seconds=time.monotonic() - started,
            sampled_peak_tree_rss_bytes=peak,
            output_exists=outputs,
        )
        if reason == "completed" and not all(outputs.values()):
            payload["status"] = "missing_output"
        _atomic_json(record, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["complete"] else (124 if reason == "timeout" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
