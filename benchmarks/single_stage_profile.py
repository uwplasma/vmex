#!/usr/bin/env python3
"""Profile the two single-stage optimization examples end to end.

Each example runs in a fresh process inside an empty temporary directory, so
compilation, the optimization, the verification solve and every output file
fall inside the measured wall time.  Peak memory is the child's maximum
resident set size reported by ``os.wait4``.  Each example writes a
``*_summary.json`` with its trials, solves and final values against its
targets; this script copies that summary into the record unchanged.

    OMP_NUM_THREADS=4 VMEX_COMPILATION_CACHE=disabled \
    PYTHONPATH=.:<ESSOS checkout with uwplasma/ESSOS#58> \
    python benchmarks/single_stage_profile.py \
        --runs fixed:smoke:baseline fixed:smoke fixed:smoke:baseline fixed:smoke \
               free:smoke fixed:full \
        --baseline-ref f09288b3 --output benchmarks/single_stage_profile_m4.json

``smoke`` sets ``VMEX_EXAMPLES_CI=1``, the capped wiring check CI runs;
``full`` runs the shipped budget.  A ``:baseline`` run executes the example as
it was at ``--baseline-ref`` (extracted with ``git show`` next to this tree's
``examples/data``) against the same library, so alternating baseline and
candidate runs compare the examples under the same machine load.  Wall times
on a shared machine are diagnostic samples: compare only such pairs, each
recorded with its load average.  A run that exceeds ``--timeout`` is killed
and recorded as capped; a run that exits non-zero is recorded with its exit
status.  Neither is reported as a result it did not produce.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _provenance import assert_repo_vmex, git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "fixed": "single_stage_optimization",
    "free": "single_stage_free_boundary_optimization",
    "free_beta": "single_stage_free_boundary_optimization_finite_beta",
}
MODES = ("smoke", "full")
#: Final values both the current and the baseline examples print.
REPORTED = {
    "min |iota|": re.compile(r"Minimum \|iota\| = ([0-9.eE+-]+)"),
    "aspect": re.compile(r"Aspect ratio = ([0-9.eE+-]+)"),
    "B.n/B RMS percent": re.compile(r"B\.n/B: area-weighted RMS = ([0-9.eE+-]+)%"),
    "free-boundary final line": re.compile(r"(\[final\] QA = .*)"),
}


def parse_run(spec: str) -> tuple[str, str, bool]:
    """``example:mode`` or ``example:mode:baseline`` -> validated triple."""
    parts = spec.split(":")
    valid = (len(parts) in (2, 3) and parts[0] in EXAMPLES and parts[1] in MODES
             and (len(parts) == 2 or parts[2] == "baseline"))
    if not valid:
        raise argparse.ArgumentTypeError(
            f"run {spec!r} must be one of {sorted(EXAMPLES)} ':' one of {MODES}, "
            "optionally followed by ':baseline'")
    return parts[0], parts[1], len(parts) == 3


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _essos_commit() -> str | None:
    spec = importlib.util.find_spec("essos")
    if spec is None or spec.origin is None:
        return None
    probe = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(spec.origin).parent,
        capture_output=True, text=True, check=False)
    return probe.stdout.strip() or None


def environment() -> dict[str, object]:
    import vmex

    assert_repo_vmex(vmex.__file__, REPO)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "VMEX_COMPILATION_CACHE": os.environ.get("VMEX_COMPILATION_CACHE"),
        "versions": {
            "vmex": getattr(vmex, "__version__", None),
            **{name: _version(name) for name in ("jax", "jaxlib", "numpy", "scipy", "essos")},
            "essos_commit": _essos_commit(),
        },
    }


def run_example(example: str, mode: str, timeout: float,
                baseline_commit: str | None = None) -> dict[str, object]:
    """Run one example in a fresh process and return its measured row."""
    stem = EXAMPLES[example]
    relative = f"examples/optimization/{stem}.py"
    env = {key: value for key, value in os.environ.items() if key != "VMEX_EXAMPLES_CI"}
    env["MPLBACKEND"] = "Agg"
    if mode == "smoke":
        env["VMEX_EXAMPLES_CI"] = "1"
    with tempfile.TemporaryDirectory(prefix="vmex-single-stage-") as scratch:
        script = REPO / relative
        if baseline_commit is not None:
            # The example locates its deck relative to itself, so give the old
            # script this tree's examples/data beside it.
            tree = Path(scratch) / "baseline"
            script = tree / relative
            script.parent.mkdir(parents=True)
            script.write_text(_git("show", f"{baseline_commit}:{relative}"))
            (tree / "examples" / "data").symlink_to(REPO / "examples" / "data")
        scratch = str(Path(scratch) / "run")
        Path(scratch).mkdir()
        log_path = Path(scratch) / "stdout.log"
        load_before = os.getloadavg()
        started = time.perf_counter()
        capped = False
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, str(script)], cwd=scratch, env=env,
                stdout=log, stderr=subprocess.STDOUT)
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    break
                if time.perf_counter() - started > timeout:
                    process.kill()
                    capped = True
                    _, status, usage = os.wait4(process.pid, 0)
                    break
                time.sleep(1.0)
        wall = time.perf_counter() - started
        exit_code = os.waitstatus_to_exitcode(status)
        process.returncode = exit_code  # reaped by os.wait4 above
        # ru_maxrss is bytes on macOS and kibibytes on Linux.
        peak_bytes = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        summary_path = Path(scratch) / f"{stem}_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        text = log_path.read_text(errors="replace")
        lines = text.splitlines()
    reported = {}
    for name, pattern in REPORTED.items():
        match = pattern.search(text)
        if match is not None:
            reported[name] = (match.group(1) if name.endswith("line")
                              else float(match.group(1)))
    row: dict[str, object] = {
        "example": relative,
        "source": "working tree" if baseline_commit is None else f"git show {baseline_commit}",
        "mode": mode,
        "wall_seconds": round(wall, 1),
        "peak_rss_gib": round(peak_bytes / 2**30, 2),
        "exit_code": exit_code,
        "capped_at_timeout": capped,
        "load_average_before": [round(value, 2) for value in load_before],
        "load_average_after": [round(value, 2) for value in os.getloadavg()],
        "reported": reported,
        "summary": summary,
    }
    if summary is None:
        row["stdout_tail"] = lines[-15:]
    else:
        # The per-stage lines are the constrained method's trajectory.
        row["stage_lines"] = [line for line in lines if line.startswith("[stage ")]
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", type=parse_run,
                        default=[("fixed", "smoke", False), ("free", "smoke", False)],
                        help="example:mode[:baseline] specs, run in the order given")
    parser.add_argument("--baseline-ref", default="origin/main",
                        help="git revision whose examples the ':baseline' runs execute")
    parser.add_argument("--timeout", type=float, default=90 * 60,
                        help="seconds before a run is killed and recorded as capped")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline_commit = _git("rev-parse", args.baseline_ref).strip()

    specs = [f"{example}:{mode}" + (":baseline" if baseline else "")
             for example, mode, baseline in args.runs]
    record: dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "_provenance": {
            **git_state(REPO),
            "baseline_commit": baseline_commit,
            "command": "python benchmarks/single_stage_profile.py --runs " + " ".join(specs)
                       + f" --baseline-ref {baseline_commit[:8]} --timeout {args.timeout:g}",
        },
        "timing_note": (
            "Wall times are diagnostic samples on a shared machine, each with its load "
            "average; compare only adjacent baseline and candidate runs of this record."),
        "environment": environment(),
        "runs": [],
    }
    for (example, mode, baseline), spec in zip(args.runs, specs):
        print(f"running {spec}", flush=True)
        row = run_example(example, mode, args.timeout,
                          baseline_commit if baseline else None)
        record["runs"].append(row)
        summary = row["summary"] or {}
        print(f"  {row['wall_seconds']} s, {row['peak_rss_gib']} GiB, exit {row['exit_code']}, "
              f"met = {summary.get('met')}", flush=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
