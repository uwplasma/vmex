#!/usr/bin/env python3
"""Profile the QI optimization example end to end with the effort counters.

Each run executes ``examples/optimization/QI_optimization.py`` -- this tree's,
or with ``:baseline`` the version at ``--baseline-ref`` -- in a fresh process
inside an empty directory, so imports, compilation, the optimization, the final
solve and every output file fall inside the measured wall time.  The child runs
the script unmodified through ``runpy`` and only observes it: it keeps each
``VmecProblem`` the script builds and, after each SciPy ``least_squares`` call
on one, records that stage's evaluations, failed trials and counter deltas.
Final values are the lines the script prints.

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VMEX_COMPILATION_CACHE=disabled \
    PYTHONPATH=. python benchmarks/qi_optimization_profile.py \
        --runs full:baseline full --baseline-ref 9a6b5efc \
        --output benchmarks/qi_optimization_profile_office.json

``smoke`` sets ``VMEX_EXAMPLES_CI=1``.  A timing counts only when the load
average at both ends is within the machine's limit (plan section 7); a run past
``--timeout`` is killed and recorded as capped with the stages it finished.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _provenance import git_state  # noqa: E402
from single_stage_profile import _git, environment  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = "examples/optimization/QI_optimization.py"
NUMBER = r"([0-9.eE+-]+)"
FINAL = ("constructed QI", "aspect", "mean iota", "min |iota|", "mirror", "elongation")
CONSTANTS = ("nfp", "DATA", "MAX_MODES", "MAX_NFEV", "ASPECT_TARGET", "IOTA_FLOOR",
             "MIRROR_LIMIT", "ELONGATION_LIMIT")


def child(script: str, row_path: str) -> None:
    """Run ``script`` as ``__main__`` and write its stages to ``row_path``."""
    import runpy

    import numpy as np
    import scipy.optimize

    from vmex import optimize as opt
    from vmex.core import implicit as imp

    row: dict[str, object] = {"stages": []}
    problems: list[tuple[object, dict, float]] = []

    def counters(problem) -> dict:
        return dict(imp._SOLVE_STATS.get(problem.metadata.get("config")) or {})

    def write() -> None:
        Path(row_path).write_text(json.dumps(row, indent=1, default=str))

    build = opt.VmecProblem.from_tuples

    def observed_build(*args, **kwargs):
        started = time.perf_counter()
        problem = build(*args, **kwargs)
        problems.append((problem, counters(problem), time.perf_counter() - started))
        return problem

    least_squares = scipy.optimize.least_squares

    def observed_least_squares(fun, x0, *args, **kwargs):
        started = time.perf_counter()
        result = least_squares(fun, x0, *args, **kwargs)
        seconds = time.perf_counter() - started
        if problems:  # the example builds each stage's problem just before its solve
            problem, before, build_seconds = problems[-1]
            after = counters(problem)
            row["stages"].append({
                "dofs": int(np.size(x0)), "build_seconds": round(build_seconds, 1),
                "least_squares_seconds": round(seconds, 1), "nfev": int(result.nfev),
                "njev": None if result.njev is None else int(result.njev),
                "status": int(result.status), "cost": float(result.cost),
                "optimality": float(result.optimality),
                "failed_trials": problem.metadata["holder"].get("failed_trials"),
                "x": [float(value) for value in result.x],
                "counters": {key: value - before[key] if isinstance(value, (int, float))
                             and isinstance(before.get(key), (int, float)) else value
                             for key, value in after.items()},
            })
            problem.input_from_x(result.x).to_indata(
                str(Path(row_path).with_name(f"input.stage{len(row['stages'])}")))
            write()
        return result

    opt.VmecProblem.from_tuples = staticmethod(observed_build)
    scipy.optimize.least_squares = observed_least_squares
    sys.argv = [script]
    names = runpy.run_path(script, run_name="__main__")
    row["constants"] = {name: names.get(name) for name in CONSTANTS}
    row["min |iota| final"] = float(opt.EquilibriumReporter(
        ("min |iota|", opt.min_abs_iota, ".6f"))("final", names["final_equilibrium"])["min |iota|"])
    write()


def run(mode: str, timeout: float, baseline_commit: str | None, log_copy: Path) -> dict[str, object]:
    """Run the example in a fresh process, keep its stdout, and return its measured row."""
    env = {key: value for key, value in os.environ.items() if key != "VMEX_EXAMPLES_CI"}
    env["MPLBACKEND"] = "Agg"
    if mode == "smoke":
        env["VMEX_EXAMPLES_CI"] = "1"
    with tempfile.TemporaryDirectory(prefix="vmex-qi-profile-") as scratch:
        script = REPO / EXAMPLE
        if baseline_commit is not None:  # the example finds its deck beside itself
            script = Path(scratch) / "baseline" / EXAMPLE
            script.parent.mkdir(parents=True)
            script.write_text(_git("show", f"{baseline_commit}:{EXAMPLE}"))
            (script.parents[1] / "data").symlink_to(REPO / "examples" / "data")
        cwd = Path(scratch) / "run"
        cwd.mkdir()
        log_path, row_path = cwd / "stdout.log", cwd / "row.json"
        load_before, started, capped = os.getloadavg(), time.perf_counter(), False
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, __file__, "--child", str(script), str(row_path)],
                cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
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
        process.returncode = exit_code = os.waitstatus_to_exitcode(status)
        text = log_path.read_text(errors="replace")
        log_copy.write_text(text)
        for deck in cwd.glob("input.*"):  # stage decks and the example's own output
            shutil.copy(deck, log_copy.with_name(f"{log_copy.stem}.{deck.name}"))
        observed = json.loads(row_path.read_text()) if row_path.exists() else {}
    peak = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    final = {}
    for name in FINAL:
        found = re.findall(rf"\[final\][^\n]*?{re.escape(name)} = {NUMBER}", text)
        if found:
            final[name] = float(found[-1])
    validation = re.search(rf"QI total {NUMBER}; independent fine-grid validation {NUMBER}", text)
    if validation:
        final["fine-grid validation"] = float(validation.group(2))
    constants = observed.get("constants") or {}
    met = None
    if {"min |iota|", "mirror", "elongation"} <= set(final) and constants:
        met = {"iota": final["min |iota|"] >= constants["IOTA_FLOOR"],
               "mirror": final["mirror"] <= constants["MIRROR_LIMIT"],
               "elongation": final["elongation"] <= constants["ELONGATION_LIMIT"]}
    stages = observed.get("stages", [])
    row: dict[str, object] = {
        "source": "working tree" if baseline_commit is None else f"git show {baseline_commit[:8]}",
        "mode": mode, "wall_seconds": round(wall, 1), "peak_rss_gib": round(peak / 2**30, 2),
        "exit_code": exit_code, "capped_at_timeout": capped,
        "load_average_before": [round(value, 2) for value in load_before],
        "load_average_after": [round(value, 2) for value in os.getloadavg()],
        "constants": constants, "final": final, "constraints_met": met,
        "failed_trials": sum(stage["failed_trials"] or 0 for stage in stages) if stages else None,
        "stages": stages,
        "stage_lines": [line for line in text.splitlines() if line.startswith("[QI mode ")],
    }
    if exit_code != 0 or capped:
        row["stdout_tail"] = text.splitlines()[-15:]
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", default=["smoke"],
                        choices=["smoke", "full", "smoke:baseline", "full:baseline"])
    parser.add_argument("--baseline-ref", default="origin/main")
    parser.add_argument("--timeout", type=float, default=4 * 3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline_commit = _git("rev-parse", args.baseline_ref).strip()
    record: dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "_provenance": {
            **git_state(REPO), "baseline_commit": baseline_commit,
            "command": f"python benchmarks/qi_optimization_profile.py --runs {' '.join(args.runs)} "
                       f"--baseline-ref {baseline_commit[:8]} --timeout {args.timeout:g}",
        },
        "environment": {**environment(), "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
                        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
        "runs": [],
    }
    for spec in args.runs:
        mode, _, baseline = spec.partition(":")
        print(f"running {spec}", flush=True)
        row = run(mode, args.timeout, baseline_commit if baseline else None,
                  args.output.with_suffix(f".{len(record['runs'])}.log"))
        record["runs"].append(row)
        print(f"  {row['wall_seconds']} s, exit {row['exit_code']}, capped {row['capped_at_timeout']}, "
              f"final {row['final']}, failed trials {row['failed_trials']}", flush=True)
        args.output.write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(*sys.argv[2:4])
    else:
        raise SystemExit(main())
