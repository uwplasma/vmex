"""Run the complete native polish from the original input deck (R7.4).

One command, no cached physical state: the ordinary VMEX solve, the lift into
the native spline representation, and the declared continuation schedule
(the R3-R5 schedule that first reached the force target), all in the accurate
coefficient-first split-jet evaluator for the sparse stages.  Every stage runs
in a fresh process under ``run_checked.py`` with an isolated compilation cache
directory, so a first invocation measures a cold cache and a repeat with the
same ``--prefix`` measures a cache reload.  Only the final state receives the
independent point-force certificate; intermediate stages record the in-loop
quadrature force norm and a signed-Jacobian check.

Example::

    .venv/bin/python benchmarks/polish_recovery_cold.py --prefix artifacts/r7/cold-1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

INPUT = Path("examples/data/input.shaped_tokamak_pressure")
INPUT_SHA256 = "5b2740db6dbc91d6a1b43c283b95c959397a608566aeb54da7489203e1fe5102"

# (label, sparse-driver arguments).  Inserted-knot counts, radial order, and
# angular enrichment reproduce the historical continuation records
# polish_recovery_r4_* and r5_basis191_stationary.
SPARSE_SCHEDULE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("insert4", ("--insert-count", "4", "--max-steps", "2")),
    ("insert8", ("--insert-count", "8", "--max-steps", "2")),
    ("insert12a", ("--insert-count", "12", "--max-steps", "2")),
    ("order6", ("--reanchor", "--radial-order", "6", "--max-steps", "2")),
    *(
        (f"order6_insert12_{index}", ("--insert-count", "12", "--radial-order", "6", "--max-steps", "2"))
        for index in range(3)
    ),
    ("angular17", ("--maximum-m", "17", "--radial-order", "6", "--max-steps", "2")),
    ("angular19", ("--maximum-m", "19", "--radial-order", "6", "--max-steps", "2")),
    *(
        (f"m19_insert12_{index}", ("--insert-count", "12", "--radial-order", "6", "--max-steps", "2"))
        for index in range(6)
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--stage-timeout", type=float, default=900.0)
    parser.add_argument("--rss-gib", type=float, default=12.0)
    parser.add_argument("--stop-after", help="optional stage label to stop after (bounded probes)")
    args = parser.parse_args()
    if _sha256(INPUT) != INPUT_SHA256:
        raise ValueError("input deck hash mismatch; refusing a substituted problem")
    prefix = args.prefix
    prefix.mkdir(parents=True, exist_ok=True)
    cache = (prefix / "jax-cache").resolve()
    env = dict(os.environ, JAX_ENABLE_X64="1", VMEX_COMPILATION_CACHE_DIR=str(cache), JAX_COMPILATION_CACHE_DIR=str(cache))
    cache_existed = cache.exists() and any(cache.iterdir())
    python = sys.executable
    manifest = {
        "schema": "vmex.polish-recovery-cold/1",
        "input": str(INPUT),
        "input_sha256": INPUT_SHA256,
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "generator_sha256": _sha256(Path(__file__)),
        "compilation_cache": "reload" if cache_existed else "cold (empty dedicated directory)",
        "stages": [],
        "complete": False,
    }
    record = prefix / "cold.json"

    def run(label: str, command: list[str], outputs: list[Path]) -> None:
        watch = [python, "benchmarks/run_checked.py", "--record", str(prefix / f"{label}-run.json")]
        watch += ["--timeout", str(args.stage_timeout), "--rss-gib", str(args.rss_gib)]
        for output in outputs:
            watch += ["--expected-output", str(output)]
        started = time.perf_counter()
        completed = subprocess.run(watch + ["--"] + command, env=env, capture_output=True, text=True)
        outcome = json.loads((prefix / f"{label}-run.json").read_text())
        stage = {
            "label": label,
            "command": command,
            "wall_seconds": time.perf_counter() - started,
            "status": outcome.get("status"),
            "returncode": completed.returncode,
            "peak_rss_bytes": outcome.get("sampled_peak_tree_rss_bytes"),
        }
        manifest["stages"].append(stage)
        record.write_text(json.dumps(manifest, indent=2))
        if completed.returncode != 0 or outcome.get("status") != "completed":
            raise SystemExit(f"stage {label} failed: see {prefix / (label + '-run.json')}")

    wout_dir = prefix / "p0"
    wout = wout_dir / "wout_shaped_tokamak_pressure.nc"
    run("p0_ordinary", [str(Path(python).with_name("vmex")), str(INPUT), "--outdir", str(wout_dir)], [wout])
    state = prefix / "lift_state.npz"
    run(
        "lift_p3",
        [python, "benchmarks/polish_recovery_p3.py", "--wout", str(wout), "--max-steps", "4",
         "--output-json", str(prefix / "lift.json"), "--output-state", str(state)],
        [state],
    )
    for label, script, extra in (
        ("refine_bisect", "polish_recovery_refine.py", ()),
        ("angular15", "polish_recovery_angular.py", ("--maximum-m", "15")),
        ("refine_adaptive8", "polish_recovery_refine.py", ("--adaptive-count", "8")),
    ):
        output = prefix / f"{label}_state.npz"
        run(
            label,
            [python, f"benchmarks/{script}", "--input-state", str(state), *extra,
             "--output-json", str(prefix / f"{label}.json"), "--output-state", str(output)],
            [output],
        )
        state = output
    sparse_common = ("--stable-derivatives", "--linearization", "local-normal", "--linear-solver", "normal")
    for label, extra in SPARSE_SCHEDULE:
        output = prefix / f"{label}_state.npz"
        run(
            label,
            [python, "benchmarks/polish_recovery_sparse.py", "--input-state", str(state), *sparse_common, *extra,
             "--certificate", "none", "--output-json", str(prefix / f"{label}.json"), "--output-state", str(output)],
            [output],
        )
        state = output
        if args.stop_after == label:
            manifest["stopped_after"] = label
            record.write_text(json.dumps(manifest, indent=2))
            return
    final = prefix / "stationary_state.npz"
    run(
        "stationary_certified",
        [python, "benchmarks/polish_recovery_sparse.py", "--input-state", str(state), *sparse_common,
         "--stationarity-hessian", "exact", "--stationarity-refinement", "--max-steps", "4",
         "--certificate", "point", "--output-json", str(prefix / "stationary.json"), "--output-state", str(final)],
        [final],
    )
    result = json.loads((prefix / "stationary.json").read_text())["checks"]
    manifest["result"] = {
        key: result.get(key)
        for key in (
            "independent_epsilon_B",
            "independent_force_rms_N_per_m3",
            "final_projected_gradient_frobenius_relative",
            "stationarity_pass",
            "stationarity_certified_object",
            "final_gauge_residual_norm",
            "minimum_signed_jacobian",
        )
    }
    manifest["final_state_sha256"] = _sha256(final)
    manifest["total_wall_seconds"] = sum(stage["wall_seconds"] for stage in manifest["stages"])
    manifest["complete"] = True
    record.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["result"], indent=1), manifest["total_wall_seconds"])


if __name__ == "__main__":
    main()
