"""Batch non-axis parity scan using VMEC2000 stage-trace comparator.

This helper enumerates `input.*` files (typically from `simsopt/tests/test_files`)
and runs `vmec2000_exec_stage_trace_compare.py` with a short iteration budget.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import vmec_jax.api as vj


def _first_mismatch(stdout: str) -> tuple[str, float, float] | None:
    m = re.search(
        r"field=([A-Za-z0-9_]+).*?vmec2000=([0-9.eE+-]+).*?vmec_jax=([0-9.eE+-]+)",
        stdout,
        flags=re.S,
    )
    if not m:
        return None
    return m.group(1), float(m.group(2)), float(m.group(3))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs-root",
        type=str,
        default=str((Path(__file__).resolve().parents[3] / "simsopt" / "tests" / "test_files")),
    )
    p.add_argument("--glob", type=str, default="input.*")
    p.add_argument("--single-ns", type=int, default=13)
    p.add_argument("--max-iter", type=int, default=1)
    p.add_argument("--vmec-timeout", type=float, default=60.0)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--max-cases", type=int, default=8)
    p.add_argument("--allow-lasym", action="store_true")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    compare_script = repo_root / "tools" / "diagnostics" / "vmec2000_exec_stage_trace_compare.py"
    inputs_root = Path(args.inputs_root).expanduser().resolve()
    if not inputs_root.exists():
        raise SystemExit(f"Missing inputs root: {inputs_root}")

    chosen: list[Path] = []
    supported_profile_types = {"power_series", "two_power"}
    skipped_profile = 0
    for pth in sorted(inputs_root.glob(args.glob)):
        try:
            cfg, indata = vj.load_input(pth)
        except Exception:
            continue
        if int(cfg.ntor) <= 0 or int(cfg.nfp) <= 1:
            continue
        if (not bool(args.allow_lasym)) and bool(cfg.lasym):
            continue
        unsupported = False
        for key in ("PMASS_TYPE", "PIOTA_TYPE", "PCURR_TYPE"):
            v = indata.get(key, None)
            if v is None:
                continue
            if str(v).strip().lower() not in supported_profile_types:
                unsupported = True
                break
        if unsupported:
            skipped_profile += 1
            continue
        chosen.append(pth)
        if len(chosen) >= int(args.max_cases):
            break

    if not chosen:
        raise SystemExit("No non-axis input files matched filters.")

    print("non-axis parity batch:")
    print(f"  inputs_root={inputs_root}")
    print(f"  cases={len(chosen)} single_ns={args.single_ns} max_iter={args.max_iter}")
    if skipped_profile > 0:
        print(f"  skipped_unsupported_profiles={skipped_profile}")

    n_fail = 0
    for input_path in chosen:
        cmd = [
            sys.executable,
            str(compare_script),
            "--input",
            str(input_path),
            "--single-ns",
            str(int(args.single_ns)),
            "--max-iter",
            str(int(args.max_iter)),
            "--dump-level",
            "lite",
            "--vmec-timeout",
            str(float(args.vmec_timeout)),
            "--rtol",
            str(float(args.rtol)),
            "--atol",
            str(float(args.atol)),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
        dt = time.perf_counter() - t0
        out = proc.stdout + "\n" + proc.stderr
        mm = _first_mismatch(out)
        if mm is None and proc.returncode == 0:
            print(f"  PASS  {input_path.name:50s}  {dt:6.2f}s")
            continue
        n_fail += 1
        if mm is None:
            tail = " | ".join([ln for ln in out.splitlines()[-3:] if ln.strip()])
            print(f"  FAIL  {input_path.name:50s}  {dt:6.2f}s  {tail}")
        else:
            field, vmec_v, jax_v = mm
            print(
                f"  FAIL  {input_path.name:50s}  {dt:6.2f}s  "
                f"{field}: vmec={vmec_v:.3e} jax={jax_v:.3e}"
            )

    if n_fail > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
