"""Driver helpers for bundled example paths and lightweight I/O wrappers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


def example_paths(
    case: str,
    *,
    root: str | Path | None = None,
    package_file: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Return ``(input_path, wout_path)`` for a bundled example case."""

    if root is None:
        if package_file is None:
            package_file = __file__
        root_path = Path(package_file).resolve().parents[1]
    else:
        root_path = Path(root)
    data_dir = root_path / "examples" / "data"
    input_path = data_dir / f"input.{case}"
    wout_path = data_dir / f"wout_{case}_reference.nc"
    if not wout_path.exists():
        wout_path = data_dir / f"wout_{case}.nc"
    if not wout_path.exists():
        wout_path = None
    return input_path, wout_path


def load_example(
    case: str,
    *,
    root: str | Path | None = None,
    with_wout: bool = True,
    grid=None,
    example_data_type: Callable[..., Any],
    example_paths_func: Callable[..., tuple[Path, Path | None]],
    load_config_func: Callable[..., tuple[Any, Any]],
    free_boundary_static_inputs_func: Callable[..., tuple[Any, Any]],
    build_static_func: Callable[..., Any],
    read_wout_func: Callable[..., Any],
    state_from_wout_func: Callable[..., Any],
):
    """Load a bundled example case using driver-injected dependencies."""

    input_path, wout_path = example_paths_func(case, root=root)
    cfg, indata = load_config_func(str(input_path))
    fb_meta, fb_extcur = free_boundary_static_inputs_func(cfg, load_fields=False, strict=False)
    static = build_static_func(cfg, grid=grid, mgrid_metadata=fb_meta, free_boundary_extcur=fb_extcur)
    if with_wout and wout_path is not None:
        wout = read_wout_func(wout_path)
        state = state_from_wout_func(wout)
    else:
        wout = None
        state = None
    return example_data_type(
        input_path=input_path,
        wout_path=wout_path,
        cfg=cfg,
        indata=indata,
        static=static,
        wout=wout,
        state=state,
    )


def load_input(path: str | Path, *, load_config_func: Callable[..., tuple[Any, Any]]):
    """Convenience wrapper around the configured input loader."""

    return load_config_func(str(path))


def load_wout(path: str | Path, *, read_wout_func: Callable[..., Any]):
    """Convenience wrapper around the configured ``wout`` reader."""

    return read_wout_func(path)


def save_npz(path: str | Path, **arrays) -> Path:
    """Save arrays into a NumPy ``.npz`` file and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def print_fixed_boundary_intro(
    *,
    input_path: str | Path,
    cfg: Any,
    solver: str,
    use_initial_guess: bool,
    max_iter: int,
    step_size: float,
    history_size: int,
    print_func: Callable[..., Any] = print,
) -> None:
    """Print the concise vmec_jax fixed-boundary driver banner."""

    mode = "initial guess" if bool(use_initial_guess) else f"{solver} solve"
    print_func(f"[vmec_jax] fixed-boundary run ({mode})", flush=True)
    print_func(f"[vmec_jax] input={input_path}", flush=True)
    print_func(
        f"[vmec_jax] ns={cfg.ns} mpol={cfg.mpol} ntor={cfg.ntor} nfp={cfg.nfp}",
        flush=True,
    )
    if not bool(use_initial_guess):
        print_func(
            f"[vmec_jax] max_iter={int(max_iter)} step_size={float(step_size)} "
            f"history_size={int(history_size)}",
            flush=True,
        )


def print_vmec2000_run_header(
    *,
    input_path: str | Path,
    version: str = "vmec_jax",
    now: datetime | None = None,
    print_func: Callable[..., Any] = print,
) -> None:
    """Print the VMEC2000-style run header used by interactive CLI solves."""

    now = datetime.now() if now is None else now
    date_str = now.strftime("%b %d,%Y")
    time_str = now.strftime("%H:%M:%S")
    input_name = Path(input_path).name.upper()
    print_func(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -", flush=True)
    print_func("  SEQ =    1 TIME SLICE  0.0000E+00", flush=True)
    print_func(f"  PROCESSING {input_name}", flush=True)
    print_func(f"  THIS IS PARVMEC (PARALLEL VMEC), VERSION {version}", flush=True)
    print_func("  Lambda: Full Radial Mesh. L-Force: hybrid full/half.", flush=True)
    print_func("", flush=True)
    print_func(f"  COMPUTER:    OS:    RELEASE:   DATE = {date_str}  TIME = {time_str}", flush=True)
    print_func("", flush=True)


def print_vmec2000_run_summary(
    *,
    input_path: str | Path,
    result: Any,
    niter_stage: int,
    total_time: float,
    print_func: Callable[..., Any] = print,
) -> None:
    """Print the VMEC2000-style final summary for a fixed-boundary solve."""

    converged = bool(getattr(result, "diagnostics", {}).get("converged", False))
    if not converged and int(getattr(result, "n_iter", 0)) >= int(niter_stage):
        print_func(" Try increasing NITER or PRE_NITER if the preconditioner is on.", flush=True)
    print_func("", flush=True)
    if converged:
        print_func(" EXECUTION TERMINATED NORMALLY", flush=True)
    else:
        print_func(" EXECUTION FINISHED WITHOUT REQUESTED CONVERGENCE", flush=True)
    print_func("", flush=True)
    case_name = Path(input_path).name
    if case_name.startswith("input."):
        case_name = case_name.split("input.", 1)[-1]
    print_func(f" FILE : {case_name}", flush=True)
    ijacob = int(getattr(result, "diagnostics", {}).get("ijacob", 0))
    print_func(f" NUMBER OF JACOBIAN RESETS = {ijacob:4d}", flush=True)
    total_time = max(0.0, float(total_time))
    print_func("", flush=True)
    print_func(f"    TOTAL COMPUTATIONAL TIME (SEC)         {total_time:8.2f}", flush=True)
    print_func("    TIME TO INPUT/OUTPUT                   0.00", flush=True)
    print_func("       READ IN DATA                        0.00", flush=True)
    print_func("       WRITE OUT DATA TO WOUT              0.00", flush=True)
    print_func(f"    TIME IN FUNCT3D                        {total_time:8.2f}", flush=True)
    print_func("       BCOVAR FIELDS                       0.00", flush=True)
    print_func("       FOURIER TRANSFORM                   0.00", flush=True)
    print_func("       INVERSE FOURIER TRANSFORM           0.00", flush=True)
    print_func("       FORCES AND SYMMETRIZE               0.00", flush=True)
    print_func("       RESIDUE                             0.00", flush=True)
    print_func("       EQFORCE                             0.00", flush=True)
    print_func("", flush=True)
    print_func(" NO. OF PROCS:     1", flush=True)
    print_func(" PARVMEC     :     T", flush=True)
    print_func(" LPRECOND    :     F", flush=True)
    print_func(" LV3FITCALL  :     F", flush=True)
