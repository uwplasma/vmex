#!/usr/bin/env python
"""Scan one Boozer ``|B|`` mode and compare smooth vs legacy QI metrics.

This is a cheap diagnostic for QI objective noisiness.  It runs one VMEC-JAX
equilibrium, one Boozer transform, then perturbs a selected Boozer ``bmnc``
coefficient directly and evaluates:

- ``smooth_qi_total``: the differentiable objective used by vmec_jax.
- ``legacy_qi_total``: the non-differentiable Goodman-style branch/shuffle
  diagnostic used as a ranking gate.

Edit the variables below to change the input, resolution, scanned coefficient,
or scan range.  Outputs are written under ``results/`` and are ignored by git.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vmec_jax as vj
from vmec_jax._compat import enable_x64, jnp
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.optimization import rebuild_indata_with_resolution
from vmec_jax.quasi_isodynamic.legacy import legacy_qi_branch_shuffle_diagnostic_from_boozer_output
from vmec_jax.quasi_isodynamic import (
    quasi_isodynamic_residual_from_boozer_output,
    quasi_isodynamic_residual_from_state,
)


enable_x64(True)


# ---------------------------------------------------------------------------
# User-editable controls
# ---------------------------------------------------------------------------

INPUT_FILE = Path("examples/data/input.minimal_seed_nfp2")
OUTPUT_DIR = Path("results/omnigenity_compare/qi_boozer_mode_scan")

VMEC_MPOL = 6
VMEC_NTOR = 6
SOLVER_DEVICE = None  # None uses JAX default; set "cpu" or "gpu" to force one backend.

SURFACES = np.linspace(0.1, 1.0, 6)
QI_MBOZ = 10
QI_NBOZ = 10
QI_NPHI = 61
QI_NALPHA = 13
QI_N_BOUNCE = 21
QI_SOFTNESS = 2.0e-2
QI_BRANCH_WIDTH_SOFTNESS = 2.0e-2
QI_PROFILE_WEIGHT = 0.1
QI_SHUFFLE_PROFILE_WEIGHT = 1.0
QI_SHUFFLE_PROFILE_SOFTNESS = 2.0e-2
QI_PHIMIN = 0.0

# Set SCAN_MODE_INDEX to an integer to force a mode, or leave None to scan the
# largest non-axisymmetric cosine Boozer mode in the computed spectrum.
SCAN_MODE_INDEX = None
SCAN_SCALE_FACTORS = np.linspace(0.75, 1.25, 21)
LEGACY_NPHI_OUT = 401


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")
    print(f"wrote {path}")


def _timed(label: str, fn):
    t0 = time.perf_counter()
    value = fn()
    wall_s = time.perf_counter() - t0
    print(f"{label}: {wall_s:.2f} s")
    return value, wall_s


def _resolved_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _select_scan_mode_index(booz: dict, mode_index: int | None = None) -> int:
    bmnc = np.asarray(booz["bmnc_b"], dtype=float)
    xm = np.asarray(booz["ixm_b"], dtype=int)
    xn = np.asarray(booz["ixn_b"], dtype=int)
    if mode_index is not None:
        if int(mode_index) < 0 or int(mode_index) >= bmnc.shape[1]:
            raise ValueError(f"SCAN_MODE_INDEX={mode_index} outside [0, {bmnc.shape[1]})")
        return int(mode_index)

    amplitudes = np.linalg.norm(bmnc, axis=0)
    axis_mask = (xm == 0) & (xn == 0)
    amplitudes = np.where(axis_mask, -np.inf, amplitudes)
    selected = int(np.argmax(amplitudes))
    if not np.isfinite(amplitudes[selected]):
        raise ValueError("Could not find a non-axisymmetric Boozer mode to scan")
    return selected


def _scaled_boozer_mode(booz: dict, *, mode_index: int, scale: float) -> dict:
    out = dict(booz)
    bmnc = np.asarray(booz["bmnc_b"], dtype=float).copy()
    bmnc[:, int(mode_index)] *= float(scale)
    out["bmnc_b"] = bmnc
    return out


def _roughness(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return 0.0
    span = max(float(np.ptp(arr)), np.finfo(float).eps)
    return float(np.max(np.abs(np.diff(arr, n=2))) / span)


def evaluate_boozer_mode_scan(
    booz: dict,
    *,
    scales,
    mode_index: int | None = None,
    nfp: int | None = None,
) -> dict:
    """Return smooth/legacy QI metrics for a direct Boozer-coefficient scan."""
    selected = _select_scan_mode_index(booz, mode_index)
    xm = np.asarray(booz["ixm_b"], dtype=int)
    xn = np.asarray(booz["ixn_b"], dtype=int)
    bmnc = np.asarray(booz["bmnc_b"], dtype=float)
    nfp_local = int(nfp) if nfp is not None else int(np.asarray(booz["nfp_b"]).ravel()[0])

    rows = []
    for scale in scales:
        trial = _scaled_boozer_mode(booz, mode_index=selected, scale=float(scale))
        smooth = quasi_isodynamic_residual_from_boozer_output(
            trial,
            nfp=nfp_local,
            nphi=QI_NPHI,
            nalpha=QI_NALPHA,
            n_bounce=QI_N_BOUNCE,
            softness=QI_SOFTNESS,
            branch_width_softness=QI_BRANCH_WIDTH_SOFTNESS,
            profile_weight=QI_PROFILE_WEIGHT,
            shuffle_profile_weight=QI_SHUFFLE_PROFILE_WEIGHT,
            shuffle_profile_softness=QI_SHUFFLE_PROFILE_SOFTNESS,
            phimin=QI_PHIMIN,
        )
        legacy = legacy_qi_branch_shuffle_diagnostic_from_boozer_output(
            trial,
            nfp=nfp_local,
            nphi=QI_NPHI,
            nalpha=QI_NALPHA,
            n_bounce=QI_N_BOUNCE,
            nphi_out=LEGACY_NPHI_OUT,
            phimin=QI_PHIMIN,
        )
        rows.append(
            {
                "scale": float(scale),
                "smooth_qi_total": float(np.asarray(smooth["total"])),
                "legacy_qi_total": float(legacy["total"]),
            }
        )

    smooth_values = [row["smooth_qi_total"] for row in rows]
    legacy_values = [row["legacy_qi_total"] for row in rows]
    return {
        "mode_index": selected,
        "mode_m": int(xm[selected]),
        "mode_n": int(xn[selected]),
        "base_bmnc_l2": float(np.linalg.norm(bmnc[:, selected])),
        "rows": rows,
        "smooth_roughness": _roughness(smooth_values),
        "legacy_roughness": _roughness(legacy_values),
        "smooth_min_scale": float(rows[int(np.argmin(smooth_values))]["scale"]),
        "legacy_min_scale": float(rows[int(np.argmin(legacy_values))]["scale"]),
    }


def _run_vmec_jax_boozer() -> tuple[dict, float, float]:
    input_path = _resolved_path(INPUT_FILE)
    _cfg, indata = vj.load_config(str(input_path))
    indata = rebuild_indata_with_resolution(indata, mpol=VMEC_MPOL, ntor=VMEC_NTOR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rebuilt_input = OUTPUT_DIR / "input.vmec_jax_rebuilt"
    vj.write_indata(rebuilt_input, indata)
    print(f"Running vmec_jax on {rebuilt_input} ...")
    run, solve_wall_s = _timed(
        "vmec_jax fixed-boundary solve",
        lambda: vj.run_fixed_boundary(
            rebuilt_input,
            verbose=False,
            solver_device=SOLVER_DEVICE,
        ),
    )

    geom = eval_geom(run.state, run.static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
    flux = vj.flux_profiles_from_indata(run.indata, run.static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(run.static.s))
    qi, booz_wall_s = _timed(
        "shared Boozer/QI field",
        lambda: quasi_isodynamic_residual_from_state(
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=signgs,
            flux_local=flux,
            prof_local={"pressure": pressure},
            pressure_local=pressure,
            surfaces=SURFACES,
            mboz=QI_MBOZ,
            nboz=QI_NBOZ,
            nphi=QI_NPHI,
            nalpha=QI_NALPHA,
            n_bounce=QI_N_BOUNCE,
            softness=QI_SOFTNESS,
            branch_width_softness=QI_BRANCH_WIDTH_SOFTNESS,
            profile_weight=QI_PROFILE_WEIGHT,
            shuffle_profile_weight=QI_SHUFFLE_PROFILE_WEIGHT,
            shuffle_profile_softness=QI_SHUFFLE_PROFILE_SOFTNESS,
            phimin=QI_PHIMIN,
            jit_booz=False,
        ),
    )
    return qi["booz"], solve_wall_s, booz_wall_s


def _write_plot(summary: dict, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skipping plot: {type(exc).__name__}: {exc}")
        return

    rows = summary["rows"]
    scale = [row["scale"] for row in rows]
    smooth = [row["smooth_qi_total"] for row in rows]
    legacy = [row["legacy_qi_total"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax1.plot(scale, smooth, "o-", label="smooth QI objective")
    ax1.set_xlabel("Boozer coefficient scale factor")
    ax1.set_ylabel("smooth QI total")
    ax2 = ax1.twinx()
    ax2.plot(scale, legacy, "s--", color="tab:orange", label="legacy QI diagnostic")
    ax2.set_ylabel("legacy QI total")
    title = f"QI metric scan: mode m={summary['mode_m']}, n={summary['mode_n']}"
    ax1.set_title(title)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    booz, solve_wall_s, booz_wall_s = _run_vmec_jax_boozer()
    scan, scan_wall_s = _timed(
        "Boozer mode scan",
        lambda: evaluate_boozer_mode_scan(
            booz,
            scales=SCAN_SCALE_FACTORS,
            mode_index=SCAN_MODE_INDEX,
        ),
    )
    summary = {
        "input_file": str(INPUT_FILE),
        "vmec_resolution": {"mpol": VMEC_MPOL, "ntor": VMEC_NTOR},
        "surfaces": np.asarray(SURFACES, dtype=float),
        "qi_resolution": {
            "mboz": QI_MBOZ,
            "nboz": QI_NBOZ,
            "nphi": QI_NPHI,
            "nalpha": QI_NALPHA,
            "n_bounce": QI_N_BOUNCE,
            "phimin": QI_PHIMIN,
        },
        "wall_time_s": {
            "vmec_solve": solve_wall_s,
            "booz_qi": booz_wall_s,
            "scan": scan_wall_s,
        },
        **scan,
    }
    _write_json(OUTPUT_DIR / "qi_boozer_mode_scan.json", summary)
    _write_plot(summary, OUTPUT_DIR / "qi_boozer_mode_scan.png")
    print(
        "mode "
        f"(m={summary['mode_m']}, n={summary['mode_n']}) "
        f"smooth_min_scale={summary['smooth_min_scale']:.3f}, "
        f"legacy_min_scale={summary['legacy_min_scale']:.3f}, "
        f"smooth_roughness={summary['smooth_roughness']:.3e}, "
        f"legacy_roughness={summary['legacy_roughness']:.3e}"
    )


if __name__ == "__main__":
    main()
