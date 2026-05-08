#!/usr/bin/env python
"""Compare vmec_jax QI diagnostics against omnigenity_optimization.

This is a diagnostic script, not a polished gallery example.  It evaluates the
bundled NFP=2 QI seed with vmec_jax/booz_xform_jax.  Set
``RUN_REFERENCE_OMNIGENITY = True`` below to also evaluate the original
``qi_functions.py`` objectives on the same VMEC input when the local SIMSOPT and
omnigenity_optimization checkout are available.

The goal is to audit objective definitions before launching long sweeps:

- smooth vmec_jax QI residual variants,
- mirror-ratio and elongation penalties,
- optional ``phimin`` interval shifts,
- reference omnigenity residual / MirrorRatioPen / MaxElongationPen.

Edit the top-level variables below to change the input, resolution, or variants.
Outputs are written under ``results/`` and are ignored by git.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
from pathlib import Path
import queue as queue_module
import sys
import time

import numpy as np
from scipy.interpolate import UnivariateSpline

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vmec_jax as vj
from vmec_jax._compat import enable_x64, jnp
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.optimization import rebuild_indata_with_resolution
from vmec_jax.quasi_isodynamic import (
    lgradb_penalty_from_state,
    max_elongation_penalty_from_state,
    mirror_ratio_penalty_from_boozer_output,
    quasi_isodynamic_residual_from_boozer_output,
    quasi_isodynamic_residual_from_state,
)


enable_x64(True)


# ---------------------------------------------------------------------------
# User-editable diagnostic controls
# ---------------------------------------------------------------------------

OMNIGENITY_ROOT = Path("~/local/omnigenity_optimization").expanduser()
INPUT_FILE = Path("examples/data/input.nfp2_QI")
OUTPUT_DIR = Path("results/omnigenity_compare/qi_objective")

VMEC_MPOL = 6
VMEC_NTOR = 6
SOLVER_DEVICE = None  # None uses JAX default; set "cpu" or "gpu" to force one backend.

SURFACES = np.linspace(0.1, 1.0, 6)
QI_MBOZ = 10  # Increase to 18-40 for research-quality audits.
QI_NBOZ = 10  # Increase with QI_MBOZ when checking final high-mode designs.
QI_NPHI = 61  # Increase to 151+ for production comparisons.
QI_NALPHA = 13  # Increase to 31+ for production comparisons.
QI_N_BOUNCE = 21  # Increase to 51+ for production comparisons.
QI_SOFTNESS = 2.0e-2
QI_BRANCH_WIDTH_SOFTNESS = 2.0e-2
QI_ALIGNED_PROFILE_SOFTNESS = 2.0e-2
QI_ALIGNED_PROFILE_TRAP_LEVEL = 0.65
QI_ALIGNED_PROFILE_TRAP_SOFTNESS = 5.0e-2

MAX_MIRROR_RATIO = 0.21
MAX_ELONGATION = 8.0
LGRADB_THRESHOLD = 0.30

# The reference code chooses either 0 or pi/nfp depending on where the first
# well starts. Scan both here so disagreements are visible before optimization.
PHIMIN_FACTORS = (0.0, 0.5)  # multiply by 2*pi/nfp

QI_VARIANTS = (
    {
        "name": "width_only",
        "width_weight": 1.0,
        "branch_width_weight": 0.0,
        "profile_weight": 0.0,
        "aligned_profile_weight": 0.0,
    },
    {
        "name": "branch_width",
        "width_weight": 1.0,
        "branch_width_weight": 1.0,
        "profile_weight": 0.0,
        "aligned_profile_weight": 0.0,
    },
    {
        "name": "branch_width_aligned_profile",
        "width_weight": 1.0,
        "branch_width_weight": 1.0,
        "profile_weight": 0.0,
        "aligned_profile_weight": 0.5,
    },
)

RUN_REFERENCE_OMNIGENITY = False  # Set True for the slower SIMSOPT/omnigenity leg.
REFERENCE_NPHI_OUT = 401  # Increase to 2000 to match the original reference script.
REFERENCE_TIMEOUT_S = 300.0  # Child-process timeout for the optional reference leg.


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


def _run_vmec_jax_equilibrium() -> tuple[object, float]:
    input_path = _resolved_path(INPUT_FILE)
    _cfg, indata = vj.load_config(str(input_path))
    indata = rebuild_indata_with_resolution(indata, mpol=VMEC_MPOL, ntor=VMEC_NTOR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rebuilt_input = OUTPUT_DIR / "input.vmec_jax_rebuilt"
    vj.write_indata(rebuilt_input, indata)
    print(f"Running vmec_jax on {rebuilt_input} ...")
    return _timed(
        "vmec_jax fixed-boundary solve",
        lambda: vj.run_fixed_boundary(
            rebuilt_input,
            verbose=False,
            solver_device=SOLVER_DEVICE,
        ),
    )


def _slice_boozer_surface(booz: dict, surface_index: int) -> dict:
    out = dict(booz)
    for key in ("bmnc_b", "bmns_b", "iota_b", "s_b"):
        value = out.get(key)
        if value is not None:
            out[key] = value[surface_index : surface_index + 1]
    return out


def _legacy_get_branches(phi_b: np.ndarray, b_alpha: np.ndarray, b_level: float) -> tuple[float, float]:
    """Return the first and last bounce crossings used by qi_functions.py.

    This is intentionally NumPy/SciPy-only and non-differentiable.  It is a
    diagnostic copy of the legacy omnigenity metric so we can score the same
    booz_xform_jax spectrum without relying on the SIMSOPT Boozer subprocess.
    """
    diffs = b_alpha - float(b_level)
    diffsgn = diffs[:-1] * diffs[1:]
    inds = np.where(diffsgn < 0)[0]
    if b_level <= np.min(b_alpha):
        phi_min = float(phi_b[int(np.argmin(b_alpha))])
        return phi_min, phi_min
    if b_level >= np.max(b_alpha):
        return float(phi_b[0]), float(phi_b[-1])
    if len(inds) < 2:
        inds = np.where(diffsgn <= 0)[0]
        split = None
        for idx in range(1, len(inds)):
            if inds[idx] != inds[idx - 1] + 1:
                split = [inds[idx - 1], inds[-1]]
                break
        if split is not None:
            inds = np.asarray(split, dtype=int)
    if len(inds) > 2:
        inds = np.asarray([inds[0], inds[-1]], dtype=int)
    if len(inds) < 2:
        return float(phi_b[0]), float(phi_b[-1])

    def _crossing(ind: int, *, right_endpoint: bool) -> float:
        dy = b_alpha[ind] - b_alpha[ind + 1]
        dx = phi_b[ind] - phi_b[ind + 1]
        slope = dy / dx
        intercept = b_alpha[ind] - slope * phi_b[ind]
        if slope == 0.0:
            return float(phi_b[ind + int(right_endpoint)])
        return float((b_level - intercept) / slope)

    return _crossing(int(inds[0]), right_endpoint=False), _crossing(int(inds[1]), right_endpoint=True)


def _legacy_qi_residual_from_boozer_output(
    booz: dict,
    *,
    nfp: int,
    nphi: int,
    nalpha: int,
    n_bounce: int,
    nphi_out: int,
    phimin: float,
    weights=None,
) -> dict:
    """Evaluate the branch-squash/stretch/shuffle QI diagnostic on Boozer modes."""
    bmnc_b = np.asarray(booz["bmnc_b"], dtype=float)
    xm_b = np.asarray(booz["ixm_b"], dtype=float)
    xn_b = np.asarray(booz["ixn_b"], dtype=float)
    iota_b = np.asarray(booz["iota_b"], dtype=float)
    if bmnc_b.ndim != 2:
        raise ValueError(f"bmnc_b must have shape (nsurf, nmodes), got {bmnc_b.shape}")
    nsurf = bmnc_b.shape[0]
    if weights is None:
        weights_arr = np.ones(nsurf)
    else:
        weights_arr = np.sqrt(np.asarray(weights, dtype=float))
    if weights_arr.shape[0] != nsurf:
        raise ValueError("weights must have one value per Boozer surface")

    phimax = float(phimin) + 2.0 * np.pi / float(nfp)
    phi_1d = np.linspace(float(phimin), phimax, int(nphi))
    phis2d = np.tile(phi_1d, (int(nalpha), 1)).T
    b_levels = np.linspace(0.0, 1.0, int(n_bounce))
    out = np.zeros((nsurf, int(nalpha), int(nphi_out)))
    surface_totals = []

    for surf in range(nsurf):
        iota = float(iota_b[surf])
        theta_min = -iota * float(phimin)
        theta_max = theta_min + 2.0 * np.pi
        thetas2d = np.tile(np.linspace(theta_min, theta_max, int(nalpha)), (int(nphi), 1)) + iota * phis2d
        angle = thetas2d[:, :, None] * xm_b[None, None, :] - phis2d[:, :, None] * xn_b[None, None, :]
        bmag = np.sum(bmnc_b[surf][None, None, :] * np.cos(angle), axis=-1)
        bmin = float(np.min(bmag))
        bmax = float(np.max(bmag))
        denom = max(bmax - bmin, np.finfo(float).tiny)
        bnorm = (bmag - bmin) / denom

        bounce_widths = np.zeros((int(nalpha), int(n_bounce)))
        phi_crossings = np.zeros((int(nalpha), 2 * int(n_bounce) - 1))
        shuffled_crossings = np.zeros_like(phi_crossings)
        weights_alpha = np.zeros(int(nalpha))

        for ialpha in range(int(nalpha)):
            profile = np.array(bnorm[:, ialpha], copy=True)
            phi_profile = phis2d[:, ialpha]
            min_index = int(np.argmin(profile))

            left = np.array(profile[: min_index + 1], copy=True)
            phi_left = phi_profile[: min_index + 1]
            right = np.array(profile[min_index:], copy=True)
            phi_right = phi_profile[min_index:]

            left_max = int(np.argmax(left))
            left[:left_max] = left[left_max]
            for idx in range(len(left) - 1):
                if left[idx] <= left[idx + 1]:
                    stop = len(left) - 1
                    for jdx in range(idx + 1, len(left)):
                        if left[jdx] < left[idx]:
                            stop = jdx
                            break
                    left[idx:stop] = left[idx]

            right_max = int(np.argmax(right))
            right[right_max:] = right[right_max]
            for jdx in range(len(right) - 1, 1, -1):
                if right[jdx - 1] >= right[jdx]:
                    stop = 0
                    for kdx in range(jdx - 1, 1, -1):
                        if right[kdx] < right[jdx]:
                            stop = kdx
                            break
                    right[stop + 1 : jdx] = right[jdx]

            pmax = 50
            pmin = 15
            if len(left) > 1:
                x_left = (phi_left - phi_left[0]) / max(phi_left[-1] - phi_left[0], np.finfo(float).eps)
                left_half = x_left < 0.5
                f_left = left_half * (1.0 - left[0]) * ((np.cos(2 * np.pi * x_left) + 1.0) / 2.0) ** pmax
                f_left += (~left_half) * (-left[-1]) * ((np.cos(2 * np.pi * x_left) + 1.0) / 2.0) ** pmin
                left = left + f_left
            if len(right) > 1:
                x_right = (phi_right - phi_right[0]) / max(phi_right[-1] - phi_right[0], np.finfo(float).eps)
                right_half = x_right < 0.5
                f_right = right_half * (-right[0]) * ((np.cos(2 * np.pi * x_right) + 1.0) / 2.0) ** pmin
                f_right += (~right_half) * (1.0 - right[-1]) * ((np.cos(2 * np.pi * x_right) + 1.0) / 2.0) ** pmax
                right = right + f_right

            squashed = np.concatenate((left[:-1], right))
            diff = profile - squashed
            weights_alpha[ialpha] = (phimax - float(phimin)) / max(
                float(UnivariateSpline(phi_profile, diff * diff, k=1, s=0).integral(float(phimin), phimax)),
                np.finfo(float).eps,
            )

            for jlevel, level in enumerate(b_levels):
                phi_left_cross, phi_right_cross = _legacy_get_branches(phi_profile, squashed, float(level))
                bounce_widths[ialpha, jlevel] = phi_right_cross - phi_left_cross
                phi_crossings[ialpha, int(n_bounce) - jlevel - 1] = phi_left_cross
                phi_crossings[ialpha, int(n_bounce) + jlevel - 1] = phi_right_cross

        weights_alpha = weights_alpha / max(float(np.sum(weights_alpha)), np.finfo(float).eps)
        mean_widths = np.sum(bounce_widths * weights_alpha[:, None], axis=0)
        shuffled_levels = np.concatenate((np.flip(b_levels), b_levels[1:]))
        phi_eval = np.linspace(float(phimin), phimax, int(nphi_out))

        for ialpha in range(int(nalpha)):
            delta_widths = 0.5 * (bounce_widths[ialpha, :] - mean_widths)
            left_crossings = np.array(phi_crossings[ialpha, : int(n_bounce)], copy=True)
            right_crossings = np.array(phi_crossings[ialpha, int(n_bounce) - 1 :], copy=True)
            left_crossings += np.flip(delta_widths)
            right_crossings -= delta_widths
            for idx in range(int(n_bounce) - 1):
                if left_crossings[idx + 1] - left_crossings[idx] < 0:
                    right_crossings[-idx - 2] += left_crossings[idx] - left_crossings[idx + 1] + 1.0e-12
                    left_crossings[idx + 1] = left_crossings[idx] + 1.0e-12
                if right_crossings[-idx - 1] - right_crossings[-idx - 2] < 0:
                    left_crossings[idx + 1] += right_crossings[-idx - 1] - right_crossings[-idx - 2] - 1.0e-12
                    right_crossings[-idx - 2] = right_crossings[-idx - 1] - 1.0e-12
            shuffled_crossings[ialpha, : int(n_bounce)] = left_crossings
            shuffled_crossings[ialpha, int(n_bounce) - 1 :] = right_crossings
            for idx in range(1, shuffled_crossings.shape[1]):
                if shuffled_crossings[ialpha, idx] <= shuffled_crossings[ialpha, idx - 1]:
                    shuffled_crossings[ialpha, idx] = shuffled_crossings[ialpha, idx - 1] + 1.0e-12

            original = UnivariateSpline(phis2d[:, ialpha], bnorm[:, ialpha], k=1, s=0)
            shuffled = UnivariateSpline(shuffled_crossings[ialpha, :], shuffled_levels, k=1, s=0)
            out[surf, ialpha, :] = weights_arr[surf] * (shuffled(phi_eval) - original(phi_eval)) / np.sqrt(
                int(nphi_out)
            )

        out[surf, :, :] = out[surf, :, :] / np.sqrt(int(nalpha))
        surface_totals.append(float(np.dot(out[surf].ravel(), out[surf].ravel())))

    residuals = out.ravel()
    return {
        "residuals1d": residuals,
        "total": float(np.dot(residuals, residuals)),
        "surface_totals": surface_totals,
        "residual_size": int(residuals.size),
    }


def evaluate_vmec_jax() -> dict:
    run, solve_wall_s = _run_vmec_jax_equilibrium()
    wout = vj.write_wout_from_fixed_boundary_run(OUTPUT_DIR / "wout_vmec_jax.nc", run)

    geom = eval_geom(run.state, run.static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
    flux = vj.flux_profiles_from_indata(run.indata, run.static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(run.static.s))
    period = 2.0 * np.pi / int(run.static.cfg.nfp)
    print("Computing one shared vmec_jax Boozer transform for QI variants ...")
    base_qi, booz_wall_s = _timed(
        "vmec_jax shared Boozer/QI field",
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
            width_weight=1.0,
            branch_width_weight=0.0,
            branch_width_softness=QI_BRANCH_WIDTH_SOFTNESS,
            profile_weight=0.0,
            aligned_profile_weight=0.0,
            aligned_profile_softness=QI_ALIGNED_PROFILE_SOFTNESS,
            aligned_profile_trap_level=QI_ALIGNED_PROFILE_TRAP_LEVEL,
            aligned_profile_trap_softness=QI_ALIGNED_PROFILE_TRAP_SOFTNESS,
            phimin=0.0,
            jit_booz=False,
        ),
    )
    booz = base_qi["booz"]

    variants: list[dict] = []
    for phimin_factor in PHIMIN_FACTORS:
        phimin = float(phimin_factor) * period
        print(f"Evaluating legacy branch-shuffle QI diagnostic phimin={phimin:.6g} ...")
        legacy_qi, legacy_wall_s = _timed(
            f"legacy branch-shuffle phimin={phimin_factor}",
            lambda phimin=phimin: _legacy_qi_residual_from_boozer_output(
                booz,
                nfp=int(run.static.cfg.nfp),
                nphi=QI_NPHI,
                nalpha=QI_NALPHA,
                n_bounce=QI_N_BOUNCE,
                nphi_out=REFERENCE_NPHI_OUT,
                phimin=phimin,
            ),
        )
        variants.append(
            {
                "name": "legacy_branch_shuffle",
                "phimin_factor": float(phimin_factor),
                "phimin": phimin,
                "total": float(legacy_qi["total"]),
                "width_total": None,
                "branch_width_total": None,
                "profile_total": None,
                "aligned_profile_total": None,
                "surface_totals": legacy_qi["surface_totals"],
                "mirror_ratio": None,
                "wall_time_s": legacy_wall_s,
                "residual_size": int(legacy_qi["residual_size"]),
                "diagnostic_only": True,
            }
        )
        for variant in QI_VARIANTS:
            print(f"Evaluating vmec_jax QI variant {variant['name']} phimin={phimin:.6g} ...")
            qi, wall_s = _timed(
                f"vmec_jax qi {variant['name']} phimin={phimin_factor}",
                lambda variant=variant, phimin=phimin: quasi_isodynamic_residual_from_boozer_output(
                    booz,
                    nfp=int(run.static.cfg.nfp),
                    nphi=QI_NPHI,
                    nalpha=QI_NALPHA,
                    n_bounce=QI_N_BOUNCE,
                    softness=QI_SOFTNESS,
                    width_weight=variant["width_weight"],
                    branch_width_weight=variant["branch_width_weight"],
                    branch_width_softness=QI_BRANCH_WIDTH_SOFTNESS,
                    profile_weight=variant["profile_weight"],
                    aligned_profile_weight=variant["aligned_profile_weight"],
                    aligned_profile_softness=QI_ALIGNED_PROFILE_SOFTNESS,
                    aligned_profile_trap_level=QI_ALIGNED_PROFILE_TRAP_LEVEL,
                    aligned_profile_trap_softness=QI_ALIGNED_PROFILE_TRAP_SOFTNESS,
                    phimin=phimin,
                ),
            )
            mirror = mirror_ratio_penalty_from_boozer_output(
                _slice_boozer_surface(booz, 0),
                nfp=int(run.static.cfg.nfp),
                threshold=MAX_MIRROR_RATIO,
                ntheta=96,
                nphi=96,
            )
            variants.append(
                {
                    "name": variant["name"],
                    "phimin_factor": float(phimin_factor),
                    "phimin": phimin,
                    "total": float(np.asarray(qi["total"])),
                    "width_total": float(np.dot(np.asarray(qi["width_residuals1d"]), np.asarray(qi["width_residuals1d"]))),
                    "branch_width_total": float(
                        np.dot(
                            np.asarray(qi["branch_width_residuals1d"]),
                            np.asarray(qi["branch_width_residuals1d"]),
                        )
                    ),
                    "profile_total": float(
                        np.dot(np.asarray(qi["profile_residuals1d"]), np.asarray(qi["profile_residuals1d"]))
                    ),
                    "aligned_profile_total": float(
                        np.dot(
                            np.asarray(qi["aligned_profile_residuals1d"]),
                            np.asarray(qi["aligned_profile_residuals1d"]),
                        )
                    ),
                    "mirror_ratio": float(np.max(np.asarray(mirror["mirror_ratio"]))),
                    "wall_time_s": wall_s,
                    "residual_size": int(np.asarray(qi["residuals1d"]).size),
                    "diagnostic_only": False,
                }
            )

    elongation = max_elongation_penalty_from_state(
        state=run.state,
        static=run.static,
        threshold=MAX_ELONGATION,
        ntheta=48,
        nphi=16,
    )
    lgradb = lgradb_penalty_from_state(
        state=run.state,
        static=run.static,
        indata=run.indata,
        signgs=signgs,
        flux_local=flux,
        threshold=LGRADB_THRESHOLD,
        ntheta=9,
        nphi=7,
    )
    return {
        "backend": "vmec_jax",
        "input_file": INPUT_FILE,
        "wout_path": OUTPUT_DIR / "wout_vmec_jax.nc",
        "solve_wall_time_s": solve_wall_s,
        "aspect": float(np.asarray(wout.aspect)),
        "mean_iota": float(np.mean(np.asarray(wout.iotas)[1:])),
        "nfp": int(np.asarray(wout.nfp)),
        "booz_wall_time_s": booz_wall_s,
        "variants": variants,
        "max_elongation": float(np.asarray(elongation["max_elongation"])),
        "max_elongation_excess": float(np.asarray(elongation["penalty"])),
        "min_lgradb": float(np.min(np.asarray(lgradb["L_grad_B"]))),
        "lgradb_excess_max": max(0.0, float(np.max(np.asarray(lgradb["excess"])))),
    }


def _evaluate_reference_omnigenity_impl() -> dict:
    if not OMNIGENITY_ROOT.exists():
        return {
            "backend": "omnigenity_reference",
            "skipped": True,
            "reason": f"{OMNIGENITY_ROOT} does not exist",
        }

    sys.path.insert(0, str(OMNIGENITY_ROOT))
    try:
        from qi_functions import MaxElongationPen, MirrorRatioPen, QuasiIsodynamicResidual
        from simsopt.mhd import Vmec
        from simsopt.util import MpiPartition
    except Exception as exc:
        return {
            "backend": "omnigenity_reference",
            "skipped": True,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    reference_input = OMNIGENITY_ROOT / "inputs" / INPUT_FILE.name
    if not reference_input.exists():
        reference_input = _resolved_path(INPUT_FILE)
    print(f"Running SIMSOPT/omnigenity reference on {reference_input} ...")

    mpi = MpiPartition()
    vmec = Vmec(str(reference_input), mpi=mpi, verbose=False)
    vmec.indata.mpol = VMEC_MPOL
    vmec.indata.ntor = VMEC_NTOR
    qi, qi_wall_s = _timed(
        "reference QuasiIsodynamicResidual",
        lambda: QuasiIsodynamicResidual(
            vmec,
            snorms=SURFACES,
            nphi=QI_NPHI,
            nalpha=QI_NALPHA,
            nBj=QI_N_BOUNCE,
            mpol=QI_MBOZ,
            ntor=QI_NBOZ,
            nphi_out=REFERENCE_NPHI_OUT,
            arr_out=True,
        ),
    )
    mirror, mirror_wall_s = _timed("reference MirrorRatioPen", lambda: MirrorRatioPen(vmec, t=MAX_MIRROR_RATIO))
    elongation, elongation_wall_s = _timed("reference MaxElongationPen", lambda: MaxElongationPen(vmec, t=MAX_ELONGATION))
    with contextlib.suppress(Exception):
        if getattr(vmec, "output_file", None):
            Path(vmec.output_file).replace(OUTPUT_DIR / "wout_reference.nc")
    return {
        "backend": "omnigenity_reference",
        "input_file": reference_input,
        "qi_total": float(np.dot(np.asarray(qi, dtype=float), np.asarray(qi, dtype=float))),
        "qi_residual_size": int(np.asarray(qi).size),
        "mirror_penalty": float(mirror),
        "elongation_penalty": float(elongation),
        "aspect": float(vmec.aspect()),
        "mean_iota": float(vmec.mean_iota()),
        "timing_s": {
            "qi": qi_wall_s,
            "mirror": mirror_wall_s,
            "elongation": elongation_wall_s,
        },
    }


def _reference_omnigenity_worker(result_queue: mp.Queue) -> None:
    try:
        result_queue.put({"ok": True, "payload": _evaluate_reference_omnigenity_impl()})
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "payload": {
                    "backend": "omnigenity_reference",
                    "skipped": True,
                    "reason": f"{type(exc).__name__}: {exc}",
                },
            }
        )


def evaluate_reference_omnigenity() -> dict:
    if not RUN_REFERENCE_OMNIGENITY:
        return {"backend": "omnigenity_reference", "skipped": True}

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_reference_omnigenity_worker,
        args=(result_queue,),
        name="omnigenity_qi_reference",
    )
    process.start()
    process.join(REFERENCE_TIMEOUT_S)

    if process.is_alive():
        process.terminate()
        process.join(5.0)
        return {
            "backend": "omnigenity_reference",
            "skipped": True,
            "reason": f"timeout after {REFERENCE_TIMEOUT_S:.1f} s",
        }

    if process.exitcode != 0:
        return {
            "backend": "omnigenity_reference",
            "skipped": True,
            "reason": f"child process exited with code {process.exitcode}",
        }

    try:
        message = result_queue.get(timeout=5.0)
    except queue_module.Empty:
        return {
            "backend": "omnigenity_reference",
            "skipped": True,
            "reason": "child process produced no result",
        }

    if message.get("ok", False):
        return message["payload"]
    return message["payload"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vmec_jax_summary = evaluate_vmec_jax()
    reference_summary = evaluate_reference_omnigenity()
    summary = {
        "input_file": INPUT_FILE,
        "vmec_resolution": {"mpol": VMEC_MPOL, "ntor": VMEC_NTOR},
        "qi_grid": {
            "surfaces": SURFACES,
            "mboz": QI_MBOZ,
            "nboz": QI_NBOZ,
            "nphi": QI_NPHI,
            "nalpha": QI_NALPHA,
            "n_bounce": QI_N_BOUNCE,
            "softness": QI_SOFTNESS,
            "branch_width_softness": QI_BRANCH_WIDTH_SOFTNESS,
        },
        "vmec_jax": vmec_jax_summary,
        "omnigenity_reference": reference_summary,
    }
    _write_json(OUTPUT_DIR / "qi_objective_comparison.json", summary)

    print("\nvmec_jax QI variants:")
    for row in vmec_jax_summary["variants"]:
        mirror_text = "n/a" if row.get("mirror_ratio") is None else f"{row['mirror_ratio']:.4f}"
        print(
            f"  {row['name']:28s} phimin={row['phimin_factor']:.1f} "
            f"total={row['total']:.6e} mirror={mirror_text}"
        )
    if not reference_summary.get("skipped", False):
        print(
            "\nReference omnigenity: "
            f"QI={reference_summary['qi_total']:.6e}, "
            f"mirror_penalty={reference_summary['mirror_penalty']:.6e}, "
            f"elongation_penalty={reference_summary['elongation_penalty']:.6e}"
        )
    else:
        print(f"\nReference omnigenity skipped: {reference_summary.get('reason', 'disabled')}")


if __name__ == "__main__":
    main()
