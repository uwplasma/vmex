#!/usr/bin/env python
"""Print a privacy-preserving first-iteration diagnosis for one VMEC input deck.

Usage::

    python tools/diagnose_input.py path/to/input.case

This intentionally stops before a full equilibrium solve or NESTOR call.  Its
default output is safe to share: it omits the input path, every parsed input
value, resolution, profile name, axis value, coefficient, and force magnitude.
Use ``--details`` only for local diagnosis of a non-confidential deck.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np

from vmex.core.geometry import half_mesh_jacobian
from vmex.core.input import VmecInput
from vmex.core.solver import _geometry, _initial_state, evaluate_forces, prepare_runtime


def _named_arrays(value: Any, prefix: str = ""):
    if is_dataclass(value):
        for field in fields(value):
            name = f"{prefix}.{field.name}" if prefix else field.name
            yield from _named_arrays(getattr(value, field.name), name)
        return
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        for name in value._fields:
            child = f"{prefix}.{name}" if prefix else name
            yield from _named_arrays(getattr(value, name), child)
        return
    try:
        array = np.asarray(value)
    except Exception:
        return
    if np.issubdtype(array.dtype, np.number):
        yield prefix, array


def _nonfinite_names(value: Any, prefix: str) -> list[str]:
    return [name for name, array in _named_arrays(value, prefix) if not np.isfinite(array).all()]


def _range(value: Any) -> str:
    array = np.asarray(value, dtype=float)
    if not array.size:
        return "empty"
    if not np.isfinite(array).all():
        return "non-finite"
    return f"[{array.min():.6e}, {array.max():.6e}]"


def diagnose(path: Path, *, details: bool = False) -> int:
    inp = VmecInput.from_file(path)
    rt = prepare_runtime(inp)
    state = _initial_state(rt.setup)
    _, geometry = _geometry(state, rt)
    jacobian = half_mesh_jacobian(geometry, s=rt.setup.s_full)
    gc, residuals, diagnostics = evaluate_forces(state, rt)

    axis_supplied = any(
        np.any(np.asarray(value) != 0.0)
        for value in (inp.raxis_c, inp.raxis_s, inp.zaxis_c, inp.zaxis_s)
    )
    setup_bad = _nonfinite_names(rt.setup, "setup")
    raw_bad = _nonfinite_names(residuals, "residual")
    update_bad = (
        _nonfinite_names(gc, "force")
        + _nonfinite_names(diagnostics.preconditioned, "preconditioned")
        + _nonfinite_names(diagnostics.cache, "preconditioner_cache")
    )
    diagnostic_bad = _nonfinite_names(
        (diagnostics.wb, diagnostics.wp, diagnostics.r00, diagnostics.z00),
        "diagnostics",
    )
    force_bad = raw_bad + update_bad + diagnostic_bad
    fsq = (float(residuals.fsqr), float(residuals.fsqz), float(residuals.fsql))
    pre = diagnostics.preconditioned

    print("VMEX first-iteration diagnostic (shareable, input details redacted)")
    print(
        "runtime: "
        f"jax={jax.__version__} backend={jax.default_backend()} "
        f"x64={bool(jax.config.jax_enable_x64)} "
        f"VMEX_FAST_COMPILE={os.environ.get('VMEX_FAST_COMPILE', '<default>')}"
    )
    sqrt_g = np.asarray(jacobian.sqrt_g)[1:]
    jacobian_finite = np.isfinite(sqrt_g).all()
    jacobian_good = jacobian_finite and not bool(jacobian.jacobian_sign_changed)
    print(f"setup arrays finite: {'PASS' if not setup_bad else 'FAIL'}")
    print(f"initial Jacobian valid: {'PASS' if jacobian_good else 'FAIL'}")
    print(f"raw force residuals finite: {'PASS' if not raw_bad else 'FAIL'}")
    print(f"preconditioned update finite: {'PASS' if not update_bad else 'FAIL'}")
    print(f"other force diagnostics finite: {'PASS' if not diagnostic_bad else 'FAIL'}")

    if details:
        print("\nLOCAL DETAILS — do not share for a confidential deck")
        print(f"input: {path.resolve()}")
        print(
            "resolution: "
            f"nfp={inp.nfp} mpol={inp.mpol} ntor={inp.ntor} "
            f"ntheta={rt.resolution.ntheta} nzeta={rt.resolution.nzeta} "
            f"ns={rt.resolution.ns} lasym={inp.lasym} lfreeb={inp.lfreeb}"
        )
        print(
            "controls: "
            f"phiedge={inp.phiedge:.17g} delt={inp.delt:.17g} "
            f"gamma={inp.gamma:.17g} ncurr={inp.ncurr} curtor={inp.curtor:.17g} "
            f"precon_type={inp.precon_type!r} "
            f"prec2d_threshold={inp.prec2d_threshold:.3e}"
        )
        print(
            "profiles: "
            f"pmass={inp.pmass_type!r} pcurr={inp.pcurr_type!r} "
            f"piota={inp.piota_type!r} phips={_range(rt.setup.phips)} "
            f"lamscale={float(rt.setup.lamscale):.6e} mass={_range(rt.setup.mass)}"
        )
        print(
            "axis: "
            f"input={'supplied' if axis_supplied else 'missing -> inferred'} "
            f"R(0)={float(np.sum(np.asarray(rt.setup.raxis_c))):.6e} "
            f"Z-coeff-max={float(np.max(np.abs(np.asarray(rt.setup.zaxis_s)))):.6e}"
        )
        print(
            "jacobian: "
            f"sign_changed={bool(jacobian.jacobian_sign_changed)} "
            f"sqrt_g(interior)={_range(sqrt_g)} "
            f"zero_count={int(np.count_nonzero(sqrt_g == 0.0))}"
        )
        print(
            "residuals: "
            f"FSQR={fsq[0]:.6e} FSQZ={fsq[1]:.6e} FSQL={fsq[2]:.6e}; "
            f"pre=({float(pre.fsqr1):.6e}, {float(pre.fsqz1):.6e}, "
            f"{float(pre.fsql1):.6e})"
        )
        bad = setup_bad + force_bad
        print("nonfinite arrays: " + (", ".join(bad) if bad else "none"))

    if details:
        likely: list[str] = []
        if inp.phiedge == 0.0 or float(rt.setup.lamscale) == 0.0:
            likely.append(
                "zero effective toroidal flux: check PHIEDGE and APHI; "
                "the force/lambda normalizations are singular"
            )
        if bool(jacobian.jacobian_sign_changed):
            likely.append(
                "bad initial axis/boundary Jacobian: remove the supplied "
                "RAXIS_*/ZAXIS_* to trigger inference, or seed them from the "
                "VMEC2000 output"
            )
        if setup_bad:
            likely.append(
                "a parsed profile/setup array is already non-finite before iteration 1"
            )
        raw_finite = np.isfinite(np.asarray(fsq)).all()
        pre_finite = np.isfinite(np.asarray([
            pre.fsqr1, pre.fsqz1, pre.fsql1,
        ], dtype=float)).all() and not _nonfinite_names(gc, "force")
        if raw_finite and not pre_finite:
            likely.append(
                "raw force residuals are finite but the update is not: the 1D "
                "radial/lambda preconditioner is the first failing stage"
            )
        if inp.gamma == 1.0:
            likely.append("GAMMA=1 makes the printed MHD-energy expression singular")
        if not likely and force_bad:
            likely.append(
                "non-finite values first appear inside the "
                "geometry/field/preconditioner force pass"
            )
        assessment = "; ".join(likely) if likely else "first force pass is finite"
    elif setup_bad:
        assessment = "D01_SETUP_NONFINITE"
    elif not jacobian_good:
        assessment = "D02_INITIAL_JACOBIAN"
    elif raw_bad:
        assessment = "D03_RAW_FORCE_NONFINITE"
    elif update_bad:
        assessment = "D04_PRECONDITIONED_UPDATE_NONFINITE"
    elif diagnostic_bad:
        assessment = "D05_FORCE_DIAGNOSTIC_NONFINITE"
    else:
        assessment = "OK_FIRST_FORCE_PASS_FINITE"
    print(f"assessment: {assessment}")
    return 1 if setup_bad or force_bad or not jacobian_good else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="VMEC &INDATA or VMEC++ JSON deck")
    parser.add_argument(
        "--details", action="store_true",
        help="print input-derived values for local use only; never share for a confidential deck",
    )
    args = parser.parse_args()
    try:
        return diagnose(args.input, details=args.details)
    except Exception as exc:
        print(f"diagnostic failed: {type(exc).__name__}")
        if args.details:
            print(f"local detail: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
