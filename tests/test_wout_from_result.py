"""``wout_from_result`` must equal the long ``wout_from_state`` call exactly.

Fourteen examples spell out the same six arguments to ``wout_from_state``:
the three force residuals, the iteration count, the converged flag and, on
the free-boundary lane, ``vacuum_output``.  The helper exists so an example
does not have to teach that incantation.  These tests pin it field by field
against the form the examples use today, on both lanes, so the two cannot
drift apart.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("netCDF4")

import vmex as vj  # noqa: E402
from vmex.core.wout import wout_from_result, wout_from_state  # noqa: E402
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"


def _assert_same_wout(produced, expected) -> None:
    """Every field of the two datasets is identical, not merely close."""
    produced_fields = {f.name for f in dataclasses.fields(produced)}
    expected_fields = {f.name for f in dataclasses.fields(expected)}
    assert produced_fields == expected_fields
    differing = []
    for name in sorted(produced_fields):
        a = getattr(produced, name)
        b = getattr(expected, name)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                differing.append(name)
        elif a != b:
            differing.append(name)
    assert not differing, f"wout_from_result differs from wout_from_state in {differing}"


def _solve_fixed():
    inp = vj.VmecInput.from_file(DATA / "input.solovev")
    return inp, vj.solve_multigrid(inp, verbose=False)


def test_matches_the_six_argument_form_on_a_fixed_boundary_solve():
    inp, result = _solve_fixed()
    expected = wout_from_state(
        inp=inp, state=result.state,
        fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
        niter=int(result.iterations), converged=bool(result.converged))
    _assert_same_wout(wout_from_result(inp, result), expected)


def test_a_fixed_boundary_result_carries_no_vacuum_tables():
    """``result.vacuum`` is None off the free-boundary lane, so one call serves both."""
    _, result = _solve_fixed()
    assert result.vacuum is None
    assert wout_from_result.__module__ == "vmex.core.wout"


@pytest.mark.full  # a NESTOR solve, so nightly rather than per-PR
def test_matches_the_six_argument_form_on_a_free_boundary_solve():
    mgrid = DATA / "mgrid_cth_like_lasym_small.nc"
    if not mgrid.is_file():  # release asset, not tracked in git
        pytest.skip(f"{mgrid.name} is absent; fetch the reference-nc bundle")
    inp = vj.VmecInput.from_file(DATA / "input.cth_like_free_bdy_lasym_small")
    result = vj.solve_free_boundary(inp, mgrid_path=mgrid,
                                    error_on_no_convergence=False)
    assert result.vacuum is not None, "this lane must exercise vacuum_output"
    expected = wout_from_state(
        inp=inp, state=result.state,
        fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
        niter=int(result.iterations), converged=bool(result.converged),
        vacuum_output=result.vacuum)
    _assert_same_wout(wout_from_result(inp, result), expected)


def test_state_argument_exports_a_different_state():
    """``state=`` overrides the exported state; everything else still comes from the result."""
    inp, result = _solve_fixed()
    moved = dataclasses.replace(result.state, R_cos=result.state.R_cos * 1.01)
    produced = wout_from_result(inp, result, state=moved)
    expected = wout_from_state(
        inp=inp, state=moved,
        fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
        niter=int(result.iterations), converged=bool(result.converged))
    _assert_same_wout(produced, expected)
    assert not np.array_equal(np.asarray(produced.rmnc),
                              np.asarray(wout_from_result(inp, result).rmnc))


def test_overrides_win_over_the_values_taken_from_the_result():
    inp, result = _solve_fixed()
    produced = wout_from_result(inp, result, niter=4321, converged=False,
                                input_extension="override_case")
    assert int(produced.niter) == 4321
    assert produced.input_extension == "override_case"
    # converged=False is the VMEC2000 "more iterations needed" flag, not 0
    assert int(produced.ier_flag) == 2


def test_toroidal_flux_follows_fileout_for_a_nonuniform_aphi():
    """``phi`` is fileout.f's midpoint sum of the half-mesh ``phips``.

    ``phi(js) = signgs*twopi*hs*sum(phip(2:js))``.  With APHI = 0.5 0.5 the
    normalized flux is ``0.5 s + 0.5 s^2``, whose derivative is linear, so the
    midpoint sum is exact; a right-endpoint sum of the full-mesh ``phipf``
    overshoots by ``PHIEDGE * 0.5 / (ns - 1)`` (5 % at ns = 11).
    """
    from vmex.core.solver import _initial_state, prepare_runtime, resolution_from_input

    aphi = np.zeros(20)
    aphi[:2] = 0.5
    inp = dataclasses.replace(vj.VmecInput.from_file(DATA / "input.circular_tokamak"), aphi=aphi, ns_array=np.array([11]))
    state = _initial_state(prepare_runtime(inp, resolution_from_input(inp)).setup)
    wout = wout_from_state(inp=inp, state=state, fsqr=1.0, fsqz=1.0, fsql=1.0, converged=False)
    s = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(wout.phi, inp.phiedge * (0.5 * s + 0.5 * s**2), rtol=1e-13, atol=1e-15)
