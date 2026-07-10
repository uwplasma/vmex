"""Free-boundary tests: NESTOR operator A/B vs legacy + end-to-end golden run.

A/B lane
--------
``vmec_jax.core.vacuum`` is a cleaned port of the legacy parity-proven JAX
NESTOR operator (``solvers/free_boundary/jax_nestor_operator.py`` host
tables + ``solvers/free_boundary/adjoint/vmec_nestor.py`` assembly).  The
A/B tests build identical boundary + external-field inputs from the golden
free-boundary deck and require the core operator to reproduce the legacy
operator's ``potvac / rhs / mode_matrix / grpmn / gsource`` to 1e-10
(max-normalized; measured agreement is ~5e-12, pure float-reassociation).

End-to-end lane
---------------
The golden VMEC2000 run of ``input.cth_like_free_bdy_lasym_small`` is only
*partially converged* (NITER=1000 exhausted with fsq ~ 1e-1 and 17 Jacobian
resets; the deck header calls it a bounded LASYM smoke fixture).  Past
vacuum turn-on the trajectory is chaotic, so the golden comparison is
structural + coarse:

- the vacuum activation banner appears and the turn-on iteration matches
  the golden stdout (53) to within a few iterations,
- the first-call vacuum diagnostics (``2*pi*a*-BPOL``, ``TOROIDAL
  CURRENT``, ``R*BTOR``) match the golden print block,
- the final ``fsqr`` is within 10x of the golden stdout's final value,
- the edge ``rmnc/zmns`` rows agree with the golden wout to a few percent
  of the dominant coefficient (per-coefficient rtol is meaningless between
  two chaotic unconverged trajectories; measured: ~1% for R, ~4% for Z).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vmec_jax.core import freeboundary as FB  # noqa: E402
from vmec_jax.core import vacuum as V  # noqa: E402
from vmec_jax.core.errors import MgridNotFoundError  # noqa: E402
from vmec_jax.core.input import VmecInput  # noqa: E402
from vmec_jax.core.mgrid import MgridField, read_mgrid  # noqa: E402
from vmec_jax.core.solver import (  # noqa: E402
    _initial_state, prepare_runtime, resolution_from_input,
)

REPO = Path(__file__).resolve().parents[2]
DECK = REPO / "examples" / "data" / "input.cth_like_free_bdy_lasym_small"
MGRID = REPO / "examples" / "data" / "mgrid_cth_like_lasym_small.nc"
CASE = "cth_like_free_bdy_lasym_small"


# ---------------------------------------------------------------------------
# Shared fixtures: identical inputs for the core and legacy operators
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ab_inputs():
    """Boundary geometry + bexni from the golden deck's initial state."""
    inp = VmecInput.from_file(DECK)
    res = resolution_from_input(inp)
    rt = prepare_runtime(inp, res)
    state = _initial_state(rt.setup)
    basis = V.vacuum_basis(
        mf=int(inp.mpol) + 1, nf=int(inp.ntor),
        ntheta3=res.ntheta3, nzeta=res.nzeta, nfp=res.nfp,
        lasym=res.lasym, wint=np.asarray(rt.trig.wint),
    )
    rmnc, zmns, rmns, zmnc = FB._edge_fourier(state, rt)
    boundary = FB.boundary_from_coefficients(
        rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc, modes=rt.modes, basis=basis
    )
    ctor, _rbtor, axis_r, axis_z, _bsq3, _pres = FB._vacuum_scalars(state, rt)
    field = MgridField.from_mgrid_data(
        read_mgrid(MGRID),
        extcur=np.asarray(inp.extcur, dtype=float)[: read_mgrid(MGRID).nextcur],
    )
    phi = (np.asarray(basis.zeta) * basis.onp).reshape(basis.ntheta3, basis.nzeta)
    br_c, bp_c, bz_c = field.b_cyl(np.asarray(boundary.R), phi, np.asarray(boundary.Z))
    br_a, bp_a, bz_a = FB.axis_current_field(
        R=np.asarray(boundary.R), Z=np.asarray(boundary.Z),
        axis_r=np.asarray(axis_r), axis_z=np.asarray(axis_z),
        nfp=res.nfp, plascur=float(ctor),
    )
    ext = FB.external_field_channels(
        boundary=boundary,
        br=np.asarray(br_c) + br_a, bp=np.asarray(bp_c) + bp_a,
        bz=np.asarray(bz_c) + bz_a,
        basis=basis, signgs=int(rt.setup.signgs),
    )
    return dict(inp=inp, res=res, rt=rt, basis=basis, boundary=boundary, ext=ext)


def _rel_max(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def test_vacuum_basis_matches_legacy(ab_inputs):
    """Core precal tables reproduce the legacy builders exactly (or to eps)."""
    from vmec_jax.solvers.free_boundary import jax_nestor_operator as L

    basis: V.VacuumBasis = ab_inputs["basis"]
    lb = L.build_vmec_mode_basis(
        ntheta=basis.ntheta3, nzeta=basis.nzeta, nfp=basis.nfp,
        mf=basis.mf, nf=basis.nf, lasym=basis.lasym, wint=basis.wint,
    )
    for key in ("xmpot", "n_raw", "imirr", "imirr_full"):
        np.testing.assert_array_equal(np.asarray(getattr(basis, key)), np.asarray(lb[key]))
    for key in ("sin_phase", "cos_phase", "sinmni", "cosmni", "cmns", "theta", "zeta"):
        np.testing.assert_array_equal(np.asarray(getattr(basis, key)), np.asarray(lb[key]))
    tables = L.ensure_vmec_nonsingular_kernel_tables(
        basis=lb, nv=basis.nzeta, nvper=basis.nvper
    )
    for key in ("tanu", "tanv", "cosuv", "sinuv", "cosper", "sinper"):
        np.testing.assert_array_equal(np.asarray(getattr(basis, key)), np.asarray(tables[key]))
    # cosui/sinui/cosv_tab/sinv_tab: vectorized vs per-m Fortran-style loop —
    # identical up to one float reassociation.
    for key in ("cosv_tab", "sinv_tab", "cosui", "sinui"):
        np.testing.assert_allclose(
            np.asarray(getattr(basis, key)), np.asarray(tables[key]), rtol=0, atol=1e-14
        )


def test_nestor_operator_matches_legacy(ab_inputs, monkeypatch):
    """Core full NESTOR update == legacy JAX NESTOR operator (rtol 1e-10)."""
    from vmec_jax.solvers.free_boundary import jax_nestor_operator as L

    # The legacy jit wrapper returns a python str inside its output dict,
    # which current JAX rejects at lowering time; run the legacy operator
    # in its non-jitted lane (identical math).
    monkeypatch.setenv("VMEC_JAX_FREEB_JAX_NESTOR_JIT_OPERATOR", "0")

    basis: V.VacuumBasis = ab_inputs["basis"]
    boundary = ab_inputs["boundary"]
    rt = ab_inputs["rt"]
    bexni = np.asarray(ab_inputs["ext"]["bexni"], dtype=float)
    signgs = int(rt.setup.signgs)

    solver = V.make_vacuum_solver(basis, signgs=signgs)
    potvac, mode_matrix, bvec_nonsing, rhs, gsource, grpmn = solver.full(
        boundary, jnp.asarray(bexni)
    )

    lb = basis.as_legacy_basis()
    tables = L.ensure_vmec_nonsingular_kernel_tables(
        basis=lb, nv=basis.nzeta, nvper=basis.nvper
    )

    class _Sample:
        pass

    sample = _Sample()
    for name in ("R", "Z", "Ru", "Zu", "Rv", "Zv", "ruu", "ruv", "rvv", "zuu", "zuv", "zvv"):
        setattr(sample, name, np.asarray(getattr(boundary, name)))

    (_phi, potvac_l, rhs_l, mm_l, grpmn_l, gsource_l, _jit, _hit) = (
        L.solve_vmec_like_mode_with_jax_nestor_operator(
            sample=sample, basis=lb, tables=tables, bexni=bexni,
            signgs=signgs, nvper=basis.nvper, include_analytic=True,
        )
    )

    assert _rel_max(potvac, potvac_l) < 1e-10
    assert _rel_max(rhs, rhs_l) < 1e-10
    assert _rel_max(mode_matrix, mm_l) < 1e-10
    assert _rel_max(grpmn, grpmn_l) < 1e-10
    assert _rel_max(gsource, gsource_l) < 1e-10

    # scalpot.f skip branch: cached matrix + cached non-singular source with
    # a freshly recomputed analytic source must reproduce the full solve
    # when the geometry/source are unchanged.
    potvac_skip, rhs_skip = solver.skip(
        boundary, jnp.asarray(bexni), bvec_nonsing, mode_matrix
    )
    np.testing.assert_allclose(np.asarray(potvac_skip), np.asarray(potvac), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(rhs_skip), np.asarray(rhs), rtol=1e-12, atol=1e-14)


def test_vacuum_first_call_diagnostics(ab_inputs):
    """vacuum.f first-call print block values against the golden stdout."""
    basis: V.VacuumBasis = ab_inputs["basis"]
    boundary = ab_inputs["boundary"]
    rt = ab_inputs["rt"]
    ext = ab_inputs["ext"]
    solver = V.make_vacuum_solver(basis, signgs=int(rt.setup.signgs))
    potvac, *_ = solver.full(boundary, jnp.asarray(ext["bexni"]))
    _bsq, bsubu_s, bsubv_s, _bu, _bv = V.vacuum_channels(
        basis=basis, potvac=potvac,
        bexu=jnp.asarray(ext["bexu"]), bexv=jnp.asarray(ext["bexv"]),
        guu=jnp.asarray(ext["guu"]), guv=jnp.asarray(ext["guv"]),
        gvv=jnp.asarray(ext["gvv"]),
    )
    wint2 = np.asarray(basis.wint).reshape(basis.ntheta3, basis.nzeta)
    fac = 1.0e-6 / FB.MU0
    bsubuvac = float(np.sum(np.asarray(bsubu_s) * wint2)) * float(rt.setup.signgs) * 2 * np.pi
    bsubvvac = float(np.sum(np.asarray(bsubv_s) * wint2))
    # Golden stdout: 4.69E-02, 4.32E-02, -4.59E-01 (first vacuum call).
    assert bsubuvac * fac == pytest.approx(4.69e-2, abs=3e-3)
    assert bsubvvac == pytest.approx(-4.59e-1, abs=5e-3)
    state = _initial_state(ab_inputs["rt"].setup)
    ctor, *_rest = FB._vacuum_scalars(state, ab_inputs["rt"])
    assert float(ctor) * fac == pytest.approx(4.32e-2, abs=2e-4)


# ---------------------------------------------------------------------------
# End-to-end golden run
# ---------------------------------------------------------------------------


def _golden_stdout(golden_dir: Path) -> str:
    path = golden_dir / CASE / "stdout.txt"
    if not path.is_file():
        pytest.skip(f"golden stdout missing: {path}")
    return path.read_text()


@pytest.mark.full
def test_free_boundary_end_to_end_golden(golden_dir):
    """Run the golden free-boundary deck with the core solver (structural).

    The golden run is unconverged (NITER exhausted, fsq ~ 1e-1, chaotic past
    turn-on), so this checks iteration *structure* and coarse values — see
    the module docstring for the tolerance rationale.
    """
    stdout_g = _golden_stdout(golden_dir)
    m = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)\s+ITERATIONS", stdout_g)
    assert m, "golden stdout lacks the vacuum banner"
    golden_turnon = int(m.group(1))
    final_line = [ln for ln in stdout_g.splitlines() if re.match(r"^\s*\d+\s+[\d.E+-]+", ln)][-1]
    golden_final_fsqr = float(final_line.split()[1])

    inp = VmecInput.from_file(DECK)
    lines: list[str] = []
    result = FB.solve_free_boundary(
        inp, mgrid_path=MGRID, verbose=True,
        emit=lambda *a, **k: lines.append(a[0] if a else ""),
        error_on_no_convergence=False,
    )
    out = "".join(lines)

    m2 = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)\s+ITERATIONS", out)
    assert m2, "core run never activated the vacuum field"
    turnon = int(m2.group(1))
    # Same fixed-boundary physics up to activation: expect the same turn-on
    # iteration (53 in the golden stdout) modulo float-order jitter.
    assert abs(turnon - golden_turnon) <= 5

    assert "In VACUUM" in out
    assert result.iterations == int(inp.niter_array[0])
    assert not result.converged  # golden doesn't converge either
    # Final fsqr within 10x of the golden stdout's final printed value.
    assert result.fsqr <= 10.0 * golden_final_fsqr

    # Edge rmnc/zmns vs golden wout: a few percent of the dominant
    # coefficient (both trajectories are unconverged; documented above).
    netCDF4 = pytest.importorskip("netCDF4")
    wout = golden_dir / CASE / f"wout_{CASE}.nc"
    with netCDF4.Dataset(wout) as ds:
        g_rmnc = np.asarray(ds.variables["rmnc"][:])[-1]
        g_zmns = np.asarray(ds.variables["zmns"][:])[-1]
        g_xm = np.asarray(ds.variables["xm"][:]).astype(int)
        g_xn = np.asarray(ds.variables["xn"][:]).astype(int)
    mine = {
        (int(m_), int(n_)): k for k, (m_, n_) in enumerate(zip(result.xm, result.xn))
    }
    idx = np.asarray([mine[(m_, n_)] for m_, n_ in zip(g_xm, g_xn)])
    r_err = np.abs(result.rmnc[-1][idx] - g_rmnc).max() / np.abs(g_rmnc).max()
    z_err = np.abs(result.zmns[-1][idx] - g_zmns).max() / np.abs(g_zmns).max()
    assert r_err < 0.05, f"edge rmnc scale-relative error {r_err}"
    assert z_err < 0.08, f"edge zmns scale-relative error {z_err}"


# ---------------------------------------------------------------------------
# Missing-mgrid fallback policy
# ---------------------------------------------------------------------------


def test_missing_mgrid_raises(tmp_path):
    """solve_free_boundary surfaces MgridNotFoundError for a missing file."""
    inp = VmecInput.from_file(DECK)
    with pytest.raises(MgridNotFoundError):
        FB.solve_free_boundary(inp, mgrid_path=tmp_path / "mgrid_missing.nc")


def test_cli_missing_mgrid_fallback_warns(tmp_path):
    """CLI policy: missing mgrid -> fixed-boundary fallback warning (VMEC2000)."""
    import types

    from vmec_jax.core.cli import _free_boundary_plan

    deck = tmp_path / "input.cth_like_free_bdy_lasym_small"
    deck.write_text(DECK.read_text())  # mgrid deliberately NOT copied
    inp = VmecInput.from_file(deck)
    messages: list[str] = []
    args = types.SimpleNamespace(coils=None)
    plan = _free_boundary_plan(args, inp, deck,
                               emit=lambda s, **k: messages.append(str(s)))
    assert plan is None  # fixed-boundary fallback
    assert any("FIXED-BOUNDARY" in msg for msg in messages)
    assert any("mgrid file not found" in msg for msg in messages)
