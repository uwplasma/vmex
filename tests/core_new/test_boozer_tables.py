"""Tests for ``vmec_jax.core.boozer_tables`` (traceable Boozer input tables).

Validated here (mirroring the validation this function shipped with in the
sfincs_jax flagship example ``examples/optimize_QA_bootstrap.py``):

1. wout-engine parity on a small 3D equilibrium (LandremanPaul2021 QA lowres,
   ns=7): the traceable single-surface tables reproduce the host wout engine
   on the same converged state —
   - ``bmnc`` (identical quadrature) to < 1e-10 relative,
   - ``rmnc``/``zmns`` against the same VMEC odd-m parity interpolation of
     the wout full-mesh tables to < 1e-10 relative,
   - ``bsubumnc``/``bsubvmnc`` to < 5e-3 and ``lmns`` to < 2e-2 relative —
     the half-mesh finite-difference level of the wout engine's own grid
     treatment at this tiny ns (measured ~1e-3 at production ns),
   - ``iota``, ``G = bvco`` and ``I = buco`` at the surface to < 1e-10;
2. traceability: the function jits (same values as eager), and
   ``jax.grad`` of ``sum(bmnc**2)`` through the full implicit-equilibrium
   chain (``solve_implicit`` -> tables) w.r.t. the boundary coefficients is
   finite and nonzero.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from vmec_jax.core import implicit as im
from vmec_jax.core import solver
from vmec_jax.core.boozer_tables import boozer_input_tables
from vmec_jax.core.input import VmecInput
from vmec_jax.core.wout import wout_from_state

DATA_DIR = Path(__file__).resolve().parents[2] / "examples" / "data"

NS = 7                     # radial surfaces (CI-scale, as in the sfincs_jax example)
FTOL = 1e-11
MAX_ITERATIONS = 4000
J = NS // 2                # half-mesh radial row under test


@pytest.fixture(scope="module")
def qa_case():
    """Converged LandremanPaul2021 QA lowres solve + wout + traceable tables."""
    inp = VmecInput.from_file(str(DATA_DIR / "input.LandremanPaul2021_QA_lowres"))
    inp = dataclasses.replace(
        inp,
        ns_array=np.asarray([NS]),
        ftol_array=np.asarray([FTOL]),
        niter_array=np.asarray([MAX_ITERATIONS]),
    )
    cfg = im.make_config(inp)
    p0 = im.params_from_input(inp)
    result = solver.solve(inp, cfg.resolution, ftol=cfg.ftol,
                          max_iterations=cfg.max_iterations, mode="cli")
    assert result.converged
    rt = im.runtime_from_params(p0, cfg)
    wout = wout_from_state(inp=inp, state=result.state, fsqr=result.fsqr,
                           fsqz=result.fsqz, fsql=result.fsql)
    tabs = boozer_input_tables(result.state, rt, J)
    return inp, cfg, p0, result, rt, wout, tabs


def _max_rel(tabs, name, ref_row, xm_ref, xn_ref) -> float:
    """Max |mine - ref| / max|ref| over the modes both tables carry."""
    mine, ref = [], []
    for i, (m, n) in enumerate(zip(tabs["xm"], tabs["xn"])):
        hits = np.where((xm_ref == m) & (xn_ref == n))[0]
        if hits.size:
            mine.append(float(np.asarray(tabs[name])[i]))
            ref.append(float(ref_row[hits[0]]))
    assert len(ref) > 4, f"too few shared modes for {name}"
    mine, ref = np.asarray(mine), np.asarray(ref)
    return float(np.max(np.abs(mine - ref)) / max(np.max(np.abs(ref)), 1e-30))


# ==========================================================================
# 1. wout-engine parity
# ==========================================================================


def test_bmnc_matches_wout_engine(qa_case):
    _, _, _, _, _, wout, tabs = qa_case
    xm_nyq = np.asarray(wout.xm_nyq, dtype=int)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=int)
    # |B| spectrum: identical quadrature -> near machine precision
    assert _max_rel(tabs, "bmnc", np.asarray(wout.bmnc)[J], xm_nyq, xn_nyq) < 1e-10


def test_rmnc_zmns_match_parity_interpolated_wout(qa_case):
    _, _, _, _, _, wout, tabs = qa_case
    xm = np.asarray(wout.xm, dtype=int)
    xn = np.asarray(wout.xn, dtype=int)
    s = np.linspace(0.0, 1.0, NS)
    sqrt_s = np.sqrt(s)
    s_half_j = 0.5 * (s[J] + s[J - 1])

    def parity_half(full_table):
        a = np.asarray(full_table)[J - 1]
        b = np.asarray(full_table)[J]
        even = 0.5 * (a + b)
        odd = 0.5 * (a / sqrt_s[J - 1] + b / sqrt_s[J]) * np.sqrt(s_half_j)
        return np.where(xm % 2 == 0, even, odd)

    assert _max_rel(tabs, "rmnc", parity_half(wout.rmnc), xm, xn) < 1e-10
    assert _max_rel(tabs, "zmns", parity_half(wout.zmns), xm, xn) < 1e-10


def test_covariant_fields_and_lambda_match_at_half_mesh_fd_level(qa_case):
    _, _, _, _, _, wout, tabs = qa_case
    xm_nyq = np.asarray(wout.xm_nyq, dtype=int)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=int)
    # covariant fields: same half-mesh data, wout-engine grid differences only
    assert _max_rel(tabs, "bsubumnc", np.asarray(wout.bsubumnc)[J], xm_nyq, xn_nyq) < 5e-3
    assert _max_rel(tabs, "bsubvmnc", np.asarray(wout.bsubvmnc)[J], xm_nyq, xn_nyq) < 5e-3
    # lambda (reconstructed from angular derivatives, wout normalization;
    # dominated by the half-mesh finite-difference level at this tiny ns)
    assert _max_rel(tabs, "lmns", np.asarray(wout.lmns)[J],
                    np.asarray(wout.xm, dtype=int), np.asarray(wout.xn, dtype=int)) < 2e-2


def test_iota_and_boozer_currents_match_wout(qa_case):
    _, _, _, _, _, wout, tabs = qa_case
    assert abs(float(tabs["iota"]) - float(np.asarray(wout.iotas)[J])) < 1e-10
    # normalize G and I by the dominant covariant scale: I (the enclosed
    # toroidal current) is ~1e-17 for this near-vacuum QA, so a pure
    # relative comparison would be ill-posed
    g_ref = float(np.asarray(wout.bvco)[J])
    i_ref = float(np.asarray(wout.buco)[J])
    scale = max(abs(g_ref), abs(i_ref), 1e-30)
    assert abs(float(tabs["G"]) - g_ref) / scale < 1e-10
    assert abs(float(tabs["I"]) - i_ref) / scale < 1e-10


# ==========================================================================
# 2. traceability: jit and end-to-end gradient
# ==========================================================================


def test_tables_jit_match_eager(qa_case):
    _, _, _, result, rt, _, tabs = qa_case
    jitted = jax.jit(lambda st: boozer_input_tables(st, rt, J))(result.state)
    for key in ("rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc",
                "iota", "G", "I"):
        np.testing.assert_allclose(
            np.asarray(jitted[key]), np.asarray(tabs[key]),
            rtol=1e-12, atol=1e-14, err_msg=key)


def test_grad_of_bmnc_through_implicit_solve_is_finite(qa_case):
    """d(sum bmnc^2)/d(boundary) through solve_implicit is finite, nonzero."""
    _, cfg, p0, _, _, _, _ = qa_case

    def loss(params):
        state = im.solve_implicit(params, cfg)
        rt = im.runtime_from_params(params, cfg)
        return jnp.sum(boozer_input_tables(state, rt, J)["bmnc"] ** 2)

    grad = jax.grad(loss)(p0)
    g_rbc = np.asarray(grad.rbc)
    g_zbs = np.asarray(grad.zbs)
    assert np.all(np.isfinite(g_rbc)) and np.all(np.isfinite(g_zbs))
    assert float(np.max(np.abs(g_rbc))) > 0.0
