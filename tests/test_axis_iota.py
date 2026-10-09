"""``opt.geometric_iota`` / ``opt.axis_iota``: the iota without its enclosed-current part, and its axis value.

A prescribed-current deck (LP QA at 0.5% beta, 3 x 3 modes) with no current, where the geometric iota is the
iota, and with a bootstrap-like current ``I' ~ s^(1/4) (1 - s)``, whose part of iota (0.1) is steep off the axis.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy

from vmex.core import optimize as opt  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.statephysics import _iotas_half  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")  # full solves: run jitted
DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
KNOTS = np.r_[0.0, np.geomspace(1e-6, 1.0, 60)]


def _solve(ns, *, current=True):
    inp = VmecInput.from_file(DATA_DIR / "input.LandremanPaul2021_QA_beta0p5_bootstrap").change_resolution(
        mpol=3, ntor=3, ntheta=12, nzeta=12)
    if current:
        inp = dataclasses.replace(inp, pcurr_type="line_segment_ip", ac_aux_s=KNOTS,
                                  ac_aux_f=KNOTS**0.25 * (1.0 - KNOTS))
    else:
        inp = dataclasses.replace(inp, curtor=0.0)
    inp = dataclasses.replace(inp, ns_array=np.array([ns]), ftol_array=np.array([1e-14]),
                              niter_array=np.array([40000]))
    eq = opt.solve_equilibrium(inp)
    assert eq.result.converged
    return eq


@pytest.fixture(scope="module")
def bootstrap_like():
    return _solve(21)


def test_geometric_iota_is_iota_without_current():
    eq = _solve(21, current=False)
    geo, iota = np.asarray(opt.geometric_iota(eq.state, eq.runtime)), np.asarray(_iotas_half(eq.state, eq.runtime))
    np.testing.assert_allclose(geo[1:], iota[1:], rtol=0, atol=1e-14)
    assert abs(float(opt.axis_iota(eq.state, eq.runtime)) - (15 * iota[1] - 10 * iota[2] + 3 * iota[3]) / 8) < 1e-14


def test_axis_iota_drops_the_current_part_and_drifts_less_with_ns(bootstrap_like):
    """iota - iota_geo = icurv / (phips <guu/sqrt(g)>), steep off the axis: VMEC's iotaf[0] extrapolates it onto
    the axis and drifts with ns; the axis value has none of it and moves 3-4x less (ns 21 -> 41)."""
    eqs = (bootstrap_like, _solve(41))
    axis = [float(opt.axis_iota(eq.state, eq.runtime)) for eq in eqs]
    naive = [float(eq.wout.iotaf[0]) for eq in eqs]
    assert min(abs(n - a) for n, a in zip(naive, axis)) > 0.04  # ~0.05 of the 0.1 current part
    assert abs(axis[1] - axis[0]) < 0.4 * abs(naive[1] - naive[0])


def test_prescribed_iota_axis_is_the_profile_value():
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    ai = np.zeros_like(np.asarray(inp.ai, dtype=float))
    ai[:2] = 1.0, -0.4
    inp = dataclasses.replace(inp, ai=ai)
    eq = opt.solve_equilibrium(inp)
    assert float(opt.axis_iota(eq.state, eq.runtime)) == pytest.approx(float(eq.runtime.setup.iotaf[0]), abs=1e-14)
    assert float(opt.axis_iota(eq.state, eq.runtime)) == pytest.approx(1.0, abs=1e-12)


def test_axis_iota_jvp_matches_central_difference(bootstrap_like):
    eq = bootstrap_like
    tangent = jax.tree.map(jnp.zeros_like, eq.state)
    tangent = dataclasses.replace(tangent, L_sin=tangent.L_sin.at[2, 1].set(1.0),
                                  R_cos=tangent.R_cos.at[3, 1].set(1.0))
    value, jvp = jax.jvp(lambda s: opt.axis_iota(s, eq.runtime), (eq.state,), (tangent,))
    h = 1e-6
    shifted = [jax.tree.map(lambda a, t: a + sign * h * t, eq.state, tangent) for sign in (1, -1)]
    fd = (opt.axis_iota(shifted[0], eq.runtime) - opt.axis_iota(shifted[1], eq.runtime)) / (2 * h)
    assert np.isfinite(float(value)) and abs(float(jvp)) > 0
    np.testing.assert_allclose(float(jvp), float(fd), rtol=1e-5, atol=1e-10)


def test_axis_field_strength_is_the_axis_mode_of_b(bootstrap_like):
    """The angle mean of |B| extrapolated to the axis: the wout (0, 0) |B| harmonic extrapolated the same way."""
    eq = bootstrap_like
    b00 = np.asarray(eq.wout.bmnc)[:, 0]  # half mesh; (m, n) = (0, 0) is the first Nyquist mode
    value = float(opt.axis_field_strength(eq.state, eq.runtime))
    assert value == pytest.approx(15 / 8 * b00[1] - 5 / 4 * b00[2] + 3 / 8 * b00[3], rel=1e-6)
    assert value == pytest.approx(1.5 * b00[1] - 0.5 * b00[2], rel=2e-3)
