"""Targeted unit tests for :mod:`vmex.core.setup` corner branches.

Covers the ``APHI`` toroidal-flux polynomial map (identity short-circuit
and a genuine polynomial, checked against the exact antiderivative), the
degenerate ``ns < 2`` radial grid, and the lasym delta-rotation guards
(``convert_sym``/``convert_asym`` m=1 phase alignment in readin.f).
"""

from __future__ import annotations

import numpy as np
import pytest

from vmex.core import setup as su


def test_torflux_identity_short_circuit():
    torflux, deriv = su._torflux_functions(np.asarray([1.0, 0.0, 0.0]))
    x = np.linspace(0.0, 1.0, 7)
    np.testing.assert_allclose(np.asarray(torflux(x)), x, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(deriv(x)), np.ones_like(x), rtol=0, atol=0)
    # empty APHI defaults to the identity too
    torflux, deriv = su._torflux_functions(np.asarray([]))
    np.testing.assert_allclose(np.asarray(deriv(0.3)), 1.0)


def test_torflux_polynomial_matches_antiderivative():
    # aphi = [0.5, 0.5]: torflux_deriv(x) = 0.5 + 1.0*x (i * aphi_i * x**(i-1));
    # the 101-point trapezoid rule is exact for a linear integrand.
    torflux, deriv = su._torflux_functions(np.asarray([0.5, 0.5]))
    x = np.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(np.asarray(deriv(x)), 0.5 + x, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(torflux(x)), 0.5 * x + 0.5 * x**2,
                               rtol=1e-12, atol=1e-14)


def test_radial_grids_degenerate_ns():
    grids = su.radial_grids(1)
    assert grids.s_full.shape == (1,)
    np.testing.assert_array_equal(np.asarray(grids.s_full), [0.0])
    np.testing.assert_array_equal(np.asarray(grids.sqrts), [1.0])
    assert float(grids.hs) == 1.0


def _m1_arrays(ntor=1, mpol=3):
    rbc = np.zeros((2 * ntor + 1, mpol))
    rbs = np.zeros_like(rbc)
    zbc = np.zeros_like(rbc)
    zbs = np.zeros_like(rbc)
    return rbc, rbs, zbc, zbs


def test_lasym_delta_rotation_guards():
    ntor = 1
    # mpol < 2: unrotated
    rbc, rbs, zbc, zbs = (np.zeros((3, 1)),) * 4
    out = su._lasym_delta_rotation(rbc, rbs, zbc, zbs, mpol=1, ntor=ntor)
    assert out[0] is rbc

    # degenerate m=1 content (denominator zero): unrotated
    rbc, rbs, zbc, zbs = _m1_arrays()
    out = su._lasym_delta_rotation(rbc, rbs, zbc, zbs, mpol=3, ntor=ntor)
    assert out[0] is rbc

    # rbs(0,1) == zbc(0,1) -> delta = 0: unrotated
    rbc, rbs, zbc, zbs = _m1_arrays()
    rbc[ntor, 1] = 1.0
    rbs[ntor, 1] = 0.25
    zbc[ntor, 1] = 0.25
    out = su._lasym_delta_rotation(rbc, rbs, zbc, zbs, mpol=3, ntor=ntor)
    assert out[0] is rbc


def test_lasym_delta_rotation_aligns_m1_modes():
    """After rotation the m=1 antisymmetric content satisfies rbs = zbc."""
    ntor = 1
    rbc, rbs, zbc, zbs = _m1_arrays()
    rbc[ntor, 1] = 1.0
    zbs[ntor, 1] = 0.8
    rbs[ntor, 1] = 0.3
    zbc[ntor, 1] = 0.1
    rbc2, rbs2, zbc2, zbs2 = su._lasym_delta_rotation(
        rbc, rbs, zbc, zbs, mpol=3, ntor=ntor)
    assert rbs2[ntor, 1] == pytest.approx(zbc2[ntor, 1], abs=1e-12)
    # the rotation is a pure phase: the m=1 quadratic invariant is preserved
    inv0 = rbc[ntor, 1] ** 2 + rbs[ntor, 1] ** 2 + zbc[ntor, 1] ** 2 + zbs[ntor, 1] ** 2
    inv1 = rbc2[ntor, 1] ** 2 + rbs2[ntor, 1] ** 2 + zbc2[ntor, 1] ** 2 + zbs2[ntor, 1] ** 2
    assert inv1 == pytest.approx(inv0, rel=1e-12)


# ---------------------------------------------------------------------------
# The jitted setup lane (one XLA program per rung instead of ~110)
# ---------------------------------------------------------------------------


@pytest.fixture
def _jit_enabled():
    """The repo conftest disables jit globally; the lane needs it on."""
    import jax

    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


def _solovev_input():
    from pathlib import Path

    from vmex.core.input import VmecInput

    root = Path(__file__).resolve().parents[1]
    return VmecInput.from_file(str(root / "examples" / "data" / "input.solovev"))


def test_setup_lane_is_reused_across_rungs_and_compiles_once(_jit_enabled):
    """Every radial rung of one deck hits the same lane executable.

    The lane exists to replace ~110 single-op XLA programs per rung with
    one, which only pays off if the second rung reuses the first rung's
    executable.  Counted from JAX's own compile events, with jit enabled,
    because the suite otherwise runs with ``jax_disable_jit`` and would
    never execute the compiled path at all.
    """
    import jax
    from jax import monitoring

    from vmex.core import solver as S

    compiles = []
    monitoring.register_event_duration_secs_listener(
        lambda event, secs, **kw: compiles.append(kw.get("fun_name", "?"))
        if event == "/jax/core/compile/backend_compile_duration" else None)

    inp = _solovev_input()
    su._setup_lane.cache_clear()
    jax.clear_caches()
    first = su.run_setup(inp, S.resolution_from_input(inp, ns=11),
                         lconm1=True, infer_axis_if_missing=False)
    after_first = len(compiles)
    second = su.run_setup(inp, S.resolution_from_input(inp, ns=11),
                          lconm1=True, infer_axis_if_missing=False)
    assert len(compiles) == after_first, (
        f"a repeat run_setup compiled {compiles[after_first:]}; the lane key "
        "is not stable across calls")
    assert after_first <= 3, (
        f"one rung asked for {after_first} XLA programs ({compiles}); the "
        "array tail is dispatching op by op again")
    np.testing.assert_array_equal(np.asarray(first.mass), np.asarray(second.mass))

    # A different rung is a different structure, so it compiles its own lane
    # and does not disturb the first one.
    su.run_setup(inp, S.resolution_from_input(inp, ns=17),
                 lconm1=True, infer_axis_if_missing=False)
    again = su.run_setup(inp, S.resolution_from_input(inp, ns=11),
                         lconm1=True, infer_axis_if_missing=False)
    np.testing.assert_array_equal(np.asarray(again.mass), np.asarray(first.mass))


def _dispatched_one_op_at_a_time(inp, resolution, *, lconm1=True):
    """``run_setup``'s array tail with every op dispatched on its own.

    This is what the function did before the lane, and it is the oracle the
    lane has to reproduce exactly.  ``jax.disable_jit()`` is *not* that
    oracle: it also reroutes ``jnp`` helpers that jit internally, and
    ``jnp.linspace`` alone then moves the radial mesh by one ulp.
    """
    from vmex.core.fourier import mode_table, trig_tables
    from vmex.core.transforms import odd_m_sqrt_s_scaling

    modes = mode_table(resolution.mpol, resolution.ntor)
    trig = trig_tables(resolution)
    grids = su.radial_grids(resolution.ns)
    boundary = su.boundary_from_input(inp, modes=modes, trig=trig, lconm1=lconm1)
    profiles = su.flux_profiles(inp, grids, r00=boundary.r00,
                                signgs=boundary.signgs, lflip=boundary.lflip)
    axis = su._axis_arrays(inp, grids.s_full.dtype)
    state = su.interior_guess(
        boundary_R_cos=boundary.R_cos, boundary_R_sin=boundary.R_sin,
        boundary_Z_cos=boundary.Z_cos, boundary_Z_sin=boundary.Z_sin,
        raxis_c=axis[0], raxis_s=axis[1], zaxis_c=axis[2], zaxis_s=axis[3],
        modes=modes, trig=trig, s=grids.s_full)
    expected = {
        "s_full": grids.s_full, "s_half": grids.s_half, "sqrts": grids.sqrts,
        "shalf": grids.shalf, "sm": grids.sm, "sp": grids.sp, "hs": grids.hs,
        "scalxc": odd_m_sqrt_s_scaling(grids.s_full, resolution.mpol),
        "boundary_R_cos": boundary.R_cos, "boundary_R_sin": boundary.R_sin,
        "boundary_Z_cos": boundary.Z_cos, "boundary_Z_sin": boundary.Z_sin,
        "raxis_c": axis[0], "raxis_s": axis[1],
        "zaxis_c": axis[2], "zaxis_s": axis[3],
        "R_cos": state[0], "R_sin": state[1], "Z_cos": state[2],
        "Z_sin": state[3], "lambda_cos": state[4], "lambda_sin": state[5],
    }
    expected.update({name: value for name, value in profiles.items()})
    return expected


def test_setup_lane_matches_the_eager_dispatch_bit_for_bit(_jit_enabled):
    """The lane must reproduce op-at-a-time dispatch exactly, not closely.

    Compiled at the default XLA optimization level the lane's arithmetic is
    contracted (a multiply and an add become one FMA) and the m=0 row of the
    initial guess moves by one ulp -- a different initial guess, a different
    trajectory and a different final ``fsqr``.  The lane is therefore
    compiled at optimization level 0, and this is the test that holds it.
    """
    from vmex.core import solver as S

    inp = _solovev_input()
    resolution = S.resolution_from_input(inp, ns=11)
    lane = su.run_setup(inp, resolution, lconm1=True,
                        infer_axis_if_missing=False)
    expected = _dispatched_one_op_at_a_time(inp, resolution)
    assert expected, "the oracle built nothing"
    for field, want in expected.items():
        got, want = np.asarray(getattr(lane, field)), np.asarray(want)
        assert got.dtype == want.dtype and got.shape == want.shape, field
        assert np.array_equal(got, want), (
            f"{field}: the lane and the op-at-a-time dispatch differ "
            f"(max |delta| {np.max(np.abs(got - want)) if got.size else 0})")


def test_zero_carry_constants_are_transfers_not_programs(_jit_enabled):
    """Host-built zeros compile nothing, and the carry copy is one program."""
    import jax
    from jax import monitoring

    from vmex.core import solver as S

    compiles = []
    monitoring.register_event_duration_secs_listener(
        lambda event, secs, **kw: compiles.append(kw.get("fun_name", "?"))
        if event == "/jax/core/compile/backend_compile_duration" else None)
    jax.clear_caches()
    for shape in ((7,), (8,), (9, 3), ()):
        value = S._host_zeros(shape, np.float64)
        assert np.asarray(value).shape == shape
        assert not np.any(np.asarray(value))
    assert compiles == [], f"_host_zeros compiled {compiles}"

    # One jitted copy for the whole carry, and every output leaf a distinct
    # buffer -- which is what the donated block/while lanes require.
    tree = {"a": jax.numpy.ones((4,)), "b": None}
    tree["b"] = tree["a"]
    copied = S._distinct_buffers(tree)
    pointers = {leaf.unsafe_buffer_pointer() for leaf in jax.tree.leaves(copied)}
    assert len(pointers) == len(jax.tree.leaves(copied)), "aliased output buffers"
    assert not (pointers & {tree["a"].unsafe_buffer_pointer()}), "aliased the input"
