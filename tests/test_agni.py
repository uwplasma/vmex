"""Gates for the VMEX -> AGNI equilibrium adapter (``vmex.core.agni``).

The adapter is checked against identities it must satisfy by construction of
the PEST chart (root residual, ``psi_r``, the ``B`` representation, ``J^phi``),
against central differences of its own ``sqrtg``, and end to end through
``agnimhd.growth_rate`` in optimize mode.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

agnimhd = pytest.importorskip("agnimhd")

import vmex as vj
from vmex.core import agni
from vmex.core.input import VmecInput
from vmex.core.solver import prepare_runtime, resolution_from_input

jax.config.update("jax_enable_x64", True)

DECK = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.nfp2_QA_finite_beta"
AUTOMORPHISM = {"eps": 0.05, "x_0": 0.6, "m_1": 2.5, "m_2": 3.0}
SHAPE = (6, 7, 5)
AD_SHAPE = (6, 9, 6)  # SHAPE's top mode is a numerical null; this grid has an isolated unstable one


@pytest.fixture(scope="module")
def deck():
    """Converged nfp=2 QA at about 2 % beta, its runtime, and the AGNI node set."""
    inp = VmecInput.from_file(DECK)
    inp = dataclasses.replace(inp, ns_array=np.asarray([21]), ftol_array=np.asarray([1e-12]),
                              niter_array=np.asarray([8000]))
    state = vj.solve_multigrid(inp, verbose=False).state
    runtime = prepare_runtime(inp, resolution_from_input(inp, ns=21))
    basis = agnimhd.Basis(*SHAPE, radial="lobatto", automorphism=AUTOMORPHISM, restrict=True)
    nodes, diffmat = basis.nodes_and_diffmat(inp.nfp, 0)
    return inp, state, runtime, nodes, diffmat


@pytest.fixture(scope="module")
def mapped(deck):
    inp, state, runtime, nodes, _ = deck
    emap = agni.make_agni_map(runtime, nodes, inp=inp)
    eq, extras = jax.jit(lambda pair: emap(pair, extras=True))((state, runtime))
    return eq, {k: np.asarray(v) for k, v in extras.items()}


def test_nodes_are_rho_major_and_profiles_are_surface_functions(deck, mapped):
    _, _, _, nodes, _ = deck
    eq, extras = mapped
    pts = extras["points"].reshape(*SHAPE, 3)
    np.testing.assert_array_equal(pts[:, 0, 0, 0], np.asarray(nodes["rho"]))
    np.testing.assert_array_equal(pts[0, :, 0, 1], np.asarray(nodes["theta"]))
    np.testing.assert_array_equal(pts[0, 0, :, 2], np.asarray(nodes["zeta"]))
    for name in ("iota", "p", "p_r", "psi_r"):
        surface = np.asarray(getattr(eq, name)).reshape(SHAPE[0], -1)
        assert np.ptp(surface, axis=1).max() == 0.0 or np.ptp(surface, axis=1).max() < 1e-13 * np.abs(surface).max()


def test_pest_root_and_jacobian_sign(mapped):
    eq, extras = mapped
    assert np.abs(extras["root_residual"]).max() < 1e-12
    sqrtg = np.asarray(eq.sqrtg)
    assert np.all(sqrtg * sqrtg[0] > 0.0)
    assert np.all(np.asarray(eq.g_sup_rr) > 0.0)
    assert float(eq.a) > 0.0 and np.isfinite(float(eq.Psi))


def test_flux_and_field_representations(deck, mapped):
    inp, _, runtime, _, _ = deck
    eq, extras = mapped
    rho = extras["points"][:, 0]
    Psi = float(eq.Psi)
    assert Psi == pytest.approx(float(inp.phiedge), rel=1e-14)
    psi_r = np.asarray(eq.psi_r)
    np.testing.assert_allclose(psi_r, rho * Psi / np.pi, rtol=1e-14)
    sqrtg = np.asarray(eq.sqrtg)
    np.testing.assert_allclose(extras["B_dot_grad_phi"] * sqrtg / psi_r, 1.0, rtol=1e-10)
    e, iota = extras["e"], np.asarray(eq.iota)
    B = (psi_r / sqrtg)[:, None] * (iota[:, None] * e[:, :, 1] + e[:, :, 2])
    assert np.linalg.norm(B - extras["B"], axis=1).max() / np.linalg.norm(extras["B"], axis=1).max() < 1e-10


def test_toroidal_current_is_j_dot_grad_phi(mapped):
    eq, extras = mapped
    contravariant = np.linalg.solve(extras["e"], extras["J"][..., None])[..., 0]
    expected = np.asarray(eq.J_sup_zeta)
    assert np.abs(contravariant[:, 2] - expected).max() < 1e-10 * np.abs(expected).max()
    np.testing.assert_allclose(np.asarray(eq.abs_J), np.linalg.norm(extras["J"], axis=1), rtol=1e-13)


def test_validate_passes_on_the_concrete_output(mapped):
    eq, _ = mapped
    assert eq.validate() is eq


def test_sqrtg_partials_match_central_differences(deck):
    """One tensor grid holds two base points per axis and their +-h neighbours."""
    inp, state, runtime, _, _ = deck
    h = 1e-5

    def stencil(c):
        return np.array([c[0], c[0] + h, c[0] - h, c[1], c[1] + h, c[1] - h])

    nodes = {"rho": stencil((0.4, 0.8)), "theta": stencil((0.3, 2.1)), "zeta": stencil((0.1, 0.7))}
    eq = jax.jit(agni.make_agni_map(runtime, nodes, inp=inp, radius_quadrature=(4, 8, 2)))((state, runtime))
    shape, base = (6, 6, 6), [0, 3]

    def along(values, axis, indices):
        picked = np.moveaxis(np.asarray(values).reshape(shape), axis, 0)[indices]
        return picked[:, base][:, :, base]  # other axes at their base points

    for axis, field in enumerate(("sqrtg_r", "sqrtg_v", "sqrtg_p")):
        fd = (along(eq.sqrtg, axis, [1, 4]) - along(eq.sqrtg, axis, [2, 5])) / (2.0 * h)
        np.testing.assert_allclose(along(getattr(eq, field), axis, base), fd, rtol=1e-6, atol=1e-8 * np.abs(fd).max())


def test_eager_and_jit_agree(deck):
    inp, state, runtime, _, _ = deck
    nodes = {"rho": np.array([0.5]), "theta": np.array([0.4, 3.0]), "zeta": np.array([0.2])}
    emap = agni.make_agni_map(runtime, nodes, inp=inp, radius_quadrature=(4, 8, 2))
    jitted = jax.jit(emap)((state, runtime))
    eager = emap((state, runtime))
    for name in agnimhd.equilibrium.REQUIRED_ARRAYS + ("J_cross_grad_rho", "B_dot_grad_grad_rho"):
        np.testing.assert_allclose(np.asarray(getattr(eager, name)), np.asarray(getattr(jitted, name)),
                                   rtol=1e-12, atol=1e-12 * np.abs(np.asarray(getattr(jitted, name))).max())
    assert float(eager.a) == pytest.approx(float(jitted.a), rel=1e-13)


def test_density_weight_is_passed_through(deck):
    inp, state, runtime, _, _ = deck
    nodes = {"rho": np.array([0.4, 0.9]), "theta": np.array([1.0]), "zeta": np.array([0.3])}
    emap = agni.make_agni_map(runtime, nodes, inp=inp, radius_quadrature=(4, 8, 2),
                              density=lambda rho: 1.0 - 0.5 * rho ** 2)
    np.testing.assert_allclose(np.asarray(emap((state, runtime)).density), [1.0 - 0.08, 1.0 - 0.405])
    with pytest.raises(ValueError, match="one entry per node"):
        agni.make_agni_map(runtime, nodes, density=np.ones(3))


def test_growth_rate_gradient_matches_finite_differences(deck):
    """``d(gamma^2)`` through the map, along a state direction and a pressure direction.

    The state direction scales the ``Z`` sine coefficients (vertical elongation);
    the runtime direction scales the pressure (``setup.mass``) alone, which moves
    ``p`` and ``p_r`` but not the geometry or the field. Both are partial
    derivatives at fixed force-balance residual, which is what optimize mode
    returns. The reverse pass runs op by op: jitting it costs minutes of XLA
    compilation for the same answer.
    """
    inp, state, runtime, _, _ = deck
    basis = agnimhd.Basis(*AD_SHAPE, radial="lobatto", automorphism=AUTOMORPHISM, restrict=True)
    nodes, diffmat = basis.nodes_and_diffmat(inp.nfp, 0)
    emap = agni.make_agni_map(runtime, nodes, inp=inp)
    config = agnimhd.AssemblyConfig()
    probe = agnimhd.SolverConfig(eigensolver="eigsh", sigma=1e-2, eigsh_tol=1e-12)
    gamma2, _, residual = agnimhd.eigenpair(jax.jit(emap)((state, runtime)), diffmat, config, probe)
    assert float(gamma2) > 0.0 and float(residual) < 1e-6  # an isolated unstable mode
    solver = dataclasses.replace(probe, sigma=2.0 * float(gamma2))

    def value(pair):
        return agnimhd.growth_rate(pair, diffmat, config, solver, equilibrium_map=emap)

    grad_state, grad_runtime = jax.grad(value, allow_int=True)((state, runtime))
    leaves = jax.tree_util.tree_leaves((grad_state, grad_runtime))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating))
    ad_state = float(jnp.vdot(grad_state.Z_sin, state.Z_sin))
    ad_pressure = float(jnp.vdot(grad_runtime.setup.mass, runtime.setup.mass))
    assert np.isfinite(ad_state) and np.isfinite(ad_pressure) and ad_state != 0.0 and ad_pressure != 0.0

    def shifted_state(t):
        return dataclasses.replace(state, Z_sin=state.Z_sin * (1.0 + t))

    def scaled_pressure(t):
        setup = dataclasses.replace(runtime.setup, mass=runtime.setup.mass * (1.0 + t))
        return dataclasses.replace(runtime, setup=setup)

    h = 1e-4
    fd_state = (value((shifted_state(h), runtime)) - value((shifted_state(-h), runtime))) / (2 * h)
    fd_pressure = (value((state, scaled_pressure(h))) - value((state, scaled_pressure(-h)))) / (2 * h)
    assert ad_state == pytest.approx(float(fd_state), rel=1e-5)
    assert ad_pressure == pytest.approx(float(fd_pressure), rel=1e-5)


def test_jitted_value_and_grad_matches_finite_differences(deck):
    """The map sits inside an optimization loop: its jitted reverse pass must be exact."""
    inp, state, runtime, nodes, _ = deck
    emap = agni.make_agni_map(runtime, nodes, inp=inp, radius_quadrature=(4, 8, 2))

    def objective(pair):
        eq = emap(pair)
        return jnp.sum(eq.sqrtg * eq.p) + 1e-3 * jnp.sum(eq.sqrtg_r * eq.g_sup_rr) + jnp.sum(
            eq.J_cross_grad_rho * eq.B_dot_grad_grad_rho) + eq.a

    value, (grad_state, _) = jax.jit(jax.value_and_grad(objective, allow_int=True))((state, runtime))
    shifted = lambda t: (dataclasses.replace(state, Z_sin=state.Z_sin * (1.0 + t)), runtime)  # noqa: E731
    h = 1e-5
    fd = (objective(shifted(h)) - objective(shifted(-h))) / (2.0 * h)
    assert float(value) == pytest.approx(float(objective((state, runtime))), rel=1e-13)
    assert float(jnp.vdot(grad_state.Z_sin, state.Z_sin)) == pytest.approx(float(fd), rel=1e-7)


def test_unsupported_inputs_are_rejected(deck):
    inp, _, runtime, nodes, _ = deck
    with pytest.raises(NotImplementedError, match="APHI"):
        agni.make_agni_map(runtime, nodes, inp=dataclasses.replace(inp, aphi=np.r_[1.0, 0.1, np.zeros(18)]))
    # Without inp the identity map is inferred from the runtime flux profiles.
    warped = dataclasses.replace(inp, aphi=np.r_[0.9, 0.1, np.zeros(18)])
    with pytest.raises(NotImplementedError, match="APHI"):
        agni.make_agni_map(prepare_runtime(warped, resolution_from_input(warped, ns=21)), nodes)
    for flag in ("lasym", "lflip"):
        flagged = dataclasses.replace(runtime, setup=dataclasses.replace(runtime.setup, **{flag: True}))
        with pytest.raises(NotImplementedError, match=flag):
            agni.make_agni_map(flagged, nodes)
    with pytest.raises(ValueError, match="field period"):
        agni.make_agni_map(runtime, {**nodes, "zeta": np.asarray(nodes["zeta"]) + 2.0 * np.pi})
