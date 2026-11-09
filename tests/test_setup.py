"""Tests for ``vmex.core.setup`` (profil1d.f / readin.f / profil3d.f).

Field-by-field parity with the legacy setup chain (flux profiles, boundary
processing incl. theta flip and the lconm1 m=1 conversion, initial guess,
guess_axis) was proven by the A/B suite that retired with the legacy tree.
Kept here are the analytic VMEC2000 grid conventions, well-posedness of the
initial state through the proven geometry chain, the readin.f theta-flip
behaviour, and jit-compatibility of the state-producing path.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core import implicit as im
from vmex.core.fourier import mode_table, trig_tables
from vmex.core.scaling import scale_input
from vmex.core.geometry import (
    half_mesh_jacobian,
    real_space_geometry,
    sqrt_s_half_mesh,
)
from vmex.core.input import VmecInput
from vmex.core.setup import (
    boundary_volume,
    rescale_boundary_volume,
    _geometry_state_arrays,
    _internal_blocks_from_input,
    interior_guess,
    run_setup,
)
from vmex.core.solver import resolution_from_input

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"

RTOL = 1e-12
ATOL = 1e-13


def geometry_state(setup, *, modes):
    """Evolution-basis state -> ``real_space_geometry`` keyword inputs."""
    return _geometry_state_arrays(
        (setup.R_cos, setup.R_sin, setup.Z_cos, setup.Z_sin, setup.lambda_cos, setup.lambda_sin),
        modes=modes, lthreed=setup.lthreed, lasym=setup.lasym, lconm1=setup.lconm1)

SIGNGS = -1

# 2D sym pressure-driven / 2D sym nfp=5 current-driven (two_power, phiedge<0)
# / 3D sym (lconm1 m=1, axis inference) / lasym (axis inference) / high-mpol
# multigrid deck (legacy RAXIS name).
CASES = [
    "solovev",
    "cth_like_fixed_bdy",
    "li383_low_res",
    "up_down_asymmetric_tokamak",
    "DSHAPE",
]


def _allclose(new, old, name):
    np.testing.assert_allclose(
        np.asarray(new), np.asarray(old), rtol=RTOL, atol=ATOL, err_msg=f"{name} mismatch"
    )


@pytest.fixture(scope="module", params=CASES, ids=CASES)
def case(request):
    """Build the core setup for one input deck."""
    name = request.param
    deck = DATA_DIR / f"input.{name}"

    inp = VmecInput.from_file(deck)
    res = resolution_from_input(inp)
    ns = int(res.ns)
    s = np.linspace(0.0, 1.0, ns)
    setup = run_setup(inp, res)
    modes_new = mode_table(res.mpol, res.ntor)
    trig_new = trig_tables(res)

    geom = real_space_geometry(
        **geometry_state(setup, modes=modes_new),
        modes=modes_new, trig=trig_new, s=setup.s_full,
    )
    jac = half_mesh_jacobian(geom, s=setup.s_full)

    return SimpleNamespace(
        name=name, s=s, ns=ns, inp=inp, res=res, setup=setup,
        modes_new=modes_new, trig_new=trig_new, geom=geom, jac=jac,
    )


# ---------------------------------------------------------------------------
# Radial grids (profil1d.f conventions, analytic references)
# ---------------------------------------------------------------------------


def test_radial_grids_conventions(case):
    setup, s = case.setup, case.s
    _allclose(setup.s_full, s, "s_full")
    hs = s[1] - s[0]
    _allclose(setup.hs, hs, "hs")
    # profil1d.f: s_half(i) = hs*|i-1.5| — the axis slot repeats the first
    # interior half-mesh value; interior slots are the full-mesh midpoints.
    _allclose(setup.s_half[1:], 0.5 * (s[1:] + s[:-1]), "s_half interior")
    _allclose(setup.s_half[0], 0.5 * hs, "s_half axis slot")
    sqrts_ref = np.sqrt(s)
    sqrts_ref[-1] = 1.0
    _allclose(setup.sqrts, sqrts_ref, "sqrts")
    _allclose(setup.shalf, sqrt_s_half_mesh(jnp.asarray(s)), "shalf")
    # profil1d.f sm/sp odd-m half-mesh weights.
    sm_ref = np.zeros(case.ns)
    sm_ref[1:] = np.asarray(setup.shalf)[1:] / sqrts_ref[1:]
    sp_ref = np.zeros(case.ns)
    sp_ref[1:-1] = np.asarray(setup.shalf)[2:] / sqrts_ref[1:-1]
    sp_ref[-1] = 1.0 / sqrts_ref[-1]
    sp_ref[0] = sm_ref[1]
    _allclose(setup.sm, sm_ref, "sm")
    _allclose(setup.sp, sp_ref, "sp")


def test_profiles_and_current_lane(case):
    setup = case.setup
    assert setup.signgs == SIGNGS
    assert setup.lflip is False
    # phips carries a zeroed axis slot; lamscale is finite and positive.
    assert float(np.asarray(setup.phips)[0]) == 0.0
    assert np.isfinite(float(setup.lamscale)) and float(setup.lamscale) > 0.0
    assert float(setup.psi_edge) == pytest.approx(
        setup.signgs * float(case.inp.phiedge) / (2.0 * np.pi))
    assert setup.ncurr == int(case.inp.ncurr)
    if setup.ncurr == 1 and abs(float(case.inp.curtor)) > 0 and np.any(
        np.asarray(case.inp.ac)
    ):
        # The current-driven deck must exercise a nonzero pcurr lane.
        assert np.any(np.asarray(setup.icurv) != 0.0)


def test_signed_flux_updates_with_implicit_parameters():
    inp = VmecInput.from_file(DATA_DIR / "input.nfp1_QI")
    cfg = im.make_config(inp, ns=9)
    params = im.params_from_input(inp, device=None)
    scaled = dataclasses.replace(params, phiedge=1.2 * params.phiedge)
    original = im.runtime_from_params(params, cfg).setup
    updated = im.runtime_from_params(scaled, cfg).setup
    np.testing.assert_allclose(updated.psi_edge, 1.2 * original.psi_edge)
    np.testing.assert_allclose(updated.psi_half, 1.2 * original.psi_half)


# ---------------------------------------------------------------------------
# Boundary / initial state (readin.f + profil3d.f) properties
# ---------------------------------------------------------------------------


def test_edge_row_is_processed_boundary(case):
    setup = case.setup
    _allclose(setup.R_cos[-1], setup.boundary_R_cos, "edge row R_cos")
    _allclose(setup.Z_sin[-1], setup.boundary_Z_sin, "edge row Z_sin")


def test_axis_coefficients(case):
    setup = case.setup
    input_axis_given = any(
        np.any(np.asarray(a) != 0.0)
        for a in (case.inp.raxis_c, case.inp.raxis_s, case.inp.zaxis_c, case.inp.zaxis_s)
    )
    if input_axis_given:
        # Axis given in the deck: used verbatim (profil3d.f).
        _allclose(setup.raxis_c, case.inp.raxis_c, "raxis_c")
        _allclose(setup.zaxis_s, case.inp.zaxis_s, "zaxis_s")
    else:
        # Axis inferred (guess_axis.f lane): must be usable, i.e. nonzero R.
        assert float(np.asarray(setup.raxis_c)[0]) > 0.0


# ---------------------------------------------------------------------------
# Well-posedness: the initial state through the proven new geometry chain
# ---------------------------------------------------------------------------


def test_initial_state_jacobian_has_no_sign_change(case):
    assert not bool(case.jac.jacobian_sign_changed), (
        f"{case.name}: initial guess flags a Jacobian sign change"
    )
    # And the interior Jacobian sign agrees with signgs = -1 (jacobian.f).
    tau_interior = np.asarray(case.jac.tau)[1:]
    assert np.all(np.sign(tau_interior) == case.setup.signgs)


# ---------------------------------------------------------------------------
# Theta flip (readin.f lflip + init_geometry.f90 flip_theta)
# ---------------------------------------------------------------------------


def test_theta_flip_recovers_good_jacobian(case):
    if case.name != "solovev":
        pytest.skip("flip check runs once on the solovev deck")
    # Negating ZBS reverses the poloidal orientation: readin.f must detect
    # rtest*ztest < 0, flip theta -> pi - theta, and negate iotas/chips.
    flipped_inp = dataclasses.replace(case.inp, zbs=-np.asarray(case.inp.zbs))
    flipped = run_setup(flipped_inp, case.res)
    assert flipped.lflip is True
    assert flipped.signgs == SIGNGS
    _allclose(flipped.iotas, -np.asarray(case.setup.iotas), "flipped iotas")
    _allclose(flipped.chips, -np.asarray(case.setup.chips), "flipped chips")
    geom = real_space_geometry(
        **geometry_state(flipped, modes=case.modes_new),
        modes=case.modes_new, trig=case.trig_new, s=flipped.s_full,
    )
    jac = half_mesh_jacobian(geom, s=flipped.s_full)
    assert not bool(jac.jacobian_sign_changed)
    assert np.all(np.sign(np.asarray(jac.tau)[1:]) == SIGNGS)


# ---------------------------------------------------------------------------
# jit-compatibility of the state-producing path
# ---------------------------------------------------------------------------


def test_interior_guess_is_jittable(case):
    setup, modes, trig = case.setup, case.modes_new, case.trig_new

    def produce_state(args):
        b_rc, b_rs, b_zc, b_zs, r_c, r_s, z_c, z_s = args
        return interior_guess(
            boundary_R_cos=b_rc, boundary_R_sin=b_rs,
            boundary_Z_cos=b_zc, boundary_Z_sin=b_zs,
            raxis_c=r_c, raxis_s=r_s, zaxis_c=z_c, zaxis_s=z_s,
            modes=modes, trig=trig, s=setup.s_full,
        )

    args = (setup.boundary_R_cos, setup.boundary_R_sin, setup.boundary_Z_cos,
            setup.boundary_Z_sin, setup.raxis_c, setup.raxis_s,
            setup.zaxis_c, setup.zaxis_s)
    eager = produce_state(args)
    jitted = jax.jit(produce_state)(args)
    names = ("R_cos", "R_sin", "Z_cos", "Z_sin", "lambda_cos", "lambda_sin")
    for name, a, b in zip(names, eager, jitted):
        _allclose(b, a, f"jit {name}")
    # The jitted state reproduces the RunSetup state (same axis inputs).
    for name, a in zip(names, jitted):
        _allclose(a, getattr(setup, name), f"jit vs run_setup {name}")


def test_geometry_chain_from_setup_is_jittable(case):
    setup, modes, trig = case.setup, case.modes_new, case.trig_new

    def sign_flag(state_setup):
        geom = real_space_geometry(
            **geometry_state(state_setup, modes=modes),
            modes=modes, trig=trig, s=state_setup.s_full,
        )
        jac = half_mesh_jacobian(geom, s=state_setup.s_full)
        return jac.jacobian_sign_changed, jac.tau

    flag_eager, tau_eager = sign_flag(setup)
    flag_jit, tau_jit = jax.jit(sign_flag)(setup)
    assert bool(flag_eager) == bool(flag_jit) == False  # noqa: E712
    _allclose(tau_jit, tau_eager, "tau (jit)")


@pytest.mark.parametrize("rfix", [False, True])
def test_tvolume_scales_with_length_and_has_the_documented_null_direction(rfix):
    """scale_input keeps a TVOLUME deck's r_scale, and scaling the rescaled
    harmonics leaves the rescaled boundary unchanged."""
    inp = dataclasses.replace(VmecInput.from_file(DATA_DIR / "input.li383_low_res"),
                              tvolume=4.0, lvolume_rfix=rfix)
    scaled = scale_input(inp, r_scale=2.0)
    assert scaled.tvolume == 32.0
    assert scale_input(dataclasses.replace(inp, tvolume=-1.0), r_scale=2.0).tvolume == -1.0
    base = rescale_boundary_volume(inp.rbc, inp.rbs, inp.zbc, inp.zbs, inp)
    doubled = rescale_boundary_volume(scaled.rbc, scaled.rbs, scaled.zbc, scaled.zbs, scaled)
    np.testing.assert_allclose(doubled[0], 2.0 * np.asarray(base[0]), atol=1e-12)
    factor = np.where(np.arange(inp.mpol) > 0, 1.1, 1.0) if rfix else 1.1
    moved = rescale_boundary_volume(*(np.asarray(a) * factor for a in
                                      (inp.rbc, inp.rbs, inp.zbc, inp.zbs)), inp)
    np.testing.assert_allclose(moved[0], base[0], atol=1e-11)
    np.testing.assert_allclose(moved[3], base[3], atol=1e-11)


@pytest.mark.parametrize("rfix", [False, True])
def test_tvolume_rescales_boundary_to_the_exact_volume(rfix, tmp_path):
    """TVOLUME hits the target exactly, LVOLUME_RFIX holds m = 0, and the
    traced rescaling makes the volume independent of the boundary dofs."""
    inp = VmecInput.from_file(
        Path(__file__).resolve().parents[1] / "examples/data/input.li383_low_res")
    inp = dataclasses.replace(inp, tvolume=4.0, lvolume_rfix=rfix)
    assert VmecInput.from_file(inp.to_indata(tmp_path / "input.v")).tvolume == 4.0

    def volume(rbc):  # brute-force 2 pi^2 <R^2 Z_theta> on a fine grid
        rbc, _, _, zbs = rescale_boundary_volume(rbc, inp.rbs, inp.zbc, inp.zbs, inp)
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 64, endpoint=False),
                           np.linspace(0, 2 * np.pi / inp.nfp, 32, endpoint=False))
        m, n = np.arange(inp.mpol), np.arange(-inp.ntor, inp.ntor + 1)[:, None]
        a = m * u[..., None, None] - n * inp.nfp * v[..., None, None]
        r = jnp.sum(rbc * np.cos(a), axis=(-2, -1))
        zu = jnp.sum(m * zbs * np.cos(a), axis=(-2, -1))
        return 2 * np.pi**2 * jnp.abs(jnp.mean(r**2 * zu))

    rbc = jnp.asarray(inp.rbc)
    np.testing.assert_allclose(volume(rbc), 4.0, rtol=1e-13)
    assert float(jnp.max(jnp.abs(jax.grad(volume)(rbc)))) < 1e-12
    scaled = rescale_boundary_volume(rbc, inp.rbs, inp.zbc, inp.zbs, inp)
    assert np.allclose(scaled[0][:, 0], rbc[:, 0]) == rfix
    rescaled = dataclasses.replace(inp, **dict(zip(("rbc", "rbs", "zbc", "zbs"),
                                                   map(np.asarray, scaled))))
    np.testing.assert_allclose(boundary_volume(rescaled), 4.0, rtol=1e-13)


@pytest.mark.parametrize("rfix", [False, True])
@pytest.mark.parametrize("nfp, lasym, flip", [(1, False, 1.0), (5, False, -1.0), (2, True, 1.0)])
def test_tvolume_elliptical_torus_closed_form(rfix, nfp, lasym, flip):
    """R = R0 + a cos u + s sin u, Z = b sin u + e cos u encloses 2 pi^2 R0 (a b - s e)
    for any nfp and orientation.  Scaling every harmonic gives lambda^3 V = T; holding
    m = 0 gives c^2 V = T, where Pappus' theorem is exact (the centroid stays at R0)."""
    r0, a, b, (s, e), target = 3.0, 0.8, 1.3, ((0.2, 0.3) if lasym else (0.0, 0.0)), 40.0
    inp = dataclasses.replace(
        VmecInput.from_file(DATA_DIR / "input.circular_tokamak"), nfp=nfp, lasym=lasym,
        tvolume=target, lvolume_rfix=rfix)
    mode = lambda m, v: np.eye(1, inp.mpol, m) * v  # noqa: E731
    rbc, rbs = mode(0, r0) + mode(1, a), mode(1, s)
    zbs, zbc = flip * mode(1, b), flip * mode(1, e)
    volume = 2 * np.pi**2 * r0 * (a * b - s * e)
    scale = np.sqrt(target / volume) if rfix else np.cbrt(target / volume)
    out = rescale_boundary_volume(rbc, rbs, zbc, zbs, inp)
    held = np.eye(1, inp.mpol, 0)
    for got, ref in zip(out, (rbc, rbs, zbc, zbs)):
        want = scale * ref + ((1 - scale) * ref * held if rfix else 0.0)
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("tvolume, lfreeb", [(-1.0, False), (0.0, False), (4.0, True)])
def test_tvolume_is_a_noop_when_off_or_free_boundary(tvolume, lfreeb):
    """Missing, zero and negative targets leave the boundary alone; so does a free-boundary
    run (VMEC2000 RESCALE_BOUNDARY requires .not. lfreeb): the coils set the size."""
    inp = VmecInput.from_file(DATA_DIR / "input.li383_low_res")
    inp = dataclasses.replace(inp, tvolume=tvolume, lfreeb=lfreeb, mgrid_file="mgrid.nc")
    arrays = (inp.rbc, inp.rbs, inp.zbc, inp.zbs)
    for got, ref in zip(rescale_boundary_volume(*arrays, inp), arrays):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    base = _internal_blocks_from_input(dataclasses.replace(inp, tvolume=-1.0))[0]
    for name, block in _internal_blocks_from_input(inp)[0].items():
        np.testing.assert_array_equal(block, base[name])


def _vmec2000_rescale(inp, rbc, zbs):
    """Port of RESCALE_BOUNDARY (STELLOPT VMEC2000, vmec_input.f), stellarator symmetric:
    volume 2 pi |<R Z dR/du>| and area |<Z dR/du>| on a (u, v) fraction grid, Pappus
    major radius V / (2 pi A), and all harmonics scaled before the m = 0 row is restored."""
    u, v = np.meshgrid(np.arange(64) / 64, np.arange(64) / 64, indexing="ij")
    n = np.arange(-inp.ntor, inp.ntor + 1)[:, None]
    ang = 2 * np.pi * (np.arange(inp.mpol) * u[..., None, None] - n * v[..., None, None])
    rz = lambda r, z: (np.sum(r * np.cos(ang), (-2, -1)), np.sum(z * np.sin(ang), (-2, -1)),  # noqa: E731
                       -2 * np.pi * np.sum(np.arange(inp.mpol) * r * np.sin(ang), (-2, -1)))
    r, z, ru = rz(rbc, zbs)
    vol, area = 2 * np.pi * abs(np.mean(r * z * ru)), abs(np.mean(z * ru))
    if not inp.lvolume_rfix:
        return rbc * np.cbrt(inp.tvolume / vol), zbs * np.cbrt(inp.tvolume / vol), vol
    ar00 = vol / (2 * np.pi * area)
    f = np.sqrt(inp.tvolume / (2 * np.pi * ar00) / area)
    keep = np.zeros_like(rbc, dtype=bool)
    keep[inp.ntor:, 0] = True
    return np.where(keep, rbc, rbc * f), np.where(keep, zbs, zbs * f), vol


@pytest.mark.parametrize("rfix", [False, True])
def test_tvolume_matches_the_vmec2000_reference_formula(rfix):
    """Without LVOLUME_RFIX VMEX equals the VMEC2000 harmonics to round-off.  With it
    VMEC2000 is a Pappus estimate (volume off the target at the percent level for a
    shaped 3D boundary); VMEX is exact, and the two agree to that estimate's error."""
    inp = dataclasses.replace(VmecInput.from_file(DATA_DIR / "input.li383_low_res"),
                              tvolume=4.0, lvolume_rfix=rfix)
    rbc_ref, zbs_ref, _ = _vmec2000_rescale(inp, np.asarray(inp.rbc), np.asarray(inp.zbs))
    rbc, _, _, zbs = rescale_boundary_volume(inp.rbc, inp.rbs, inp.zbc, inp.zbs, inp)
    tol = 2e-3 if rfix else 1e-13
    np.testing.assert_allclose(rbc, rbc_ref, rtol=0.0, atol=tol)
    np.testing.assert_allclose(zbs, zbs_ref, rtol=0.0, atol=tol)
