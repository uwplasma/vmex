"""A/B equivalence tests: ``vmec_jax.core.setup`` vs the legacy setup path.

Old implementations under test (left untouched):

- ``vmec_jax.energy.flux_profiles_from_indata``           (profil1d.f fluxes)
- ``vmec_jax.profiles.eval_profiles``                     (piota/pcurr/pmass)
- ``vmec_jax.solvers.fixed_boundary.profiles``            (mass / icurv lanes)
- ``vmec_jax.kernels.residue.vmec_scalxc_from_s``         (profil3d.f scalxc)
- ``vmec_jax.boundary`` (readin.f boundary chain incl. theta flip and the
  lconm1 m=1 conversion)
- ``vmec_jax.init_guess.initial_guess_from_boundary``     (profil3d.f guess)
- ``vmec_jax.init_guess._recompute_axis_from_state_vmec`` (guess_axis.f)

New implementation: ``vmec_jax.core.setup`` (radial_grids / flux_profiles /
boundary_from_input / interior_guess / guess_axis / run_setup), consumed
through the already-proven new chain ``core.geometry.real_space_geometry ->
half_mesh_jacobian`` for the well-posedness (no Jacobian sign change) checks.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

# Pin the legacy kernels to their CPU DFT lane (deterministic A/B reference).
os.environ.setdefault("VMEC_JAX_TOMNSPS_FFT", "0")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from vmec_jax.boundary import (
    _boundary_helical_from_internal,
    _boundary_internal_flip_theta,
    _boundary_internal_from_helical,
    boundary_apply_vmec_m1_constraint,
    boundary_input_from_indata,
)
from vmec_jax.config import load_config
from vmec_jax.energy import flux_profiles_from_indata
from vmec_jax.init_guess import (
    _recompute_axis_from_state_vmec,
    initial_guess_from_boundary,
)
from vmec_jax.kernels.residue import vmec_scalxc_from_s
from vmec_jax.profiles import eval_profiles
from vmec_jax.solvers.fixed_boundary.profiles import (
    _icurv_full_mesh_from_indata,
    _mass_half_mesh_from_indata,
)
from vmec_jax.static import build_static

from vmec_jax.core.fourier import Resolution, mode_table, trig_tables
from vmec_jax.core.geometry import (
    half_mesh_jacobian,
    real_space_geometry,
    sqrt_s_half_mesh,
)
from vmec_jax.core.input import VmecInput
from vmec_jax.core.setup import (
    geometry_state,
    guess_axis,
    interior_guess,
    run_setup,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "examples" / "data"

RTOL = 1e-12
ATOL = 1e-13

SIGNGS = -1

# 2D sym pressure-driven / 2D sym nfp=5 current-driven (two_power, phiedge<0)
# / 3D sym (lconm1 m=1, axis inference) / lasym (axis inference) / high-mpol
# multigrid deck (legacy RAXIS name, ns=128).
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
    """Build the legacy setup and the new setup for one input deck."""
    name = request.param
    deck = DATA_DIR / f"input.{name}"

    # ------------------------------------------------------------------ old
    cfg, indata = load_config(str(deck))
    static = build_static(cfg)
    s = np.asarray(static.s)
    ns = int(s.shape[0])
    s_half_old = np.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])])

    old_flux = flux_profiles_from_indata(indata, jnp.asarray(s), signgs=SIGNGS)
    phips_old = np.asarray(old_flux.phips).copy()
    phips_old[0] = 0.0

    prof_half = eval_profiles(indata, s_half_old)
    prof_full = eval_profiles(indata, s)
    iotas_old = np.asarray(prof_half.get("iota", np.zeros(ns))).copy()
    iotas_old[0] = 0.0
    iotaf_old = np.asarray(prof_full.get("iota", np.zeros(ns)))
    # profil1d.f: chips = torflux_edge * polflux_deriv = phips * iotas for the
    # default APHI (torflux == identity), which all decks here use.
    chips_old = phips_old * iotas_old

    # Old boundary in solver convention, assembled from the legacy building
    # blocks but with the readin.f theta-flip rule computed over *all* n
    # columns.  (The legacy top-level ``boundary_from_input_convention``
    # sums ``rbcc[1:, 1]``, skipping n = 0; VMEC2000's ``rtest =
    # SUM(rbcc(1:ntor1, m1))`` covers n = 0..ntor because the readin.f
    # pointers are 1-based.  For li383 the two rules disagree — the legacy
    # code flips where VMEC2000 does not — so the test wires the flip
    # decision itself and A/Bs everything downstream of it.)
    boundary_in = boundary_input_from_indata(indata, static.modes)
    internal = _boundary_internal_from_helical(
        boundary_in, static.modes, lthreed=cfg.lthreed, lasym=cfg.lasym
    )
    rtest = float(np.sum(np.asarray(internal.rbcc)[:, 1]))
    ztest = float(np.sum(np.asarray(internal.zbsc)[:, 1]))
    if rtest * ztest < 0.0:
        internal = _boundary_internal_flip_theta(
            internal, lthreed=cfg.lthreed, lasym=cfg.lasym
        )
    old_boundary = _boundary_helical_from_internal(
        internal, static.modes, lthreed=cfg.lthreed, lasym=cfg.lasym
    )
    if cfg.lthreed or cfg.lasym:
        old_boundary_con = boundary_apply_vmec_m1_constraint(
            old_boundary, static.modes, lthreed=cfg.lthreed, lasym=cfg.lasym
        )
    else:
        old_boundary_con = old_boundary
    mode_scale = np.asarray(static.mode_scale_internal)

    m_old = np.asarray(static.modes.m)
    n_old = np.asarray(static.modes.n)
    idx00 = int(np.nonzero((m_old == 0) & (n_old == 0))[0][0])
    r00 = float(np.asarray(old_boundary.R_cos)[idx00])
    gamma = float(indata.get_float("GAMMA", 0.0))
    mass_old = _mass_half_mesh_from_indata(
        indata=indata, s_full=jnp.asarray(s), phips=jnp.asarray(phips_old),
        r00=r00, gamma=gamma,
    )
    icurv_old = _icurv_full_mesh_from_indata(
        indata=indata, s_full=jnp.asarray(s), signgs=SIGNGS
    )
    scalxc_old = vmec_scalxc_from_s(s=jnp.asarray(s), mpol=int(cfg.mpol))

    old_state = initial_guess_from_boundary(
        static, old_boundary, indata, vmec_project=False, infer_axis_if_missing=True
    )

    # ------------------------------------------------------------------ new
    inp = VmecInput.from_file(deck)
    res = Resolution(
        mpol=int(cfg.mpol), ntor=int(cfg.ntor), ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta), nfp=int(cfg.nfp), lasym=bool(cfg.lasym), ns=ns,
    )
    setup = run_setup(inp, res)
    modes_new = mode_table(res.mpol, res.ntor)
    trig_new = trig_tables(res)

    geom = real_space_geometry(
        **geometry_state(setup, modes=modes_new),
        modes=modes_new, trig=trig_new, s=setup.s_full,
    )
    jac = half_mesh_jacobian(geom, s=setup.s_full)

    return SimpleNamespace(
        name=name, cfg=cfg, indata=indata, static=static, s=s, ns=ns,
        old_flux=old_flux, phips_old=phips_old, iotas_old=iotas_old,
        iotaf_old=iotaf_old, chips_old=chips_old, mass_old=mass_old,
        icurv_old=icurv_old, scalxc_old=scalxc_old,
        old_boundary_con=old_boundary_con, mode_scale=mode_scale,
        old_state=old_state, inp=inp, res=res, setup=setup,
        modes_new=modes_new, trig_new=trig_new, geom=geom, jac=jac,
    )


# ---------------------------------------------------------------------------
# Radial grids and scalxc (profil1d.f / profil3d.f)
# ---------------------------------------------------------------------------


def test_radial_grids_match_old(case):
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


def test_scalxc_matches_old(case):
    _allclose(case.setup.scalxc, case.scalxc_old, "scalxc")


# ---------------------------------------------------------------------------
# 1D profiles (profil1d.f)
# ---------------------------------------------------------------------------


def test_flux_profiles_match_old(case):
    setup = case.setup
    assert setup.signgs == SIGNGS
    assert setup.lflip is False
    _allclose(setup.phips, case.phips_old, "phips")
    _allclose(setup.phipf, case.old_flux.phipf, "phipf")
    _allclose(setup.chipf, case.old_flux.chipf, "chipf")
    _allclose(setup.chips, case.chips_old, "chips")
    _allclose(setup.iotas, case.iotas_old, "iotas")
    _allclose(setup.iotaf, case.iotaf_old, "iotaf")
    _allclose(setup.lamscale, case.old_flux.lamscale, "lamscale")


def test_mass_and_icurv_match_old(case):
    _allclose(case.setup.mass, case.mass_old, "mass")
    # Integrated pcurr lanes ('two_power'/'gauss_trunc') now use VMEC2000's
    # exact 10-point Gauss-Legendre rule (profile_functions.f); the legacy
    # evaluator used 16 points, deviating from VMEC2000 by ~2e-6 relative
    # (bug exposed by the end-to-end cth_like_fixed_bdy wout parity test).
    if str(case.inp.pcurr_type) in ("two_power", "gauss_trunc"):
        np.testing.assert_allclose(np.asarray(case.setup.icurv),
                                   np.asarray(case.icurv_old),
                                   rtol=5e-6, atol=1e-12,
                                   err_msg="icurv (quadrature lane) mismatch")
    else:
        _allclose(case.setup.icurv, case.icurv_old, "icurv")
    assert case.setup.ncurr == int(case.indata.get_int("NCURR", 0))
    if case.setup.ncurr == 1 and abs(float(case.inp.curtor)) > 0 and np.any(
        np.asarray(case.inp.ac)
    ):
        # The current-driven deck must exercise a nonzero pcurr lane.
        assert np.any(np.asarray(case.setup.icurv) != 0.0)


# ---------------------------------------------------------------------------
# Boundary processing (readin.f) and initial state (profil3d.f)
# ---------------------------------------------------------------------------


def test_boundary_matches_old(case):
    # Old chain: readin.f accumulation + theta flip + lconm1 m=1 conversion,
    # then the profil3d.f mscale*nscale internal normalization.
    setup, scale = case.setup, case.mode_scale
    old = case.old_boundary_con
    _allclose(setup.boundary_R_cos, np.asarray(old.R_cos) * scale, "boundary_R_cos")
    _allclose(setup.boundary_R_sin, np.asarray(old.R_sin) * scale, "boundary_R_sin")
    _allclose(setup.boundary_Z_cos, np.asarray(old.Z_cos) * scale, "boundary_Z_cos")
    _allclose(setup.boundary_Z_sin, np.asarray(old.Z_sin) * scale, "boundary_Z_sin")


def test_initial_state_matches_old(case):
    setup, old = case.setup, case.old_state
    _allclose(setup.R_cos, old.Rcos, "R_cos")
    _allclose(setup.R_sin, old.Rsin, "R_sin")
    _allclose(setup.Z_cos, old.Zcos, "Z_cos")
    _allclose(setup.Z_sin, old.Zsin, "Z_sin")
    _allclose(setup.lambda_cos, old.Lcos, "lambda_cos")
    _allclose(setup.lambda_sin, old.Lsin, "lambda_sin")
    # The edge surface is exactly the processed boundary.
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
# guess_axis (guess_axis.f) A/B against the legacy port
# ---------------------------------------------------------------------------


def test_guess_axis_matches_old(case):
    geom, setup = case.geom, case.setup
    new_axis = guess_axis(geom, s=setup.s_full, trig=case.trig_new, signgs=SIGNGS)
    old_axis = _recompute_axis_from_state_vmec(
        case.static,
        pr1_even=np.asarray(geom.R_even), pr1_odd=np.asarray(geom.R_odd),
        pz1_even=np.asarray(geom.Z_even), pz1_odd=np.asarray(geom.Z_odd),
        pru_even=np.asarray(geom.dR_dtheta_even), pru_odd=np.asarray(geom.dR_dtheta_odd),
        pzu_even=np.asarray(geom.dZ_dtheta_even), pzu_odd=np.asarray(geom.dZ_dtheta_odd),
        signgs=SIGNGS, trig=case.static.trig_vmec,
    )
    # Both return (raxis_c, raxis_s, zaxis_c, zaxis_s) (old names *_cc/*_cs).
    for new, old, name in zip(
        new_axis, old_axis, ("raxis_c", "raxis_s", "zaxis_c", "zaxis_s")
    ):
        _allclose(new, old, name)


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
