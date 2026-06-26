from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax import solve as solve_mod
from vmec_jax._compat import has_jax, jax, jnp
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.kernels.tomnsp import TomnspsRZL


pytestmark = pytest.mark.skipif(not has_jax(), reason="residual optimizers require JAX")


class _TinyInData:
    scalars = {}
    indexed = {}

    def get_float(self, name, default=0.0):
        return {"FTOL": 0.0, "TCON0": 1.0, "GAMMA": 0.0}.get(name, default)

    def get_bool(self, name, default=False):
        return {"LFORBAL": False, "LRFP": False}.get(name, default)

    def get_int(self, name, default=0):
        return {"NCURR": 0}.get(name, default)


def _tiny_static(*, lasym: bool = False):
    cfg = SimpleNamespace(
        ns=3,
        mpol=2,
        ntor=0,
        nfp=1,
        ntheta=4,
        nzeta=1,
        lasym=bool(lasym),
        lthreed=True,
        lconm1=True,
    )
    modes = SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, 0]), K=2)
    return SimpleNamespace(
        cfg=cfg,
        s=jnp.asarray([0.0, 0.5, 1.0]),
        modes=modes,
        trig_vmec=SimpleNamespace(name="fake-trig"),
        tomnsps_masks={"tiny": True},
    )


def _state_with_interior_rcos(value: float = 2.0) -> VMECState:
    layout = StateLayout(ns=3, K=2, lasym=False)
    z = np.zeros((layout.ns, layout.K), dtype=float)
    rcos = z.copy()
    rcos[1, 0] = float(value)
    return VMECState(layout=layout, Rcos=rcos, Rsin=z, Zcos=z, Zsin=z, Lcos=z, Lsin=z)


def _zeros_like_tomnsps(state: VMECState):
    dtype = jnp.asarray(state.Rcos).dtype
    return jnp.zeros((3, 2, 1), dtype=dtype)


def _single_r_residual(state: VMECState) -> TomnspsRZL:
    z = _zeros_like_tomnsps(state)
    x = jnp.asarray(state.Rcos)[1, 0]
    return TomnspsRZL(
        frcc=z.at[1, 0, 0].set(x),
        frss=None,
        fzsc=z,
        fzcs=None,
        flsc=z,
        flcs=None,
    )


def _nonfinite_trial_residual(state: VMECState) -> TomnspsRZL:
    z = _zeros_like_tomnsps(state)
    x = jnp.asarray(state.Rcos)[1, 0]
    r = jax.lax.cond(
        x < jnp.asarray(1.9, dtype=x.dtype),
        lambda _: jnp.asarray(jnp.nan, dtype=x.dtype),
        lambda _: x,
        operand=None,
    )
    return TomnspsRZL(
        frcc=z.at[1, 0, 0].set(r),
        frss=None,
        fzsc=z,
        fzcs=None,
        flsc=z,
        flcs=None,
    )


def _lasym_optional_only_residual(state: VMECState) -> TomnspsRZL:
    z = _zeros_like_tomnsps(state)
    x = jnp.asarray(state.Rcos)[1, 0]

    def block(scale: float):
        return z.at[1, 0, 0].set(jnp.asarray(scale, dtype=x.dtype) * x)

    return TomnspsRZL(
        frcc=z,
        frss=block(0.25),
        fzsc=z,
        fzcs=block(0.50),
        flsc=z,
        flcs=block(0.75),
        frsc=block(1.00),
        frcs=block(1.25),
        fzcc=block(1.50),
        fzss=block(1.75),
        flcc=block(2.00),
        flss=block(2.25),
    )


def _install_fake_physics(monkeypatch, residual_from_state, *, sqrtg_value: float = 1.0):
    import vmec_jax.boundary as boundary_mod
    import vmec_jax.energy as energy_mod
    import vmec_jax.kernels.forces as forces_mod
    import vmec_jax.kernels.residue as residue_mod

    monkeypatch.setattr(
        energy_mod,
        "flux_profiles_from_indata",
        lambda _indata, s, signgs: SimpleNamespace(
            chipf=jnp.zeros_like(jnp.asarray(s)),
            phips=jnp.ones_like(jnp.asarray(s)),
            phipf=jnp.ones_like(jnp.asarray(s)),
        ),
    )
    monkeypatch.setattr(
        boundary_mod,
        "boundary_from_indata",
        lambda _indata, _modes: SimpleNamespace(R_cos=np.asarray([0.0, 0.0])),
    )
    monkeypatch.setattr(solve_mod, "_mass_half_mesh_from_indata", lambda **_kwargs: jnp.zeros(3))
    monkeypatch.setattr(solve_mod, "_pressure_half_mesh_from_indata", lambda **_kwargs: jnp.zeros(3))
    monkeypatch.setattr(solve_mod, "_icurv_full_mesh_from_indata", lambda **_kwargs: jnp.zeros(3))
    monkeypatch.setattr(
        solve_mod,
        "_vmec_force_flux_profiles",
        lambda **kwargs: (jnp.asarray(kwargs["phipf"]), jnp.asarray(kwargs["chipf"]), jnp.asarray(kwargs["chipf"])),
    )

    def fake_forces(*, state, **_kwargs):
        sqrtg = jnp.full((3, 1, 1), float(sqrtg_value), dtype=jnp.asarray(state.Rcos).dtype)
        return SimpleNamespace(state=state, bc=SimpleNamespace(jac=SimpleNamespace(sqrtg=sqrtg)))

    monkeypatch.setattr(forces_mod, "vmec_forces_rz_from_wout", fake_forces)
    monkeypatch.setattr(
        forces_mod,
        "vmec_residual_internal_from_kernels",
        lambda k, **_kwargs: residual_from_state(k.state),
    )
    monkeypatch.setattr(
        residue_mod,
        "vmec_force_norms_from_bcovar_dynamic",
        lambda **_kwargs: SimpleNamespace(
            r1=jnp.asarray(1.0),
            fnorm=jnp.asarray(1.0),
            fnormL=jnp.asarray(1.0),
        ),
    )


def test_lbfgs_auto_scales_accepts_one_step_and_stops_on_nonfinite_trial(monkeypatch):
    static = _tiny_static()
    indata = _TinyInData()

    _install_fake_physics(monkeypatch, _single_r_residual)
    accepted = solve_mod.solve_fixed_boundary_lbfgs_vmec_residual(
        _state_with_interior_rcos(2.0),
        static,
        indata=indata,
        signgs=1,
        include_constraint_force=False,
        max_iter=1,
        step_size=1.0,
        max_backtracks=0,
        verbose=False,
    )

    assert accepted.n_iter == 1
    assert accepted.diagnostics["objective_scale"] == pytest.approx(0.25)
    np.testing.assert_allclose(accepted.w_history, [1.0, 0.25])
    np.testing.assert_allclose(accepted.step_history, [1.0])

    _install_fake_physics(monkeypatch, _nonfinite_trial_residual)
    stopped = solve_mod.solve_fixed_boundary_lbfgs_vmec_residual(
        _state_with_interior_rcos(2.0),
        static,
        indata=indata,
        signgs=1,
        include_constraint_force=False,
        max_iter=1,
        step_size=1.0,
        max_backtracks=0,
        verbose=False,
    )

    assert stopped.n_iter == 0
    np.testing.assert_allclose(stopped.w_history, [1.0])
    np.testing.assert_allclose(stopped.step_history, [1.0])


def test_lbfgs_builds_missing_trig_jits_gradient_and_warns_on_negative_jacobian(monkeypatch, capsys):
    import vmec_jax.kernels.tomnsp as tomnsp_mod

    static = _tiny_static()
    static.trig_vmec = None
    generated_trig = SimpleNamespace(name="generated-trig")
    trig_calls = []
    monkeypatch.setattr(
        tomnsp_mod,
        "vmec_trig_tables",
        lambda **kwargs: trig_calls.append(kwargs) or generated_trig,
    )
    _install_fake_physics(monkeypatch, _single_r_residual, sqrtg_value=-1.0)

    result = solve_mod.solve_fixed_boundary_lbfgs_vmec_residual(
        _state_with_interior_rcos(2.0),
        static,
        indata=_TinyInData(),
        signgs=1,
        include_constraint_force=False,
        max_iter=1,
        jit_grad=True,
        verbose=True,
    )

    assert trig_calls
    assert trig_calls[0]["ntheta"] == 4
    assert trig_calls[0]["mmax"] == 1
    assert result.n_iter >= 0
    assert result.diagnostics["objective_scale"] == pytest.approx(0.25)
    assert "initial Jacobian has non-positive entries" in capsys.readouterr().out


def test_gn_lasym_optional_blocks_auto_scale_and_accept_real_cg_step(monkeypatch):
    static = _tiny_static(lasym=True)
    _install_fake_physics(monkeypatch, _lasym_optional_only_residual)

    result = solve_mod.solve_fixed_boundary_gn_vmec_residual(
        _state_with_interior_rcos(2.0),
        static,
        indata=_TinyInData(),
        signgs=1,
        include_constraint_force=False,
        apply_m1_constraints=False,
        damping=1.0e-3,
        cg_tol=1.0e-7,
        cg_maxiter=20,
        max_iter=1,
        jit_kernels=False,
        verbose=False,
    )

    unscaled_initial = result.fsqr2_history[0] + result.fsqz2_history[0] + result.fsql2_history[0]
    assert result.n_iter == 1
    assert result.fsqr2_history[0] > 0.0
    assert result.fsqz2_history[0] > 0.0
    assert result.fsql2_history[0] > 0.0
    assert result.diagnostics["objective_scale"] == pytest.approx(1.0 / unscaled_initial)
    assert result.diagnostics["damping_mode"] == "fixed"
    assert result.diagnostics["cg_tol_mode"] == "fixed"
    assert result.w_history[0] == pytest.approx(1.0)
    assert result.w_history[1] < result.w_history[0]


def test_gn_retries_with_more_damping_then_fallback_descent_accepts(monkeypatch):
    import jax.scipy.sparse.linalg as sparse_linalg

    static = _tiny_static()
    _install_fake_physics(monkeypatch, _single_r_residual)

    cg_calls = []

    def fake_cg(matvec, b, *, tol, maxiter):
        cg_calls.append((float(tol), int(maxiter), np.asarray(matvec(jnp.zeros_like(b))).shape))
        return -b, None

    monkeypatch.setattr(sparse_linalg, "cg", fake_cg)

    result = solve_mod.solve_fixed_boundary_gn_vmec_residual(
        _state_with_interior_rcos(2.0),
        static,
        indata=_TinyInData(),
        signgs=1,
        include_constraint_force=False,
        objective_scale=2.0,
        damping=1.0e-6,
        damping_increase=10.0,
        max_retries=1,
        max_damping=1.0e-3,
        cg_tol=1.0e-5,
        cg_maxiter=3,
        max_iter=1,
        max_backtracks=0,
        jit_kernels=False,
        verbose=False,
    )

    assert len(cg_calls) == 2
    assert cg_calls[0][:2] == pytest.approx((1.0e-5, 3))
    assert result.n_iter == 1
    assert result.diagnostics["objective_scale"] == pytest.approx(2.0)
    np.testing.assert_allclose(result.w_history, [8.0, 0.0])
    np.testing.assert_allclose(result.step_history, [1.0])
