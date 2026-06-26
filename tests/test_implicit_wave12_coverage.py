from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import has_jax
from vmec_jax.state import StateLayout, VMECState


pytestmark = pytest.mark.skipif(not has_jax(), reason="implicit residual custom VJP branches require JAX")


class _Indata:
    def __init__(self, scalars=None):
        self.scalars = {str(k).upper(): v for k, v in dict(scalars or {}).items()}

    def get(self, name, default=None):
        return self.scalars.get(str(name).upper(), default)

    def get_bool(self, name, default=False):
        value = self.get(name, default)
        if isinstance(value, list):
            value = value[0] if value else default
        return bool(value)

    def get_float(self, name, default=0.0):
        value = self.get(name, default)
        if isinstance(value, list):
            value = value[0] if value else default
        return float(value)

    def get_int(self, name, default=0):
        value = self.get(name, default)
        if isinstance(value, list):
            value = value[0] if value else default
        return int(value)


def _state(*, xp=np) -> VMECState:
    layout = StateLayout(ns=2, K=1, lasym=False)
    zeros = xp.zeros((2, 1), dtype=float)
    return VMECState(
        layout=layout,
        Rcos=xp.asarray([[1.0], [2.0]], dtype=float),
        Rsin=zeros,
        Zcos=zeros,
        Zsin=xp.asarray([[0.0], [0.5]], dtype=float),
        Lcos=zeros,
        Lsin=xp.asarray([[0.0], [0.1]], dtype=float),
    )


def _static(*, lasym: bool = False):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            ns=2,
            nfp=1,
            mpol=1,
            ntor=0,
            ntheta=2,
            nzeta=1,
            lasym=bool(lasym),
            lconm1=True,
        ),
        modes=SimpleNamespace(m=np.asarray([0], dtype=int), n=np.asarray([0], dtype=int), K=1),
        basis=None,
        grid=SimpleNamespace(theta=np.asarray([0.0, np.pi]), zeta=np.asarray([0.0])),
        s=np.asarray([0.0, 1.0]),
        trig_vmec=SimpleNamespace(),
    )


def _install_residual_fakes(monkeypatch, implicit, jnp, state0):
    import vmec_jax.boundary as boundary_module
    import vmec_jax.init_guess as init_guess_module
    import vmec_jax.preconditioner_1d_jax as preconditioner_module
    import vmec_jax.kernels.forces as forces_module
    import vmec_jax.kernels.residue as residue_module
    from vmec_jax.boundary import BoundaryCoeffs
    from vmec_jax.kernels.tomnsp import TomnspsRZL

    boundary = BoundaryCoeffs(
        R_cos=np.asarray([2.0]),
        R_sin=np.asarray([0.0]),
        Z_cos=np.asarray([0.0]),
        Z_sin=np.asarray([0.5]),
    )

    def fake_initial_guess(_static, boundary_arg, *_args, dtype=None, **_kwargs):
        dtype = dtype or jnp.asarray(state0.Rcos).dtype
        return VMECState(
            layout=state0.layout,
            Rcos=jnp.asarray(state0.Rcos, dtype=dtype).at[-1].set(jnp.asarray(boundary_arg.R_cos, dtype=dtype)),
            Rsin=jnp.asarray(state0.Rsin, dtype=dtype).at[-1].set(jnp.asarray(boundary_arg.R_sin, dtype=dtype)),
            Zcos=jnp.asarray(state0.Zcos, dtype=dtype).at[-1].set(jnp.asarray(boundary_arg.Z_cos, dtype=dtype)),
            Zsin=jnp.asarray(state0.Zsin, dtype=dtype).at[-1].set(jnp.asarray(boundary_arg.Z_sin, dtype=dtype)),
            Lcos=jnp.asarray(state0.Lcos, dtype=dtype),
            Lsin=jnp.asarray(state0.Lsin, dtype=dtype),
        )

    def fake_residual_from_kernels(k, **_kwargs):
        r = jnp.asarray(k.state.Rcos)[:, :, None]
        z = jnp.asarray(k.state.Zsin)[:, :, None]
        l = jnp.asarray(k.state.Lsin)[:, :, None]
        return TomnspsRZL(
            frcc=r + 0.25 * z,
            frss=None,
            fzsc=0.5 * z,
            fzcs=None,
            flsc=l + 0.1 * r,
            flcs=None,
        )

    monkeypatch.setattr(
        implicit,
        "flux_profiles_from_indata",
        lambda *_args, **_kwargs: SimpleNamespace(
            phipf=jnp.asarray([0.0, 1.0]),
            phips=jnp.asarray([0.0, 1.0]),
            chipf=jnp.asarray([0.0, 0.2]),
        ),
    )
    monkeypatch.setattr(implicit, "_mass_half_mesh_from_indata", lambda **_kwargs: jnp.asarray([0.0, 1.0]))
    monkeypatch.setattr(implicit, "_pressure_half_mesh_from_indata", lambda **_kwargs: jnp.asarray([0.0, 0.0]))
    monkeypatch.setattr(implicit, "_icurv_full_mesh_from_indata", lambda **_kwargs: jnp.asarray([0.0, 0.0]))
    monkeypatch.setattr(
        implicit,
        "_vmec_force_flux_profiles",
        lambda **kwargs: (kwargs["phipf"], kwargs["chipf"], kwargs["chipf"]),
    )
    monkeypatch.setattr(boundary_module, "boundary_from_indata", lambda *_args, **_kwargs: boundary)
    monkeypatch.setattr(init_guess_module, "initial_guess_from_boundary", fake_initial_guess)
    monkeypatch.setattr(implicit, "eval_geom", lambda *_args, **_kwargs: SimpleNamespace(sqrtg=np.ones((2, 1, 1))))
    monkeypatch.setattr(implicit, "signgs_from_sqrtg", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        implicit,
        "solve_fixed_boundary_residual_iter",
        lambda state, *_args, **_kwargs: SimpleNamespace(state=state, n_iter=1, fsqz2_history=np.asarray([1e-8])),
    )
    monkeypatch.setattr(forces_module, "vmec_forces_rz_from_wout", lambda state, **_kwargs: SimpleNamespace(state=state, bc=state))
    monkeypatch.setattr(forces_module, "vmec_residual_internal_from_kernels", fake_residual_from_kernels)
    monkeypatch.setattr(residue_module, "vmec_apply_m1_constraints", lambda *, frzl, **_kwargs: frzl)
    monkeypatch.setattr(residue_module, "vmec_zero_m1_zforce", lambda *, frzl, **_kwargs: frzl)
    monkeypatch.setattr(residue_module, "vmec_apply_scalxc_to_tomnsps", lambda *, frzl, **_kwargs: frzl)
    monkeypatch.setattr(
        residue_module,
        "vmec_force_norms_from_bcovar_dynamic",
        lambda **_kwargs: SimpleNamespace(r1=1.0, fnorm=1.0, fnormL=1.0),
    )
    monkeypatch.setattr(preconditioner_module, "lambda_preconditioner_cached", lambda **_kwargs: jnp.ones((2, 1, 1)))


def _base_options(implicit, *, residual_adjoint_mode="auto", residual_tangent_mode="opaque"):
    return implicit.ImplicitFixedBoundaryOptions(
        cg_max_iter=4,
        cg_tol=1.0e-10,
        damping=0.2,
        residual_adjoint_mode=residual_adjoint_mode,
        residual_tangent_mode=residual_tangent_mode,
        jac_chunk_size=1,
    )


def _solve_from_scale(implicit, state0, static, jnp, scale, *, options):
    return implicit.solve_fixed_boundary_state_implicit_vmec_residual(
        state0,
        static,
        indata=_Indata({"NCURR": 0, "LRFP": False, "GAMMA": 0.0, "TCON0": 1.0}),
        signgs=1,
        max_iter=1,
        step_size=0.2,
        implicit=options,
        edge_Rcos=scale * jnp.asarray([2.0]),
        edge_Rsin=jnp.asarray([0.0]),
        edge_Zcos=jnp.asarray([0.0]),
        edge_Zsin=scale * jnp.asarray([0.5]),
    )


def test_vmec_residual_custom_vjp_routes_active_direct_bicgstab(monkeypatch):
    import jax.scipy.sparse.linalg as sparse_linalg
    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state0 = _state(xp=jnp)
    static = _static()
    _install_residual_fakes(monkeypatch, implicit, jnp, state0)
    calls = []

    def fake_bicgstab(matvec, b, *, tol, atol, maxiter):
        calls.append((np.asarray(matvec(jnp.ones_like(b))).shape, tol, atol, maxiter))
        return 0.5 * b, None

    monkeypatch.setattr(sparse_linalg, "bicgstab", fake_bicgstab)
    options = _base_options(implicit, residual_adjoint_mode="direct")

    def objective(scale):
        st = _solve_from_scale(implicit, state0, static, jnp, scale, options=options)
        return st.Rcos[-1, 0] + 0.3 * st.Zsin[-1, 0]

    grad = float(jax.grad(objective)(1.1))

    assert calls == [((1,), 1.0e-10, 0.0, 4)]
    assert np.isfinite(grad)


def test_vmec_residual_custom_vjp_routes_active_lineax_success(monkeypatch):
    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state0 = _state(xp=jnp)
    static = _static()
    _install_residual_fakes(monkeypatch, implicit, jnp, state0)
    calls = []

    def fake_lineax(matvec, b, *, tol, max_iter, **_kwargs):
        calls.append((np.asarray(matvec(jnp.ones_like(b))).shape, tol, max_iter))
        return 0.25 * b, True, {"num_steps": jnp.asarray(3)}

    monkeypatch.setattr(implicit, "_lineax_bicgstab_solve", fake_lineax)
    options = _base_options(implicit, residual_adjoint_mode="lineax")

    def objective(scale):
        st = _solve_from_scale(implicit, state0, static, jnp, scale, options=options)
        return st.Rcos[-1, 0] - 0.2 * st.Zsin[-1, 0]

    grad = float(jax.grad(objective)(1.05))

    assert calls == [((1,), 1.0e-10, 4)]
    assert np.isfinite(grad)


def test_vmec_residual_custom_vjp_active_cg_and_full_cg_fallbacks(monkeypatch):
    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state0 = _state(xp=jnp)
    static = _static()
    _install_residual_fakes(monkeypatch, implicit, jnp, state0)
    calls = []
    original_cg = implicit._cg_solve

    def recording_cg(matvec, rhs, *, tol, max_iter):
        calls.append((np.asarray(rhs).shape, tol, max_iter))
        matvec(jnp.ones_like(rhs))
        return original_cg(matvec, rhs, tol=tol, max_iter=max_iter)

    monkeypatch.setattr(implicit, "_cg_solve", recording_cg)
    options = _base_options(implicit, residual_adjoint_mode="auto")

    def objective(scale):
        st = _solve_from_scale(implicit, state0, static, jnp, scale, options=options)
        return 0.1 * st.Rcos[-1, 0] + st.Zsin[-1, 0]

    active_grad = float(jax.grad(objective)(1.0))
    monkeypatch.setenv("VMEC_JAX_IMPLICIT_DISABLE_REDUCED_ACTIVE", "1")
    full_grad = float(jax.grad(objective)(1.0))

    assert calls[0] == ((1,), 1.0e-10, 4)
    assert calls[1] == ((12,), 1.0e-10, 4)
    assert np.isfinite(active_grad)
    assert np.isfinite(full_grad)


def test_vmec_residual_custom_jvp_routes_lineax_and_rejects_lasym(monkeypatch):
    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state0 = _state(xp=jnp)
    static = _static()
    _install_residual_fakes(monkeypatch, implicit, jnp, state0)
    calls = []

    def fake_lineax(matvec, rhs, *, tol, max_iter, **_kwargs):
        calls.append((np.asarray(matvec(jnp.ones_like(rhs))).shape, tol, max_iter))
        return rhs, True, {"num_steps": jnp.asarray(1)}

    monkeypatch.setattr(implicit, "_lineax_bicgstab_solve", fake_lineax)
    options = _base_options(implicit, residual_tangent_mode="lineax")

    def scalar(scale):
        st = _solve_from_scale(implicit, state0, static, jnp, scale, options=options)
        return st.Rcos[-1, 0] + 0.2 * st.Zsin[-1, 0]

    value, tangent = jax.jvp(scalar, (jnp.asarray(1.0),), (jnp.asarray(0.1),))

    assert calls == [((1,), 1.0e-10, 4)]
    assert float(value) == pytest.approx(2.1)
    assert np.isfinite(float(tangent))

    with pytest.raises(NotImplementedError, match="lasym=False"):
        jax.jvp(
            lambda scale: _solve_from_scale(
                implicit,
                state0,
                _static(lasym=True),
                jnp,
                scale,
                options=options,
            ).Rcos[-1, 0],
            (jnp.asarray(1.0),),
            (jnp.asarray(0.1),),
        )
