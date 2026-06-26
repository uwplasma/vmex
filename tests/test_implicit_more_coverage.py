from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import has_jax
from vmec_jax.state import StateLayout, VMECState


pytestmark = pytest.mark.skipif(not has_jax(), reason="additional implicit coverage requires JAX")


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
        Rcos=xp.asarray([[1.1], [2.0]], dtype=float),
        Rsin=zeros,
        Zcos=zeros,
        Zsin=xp.asarray([[0.2], [0.5]], dtype=float),
        Lcos=xp.asarray([[0.0], [0.03]], dtype=float),
        Lsin=xp.asarray([[0.0], [0.07]], dtype=float),
    )


def _static():
    return SimpleNamespace(
        cfg=SimpleNamespace(
            ns=2,
            nfp=1,
            mpol=1,
            ntor=0,
            ntheta=2,
            nzeta=1,
            lasym=False,
            lconm1=True,
        ),
        modes=SimpleNamespace(m=np.asarray([0], dtype=int), n=np.asarray([0], dtype=int), K=1),
        basis=None,
        grid=SimpleNamespace(theta=np.asarray([0.0, np.pi]), zeta=np.asarray([0.0])),
        s=np.asarray([0.0, 1.0]),
        trig_vmec=SimpleNamespace(),
    )


def _install_residual_fakes(monkeypatch, implicit, jnp, state0: VMECState) -> None:
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

    def fake_initial_guess(_static_arg, boundary_arg, *_args, dtype=None, **_kwargs):
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
        rs = jnp.asarray(k.state.Rsin)[:, :, None]
        zc = jnp.asarray(k.state.Zcos)[:, :, None]
        zs = jnp.asarray(k.state.Zsin)[:, :, None]
        lc = jnp.asarray(k.state.Lcos)[:, :, None]
        ls = jnp.asarray(k.state.Lsin)[:, :, None]
        return TomnspsRZL(
            frcc=r + 0.25 * zs,
            frss=0.2 * r - 0.1 * zs,
            fzsc=0.5 * zs + 0.1 * r,
            fzcs=0.3 * zc + 0.05 * r,
            flsc=ls + 0.1 * r,
            flcs=lc + 0.2 * ls + 0.03 * r,
            frsc=rs + 0.04 * r,
            frcs=rs - 0.02 * zs,
            fzcc=zc + 0.07 * r,
            fzss=zs - 0.03 * r,
            flcc=lc + 0.11 * ls,
            flss=ls - 0.05 * lc,
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
        lambda state, *_args, **_kwargs: SimpleNamespace(
            state=state,
            n_iter=3,
            fsqz2_history=np.asarray([1.0e-8]),
        ),
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


def _options(implicit, *, residual_adjoint_mode="auto", residual_tangent_mode="opaque"):
    return implicit.ImplicitFixedBoundaryOptions(
        cg_max_iter=4,
        cg_tol=1.0e-10,
        damping=0.2,
        residual_adjoint_mode=residual_adjoint_mode,
        residual_tangent_mode=residual_tangent_mode,
        jac_chunk_size=None,
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


def test_implicit_adjoint_helpers_cover_shape_errors_and_dense_callback(monkeypatch):
    import vmec_jax.implicit_adjoint_helpers as helpers
    from vmec_jax._compat import jnp
    from vmec_jax.implicit_adjoint_helpers import (
        default_jac_chunk_size,
        dense_adjoint_from_jacobian,
        first_transpose_result,
        full_active_keep_indices,
        validate_active_adjoint_shapes,
        validate_full_adjoint_shapes,
    )

    np.testing.assert_array_equal(np.asarray(full_active_keep_indices(jnp.ones(3))), np.arange(3))
    with pytest.raises(ValueError, match="one-dimensional"):
        full_active_keep_indices(jnp.ones((1, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        full_active_keep_indices(-1)
    with pytest.raises(ValueError, match="x_active_star must be one-dimensional"):
        default_jac_chunk_size(jnp.ones((1, 2)), None)
    with pytest.raises(ValueError, match="x_active_star must be one-dimensional"):
        validate_active_adjoint_shapes(jnp.ones(2), jnp.ones(2), jnp.ones((1, 2)))
    with pytest.raises(ValueError, match="residual_star_active must be non-empty"):
        validate_active_adjoint_shapes(jnp.ones(0), jnp.ones(0), jnp.ones(1))
    with pytest.raises(ValueError, match="full residual must be non-empty"):
        validate_full_adjoint_shapes(jnp.ones(0), jnp.ones(1))
    with pytest.raises(ValueError, match="empty tuple"):
        first_transpose_result(())
    assert first_transpose_result(jnp.asarray([4.0]))[0] == pytest.approx(4.0)

    calls = []

    def dense_host(J, b, damping):
        calls.append((np.asarray(J).shape, np.asarray(b).shape, float(np.asarray(damping))))
        return np.asarray([6.0, -2.0])

    def fake_pure_callback(callback, out_shape, J, b, damping):
        assert out_shape.shape == (2,)
        return jnp.asarray(callback(J, b, damping), dtype=out_shape.dtype)

    monkeypatch.setattr(helpers.jax, "pure_callback", fake_pure_callback)
    lam = dense_adjoint_from_jacobian(
        jnp.eye(2),
        jnp.asarray([1.0, 2.0]),
        damping=0.5,
        mode="dense",
        dense_transpose_lstsq_host=dense_host,
        is_traced=lambda *_args: True,
    )

    np.testing.assert_allclose(np.asarray(lam), [6.0, -2.0])
    assert calls == [((2, 2), (2,), 0.5)]


def test_vmec_residual_jitted_primal_uses_traced_solve_callback(monkeypatch):
    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state0 = _state(xp=jnp)
    static = _static()
    _install_residual_fakes(monkeypatch, implicit, jnp, state0)
    options = _options(implicit)

    @jax.jit
    def edge_value(scale):
        st = _solve_from_scale(implicit, state0, static, jnp, scale, options=options)
        return st.Rcos[-1, 0] + st.Zsin[-1, 0]

    assert float(edge_value(jnp.asarray(1.2))) == pytest.approx(3.0)


def test_vmec_residual_keep_all_tangent_direct_bicgstab(monkeypatch):
    import jax.scipy.sparse.linalg as sparse_linalg
    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state0 = _state(xp=jnp)
    static = _static()
    _install_residual_fakes(monkeypatch, implicit, jnp, state0)
    monkeypatch.setenv("VMEC_JAX_IMPLICIT_KEEP_ALL_ACTIVE", "1")
    calls = []

    def fake_bicgstab(matvec, rhs, *, tol, atol, maxiter):
        calls.append((np.asarray(matvec(jnp.ones_like(rhs))).shape, np.asarray(rhs).shape, tol, atol, maxiter))
        return 0.5 * rhs, None

    monkeypatch.setattr(sparse_linalg, "bicgstab", fake_bicgstab)
    options = _options(implicit, residual_tangent_mode="direct")

    def scalar(scale):
        st = _solve_from_scale(implicit, state0, static, jnp, scale, options=options)
        return st.Rcos[-1, 0] + 0.2 * st.Zsin[-1, 0]

    value, tangent = jax.jvp(scalar, (jnp.asarray(1.1),), (jnp.asarray(0.1),))

    assert calls == [((2,), (2,), 1.0e-10, 0.0, 4)]
    assert float(value) == pytest.approx(2.31)
    assert float(tangent) == pytest.approx(0.21)


def test_vmec_residual_keep_all_tangent_chunked_default_and_cg_fallback(monkeypatch):
    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state0 = _state(xp=jnp)
    static = _static()
    _install_residual_fakes(monkeypatch, implicit, jnp, state0)
    monkeypatch.setenv("VMEC_JAX_IMPLICIT_KEEP_ALL_ACTIVE", "1")

    chunked_options = _options(implicit, residual_tangent_mode="chunked")

    def chunked_scalar(scale):
        st = _solve_from_scale(implicit, state0, static, jnp, scale, options=chunked_options)
        return st.Rcos[-1, 0] - 0.1 * st.Zsin[-1, 0]

    chunked_value, chunked_tangent = jax.jvp(chunked_scalar, (jnp.asarray(0.9),), (jnp.asarray(0.2),))
    assert float(chunked_value) == pytest.approx(1.755)
    assert float(chunked_tangent) == pytest.approx(0.39)

    calls = []

    def fake_lineax(_matvec, b, *, tol, max_iter, **_kwargs):
        calls.append(("lineax", np.asarray(b).shape, tol, max_iter))
        return None, False, {}

    def fake_cg(matvec, rhs, *, tol, max_iter):
        calls.append(("cg", np.asarray(matvec(jnp.ones_like(rhs))).shape, np.asarray(rhs).shape, tol, max_iter))
        return jnp.zeros_like(rhs)

    monkeypatch.setattr(implicit, "_lineax_bicgstab_solve", fake_lineax)
    monkeypatch.setattr(implicit, "_cg_solve", fake_cg)
    cg_options = _options(implicit, residual_tangent_mode="auto")

    def cg_scalar(scale):
        st = _solve_from_scale(implicit, state0, static, jnp, scale, options=cg_options)
        return 0.5 * st.Rcos[-1, 0] + st.Zsin[-1, 0]

    _value, tangent = jax.jvp(cg_scalar, (jnp.asarray(1.0),), (jnp.asarray(0.1),))

    assert calls == [
        ("lineax", (2,), 1.0e-10, 4),
        ("cg", (2,), (2,), 1.0e-10, 4),
    ]
    assert float(tangent) == pytest.approx(0.15)


def test_vmec_residual_keep_all_backward_lineax_bad_num_steps(monkeypatch):
    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state0 = _state(xp=jnp)
    static = _static()
    _install_residual_fakes(monkeypatch, implicit, jnp, state0)
    monkeypatch.setenv("VMEC_JAX_IMPLICIT_KEEP_ALL_ACTIVE", "true")
    calls = []

    class BadNumSteps:
        def __array__(self, dtype=None):
            raise RuntimeError("bad host conversion")

    def fake_lineax(matvec, b, *, tol, max_iter, **_kwargs):
        calls.append((np.asarray(matvec(jnp.ones_like(b))).shape, np.asarray(b).shape, tol, max_iter))
        return 0.25 * b, True, {"num_steps": BadNumSteps()}

    monkeypatch.setattr(implicit, "_lineax_bicgstab_solve", fake_lineax)
    options = _options(implicit, residual_adjoint_mode="lineax")

    def objective(scale):
        st = _solve_from_scale(implicit, state0, static, jnp, scale, options=options)
        return 0.3 * st.Rcos[-1, 0] - 0.4 * st.Zsin[-1, 0]

    grad = float(jax.grad(objective)(jnp.asarray(1.05)))

    assert calls == [((2,), (2,), 1.0e-10, 4)]
    assert np.isfinite(grad)
