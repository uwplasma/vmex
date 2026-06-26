from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.state import StateLayout, VMECState


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


def _state(ns: int, k: int, *, xp=np) -> VMECState:
    layout = StateLayout(ns=ns, K=k, lasym=False)
    zeros = xp.zeros((ns, k), dtype=float)
    base = xp.arange(float(ns * k), dtype=float).reshape(ns, k)
    return VMECState(
        layout=layout,
        Rcos=1.0 + 0.1 * base,
        Rsin=zeros,
        Zcos=zeros,
        Zsin=0.2 + 0.05 * base,
        Lcos=0.03 * base,
        Lsin=0.04 + 0.02 * base,
    )


def _static(ns: int, m, n, *, lasym: bool = False):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            ns=ns,
            nfp=1,
            mpol=int(max(m)) + 1 if len(m) else 0,
            ntor=int(max(abs(x) for x in n)) if len(n) else 0,
            ntheta=2,
            nzeta=1,
            lasym=bool(lasym),
            lconm1=True,
        ),
        modes=SimpleNamespace(m=np.asarray(m, dtype=int), n=np.asarray(n, dtype=int), K=len(m)),
        basis=None,
        grid=SimpleNamespace(theta=np.asarray([0.0, np.pi]), zeta=np.asarray([0.0])),
        s=np.linspace(0.0, 1.0, ns),
        trig_vmec=SimpleNamespace(),
    )


def test_implicit_wrappers_raise_without_jax(monkeypatch):
    import vmec_jax.implicit as implicit

    monkeypatch.setattr(implicit, "has_jax", lambda: False)
    state = _state(2, 1)
    static = _static(2, [0], [0])

    with pytest.raises(ImportError, match="solve_lambda_state_implicit requires JAX"):
        implicit.solve_lambda_state_implicit(
            state,
            static,
            phipf=np.ones(2),
            chipf=np.ones(2),
            signgs=1,
            lamscale=np.ones(2),
        )

    with pytest.raises(ImportError, match="solve_fixed_boundary_state_implicit requires JAX"):
        implicit.solve_fixed_boundary_state_implicit(
            state,
            static,
            phipf=np.ones(2),
            chipf=np.ones(2),
            signgs=1,
            lamscale=np.ones(2),
            pressure=np.zeros(2),
        )

    with pytest.raises(ImportError, match="solve_fixed_boundary_state_implicit_vmec_residual requires JAX"):
        implicit.solve_fixed_boundary_state_implicit_vmec_residual(
            state,
            static,
            indata=_Indata(),
            signgs=1,
        )


def test_lambda_implicit_backward_masks_gauge_and_uses_hvp(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state = _state(2, 2, xp=jnp)
    static = _static(2, [0, 1], [0, 0])
    static.s = np.asarray([0.0])  # exercise the single-radial-point weight guard without a solve.

    def fake_eval_geom(st, _static):
        shape = jnp.asarray(st.Rcos).shape
        return SimpleNamespace(
            g_tt=jnp.ones(shape),
            g_tp=jnp.zeros(shape),
            g_pp=2.0 * jnp.ones(shape),
            sqrtg=jnp.ones(shape),
        )

    monkeypatch.setattr(implicit, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(
        implicit,
        "eval_fourier_dtheta",
        lambda Lcos, Lsin, *_args, **_kwargs: jnp.asarray(Lcos) + 0.5 * jnp.asarray(Lsin),
    )
    monkeypatch.setattr(
        implicit,
        "eval_fourier_dzeta_phys",
        lambda Lcos, Lsin, *_args, **_kwargs: 0.25 * jnp.asarray(Lcos) - jnp.asarray(Lsin),
    )

    def fake_bsup_from_sqrtg_lambda(*, lam_u, lam_v, phipf, chipf, lamscale, **_kwargs):
        return (
            jnp.asarray(phipf)[:, None] * lam_u + jnp.asarray(chipf)[:, None],
            jnp.asarray(lamscale)[:, None] * lam_v + 0.1 * jnp.asarray(chipf)[:, None],
        )

    monkeypatch.setattr(implicit, "bsup_from_sqrtg_lambda", fake_bsup_from_sqrtg_lambda)
    monkeypatch.setattr(implicit, "solve_lambda_gd", lambda state0, *_args, **_kwargs: SimpleNamespace(state=state0))

    base_phipf = jnp.asarray([1.0, 1.3])
    base_chipf = jnp.asarray([0.2, 0.4])
    base_lamscale = jnp.asarray([0.7, 1.1])
    options = implicit.ImplicitLambdaOptions(cg_max_iter=12, cg_tol=1.0e-12, damping=0.2)

    def active_objective(scale):
        st = implicit.solve_lambda_state_implicit(
            state,
            static,
            phipf=scale * base_phipf,
            chipf=scale * base_chipf,
            signgs=1,
            lamscale=scale * base_lamscale,
            implicit=options,
        )
        return jnp.sum(st.Lcos[:, 1]) + 0.25 * jnp.sum(st.Lsin[:, 1])

    def gauge_objective(scale):
        st = implicit.solve_lambda_state_implicit(
            state,
            static,
            phipf=scale * base_phipf,
            chipf=scale * base_chipf,
            signgs=1,
            lamscale=scale * base_lamscale,
            implicit=options,
        )
        return jnp.sum(st.Lcos[:, 0]) + jnp.sum(st.Lsin[:, 0])

    active_grad = float(jax.grad(active_objective)(1.2))
    gauge_grad = float(jax.grad(gauge_objective)(1.2))

    assert np.isfinite(active_grad)
    assert abs(active_grad) > 1.0e-8
    assert gauge_grad == pytest.approx(0.0, abs=1.0e-12)


def test_fixed_boundary_lbfgs_falls_back_to_gd_when_no_step(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.implicit as implicit
    from vmec_jax._compat import jnp

    state = _state(2, 2, xp=jnp)
    static = _static(2, [0, 1], [0, 0])
    calls = []

    def fake_lbfgs(state0, *_args, **_kwargs):
        return SimpleNamespace(state=state0, n_iter=0, grad_rms_history=[], diagnostics={})

    def fake_gd(state0, *_args, **kwargs):
        calls.append(kwargs)
        out = VMECState(
            layout=state0.layout,
            Rcos=jnp.asarray(state0.Rcos) + 10.0,
            Rsin=state0.Rsin,
            Zcos=state0.Zcos,
            Zsin=state0.Zsin,
            Lcos=state0.Lcos,
            Lsin=state0.Lsin,
        )
        return SimpleNamespace(state=out, grad_rms_history=[0.0], diagnostics={"grad_tol": 1.0})

    monkeypatch.setattr(implicit, "solve_fixed_boundary_lbfgs", fake_lbfgs)
    monkeypatch.setattr(implicit, "solve_fixed_boundary_gd", fake_gd)

    out = implicit.solve_fixed_boundary_state_implicit(
        state,
        static,
        phipf=jnp.ones(2),
        chipf=jnp.ones(2),
        signgs=1,
        lamscale=jnp.ones(2),
        pressure=jnp.zeros(2),
        solver="lbfgs",
        max_iter=3,
        step_size=2.0,
        grad_tol=1.0e-4,
    )

    assert len(calls) == 1
    assert calls[0]["max_iter"] == 50
    assert calls[0]["step_size"] == pytest.approx(0.2)
    assert calls[0]["grad_tol"] == pytest.approx(1.0e-4)
    np.testing.assert_allclose(np.asarray(out.Rcos), np.asarray(state.Rcos) + 10.0)


def test_fixed_boundary_backward_runs_hvp_and_direct_edge_cotangent(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state = _state(2, 2, xp=jnp)
    static = _static(2, [0, 1], [0, 0])
    cg_calls = []

    def fake_gd(state0, *_args, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin, **_kwargs):
        return SimpleNamespace(
            state=VMECState(
                layout=state0.layout,
                Rcos=jnp.asarray(state0.Rcos).at[-1].set(edge_Rcos),
                Rsin=jnp.asarray(state0.Rsin).at[-1].set(edge_Rsin),
                Zcos=jnp.asarray(state0.Zcos).at[-1].set(edge_Zcos),
                Zsin=jnp.asarray(state0.Zsin).at[-1].set(edge_Zsin),
                Lcos=state0.Lcos,
                Lsin=state0.Lsin,
            ),
            grad_rms_history=[0.0],
            diagnostics={"grad_tol": 1.0},
        )

    def fake_eval_geom(st, _static):
        r = jnp.asarray(st.Rcos)[:, :, None]
        z = jnp.asarray(st.Zsin)[:, :, None]
        sqrtg = 1.0 + 0.02 * r + 0.03 * z
        return SimpleNamespace(
            state=st,
            g_tt=jnp.ones_like(sqrtg),
            g_tp=jnp.zeros_like(sqrtg),
            g_pp=1.5 * jnp.ones_like(sqrtg),
            sqrtg=sqrtg,
        )

    def fake_bsup_from_geom(g, *, phipf, chipf, lamscale, **_kwargs):
        return (
            jnp.asarray(phipf)[:, None, None] + 0.1 * jnp.asarray(g.state.Rcos)[:, :, None],
            jnp.asarray(chipf)[:, None, None]
            + jnp.asarray(lamscale)[:, None, None]
            + 0.2 * jnp.asarray(g.state.Zsin)[:, :, None],
        )

    def fake_cg(matvec, b, **kwargs):
        cg_calls.append(kwargs)
        hvp = matvec(jnp.ones_like(b))
        assert hvp.shape == b.shape
        return jnp.zeros_like(b)

    monkeypatch.setattr(implicit, "solve_fixed_boundary_gd", fake_gd)
    monkeypatch.setattr(implicit, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(implicit, "bsup_from_geom", fake_bsup_from_geom)
    monkeypatch.setattr(implicit, "b2_from_bsup", lambda _g, u, v: u * u + v * v)
    monkeypatch.setattr(implicit, "_cg_solve", fake_cg)

    edge_r = jnp.asarray([1.0, -0.5])
    edge_s = jnp.asarray([0.25, -0.125])

    def objective(alpha):
        st = implicit.solve_fixed_boundary_state_implicit(
            state,
            static,
            phipf=jnp.asarray([1.0, 1.2]),
            chipf=jnp.asarray([0.1, 0.2]),
            signgs=1,
            lamscale=jnp.asarray([0.8, 0.9]),
            pressure=jnp.asarray([0.0, 0.1]),
            solver="gd",
            edge_Rcos=alpha * edge_r,
            edge_Rsin=jnp.zeros(2),
            edge_Zcos=jnp.zeros(2),
            edge_Zsin=alpha * edge_s,
            implicit_converge_tol=1.0e-3,
            implicit_zero_unconverged=False,
            implicit=implicit.ImplicitFixedBoundaryOptions(cg_max_iter=4, cg_tol=1.0e-12, damping=0.3),
        )
        return jnp.sum(jnp.asarray([2.0, -1.0]) * st.Rcos[-1]) + 0.5 * jnp.sum(st.Zsin[-1])

    grad = float(jax.grad(objective)(1.5))

    assert len(cg_calls) == 1
    assert np.isfinite(grad)
    assert abs(grad) > 0.1


def test_fixed_boundary_implicit_scalar_grad_matches_central_fd(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state = _state(2, 2, xp=jnp)
    static = _static(2, [0, 1], [0, 0])

    def fake_gd(state0, *_args, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin, **_kwargs):
        return SimpleNamespace(
            state=VMECState(
                layout=state0.layout,
                Rcos=jnp.asarray(state0.Rcos).at[-1].set(edge_Rcos),
                Rsin=jnp.asarray(state0.Rsin).at[-1].set(edge_Rsin),
                Zcos=jnp.asarray(state0.Zcos).at[-1].set(edge_Zcos),
                Zsin=jnp.asarray(state0.Zsin).at[-1].set(edge_Zsin),
                Lcos=state0.Lcos,
                Lsin=state0.Lsin,
            ),
            grad_rms_history=[0.0],
            diagnostics={"grad_tol": 1.0},
        )

    def fake_eval_geom(st, _static):
        r = jnp.asarray(st.Rcos)[:, :, None]
        z = jnp.asarray(st.Zsin)[:, :, None]
        return SimpleNamespace(
            state=st,
            g_tt=1.0 + 0.1 * r,
            g_tp=0.05 * z,
            g_pp=1.5 + 0.2 * r,
            sqrtg=1.0 + 0.02 * r + 0.03 * z,
        )

    def fake_bsup_from_geom(g, *, phipf, chipf, lamscale, **_kwargs):
        return (
            jnp.asarray(phipf)[:, None, None] + 0.1 * jnp.asarray(g.state.Rcos)[:, :, None],
            jnp.asarray(chipf)[:, None, None] + 0.2 * jnp.asarray(g.state.Zsin)[:, :, None],
        )

    monkeypatch.setattr(implicit, "solve_fixed_boundary_gd", fake_gd)
    monkeypatch.setattr(implicit, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(implicit, "bsup_from_geom", fake_bsup_from_geom)
    monkeypatch.setattr(implicit, "b2_from_bsup", lambda _g, u, v: u * u + v * v)
    monkeypatch.setattr(implicit, "_cg_solve", lambda matvec, b, **_kwargs: jnp.zeros_like(b) + 0.0 * matvec(b))

    edge_r = jnp.asarray([1.0, -0.5], dtype=jnp.float64)
    edge_z = jnp.asarray([0.25, -0.125], dtype=jnp.float64)
    weights_r = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
    weights_z = jnp.asarray([0.5, 1.5], dtype=jnp.float64)

    def objective(alpha):
        st = implicit.solve_fixed_boundary_state_implicit(
            state,
            static,
            phipf=jnp.asarray([1.0, 1.2], dtype=jnp.float64),
            chipf=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
            signgs=1,
            lamscale=jnp.asarray([0.8, 0.9], dtype=jnp.float64),
            pressure=jnp.asarray([0.0, 0.1], dtype=jnp.float64),
            solver="gd",
            edge_Rcos=alpha * edge_r,
            edge_Rsin=jnp.zeros(2, dtype=jnp.float64),
            edge_Zcos=jnp.zeros(2, dtype=jnp.float64),
            edge_Zsin=alpha * edge_z,
            implicit_converge_tol=1.0e-3,
            implicit_zero_unconverged=False,
            implicit=implicit.ImplicitFixedBoundaryOptions(cg_max_iter=4, cg_tol=1.0e-12, damping=0.3),
        )
        return jnp.sum(weights_r * st.Rcos[-1]) + jnp.sum(weights_z * st.Zsin[-1])

    alpha0 = jnp.asarray(1.25, dtype=jnp.float64)
    eps = jnp.asarray(1.0e-5, dtype=jnp.float64)
    grad_ad = jax.grad(objective)(alpha0)
    grad_fd = (objective(alpha0 + eps) - objective(alpha0 - eps)) / (2.0 * eps)

    assert np.isfinite(float(np.asarray(grad_ad)))
    assert np.isfinite(float(np.asarray(grad_fd)))
    np.testing.assert_allclose(np.asarray(grad_ad), np.asarray(grad_fd), rtol=1.0e-9, atol=1.0e-11)


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
        lambda state, *_args, **_kwargs: SimpleNamespace(state=state, n_iter=1, fsqz2_history=np.asarray([1.0e-8])),
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


def test_vmec_residual_boundary_vjp_chunked_active_path(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state0 = _state(2, 1, xp=jnp)
    static = _static(2, [0], [0])
    _install_residual_fakes(monkeypatch, implicit, jnp, state0)
    options = implicit.ImplicitFixedBoundaryOptions(
        cg_max_iter=4,
        cg_tol=1.0e-12,
        damping=0.25,
        residual_adjoint_mode="chunked",
        jac_chunk_size=1,
    )

    def objective(scale):
        st = implicit.solve_fixed_boundary_state_implicit_vmec_residual(
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
        return st.Rcos[-1, 0] + 0.2 * st.Zsin[-1, 0]

    grad = float(jax.grad(objective)(1.1))

    assert np.isfinite(grad)
    assert abs(grad) > 0.1


def test_vmec_residual_boundary_jvp_chunked_tangent_path(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state0 = _state(2, 1, xp=jnp)
    static = _static(2, [0], [0])
    _install_residual_fakes(monkeypatch, implicit, jnp, state0)
    options = implicit.ImplicitFixedBoundaryOptions(
        cg_max_iter=4,
        cg_tol=1.0e-12,
        damping=0.25,
        residual_tangent_mode="chunked",
        jac_chunk_size=1,
    )

    def scalar(edge_r):
        st = implicit.solve_fixed_boundary_state_implicit_vmec_residual(
            state0,
            static,
            indata=_Indata({"NCURR": 0, "LRFP": False, "GAMMA": 0.0, "TCON0": 1.0}),
            signgs=1,
            max_iter=1,
            step_size=0.2,
            implicit=options,
            edge_Rcos=edge_r,
            edge_Rsin=jnp.asarray([0.0]),
            edge_Zcos=jnp.asarray([0.0]),
            edge_Zsin=jnp.asarray([0.5]),
        )
        return st.Rcos[-1, 0] + 0.2 * st.Zsin[-1, 0]

    value, tangent = jax.jvp(scalar, (jnp.asarray([2.0]),), (jnp.asarray([0.3]),))

    assert float(value) == pytest.approx(2.0 + 0.2 * 0.5)
    assert float(tangent) == pytest.approx(0.3)


def test_dense_transpose_lstsq_host_matches_augmented_system_with_damping():
    import vmec_jax.implicit as implicit

    J = np.asarray(
        [
            [2.0, -1.0],
            [0.5, 3.0],
            [1.5, 0.25],
        ]
    )
    b = np.asarray([1.0, -2.0])
    damping = 0.4

    lam = implicit._dense_transpose_lstsq_host(J, b, damping)
    A = np.concatenate([J.T, np.sqrt(damping) * np.eye(J.shape[0])], axis=0)
    rhs = np.concatenate([b, np.zeros(J.shape[0])], axis=0)
    expected, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    assert lam.dtype == J.dtype
    np.testing.assert_allclose(lam, expected, rtol=1.0e-12, atol=1.0e-12)


def test_pack_named_residual_parts_applies_projector_per_block():
    import vmec_jax.implicit as implicit
    from vmec_jax._compat import jnp

    packed = implicit._pack_named_residual_parts(
        (
            ("r", jnp.asarray([[1.0, 2.0], [3.0, 4.0]])),
            ("z", jnp.asarray([10.0, 20.0, 30.0])),
            ("l", jnp.asarray([[100.0], [200.0]])),
        ),
        projector={
            "r": jnp.asarray([3, 1]),
            "l": jnp.asarray([0]),
        },
    )

    np.testing.assert_allclose(np.asarray(packed), [4.0, 2.0, 10.0, 20.0, 30.0, 100.0])


def test_linear_map_jacobian_columns_chunks_exactly_and_rejects_bad_chunk():
    pytest.importorskip("jax")

    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jnp

    enable_x64(True)
    matrix = jnp.asarray(
        [
            [1.0, 2.0, -1.0, 0.5],
            [0.0, -3.0, 4.0, 2.0],
            [5.0, 0.25, 0.0, -2.0],
        ]
    )

    def linear_map(x):
        return matrix @ x

    J = implicit._linear_map_jacobian_columns(
        linear_map,
        input_size=4,
        output_size=3,
        dtype=matrix.dtype,
        chunk_size=2,
    )

    np.testing.assert_allclose(np.asarray(J), np.asarray(matrix), rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        implicit._linear_map_jacobian_columns(
            linear_map,
            input_size=4,
            output_size=3,
            dtype=matrix.dtype,
            chunk_size=0,
        )


def test_zero_m1_zforce_flag_tracks_short_and_converged_histories():
    import vmec_jax.implicit as implicit

    short_run = implicit._zero_m1_zforce_flag_from_result(
        SimpleNamespace(n_iter=1, fsqz2_history=[]),
        float,
    )
    assert short_run == pytest.approx(1.0)
    assert implicit._zero_m1_zforce_flag_from_result(
        SimpleNamespace(n_iter=4, fsqz2_history=[2.0e-7]),
        np.float32,
    ).dtype == np.dtype(np.float32)
    assert float(
        implicit._zero_m1_zforce_flag_from_result(
            SimpleNamespace(n_iter=4, fsqz2_history=[2.0e-4]),
            float,
        )
    ) == pytest.approx(0.0)


def test_profile_logs_include_elapsed_payloads(monkeypatch, capsys):
    import vmec_jax.implicit as implicit

    monkeypatch.setenv("VMEC_JAX_PROFILE_BACKWARD", "1")
    monkeypatch.setenv("VMEC_JAX_PROFILE_RESIDUAL", "yes")
    monkeypatch.setattr(implicit.time, "perf_counter", lambda: 12.5)

    implicit._vmec_backward_profile_log("bwd", start=10.0, columns=3)
    implicit._vmec_residual_profile_log("resid", start=11.0, rows=4)

    out = capsys.readouterr().out
    assert "[vmec_jax backward]" in out
    assert "'stage': 'bwd'" in out
    assert "'elapsed_s': 2.5" in out
    assert "'columns': 3" in out
    assert "[vmec_jax residual]" in out
    assert "'stage': 'resid'" in out
    assert "'elapsed_s': 1.5" in out
    assert "'rows': 4" in out


def test_lineax_bicgstab_marks_device_get_failures_unsuccessful(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.implicit as implicit
    from vmec_jax._compat import jnp

    class FakeLineax:
        class FunctionLinearOperator:
            def __init__(self, matvec, input_structure):
                self.matvec = matvec
                self.input_structure = input_structure

        class BiCGStab:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        @staticmethod
        def linear_solve(*_args, **_kwargs):
            return SimpleNamespace(value=jnp.asarray([1.0, -2.0]), stats={"num_steps": 2})

    monkeypatch.setattr(implicit, "lx", FakeLineax)
    monkeypatch.setattr(implicit.jax, "device_get", lambda _x: (_ for _ in ()).throw(RuntimeError("device")))

    value, success, stats = implicit._lineax_bicgstab_solve(
        lambda x: x,
        jnp.asarray([1.0, -2.0]),
        x0=jnp.asarray([0.5, 0.5]),
        tol=1.0e-6,
        max_iter=3,
    )

    np.testing.assert_allclose(np.asarray(value), [1.0, -2.0])
    assert success is False
    assert stats == {"num_steps": 2}


def test_lambda_implicit_primal_uses_radial_spacing_branch(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.implicit as implicit
    from vmec_jax._compat import jnp

    state = _state(2, 1, xp=jnp)
    static = _static(2, [0], [0])

    monkeypatch.setattr(
        implicit,
        "eval_geom",
        lambda st, _static: SimpleNamespace(
            g_tt=jnp.ones_like(jnp.asarray(st.Rcos)),
            g_tp=jnp.zeros_like(jnp.asarray(st.Rcos)),
            g_pp=jnp.ones_like(jnp.asarray(st.Rcos)),
            sqrtg=jnp.ones_like(jnp.asarray(st.Rcos)),
        ),
    )
    monkeypatch.setattr(
        implicit,
        "solve_lambda_gd",
        lambda state0, *_args, **_kwargs: SimpleNamespace(
            state=VMECState(
                layout=state0.layout,
                Rcos=state0.Rcos,
                Rsin=state0.Rsin,
                Zcos=state0.Zcos,
                Zsin=state0.Zsin,
                Lcos=jnp.asarray(state0.Lcos) + 1.0,
                Lsin=state0.Lsin,
            )
        ),
    )

    out = implicit.solve_lambda_state_implicit(
        state,
        static,
        phipf=jnp.ones(2),
        chipf=jnp.ones(2),
        signgs=1,
        lamscale=jnp.ones(2),
    )

    np.testing.assert_allclose(np.asarray(out.Lcos), np.asarray(state.Lcos) + 1.0)


def test_fixed_boundary_backward_zeros_inactive_edge_cotangents(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state = _state(1, 1, xp=jnp)
    static = _static(1, [0], [0])
    lbfgs_calls = []

    def fake_gd(state0, *_args, **kwargs):
        return SimpleNamespace(state=state0, grad_rms_history=[0.0], diagnostics={"grad_tol": 1.0})

    def fake_lbfgs(state0, *_args, **kwargs):
        lbfgs_calls.append(kwargs)
        return SimpleNamespace(state=state0, grad_rms_history=[0.0], diagnostics={"grad_tol": 1.0})

    def fake_eval_geom(st, _static):
        shape = jnp.asarray(st.Rcos)[:, :, None].shape
        return SimpleNamespace(
            g_tt=jnp.ones(shape),
            g_tp=jnp.zeros(shape),
            g_pp=jnp.ones(shape),
            sqrtg=jnp.ones(shape),
        )

    monkeypatch.setattr(implicit, "solve_fixed_boundary_gd", fake_gd)
    monkeypatch.setattr(implicit, "solve_fixed_boundary_lbfgs", fake_lbfgs)
    monkeypatch.setattr(implicit, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(
        implicit,
        "bsup_from_geom",
        lambda _g, *, phipf, chipf, lamscale, **_kwargs: (
            jnp.asarray(phipf)[:, None, None],
            jnp.asarray(chipf)[:, None, None] + jnp.asarray(lamscale)[:, None, None],
        ),
    )
    monkeypatch.setattr(implicit, "b2_from_bsup", lambda _g, u, v: u * u + v * v)
    monkeypatch.setattr(implicit, "_cg_solve", lambda _matvec, b, **_kwargs: jnp.zeros_like(b))

    def objective(alpha):
        st = implicit.solve_fixed_boundary_state_implicit(
            state,
            static,
            phipf=alpha * jnp.ones(1),
            chipf=jnp.ones(1),
            signgs=1,
            lamscale=jnp.ones(1),
            pressure=jnp.zeros(1),
            solver="lbfgs",
            implicit=implicit.ImplicitFixedBoundaryOptions(cg_max_iter=2, cg_tol=1.0e-12, damping=0.1),
        )
        return jnp.sum(st.Rcos)

    grad = float(jax.grad(objective)(2.0))

    assert grad == pytest.approx(0.0)
    assert len(lbfgs_calls) == 1
