from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import has_jax
from vmec_jax.state import StateLayout, VMECState


pytestmark = pytest.mark.skipif(not has_jax(), reason="implicit sensitivity tests require JAX")


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
    rcos = xp.ones((ns, k), dtype=float)
    return VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=zeros,
        Zcos=zeros,
        Zsin=zeros,
        Lcos=zeros,
        Lsin=zeros,
    )


def _static(ns: int, m, n):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            ns=ns,
            nfp=1,
            mpol=int(max(m)) + 1 if len(m) else 0,
            ntor=int(max(abs(x) for x in n)) if len(n) else 0,
            ntheta=2,
            nzeta=1,
            lasym=False,
            lconm1=True,
        ),
        modes=SimpleNamespace(m=np.asarray(m, dtype=int), n=np.asarray(n, dtype=int), K=len(m)),
        basis=None,
        grid=SimpleNamespace(theta=np.asarray([0.0, np.pi]), zeta=np.asarray([0.0])),
        s=np.linspace(0.0, 1.0, ns),
        trig_vmec=SimpleNamespace(),
    )


def _central_difference(fun, x: float, *, eps: float = 1.0e-5) -> float:
    return (float(fun(x + eps)) - float(fun(x - eps))) / (2.0 * eps)


def test_lambda_custom_vjp_matches_closed_form_active_sensitivity(monkeypatch):
    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state0 = _state(1, 2, xp=jnp)
    static = _static(1, [0, 1], [0, 0])

    def fake_eval_geom(st, _static):
        shape = jnp.asarray(st.Rcos).shape
        return SimpleNamespace(
            g_tt=jnp.ones(shape),
            g_tp=jnp.zeros(shape),
            g_pp=jnp.ones(shape),
            sqrtg=jnp.ones(shape),
        )

    def fake_solve_lambda_gd(state, _static, *, phipf, chipf, **_kwargs):
        lcos = jnp.zeros_like(jnp.asarray(state.Lcos)).at[0, 1].set(-jnp.asarray(chipf)[0] / jnp.asarray(phipf)[0])
        return SimpleNamespace(state=VMECState(state.layout, state.Rcos, state.Rsin, state.Zcos, state.Zsin, lcos, state.Lsin))

    monkeypatch.setattr(implicit, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(implicit, "eval_fourier_dtheta", lambda Lcos, *_args, **_kwargs: jnp.asarray(Lcos))
    monkeypatch.setattr(implicit, "eval_fourier_dzeta_phys", lambda Lcos, *_args, **_kwargs: jnp.zeros_like(Lcos))
    monkeypatch.setattr(
        implicit,
        "bsup_from_sqrtg_lambda",
        lambda *, lam_u, phipf, chipf, **_kwargs: (
            jnp.asarray(phipf)[:, None] * jnp.asarray(lam_u) + jnp.asarray(chipf)[:, None],
            jnp.zeros_like(lam_u),
        ),
    )
    monkeypatch.setattr(implicit, "solve_lambda_gd", fake_solve_lambda_gd)

    phipf0 = 2.5
    weight = 1.7
    options = implicit.ImplicitLambdaOptions(cg_max_iter=8, cg_tol=1.0e-13, damping=0.0)

    def objective(chipf0):
        st = implicit.solve_lambda_state_implicit(
            state0,
            static,
            phipf=jnp.asarray([phipf0]),
            chipf=jnp.asarray([chipf0]),
            signgs=1,
            lamscale=jnp.ones(1),
            max_iter=1,
            implicit=options,
        )
        return weight * st.Lcos[0, 1]

    grad_ad = float(jax.grad(objective)(0.4))
    grad_fd = _central_difference(objective, 0.4)

    assert grad_ad == pytest.approx(-weight / phipf0, rel=0.0, abs=1.0e-11)
    assert grad_ad == pytest.approx(grad_fd, rel=0.0, abs=1.0e-8)


def test_fixed_boundary_energy_custom_vjp_profile_sensitivity_matches_fd(monkeypatch):
    import vmec_jax.implicit as implicit
    from vmec_jax._compat import enable_x64, jax, jnp

    enable_x64(True)
    state0 = _state(2, 1, xp=jnp)
    static = _static(2, [0], [0])

    def fake_solve_fixed_boundary_gd(state, _static, *, phipf, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin, **_kwargs):
        rcos = (
            jnp.asarray(state.Rcos)
            .at[0, 0]
            .set(jnp.asarray(phipf)[0])
            .at[-1, :]
            .set(jnp.asarray(edge_Rcos))
        )
        solved = VMECState(
            state.layout,
            rcos,
            jnp.asarray(state.Rsin).at[-1, :].set(jnp.asarray(edge_Rsin)),
            jnp.asarray(state.Zcos).at[-1, :].set(jnp.asarray(edge_Zcos)),
            jnp.asarray(state.Zsin).at[-1, :].set(jnp.asarray(edge_Zsin)),
            state.Lcos,
            state.Lsin,
        )
        return SimpleNamespace(state=solved, grad_rms_history=[0.0], diagnostics={"grad_tol": 1.0})

    def fake_eval_geom(st, _static):
        return SimpleNamespace(state=st, sqrtg=jnp.ones(jnp.asarray(st.Rcos).shape + (1,)))

    monkeypatch.setattr(implicit, "solve_fixed_boundary_gd", fake_solve_fixed_boundary_gd)
    monkeypatch.setattr(implicit, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(
        implicit,
        "bsup_from_geom",
        lambda _g, *, phipf, **_kwargs: (jnp.asarray(phipf)[:, None, None], jnp.zeros((2, 1, 1))),
    )
    monkeypatch.setattr(implicit, "b2_from_bsup", lambda g, u, _v: (jnp.asarray(g.state.Rcos)[:, :, None] - u) ** 2)

    phipf_base = jnp.asarray([1.25, -0.4])
    output_weight = 2.0
    options = implicit.ImplicitFixedBoundaryOptions(cg_max_iter=8, cg_tol=1.0e-13, damping=0.0)

    def objective(scale):
        st = implicit.solve_fixed_boundary_state_implicit(
            state0,
            static,
            phipf=scale * phipf_base,
            chipf=jnp.zeros(2),
            signgs=1,
            lamscale=jnp.ones(2),
            pressure=jnp.zeros(2),
            solver="gd",
            max_iter=1,
            implicit_converge_tol=0.5,
            implicit=options,
        )
        return output_weight * st.Rcos[0, 0]

    grad_ad = float(jax.grad(objective)(0.8))
    grad_fd = _central_difference(objective, 0.8)

    assert grad_ad == pytest.approx(output_weight * float(phipf_base[0]), rel=0.0, abs=1.0e-11)
    assert grad_ad == pytest.approx(grad_fd, rel=0.0, abs=1.0e-8)


def test_vmec_residual_custom_vjp_active_boundary_sensitivity_matches_fd(monkeypatch):
    import vmec_jax.boundary as boundary_module
    import vmec_jax.implicit as implicit
    import vmec_jax.init_guess as init_guess_module
    import vmec_jax.preconditioner_1d_jax as preconditioner_module
    import vmec_jax.kernels.forces as forces_module
    import vmec_jax.kernels.residue as residue_module
    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.boundary import BoundaryCoeffs
    from vmec_jax.kernels.tomnsp import TomnspsRZL

    enable_x64(True)
    state0 = _state(2, 1, xp=jnp)
    static = _static(2, [0], [0])
    boundary = BoundaryCoeffs(
        R_cos=np.asarray([1.6]),
        R_sin=np.asarray([0.0]),
        Z_cos=np.asarray([0.0]),
        Z_sin=np.asarray([0.0]),
    )

    def fake_initial_guess(_static, boundary_arg, *_args, dtype=None, **_kwargs):
        dtype = dtype or jnp.asarray(state0.Rcos).dtype
        edge_r = jnp.asarray(boundary_arg.R_cos, dtype=dtype)
        edge_z = jnp.asarray(boundary_arg.Z_sin, dtype=dtype)
        zeros = jnp.zeros_like(jnp.asarray(state0.Rcos, dtype=dtype))
        return VMECState(
            state0.layout,
            zeros.at[0, :].set(edge_r).at[-1, :].set(edge_r),
            zeros,
            zeros,
            zeros.at[0, :].set(edge_z).at[-1, :].set(edge_z),
            zeros,
            zeros,
        )

    def fake_residual_from_kernels(k, **_kwargs):
        r = jnp.asarray(k.state.Rcos)[:, :, None]
        zeros = jnp.zeros_like(r)
        return TomnspsRZL(
            frcc=r - r[-1:, :, :],
            frss=None,
            fzsc=zeros,
            fzcs=None,
            flsc=zeros,
            flcs=None,
        )

    monkeypatch.setattr(
        implicit,
        "flux_profiles_from_indata",
        lambda *_args, **_kwargs: SimpleNamespace(
            phipf=jnp.asarray([0.0, 1.0]),
            phips=jnp.asarray([0.0, 1.0]),
            chipf=jnp.asarray([0.0, 0.0]),
        ),
    )
    monkeypatch.setattr(implicit, "_mass_half_mesh_from_indata", lambda **_kwargs: jnp.asarray([0.0, 1.0]))
    monkeypatch.setattr(implicit, "_pressure_half_mesh_from_indata", lambda **_kwargs: jnp.asarray([0.0, 0.0]))
    monkeypatch.setattr(implicit, "_icurv_full_mesh_from_indata", lambda **_kwargs: jnp.asarray([0.0, 0.0]))
    monkeypatch.setattr(implicit, "_vmec_force_flux_profiles", lambda **kwargs: (kwargs["phipf"], kwargs["chipf"], None))
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

    edge_base = jnp.asarray([1.6])
    output_weight = 1.3
    options = implicit.ImplicitFixedBoundaryOptions(
        cg_max_iter=8,
        cg_tol=1.0e-13,
        damping=0.0,
        residual_adjoint_mode="auto",
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
            edge_Rcos=scale * edge_base,
            edge_Rsin=jnp.asarray([0.0]),
            edge_Zcos=jnp.asarray([0.0]),
            edge_Zsin=jnp.asarray([0.0]),
        )
        return output_weight * st.Rcos[0, 0]

    grad_ad = float(jax.grad(objective)(0.75))
    grad_fd = _central_difference(objective, 0.75)

    assert grad_ad == pytest.approx(output_weight * float(edge_base[0]), rel=0.0, abs=1.0e-11)
    assert grad_ad == pytest.approx(grad_fd, rel=0.0, abs=1.0e-8)
