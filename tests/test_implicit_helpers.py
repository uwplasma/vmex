from __future__ import annotations

import numpy as np
import pytest


def _mode_index(modes, m: int, n: int) -> int:
    for k, (mm, nn) in enumerate(zip(np.asarray(modes.m), np.asarray(modes.n))):
        if int(mm) == int(m) and int(nn) == int(n):
            return k
    raise KeyError((m, n))


def test_update_stellsym_feasible_state_supports_reverse_mode(load_case_circular_tokamak):
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.implicit import (
        _mode00_index,
        _pack_stellsym_feasible_state,
        _stellsym_feasible_indices,
        _update_stellsym_feasible_state,
    )

    enable_x64(True)

    _cfg, _indata, static, _bdy, st0 = load_case_circular_tokamak
    idx00 = _mode00_index(static.modes)
    rz_idx, lam_idx, ns, K = _stellsym_feasible_indices(static, idx00=idx00, mask_lambda_axis=True)
    x0 = _pack_stellsym_feasible_state(st0, rz_idx=rz_idx, lam_idx=lam_idx)

    def objective(x):
        st = _update_stellsym_feasible_state(st0, x, rz_idx=rz_idx, lam_idx=lam_idx, ns=ns, K=K)
        return jnp.sum(jnp.asarray(st.Rcos) ** 2) + jnp.sum(jnp.asarray(st.Zsin) ** 2) + jnp.sum(jnp.asarray(st.Lsin) ** 2)

    grad = np.asarray(jax.grad(objective)(x0))
    assert grad.shape == tuple(np.asarray(x0).shape)
    assert np.all(np.isfinite(grad))


def test_initial_guess_vmec_project_edge_rc01_gradient_matches_internal_scale():
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.boundary import BoundaryCoeffs
    from vmec_jax.config import VMECConfig
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.namelist import InData
    from vmec_jax.static import build_static

    enable_x64(True)

    cfg = VMECConfig(mpol=3, ntor=2, ns=5, nfp=1, lasym=False, lconm1=True, lthreed=True, ntheta=16, nzeta=8)
    static = build_static(cfg)
    K = int(static.modes.K)

    k00 = _mode_index(static.modes, 0, 0)
    k01 = _mode_index(static.modes, 0, 1)

    base_Rcos = np.zeros((K,), dtype=float)
    base_Rsin = np.zeros((K,), dtype=float)
    base_Zcos = np.zeros((K,), dtype=float)
    base_Zsin = np.zeros((K,), dtype=float)
    base_Rcos[k00] = 3.0
    indata = InData(scalars={"RAXIS_CC": [3.0], "ZAXIS_CS": [0.0]}, indexed={})

    def edge_coeff(alpha):
        boundary = BoundaryCoeffs(
            R_cos=jnp.asarray(base_Rcos).at[k01].set(alpha),
            R_sin=jnp.asarray(base_Rsin),
            Z_cos=jnp.asarray(base_Zcos),
            Z_sin=jnp.asarray(base_Zsin),
        )
        st = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
        return st.Rcos[-1, k01]

    alpha0 = 1.2
    eps = 1e-6
    grad_ad = float(jax.grad(edge_coeff)(alpha0))
    grad_fd = float((edge_coeff(alpha0 + eps) - edge_coeff(alpha0 - eps)) / (2.0 * eps))
    expected = float(np.asarray(static.mode_scale_internal)[k01])

    assert np.isfinite(grad_ad)
    assert np.isfinite(grad_fd)
    assert grad_ad == pytest.approx(expected, rel=0.0, abs=1e-12)
    assert grad_ad == pytest.approx(grad_fd, rel=0.0, abs=1e-7)


def test_fixed_boundary_residual_implicit_primal_matches_reference_mode(load_case_circular_tokamak):
    pytest.importorskip("jax")

    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.implicit import solve_fixed_boundary_state_implicit_vmec_residual
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state
    _cfg, indata, static, boundary, state_init = load_case_circular_tokamak
    signgs0 = signgs_from_sqrtg(np.asarray(eval_geom(state_init, static).sqrtg), axis_index=1)

    direct = solve_fixed_boundary_residual_iter(
        state_init,
        static,
        indata=indata,
        signgs=int(signgs0),
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=True,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
    ).state

    wrapped = solve_fixed_boundary_state_implicit_vmec_residual(
        state_init,
        static,
        indata=indata,
        signgs=int(signgs0),
        state0_host=state_init,
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        ftol=float(indata.get_float("FTOL", 1e-14)),
        edge_Rcos=np.asarray(boundary.R_cos),
        edge_Rsin=np.asarray(boundary.R_sin),
        edge_Zcos=np.asarray(boundary.Z_cos),
        edge_Zsin=np.asarray(boundary.Z_sin),
    )

    assert np.asarray(pack_state(wrapped)) == pytest.approx(np.asarray(pack_state(direct)), rel=0.0, abs=1e-12)
