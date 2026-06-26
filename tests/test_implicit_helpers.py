from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.state import StateLayout, VMECState


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


def test_stellsym_active_keep_scatter_supports_reverse_mode(load_case_circular_tokamak):
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.implicit import (
        _mode00_index,
        _pack_stellsym_feasible_state,
        _stellsym_feasible_indices,
        _stellsym_structural_active_keep_indices,
    )

    enable_x64(True)

    _cfg, _indata, static, _bdy, st0 = load_case_circular_tokamak
    idx00 = _mode00_index(static.modes)
    rz_idx, lam_idx, _ns, K = _stellsym_feasible_indices(static, idx00=idx00, mask_lambda_axis=True)
    x_full0 = _pack_stellsym_feasible_state(st0, rz_idx=rz_idx, lam_idx=lam_idx)
    keep = _stellsym_structural_active_keep_indices(
        rz_idx=np.asarray(rz_idx),
        lam_idx=np.asarray(lam_idx),
        K=int(K),
        idx00=idx00,
    )
    x0 = jnp.take(x_full0, keep)

    def objective(x):
        rebuilt = x_full0.at[keep].set(x, indices_are_sorted=True, unique_indices=True)
        return jnp.sum(rebuilt * rebuilt)

    grad = np.asarray(jax.grad(objective)(x0))
    assert grad.shape == tuple(np.asarray(x0).shape)
    assert np.all(np.isfinite(grad))


def test_stellsym_reduced_lambda_mn_coords_roundtrip_and_support_reverse_mode(load_case_circular_tokamak):
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.implicit import (
        _mode00_index,
        _pack_stellsym_reduced_state,
        _stellsym_feasible_indices_np,
        _stellsym_lambda_mn_indices,
        _stellsym_reduced_z_indices,
        _update_stellsym_reduced_state,
    )

    enable_x64(True)

    _cfg, _indata, static, _bdy, st0 = load_case_circular_tokamak
    idx00 = _mode00_index(static.modes)
    rz_idx_np, _lam_idx_np, ns, K = _stellsym_feasible_indices_np(static, idx00=idx00, mask_lambda_axis=True)
    rz_idx = jnp.asarray(rz_idx_np, dtype=jnp.int32)
    z_idx = _stellsym_reduced_z_indices(rz_idx=rz_idx_np, K=int(K), idx00=idx00)
    lam_sc_idx, lam_cs_idx, lam_maps = _stellsym_lambda_mn_indices(
        static,
        idx00=idx00,
        mask_lambda_axis=True,
    )

    x0 = _pack_stellsym_reduced_state(
        st0,
        rz_idx=rz_idx,
        z_idx=z_idx,
        lam_sc_idx=lam_sc_idx,
        lam_cs_idx=lam_cs_idx,
        lam_maps=lam_maps,
    )
    st1 = _update_stellsym_reduced_state(
        st0,
        x0,
        rz_idx=rz_idx,
        z_idx=z_idx,
        lam_sc_idx=lam_sc_idx,
        lam_cs_idx=lam_cs_idx,
        lam_maps=lam_maps,
        ns=ns,
        K=K,
    )

    assert np.asarray(st1.Rcos) == pytest.approx(np.asarray(st0.Rcos), rel=0.0, abs=1e-12)
    assert np.asarray(st1.Zsin) == pytest.approx(np.asarray(st0.Zsin), rel=0.0, abs=1e-12)
    assert np.asarray(st1.Lsin) == pytest.approx(np.asarray(st0.Lsin), rel=0.0, abs=1e-12)

    def objective(x):
        st = _update_stellsym_reduced_state(
            st0,
            x,
            rz_idx=rz_idx,
            z_idx=z_idx,
            lam_sc_idx=lam_sc_idx,
            lam_cs_idx=lam_cs_idx,
            lam_maps=lam_maps,
            ns=ns,
            K=K,
        )
        return (
            jnp.sum(jnp.asarray(st.Rcos) ** 2)
            + jnp.sum(jnp.asarray(st.Zsin) ** 2)
            + jnp.sum(jnp.asarray(st.Lsin) ** 2)
        )

    grad = np.asarray(jax.grad(objective)(x0))
    assert grad.shape == tuple(np.asarray(x0).shape)
    assert np.all(np.isfinite(grad))


def test_fixed_boundary_residual_implicit_primal_matches_default_control_path(load_case_circular_tokamak):
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
        reference_mode=False,
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


def test_implicit_profile_environment_flags_and_logs(monkeypatch, capsys):
    from vmec_jax.implicit import (
        _vmec_backward_profile_enabled,
        _vmec_backward_profile_log,
        _vmec_disable_reduced_active_enabled,
        _vmec_keep_all_active_enabled,
        _vmec_residual_profile_enabled,
        _vmec_residual_profile_log,
    )

    for name in (
        "VMEC_JAX_PROFILE_BACKWARD",
        "VMEC_JAX_PROFILE_RESIDUAL",
        "VMEC_JAX_IMPLICIT_KEEP_ALL_ACTIVE",
        "VMEC_JAX_IMPLICIT_DISABLE_REDUCED_ACTIVE",
    ):
        monkeypatch.delenv(name, raising=False)

    assert _vmec_backward_profile_enabled() is False
    assert _vmec_residual_profile_enabled() is False
    assert _vmec_keep_all_active_enabled() is False
    assert _vmec_disable_reduced_active_enabled() is False

    _vmec_backward_profile_log("silent")
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("VMEC_JAX_PROFILE_BACKWARD", "1")
    monkeypatch.setenv("VMEC_JAX_PROFILE_RESIDUAL", "yes")
    monkeypatch.setenv("VMEC_JAX_IMPLICIT_KEEP_ALL_ACTIVE", "true")
    monkeypatch.setenv("VMEC_JAX_IMPLICIT_DISABLE_REDUCED_ACTIVE", "TRUE")

    assert _vmec_backward_profile_enabled() is True
    assert _vmec_residual_profile_enabled() is True
    assert _vmec_keep_all_active_enabled() is True
    assert _vmec_disable_reduced_active_enabled() is True

    _vmec_backward_profile_log("unit_backward", count=2)
    _vmec_residual_profile_log("unit_residual", rows=3)
    out = capsys.readouterr().out
    assert "[vmec_jax backward]" in out
    assert "unit_backward" in out
    assert "'count': 2" in out
    assert "[vmec_jax residual]" in out
    assert "unit_residual" in out
    assert "'rows': 3" in out

    monkeypatch.setenv("VMEC_JAX_PROFILE_BACKWARD", "False")
    monkeypatch.setenv("VMEC_JAX_PROFILE_RESIDUAL", "no")
    monkeypatch.setenv("VMEC_JAX_IMPLICIT_KEEP_ALL_ACTIVE", "0")
    monkeypatch.setenv("VMEC_JAX_IMPLICIT_DISABLE_REDUCED_ACTIVE", "")
    assert _vmec_backward_profile_enabled() is False
    assert _vmec_residual_profile_enabled() is False
    assert _vmec_keep_all_active_enabled() is False
    assert _vmec_disable_reduced_active_enabled() is False


def test_implicit_linear_algebra_and_state_packing_helpers():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.implicit import (
        _cg_solve,
        _dense_transpose_lstsq_host,
        _flatten_L,
        _linear_map_jacobian_columns,
        _pack_named_residual_parts,
        _stop_gradient_tree,
        _unflatten_L,
        _zero_m1_zforce_flag_from_result,
        _zero_state_like,
    )
    from vmec_jax.solve import _zero_edge_rz_force_block, _zero_edge_rz_force_blocks
    from vmec_jax.kernels.tomnsp import TomnspsRZL

    layout = StateLayout(ns=2, K=3, lasym=True)
    state = VMECState(
        layout=layout,
        Rcos=jnp.arange(6.0).reshape(2, 3),
        Rsin=jnp.ones((2, 3)),
        Zcos=2.0 * jnp.ones((2, 3)),
        Zsin=3.0 * jnp.ones((2, 3)),
        Lcos=4.0 * jnp.ones((2, 3)),
        Lsin=5.0 * jnp.ones((2, 3)),
    )

    zero = _zero_state_like(state)
    assert zero.layout == layout
    for block in (zero.Rcos, zero.Rsin, zero.Zcos, zero.Zsin, zero.Lcos, zero.Lsin):
        np.testing.assert_allclose(np.asarray(block), np.zeros((2, 3)))

    stopped = _stop_gradient_tree(state)
    np.testing.assert_allclose(np.asarray(stopped.Rcos), np.asarray(state.Rcos))

    flat = _flatten_L(state.Lcos, state.Lsin)
    Lcos, Lsin = _unflatten_L(flat, shape=(2, 3))
    np.testing.assert_allclose(np.asarray(Lcos), np.asarray(state.Lcos))
    np.testing.assert_allclose(np.asarray(Lsin), np.asarray(state.Lsin))

    mat = np.asarray([[2.0, 0.0], [0.0, 4.0]])
    rhs = np.asarray([6.0, 8.0])
    np.testing.assert_allclose(_dense_transpose_lstsq_host(mat, rhs, 0.0), [3.0, 2.0])

    damping = 0.5
    eye = np.eye(2)
    expected_damped, *_ = np.linalg.lstsq(
        np.concatenate([mat.T, np.sqrt(damping) * eye], axis=0),
        np.concatenate([rhs, np.zeros((2,))], axis=0),
        rcond=None,
    )
    np.testing.assert_allclose(_dense_transpose_lstsq_host(mat, rhs, damping), expected_damped)

    sol = _cg_solve(
        lambda x: jnp.asarray([4.0 * x[0], 9.0 * x[1]]),
        jnp.asarray([8.0, 27.0]),
        tol=1.0e-10,
        max_iter=5,
    )
    np.testing.assert_allclose(np.asarray(sol), [2.0, 3.0], rtol=1e-6, atol=1e-6)

    jac = _linear_map_jacobian_columns(
        lambda x: jnp.asarray([x[0] + 2.0 * x[1], x[2] - x[0]]),
        input_size=3,
        output_size=2,
        dtype=jnp.float32,
        chunk_size=2,
    )
    np.testing.assert_allclose(np.asarray(jac), [[1.0, 2.0, 0.0], [-1.0, 0.0, 1.0]])

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        _linear_map_jacobian_columns(
            lambda x: x,
            input_size=1,
            output_size=1,
            dtype=jnp.float32,
            chunk_size=0,
        )

    packed = _pack_named_residual_parts(
        [
            ("a", jnp.asarray([[1.0, 2.0], [3.0, 4.0]])),
            ("b", jnp.asarray([5.0, 6.0, 7.0])),
        ],
        projector={"a": jnp.asarray([0, 3], dtype=jnp.int32)},
    )
    np.testing.assert_allclose(np.asarray(packed), [1.0, 4.0, 5.0, 6.0, 7.0])

    np_block = np.arange(6.0).reshape(3, 2)
    np_masked = _zero_edge_rz_force_block(np_block)
    assert isinstance(np_masked, np.ndarray)
    assert not np.shares_memory(np_masked, np_block)
    np.testing.assert_allclose(np_masked, [[0.0, 1.0], [2.0, 3.0], [0.0, 0.0]])
    np.testing.assert_allclose(np_block[-1], [4.0, 5.0])
    assert _zero_edge_rz_force_block(None) is None
    same_short = _zero_edge_rz_force_block(np.ones((1, 2)))
    np.testing.assert_allclose(same_short, [[1.0, 1.0]])

    jax_masked = _zero_edge_rz_force_block(jnp.asarray(np_block), preserve_numpy=False)
    np.testing.assert_allclose(np.asarray(jax_masked), [[0.0, 1.0], [2.0, 3.0], [0.0, 0.0]])

    frzl = TomnspsRZL(
        frcc=np_block,
        frss=np_block + 10.0,
        fzsc=np_block + 20.0,
        fzcs=np_block + 30.0,
        flsc=np_block + 40.0,
        flcs=np_block + 50.0,
    )
    masked_frzl = _zero_edge_rz_force_blocks(frzl)
    np.testing.assert_allclose(masked_frzl.frcc[-1], [0.0, 0.0])
    np.testing.assert_allclose(masked_frzl.frss[-1], [0.0, 0.0])
    np.testing.assert_allclose(masked_frzl.fzsc[-1], [0.0, 0.0])
    np.testing.assert_allclose(masked_frzl.fzcs[-1], [0.0, 0.0])
    np.testing.assert_allclose(masked_frzl.flsc[-1], [44.0, 45.0])
    np.testing.assert_allclose(masked_frzl.flcs[-1], [54.0, 55.0])

    class Result:
        def __init__(self, n_iter, fsqz2_history):
            self.n_iter = n_iter
            self.fsqz2_history = fsqz2_history

    assert float(_zero_m1_zforce_flag_from_result(Result(0, []), dtype=np.float64)) == 1.0
    assert float(_zero_m1_zforce_flag_from_result(Result(4, [1.0e-8]), dtype=np.float64)) == 1.0
    assert float(_zero_m1_zforce_flag_from_result(Result(4, [1.0e-3]), dtype=np.float64)) == 0.0
