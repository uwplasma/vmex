from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import has_jax
from vmec_jax.state import StateLayout, VMECState


pytestmark = pytest.mark.skipif(not has_jax(), reason="implicit helpers require JAX")


def _state(ns: int = 2, K: int = 3) -> VMECState:
    layout = StateLayout(ns=ns, K=K, lasym=False)
    base = np.arange(ns * K, dtype=float).reshape(ns, K)
    return VMECState(
        layout=layout,
        Rcos=base + 1.0,
        Rsin=base + 2.0,
        Zcos=base + 3.0,
        Zsin=base + 4.0,
        Lcos=base + 5.0,
        Lsin=base + 6.0,
    )


def _minimal_static(ns: int = 2, K: int = 3):
    return SimpleNamespace(
        cfg=SimpleNamespace(ns=ns, nfp=1, mpol=2, ntor=1, lasym=False),
        modes=SimpleNamespace(m=np.asarray([0, 1, 1])[:K], n=np.asarray([0, 0, 1])[:K]),
        s=np.linspace(0.0, 1.0, ns),
        grid=SimpleNamespace(theta=np.linspace(0.0, 1.0, 2), zeta=np.linspace(0.0, 1.0, 2)),
    )


def test_env_switch_helpers_accept_common_false_values(monkeypatch):
    import vmec_jax.implicit as implicit

    for value in ("", "0", "false", "no"):
        monkeypatch.setenv("VMEC_JAX_PROFILE_BACKWARD", value)
        monkeypatch.setenv("VMEC_JAX_PROFILE_RESIDUAL", value)
        monkeypatch.setenv("VMEC_JAX_IMPLICIT_KEEP_ALL_ACTIVE", value)
        monkeypatch.setenv("VMEC_JAX_IMPLICIT_DISABLE_REDUCED_ACTIVE", value)
        assert implicit._vmec_backward_profile_enabled() is False
        assert implicit._vmec_residual_profile_enabled() is False
        assert implicit._vmec_keep_all_active_enabled() is False
        assert implicit._vmec_disable_reduced_active_enabled() is False

    monkeypatch.setenv("VMEC_JAX_PROFILE_BACKWARD", "yes")
    monkeypatch.setenv("VMEC_JAX_PROFILE_RESIDUAL", "1")
    monkeypatch.setenv("VMEC_JAX_IMPLICIT_KEEP_ALL_ACTIVE", "true")
    monkeypatch.setenv("VMEC_JAX_IMPLICIT_DISABLE_REDUCED_ACTIVE", "on")
    assert implicit._vmec_backward_profile_enabled() is True
    assert implicit._vmec_residual_profile_enabled() is True
    assert implicit._vmec_keep_all_active_enabled() is True
    assert implicit._vmec_disable_reduced_active_enabled() is True


def test_lambda_pack_roundtrip_zero_state_and_projected_residual_parts():
    from vmec_jax._compat import jnp
    from vmec_jax.implicit import _flatten_L, _pack_named_residual_parts, _unflatten_L, _zero_state_like

    Lcos = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    Lsin = jnp.asarray([[5.0, 6.0], [7.0, 8.0]])

    flat = _flatten_L(Lcos, Lsin)
    out_cos, out_sin = _unflatten_L(flat, shape=(2, 2))

    np.testing.assert_allclose(np.asarray(out_cos), np.asarray(Lcos))
    np.testing.assert_allclose(np.asarray(out_sin), np.asarray(Lsin))

    zero = _zero_state_like(_state(ns=2, K=2))
    assert zero.layout == StateLayout(ns=2, K=2, lasym=False)
    for leaf in (zero.Rcos, zero.Rsin, zero.Zcos, zero.Zsin, zero.Lcos, zero.Lsin):
        np.testing.assert_allclose(np.asarray(leaf), 0.0)

    parts = [("a", jnp.asarray([[1.0, 2.0], [3.0, 4.0]])), ("b", jnp.asarray([5.0, 6.0]))]
    packed = _pack_named_residual_parts(parts, projector={"a": jnp.asarray([0, 3]), "b": jnp.asarray([1])})
    np.testing.assert_allclose(np.asarray(packed), [1.0, 4.0, 6.0])


def test_cg_solve_matches_spd_system_and_honors_initial_guess_for_zero_iterations():
    from vmec_jax._compat import jnp
    from vmec_jax.implicit import _cg_solve

    A = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    b = jnp.asarray([1.0, 2.0])

    x = _cg_solve(lambda v: A @ v, b, tol=1.0e-14, max_iter=8)
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(np.asarray(A), np.asarray(b)), rtol=1e-10, atol=1e-10)

    x0 = jnp.asarray([9.0, -2.0])
    unchanged = _cg_solve(lambda v: A @ v, b, x0=x0, tol=1.0e-14, max_iter=0)
    np.testing.assert_allclose(np.asarray(unchanged), np.asarray(x0))


def test_zero_m1_flag_uses_iteration_count_and_final_fsqz():
    from vmec_jax.implicit import _zero_m1_zforce_flag_from_result

    early = _zero_m1_zforce_flag_from_result(SimpleNamespace(n_iter=1, fsqz2_history=[10.0]), np.float32)
    converged = _zero_m1_zforce_flag_from_result(SimpleNamespace(n_iter=4, fsqz2_history=[1.0e-8]), np.float64)
    unconverged = _zero_m1_zforce_flag_from_result(SimpleNamespace(n_iter=4, fsqz2_history=[1.0e-3]), np.float64)
    empty_history = _zero_m1_zforce_flag_from_result(SimpleNamespace(n_iter=4, fsqz2_history=[]), np.float64)

    assert early.dtype == np.dtype(np.float32)
    assert float(early) == 1.0
    assert float(converged) == 1.0
    assert float(unconverged) == 0.0
    assert float(empty_history) == 0.0


def test_stellsym_feasible_pack_update_and_keep_indices_mask_constraints():
    from vmec_jax.implicit import (
        _pack_stellsym_feasible_state,
        _stellsym_feasible_indices_np,
        _stellsym_structural_active_keep_indices,
        _update_stellsym_feasible_state,
    )

    state = _state(ns=3, K=3)
    static = _minimal_static(ns=3, K=3)

    rz_idx, lam_idx, ns, K = _stellsym_feasible_indices_np(static, idx00=0, mask_lambda_axis=True)
    assert ns == 3
    assert K == 3
    assert 0 in rz_idx
    assert (ns - 1) * K not in rz_idx
    assert 0 not in lam_idx
    assert 3 not in lam_idx

    packed = _pack_stellsym_feasible_state(state, rz_idx=rz_idx, lam_idx=lam_idx)
    updated = _update_stellsym_feasible_state(
        state,
        packed + 100.0,
        rz_idx=rz_idx,
        lam_idx=lam_idx,
        ns=ns,
        K=K,
    )
    np.testing.assert_allclose(np.ravel(np.asarray(updated.Rcos))[rz_idx], np.ravel(np.asarray(state.Rcos))[rz_idx] + 100.0)
    np.testing.assert_allclose(np.ravel(np.asarray(updated.Zsin))[rz_idx], np.ravel(np.asarray(state.Zsin))[rz_idx] + 100.0)
    np.testing.assert_allclose(np.ravel(np.asarray(updated.Lsin))[lam_idx], np.ravel(np.asarray(state.Lsin))[lam_idx] + 100.0)
    np.testing.assert_allclose(np.asarray(updated.Rsin), np.asarray(state.Rsin))

    keep = _stellsym_structural_active_keep_indices(rz_idx=rz_idx, lam_idx=lam_idx, K=K, idx00=0)
    assert keep.shape[0] < (2 * len(rz_idx) + len(lam_idx))
    assert not np.any(np.asarray(keep) == len(rz_idx))


def test_fixed_boundary_implicit_rejects_invalid_solver_and_partial_edge_inputs():
    from vmec_jax.implicit import solve_fixed_boundary_state_implicit

    state = _state()
    static = _minimal_static()

    with pytest.raises(ValueError, match="solver must be 'gd' or 'lbfgs'"):
        solve_fixed_boundary_state_implicit(
            state,
            static,
            phipf=np.ones(2),
            chipf=np.ones(2),
            signgs=1,
            lamscale=1.0,
            pressure=np.zeros(2),
            solver="newton",
        )

    with pytest.raises(ValueError, match="must be provided together"):
        solve_fixed_boundary_state_implicit(
            state,
            static,
            phipf=np.ones(2),
            chipf=np.ones(2),
            signgs=1,
            lamscale=1.0,
            pressure=np.zeros(2),
            edge_Rcos=np.ones(3),
        )
