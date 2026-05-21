from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.solve as solve
from vmec_jax._compat import has_jax, jnp
from vmec_jax.state import StateLayout, VMECState


pytestmark = pytest.mark.skipif(not has_jax(), reason="fixed-boundary L-BFGS requires JAX")


def _static():
    return SimpleNamespace(
        cfg=SimpleNamespace(nfp=1),
        modes=SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, 0])),
        s=np.asarray([0.0, 0.5, 1.0]),
        grid=SimpleNamespace(theta=np.asarray([0.0, np.pi]), zeta=np.asarray([0.0])),
    )


def _state(*, interior: float = 2.0) -> VMECState:
    layout = StateLayout(ns=3, K=2, lasym=False)
    zeros = np.zeros((3, 2), dtype=float)
    rcos = zeros.copy()
    rcos[:, 0] = 1.0
    rcos[1, 1] = interior
    rcos[-1, 1] = 0.25
    return VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )


def _install_quadratic_geometry(monkeypatch, *, offset: float = 1.0):
    def fake_eval_geom(state, _static):
        # Positive Jacobian with a real derivative in the interior coefficients.
        rsum = jnp.sum(jnp.asarray(state.Rcos), axis=1)
        return SimpleNamespace(sqrtg=rsum[:, None, None] ** 2 + offset)

    monkeypatch.setattr(solve, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(solve, "bsup_from_geom", lambda _g, **_kwargs: (0.0, 0.0))
    monkeypatch.setattr(solve, "b2_from_bsup", lambda g, _bsupu, _bsupv: jnp.ones_like(g.sqrtg))


def test_solve_fixed_boundary_lbfgs_accepts_step_and_records_history(monkeypatch):
    _install_quadratic_geometry(monkeypatch)
    state0 = _state(interior=2.0)

    result = solve.solve_fixed_boundary_lbfgs(
        state0,
        _static(),
        phipf=np.ones(3),
        chipf=np.zeros(3),
        signgs=1,
        lamscale=1.0,
        pressure=np.zeros(3),
        max_iter=3,
        step_size=0.25,
        history_size=2,
        grad_tol=0.0,
        max_backtracks=2,
        jit_grad=False,
        preconditioner="mode_diag",
        precond_exponent=0.5,
        verbose=False,
    )

    assert result.n_iter >= 1
    assert result.w_history[-1] < result.w_history[0]
    assert result.step_history.shape[0] >= 1
    assert result.diagnostics["history_size"] == 2
    assert result.diagnostics["preconditioner"] == "mode_diag"
    assert np.asarray(result.state.Rcos)[1, 1] != pytest.approx(np.asarray(state0.Rcos)[1, 1])
    np.testing.assert_allclose(np.asarray(result.state.Rcos)[-1], np.asarray(state0.Rcos)[-1])


def test_solve_fixed_boundary_lbfgs_rejects_invalid_initial_jacobian(monkeypatch):
    monkeypatch.setattr(
        solve,
        "eval_geom",
        lambda state, _static: SimpleNamespace(sqrtg=-jnp.ones((state.Rcos.shape[0], 1, 1), dtype=jnp.asarray(state.Rcos).dtype)),
    )
    monkeypatch.setattr(solve, "bsup_from_geom", lambda _g, **_kwargs: (0.0, 0.0))
    monkeypatch.setattr(solve, "b2_from_bsup", lambda g, _bsupu, _bsupv: jnp.ones_like(g.sqrtg))

    with pytest.raises(ValueError, match="invalid Jacobian"):
        solve.solve_fixed_boundary_lbfgs(
            _state(interior=1.0),
            _static(),
            phipf=np.ones(3),
            chipf=np.zeros(3),
            signgs=1,
            lamscale=1.0,
            pressure=np.zeros(3),
            max_iter=1,
            verbose=False,
        )
