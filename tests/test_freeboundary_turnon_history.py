"""A change of force operator must not inherit a converged growth reference."""
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vmex.core import solver
from vmex.core.residuals import ForceResiduals, PreconditionedResiduals
from tests.test_freeboundary_turnon_convergence import model


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('compiled', [False, True])
def test_turnon_rebases_growth_history_but_keeps_real_growth_protection(monkeypatch, strict, compiled):
    rt, carry, _ = model(monkeypatch, strict, max_iterations=30)
    carry = dataclasses.replace(carry, fsq=jnp.asarray(1e-14))
    original = solver._evaluate

    def evaluate(state, cache, iteration, *args, **kwargs):
        e = original(state, cache, iteration, *args, **kwargs)
        # A healthy, changed operator, followed by actual divergent growth.
        pre = jnp.where(iteration >= 15, 1e-2, 1e-8)
        raw = jnp.asarray(1e-6)
        return dataclasses.replace(e,
            gc=jax.tree.map(lambda x: jnp.ones_like(x) * .01, state),
            residuals=ForceResiduals(raw, raw, raw, raw, raw, raw, raw),
            pre=PreconditionedResiduals(pre, pre, pre))

    monkeypatch.setattr(solver, '_evaluate', evaluate)
    turnon = solver._make_body(rt, evaluation_state=carry.state)
    normal = solver._make_body(rt)
    if compiled:
        turnon, normal = jax.jit(turnon), jax.jit(normal)
    with jax.disable_jit(not compiled):
        result = turnon(carry)
        np.testing.assert_allclose(float(result.res0), 3e-8 if strict else 1e-14, rtol=1e-6, atol=0)
        while int(result.iteration) < 15:
            result = normal(result)
        if strict:
            # The finite activation jump causes no backoff or lost progress.
            np.testing.assert_allclose(result.time_step, carry.time_step)
            assert np.max(np.abs(np.asarray(result.state.R_cos))) > 0
            result = normal(result)  # records the new, genuinely large residual
            result = normal(result)  # growth detection uses the previous residual
            np.testing.assert_allclose(result.time_step, carry.time_step / 1.03)
            assert int(result.iter1) == 16
        else:
            # Preserve the legacy VMEC-compatible turn-on behavior.
            assert float(result.time_step) < float(carry.time_step)


def test_ordinary_fixed_boundary_restart_reference_is_unchanged(monkeypatch):
    rt, carry, _ = model(monkeypatch, True)
    carry = dataclasses.replace(carry, fsq=jnp.asarray(1e-14))
    result = solver._make_body(rt)(carry)
    np.testing.assert_allclose(result.res0, 1e-14, atol=0, rtol=1e-6)
