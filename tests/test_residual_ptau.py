from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.solvers.fixed_boundary.residual import config as residual_config
from vmec_jax.solvers.fixed_boundary.residual import policy as residual_policy
from vmec_jax.solvers.fixed_boundary.residual.ptau import resolve_bad_jacobian_tau_selection


def _startup_policy(**overrides):
    policy = dict(
        badjac_state_probe=False,
        badjac_initial_state_probe_iters=0,
        dump_ptau_state=False,
    )
    policy.update(overrides)
    return SimpleNamespace(**policy)


def _selection_kwargs(**overrides):
    timings: list[str] = []
    calls: dict[str, int] = {"ptau": 0, "device": 0, "state": 0}

    def ptau_minmax(_k):
        calls["ptau"] += 1
        return 1.0, 2.0

    def device_get(*values):
        calls["device"] += 1
        return tuple(float(value) for value in values)

    def state_tau(**_kwargs):
        calls["state"] += 1
        return 3.0, 4.0

    kwargs = dict(
        reference_mode=True,
        vmec2000_control=True,
        accepted_control_ptau_host=(1.0, 2.0),
        k=SimpleNamespace(),
        state=SimpleNamespace(),
        iter_idx=1,
        startup_policy=_startup_policy(),
        badjac_use_state=False,
        ptau_tol=0.0,
        static=SimpleNamespace(modes=SimpleNamespace(), cfg=SimpleNamespace(lconm1=True, lthreed=True)),
        trig=SimpleNamespace(),
        s=np.array([0.0, 1.0]),
        host_update_assembly=True,
        timing_enabled=True,
        perf_counter=lambda: 1.0,
        record_timing=lambda name, _start: timings.append(name),
        ptau_minmax_from_k_host_func=ptau_minmax,
        device_get_floats_func=device_get,
        should_probe_bad_jacobian_state_func=residual_config.should_probe_bad_jacobian_state,
        bad_jacobian_requires_state_jacobian_func=residual_policy.bad_jacobian_requires_state_jacobian,
        bad_jacobian_tau_decision_func=residual_policy.bad_jacobian_tau_decision,
        select_bad_jacobian_decision_func=residual_policy.select_bad_jacobian_decision,
        state_tau_minmax_from_vmec_state_func=state_tau,
        tree_has_tracer_func=lambda _value: False,
        jacobian_from_state_func=lambda **_kwargs: None,
        jnp_module=np,
        numpy_patch_context=None,
    )
    kwargs.update(overrides)
    return kwargs, timings, calls


def test_resolve_bad_jacobian_tau_selection_uses_accepted_ptau_without_state_probe() -> None:
    kwargs, timings, calls = _selection_kwargs()

    result = resolve_bad_jacobian_tau_selection(**kwargs)

    assert not result.bad_jacobian
    assert result.min_tau == 1.0
    assert result.max_tau == 2.0
    assert result.min_tau_ptau == 1.0
    assert result.max_tau_ptau == 2.0
    assert np.isnan(result.min_tau_state)
    assert np.isnan(result.max_tau_state)
    assert result.bad_jacobian_ptau is False
    assert result.bad_jacobian_state is False
    assert calls == {"ptau": 0, "device": 0, "state": 0}
    assert timings == []


def test_resolve_bad_jacobian_tau_selection_can_select_state_authority() -> None:
    kwargs, timings, calls = _selection_kwargs(
        accepted_control_ptau_host=(-1.0, 2.0),
        badjac_use_state=True,
        startup_policy=_startup_policy(badjac_state_probe=True, badjac_initial_state_probe_iters=2),
    )

    result = resolve_bad_jacobian_tau_selection(**kwargs)

    assert not result.bad_jacobian
    assert result.min_tau == 3.0
    assert result.max_tau == 4.0
    assert result.min_tau_ptau == -1.0
    assert result.max_tau_ptau == 2.0
    assert result.min_tau_state == 3.0
    assert result.max_tau_state == 4.0
    assert result.bad_jacobian_ptau is True
    assert result.bad_jacobian_state is False
    assert calls == {"ptau": 0, "device": 0, "state": 1}
    assert "iteration_control_badjac_state_jacobian" in timings
