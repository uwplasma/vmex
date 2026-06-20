from __future__ import annotations

import pytest

from vmec_jax.solvers.fixed_boundary.residual.preconditioner_payload import (
    materialize_accepted_control_payload,
)


class _FakeJax:
    calls = []

    @classmethod
    def device_get(cls, value):
        cls.calls.append(value)
        return value


def _host_from_payload(payload, *, device_get_floats):
    if payload == "unused":
        return 0.0, None, False
    fsq1, min_tau, max_tau = device_get_floats(*payload)
    return fsq1, (min_tau, max_tau), True


def test_materialize_accepted_control_payload_uses_existing_payload() -> None:
    stats = {"iteration_control_fsq1_payload_get": 0.0, "iteration_control_fsq1_direct_get": 0.0}

    out = materialize_accepted_control_payload(
        accepted_control_ptau_payload=(1.25, -0.5, 2.0),
        use_control_payload=True,
        fsq1_j=9.0,
        k="kernel",
        ptau_pshalf_jax="pshalf",
        ptau_ohs_jax="ohs",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=iter([10.0, 10.25]).__next__,
        jax_module=_FakeJax,
        device_get_floats=lambda *vals: tuple(float(v) for v in vals),
        accepted_control_ptau_host_from_payload=_host_from_payload,
        scan_math_kernel_arrays_from_k=lambda _k: (_k,),
        accepted_control_payload_jit=lambda: None,
    )

    assert out.fsq1 == 1.25
    assert out.accepted_control_ptau_host == (-0.5, 2.0)
    assert out.control_payload_used is True
    assert stats["iteration_control_fsq1_payload_get"] == pytest.approx(0.25)
    assert stats["iteration_control_fsq1_direct_get"] == 0.0


def test_materialize_accepted_control_payload_builds_payload_when_requested() -> None:
    stats = {"iteration_control_fsq1_payload_get": 0.0, "iteration_control_fsq1_direct_get": 0.0}
    payload_calls = []

    def payload_fn(fsq1_j, *args):
        payload_calls.append((fsq1_j, args))
        return (2.5, -1.0, 3.0)

    out = materialize_accepted_control_payload(
        accepted_control_ptau_payload=None,
        use_control_payload=True,
        fsq1_j=2.5,
        k="kernel",
        ptau_pshalf_jax="pshalf",
        ptau_ohs_jax="ohs",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=iter([20.0, 20.5]).__next__,
        jax_module=_FakeJax,
        device_get_floats=lambda *vals: tuple(float(v) for v in vals),
        accepted_control_ptau_host_from_payload=_host_from_payload,
        scan_math_kernel_arrays_from_k=lambda _k: ("ptau",),
        accepted_control_payload_jit=lambda: payload_fn,
    )

    assert payload_calls == [(2.5, ("ptau", "pshalf", "ohs"))]
    assert out.fsq1 == 2.5
    assert out.accepted_control_ptau_host == (-1.0, 3.0)
    assert out.control_payload_used is True
    assert stats["iteration_control_fsq1_payload_get"] == pytest.approx(0.5)
    assert stats["iteration_control_fsq1_direct_get"] == 0.0


def test_materialize_accepted_control_payload_falls_back_to_direct_device_get() -> None:
    _FakeJax.calls = []
    stats = {"iteration_control_fsq1_payload_get": 0.0, "iteration_control_fsq1_direct_get": 0.0}

    out = materialize_accepted_control_payload(
        accepted_control_ptau_payload="unused",
        use_control_payload=True,
        fsq1_j=7.0,
        k="kernel",
        ptau_pshalf_jax="pshalf",
        ptau_ohs_jax="ohs",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=iter([30.0, 30.1, 40.0, 40.2]).__next__,
        jax_module=_FakeJax,
        device_get_floats=lambda *vals: tuple(float(v) for v in vals),
        accepted_control_ptau_host_from_payload=_host_from_payload,
        scan_math_kernel_arrays_from_k=lambda _k: None,
        accepted_control_payload_jit=lambda: None,
    )

    assert out.fsq1 == 7.0
    assert out.accepted_control_ptau_host is None
    assert out.control_payload_used is False
    assert _FakeJax.calls == [7.0]
    assert stats["iteration_control_fsq1_payload_get"] == pytest.approx(0.1)
    assert stats["iteration_control_fsq1_direct_get"] == pytest.approx(0.2)
