from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tools.diagnostics.performance import accelerated_scan_cache_probe as probe
from vmec_jax.state import StateLayout, VMECState


def _state() -> VMECState:
    layout = StateLayout(ns=3, K=2, lasym=False)
    zeros = np.zeros((3, 2), dtype=float)
    rcos = zeros.copy()
    rcos[:, 0] = 1.0
    rcos[:, 1] = 0.2
    return VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )


def test_perturb_boundary_edge_changes_only_requested_edge_coefficient() -> None:
    state = _state()

    perturbed = probe._perturb_boundary_edge(state, coefficient_index=1, perturbation=1.0e-4)

    np.testing.assert_allclose(np.asarray(perturbed.Rcos)[:-1], np.asarray(state.Rcos)[:-1])
    assert np.asarray(perturbed.Rcos)[-1, 1] == np.asarray(state.Rcos)[-1, 1] + 1.0e-4
    np.testing.assert_allclose(np.asarray(perturbed.Rsin), np.asarray(state.Rsin))
    np.testing.assert_allclose(np.asarray(perturbed.Zcos), np.asarray(state.Zcos))
    np.testing.assert_allclose(np.asarray(perturbed.Zsin), np.asarray(state.Zsin))


def test_run_probe_reports_cache_reuse_with_same_shape_boundary_update(monkeypatch, tmp_path) -> None:
    state = _state()
    cfg = SimpleNamespace(nfp=4, ns=35, mpol=2, ntor=2, lasym=False)
    indata = SimpleNamespace(get_float=lambda _name, default=1.0: 0.9)
    static = object()
    calls = []

    monkeypatch.setattr(probe, "_build_initial_state", lambda *_args, **_kwargs: (cfg, indata, static, state))
    monkeypatch.setattr(probe, "_clear_caches", lambda: None)

    def fake_run_accelerated_scan(**kwargs):
        calls.append(kwargs["state"])
        index = len(calls)
        return {
            "wall_s": 1.0 / index,
            "n_iter": 3,
            "final_fsq_total": float(index),
            "accelerated_scan": True,
            "scan_path": "accelerated",
            "vmec2000_scan": False,
            "scan_runner_cache_hit_count": 0 if index == 1 else 1,
            "scan_runner_cache_miss_count": 1 if index == 1 else 0,
            "scan_runner_cache_bypass_count": 0,
            "scan_run_setup_s": 0.0,
            "scan_device_run_s": 0.0,
            "scan_device_ready_s": 0.0,
            "scan_total_s": 0.0,
            "timing": {},
        }

    monkeypatch.setattr(probe, "_run_accelerated_scan", fake_run_accelerated_scan)

    args = SimpleNamespace(
        input=tmp_path / "input.fake",
        vmec_project=False,
        coefficient_index=1,
        perturbation=1.0e-4,
        signgs=-1,
        step_size=None,
        ftol=None,
        iters=3,
        jit_forces=True,
        precond_radial_alpha=0.0,
        precond_lambda_alpha=0.0,
        state_only=False,
        jax_platforms=None,
    )

    result = probe.run_probe(args)

    assert result["cache_reuse_pass"] is True
    assert result["cache_reuse_summary"]["same_shape_cache_hit"] is True
    assert result["cache_reuse_summary"]["first"]["miss_count"] == 1
    assert result["cache_reuse_summary"]["first"]["hit_fraction"] == 0.0
    assert result["cache_reuse_summary"]["second"]["hit_count"] == 1
    assert result["cache_reuse_summary"]["second"]["hit_fraction"] == 1.0
    assert result["cache_reuse_summary"]["second_to_first_wall_ratio"] == 0.5
    assert result["cache_reuse_summary"]["first_to_second_wall_speedup"] == 2.0
    assert len(calls) == 2
    assert np.asarray(calls[1].Rcos)[-1, 1] != np.asarray(calls[0].Rcos)[-1, 1]
