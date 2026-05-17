from types import SimpleNamespace

import numpy as np

from vmec_jax.namelist import InData
from vmec_jax.optimization import FixedBoundaryExactOptimizer


def _wout_ready_optimizer() -> FixedBoundaryExactOptimizer:
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._static = SimpleNamespace(cfg=SimpleNamespace())
    opt._indata = InData(scalars={}, indexed={}, source_path=None)
    opt._flux = object()
    opt._signgs = 1
    opt._profile = {}
    opt._exact_cache = {}
    opt._exact_state_cache = {}
    opt._exact_state_key_by_id = {}
    return opt


def test_save_wout_cache_miss_uses_exact_accepted_solve(monkeypatch, tmp_path) -> None:
    opt = _wout_ready_optimizer()
    params = np.asarray([1.0, -0.25])
    exact_state = SimpleNamespace(label="exact-accepted")
    solve_calls = []
    captured = {}

    def solve_forward(params_arg, *, trial=False):
        solve_calls.append((np.asarray(params_arg, dtype=float).copy(), bool(trial)))
        return exact_state

    def write_wout(path, run, **kwargs):
        captured.update({"path": path, "state": run.state, "kwargs": kwargs})

    opt._solve_forward = solve_forward
    monkeypatch.setattr("vmec_jax.driver.write_wout_from_fixed_boundary_run", write_wout)

    opt.save_wout(tmp_path / "wout_final.nc", params=params)

    assert len(solve_calls) == 1
    np.testing.assert_allclose(solve_calls[0][0], params)
    assert solve_calls[0][1] is False
    assert captured["state"] is exact_state
    assert captured["path"] == str(tmp_path / "wout_final.nc")


def test_save_wout_recomputes_when_supplied_state_is_stale(monkeypatch, tmp_path) -> None:
    opt = _wout_ready_optimizer()
    params = np.asarray([0.5])
    stale_state = SimpleNamespace(label="rejected-trial")
    exact_state = SimpleNamespace(label="exact-accepted")
    captured = {}

    def solve_forward(params_arg, *, trial=False):
        np.testing.assert_allclose(params_arg, params)
        assert trial is False
        return exact_state

    def write_wout(_path, run, **_kwargs):
        captured["state"] = run.state

    opt._solve_forward = solve_forward
    monkeypatch.setattr("vmec_jax.driver.write_wout_from_fixed_boundary_run", write_wout)

    opt.save_wout(tmp_path / "wout_final.nc", params=params, state=stale_state)

    assert captured["state"] is exact_state


def test_save_wout_trusts_known_exact_state_for_matching_params(monkeypatch, tmp_path) -> None:
    opt = _wout_ready_optimizer()
    params = np.asarray([0.75])
    exact_state = SimpleNamespace(label="exact-accepted")
    captured = {}

    key = opt._exact_cache_key(params)
    opt._remember_exact_state(key, exact_state)
    opt._solve_forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("matching accepted state should not be recomputed")
    )
    monkeypatch.setattr(
        "vmec_jax.driver.write_wout_from_fixed_boundary_run",
        lambda _path, run, **_kwargs: captured.update({"state": run.state}),
    )

    opt.save_wout(tmp_path / "wout_final.nc", params=params, state=exact_state)

    assert captured["state"] is exact_state
