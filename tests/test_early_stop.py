"""Early stop of a solve that cannot converge within its budget (:class:`vmex.core.solver.EarlyStop`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

from vmex.core import multigrid
from vmex.core.input import VmecInput
from vmex.core.solver import EarlyStop

DECK = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.cth_like_fixed_bdy"


def _checks(rate, iterations, f0=1.0, block=10):
    """Block-end iteration counts and best residuals of a residual falling ``rate`` decades per iteration."""
    k = np.arange(block - 1, iterations, block)
    return k, f0 * 10.0 ** (-np.asarray(rate(k), dtype=float))


def test_steady_rate_that_reaches_the_target_is_never_stopped():
    stop = EarlyStop(target=1e-13, window=100)
    # 13 decades in 1300 iterations against a budget of 1400: tight, and never stopped on the way
    k, f = _checks(lambda k: k / 100.0, 1300)
    assert not any(stop.exhausts_budget(k[:n], f[:n], 1400) for n in range(1, k.size + 1))


def test_slowing_rate_that_still_converges_is_not_stopped():
    stop = EarlyStop(target=1e-13, window=100)
    # fast at first, then a third of the rate: 13 decades reached at iteration 2700 of 3000
    k, f = _checks(lambda k: np.where(k < 500, k / 50.0, 10.0 + (k - 500) / 733.0), 2700)
    assert f[-1] <= 1.01e-13
    assert not any(stop.exhausts_budget(k[:n], f[:n], 3000) for n in range(1, k.size + 1))


def test_rate_too_slow_for_the_budget_is_stopped_early():
    stop = EarlyStop(target=1e-13, window=100)
    k, f = _checks(lambda k: k / 10000.0, 10000)  # 13 decades take 130000 iterations
    stops = [n for n in range(1, k.size + 1) if stop.exhausts_budget(k[:n], f[:n], 10000)]
    assert stops and k[stops[0] - 1] < 300


def test_flat_residual_is_stopped_and_early_or_reached_ones_are_not():
    stop = EarlyStop(target=1e-13, window=100)
    k = np.arange(9, 400, 10)
    flat = np.full(k.size, 1e-6)
    assert not stop.exhausts_budget(k[:19], flat[:19], 10000)  # before 2 * window
    assert stop.exhausts_budget(k, flat, 10000)
    assert not stop.exhausts_budget(k, np.full(k.size, 1e-14), 10000)  # target reached


def _solve(ftol, niter, early_stop):
    inp = VmecInput.from_file(str(DECK))
    return multigrid.solve_multigrid(
        inp, ns_array=[5, 9, 15], ftol_array=[1e-8, 1e-10, ftol], niter_array=[niter] * 3,
        raise_on_max_iterations=False, early_stop=early_stop)


def test_unreachable_tolerance_stops_before_the_budget():
    result = _solve(1e-40, 4000, EarlyStop(target=1e-40, window=50))
    assert not result.converged
    assert result.iterations < 1000
    assert np.all(result.fsq_history[:result.iterations, :3].sum(axis=1) > 0.0)


def test_converging_solve_is_unchanged():
    plain = _solve(1e-14, 4000, None)
    stopped = _solve(1e-14, 4000, EarlyStop(target=1e-14, window=50))
    assert plain.converged and stopped.converged
    assert plain.iterations == stopped.iterations > 100
    for a, b in zip(jax.tree.leaves(plain.state), jax.tree.leaves(stopped.state)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("window", [50])
def test_early_stop_keys_implicit_configs(window):
    from dataclasses import replace

    from vmex.core import implicit as im

    cfg = im.make_config(VmecInput.from_file(str(DECK)), multigrid=True)
    stopped = im._canonical_config(replace(cfg, early_stop=EarlyStop(target=1e-12, window=window)))
    assert im._canonical_config(cfg) is not stopped
    assert im._canonical_config(replace(cfg, early_stop=EarlyStop(target=1e-12, window=window))) is stopped


def test_trial_config_stops_early_at_the_accepted_residual():
    from vmex.core import freeboundary_vc as fvc

    inp = VmecInput.from_file(str(DECK))
    cfg = fvc._trial_config(inp, 1e-11, jax.devices()[0])
    assert cfg.ftol == max(1e-11, float(np.asarray(inp.ftol_array)[-1]))
    assert cfg.early_stop == EarlyStop(target=cfg.max_fsq_ratio * cfg.ftol, window=fvc._TRIAL_RATE_WINDOW)
