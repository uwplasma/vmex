"""Small, mandatory accelerator checks for the manual GPU CI lane."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import jax
import numpy as np
import pytest

from vmex.core import device as device_policy
from vmex.core import implicit as im
from vmex.core import multigrid, solver
from vmex.core.input import VmecInput


DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


try:
    GPU = jax.devices("gpu")[0]
except RuntimeError:
    GPU = None


def _gpu():
    assert GPU is not None
    return GPU


def _platform(array) -> str:
    device = array.device
    return (device() if callable(device) else device).platform


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(GPU is None, reason="GPU unavailable"),
    pytest.mark.usefixtures("_module_jit_enabled"),
]


def test_gpu_is_default_without_platform_environment_pins():
    """The dedicated lane must fail rather than silently exercise the CPU."""
    _gpu()
    assert "JAX_PLATFORMS" not in os.environ
    assert "JAX_PLATFORM_NAME" not in os.environ
    assert jax.default_backend() == "gpu"
    assert _platform(jax.numpy.ones(())) == "gpu"


def test_implicit_auto_prefers_cpu_but_none_follows_jax_gpu():
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    automatic = im.params_from_input(inp)
    following_jax = im.params_from_input(inp, device=None)

    assert {_platform(x) for x in jax.tree.leaves(automatic)} == {"cpu"}
    assert {_platform(x) for x in jax.tree.leaves(following_jax)} == {"gpu"}


def test_explicit_forward_solve_cpu_gpu_parity():
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    results = {
        platform: solver.solve(
            inp, ftol=1e-12, max_iterations=1000, mode="jit", device=platform
        )
        for platform in ("cpu", "gpu")
    }

    assert results["cpu"].converged and results["gpu"].converged
    assert results["cpu"].iterations == results["gpu"].iterations
    assert _platform(results["cpu"].state.R_cos) == "cpu"
    assert _platform(results["gpu"].state.R_cos) == "gpu"
    np.testing.assert_allclose(
        results["gpu"].fsq_history,
        results["cpu"].fsq_history,
        rtol=5e-10,
        atol=1e-14,
    )
    for gpu_leaf, cpu_leaf in zip(
        jax.tree.leaves(results["gpu"].state),
        jax.tree.leaves(results["cpu"].state),
        strict=True,
    ):
        np.testing.assert_allclose(gpu_leaf, cpu_leaf, rtol=5e-10, atol=1e-12)

    # Committed hot starts must not defeat an explicit opposite-device request.
    for platform, seed in (("gpu", results["cpu"]), ("cpu", results["gpu"])):
        restarted = solver.solve(
            inp,
            ftol=1e-12,
            max_iterations=1000,
            mode="jit",
            initial_state=seed.state,
            device=platform,
        )
        assert restarted.converged
        assert _platform(restarted.state.R_cos) == platform


def test_multigrid_auto_moves_state_across_policy_threshold(monkeypatch):
    """A CPU coarse stage must not commit an AUTO-selected fine stage to CPU."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    inp = dataclasses.replace(
        inp,
        ns_array=np.asarray([5, 11]),
        ftol_array=np.asarray([1e-9, 1e-12]),
        niter_array=np.asarray([1000, 1000]),
    )
    monkeypatch.setattr(device_policy, "GPU_MIN_ITERATION_WORK", 500)
    seen = []
    solve_stage = multigrid._solve_stage

    def recording_solve_stage(*args, **kwargs):
        carry = solve_stage(*args, **kwargs)
        seen.append(_platform(carry.state.R_cos))
        return carry

    monkeypatch.setattr(multigrid, "_solve_stage", recording_solve_stage)
    result = multigrid.solve_multigrid(inp, mode="jit", device="auto")
    assert result.converged
    assert seen == ["cpu", "gpu"]


def test_explicit_implicit_gradient_cpu_gpu_parity():
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")

    def value_and_gradient(platform):
        params = im.params_from_input(inp, device=platform)
        value, gradient = jax.value_and_grad(
            lambda p: im.run(
                inp,
                p,
                ftol=1e-12,
                max_iterations=1000,
                device=platform,
            ).wb
        )(params)
        return value, gradient.rbc[inp.ntor, 1], gradient

    cpu_value, cpu_gradient, cpu_tree = value_and_gradient("cpu")
    gpu_value, gpu_gradient, gpu_tree = value_and_gradient(_gpu())

    assert {_platform(x) for x in jax.tree.leaves(cpu_tree)} == {"cpu"}
    assert {_platform(x) for x in jax.tree.leaves(gpu_tree)} == {"gpu"}
    np.testing.assert_allclose(gpu_value, cpu_value, rtol=1e-11)
    np.testing.assert_allclose(gpu_gradient, cpu_gradient, rtol=2e-7, atol=1e-12)
