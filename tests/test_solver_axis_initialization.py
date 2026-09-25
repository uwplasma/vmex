"""Production-axis initialization parity with VMEC2000."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from vmex.core.input import VmecInput
from vmex.core.setup import run_setup
from vmex.core.solver import prepare_runtime, resolution_from_input


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_production_runtime_does_not_preinfer_missing_axis() -> None:
    """eqsolve, not setup, owns VMEC2000's one allowed guess_axis transfer."""
    inp = VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
    zeros = np.zeros(inp.ntor + 1)
    inp = dataclasses.replace(
        inp,
        raxis_c=zeros,
        raxis_s=zeros,
        zaxis_c=zeros,
        zaxis_s=zeros,
    )
    resolution = resolution_from_input(inp)

    runtime = prepare_runtime(inp, resolution)
    np.testing.assert_array_equal(runtime.setup.raxis_c, zeros)
    np.testing.assert_array_equal(runtime.setup.raxis_s, zeros)
    np.testing.assert_array_equal(runtime.setup.zaxis_c, zeros)
    np.testing.assert_array_equal(runtime.setup.zaxis_s, zeros)

    inferred = run_setup(
        inp, resolution, infer_axis_if_missing=True,
    )
    assert np.any(np.asarray(inferred.raxis_c) != 0.0)

    # Geometry/force-kernel callers can opt into the setup convenience
    # explicitly without changing the production default.
    inferred_runtime = prepare_runtime(inp, resolution, setup=inferred)
    np.testing.assert_array_equal(
        inferred_runtime.setup.raxis_c, inferred.raxis_c,
    )


def _seed_input() -> VmecInput:
    """The A1 optimization benchmark's seed deck at its max_mode=1 resolution."""
    inp = VmecInput.from_file(DATA / "input.minimal_seed_nfp2")
    return dataclasses.replace(inp, delt=0.5).change_resolution(
        mpol=5, ntor=5, ntheta=16, nzeta=14,
    )


def _axis_decks():
    yield "minimal_seed_nfp2", _seed_input()
    for name in ("cth_like_fixed_bdy", "solovev", "LandremanPaul2021_QA_lowres",
                 "up_down_asymmetric_tokamak"):
        yield name, VmecInput.from_file(DATA / f"input.{name}")
    free = VmecInput.from_file(DATA / "input.cth_like_free_bdy_lasym_small")
    yield "cth_like_free_bdy_lasym_small", dataclasses.replace(free, lfreeb=False)


def test_traced_guess_axis_selects_the_host_grid_points() -> None:
    """The fixed-shape JAX guess_axis picks the host's per-plane grid points.

    Coefficients may differ only by the rounding of the final Fourier
    projection, eager and under jax.jit, on the decks of the axis-retry tests
    and the optimization seed deck.
    """
    import jax

    from vmex.core import solver
    from vmex.core.setup import guess_axis

    for name, inp in _axis_decks():
        runtime = prepare_runtime(inp, resolution_from_input(inp))
        state = solver._initial_state(runtime.setup)
        _, geometry = solver._geometry_lane(state, runtime, use_fft=False)
        kwargs = dict(s=runtime.setup.s_full, trig=runtime.trig, signgs=runtime.setup.signgs)
        host = [np.asarray(a) for a in guess_axis(geometry, **kwargs)]
        scale = max(float(np.max(np.abs(a))) for a in host)
        with jax.disable_jit(False):
            eager = solver._guess_axis_traced(geometry, **kwargs)
            jitted = jax.jit(lambda g: solver._guess_axis_traced(g, **kwargs))(geometry)
        for label, port in (("eager", eager), ("jit", jitted)):
            worst = max(float(np.max(np.abs(np.asarray(p) - h))) for p, h in zip(port, host))
            assert worst <= 1.0e-12 * scale, (name, label, worst, scale)


def test_traced_stage_follows_the_host_axis_retry() -> None:
    """jax.jit(_solve_stage_traced) makes the host driver's decisions.

    Solov'ev converges without a retry; the seed deck needs one axis re-guess
    after a first-iteration bad Jacobian.  Status, iterations and Jacobian
    resets are identical, and the final states agree to rounding.
    """
    import jax

    from vmex.core import solver

    for inp in (VmecInput.from_file(DATA / "input.solovev"), _seed_input()):
        runtime = prepare_runtime(inp, resolution_from_input(inp), use_fft=False)
        time_step0, nstep = solver._loop_driver_config(inp)
        with jax.disable_jit(False):
            host = solver._solve_stage(
                runtime, None, mode="jit", verbose=False, emit=None,
                time_step0=time_step0, nstep=nstep, use_fft=False,
            )
            traced = jax.jit(
                lambda rt: solver._solve_stage_traced(rt, None, time_step0=time_step0)
            )(runtime)
        assert [int(host.ier), int(host.iteration), int(host.ijacob)] == [
            int(traced.ier), int(traced.iteration), int(traced.ijacob)]
        for field in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"):
            np.testing.assert_allclose(
                np.asarray(getattr(traced.state, field)),
                np.asarray(getattr(host.state, field)), rtol=0.0, atol=1.0e-12,
            )
