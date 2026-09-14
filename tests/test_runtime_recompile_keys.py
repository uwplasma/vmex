"""Lane recompile keys of :class:`vmex.core.solver.SolverRuntime`.

Pins the perf contract that host loop-driver configuration never keys a lane
recompile, with lower-only checks and a small compiled-loop regression:

- ``nstep`` (print cadence) and ``time_step0`` (initial DELT) are call
  arguments of ``_run_loop``/``_initial_carry``, NOT runtime fields — as
  static pytree meta they forced a full lane recompile for a changed print
  cadence or initial time step;
- ``ftol`` is a traced data scalar (read inside the trace only in exact
  comparisons), so two runtimes differing only in tolerance share one
  structural key and one lowering, bit-exactly.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from vmex.core import solver
from vmex.core.input import VmecInput

DECK = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.solovev"


def _small_runtime(ftol: float = 1e-10) -> solver.SolverRuntime:
    inp = VmecInput.from_file(str(DECK))
    resolution = solver.resolution_from_input(inp, ns=5)
    return solver.prepare_runtime(inp, resolution, ftol=ftol, max_iterations=6)


def test_loop_driver_scalars_are_not_runtime_fields() -> None:
    """nstep/time_step0 are host loop-driver config, not solver context."""
    names = {f.name for f in dataclasses.fields(solver.SolverRuntime)}
    assert "nstep" not in names
    assert "time_step0" not in names


def test_initial_delt_does_not_change_lane_structure() -> None:
    """Two initial carries differing only in DELT share one lane key.

    This pins structure, not placement: commitment and sharding can also
    distinguish JAX executables (tested with actual calls below).
    """
    rt = _small_runtime()
    state = solver._initial_state(rt.setup)
    carry_a = solver._initial_carry(state, rt, ijacob=0, time_step0=0.9)
    carry_b = solver._initial_carry(state, rt, ijacob=0, time_step0=0.45)
    key_a = solver._lane_signature("block", carry_a, rt)
    key_b = solver._lane_signature("block", carry_b, rt)
    assert key_a == key_b
    # ... and the DELT value itself still reaches the carry.
    assert float(carry_a.time_step) != float(carry_b.time_step)


def test_ftol_variants_share_one_structural_key_and_lowering() -> None:
    """Runtimes differing only in ftol lower to the identical program."""
    import jax

    rt_a = _small_runtime(1e-10)
    rt_b = _small_runtime(1e-8)
    assert float(rt_a.ftol) != float(rt_b.ftol)

    leaves_a, tree_a = jax.tree_util.tree_flatten(rt_a)
    leaves_b, tree_b = jax.tree_util.tree_flatten(rt_b)
    assert tree_a == tree_b
    assert len(leaves_a) == len(leaves_b)

    state = solver._initial_state(rt_a.setup)
    carry_a = solver._initial_carry(state, rt_a, ijacob=0, time_step0=0.9)
    carry_b = solver._initial_carry(state, rt_b, ijacob=0, time_step0=0.9)
    key_a = solver._lane_signature("block", carry_a, rt_a)
    key_b = solver._lane_signature("block", carry_b, rt_b)
    assert key_a == key_b

    # Lower-only executable-identity pin: the ftol scalar is an ARGUMENT of
    # the lowered program, never a baked constant, so the lowered text is
    # byte-identical across tolerance variants at compatible placement.
    text_a = solver._block_lane.lower(carry_a, rt_a).as_text()
    text_b = solver._block_lane.lower(carry_b, rt_b).as_text()
    assert text_a == text_b
    assert not np.array_equal(float(rt_a.ftol), float(rt_b.ftol))


def test_mixed_commitment_reuses_lane_without_donating_inputs():
    """Predictor/axis-retry placement must not create another executable."""
    import jax

    with jax.disable_jit(False):
        rt = _small_runtime(1.0)
        state = solver._initial_state(rt.setup)
        target = state.R_cos.sharding
        originals = [np.asarray(x).copy() for x in jax.tree.leaves((state, rt))]
        solver._block_lane.clear_cache()
        results = []
        counts = []
        for mixed in (False, True):
            seed = dataclasses.replace(
                state, R_cos=jax.device_put(state.R_cos, target)
            ) if mixed else state
            runtime = dataclasses.replace(
                rt, rcon0=jax.device_put(rt.rcon0, target)
            ) if mixed else rt
            result = solver._run_loop(
                seed, runtime, mode="cli", ijacob=0, verbose=False, emit=None,
                time_step0=0.9, nstep=200,
                initial_residuals=(
                    tuple(jax.device_put(np.asarray(1.0), target) for _ in range(3))
                    if mixed else None
                ),
            )
            jax.block_until_ready(result)
            results.append(result)
            counts.append(solver._block_lane._cache_size())
        assert counts == [1, 1]
        for x, y in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1])):
            np.testing.assert_array_equal(x, y)
            assert x.sharding == target and y.sharding == target
        for expected, actual in zip(originals, jax.tree.leaves((state, rt))):
            np.testing.assert_array_equal(actual, expected)


def test_mixed_named_sharding_is_not_normalized(monkeypatch):
    """A single-device state must not overwrite another leaf's named layout."""
    import jax

    rt = _small_runtime(1.0)
    state = solver._initial_state(rt.setup)
    target = next(iter(state.R_cos.devices()))
    named = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.asarray([target]), ("device",)),
        jax.sharding.PartitionSpec(),
    )
    rt = dataclasses.replace(rt, rcon0=jax.device_put(rt.rcon0, named))
    seen = []

    def lane(carry, runtime):
        seen.append(runtime.rcon0.sharding)
        return dataclasses.replace(carry, done=True, iteration=0, ier=11)

    monkeypatch.setattr(solver, "_block_lane", lane)
    solver._run_loop(
        state, rt, mode="cli", ijacob=0, verbose=False, emit=None,
        time_step0=0.9, nstep=200,
    )
    assert seen == [named]
