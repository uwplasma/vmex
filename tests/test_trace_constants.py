"""Trace-time tables enter a compiled program once, however often the program evaluates them.

A traced program captures a NumPy array as a constant, one per array object.  The Boozer projection tables
(:func:`vmex.core.boozer_tables.boozer_input_tables`) and the Fourier phase tables
(:mod:`vmex.core.transforms`) are built once per grid and handed out as the same read-only arrays, so a program
that evaluates them on many surfaces, fields or geometries holds one copy of each.  Rebuilt per call they put
gigabytes of identical constants into the free-boundary design gradient's programs.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from vmex.core import boozer_tables, solver, transforms
from vmex.core.input import VmecInput

jax.config.update("jax_enable_x64", True)

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


def _constants(text: str, shape: tuple[int, ...]) -> int:
    """How many dense constants of ``shape`` (float64) a lowered program holds."""
    tensor = "x".join(map(str, shape)) + "xf64"
    return len(re.findall(rf"stablehlo\.constant dense<[^>]*> : tensor<{tensor}>", text))


@pytest.fixture(scope="module")
def solovev():
    inp = VmecInput.from_file(str(DATA_DIR / "input.solovev"))
    resolution = solver.resolution_from_input(inp)
    result = solver.solve(inp, resolution, ftol=1e-12, max_iterations=3000, mode="cli")
    rt = solver.prepare_runtime(inp, resolution, ftol=1e-12, max_iterations=3000)
    return result.state, rt


def test_projection_tables_are_built_once_and_read_only():
    first = boozer_tables._projection_tables(8, 6, 3)
    assert boozer_tables._projection_tables(8, 6, 3) is first
    xm, xn, w, cos_t, sin_t = first
    assert cos_t.shape == sin_t.shape == (8, 6, xm.size) and w.shape == xm.shape == xn.shape
    for table in first:
        assert not table.flags.writeable
    with pytest.raises(ValueError):
        cos_t[0, 0, 0] = 1.0


def test_boozer_tables_hold_one_projection_table_for_every_row(solovev):
    state, rt = solovev
    ns = int(np.asarray(rt.setup.s_full).size)
    rows = (2, ns // 2, ns - 2)

    def tables(state):
        return [{k: v for k, v in boozer_tables.boozer_input_tables(state, rt, j).items() if k not in ("xm", "xn")}
                for j in rows]

    text = jax.jit(tables).lower(state).as_text()
    xm = boozer_tables.boozer_input_tables(state, rt, rows[0])["xm"]
    # one cosine and one sine table (ntheta, nzeta, modes) for all three rows and every field projected on them
    projection = re.findall(rf"stablehlo\.constant dense<[^>]*> : tensor<\d+x\d+x{xm.size}xf64>", text)
    assert len(projection) == 2


def test_phase_tables_are_shared_by_every_geometry_evaluation(solovev):
    _, rt = solovev
    modes, trig = rt.modes, rt.trig
    for derivative in ("value", "dtheta", "dzeta"):
        table = transforms._stacked_phase(modes, trig, derivative)
        assert transforms._stacked_phase(modes, trig, derivative) is table and not table.flags.writeable
        np.testing.assert_array_equal(table, np.concatenate(transforms._phase_pair(modes, trig, derivative)))
    n_theta = int(trig.ntheta2)
    cos_u, sin_u = transforms._unscaled_phase(modes, trig, n_theta)
    assert transforms._unscaled_phase(modes, trig, n_theta)[0] is cos_u and not sin_u.flags.writeable

    ns, mnmax = 5, modes.mnmax
    coefficients = jnp.asarray(np.linspace(0.1, 1.0, ns * mnmax).reshape(ns, mnmax))

    def twice(c):
        first = transforms.fourier_to_real(c, 0.5 * c, modes=modes, trig=trig, derivatives=("value",))[0]
        second = transforms.fourier_to_real(2.0 * c, c, modes=modes, trig=trig, derivatives=("value",))[0]
        back = transforms.real_to_fourier(first + second, modes=modes, trig=trig)
        again = transforms.real_to_fourier(second, modes=modes, trig=trig)
        return first, second, back, again

    text = jax.jit(twice).lower(coefficients).as_text()
    stacked = transforms._stacked_phase(modes, trig, "value")
    assert _constants(text, stacked.shape) == 1  # both syntheses share one stacked table
    assert _constants(text, cos_u.shape) == 2  # both projections share one cosine and one sine table
