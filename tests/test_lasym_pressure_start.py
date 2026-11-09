"""LASYM finite-pressure deck starts with finite forces (VMEC2000 readin.f orientation).

``input.basic_non_stellsym_pressure`` carries ``AM``/``GAMMA`` pressure.  Its
boundary must be oriented so that ``rbc(0,1)*zbs(0,1) > 0`` and
``rbs(0,1) = zbc(0,1)`` (``readin.f`` angle shift ``delta = 0``): the
mirror-image ``Z -> -Z`` deck gives ``delta = pi/2``, defeats the ``lflip``
test, leaves ``signgs*sqrt(g) < 0`` after the axis re-guess, and the
adiabatic closure ``mass / vp**gamma`` is NaN (VMEC2000 behaves identically).
"""

from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

from vmex.core import solver
from vmex.core.input import VmecInput

DECK = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.basic_non_stellsym_pressure"


def test_lasym_pressure_deck_converges_from_first_ns():
    inp = VmecInput.from_file(str(DECK))
    assert inp.rbc[inp.ntor, 1] * inp.zbs[inp.ntor, 1] > 0
    assert inp.rbs[inp.ntor, 1] == inp.zbc[inp.ntor, 1]
    resolution = solver.resolution_from_input(inp, ns=13)
    result = solver.solve(inp, resolution, ftol=1e-6, max_iterations=400, mode="cli")
    assert np.isfinite([float(result.fsqr), float(result.fsqz), float(result.fsql)]).all()
    assert result.converged
