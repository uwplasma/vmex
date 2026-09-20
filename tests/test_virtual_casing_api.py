"""Public naming contract for prescribed-interface virtual casing."""

from pathlib import Path

import vmex as vj
from vmex.core import virtual_casing


def test_prescribed_interface_has_clear_public_name_and_compatibility_base():
    assert vj.PlasmaVacuumInterface is virtual_casing.PlasmaVacuumInterface
    assert virtual_casing.FreeBoundaryDiffProblem is virtual_casing.PlasmaVacuumInterface
    assert vj.surface_field_data_from_state is virtual_casing.surface_field_data_from_state
    assert (
        vj.surface_field_data_from_high_order
        is virtual_casing.surface_field_data_from_high_order
    )
    assert vj.surface_field_data_from_wout is virtual_casing.surface_field_data_from_wout


def test_exterior_source_grid_is_sized_from_the_boundary():
    """A fixed 32 x 32 missed the requested accuracy by four orders.

    The off-surface quadrature error decays as ``exp(-2 pi d / h)`` with ``h``
    the finest source level's largest spacing, so the sampling that reaches
    ``digits`` at one minor radius scales with ``R0 / a``.  The old constant
    default put the shipped QA boundary's achieved error at 2.5e-02 against a
    requested 1e-6; the rule returns 64 there and 4.3e-07.  A tokamak-like
    aspect ratio stays on the historical floor, so its grid does not move.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest

    from vmex.core import extender as ext

    class _Wout:
        def __init__(self, major, minor, nfp):
            # m = 0 and m = 1 only: R(theta) = major + minor cos(theta).
            self.rmnc = np.array([[major, minor]])
            self.xm = np.array([0.0, 1.0])
            self.nfp = nfp

    tokamak = _Wout(3.0, 1.0, 1)          # R0/a = 3
    assert ext._source_nphi_for_digits(tokamak, 6) == ext._DEFAULT_SOURCE_NPHI

    qa = _Wout(1.2237, 0.0768, 2)         # the shipped QA boundary
    assert ext._source_nphi_for_digits(qa, 6) == 64

    # More digits or a thinner boundary asks for more; the cap holds.
    assert ext._source_nphi_for_digits(qa, 12) > 64
    assert ext._source_nphi_for_digits(_Wout(100.0, 0.01, 1), 12) == ext._MAX_SOURCE_NPHI

    # A boundary the rule cannot read falls back rather than raising.
    assert ext._source_nphi_for_digits(object(), 6) == ext._DEFAULT_SOURCE_NPHI

    # A traced boundary is not unreadable: falling back there would silently
    # change the source resolution of a jitted caller.
    def traced(major):
        boundary = _Wout(1.2237, 0.0768, 2)
        boundary.rmnc = jnp.array([[1.0, 0.0768]]) * major
        return ext._source_nphi_for_digits(boundary, 6)
    with jax.disable_jit(False), pytest.raises(TypeError, match="traced boundary"):
        jax.jit(traced)(1.2237)

    # from_state feeds it a VmecInput instead of a wout; the same boundary has
    # to give the same answer through either, or a live equilibrium and its
    # exported wout would be extended on different grids.
    from vmex.core.input import VmecInput
    from vmex.core.wout import read_wout

    data = Path(__file__).resolve().parents[1] / "examples" / "data"
    reference = data / "wout_LandremanPaul2021_QA_lowres_reference.nc"
    if reference.exists():
        assert (ext._source_nphi_for_digits(read_wout(reference), 6)
                == ext._source_nphi_for_digits(
                    VmecInput.from_file(data / "input.LandremanPaul2021_QA_lowres"), 6)
                == 64)
    assert ext._source_nphi_for_digits(
        VmecInput.from_file(data / "input.circular_tokamak"), 6
    ) == ext._DEFAULT_SOURCE_NPHI
