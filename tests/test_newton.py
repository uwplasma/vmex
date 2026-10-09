"""Opt-in Newton free-boundary solve and its ideal-MHD stability count.

The fixture is a low-beta circular tokamak in an analytic external field: a
toroidal field ``B_phi = R0 Bt / R`` plus a weakly decaying vertical field
that balances the hoop force, tabulated to a small one-group mgrid.  It is
ideal-MHD stable, so the force Jacobian has no negative eigenvalue.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import vmex
from vmex.core.mgrid import MgridField, tabulate_cartesian_field

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

R0, A, BT, IP, BV, DECAY = 1.5, 0.4, 1.0, 1.0e5, 1.43e-2, 0.5


def _external_field(points):
    x, y, z = np.moveaxis(np.asarray(points), -1, 0)
    radius, phi = np.hypot(x, y), np.arctan2(y, x)
    b_r = DECAY * BV * z / R0
    b_z = -BV * (R0 / radius) ** DECAY
    b_phi = R0 * BT / radius
    return np.stack((b_r * np.cos(phi) - b_phi * np.sin(phi),
                     b_r * np.sin(phi) + b_phi * np.cos(phi), b_z), axis=-1)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    jax.config.update("jax_enable_x64", True)
    deck = tmp_path_factory.mktemp("newton") / "input.tokamak"
    deck.write_text(f"""&INDATA
 LFREEB=T, MGRID_FILE='x', NFP=1, MPOL=3, NTOR=0, NZETA=1, NTHETA=0
 NS_ARRAY=8, FTOL_ARRAY=1e-8, NITER_ARRAY=4000, NSTEP=100, DELT=0.9
 TCON0=1.0, NVACSKIP=3, GAMMA=0.0, PHIEDGE={np.pi * A * A * BT}
 NCURR=1, CURTOR={IP}, AC=1.0,0.0,0.0, PCURR_TYPE='power_series'
 PRES_SCALE=0.0, AM=0.0, EXTCUR=1.0, RAXIS_CC={R0}, ZAXIS_CS=0.0
 RBC(0,0)={R0}, RBC(0,1)={A}, ZBS(0,1)={A}
/
""")
    table = tabulate_cartesian_field(
        _external_field, rmin=0.7, rmax=2.3, zmin=-0.8, zmax=0.8,
        ir=33, jz=33, kp=1, nfp=1)
    return vmex.VmecInput.from_file(deck), MgridField.from_mgrid_data(
        table, extcur=np.ones(1))


@pytest.fixture(scope="module")
def stable(case):
    return vmex.solve_free_boundary_newton(*case)


def test_newton_converges_a_stable_free_boundary_case(stable):
    """The stable tokamak lands below 1e-10 with no unstable direction."""
    assert stable.converged and stable.residual < 1.0e-10
    assert stable.n_unstable == 0
    assert len(stable.modes) == 4
    assert all(eigenvalue > 0 for eigenvalue, _ in stable.modes)
    assert all(len(top) == 2 for _, top in stable.modes)


def test_newton_restarts_from_a_wout_and_can_skip_the_count(case, stable):
    """A converged wout is a valid start; ``count_modes=False`` skips eigs."""
    assert vmex.NewtonResult is type(stable)
    again = vmex.solve_free_boundary_newton(
        *case, start=stable.wout, count_modes=False)
    assert again.converged and again.residual < 1.0e-10
    assert again.n_unstable == -1 and not again.modes
    from_state = vmex.solve_free_boundary_newton(
        *case, start=stable.state, count_modes=False)
    assert from_state.converged


def test_newton_fixed_boundary_start_converges(case):
    """The fixed-boundary start also reaches a stable root below 1e-10."""
    other = vmex.solve_free_boundary_newton(*case, start="fixed")
    assert other.converged and other.n_unstable == 0
    assert other.residual < 1.0e-10
