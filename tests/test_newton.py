"""Opt-in Newton free-boundary solve.

The fixture is a low-beta circular tokamak in an analytic external field: a
toroidal field ``B_phi = R0 Bt / R`` plus a vertical field
``B_Z = -B_V (R0 / R)^n`` that balances the hoop force at ``R0``, tabulated to
a small one-group mgrid with decay index ``n = 0.5``.
"""

from __future__ import annotations

import numpy as np
import pytest

import vmex
from vmex.core.mgrid import MgridField, tabulate_cartesian_field

R0, A, BT, IP, BV = 1.5, 0.4, 1.0, 1.0e5, 1.43e-2


def _case(decay, tmp_path):
    def field(points):
        x, y, z = np.moveaxis(np.asarray(points), -1, 0)
        radius, phi = np.hypot(x, y), np.arctan2(y, x)
        b_r = decay * BV * z / R0
        b_z = -BV * (R0 / radius) ** decay
        b_phi = R0 * BT / radius
        return np.stack((b_r * np.cos(phi) - b_phi * np.sin(phi),
                         b_r * np.sin(phi) + b_phi * np.cos(phi), b_z), axis=-1)

    deck = tmp_path / f"input.tokamak_n{decay}"
    deck.write_text(f"""&INDATA
 LFREEB=T, MGRID_FILE='x', NFP=1, MPOL=3, NTOR=0, NZETA=1, NTHETA=0
 NS_ARRAY=8, FTOL_ARRAY=1e-8, NITER_ARRAY=4000, NSTEP=100, DELT=0.9
 TCON0=1.0, NVACSKIP=3, GAMMA=0.0, PHIEDGE={np.pi * A * A * BT}
 NCURR=1, CURTOR={IP}, AC=1.0,0.0,0.0, PCURR_TYPE='power_series'
 PRES_SCALE=0.0, AM=0.0, EXTCUR=1.0, RAXIS_CC={R0}, ZAXIS_CS=0.0
 RBC(0,0)={R0}, RBC(0,1)={A}, ZBS(0,1)={A}
/
""")
    table = tabulate_cartesian_field(field, rmin=0.7, rmax=2.3, zmin=-0.8, zmax=0.8,
                                     ir=33, jz=33, kp=1, nfp=1)
    return vmex.VmecInput.from_file(deck), MgridField.from_mgrid_data(table, extcur=np.ones(1))


@pytest.fixture(scope="module")
def stable_case(tmp_path_factory):
    return _case(0.5, tmp_path_factory.mktemp("newton"))


@pytest.fixture(scope="module")
def stable(stable_case):
    return vmex.solve_free_boundary_newton(*stable_case)


def test_newton_anchors_the_descent_root(stable):
    assert stable.converged and stable.residual < 1.0e-10
    # The wout carries VMEC's normalized invariant residuals at the root.
    assert max(stable.wout.fsqr, stable.wout.fsqz, stable.wout.fsql) < 1.0e-12


def test_restarts_land_on_the_same_root(stable_case, stable):
    """A wout or a state start solves the same residual (zero constraint baseline).

    Taking rcon0/zcon0 from a fixed-boundary run instead moved the volume by 7 %.
    """
    for start in (stable.wout, stable.state):
        again = vmex.solve_free_boundary_newton(*stable_case, start=start)
        assert again.converged
        assert float(again.wout.volume_p) == pytest.approx(float(stable.wout.volume_p), rel=1e-8)
