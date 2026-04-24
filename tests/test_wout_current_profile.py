from __future__ import annotations

import numpy as np

from vmec_jax._compat import has_jax, jax, jnp
from vmec_jax.namelist import InData
from vmec_jax.wout import _icurv_full_mesh_from_indata


def test_icurv_full_mesh_is_jit_safe_for_current_driven_profile():
    if not has_jax():
        return

    indata = InData(
        scalars={
            "NCURR": 1,
            "CURTOR": 2.0,
            "PCURR_TYPE": "power_series",
            "AC": [1.0, 0.0],
        },
        indexed={},
    )
    s = jnp.linspace(0.0, 1.0, 5)

    @jax.jit
    def _eval(s_grid):
        return _icurv_full_mesh_from_indata(indata=indata, s_full=s_grid, signgs=-1)

    out = np.asarray(_eval(s))

    assert out.shape == (5,)
    assert out[0] == 0.0
    assert np.all(np.isfinite(out))
    assert np.any(np.abs(out[1:]) > 0.0)
