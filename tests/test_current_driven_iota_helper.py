from __future__ import annotations

import numpy as np
import pytest


pytestmark = pytest.mark.full


def test_equilibrium_iota_profiles_from_state_matches_wout():
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64
    import vmec_jax as vj

    enable_x64(True)

    run = vj.run_fixed_boundary(
        "examples/data/input.LandremanPaul2021_QA_lowres",
        solver="vmec2000_iter",
        max_iter=5,
        multigrid_use_input_niter=False,
        ns_override=16,
        verbose=False,
    )
    chips, iotas, iotaf = vj.equilibrium_iota_profiles_from_state(
        state=run.state,
        static=run.static,
        indata=run.indata,
        signgs=run.signgs,
    )
    wout = vj.wout_from_fixed_boundary_run(run, include_fsq=True)

    chips_np = np.asarray(chips)
    iotas_np = np.asarray(iotas)
    iotaf_np = np.asarray(iotaf)

    assert np.all(np.isfinite(chips_np))
    assert np.all(np.isfinite(iotas_np))
    assert np.all(np.isfinite(iotaf_np))
    assert np.allclose(iotas_np, np.asarray(wout.iotas), rtol=1e-10, atol=1e-10)
    assert np.allclose(iotaf_np, np.asarray(wout.iotaf), rtol=1e-10, atol=1e-10)
    assert abs(float(iotaf_np[-1])) > 1e-6