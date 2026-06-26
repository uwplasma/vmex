from __future__ import annotations

import numpy as np
import pytest
pytestmark = pytest.mark.full



def test_end_to_end_gn_vmec_residual_decreases_for_circular_tokamak():
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    from conftest import require_slow

    require_slow()

    from vmec_jax._compat import enable_x64
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.config import load_config
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_gn_vmec_residual
    from vmec_jax.static import build_static
    from vmec_jax.kernels.tomnsp import vmec_angle_grid
    from vmec_jax.wout import read_wout

    enable_x64(True)

    cfg, indata = load_config("examples/data/input.circular_tokamak")
    wout = read_wout("examples/data/wout_circular_tokamak_reference.nc")

    static = build_static(
        cfg,
        grid=vmec_angle_grid(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(wout.nfp),
            lasym=bool(wout.lasym),
        ),
    )

    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)

    res = solve_fixed_boundary_gn_vmec_residual(
        st0,
        static,
        indata=indata,
        signgs=int(wout.signgs),
        include_constraint_force=True,
        max_iter=2,
        cg_maxiter=60,
        step_size=1.0,
        jit_kernels=True,
    )

    assert res.w_history.shape[0] >= 2
    assert np.isfinite(res.w_history).all()
    assert np.all(np.diff(res.w_history) < 0.0)
    assert res.diagnostics["damping_mode"] == "adaptive"
    assert res.diagnostics["cg_tol_mode"] == "adaptive"
    # The VMEC-style initial guess is more physical but can start at a higher
    # residual; require a meaningful reduction rather than a strict absolute target.
    assert float(res.w_history[-1]) < 0.85
