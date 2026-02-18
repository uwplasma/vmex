import numpy as np

import vmec_jax.api as vj


def test_vmecpp_restart_flag_no_effect_short_run():
    input_path = "examples/data/input.circular_tokamak"
    kwargs = dict(
        solver="vmec2000_iter",
        max_iter=2,
        multigrid=False,
        ns_override=13,
        verbose=False,
        performance_mode=False,
    )

    base = vj.run_fixed_boundary(input_path, **kwargs)
    alt = vj.run_fixed_boundary(input_path, vmecpp_restart=True, **kwargs)

    np.testing.assert_allclose(
        np.asarray(base.result.w_history),
        np.asarray(alt.result.w_history),
        rtol=1e-12,
        atol=1e-12,
    )
