from __future__ import annotations

import numpy as np
import pytest


def test_qi_boozer_mode_residual_is_zero_for_alpha_independent_wells():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_boozer_modes

    # B = 1 + 0.1 cos(phi) has identical wells for every field-line label alpha.
    out = quasi_isodynamic_residual_from_boozer_modes(
        bmnc_b=jnp.asarray([[1.0, 0.1]]),
        xm_b=jnp.asarray([0, 0]),
        xn_b=jnp.asarray([0, 1]),
        iota_b=jnp.asarray([0.4]),
        nfp=1,
        nphi=33,
        nalpha=9,
        n_bounce=7,
    )

    np.testing.assert_allclose(np.asarray(out["total"]), 0.0, atol=1e-28, rtol=0.0)
    assert np.asarray(out["width_residuals1d"]).shape == (9 * 7,)
    assert np.asarray(out["profile_residuals1d"]).shape == (33 * 9,)
    assert np.asarray(out["residuals1d"]).shape == (9 * 7 + 33 * 9,)


def test_qi_boozer_mode_residual_rejects_single_helicity_phase_shift():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_boozer_modes

    # A QH-like B = 1 + eps cos(theta - phi) has nearly identical bounce
    # widths across field-line label alpha, but the well location shifts. The
    # profile term should keep this from being treated as QI.
    out = quasi_isodynamic_residual_from_boozer_modes(
        bmnc_b=jnp.asarray([[1.0, 0.1]]),
        xm_b=jnp.asarray([0, 1]),
        xn_b=jnp.asarray([0, 1]),
        iota_b=jnp.asarray([0.4]),
        nfp=1,
        nphi=33,
        nalpha=9,
        n_bounce=7,
    )

    assert float(np.asarray(out["total"])) > 1.0e-4
    assert np.linalg.norm(np.asarray(out["profile_residuals1d"])) > 0.0


def test_qi_boozer_mode_residual_is_differentiable():
    pytest.importorskip("jax")

    import jax

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_boozer_modes

    xm_b = jnp.asarray([0, 1, 1])
    xn_b = jnp.asarray([0, 0, 1])

    def objective(coeffs):
        out = quasi_isodynamic_residual_from_boozer_modes(
            bmnc_b=coeffs[None, :],
            xm_b=xm_b,
            xn_b=xn_b,
            iota_b=jnp.asarray([0.4]),
            nfp=1,
            nphi=33,
            nalpha=9,
            n_bounce=7,
        )
        return out["total"]

    coeffs = jnp.asarray([1.0, 0.05, 0.08])
    value, grad = jax.value_and_grad(objective)(coeffs)

    assert np.isfinite(np.asarray(value))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.linalg.norm(np.asarray(grad[1:])) > 0.0


def test_qi_boozer_output_wrapper_matches_mode_residual_regression():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import (
        quasi_isodynamic_residual_from_boozer_modes,
        quasi_isodynamic_residual_from_boozer_output,
    )

    kwargs = dict(
        bmnc_b=jnp.asarray([[1.0, 0.08, 0.03], [1.1, 0.05, -0.02]]),
        xm_b=jnp.asarray([0, 1, 2]),
        xn_b=jnp.asarray([0, 1, 0]),
        iota_b=jnp.asarray([0.4, 0.55]),
        nfp=2,
        weights=[1.0, 2.0],
        nphi=21,
        nalpha=7,
        n_bounce=5,
        softness=1.0e-2,
        profile_weight=0.8,
    )
    direct = quasi_isodynamic_residual_from_boozer_modes(**kwargs)
    wrapped = quasi_isodynamic_residual_from_boozer_output(
        {
            "bmnc_b": kwargs["bmnc_b"],
            "ixm_b": kwargs["xm_b"],
            "ixn_b": kwargs["xn_b"],
            "iota_b": kwargs["iota_b"],
            "nfp_b": jnp.asarray(kwargs["nfp"]),
        },
        weights=kwargs["weights"],
        nphi=kwargs["nphi"],
        nalpha=kwargs["nalpha"],
        n_bounce=kwargs["n_bounce"],
        softness=kwargs["softness"],
        profile_weight=kwargs["profile_weight"],
    )

    np.testing.assert_allclose(np.asarray(direct["residuals1d"]), np.asarray(wrapped["residuals1d"]))
    np.testing.assert_allclose(np.asarray(wrapped["total"]), 0.4720832188870966, rtol=1.0e-12, atol=1.0e-14)
    assert np.asarray(wrapped["width_residuals1d"]).shape == (2 * 7 * 5,)
    assert np.asarray(wrapped["profile_residuals1d"]).shape == (2 * 21 * 7,)


def test_qi_boozer_mode_residual_validates_shapes_and_resolution():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import (
        _nearest_half_mesh_indices,
        quasi_isodynamic_residual_from_boozer_modes,
    )

    base = dict(
        bmnc_b=jnp.asarray([[1.0, 0.1]]),
        xm_b=jnp.asarray([0, 1]),
        xn_b=jnp.asarray([0, 1]),
        iota_b=jnp.asarray([0.4]),
        nfp=1,
        nphi=17,
        nalpha=7,
        n_bounce=5,
    )

    np.testing.assert_array_equal(_nearest_half_mesh_indices([0.1, 0.9], n_half=4), np.array([0, 3]))
    with pytest.raises(ValueError, match="at least one half-mesh"):
        _nearest_half_mesh_indices([0.5], n_half=0)
    with pytest.raises(ValueError, match="bmnc_b must have shape"):
        quasi_isodynamic_residual_from_boozer_modes(**{**base, "bmnc_b": jnp.asarray([1.0, 0.1])})
    with pytest.raises(ValueError, match="same mode dimension"):
        quasi_isodynamic_residual_from_boozer_modes(**{**base, "xm_b": jnp.asarray([0])})
    with pytest.raises(ValueError, match="one value per Boozer surface"):
        quasi_isodynamic_residual_from_boozer_modes(**{**base, "iota_b": jnp.asarray([0.4, 0.5])})
    with pytest.raises(ValueError, match="weights must have the same length"):
        quasi_isodynamic_residual_from_boozer_modes(**{**base, "weights": [1.0, 2.0]})
    with pytest.raises(ValueError, match="nphi >= 4"):
        quasi_isodynamic_residual_from_boozer_modes(**{**base, "nphi": 3})


def test_qi_state_residual_smoke(load_case_qh_warm_start):
    pytest.importorskip("jax")
    pytest.importorskip("booz_xform_jax")

    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_state

    _cfg, indata, static, _boundary, state = load_case_qh_warm_start
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state, static).sqrtg), axis_index=1))
    out = quasi_isodynamic_residual_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        surfaces=[0.5],
        mboz=4,
        nboz=4,
        nphi=17,
        nalpha=7,
        n_bounce=5,
        jit_booz=False,
    )

    assert np.asarray(out["width_residuals1d"]).shape == (7 * 5,)
    assert np.asarray(out["profile_residuals1d"]).shape == (17 * 7,)
    assert np.asarray(out["residuals1d"]).shape == (7 * 5 + 17 * 7,)
    assert np.isfinite(np.asarray(out["total"]))
