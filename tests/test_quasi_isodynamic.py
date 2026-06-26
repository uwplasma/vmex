from __future__ import annotations

from dataclasses import replace
from types import ModuleType, SimpleNamespace

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
        shuffle_profile_weight=0.0,
    )

    np.testing.assert_allclose(np.asarray(out["total"]), 0.0, atol=1e-28, rtol=0.0)
    assert np.asarray(out["width_residuals1d"]).shape == (9 * 7,)
    assert np.asarray(out["branch_width_residuals1d"]).shape == (9 * 7,)
    assert np.asarray(out["profile_residuals1d"]).shape == (33 * 9,)
    assert np.asarray(out["shuffle_profile_residuals1d"]).shape == (0,)
    assert np.asarray(out["residuals1d"]).shape == (2 * 9 * 7 + 33 * 9,)


@pytest.mark.py311_slow_coverage
def test_qi_residual_from_boozer_output_handles_surface_weights_and_sine_modes():
    pytest.importorskip("jax")

    from vmec_jax.quasi_isodynamic import (
        mirror_ratio_penalty_from_boozer_output,
        quasi_isodynamic_residual_from_boozer_output,
    )

    booz = {
        "bmnc_b": np.asarray([[1.0, 0.08, 0.02], [1.1, 0.04, -0.01]], dtype=float),
        "bmns_b": np.asarray([[0.0, 0.03, -0.02], [0.0, -0.01, 0.01]], dtype=float),
        "ixm_b": np.asarray([0, 0, 1], dtype=float),
        "ixn_b": np.asarray([0, 2, 2], dtype=float),
        "iota_b": np.asarray([0.42, 0.48], dtype=float),
        "nfp_b": np.asarray(2),
    }

    qi = quasi_isodynamic_residual_from_boozer_output(
        booz,
        weights=[1.0, 4.0],
        nphi=25,
        nalpha=7,
        n_bounce=5,
        shuffle_profile_weight=0.0,
        weighted_shuffle_profile_weight=0.5,
    )
    mirror = mirror_ratio_penalty_from_boozer_output(
        booz,
        weights=[1.0, 4.0],
        threshold=0.03,
        ntheta=16,
        nphi=16,
        smooth_extrema=1.0e-2,
        smooth_penalty=1.0e-2,
    )

    assert np.isfinite(float(np.asarray(qi["total"])))
    assert np.isfinite(float(np.asarray(mirror["total"])))
    assert np.asarray(qi["weighted_shuffle_profile_residuals1d"]).shape == (2 * 7 * 25,)
    assert np.asarray(qi["weighted_shuffle_alpha_weights"]).shape == (2, 7)
    np.testing.assert_allclose(np.sum(np.asarray(qi["weighted_shuffle_alpha_weights"]), axis=1), [1.0, 1.0])
    assert float(np.asarray(mirror["mirror_ratio"][0])) > float(np.asarray(mirror["mirror_ratio"][1]))


def test_qi_mirror_ratio_rank1_inputs_and_validation_edges():
    pytest.importorskip("jax")

    from vmec_jax.quasi_isodynamic import mirror_ratio_penalty_from_boozer_modes

    out = mirror_ratio_penalty_from_boozer_modes(
        bmnc_b=np.asarray([1.0, 0.2]),
        bmns_b=np.asarray([0.0, 0.05]),
        xm_b=np.asarray([0, 1]),
        xn_b=np.asarray([0, 1]),
        nfp=1,
        weights=[4.0],
        threshold=0.0,
        ntheta=16,
        nphi=16,
        smooth_extrema=1.0e-2,
        smooth_penalty=1.0e-2,
    )

    assert np.asarray(out["residuals1d"]).shape == (1,)
    assert float(np.asarray(out["mirror_ratio"][0])) > 0.0
    np.testing.assert_allclose(
        np.asarray(out["residuals1d"])[0],
        np.asarray(out["penalty"])[0] * 2.0,
    )

    with pytest.raises(ValueError, match="same mode dimension"):
        mirror_ratio_penalty_from_boozer_modes(
            bmnc_b=np.asarray([[1.0, 0.1]]),
            xm_b=np.asarray([0]),
            xn_b=np.asarray([0]),
            nfp=1,
        )
    with pytest.raises(ValueError, match="same shape"):
        mirror_ratio_penalty_from_boozer_modes(
            bmnc_b=np.asarray([[1.0, 0.1]]),
            bmns_b=np.asarray([[0.0, 0.0, 0.0]]),
            xm_b=np.asarray([0, 1]),
            xn_b=np.asarray([0, 1]),
            nfp=1,
        )
    with pytest.raises(ValueError, match="weights must have"):
        mirror_ratio_penalty_from_boozer_modes(
            bmnc_b=np.asarray([[1.0, 0.1]]),
            xm_b=np.asarray([0, 1]),
            xn_b=np.asarray([0, 1]),
            nfp=1,
            weights=[1.0, 2.0],
        )


def test_qi_boundary_elongation_proxy_tracks_ellipse_and_validates_phi():
    pytest.importorskip("jax")

    from vmec_jax.quasi_isodynamic import boundary_max_elongation_from_rz

    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    R = 3.0 + np.cos(theta)[:, None] * np.ones_like(phi)[None, :]
    Z = 0.5 * np.sin(theta)[:, None] * np.ones_like(phi)[None, :]

    out = boundary_max_elongation_from_rz(R, Z, phi=phi, smooth_extrema=1.0e-2)

    assert float(np.asarray(out["max_elongation"])) > 1.5
    assert np.all(np.isfinite(np.asarray(out["elongation"])))
    with pytest.raises(ValueError, match="phi must have"):
        boundary_max_elongation_from_rz(R, Z, phi=phi[:-1])
    with pytest.raises(ValueError, match="R and Z"):
        boundary_max_elongation_from_rz(R[:, :, None], Z)


def test_qi_residual_from_state_uses_nearest_half_mesh_and_boozer_wrapper(monkeypatch):
    pytest.importorskip("jax")

    import sys

    import vmec_jax.booz_input as booz_input
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_state

    calls = {}
    fake_booz_module = ModuleType("booz_xform_jax")

    def fake_prepare_booz_xform_constants_from_inputs(*, inputs, mboz, nboz, asym):
        calls["prepare"] = {"inputs": inputs, "mboz": mboz, "nboz": nboz, "asym": asym}
        return "constants", "grids"

    def fake_booz_xform_from_inputs(*, inputs, constants, grids, surface_indices, jit):
        calls["booz"] = {
            "inputs": inputs,
            "constants": constants,
            "grids": grids,
            "surface_indices": np.asarray(surface_indices),
            "jit": jit,
        }
        return {
            "bmnc_b": np.asarray([[1.0, 0.08], [1.05, 0.04]], dtype=float),
            "ixm_b": np.asarray([0, 0], dtype=float),
            "ixn_b": np.asarray([0, 2], dtype=float),
            "iota_b": np.asarray([0.41, 0.47], dtype=float),
            "nfp_b": np.asarray(2),
        }

    fake_booz_module.prepare_booz_xform_constants_from_inputs = fake_prepare_booz_xform_constants_from_inputs
    fake_booz_module.booz_xform_from_inputs = fake_booz_xform_from_inputs
    monkeypatch.setitem(sys.modules, "booz_xform_jax", fake_booz_module)
    monkeypatch.setattr(
        booz_input,
        "booz_xform_inputs_from_state",
        lambda **_kwargs: SimpleNamespace(rmnc=np.zeros((4, 2)), nfp=2),
    )

    out = quasi_isodynamic_residual_from_state(
        state="state",
        static=SimpleNamespace(cfg=SimpleNamespace(lasym=True)),
        indata="indata",
        signgs=-1,
        surfaces=[0.25, 0.75],
        weights=[1.0, 2.0],
        mboz=5,
        nboz=6,
        nphi=21,
        nalpha=5,
        n_bounce=4,
        branch_width_weight=0.0,
        shuffle_profile_weight=0.0,
        jit_booz=True,
    )

    assert calls["prepare"]["mboz"] == 5
    assert calls["prepare"]["nboz"] == 6
    assert calls["prepare"]["asym"] is True
    assert calls["booz"]["constants"] == "constants"
    assert calls["booz"]["grids"] == "grids"
    assert calls["booz"]["jit"] is True
    np.testing.assert_array_equal(calls["booz"]["surface_indices"], [0, 2])
    np.testing.assert_allclose(np.asarray(out["surfaces"]), [0.25, 0.75])
    np.testing.assert_array_equal(np.asarray(out["surface_indices"]), [0, 2])
    assert np.isfinite(float(np.asarray(out["total"])))


def test_qi_residual_from_state_accepts_precomputed_boozer_grid_and_scalar_surface(monkeypatch):
    pytest.importorskip("jax")

    import sys

    import vmec_jax.booz_input as booz_input
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_state

    calls = {}
    fake_booz_module = ModuleType("booz_xform_jax")

    def fail_prepare_booz_xform_constants_from_inputs(**_kwargs):
        raise AssertionError("precomputed Boozer constants should be reused")

    def fake_booz_xform_from_inputs(*, inputs, constants, grids, surface_indices, jit):
        calls["booz"] = {
            "inputs": inputs,
            "constants": constants,
            "grids": grids,
            "surface_indices": np.asarray(surface_indices),
            "jit": jit,
        }
        return {
            "bmnc_b": np.asarray([[1.0, 0.03]], dtype=float),
            "ixm_b": np.asarray([0, 0], dtype=float),
            "ixn_b": np.asarray([0, 1], dtype=float),
            "iota_b": np.asarray([0.42], dtype=float),
            "nfp_b": np.asarray(3),
        }

    fake_booz_module.prepare_booz_xform_constants_from_inputs = fail_prepare_booz_xform_constants_from_inputs
    fake_booz_module.booz_xform_from_inputs = fake_booz_xform_from_inputs
    monkeypatch.setitem(sys.modules, "booz_xform_jax", fake_booz_module)
    monkeypatch.setattr(
        booz_input,
        "booz_xform_inputs_from_state",
        lambda **_kwargs: SimpleNamespace(rmnc=np.zeros((5, 2)), nfp=3),
    )

    out = quasi_isodynamic_residual_from_state(
        state="state",
        static=SimpleNamespace(cfg=SimpleNamespace(lasym=False)),
        indata="indata",
        signgs=1,
        surfaces=0.6,
        weights=[2.0],
        nphi=17,
        nalpha=5,
        n_bounce=4,
        branch_width_weight=0.0,
        shuffle_profile_weight=0.0,
        booz_constants="precomputed-constants",
        booz_grids="precomputed-grids",
        surface_indices=np.asarray([3], dtype=np.int32),
        pressure_local=np.asarray([0.0, 1.0]),
    )

    assert calls["booz"]["constants"] == "precomputed-constants"
    assert calls["booz"]["grids"] == "precomputed-grids"
    assert calls["booz"]["jit"] is False
    np.testing.assert_array_equal(calls["booz"]["surface_indices"], [3])
    np.testing.assert_allclose(np.asarray(out["surfaces"]), [0.6])
    np.testing.assert_array_equal(np.asarray(out["surface_indices"]), [3])
    assert np.isfinite(float(np.asarray(out["total"])))


@pytest.mark.py311_slow_coverage
def test_qi_boozer_mode_residual_can_include_legacy_bounce_endpoints():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_boozer_modes

    kwargs = dict(
        bmnc_b=jnp.asarray([[1.0, 0.1]]),
        xm_b=jnp.asarray([0, 0]),
        xn_b=jnp.asarray([0, 1]),
        iota_b=jnp.asarray([0.4]),
        nfp=1,
        nphi=33,
        nalpha=9,
        n_bounce=7,
        shuffle_profile_weight=0.0,
    )
    default = quasi_isodynamic_residual_from_boozer_modes(**kwargs)
    endpoint = quasi_isodynamic_residual_from_boozer_modes(**kwargs, include_bounce_endpoints=True)

    assert float(np.asarray(default["levels"][0])) > 0.0
    np.testing.assert_allclose(np.asarray(endpoint["levels"])[[0, -1]], [0.0, 1.0])
    assert endpoint["include_bounce_endpoints"] is True
    assert np.asarray(endpoint["width_residuals1d"]).shape == np.asarray(default["width_residuals1d"]).shape
    assert np.isfinite(float(np.asarray(endpoint["total"])))

    shuffled = quasi_isodynamic_residual_from_boozer_modes(
        **{**kwargs, "shuffle_profile_weight": 1.0},
        include_bounce_endpoints=True,
    )
    assert np.isfinite(float(np.asarray(shuffled["total"])))
    assert np.asarray(shuffled["shuffle_profile_residuals1d"]).shape == (9 * 33,)

    dense = quasi_isodynamic_residual_from_boozer_modes(
        **{**kwargs, "shuffle_profile_weight": 1.0},
        include_bounce_endpoints=True,
        shuffle_profile_nphi_out=49,
    )
    assert np.isfinite(float(np.asarray(dense["total"])))
    assert np.asarray(dense["shuffle_profile_residuals1d"]).shape == (9 * 49,)
    assert np.asarray(dense["shuffle_profile"]).shape == (1, 9, 49)
    assert dense["shuffle_profile_nphi_out"] == 49


@pytest.mark.py311_slow_coverage
def test_qi_weighted_shuffle_profile_residual_is_finite_and_differentiable():
    pytest.importorskip("jax")

    import jax

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_boozer_modes

    xm_b = jnp.asarray([0, 0, 1, 2])
    xn_b = jnp.asarray([0, 1, 1, 2])

    def objective(coeffs):
        out = quasi_isodynamic_residual_from_boozer_modes(
            bmnc_b=coeffs[None, :],
            xm_b=xm_b,
            xn_b=xn_b,
            iota_b=jnp.asarray([0.47]),
            nfp=2,
            nphi=33,
            nalpha=9,
            n_bounce=7,
            width_weight=0.0,
            branch_width_weight=0.0,
            profile_weight=0.0,
            shuffle_profile_weight=0.0,
            weighted_shuffle_profile_weight=1.0,
            weighted_shuffle_profile_softness=2.0e-2,
        )
        return out["total"]

    value, grad = jax.value_and_grad(objective)(jnp.asarray([1.0, 0.09, 0.04, -0.02]))

    assert np.isfinite(float(np.asarray(value)))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.linalg.norm(np.asarray(grad[1:])) > 0.0


def test_qi_aligned_profile_branch_is_finite_and_differentiable():
    pytest.importorskip("jax")

    import jax

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_boozer_modes

    xm_b = jnp.asarray([0, 0, 1, 2])
    xn_b = jnp.asarray([0, 1, 1, 2])

    def objective(coeffs):
        out = quasi_isodynamic_residual_from_boozer_modes(
            bmnc_b=coeffs[None, :],
            xm_b=xm_b,
            xn_b=xn_b,
            iota_b=jnp.asarray([0.47]),
            nfp=2,
            nphi=33,
            nalpha=9,
            n_bounce=7,
            width_weight=0.0,
            branch_width_weight=0.0,
            profile_weight=0.0,
            aligned_profile_weight=1.0,
            aligned_profile_softness=2.0e-2,
            aligned_profile_trap_level=0.7,
            aligned_profile_trap_softness=5.0e-2,
            shuffle_profile_weight=0.0,
            weighted_shuffle_profile_weight=0.0,
        )
        return out["total"], out

    (value, out), grad = jax.value_and_grad(objective, has_aux=True)(jnp.asarray([1.0, 0.09, 0.04, -0.02]))

    assert np.isfinite(float(np.asarray(value)))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.linalg.norm(np.asarray(grad[1:])) > 0.0
    assert np.asarray(out["aligned_profile_residuals1d"]).shape == (32 * 9,)
    assert np.asarray(out["aligned_profile"]).shape == (1, 32, 9)
    assert np.all(np.isfinite(np.asarray(out["aligned_min_phi"])))
    assert np.all((np.asarray(out["aligned_profile_trap_weight"]) >= 0.0))


def test_qi_mirror_ratio_smooth_penalty_branch_is_differentiable():
    pytest.importorskip("jax")

    import jax

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import mirror_ratio_penalty_from_boozer_modes

    xm_b = jnp.asarray([0, 0, 1])
    xn_b = jnp.asarray([0, 1, 1])

    def objective(coeffs):
        out = mirror_ratio_penalty_from_boozer_modes(
            bmnc_b=coeffs[None, :],
            xm_b=xm_b,
            xn_b=xn_b,
            nfp=2,
            threshold=0.04,
            ntheta=24,
            nphi=24,
            smooth_extrema=2.0e-2,
            smooth_penalty=1.0e-2,
        )
        return out["total"]

    value, grad = jax.value_and_grad(objective)(jnp.asarray([1.0, 0.1, 0.03]))

    assert np.isfinite(float(np.asarray(value)))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(np.asarray(value)) > 0.0


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


def test_qi_branch_width_residual_rejects_misaligned_wells():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_boozer_modes

    qi_like = quasi_isodynamic_residual_from_boozer_modes(
        bmnc_b=jnp.asarray([[1.0, 0.1]]),
        xm_b=jnp.asarray([0, 0]),
        xn_b=jnp.asarray([0, 1]),
        iota_b=jnp.asarray([0.4]),
        nfp=1,
        nphi=33,
        nalpha=9,
        n_bounce=7,
        width_weight=0.0,
        branch_width_weight=1.0,
        profile_weight=0.0,
        shuffle_profile_weight=0.0,
    )
    qh_like = quasi_isodynamic_residual_from_boozer_modes(
        bmnc_b=jnp.asarray([[1.0, 0.1]]),
        xm_b=jnp.asarray([0, 1]),
        xn_b=jnp.asarray([0, 1]),
        iota_b=jnp.asarray([0.4]),
        nfp=1,
        nphi=33,
        nalpha=9,
        n_bounce=7,
        width_weight=0.0,
        branch_width_weight=1.0,
        profile_weight=0.0,
        shuffle_profile_weight=0.0,
    )

    np.testing.assert_allclose(np.asarray(qi_like["total"]), 0.0, atol=1.0e-28, rtol=0.0)
    assert float(np.asarray(qh_like["total"])) > 1.0e-4
    assert np.asarray(qh_like["branch_width_residuals1d"]).shape == (9 * 7,)


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


def test_qi_boozer_mode_residual_has_finite_saturated_sigmoid_adjoint():
    pytest.importorskip("jax")

    import jax

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_boozer_modes

    xm_b = jnp.asarray([0, 1, 2, 1])
    xn_b = jnp.asarray([0, 0, 1, 2])

    def objective(coeffs):
        out = quasi_isodynamic_residual_from_boozer_modes(
            bmnc_b=coeffs[None, :],
            xm_b=xm_b,
            xn_b=xn_b,
            iota_b=jnp.asarray([0.43]),
            nfp=2,
            nphi=33,
            nalpha=9,
            n_bounce=7,
            softness=1.0e-6,
            branch_width_softness=1.0e-6,
            shuffle_profile_softness=1.0e-6,
            profile_weight=0.1,
            shuffle_profile_weight=1.0,
        )
        return out["total"]

    value, grad = jax.value_and_grad(objective)(jnp.asarray([1.0, 0.3, -0.2, 0.15]))

    assert np.isfinite(np.asarray(value))
    assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.py311_slow_coverage
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
        width_weight=0.7,
        branch_width_weight=0.2,
        branch_width_softness=2.0e-2,
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
        width_weight=kwargs["width_weight"],
        branch_width_weight=kwargs["branch_width_weight"],
        branch_width_softness=kwargs["branch_width_softness"],
        profile_weight=kwargs["profile_weight"],
    )

    np.testing.assert_allclose(np.asarray(direct["residuals1d"]), np.asarray(wrapped["residuals1d"]))
    np.testing.assert_allclose(np.asarray(wrapped["total"]), 0.53924353395480018, rtol=1.0e-12, atol=1.0e-14)
    assert np.asarray(wrapped["width_residuals1d"]).shape == (2 * 7 * 5,)
    assert np.asarray(wrapped["profile_residuals1d"]).shape == (2 * 21 * 7,)
    assert np.asarray(wrapped["shuffle_profile_residuals1d"]).shape == (2 * 21 * 7,)


def test_qi_mirror_ratio_penalty_from_boozer_modes():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import mirror_ratio_penalty_from_boozer_modes

    flat = mirror_ratio_penalty_from_boozer_modes(
        bmnc_b=jnp.asarray([[1.0]]),
        xm_b=jnp.asarray([0]),
        xn_b=jnp.asarray([0]),
        nfp=1,
        threshold=0.05,
        ntheta=16,
        nphi=16,
    )
    np.testing.assert_allclose(np.asarray(flat["mirror_ratio"]), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(flat["residuals1d"]), 0.0, atol=1.0e-14)

    rippled = mirror_ratio_penalty_from_boozer_modes(
        bmnc_b=jnp.asarray([[1.0, 0.2]]),
        xm_b=jnp.asarray([0, 0]),
        xn_b=jnp.asarray([0, 1]),
        nfp=1,
        threshold=0.05,
        ntheta=16,
        nphi=64,
    )
    assert float(np.asarray(rippled["mirror_ratio"][0])) > 0.19
    assert float(np.asarray(rippled["total"])) > 0.0


def test_qi_boundary_elongation_penalty_is_differentiable():
    pytest.importorskip("jax")

    import jax

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import boundary_max_elongation_from_rz

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 64, endpoint=False)
    phi = jnp.linspace(0.0, 0.5 * jnp.pi, 5, endpoint=False)

    def max_elongation(vertical_scale):
        R = 1.0 + 0.1 * jnp.cos(theta)[:, None] * jnp.ones_like(phi)[None, :]
        Z = vertical_scale * 0.1 * jnp.sin(theta)[:, None] * jnp.ones_like(phi)[None, :]
        return boundary_max_elongation_from_rz(R, Z, phi=phi)["max_elongation"]

    value, grad = jax.value_and_grad(max_elongation)(jnp.asarray(3.0))
    np.testing.assert_allclose(np.asarray(value), 3.0, rtol=3.0e-2, atol=3.0e-2)
    assert float(np.asarray(grad)) > 0.5


def test_qi_boozer_mode_residual_validates_shapes_and_resolution():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_boozer_modes
    from vmec_jax.quasi_isodynamic.objectives import _nearest_half_mesh_indices

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


@pytest.mark.py311_slow_coverage
def test_qi_state_residual_smoke(load_case_qh_warm_start):
    pytest.importorskip("jax")
    pytest.importorskip("booz_xform_jax")

    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_state

    _cfg, indata, static, _boundary, state = load_case_qh_warm_start
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state, static).sqrtg), axis_index=1))
    kwargs = dict(
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
    )
    out = quasi_isodynamic_residual_from_state(**kwargs, jit_booz=False)
    out_jit = quasi_isodynamic_residual_from_state(**kwargs, jit_booz=True)

    assert np.asarray(out["width_residuals1d"]).shape == (7 * 5,)
    assert np.asarray(out["profile_residuals1d"]).shape == (17 * 7,)
    assert np.asarray(out["shuffle_profile_residuals1d"]).shape == (17 * 7,)
    assert np.asarray(out["residuals1d"]).shape == (2 * 7 * 5 + 2 * 17 * 7,)
    assert np.isfinite(np.asarray(out["total"]))
    np.testing.assert_allclose(np.asarray(out_jit["total"]), np.asarray(out["total"]), rtol=1e-10, atol=1e-12)


def test_qi_lgradb_penalty_from_state_smoke(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.quasi_isodynamic import lgradb_penalty_from_state

    _cfg, indata, static, _boundary, state = load_case_qh_warm_start
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state, static).sqrtg), axis_index=1))
    out = lgradb_penalty_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        threshold=0.30,
        ntheta=5,
        nphi=5,
    )

    assert np.asarray(out["residuals1d"]).shape == (25,)
    assert np.asarray(out["L_grad_B"]).shape == (5, 5)
    assert np.all(np.isfinite(np.asarray(out["L_grad_B"])))
    assert float(np.asarray(out["min_L_grad_B"])) > 0.0

    loose = lgradb_penalty_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        threshold=1.0e-8,
        ntheta=5,
        nphi=5,
    )
    np.testing.assert_allclose(np.asarray(loose["total"]), 0.0, atol=1.0e-14)


def test_qi_lgradb_penalty_from_state_is_differentiable(load_case_qh_warm_start):
    pytest.importorskip("jax")

    import jax

    from vmec_jax._compat import jnp
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.quasi_isodynamic import lgradb_penalty_from_state

    _cfg, indata, static, _boundary, state = load_case_qh_warm_start
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state, static).sqrtg), axis_index=1))
    Rcos0 = jnp.asarray(state.Rcos, dtype=jnp.float64)

    def objective(scale):
        trial_state = replace(state, Rcos=Rcos0 * scale)
        out = lgradb_penalty_from_state(
            state=trial_state,
            static=static,
            indata=indata,
            signgs=signgs,
            threshold=0.30,
            ntheta=5,
            nphi=5,
        )
        return out["total"]

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0, dtype=jnp.float64))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
