from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax import finite_beta
from vmec_jax.finite_beta import FiniteBetaTargets


def test_s_half_from_static_matches_vmec_half_mesh_convention():
    static = SimpleNamespace(s=jnp.asarray([0.0, 0.25, 1.0]))
    np.testing.assert_allclose(np.asarray(finite_beta._s_half_from_static(static)), [0.0, 0.125, 0.625])

    single = SimpleNamespace(s=jnp.asarray([0.0]))
    np.testing.assert_allclose(np.asarray(finite_beta._s_half_from_static(single)), [0.0])


def test_finite_beta_global_residuals_apply_one_sided_constraints(monkeypatch):
    def _fake_scalars_from_state(**_kwargs):
        return {
            "aspect": jnp.asarray(7.0),
            "min_iota": jnp.asarray(0.30),
            "mean_iota": jnp.asarray(0.35),
            "max_iota": jnp.asarray(0.80),
            "volavgB": jnp.asarray(2.50),
            "betatotal": jnp.asarray(0.04),
        }

    monkeypatch.setattr(finite_beta, "finite_beta_scalars_from_state", _fake_scalars_from_state)
    targets = FiniteBetaTargets(
        aspect_ratio=6.0,
        min_iota=0.41,
        min_average_iota=0.45,
        max_iota=0.70,
        volavgB=2.0,
        beta_total=0.05,
        aspect_weight=2.0,
        iota_weight=3.0,
        max_iota_weight=4.0,
        volavgB_weight=5.0,
        beta_weight=6.0,
    )

    residuals = finite_beta.finite_beta_global_residuals_from_state(
        state=None,
        static=None,
        indata=None,
        signgs=1,
        targets=targets,
    )

    np.testing.assert_allclose(
        np.asarray(residuals),
        [2.0, -0.33, -0.30, 0.40, 2.50, -0.06],
        rtol=1e-12,
        atol=1e-12,
    )


def test_finite_beta_global_residuals_are_zero_for_satisfied_one_sided_constraints(monkeypatch):
    def _fake_scalars_from_state(**_kwargs):
        return {
            "aspect": jnp.asarray(5.5),
            "min_iota": jnp.asarray(0.42),
            "mean_iota": jnp.asarray(0.46),
            "max_iota": jnp.asarray(0.65),
            "volavgB": jnp.asarray(2.0),
            "betatotal": jnp.asarray(0.05),
        }

    monkeypatch.setattr(finite_beta, "finite_beta_scalars_from_state", _fake_scalars_from_state)
    targets = FiniteBetaTargets(
        aspect_ratio=6.0,
        min_iota=0.41,
        min_average_iota=0.45,
        max_iota=0.70,
        volavgB=2.0,
        beta_total=0.05,
    )

    residuals = finite_beta.finite_beta_global_residuals_from_state(
        state=None,
        static=None,
        indata=None,
        signgs=1,
        targets=targets,
    )

    np.testing.assert_allclose(np.asarray(residuals), np.zeros(6))
