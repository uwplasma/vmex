from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax import finite_beta
from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.finite_beta import FiniteBetaTargets
from vmec_jax.modes import ModeTable
from vmec_jax.namelist import InData


def test_s_half_from_static_matches_vmec_half_mesh_convention():
    static = SimpleNamespace(s=jnp.asarray([0.0, 0.25, 1.0]))
    np.testing.assert_allclose(np.asarray(finite_beta._s_half_from_static(static)), [0.0, 0.125, 0.625])

    single = SimpleNamespace(s=jnp.asarray([0.0]))
    np.testing.assert_allclose(np.asarray(finite_beta._s_half_from_static(single)), [0.0])


def test_wout_like_for_state_builds_profile_and_flux_fields(monkeypatch):
    modes = ModeTable(m=np.array([0, 1]), n=np.array([0, 0]))
    static = SimpleNamespace(
        s=jnp.asarray([0.0, 0.5, 1.0]),
        modes=modes,
        cfg=SimpleNamespace(nfp=2, mpol=2, ntor=0, lasym=False),
    )
    indata = InData(scalars={"NCURR": 1, "GAMMA": 0.0, "LRFP": False}, indexed={}, source_path=None)

    monkeypatch.setattr(
        finite_beta,
        "flux_profiles_from_indata",
        lambda *_args, **_kwargs: SimpleNamespace(
            phipf=jnp.asarray([4.0, 5.0, 6.0]),
            phips=jnp.asarray([9.0, 2.0, 3.0]),
            chipf=jnp.asarray([1.0, 2.0, 3.0]),
        ),
    )
    monkeypatch.setattr(
        finite_beta,
        "eval_profiles",
        lambda *_args, **_kwargs: {"pressure": jnp.asarray([7.0, 8.0, 9.0])},
    )
    monkeypatch.setattr(
        finite_beta,
        "equilibrium_iota_profiles_from_state",
        lambda **_kwargs: (
            jnp.asarray([0.0, 1.0, 3.0]),
            jnp.asarray([0.0, 0.1, 0.2]),
            jnp.asarray([0.0, 0.15, 0.25]),
        ),
    )
    monkeypatch.setattr(finite_beta, "_chipf_from_chips", lambda chips: jnp.asarray(chips) + 10.0)
    monkeypatch.setattr(
        "vmec_jax.boundary.boundary_from_indata",
        lambda *_args, **_kwargs: BoundaryCoeffs(
            R_cos=np.array([1.5, 0.0]),
            R_sin=np.zeros(2),
            Z_cos=np.zeros(2),
            Z_sin=np.zeros(2),
        ),
    )
    monkeypatch.setattr(
        finite_beta,
        "_mass_half_mesh_from_indata",
        lambda **_kwargs: jnp.asarray([0.0, 0.2, 0.3]),
    )
    monkeypatch.setattr(
        finite_beta,
        "_icurv_full_mesh_from_indata",
        lambda **_kwargs: jnp.asarray([0.0, 0.4, 0.5]),
    )

    wout_like, pres = finite_beta._wout_like_for_state(
        state=object(),
        static=static,
        indata=indata,
        signgs=-1,
    )

    np.testing.assert_allclose(np.asarray(wout_like.phips), [0.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(wout_like.phipf), [4.0, 5.0, 6.0])
    np.testing.assert_allclose(np.asarray(wout_like.chipf), [10.0, 11.0, 13.0])
    np.testing.assert_allclose(np.asarray(wout_like.mass), [0.0, 0.2, 0.3])
    np.testing.assert_allclose(np.asarray(wout_like.icurv), [0.0, 0.4, 0.5])
    np.testing.assert_allclose(np.asarray(pres), [0.0, 8.0, 9.0])
    assert wout_like.signgs == -1
    assert wout_like.nfp == 2
    assert wout_like.lcurrent


def test_finite_beta_scalars_from_state_uses_iota_and_energy_diagnostics(monkeypatch):
    static = SimpleNamespace(s=jnp.asarray([0.0, 0.5, 1.0]), trig_vmec=object())

    monkeypatch.setattr(
        finite_beta,
        "equilibrium_aspect_ratio_from_state",
        lambda **_kwargs: jnp.asarray(6.25),
    )
    monkeypatch.setattr(
        finite_beta,
        "equilibrium_iota_profiles_from_state",
        lambda **_kwargs: (
            jnp.asarray([0.0, 1.0, 3.0]),
            jnp.asarray([0.0, 0.2, -0.4]),
            jnp.asarray([0.0, 0.3, -0.5]),
        ),
    )
    monkeypatch.setattr(
        finite_beta,
        "_wout_like_for_state",
        lambda **_kwargs: (SimpleNamespace(), jnp.asarray([0.0, 1.0, 2.0])),
    )
    monkeypatch.setattr(
        finite_beta,
        "vmec_bcovar_half_mesh_from_wout",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        finite_beta,
        "vmec_force_norms_from_bcovar_dynamic",
        lambda **_kwargs: SimpleNamespace(
            wb=jnp.asarray(8.0),
            wp=jnp.asarray(2.0),
            volume=jnp.asarray(4.0),
        ),
    )

    scalars = finite_beta.finite_beta_scalars_from_state(
        state=object(),
        static=static,
        indata=object(),
        signgs=1,
    )

    assert float(scalars["aspect"]) == 6.25
    np.testing.assert_allclose(np.asarray(scalars["iotas"]), [0.0, 0.2, -0.4])
    np.testing.assert_allclose(np.asarray(scalars["iotaf"]), [0.0, 0.3, -0.5])
    assert float(scalars["mean_iota"]) == 0.4
    assert float(scalars["min_iota"]) == 0.3
    assert float(scalars["max_iota"]) == 0.5
    assert float(scalars["betatotal"]) == 0.25
    assert float(scalars["volavgB"]) == 2.0
    assert float(scalars["wb"]) == 8.0
    assert float(scalars["wp"]) == 2.0
    assert float(scalars["volume"]) == 4.0


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
