from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.free_boundary_validation as validation
from vmec_jax.free_boundary_validation import (
    free_boundary_response_metrics,
    virtual_casing_finite_beta_boundary_diagnostics,
    wout_beta_percent,
    wout_fsq_total,
    wout_mean_iota,
)


@pytest.mark.py311_coverage_only
def test_free_boundary_response_metrics_with_synthetic_wout_like_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise WOUT-native response metrics without optional fixture assets."""

    def fake_read_wout(path: str | Path) -> SimpleNamespace:
        if Path(path).name == "reference.nc":
            return reference
        if Path(path).name == "candidate.nc":
            return candidate
        raise AssertionError(f"unexpected synthetic WOUT path: {path}")

    def fake_surface_rz_from_wout_physical(
        wout: SimpleNamespace,
        *,
        theta: np.ndarray,
        phi: np.ndarray,
        s_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert s_index == wout.ns - 1
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        R = wout.major_radius + wout.shift_R + 0.1 * np.cos(theta_grid) + 0.01 * np.cos(phi_grid)
        Z = wout.shift_Z + 0.2 * np.sin(theta_grid)
        return R, Z

    def fake_bmag_from_wout_physical(
        wout: SimpleNamespace,
        *,
        theta: np.ndarray,
        phi: np.ndarray,
        s_index: int,
    ) -> np.ndarray:
        assert s_index == wout.ns - 1
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        return wout.b0 + 0.01 * np.cos(theta_grid) + 0.02 * np.sin(phi_grid)

    reference = SimpleNamespace(
        nfp=2,
        ns=4,
        betatotal=0.01,
        aspect=5.0,
        iotaf=np.asarray([0.0, 0.2, 0.3, 0.4]),
        raxis_cc=np.asarray([1.0]),
        zaxis_cs=np.asarray([0.0]),
        major_radius=1.0,
        shift_R=0.0,
        shift_Z=0.0,
        b0=1.0,
    )
    candidate = SimpleNamespace(
        nfp=2,
        ns=4,
        beta_total=0.015,
        aspect=5.2,
        iotas=np.asarray([0.0, 0.25, 0.35, 0.45]),
        raxis_cc=np.asarray([1.05]),
        zaxis_cs=np.asarray([-0.02]),
        major_radius=1.0,
        shift_R=0.03,
        shift_Z=-0.04,
        b0=1.1,
    )

    monkeypatch.setattr(validation, "read_wout", fake_read_wout)
    monkeypatch.setattr(validation, "surface_rz_from_wout_physical", fake_surface_rz_from_wout_physical)
    monkeypatch.setattr(validation, "bmag_from_wout_physical", fake_bmag_from_wout_physical)

    metrics = free_boundary_response_metrics("reference.nc", Path("candidate.nc"), ntheta=8, nphi=4)
    metrics_dict = metrics.to_dict()

    assert metrics.reference_beta_percent == pytest.approx(1.0)
    assert metrics.candidate_beta_percent == pytest.approx(1.5)
    assert metrics.beta_delta_percent == pytest.approx(0.5)
    assert metrics.reference_aspect == pytest.approx(5.0)
    assert metrics.candidate_aspect == pytest.approx(5.2)
    assert metrics.aspect_delta == pytest.approx(0.2)
    assert metrics.reference_mean_iota == pytest.approx(0.3)
    assert metrics.candidate_mean_iota == pytest.approx(0.35)
    assert metrics.mean_iota_delta == pytest.approx(0.05)
    assert metrics.lcfs_rms_displacement == pytest.approx(0.05)
    assert metrics.lcfs_max_displacement == pytest.approx(0.05)
    assert metrics.lcfs_max_abs_dR == pytest.approx(0.03)
    assert metrics.lcfs_max_abs_dZ == pytest.approx(0.04)
    assert metrics.lcfs_b_rel_rms_delta > 0.09
    assert metrics.axis_R_shift == pytest.approx(0.05)
    assert metrics.axis_Z_shift == pytest.approx(-0.02)
    assert set(metrics_dict) == {
        "reference_beta_percent",
        "candidate_beta_percent",
        "beta_delta_percent",
        "reference_aspect",
        "candidate_aspect",
        "aspect_delta",
        "reference_mean_iota",
        "candidate_mean_iota",
        "mean_iota_delta",
        "lcfs_rms_displacement",
        "lcfs_max_displacement",
        "lcfs_max_abs_dR",
        "lcfs_max_abs_dZ",
        "lcfs_b_rel_rms_delta",
        "axis_R_shift",
        "axis_Z_shift",
    }


@pytest.mark.py311_coverage_only
def test_free_boundary_response_scalar_helpers_and_nfp_guard_without_assets() -> None:
    """Cover backend-neutral scalar helpers and the period-count compatibility guard."""

    reference = SimpleNamespace(
        nfp=2,
        betatotal=0.012,
        iotaf=np.asarray([0.0, 0.2, 0.4]),
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
    )
    fallback = SimpleNamespace(nfp=2, beta_total=0.034, iotas=np.asarray([0.0, -0.5]), fsqr=0.5, fsqz=0.25, fsql=0.125)
    no_profiles = SimpleNamespace(nfp=2)
    wrong_period = SimpleNamespace(nfp=3)

    assert wout_beta_percent(reference) == pytest.approx(1.2)
    assert wout_beta_percent(fallback) == pytest.approx(3.4)
    assert np.isnan(wout_beta_percent(no_profiles))
    assert wout_mean_iota(reference) == pytest.approx(0.3)
    assert wout_mean_iota(fallback) == pytest.approx(-0.5)
    assert np.isnan(wout_mean_iota(no_profiles))
    assert wout_fsq_total(reference) == pytest.approx(6.0)
    assert wout_fsq_total(fallback) == pytest.approx(0.875)
    assert np.isnan(wout_fsq_total(no_profiles))

    with pytest.raises(ValueError, match="nfp mismatch"):
        free_boundary_response_metrics(reference, wrong_period, ntheta=8, nphi=4)


@pytest.mark.py311_coverage_only
def test_virtual_casing_boundary_diagnostics_with_fake_functional_module() -> None:
    """Cover finite-beta postsolve metrics without importing virtual_casing_jax."""

    class FakeVirtualCasingFunctional:
        @staticmethod
        def prepare_functional_setup(
            _x,
            *,
            digits,
            nfp,
            half_period,
            surf_nt,
            surf_np,
            src_nt,
            src_np,
            trg_nt,
            trg_np,
            quad_nt,
            quad_np,
            patch_dim0,
        ):
            return SimpleNamespace(
                nfp=nfp,
                half_period=half_period,
                surf_nt=surf_nt,
                surf_np=surf_np,
                src_nt=src_nt,
                src_np=src_np,
                trg_nt=trg_nt,
                trg_np=trg_np,
                quad_nt=quad_nt,
                quad_np=quad_np,
                patch_dim0=3 if patch_dim0 is None else patch_dim0,
                patch_idx=np.asarray([0]),
                orient=1.0,
            )

        @staticmethod
        def compute_external_B_functional(x, b_total, **_kwargs):
            del x
            required = np.asarray(b_total, dtype=float).copy()
            required[0] -= 0.25
            return required

        @staticmethod
        def target_surface_normal(x, **_kwargs):
            normal = np.zeros_like(np.asarray(x, dtype=float))
            normal[0] = 1.0
            return normal

    theta = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    phi = np.linspace(0.0, np.pi, 3, endpoint=False)
    surface = np.zeros((3, theta.size, phi.size))
    surface[0] = 1.0 + 0.1 * np.cos(theta)[:, None]
    surface[1] = 0.1 * np.sin(theta)[:, None]
    surface[2] = phi[None, :]
    total_b = np.zeros_like(surface)
    total_b[0] = 2.0
    target_external = total_b.copy()
    target_external[0] -= 0.15

    diagnostics = virtual_casing_finite_beta_boundary_diagnostics(
        surface,
        total_b,
        target_external_b=target_external,
        pressure=0.02,
        mu0=1.0,
        nfp=2,
        digits=4,
        quad_nt=8,
        quad_np=6,
        vc_module=FakeVirtualCasingFunctional,
    )

    assert diagnostics.external_bnormal_residual_rms == pytest.approx(0.10)
    assert diagnostics.external_bnormal_residual_max == pytest.approx(0.10)
    expected_pressure_balance = 0.02 + (2.0**2 - 1.85**2) / 2.0
    assert diagnostics.pressure_balance_rms == pytest.approx(expected_pressure_balance)
    assert diagnostics.pressure_balance_max == pytest.approx(expected_pressure_balance)
    assert diagnostics.to_dict() == {
        "external_bnormal_residual_rms": pytest.approx(0.10),
        "external_bnormal_residual_max": pytest.approx(0.10),
        "pressure_balance_rms": pytest.approx(expected_pressure_balance),
        "pressure_balance_max": pytest.approx(expected_pressure_balance),
    }


@pytest.mark.py311_coverage_only
def test_virtual_casing_boundary_diagnostics_validates_inputs() -> None:
    class MinimalVirtualCasingFunctional:
        @staticmethod
        def prepare_functional_setup(*_args, **_kwargs):  # pragma: no cover - validation exits first
            raise AssertionError("unexpected virtual-casing call")

    surface = np.zeros((3, 2, 2))
    total_b = np.ones_like(surface)

    with pytest.raises(ValueError, match="shape"):
        virtual_casing_finite_beta_boundary_diagnostics(
            np.zeros((2, 2)),
            total_b,
            vc_module=MinimalVirtualCasingFunctional,
        )
    with pytest.raises(ValueError, match="does not match"):
        virtual_casing_finite_beta_boundary_diagnostics(
            surface,
            total_b[:, :, :1],
            vc_module=MinimalVirtualCasingFunctional,
        )
    with pytest.raises(ValueError, match="pressure"):
        virtual_casing_finite_beta_boundary_diagnostics(
            surface,
            total_b,
            pressure=np.zeros((3, 3)),
            vc_module=MinimalVirtualCasingFunctional,
        )
    with pytest.raises(ValueError, match="mu0"):
        virtual_casing_finite_beta_boundary_diagnostics(
            surface,
            total_b,
            mu0=0.0,
            vc_module=MinimalVirtualCasingFunctional,
        )
    with pytest.raises(ValueError, match="quad_nt"):
        virtual_casing_finite_beta_boundary_diagnostics(
            surface,
            total_b,
            quad_nt=1,
            vc_module=MinimalVirtualCasingFunctional,
        )
