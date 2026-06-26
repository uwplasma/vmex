from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.solvers.free_boundary.validation as validation
from vmec_jax.solvers.free_boundary.validation import (
    free_boundary_response_metrics,
    free_boundary_promotion_status,
    virtual_casing_finite_beta_boundary_diagnostics,
    virtual_casing_grid_adequacy_status,
    wout_beta_percent,
    wout_fsq_total,
    wout_mean_iota,
)


def test_virtual_casing_loader_uses_source_path_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Finite-beta diagnostics can use an optional source checkout without installation."""

    package = tmp_path / "virtual_casing_jax"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "functional.py").write_text("SENTINEL = 'loaded-from-env-path'\n")
    monkeypatch.setenv("VMEC_JAX_VIRTUAL_CASING_JAX_PATH", str(tmp_path))
    monkeypatch.delitem(sys.modules, "virtual_casing_jax", raising=False)
    monkeypatch.delitem(sys.modules, "virtual_casing_jax.functional", raising=False)

    module = validation._load_virtual_casing_functional()

    assert module.SENTINEL == "loaded-from-env-path"


def test_virtual_casing_boundary_diagnostics_reports_grid_metadata() -> None:
    """Finite-beta virtual-casing diagnostics record the grid used for promotion."""

    class FakeVirtualCasingFunctional:
        @staticmethod
        def prepare_functional_setup(
            _x: np.ndarray,
            *,
            digits: int,
            nfp: int,
            half_period: bool,
            surf_nt: int,
            surf_np: int,
            src_nt: int,
            src_np: int,
            trg_nt: int,
            trg_np: int,
            quad_nt: int,
            quad_np: int,
            patch_dim0: int | None,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                digits=digits,
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
                patch_dim0=patch_dim0,
                patch_idx=np.arange(quad_nt * quad_np),
                orient=1.0,
            )

        @staticmethod
        def compute_external_B_functional(x: np.ndarray, _b_total: np.ndarray, **_kwargs: object) -> np.ndarray:
            return np.zeros_like(x)

        @staticmethod
        def target_surface_normal(x: np.ndarray, **_kwargs: object) -> np.ndarray:
            normal = np.zeros_like(x)
            normal[0, :, :] = 1.0
            return normal

    surface = np.zeros((3, 4, 3))
    total_b = np.zeros_like(surface)
    diagnostics = virtual_casing_finite_beta_boundary_diagnostics(
        surface,
        total_b,
        target_external_b=np.zeros_like(surface),
        quad_nt=8,
        quad_np=6,
        chunk_size=5,
        target_chunk_size="auto",
        vc_module=FakeVirtualCasingFunctional,
    )
    payload = diagnostics.to_dict()

    assert diagnostics.grid_adequacy_status == "production_ready"
    assert diagnostics.surface_ntheta == 4
    assert diagnostics.surface_nphi == 3
    assert diagnostics.quad_ntheta == 8
    assert diagnostics.quad_nphi == 6
    assert diagnostics.quad_factor_theta == pytest.approx(2.0)
    assert diagnostics.quad_factor_phi == pytest.approx(2.0)
    assert payload["grid_adequacy_status"] == "production_ready"
    assert payload["chunk_size"] == 5
    assert payload["target_chunk_size"] == "auto"


def test_virtual_casing_grid_adequacy_controls_finite_beta_promotion() -> None:
    """Finite-beta direct-coil promotion needs a production-sized virtual-casing grid."""

    assert (
        virtual_casing_grid_adequacy_status(
            surface_ntheta=16,
            surface_nphi=8,
            quad_ntheta=32,
            quad_nphi=16,
        )
        == "production_ready"
    )
    underresolved = virtual_casing_grid_adequacy_status(
        surface_ntheta=16,
        surface_nphi=8,
        quad_ntheta=16,
        quad_nphi=16,
    )
    assert underresolved == "diagnostic_only_low_quad_theta"

    promotion = free_boundary_promotion_status(
        beta_percent=1.0,
        strict_components_met=True,
        final_residual_recomputed=True,
        virtual_casing_status="computed",
        virtual_casing_grid_adequacy_status=underresolved,
        direct_coil_backend=True,
    )

    assert promotion["production_candidate"] is False
    assert promotion["coil_bnormal_role"] == "diagnostic_only"
    assert promotion["promotion_blockers"] == [f"virtual_casing_grid_{underresolved}"]


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
