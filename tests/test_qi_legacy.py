from __future__ import annotations

import importlib.util
import numpy as np
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _booz_like(*, xm, xn, coeffs, iota=0.4, nfp=1):
    return {
        "bmnc_b": np.asarray([coeffs], dtype=float),
        "ixm_b": np.asarray(xm, dtype=float),
        "ixn_b": np.asarray(xn, dtype=float),
        "iota_b": np.asarray([iota], dtype=float),
        "nfp_b": np.asarray(nfp),
    }


def test_legacy_qi_branch_shuffle_diagnostic_ranks_qi_before_qh():
    from vmec_jax.qi_legacy import legacy_qi_branch_shuffle_diagnostic_from_boozer_output

    qi_like = _booz_like(xm=[0, 0], xn=[0, 1], coeffs=[1.0, 0.1])
    qh_like = _booz_like(xm=[0, 1], xn=[0, 1], coeffs=[1.0, 0.1])

    qi = legacy_qi_branch_shuffle_diagnostic_from_boozer_output(
        qi_like,
        nphi=33,
        nalpha=9,
        n_bounce=7,
        nphi_out=101,
        phimin=0.0,
    )
    qh = legacy_qi_branch_shuffle_diagnostic_from_boozer_output(
        qh_like,
        nphi=33,
        nalpha=9,
        n_bounce=7,
        nphi_out=101,
        phimin=0.0,
    )

    assert qi["residuals1d"].shape == (9 * 101,)
    assert qi["residual_size"] == 9 * 101
    assert qi["total"] < 1.0e-3
    assert qh["total"] > 100.0 * qi["total"]


def test_legacy_qi_branch_shuffle_diagnostic_supports_bmns_modes():
    from vmec_jax.qi_legacy import legacy_qi_branch_shuffle_diagnostic_from_boozer_output

    booz = _booz_like(xm=[0, 0, 1], xn=[0, 1, 1], coeffs=[1.0, 0.1, 0.02])
    booz["bmns_b"] = np.asarray([[0.0, 0.02, -0.01]], dtype=float)
    out = legacy_qi_branch_shuffle_diagnostic_from_boozer_output(
        booz,
        nphi=21,
        nalpha=5,
        n_bounce=5,
        nphi_out=41,
    )

    assert np.isfinite(out["total"])
    assert out["residual_size"] == 5 * 41


def test_smooth_qi_residual_preserves_legacy_synthetic_ranking():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.qi_legacy import legacy_qi_branch_shuffle_diagnostic_from_boozer_output
    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_boozer_modes

    qi_like = _booz_like(xm=[0, 0], xn=[0, 1], coeffs=[1.0, 0.1])
    qh_like = _booz_like(xm=[0, 1], xn=[0, 1], coeffs=[1.0, 0.1])
    legacy_qi = legacy_qi_branch_shuffle_diagnostic_from_boozer_output(
        qi_like,
        nphi=33,
        nalpha=9,
        n_bounce=7,
        nphi_out=101,
    )["total"]
    legacy_qh = legacy_qi_branch_shuffle_diagnostic_from_boozer_output(
        qh_like,
        nphi=33,
        nalpha=9,
        n_bounce=7,
        nphi_out=101,
    )["total"]

    def smooth_total(booz):
        out = quasi_isodynamic_residual_from_boozer_modes(
            bmnc_b=jnp.asarray(booz["bmnc_b"]),
            xm_b=jnp.asarray(booz["ixm_b"]),
            xn_b=jnp.asarray(booz["ixn_b"]),
            iota_b=jnp.asarray(booz["iota_b"]),
            nfp=1,
            nphi=33,
            nalpha=9,
            n_bounce=7,
            width_weight=1.0,
            branch_width_weight=0.5,
            profile_weight=0.1,
            shuffle_profile_weight=1.0,
        )
        return float(np.asarray(out["total"]))

    assert legacy_qh > legacy_qi
    assert smooth_total(qh_like) > smooth_total(qi_like)


def test_qi_boozer_mode_scan_reports_smooth_and_legacy_metrics(monkeypatch):
    pytest.importorskip("jax")

    script = ROOT / "examples" / "optimization" / "scan_qi_boozer_mode.py"
    spec = importlib.util.spec_from_file_location("scan_qi_boozer_mode", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "QI_NPHI", 21)
    monkeypatch.setattr(module, "QI_NALPHA", 7)
    monkeypatch.setattr(module, "QI_N_BOUNCE", 5)
    monkeypatch.setattr(module, "LEGACY_NPHI_OUT", 61)
    booz = _booz_like(xm=[0, 0, 1], xn=[0, 1, 1], coeffs=[1.0, 0.1, 0.04])
    summary = module.evaluate_boozer_mode_scan(
        booz,
        scales=[0.9, 1.0, 1.1],
        mode_index=1,
    )

    assert summary["mode_index"] == 1
    assert summary["mode_m"] == 0
    assert summary["mode_n"] == 1
    assert len(summary["rows"]) == 3
    assert all(np.isfinite(row["smooth_qi_total"]) for row in summary["rows"])
    assert all(np.isfinite(row["legacy_qi_total"]) for row in summary["rows"])
    assert np.isfinite(summary["smooth_roughness"])
    assert np.isfinite(summary["legacy_roughness"])
