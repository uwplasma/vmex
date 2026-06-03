from __future__ import annotations

import os
from pathlib import Path

import pytest

import vmec_jax as vj


@pytest.mark.skipif(
    os.environ.get("RUN_BOOTSTRAP_CURRENT_INTEGRATION", "").strip().lower() not in {"1", "true", "yes"},
    reason="set RUN_BOOTSTRAP_CURRENT_INTEGRATION=1 to run the VMEC/Redl fixed-point integration gate",
)
def test_bootstrap_current_fixed_point_reduces_redl_mismatch_on_finite_beta_tokamak() -> None:
    """Run a bounded real VMEC/Redl loop and require mismatch progress.

    This is intentionally optional because it launches several VMEC solves.  It
    is the finite-beta physics gate for the Redl-current profile feedback path.
    """

    root = Path(__file__).resolve().parents[1]
    indata = vj.read_indata(root / "examples" / "data" / "input.shaped_tokamak_pressure")
    profiles = vj.standard_finite_beta_profiles(1.0)
    indata = vj.with_pressure_profile(indata, profiles.pressure_pa, pres_scale=1.0)

    result = vj.bootstrap_current_fixed_point(
        indata,
        options=vj.BootstrapCurrentOptions(
            helicity_n=0,
            surfaces=(0.15, 0.30, 0.45, 0.60, 0.75, 0.90),
            n_current=24,
            policy="integrating_factor",
            damping=0.5,
            max_fixed_point_iter=3,
        ),
        ne_coeffs=profiles.ne_coeffs,
        Te_coeffs=profiles.Te_coeffs,
        Ti_coeffs=profiles.Ti_coeffs,
        Zeff_coeffs=profiles.Zeff_coeffs,
        run_kwargs={
            "max_iter": 250,
            "multigrid": False,
            "verbose": False,
            "jit_forces": "auto",
        },
    )

    assert len(result.history) >= 2
    first = result.history[0].mismatch_norm
    last = result.history[-1].mismatch_norm
    assert last < first
    assert abs(result.history[-1].curtor) > 0.0
