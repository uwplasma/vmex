from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools.diagnostics import profile_square_coil_free_boundary as profile


def test_square_coil_profile_residual_payload_keeps_solver_mode_and_history_tails():
    diagnostics = {
        "solver_mode": "parity",
        "use_scan": True,
        "performance_mode": False,
        "converged": False,
        "converged_strict": False,
        "requested_ftol": 1.0e-12,
        "final_fsqr": 1.0e-5,
        "final_fsqz": 2.0e-5,
        "final_fsql": 3.0e-6,
        "bad_resets": 0,
        "ijacob": 1,
        "free_boundary": {
            "nestor_model": "vmec2000_like_dense_integral",
            "couple_edge": True,
            "activate_fsq": 1.0e-3,
            "ivac": 3,
            "ivacskip": 0,
            "nvacskip": 2,
            "last_nestor_diagnostics": {
                "bnormal_rms": 4.0e-4,
                "bsqvac_rms": 1.5e-2,
            },
        },
        "freeb_ivac_history": np.array([1, 2, 3]),
        "freeb_ivacskip_history": np.array([0, 1, 0]),
        "freeb_full_update_history": np.array([1, 0, 1]),
        "freeb_nestor_reused_history": np.array([0, 1, 0]),
        "freeb_nestor_bnormal_rms_history": np.array([1.0e-3, 7.0e-4, 4.0e-4]),
        "include_edge_history": np.array([0, 1, 1]),
        "bad_jacobian_history": np.array([0, 0, 0]),
    }
    result = SimpleNamespace(
        n_iter=3,
        diagnostics=diagnostics,
        w_history=np.array([1.0, 0.5, 0.25, 0.125]),
        fsqr2_history=np.array([1.0e-3, 1.0e-4, 1.0e-5]),
        fsqz2_history=np.array([2.0e-3, 2.0e-4, 2.0e-5]),
        fsql2_history=np.array([3.0e-4, 3.0e-5, 3.0e-6]),
    )
    run = SimpleNamespace(result=result)

    payload = profile._final_residuals(run)

    assert payload["solver_mode"] == "parity"
    assert payload["use_scan"] is True
    assert payload["free_boundary_active"] is True
    assert payload["final_fsq_component_sum"] == pytest.approx(3.3e-5)
    assert payload["history"]["fsq_component_sum_tail"] == pytest.approx([0.0033, 0.00033, 3.3e-5])
    assert payload["history"]["freeb_ivac_tail"] == [1, 2, 3]
    assert payload["history"]["include_edge_tail"] == [0, 1, 1]
