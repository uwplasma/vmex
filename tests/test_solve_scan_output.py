from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.solve_scan_output import (
    Vmec2000ScanHistories,
    postprocess_vmec2000_scan_result,
    unpack_vmec2000_scan_histories,
    vmec2000_scan_full_history_row,
    vmec2000_scan_light_history_row,
    vmec2000_scan_minimal_history_row,
    vmec2000_state_only_scan_diagnostics,
    vmec2000_traced_scan_diagnostics,
)


def _carry(**overrides):
    base = dict(
        time_step=0.25,
        inv_tau=np.asarray([0.6, 0.7]),
        fsq_prev=1.25,
        fsq0_prev=1.5,
        flip_sign=-1.0,
        iter1=2,
        iter_offset=7,
        res0=0.8,
        res1=0.9,
        fsqr_prev_phys=0.11,
        fsqz_prev_phys=0.22,
        cache_valid=True,
        ijacob=3,
        bad_resets=4,
        bad_growth=5,
        fsqz_prev=0.33,
        r00_prev=1.1,
        z00_prev=2.2,
        w_mhd_prev=3.3,
        force_bcovar_update=True,
        state_checkpoint={"state": "checkpoint"},
        vRcc=np.asarray([[1.0]]),
        vRss=np.asarray([[2.0]]),
        vZsc=np.asarray([[3.0]]),
        vZcs=np.asarray([[4.0]]),
        vLsc=np.asarray([[5.0]]),
        vLcs=np.asarray([[6.0]]),
        vRsc=np.asarray([[7.0]]),
        vRcs=np.asarray([[8.0]]),
        vZcc=np.asarray([[9.0]]),
        vZss=np.asarray([[10.0]]),
        vLcc=np.asarray([[11.0]]),
        vLss=np.asarray([[12.0]]),
        cache_precond_diag=("diag",),
        cache_tcon=("tcon",),
        cache_norms=("norms",),
        cache_rz_scale=1.2,
        cache_l_scale=1.3,
        cache_rz_norm=1.4,
        cache_f_norm1=1.5,
        cache_prec_rz_mats=("mats",),
        cache_prec_lam_prec=np.asarray([1.6]),
        probe_count=2,
        probe_bad_jac=1,
        probe_accept=1,
        probe_fsq_start=10.0,
        probe_fsq_min=2.0,
        probe_fsq_max=12.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _pack(base, heavy):
    if heavy:
        return {**base, **heavy}
    return dict(base)


def _freeb_controls(iter2: int, iter1: int, nvacskip: int) -> tuple[int, int]:
    ivacskip = (iter2 - iter1) % nvacskip
    return (1 if ivacskip == 0 else 2), ivacskip


def test_scan_history_row_builders_match_unpacker_layouts():
    minimal = unpack_vmec2000_scan_histories(
        vmec2000_scan_minimal_history_row("fsqr", "fsqz", "fsql"),
        scan_minimal=True,
        scan_light=False,
    )
    assert minimal.fsqr == "fsqr"
    assert minimal.fsql == "fsql"

    light = unpack_vmec2000_scan_histories(
        vmec2000_scan_light_history_row(
            "fsqr",
            "fsqz",
            "fsql",
            "accepted",
            "r00",
            "z00",
            "w_mhd",
            "dt",
            "bad_jac",
        ),
        scan_minimal=False,
        scan_light=True,
    )
    assert light.accepted == "accepted"
    assert light.dt == "dt"
    assert light.bad_jac == "bad_jac"

    full = unpack_vmec2000_scan_histories(
        vmec2000_scan_full_history_row(*range(25)),
        scan_minimal=False,
        scan_light=False,
    )
    assert full.fsqr == 0
    assert full.fsql1 == 5
    assert full.accepted == 6
    assert full.ptau_min == 19
    assert full.badjac_state == 24


def test_state_only_scan_diagnostics_include_host_scalars_only_when_untraced():
    carry = SimpleNamespace(
        abort_scan=np.asarray(False),
        converged=np.asarray(True),
        ijacob=np.asarray(3),
    )
    host = vmec2000_state_only_scan_diagnostics(
        carry_final=carry,
        traced=False,
        ftol=1.0e-9,
        scan_minimal=True,
        scan_light=False,
        scan_use_precomputed=True,
        scan_use_lax_tridi=False,
        timing_report={"total_s": 1.25},
    )
    assert host["state_only"] is True
    assert host["history_none"] is True
    assert host["converged"] is True
    assert host["ijacob"] == 3
    assert host["timing"] == {"total_s": 1.25}

    traced = vmec2000_state_only_scan_diagnostics(
        carry_final=carry,
        traced=True,
        ftol=1.0e-9,
        scan_minimal=False,
        scan_light=True,
        scan_use_precomputed=False,
        scan_use_lax_tridi=True,
    )
    assert traced["light_history"] is True
    assert "converged" not in traced
    assert "ijacob" not in traced


def test_traced_scan_diagnostics_preserve_resume_state_and_policy_flags():
    resume_state = {"time_step": "traced"}
    diagnostics = vmec2000_traced_scan_diagnostics(
        resume_state=resume_state,
        scan_use_precomputed=True,
        scan_use_lax_tridi=False,
    )

    assert diagnostics["traced_scan"] is True
    assert diagnostics["resume_state"] is resume_state
    assert diagnostics["scan_use_precomputed"] is True
    assert diagnostics["scan_use_lax_tridi"] is False


def _post(histories, **kwargs):
    defaults = dict(
        carry_final=_carry(),
        vmec2000_control=False,
        ftol=0.1,
        fsq_total_target=None,
        max_iter=10,
        scan_minimal=False,
        scan_light=False,
        resume_state_mode="none",
        pack_resume_state=_pack,
        free_boundary_enabled=False,
        freeb_nvacskip=3,
        freeb_nvskip0=2,
        iter_offset0=0,
        free_boundary_iter_controls=_freeb_controls,
    )
    defaults.update(kwargs)
    carry_final = defaults.pop("carry_final")
    return postprocess_vmec2000_scan_result(histories, carry_final, **defaults)


def test_scan_output_filters_non_vmec_by_accepted_mask():
    out = _post(
        Vmec2000ScanHistories(
            fsqr=np.asarray([5.0, 4.0, 3.0]),
            fsqz=np.asarray([0.0, 0.0, 0.0]),
            fsql=np.asarray([0.0, 0.0, 0.0]),
            accepted=np.asarray([True, False, True]),
            fsqr1=np.asarray([50.0, 40.0, 30.0]),
            fsqz1=np.asarray([5.0, 4.0, 3.0]),
            fsql1=np.asarray([0.5, 0.4, 0.3]),
            zero_m1=np.asarray([0, 1, 0]),
            include_edge=np.asarray([1, 1, 0]),
        )
    )

    np.testing.assert_array_equal(out.accepted_mask, np.asarray([True, False, True]))
    np.testing.assert_allclose(out.fsqr_history, np.asarray([5.0, 3.0]))
    np.testing.assert_allclose(out.fsqr1_history, np.asarray([50.0, 30.0]))
    np.testing.assert_array_equal(out.include_edge_history, np.asarray([1, 0]))


def test_scan_output_vmec_control_ignores_accepted_mask():
    out = _post(
        Vmec2000ScanHistories(
            fsqr=np.asarray([5.0, 4.0, 3.0]),
            fsqz=np.asarray([0.0, 0.0, 0.0]),
            fsql=np.asarray([0.0, 0.0, 0.0]),
            accepted=np.asarray([False, False, True]),
            fsqr1=np.asarray([50.0, 40.0, 30.0]),
            fsqz1=np.asarray([5.0, 4.0, 3.0]),
            fsql1=np.asarray([0.5, 0.4, 0.3]),
            zero_m1=np.asarray([0, 1, 0]),
            include_edge=np.asarray([1, 1, 0]),
        ),
        vmec2000_control=True,
    )

    np.testing.assert_array_equal(out.accepted_mask, np.asarray([True, True, True]))
    np.testing.assert_allclose(out.w_history, np.asarray([5.0, 4.0, 3.0]))


def test_scan_output_empty_history_reports_infinite_final_residuals():
    out = _post(
        Vmec2000ScanHistories(
            fsqr=np.asarray([]),
            fsqz=np.asarray([]),
            fsql=np.asarray([]),
            accepted=np.asarray([], dtype=bool),
            fsqr1=np.asarray([]),
            fsqz1=np.asarray([]),
            fsql1=np.asarray([]),
            zero_m1=np.asarray([], dtype=int),
            include_edge=np.asarray([], dtype=int),
        ),
        max_iter=1,
    )

    assert out.fsqr_history.size == 0
    assert np.isinf(out.final_fsqr)
    assert np.isinf(out.final_fsqz)
    assert np.isinf(out.final_fsql)


def test_scan_output_minimal_and_light_diagnostics():
    minimal = _post(
        unpack_vmec2000_scan_histories(
            (
                np.asarray([3.0, 2.0]),
                np.asarray([0.3, 0.2]),
                np.asarray([0.03, 0.02]),
            ),
            scan_minimal=True,
            scan_light=False,
        ),
        scan_minimal=True,
    )
    assert minimal.diagnostics["fsqr_full"].shape == (0,)
    np.testing.assert_allclose(minimal.fsqr_history, np.asarray([3.0, 2.0]))
    assert minimal.fsqr1_history.shape == (0,)
    assert minimal.time_step_history.shape == (0,)

    light = _post(
        unpack_vmec2000_scan_histories(
            (
                np.asarray([3.0, 2.0]),
                np.asarray([0.3, 0.2]),
                np.asarray([0.03, 0.02]),
                np.asarray([True, False]),
                np.asarray([1.0, 2.0]),
                np.asarray([10.0, 20.0]),
                np.asarray([100.0, 200.0]),
                np.asarray([0.5, 0.25]),
                np.asarray([False, True]),
            ),
            scan_minimal=False,
            scan_light=True,
        ),
        scan_light=True,
    )

    np.testing.assert_allclose(light.diagnostics["fsqr_full"], np.asarray([3.0, 2.0]))
    np.testing.assert_allclose(light.r00_history, np.asarray([1.0]))
    np.testing.assert_allclose(light.time_step_history, np.asarray([0.5]))
    assert light.fsqr1_history.shape == (0,)


def test_scan_output_convergence_truncation_keeps_histories_aligned():
    out = _post(
        Vmec2000ScanHistories(
            fsqr=np.asarray([10.0, 0.1, 0.05, 0.04]),
            fsqz=np.asarray([10.0, 0.1, 0.05, 0.04]),
            fsql=np.asarray([10.0, 0.1, 0.05, 0.04]),
            accepted=np.asarray([True, True, True, True]),
            fsqr1=np.asarray([1.0, 2.0, 3.0, 4.0]),
            fsqz1=np.asarray([10.0, 20.0, 30.0, 40.0]),
            fsql1=np.asarray([100.0, 200.0, 300.0, 400.0]),
            zero_m1=np.asarray([0, 1, 0, 1]),
            include_edge=np.asarray([1, 1, 0, 0]),
            dt=np.asarray([0.4, 0.3, 0.2, 0.1]),
        ),
        ftol=0.2,
    )

    assert out.conv_idx == 2
    np.testing.assert_array_equal(out.accepted_mask, np.asarray([True, True, False, False]))
    np.testing.assert_allclose(out.fsqr_history, np.asarray([10.0, 0.1]))
    np.testing.assert_allclose(out.fsqr1_history, np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(out.time_step_history, np.asarray([0.4, 0.3]))
    assert out.final_fsqr == 0.1
    assert out.converged_strict is True


def test_scan_output_resume_payload_fields_and_free_boundary_cadence():
    out = _post(
        Vmec2000ScanHistories(
            fsqr=np.asarray([5.0, 4.0, 3.0]),
            fsqz=np.asarray([0.0, 0.0, 0.0]),
            fsql=np.asarray([0.0, 0.0, 0.0]),
            accepted=np.asarray([True, True, True]),
            fsqr1=np.asarray([50.0, 40.0, 30.0]),
            fsqz1=np.asarray([5.0, 4.0, 3.0]),
            fsql1=np.asarray([0.5, 0.4, 0.3]),
            zero_m1=np.asarray([0, 1, 0]),
            include_edge=np.asarray([1, 1, 0]),
            iter1=np.asarray([1, 2, 2]),
        ),
        resume_state_mode="full",
        free_boundary_enabled=True,
        iter_offset0=10,
    )

    resume = out.resume_state
    assert resume is not None
    assert resume["iter_offset"] == 10
    assert resume["prev_rz_fsq"] == 0.33
    assert resume["freeb_ivac"] == 2
    assert resume["freeb_ivacskip"] == 2
    assert resume["freeb_nvacskip"] == 3
    assert resume["state_checkpoint"] == {"state": "checkpoint"}
    np.testing.assert_array_equal(resume["vRcc"], np.asarray([[1.0]]))
    np.testing.assert_array_equal(out.freeb_ivacskip_full, np.asarray([1, 1, 2]))
    np.testing.assert_array_equal(out.freeb_ivac_full, np.asarray([2, 2, 2]))
