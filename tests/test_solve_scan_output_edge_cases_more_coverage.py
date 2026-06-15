from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.solvers.fixed_boundary.scan.output import Vmec2000ScanHistories, postprocess_vmec2000_scan_result


def _carry(**overrides):
    values = dict(
        time_step=0.125,
        inv_tau=np.asarray([0.5, 0.25]),
        fsq_prev=1.0,
        fsq0_prev=2.0,
        flip_sign=1.0,
        iter1=4,
        iter_offset=11,
        res0=0.1,
        res1=0.2,
        fsqr_prev_phys=0.3,
        fsqz_prev_phys=0.4,
        cache_valid=False,
        ijacob=5,
        bad_resets=6,
        bad_growth=7,
        fsqz_prev=0.8,
        r00_prev=1.25,
        z00_prev=-0.5,
        w_mhd_prev=3.5,
        force_bcovar_update=False,
        state_checkpoint={"checkpoint": True},
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
        cache_rz_scale=1.1,
        cache_l_scale=1.2,
        cache_rz_norm=1.3,
        cache_f_norm1=1.4,
        cache_prec_rz_mats=("mats",),
        cache_prec_lam_prec=np.asarray([1.5]),
        probe_count=4,
        probe_bad_jac=1,
        probe_accept=3,
        probe_fsq_start=8.0,
        probe_fsq_min=2.0,
        probe_fsq_max=16.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _free_boundary_iter_controls(iter2: int, iter1: int, nvacskip: int) -> tuple[int, int]:
    ivacskip = (iter2 - iter1) % nvacskip
    return (1 if ivacskip == 0 else 2), ivacskip


def _post(histories: Vmec2000ScanHistories, **overrides):
    kwargs = dict(
        carry_final=_carry(),
        vmec2000_control=False,
        ftol=1.0e-6,
        fsq_total_target=None,
        max_iter=10,
        scan_minimal=False,
        scan_light=False,
        resume_state_mode="none",
        pack_resume_state=lambda base, heavy: {"base": base, "heavy": heavy},
        free_boundary_enabled=False,
        freeb_nvacskip=3,
        freeb_nvskip0=2,
        iter_offset0=0,
        free_boundary_iter_controls=_free_boundary_iter_controls,
    )
    kwargs.update(overrides)
    carry_final = kwargs.pop("carry_final")
    return postprocess_vmec2000_scan_result(histories, carry_final, **kwargs)


def _full_histories(**overrides) -> Vmec2000ScanHistories:
    values = dict(
        fsqr=np.asarray([4.0, 3.0, 2.0, 1.0]),
        fsqz=np.asarray([0.4, 0.3, 0.2, 0.1]),
        fsql=np.asarray([0.04, 0.03, 0.02, 0.01]),
        accepted=np.asarray([True, True, True, True]),
        fsqr1=np.asarray([40.0, 30.0, 20.0, 10.0]),
        fsqz1=np.asarray([4.0, 3.0, 2.0, 1.0]),
        fsql1=np.asarray([0.4, 0.3, 0.2, 0.1]),
        zero_m1=np.asarray([0, 1, 0, 1]),
        include_edge=np.asarray([1, 1, 0, 0]),
        res0=np.asarray([8.0, 7.0, 6.0, 5.0]),
        res1=np.asarray([0.8, 0.7, 0.6, 0.5]),
        iter1=np.asarray([1, 2, 2, 3]),
        min_tau=np.asarray([0.1, 0.2, 0.3, 0.4]),
        max_tau=np.asarray([1.1, 1.2, 1.3, 1.4]),
        ptau_min=np.asarray([2.1, 2.2, 2.3, 2.4]),
        ptau_max=np.asarray([3.1, 3.2, 3.3, 3.4]),
        tau_min_state=np.asarray([4.1, 4.2, 4.3, 4.4]),
        tau_max_state=np.asarray([5.1, 5.2, 5.3, 5.4]),
        badjac_ptau=np.asarray([0, 1, 0, 1]),
        badjac_state=np.asarray([1, 0, 1, 0]),
        bad_jac=np.asarray([False, True, False, True]),
    )
    values.update(overrides)
    return Vmec2000ScanHistories(**values)


def test_fsq_total_target_marks_total_convergence_without_strict_convergence():
    out = _post(
        _full_histories(
            fsqr=np.asarray([1.0, 0.9, 0.1, 0.09]),
            fsqz=np.asarray([1.0, 0.8, 0.1, 0.09]),
            fsql=np.asarray([1.0, 0.7, 0.1, 0.09]),
        ),
        ftol=1.0e-4,
        fsq_total_target=0.31,
    )

    assert out.conv_idx == 3
    assert out.converged_strict is False
    assert out.converged_total is True
    np.testing.assert_array_equal(out.accepted_mask, np.asarray([True, True, True, False]))
    np.testing.assert_allclose(out.w_history, np.asarray([3.0, 2.4, 0.3]))
    assert out.diagnostics["converged_by_total_fsq"] is True


def test_empty_accepted_history_falls_back_to_convergence_index_for_final_values():
    out = _post(
        _full_histories(
            fsqr=np.asarray([5.0, 0.2, 0.01]),
            fsqz=np.asarray([5.0, 0.2, 0.01]),
            fsql=np.asarray([5.0, 0.2, 0.01]),
            accepted=np.asarray([False, False, False]),
            fsqr1=np.asarray([50.0, 20.0, 1.0]),
            fsqz1=np.asarray([5.0, 2.0, 0.1]),
            fsql1=np.asarray([0.5, 0.2, 0.01]),
            zero_m1=np.asarray([0, 0, 1]),
            include_edge=np.asarray([1, 1, 0]),
        ),
        ftol=0.05,
    )

    assert out.accepted_idx.size == 0
    assert out.n_iter_hist == 0
    assert out.w_history.size == 0
    assert out.final_fsqr == 0.01
    assert out.final_fsqz == 0.01
    assert out.final_fsql == 0.01
    assert out.converged_strict is True
    assert out.resume_iter_offset == 11


def test_accepted_indices_are_truncated_by_max_iter_before_history_slicing():
    out = _post(
        _full_histories(
            fsqr=np.asarray([6.0, 5.0, 4.0, 3.0]),
            fsqz=np.zeros(4),
            fsql=np.zeros(4),
            accepted=np.asarray([True, True, True, True]),
        ),
        max_iter=2,
    )

    np.testing.assert_array_equal(out.accepted_idx, np.asarray([0, 1]))
    np.testing.assert_allclose(out.fsqr_history, np.asarray([6.0, 5.0]))
    np.testing.assert_allclose(out.fsqr1_history, np.asarray([40.0, 30.0]))
    assert out.final_fsqr == 5.0
    assert out.n_iter_hist == 2
    assert out.conv_idx_print == 2


def test_free_boundary_without_iter1_history_keeps_full_vacuum_arrays_empty():
    out = _post(
        _full_histories(iter1=None),
        carry_final=_carry(iter_offset=5, iter1=2),
        free_boundary_enabled=True,
        freeb_nvacskip=4,
        freeb_nvskip0=1,
    )

    assert out.free_boundary_diag == {
        "enabled": True,
        "nvacskip": 4,
        "nvskip0": 1,
        "ivac": 2,
        "ivacskip": 3,
        "vacuum_stub": True,
    }
    assert out.iter1_full.size == 0
    assert out.freeb_ivac_full.size == 0
    assert out.freeb_ivacskip_full.size == 0
    np.testing.assert_array_equal(out.diagnostics["freeb_full_update_full"], np.zeros((0,), dtype=int))


def test_minimal_resume_state_packs_only_base_payload_and_probe_diagnostics():
    pack_calls = []

    def pack(base, heavy):
        pack_calls.append((base, heavy))
        return {"base": dict(base), "heavy": heavy}

    out = _post(
        _full_histories(),
        resume_state_mode="minimal",
        pack_resume_state=pack,
        free_boundary_enabled=True,
        freeb_nvacskip=5,
        freeb_nvskip0=3,
    )

    assert len(pack_calls) == 1
    base, heavy = pack_calls[0]
    assert heavy is None
    assert base["iter_offset"] == 15
    assert base["prev_rz_fsq"] == 0.7
    assert base["vmec2000_cache_valid"] is False
    assert base["freeb_nvacskip"] == 5
    assert "state_checkpoint" not in base
    assert out.resume_state == {"base": base, "heavy": None}
    assert out.probe_ratio == 2.0
    assert out.probe_accept_frac == 0.75
    assert out.diagnostics["probe_accept_frac"] == 0.75
