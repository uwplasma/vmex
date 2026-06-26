from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.discrete_adjoint as da
import vmec_jax.preconditioner_1d_jax as pc
import vmec_jax.kernels.forces as vf
from vmec_jax.kernels.residue import (
    _constrain_m1_pair,
    vmec_fsq_sums_from_tomnsps,
    vmec_gcx2_from_tomnsps,
    vmec_scalxc_from_s,
)
from vmec_jax.kernels.tomnsp import (
    TomnspsRZL,
    _select_mparity,
    tomnsps_masks,
    vmec_theta_sizes,
    vmec_trig_tables,
)


def _rzl(
    *,
    shape: tuple[int, int, int] = (3, 3, 1),
    value: float = 1.0,
    optional: bool = False,
) -> TomnspsRZL:
    base = np.full(shape, value, dtype=float)
    opt = np.full(shape, value + 1.0, dtype=float) if optional else None
    return TomnspsRZL(
        frcc=base,
        frss=opt,
        fzsc=2.0 * base,
        fzcs=opt,
        flsc=3.0 * base,
        flcs=opt,
        frsc=opt,
        frcs=opt,
        fzcc=opt,
        fzss=opt,
        flcc=opt,
        flss=opt,
    )


def test_vmec_forces_residual_coeffs_select_parity_and_differentiate(monkeypatch) -> None:
    def fake_project_to_modes(field, _basis):
        arr = np.asarray(field, dtype=float)
        return arr, -10.0 * arr

    monkeypatch.setattr(vf, "project_to_modes", fake_project_to_modes)
    fields = {
        name: np.full((2, 4), float(i + 1))
        for i, name in enumerate(
            (
                "armn_e",
                "armn_o",
                "brmn_e",
                "brmn_o",
                "crmn_e",
                "crmn_o",
                "azmn_e",
                "azmn_o",
                "bzmn_e",
                "bzmn_o",
                "czmn_e",
                "czmn_o",
            )
        )
    }
    zeros = np.zeros((2, 1, 1))
    kernels = vf.VmecRZForceKernels(
        **fields,
        bc=SimpleNamespace(),
        arcon_e=zeros,
        arcon_o=zeros,
        azcon_e=zeros,
        azcon_o=zeros,
        gcon=zeros,
        pr1_even=zeros,
        pr1_odd=zeros,
        pz1_even=zeros,
        pz1_odd=zeros,
        pru_even=zeros,
        pru_odd=zeros,
        pzu_even=zeros,
        pzu_odd=zeros,
        prv_even=zeros,
        prv_odd=zeros,
        pzv_even=zeros,
        pzv_odd=zeros,
    )
    static = SimpleNamespace(
        modes=SimpleNamespace(m=np.asarray([0, 1, 2, 3]), n=np.asarray([0, 1, -1, 2])),
        grid=SimpleNamespace(nfp=3),
        basis=None,
    )

    coeffs = vf.rz_residual_coeffs_from_kernels(kernels, static=static)

    m = static.modes.m[None, :]
    n_phys = (static.modes.n * static.grid.nfp)[None, :]

    def selected(e_name: str, o_name: str) -> np.ndarray:
        even = fields[e_name]
        odd = fields[o_name]
        return np.where((static.modes.m % 2)[None, :] == 0, even, odd)

    a_r = selected("armn_e", "armn_o")
    b_r = selected("brmn_e", "brmn_o")
    c_r = selected("crmn_e", "crmn_o")
    a_z = selected("azmn_e", "azmn_o")
    b_z = selected("bzmn_e", "bzmn_o")
    c_z = selected("czmn_e", "czmn_o")
    np.testing.assert_allclose(np.asarray(coeffs.gcr_cos), a_r - m * (-10.0 * b_r) - n_phys * (-10.0 * c_r))
    np.testing.assert_allclose(np.asarray(coeffs.gcr_sin), -10.0 * a_r + m * b_r + n_phys * c_r)
    np.testing.assert_allclose(np.asarray(coeffs.gcz_cos), a_z - m * (-10.0 * b_z) - n_phys * (-10.0 * c_z))
    np.testing.assert_allclose(np.asarray(coeffs.gcz_sin), -10.0 * a_z + m * b_z + n_phys * c_z)


def test_vmec_residue_scalxc_empty_and_m1_noop_branches() -> None:
    assert np.asarray(vmec_scalxc_from_s(s=np.asarray([]), mpol=3)).shape == (0, 3)
    assert np.asarray(vmec_scalxc_from_s(s=np.asarray([0.0, 1.0]), mpol=0)).shape == (2, 0)

    gcr, gcz = _constrain_m1_pair(gcr=np.asarray([1.0, 2.0]), gcz=np.asarray([3.0, 5.0]), lconm1=False)
    np.testing.assert_allclose(np.asarray(gcr), [1.0, 2.0])
    np.testing.assert_allclose(np.asarray(gcz), [3.0, 5.0])


def test_vmec_residue_edge_policy_and_optional_blocks_are_accounted() -> None:
    frzl = _rzl(shape=(3, 2, 1), value=1.0, optional=False)

    no_edge = vmec_fsq_sums_from_tomnsps(
        frzl=frzl,
        apply_m1_constraints=False,
        apply_scalxc=False,
        include_edge=False,
    )
    with_edge = vmec_fsq_sums_from_tomnsps(
        frzl=frzl,
        apply_m1_constraints=False,
        apply_scalxc=False,
        include_edge=True,
    )

    assert no_edge.gcr2_blocks == {"frcc": 4.0}
    assert with_edge.gcr2_blocks == {"frcc": 6.0}
    assert no_edge.gcz2_blocks == {"fzsc": 16.0}
    assert with_edge.gcz2_blocks == {"fzsc": 24.0}
    assert no_edge.gcl2_blocks == {"flsc": 54.0}

    gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
        frzl=frzl,
        apply_m1_constraints=False,
        apply_scalxc=False,
        include_edge=True,
    )
    np.testing.assert_allclose(np.asarray([gcr2, gcz2, gcl2]), [6.0, 24.0, 54.0])


def test_vmec_tomnsp_masks_cache_and_invalid_trig_edges() -> None:
    assert vmec_theta_sizes(5, lasym=True) == (4, 3, 4)
    with pytest.raises(ValueError, match="nzeta must be positive"):
        vmec_trig_tables(ntheta=4, nzeta=0, nfp=1, mmax=0, nmax=0, lasym=False)

    masks = tomnsps_masks(ns=4, mpol=3, include_edge=False, dtype=np.float64)
    cached = tomnsps_masks(ns=4, mpol=3, include_edge=False, dtype=np.float64)
    assert cached is masks
    np.testing.assert_array_equal(np.asarray(masks.mask_even), [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(
        np.asarray(masks.mask_rz[:, :, 0]),
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
    )
    np.testing.assert_array_equal(np.asarray(masks.mask_l[:, :, 0])[0], [0.0, 0.0, 0.0])

    edge_masks = tomnsps_masks(ns=4, mpol=3, include_edge=True, dtype=np.float64)
    np.testing.assert_array_equal(np.asarray(edge_masks.mask_rz[:, :, 0])[-1], [1.0, 1.0, 1.0])
    selected = _select_mparity(
        np.full((1, 3, 1), 10.0),
        np.full((1, 3, 1), -2.0),
        edge_masks.mask_even_j,
    )
    np.testing.assert_allclose(np.asarray(selected)[0, :, 0], [10.0, -2.0, 10.0])


def test_preconditioner_profile_weights_and_env_flag_resolution(monkeypatch) -> None:
    sqrt_sf, sqrt_sh = pc._sqrt_profiles_from_ns(3, dtype=np.float64)
    np.testing.assert_allclose(np.asarray(sqrt_sf), [0.0, np.sqrt(0.5), 1.0])
    np.testing.assert_allclose(np.asarray(sqrt_sh), [0.5, np.sqrt(0.75)])

    sm, sp = pc._sm_sp_from_profiles(sqrt_sf, sqrt_sh)
    np.testing.assert_allclose(np.asarray(sm), [1.0 / np.sqrt(2.0), np.sqrt(0.75)])
    np.testing.assert_allclose(np.asarray(sp), [1.0 / np.sqrt(2.0), np.sqrt(1.5)])

    cfg_sym = SimpleNamespace(ntheta=6, nzeta=2, lasym=False)
    cfg_asym = SimpleNamespace(ntheta=6, nzeta=2, lasym=True)
    np.testing.assert_allclose(
        np.asarray(pc._wint_from_config(cfg=cfg_sym, dtype=np.float64)),
        [1.0 / 12.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 12.0],
    )
    np.testing.assert_allclose(np.asarray(pc._wint_from_config(cfg=cfg_asym, dtype=np.float64)), np.full(6, 1.0 / 12.0))

    monkeypatch.setenv("VMEC_JAX_TRIDI_PRECOMPUTE", "yes")
    monkeypatch.setenv("VMEC_JAX_TRIDI_SOLVE", "force")
    assert pc._resolve_tridi_flags(use_precomputed=None, use_lax_tridi=None) == (True, True)
    assert pc._resolve_tridi_flags(use_precomputed=False, use_lax_tridi=False) == (False, False)


def test_preconditioner_tridiagonal_solvers_match_dense_reference() -> None:
    a = np.asarray([[0.3], [0.2], [0.0]])
    d = np.asarray([[2.0], [2.5], [3.0]])
    b = np.asarray([[0.0], [0.4], [0.1]])
    rhs = np.asarray([[1.0], [2.0], [4.0]])[:, :, None]
    dense = np.asarray([[2.0, 0.3, 0.0], [0.4, 2.5, 0.2], [0.0, 0.1, 3.0]])
    expected = np.linalg.solve(dense, rhs[:, 0, 0])[:, None, None]

    direct = pc._tridi_solve_batched_jmin0(a, d, b, rhs, use_lax_tridi=False)
    cp, inv = pc._tridi_precompute_coeffs(a, d, b)
    precomputed = pc._tridi_solve_precomputed(b, cp, inv, rhs)
    dl_t, d_t, du_t = pc._tridi_pretranspose_for_lax(a, d, b)
    lax_pretransposed = pc._tridi_solve_batched_jmin0_lax_pretransposed(dl_t, d_t, du_t, rhs)

    np.testing.assert_allclose(np.asarray(direct), expected, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(precomputed), expected, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(lax_pretransposed), expected, rtol=1e-6)
    assert np.asarray(pc._tridi_solve_batched_jmin0(a[:0], d[:0], b[:0], rhs[:0])).shape == (0, 1, 1)


def test_preconditioner_apply_preserves_absent_optional_blocks_and_axis_rule() -> None:
    cfg = SimpleNamespace(lthreed=False, lasym=False)
    shape = (3, 2, 1)
    mats = {
        "ar": np.zeros(shape),
        "br": np.zeros(shape),
        "dr": np.full(shape, 2.0),
        "az": np.zeros(shape),
        "bz": np.zeros(shape),
        "dz": np.full(shape, 4.0),
    }
    frzl = TomnspsRZL(
        frcc=np.arange(6.0).reshape(shape),
        frss=None,
        fzsc=np.arange(6.0, 12.0).reshape(shape),
        fzcs=None,
        flsc=np.ones(shape),
        flcs=None,
    )

    out = pc.rz_preconditioner_apply(frzl_in=frzl, mats=mats, jmax=3, cfg=cfg, use_precomputed=True)

    assert out.frss is None
    assert out.fzcs is None
    np.testing.assert_allclose(np.asarray(out.frcc)[:, 0, :], np.asarray(frzl.frcc)[:, 0, :] / 2.0)
    np.testing.assert_allclose(np.asarray(out.fzsc)[:, 0, :], np.asarray(frzl.fzsc)[:, 0, :] / 4.0)
    assert float(np.asarray(out.frcc)[0, 1, 0]) == 0.0
    assert float(np.asarray(out.fzsc)[0, 1, 0]) == 0.0


def test_discrete_adjoint_trace_helpers_validate_lengths_and_compact_diagnostics() -> None:
    diagnostics = {
        "iter2_history": [1, 2, 3],
        "step_status_history": ["momentum", "rejected", "restart_bad_jacobian"],
        "timing": {"iterations": 3, "solve_s": 1.25, "label": "skip", "enabled": True},
        "converged": np.bool_(False),
        "final_fsq": np.float64(2.5),
    }
    result = SimpleNamespace(diagnostics=diagnostics)

    trace = da.residual_iteration_trace_from_result(result)

    np.testing.assert_array_equal(trace.state_advanced, [True, False, False])
    np.testing.assert_array_equal(trace.iter2, [1, 2, 3])
    compact = da._compact_tape_diagnostics(diagnostics)
    assert compact == {"timing": {"iterations": 3, "solve_s": 1.25}, "converged": False, "final_fsq": 2.5}

    bad = SimpleNamespace(diagnostics={"iter2_history": [1, 2], "time_step_history": [1.0]})
    with pytest.raises(ValueError, match="inconsistent residual trace lengths"):
        da.residual_iteration_trace_from_result(bad)
    with pytest.raises(TypeError, match="diagnostics must be a dict"):
        da.residual_iteration_trace_from_result(SimpleNamespace(diagnostics=None))


def test_discrete_adjoint_dynamic_support_guards_and_velocity_helpers() -> None:
    supported = {
        "branch": "strict_update",
        "step_status": "momentum",
        "restart_reason": "none",
        "restart_path": "momentum_accept",
    }
    restart = {
        "branch": "strict_update",
        "step_status": "restart_bad_progress",
        "restart_path": "catastrophic_nonfinite",
    }
    assert da._dynamic_replay_trace_supported(supported)
    assert not da._dynamic_replay_trace_supported({**supported, "restart_reason": "restart"})
    assert da._dynamic_restart_trace_supported(restart)
    assert not da._dynamic_restart_trace_supported({**restart, "restart_path": "ordinary_restart"})

    one = np.ones((1, 1))
    block = da.strict_update_velocity_block(
        b1=0.5,
        fac=2.0,
        force_scale=3.0,
        flip_sign=-1.0,
        vRcc_before=one,
        vRss_before=one,
        vZsc_before=one,
        vZcs_before=one,
        vLsc_before=one,
        vLcs_before=one,
        frcc_u=2.0 * one,
        frss_u=2.0 * one,
        fzsc_u=2.0 * one,
        fzcs_u=2.0 * one,
        flsc_u=2.0 * one,
        flcs_u=2.0 * one,
    )
    np.testing.assert_allclose(np.asarray(block["vRcc_after"]), -11.0 * one)
    assert block["vRsc_after"] is None

    limited = da.strict_update_velocity_limit(
        dt_eff=2.0,
        max_update_rms=1.0,
        limit_update_rms=True,
        vRcc=one,
        vRss=one,
        vZsc=one,
        vZcs=one,
        vLsc=one,
        vLcs=one,
    )
    assert float(np.asarray(limited["update_rms_scale"])) < 1.0
    np.testing.assert_allclose(
        np.asarray(limited["update_rms_postclip"]),
        1.0,
        rtol=1e-6,
    )

    skipped = da.strict_update_velocity_limit(
        dt_eff=2.0,
        max_update_rms=1.0,
        limit_update_rms=False,
        need_update_rms=False,
        vRcc=one,
        vRss=one,
        vZsc=one,
        vZcs=one,
        vLsc=one,
        vLcs=one,
    )
    assert float(np.asarray(skipped["update_rms_preclip"])) == 0.0
    assert float(np.asarray(skipped["update_rms_scale"])) == 1.0


def test_discrete_adjoint_dynamic_safe_dt_limits_only_positive_finite_rms() -> None:
    one = np.ones((1, 1))
    limited = da._dynamic_safe_dt_from_force_arrays(
        dt_nominal=4.0,
        max_coeff_delta_rms=9.0,
        frcc=3.0 * one,
        frss=0.0 * one,
        fzsc=0.0 * one,
        fzcs=0.0 * one,
        flsc=0.0 * one,
        flcs=0.0 * one,
        frsc=0.0 * one,
        frcs=0.0 * one,
        fzcc=0.0 * one,
        fzss=0.0 * one,
        flcc=0.0 * one,
        flss=0.0 * one,
    )
    np.testing.assert_allclose(np.asarray(limited), np.sqrt(3.0))

    zero = da._dynamic_safe_dt_from_force_arrays(
        dt_nominal=4.0,
        max_coeff_delta_rms=9.0,
        frcc=0.0 * one,
        frss=0.0 * one,
        fzsc=0.0 * one,
        fzcs=0.0 * one,
        flsc=0.0 * one,
        flcs=0.0 * one,
        frsc=0.0 * one,
        frcs=0.0 * one,
        fzcc=0.0 * one,
        fzss=0.0 * one,
        flcc=0.0 * one,
        flss=0.0 * one,
    )
    assert float(np.asarray(zero)) == 4.0
