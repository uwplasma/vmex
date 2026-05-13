from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.field import TWOPI
from vmec_jax.namelist import InData
from vmec_jax.profiles import MU0
from vmec_jax.solve import (
    _apply_preconditioner,
    _can_reassemble_precond_mats,
    _enforce_field_rows_np,
    _enforce_fixed_boundary_and_axis_np,
    _free_boundary_iter_controls_vmec,
    _gc_from_frzl,
    _icurv_full_mesh_from_indata,
    _mass_half_mesh_from_indata,
    _maybe_dump_gc,
    _maybe_dump_lam_prec,
    _pressure_half_mesh_from_indata,
    _vmec_force_flux_profiles,
    _vmec_scale_m1_factors_from_mats_np,
)
from vmec_jax.state import StateLayout, VMECState


def _static(*, m=(0, 1, 2), n=(0, 1, -1), nfp=2, ns=3, mpol=3, ntor=1, lasym=False, lthreed=True):
    return SimpleNamespace(
        modes=SimpleNamespace(m=np.asarray(m), n=np.asarray(n)),
        cfg=SimpleNamespace(
            nfp=nfp,
            ns=ns,
            mpol=mpol,
            ntor=ntor,
            lasym=lasym,
            lthreed=lthreed,
        ),
    )


def _state_from_array(arr) -> VMECState:
    arr = np.asarray(arr, dtype=float)
    layout = StateLayout(ns=int(arr.shape[0]), K=int(np.prod(arr.shape[1:])), lasym=False)
    return VMECState(
        layout=layout,
        Rcos=arr.copy(),
        Rsin=arr.copy(),
        Zcos=arr.copy(),
        Zsin=arr.copy(),
        Lcos=arr.copy(),
        Lsin=arr.copy(),
    )


def _filled_forces(shape=(2, 2, 2)):
    def a(value):
        return np.full(shape, float(value))

    return SimpleNamespace(
        frcc=a(1),
        fzsc=a(2),
        flsc=a(3),
        frss=a(4),
        fzcs=a(5),
        flcs=a(6),
        frsc=a(7),
        fzcc=a(8),
        flcc=a(9),
        frcs=a(10),
        fzss=a(11),
        flss=a(12),
    )


def test_free_boundary_cadence_sanitizes_bad_residual_before_threshold(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_FREEB_ACTIVATE_FSQ", "0.5")

    ivac, ivacskip, nvacskip = _free_boundary_iter_controls_vmec(
        iter2=5,
        iter1=1,
        ivac=3,
        nvacskip=4,
        nvskip0=7,
        fsq_rz_prev=-np.inf,
    )

    assert ivac == 3
    assert ivacskip == 0
    assert nvacskip == 7

    ivac, ivacskip, nvacskip = _free_boundary_iter_controls_vmec(
        iter2=6,
        iter1=1,
        ivac=2,
        nvacskip=5,
        nvskip0=5,
        fsq_rz_prev=0.25,
    )

    assert ivac == 3
    assert ivacskip == 0
    assert nvacskip == 5


def test_gc_from_frzl_maps_force_channels_for_symmetry_classes():
    frzl = _filled_forces()

    gcr, gcz, gcl = _gc_from_frzl(frzl=frzl, cfg=SimpleNamespace(lasym=True, lthreed=True))
    assert gcr.shape == (2, 2, 2, 4)
    np.testing.assert_allclose(gcr[..., 0], 1.0)
    np.testing.assert_allclose(gcr[..., 1], 4.0)
    np.testing.assert_allclose(gcr[..., 2], 7.0)
    np.testing.assert_allclose(gcr[..., 3], 10.0)
    np.testing.assert_allclose(gcz[..., 0], 2.0)
    np.testing.assert_allclose(gcz[..., 1], 5.0)
    np.testing.assert_allclose(gcz[..., 2], 8.0)
    np.testing.assert_allclose(gcz[..., 3], 11.0)
    np.testing.assert_allclose(gcl[..., 0], 3.0)
    np.testing.assert_allclose(gcl[..., 1], 6.0)
    np.testing.assert_allclose(gcl[..., 2], 9.0)
    np.testing.assert_allclose(gcl[..., 3], 12.0)

    gcr_axisym_asym, gcz_axisym_asym, gcl_axisym_asym = _gc_from_frzl(
        frzl=frzl,
        cfg=SimpleNamespace(lasym=True, lthreed=False),
    )
    assert gcr_axisym_asym.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(gcr_axisym_asym[..., 1], 7.0)
    np.testing.assert_allclose(gcz_axisym_asym[..., 1], 8.0)
    np.testing.assert_allclose(gcl_axisym_asym[..., 1], 9.0)

    gcr_symmetric, gcz_symmetric, gcl_symmetric = _gc_from_frzl(
        frzl=frzl,
        cfg=SimpleNamespace(lasym=False, lthreed=True),
    )
    assert gcr_symmetric.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(gcr_symmetric[..., 1], 4.0)
    np.testing.assert_allclose(gcz_symmetric[..., 1], 5.0)
    np.testing.assert_allclose(gcl_symmetric[..., 1], 6.0)


def test_gc_dump_writes_selected_force_channel_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DUMP_GC", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_GC_ITER", "3")
    monkeypatch.setenv("VMEC_JAX_DUMP_GC_STAGE", "both")
    monkeypatch.setenv("VMEC_JAX_DUMP_GC_DIR", str(tmp_path))

    static = _static(ns=2, mpol=2, ntor=1, lasym=False, lthreed=True)
    _maybe_dump_gc(frzl=_filled_forces(), static=static, iter_idx=3, label="raw")

    with np.load(tmp_path / "gc_raw_ns2_iter3.npz") as data:
        assert bool(data["lthreed"])
        assert not bool(data["lasym"])
        assert data["gcr"].shape == (2, 2, 2, 2)
        np.testing.assert_allclose(data["gcr"][..., 0], 1.0)
        np.testing.assert_allclose(data["gcr"][..., 1], 4.0)
        np.testing.assert_allclose(data["gcz"][..., 1], 5.0)
        np.testing.assert_allclose(data["gcl"][..., 1], 6.0)


def test_numpy_constraint_enforcement_handles_empty_single_and_full_meshes():
    empty = _enforce_field_rows_np(np.empty((0, 3)), axis_mask=np.ones(3), edge_row=np.arange(3.0))
    assert empty.shape == (0, 3)

    one_row = _enforce_field_rows_np(
        np.array([[9.0, 9.0, 9.0]]),
        axis_mask=np.array([1.0, 0.0, 1.0]),
        edge_row=np.array([3.0, 4.0, 5.0]),
    )
    np.testing.assert_allclose(one_row, [[3.0, 0.0, 5.0]])

    one_row_lambda = _enforce_field_rows_np(
        np.array([[9.0, 9.0, 9.0]]),
        edge_row=np.array([3.0, 4.0, 5.0]),
        zero_axis=True,
    )
    np.testing.assert_allclose(one_row_lambda, [[0.0, 0.0, 0.0]])

    base = np.arange(3 * 3, dtype=float).reshape(3, 3)
    state = VMECState(
        layout=StateLayout(ns=3, K=3, lasym=False),
        Rcos=base + 10.0,
        Rsin=base + 20.0,
        Zcos=base + 30.0,
        Zsin=base + 40.0,
        Lcos=base + 50.0,
        Lsin=base + 60.0,
    )
    static = SimpleNamespace(modes=SimpleNamespace(m=np.array([0, 1, 0])))

    constrained = _enforce_fixed_boundary_and_axis_np(
        state,
        static,
        edge_Rcos=np.array([100.0, 101.0, 102.0]),
        edge_Rsin=np.array([110.0, 111.0, 112.0]),
        edge_Zcos=np.array([120.0, 121.0, 122.0]),
        edge_Zsin=np.array([130.0, 131.0, 132.0]),
        idx00=0,
    )

    np.testing.assert_allclose(constrained.Rcos[0], [10.0, 0.0, 12.0])
    np.testing.assert_allclose(constrained.Rcos[-1], [100.0, 101.0, 102.0])
    np.testing.assert_allclose(constrained.Zsin[0], [40.0, 0.0, 42.0])
    np.testing.assert_allclose(constrained.Zsin[-1], [130.0, 131.0, 132.0])
    np.testing.assert_allclose(constrained.Lcos[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(constrained.Lsin[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(constrained.Lcos[:, 0], 0.0)
    np.testing.assert_allclose(constrained.Lsin[:, 0], 0.0)


def test_preconditioner_combines_mode_weights_and_radial_dirichlet_smoothing():
    arr = np.array(
        [
            [2.0, 6.0, 9.0],
            [0.0, 12.0, 18.0],
            [8.0, 24.0, 27.0],
        ]
    )
    grad = _state_from_array(arr)

    smoothed = _apply_preconditioner(
        grad,
        _static(m=(0, 1, 2), n=(0, 1, -1), nfp=2),
        kind=" mode_diag + radial_tridi ",
        exponent=1.0,
        radial_alpha=0.5,
    )

    expected = np.array(
        [
            [2.0, 1.0, 1.0],
            [2.5, 2.25, 2.0],
            [8.0, 4.0, 3.0],
        ]
    )
    np.testing.assert_allclose(np.asarray(smoothed.Rcos), expected)
    np.testing.assert_allclose(np.asarray(smoothed.Lsin), expected)


def test_preconditioner_validation_and_short_mesh_branches():
    grad = _state_from_array(np.ones((3, 2)))
    assert _apply_preconditioner(grad, _static(m=(0, 1), n=(0, 0)), kind=", ,") is grad

    with pytest.raises(ValueError, match="exponent"):
        _apply_preconditioner(grad, _static(m=(0, 1), n=(0, 0)), kind="mode_diag", exponent=0.0)

    with pytest.raises(ValueError, match="radial_alpha"):
        _apply_preconditioner(grad, _static(m=(0, 1), n=(0, 0)), kind="radial_tridi", radial_alpha=0.0)

    with pytest.raises(ValueError, match="Unknown preconditioner"):
        _apply_preconditioner(grad, _static(m=(0, 1), n=(0, 0)), kind="spectral_magic")

    short_batched = _state_from_array(np.arange(8.0).reshape(2, 2, 2))
    unchanged = _apply_preconditioner(
        short_batched,
        _static(m=(0, 1, 2, 3), n=(0, 0, 0, 0), ns=2),
        kind="radial_tridi",
        radial_alpha=1.0,
    )
    np.testing.assert_allclose(np.asarray(unchanged.Rcos), np.arange(8.0).reshape(2, 2, 2))

    malformed = _state_from_array(np.arange(3.0))
    with pytest.raises(ValueError, match="ndim>=2"):
        _apply_preconditioner(
            malformed,
            _static(m=(0,), n=(0,)),
            kind="radial_tridi",
            radial_alpha=1.0,
        )


def test_solve_profile_helpers_use_half_mesh_mass_and_current_conventions():
    s_full = np.asarray([0.0, 0.25, 1.0])
    indata = InData(
        scalars={
            "PMASS_TYPE": "power_series",
            "AM": [2.0, 1.0],
            "PRES_SCALE": 3.0,
            "LRFP": True,
            "NCURR": 1,
            "CURTOR": 5.0,
            "PCURR_TYPE": "power_series",
            "AC": [2.0],
        },
        indexed={},
    )

    s_half = np.asarray([0.0, 0.125, 0.625])
    pressure = _pressure_half_mesh_from_indata(indata=indata, s_full=s_full)
    expected_pressure = MU0 * 3.0 * (2.0 + s_half)
    np.testing.assert_allclose(np.asarray(pressure), expected_pressure)

    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s_full,
        phips=np.asarray([10.0, 20.0, 30.0]),
        chips=np.asarray([1.0, 2.0, 4.0]),
        r00=2.0,
        gamma=2.0,
        lrfp=True,
    )
    expected_mass = expected_pressure * (np.asarray([1.0, 2.0, 4.0]) * 2.0) ** 2
    expected_mass[0] = 0.0
    np.testing.assert_allclose(np.asarray(mass), expected_mass)

    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s_full, signgs=-1)
    expected_scale = -MU0 * 5.0 / (2.0 * np.pi) / 2.0
    np.testing.assert_allclose(np.asarray(icurv), expected_scale * np.asarray([0.0, 0.25, 1.25]))


def test_flux_profile_conversion_recovers_internal_units_and_full_mesh_chips():
    phipf_physical = -TWOPI * np.asarray([1.0, 2.0, 3.0])
    chipf_physical = -TWOPI * np.asarray([0.0, 2.0, 4.0])

    phipf_internal, chipf_internal, chips_eff = _vmec_force_flux_profiles(
        phipf=phipf_physical,
        chipf=chipf_physical,
        signgs=-1,
        flux_is_internal=False,
    )

    np.testing.assert_allclose(np.asarray(phipf_internal), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(chipf_internal), [0.0, 2.0, 4.0])
    np.testing.assert_allclose(np.asarray(chips_eff), [0.0, 1.0, 3.0])


def test_m1_scale_factors_np_and_reassembly_contract_handle_zero_denominators():
    parity_mats = {
        "ard_parity": np.array([[0.0, 2.0], [0.0, -1.0]]),
        "brd_parity": np.array([[0.0, 3.0], [0.0, 1.0]]),
        "azd_parity": np.array([[0.0, 5.0], [0.0, 2.0]]),
        "bzd_parity": np.array([[0.0, 0.0], [0.0, -2.0]]),
    }
    fac_r, fac_z = _vmec_scale_m1_factors_from_mats_np(parity_mats)
    np.testing.assert_allclose(fac_r, [0.5, 1.0])
    np.testing.assert_allclose(fac_z, [0.5, 1.0])

    dr = np.zeros((2, 2, 1))
    dz = np.zeros((2, 2, 1))
    dr[:, 1, 0] = [-2.0, -1.0]
    dz[:, 1, 0] = [-6.0, 1.0]
    fac_r, fac_z = _vmec_scale_m1_factors_from_mats_np({"dr": dr, "dz": dz})
    np.testing.assert_allclose(fac_r, [0.25, 1.0])
    np.testing.assert_allclose(fac_z, [0.75, 1.0])

    required = {
        "arm_parity": np.empty(0),
        "ard_parity": np.empty(0),
        "brm_parity": np.empty(0),
        "brd_parity": np.empty(0),
        "azm_parity": np.empty(0),
        "azd_parity": np.empty(0),
        "bzm_parity": np.empty(0),
        "bzd_parity": np.empty(0),
        "cxd_full": np.empty(0),
        "delta_s": 0.25,
    }
    assert _can_reassemble_precond_mats(required)
    assert not _can_reassemble_precond_mats({key: value for key, value in required.items() if key != "delta_s"})
    assert not _can_reassemble_precond_mats(object())


def test_lambda_preconditioner_dump_uses_vmec_t_channel_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "4")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))

    static = _static(ns=2, mpol=2, ntor=1, lasym=False, lthreed=True)
    lam_prec = np.arange(1.0, 9.0).reshape(2, 2, 2)
    faclam = 10.0 * lam_prec

    _maybe_dump_lam_prec(lam_prec=lam_prec, faclam=faclam, static=static, iter_idx=3)
    assert not (tmp_path / "lam_prec_ns2_iter3.npz").exists()

    _maybe_dump_lam_prec(lam_prec=lam_prec, faclam=faclam, static=static, iter_idx=4)

    with np.load(tmp_path / "lam_prec_ns2_iter4.npz") as data:
        expected = np.transpose(lam_prec, (0, 2, 1))
        assert data["pfaclam"].shape == (2, 2, 2, 2)
        np.testing.assert_allclose(data["pfaclam"][..., 0], expected)
        np.testing.assert_allclose(data["pfaclam"][:, 1, :, 1], expected[:, 1, :])
        np.testing.assert_allclose(data["pfaclam"][:, 0, 0, 1], 0.0)
        np.testing.assert_allclose(data["faclam"][..., 0], 10.0 * expected)
        np.testing.assert_allclose(data["faclam"][:, 0, 0, 1], 0.0)
