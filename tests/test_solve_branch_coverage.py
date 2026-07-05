from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.solve as solve
from vmec_jax._compat import has_jax, jnp
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
from vmec_jax.kernels.tomnsp import TomnspsRZL


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


def _tiny_solver_static(*, ns: int = 3):
    return SimpleNamespace(
        cfg=SimpleNamespace(nfp=1),
        modes=SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, 0])),
        s=np.asarray([0.0]) if int(ns) == 1 else np.linspace(0.0, 1.0, int(ns)),
        grid=SimpleNamespace(theta=np.asarray([0.0, np.pi]), zeta=np.asarray([0.0])),
        basis=object(),
    )


def _tiny_solver_state(*, ns: int = 3, interior: float = 2.0) -> VMECState:
    ns = int(ns)
    layout = StateLayout(ns=ns, K=2, lasym=False)
    zeros = np.zeros((ns, 2), dtype=float)
    rcos = zeros.copy()
    rcos[:, 0] = 1.0
    if ns == 1:
        rcos[0, 1] = float(interior)
    else:
        rcos[1, 1] = float(interior)
        rcos[-1, 1] = 0.25
    return VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )


def _tiny_lambda_state() -> VMECState:
    state = _tiny_solver_state(ns=1)
    return VMECState(
        layout=state.layout,
        Rcos=state.Rcos,
        Rsin=state.Rsin,
        Zcos=state.Zcos,
        Zsin=state.Zsin,
        Lcos=np.asarray([[0.0, 0.3]], dtype=float),
        Lsin=np.asarray([[0.0, -0.4]], dtype=float),
    )


def _install_quadratic_geometry(monkeypatch):
    def fake_eval_geom(state, _static):
        rsum = jnp.sum(jnp.asarray(state.Rcos), axis=1)
        return SimpleNamespace(sqrtg=rsum[:, None, None] ** 2 + 1.0)

    monkeypatch.setattr(solve, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(solve, "bsup_from_geom", lambda _g, **_kwargs: (0.0, 0.0))
    monkeypatch.setattr(solve, "b2_from_bsup", lambda g, _bsupu, _bsupv: jnp.ones_like(g.sqrtg))


def _install_tiny_lambda_problem(monkeypatch):
    def fake_eval_geom(state, _static):
        shape = jnp.asarray(state.Lcos).shape
        return SimpleNamespace(
            g_tt=jnp.ones(shape),
            g_tp=jnp.zeros(shape),
            g_pp=jnp.ones(shape),
            sqrtg=jnp.ones(shape),
        )

    monkeypatch.setattr(solve, "eval_geom", fake_eval_geom)
    monkeypatch.setattr(solve, "eval_fourier_dtheta", lambda Lcos, _Lsin, *_args, **_kwargs: jnp.asarray(Lcos))
    monkeypatch.setattr(solve, "eval_fourier_dzeta_phys", lambda _Lcos, Lsin, *_args, **_kwargs: jnp.asarray(Lsin))
    monkeypatch.setattr(
        solve,
        "bsup_from_sqrtg_lambda",
        lambda *, lam_u, lam_v, **_kwargs: (lam_u, lam_v),
    )


def _negate_state(state: VMECState) -> VMECState:
    return VMECState(
        layout=state.layout,
        Rcos=-jnp.asarray(state.Rcos),
        Rsin=-jnp.asarray(state.Rsin),
        Zcos=-jnp.asarray(state.Zcos),
        Zsin=-jnp.asarray(state.Zsin),
        Lcos=-jnp.asarray(state.Lcos),
        Lsin=-jnp.asarray(state.Lsin),
    )


class _TinyResidualInData:
    scalars = {}
    indexed = {}

    def get_float(self, name, default=0.0):
        return {"FTOL": 0.0, "TCON0": 1.0, "GAMMA": 0.0}.get(str(name).upper(), default)

    def get_bool(self, name, default=False):
        return {"LFORBAL": False, "LRFP": False}.get(str(name).upper(), default)

    def get_int(self, name, default=0):
        return {"NCURR": 0}.get(str(name).upper(), default)


def _tiny_residual_static():
    cfg = SimpleNamespace(
        ns=3,
        mpol=2,
        ntor=0,
        nfp=1,
        ntheta=4,
        nzeta=1,
        lasym=False,
        lthreed=True,
        lconm1=True,
    )
    modes = SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, 0]), K=2)
    return SimpleNamespace(
        cfg=cfg,
        s=jnp.asarray([0.0, 0.5, 1.0]),
        modes=modes,
        trig_vmec=SimpleNamespace(name="fake-trig"),
    )


def _single_mode_residual(state: VMECState) -> TomnspsRZL:
    z = jnp.zeros((3, 2, 1), dtype=jnp.asarray(state.Rcos).dtype)
    x = jnp.asarray(state.Rcos)[1, 1]
    return TomnspsRZL(
        frcc=z.at[1, 1, 0].set(x),
        frss=None,
        fzsc=z,
        fzcs=None,
        flsc=z,
        flcs=None,
    )


def _nan_residual(state: VMECState) -> TomnspsRZL:
    z = jnp.zeros((3, 2, 1), dtype=jnp.asarray(state.Rcos).dtype)
    return TomnspsRZL(
        frcc=z.at[1, 1, 0].set(jnp.asarray(jnp.nan, dtype=z.dtype)),
        frss=None,
        fzsc=z,
        fzcs=None,
        flsc=z,
        flcs=None,
    )


def _install_fake_residual_physics(monkeypatch, residual_from_state):
    import vmec_jax.boundary as boundary_mod
    import vmec_jax.energy as energy_mod
    import vmec_jax.kernels.forces as forces_mod
    import vmec_jax.kernels.residue as residue_mod

    monkeypatch.setattr(
        energy_mod,
        "flux_profiles_from_indata",
        lambda _indata, s, signgs: SimpleNamespace(
            chipf=jnp.zeros_like(jnp.asarray(s)),
            phips=jnp.ones_like(jnp.asarray(s)),
            phipf=jnp.ones_like(jnp.asarray(s)),
        ),
    )
    monkeypatch.setattr(
        boundary_mod,
        "boundary_from_indata",
        lambda _indata, _modes, **_kwargs: SimpleNamespace(R_cos=np.asarray([1.0, 0.0])),
    )
    monkeypatch.setattr(solve, "_mass_half_mesh_from_indata", lambda **_kwargs: jnp.zeros(3))
    monkeypatch.setattr(solve, "_pressure_half_mesh_from_indata", lambda **_kwargs: jnp.zeros(3))
    monkeypatch.setattr(solve, "_icurv_full_mesh_from_indata", lambda **_kwargs: jnp.zeros(3))
    monkeypatch.setattr(
        solve,
        "_vmec_force_flux_profiles",
        lambda **kwargs: (jnp.asarray(kwargs["phipf"]), jnp.asarray(kwargs["chipf"]), jnp.asarray(kwargs["chipf"])),
    )

    def fake_forces(*, state, **_kwargs):
        sqrtg = jnp.ones((3, 1, 1), dtype=jnp.asarray(state.Rcos).dtype)
        return SimpleNamespace(state=state, bc=SimpleNamespace(jac=SimpleNamespace(sqrtg=sqrtg)))

    monkeypatch.setattr(forces_mod, "vmec_forces_rz_from_wout", fake_forces)
    monkeypatch.setattr(
        forces_mod,
        "vmec_residual_internal_from_kernels",
        lambda k, **_kwargs: residual_from_state(k.state),
    )
    monkeypatch.setattr(
        residue_mod,
        "vmec_force_norms_from_bcovar_dynamic",
        lambda **_kwargs: SimpleNamespace(
            r1=jnp.asarray(1.0),
            fnorm=jnp.asarray(1.0),
            fnormL=jnp.asarray(1.0),
        ),
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


def test_solve_entrypoints_raise_importerror_when_jax_unavailable(monkeypatch):
    monkeypatch.setattr(solve, "has_jax", lambda: False)
    state = object()
    static = object()
    common_flux = dict(phipf=None, chipf=None, signgs=1, lamscale=None)

    calls = [
        (
            "solve_lambda_gd requires JAX",
            lambda: solve.solve_lambda_gd(state, static, **common_flux),
        ),
        (
            "solve_fixed_boundary_gd requires JAX",
            lambda: solve.solve_fixed_boundary_gd(state, static, **common_flux),
        ),
        (
            "solve_fixed_boundary_lbfgs requires JAX",
            lambda: solve.solve_fixed_boundary_lbfgs(state, static, **common_flux),
        ),
        (
            "solve_fixed_boundary_lbfgs_vmec_residual requires JAX",
            lambda: solve.solve_fixed_boundary_lbfgs_vmec_residual(state, static, indata=object(), signgs=1),
        ),
        (
            "solve_fixed_boundary_gn_vmec_residual requires JAX",
            lambda: solve.solve_fixed_boundary_gn_vmec_residual(state, static, indata=object(), signgs=1),
        ),
        (
            "solve_fixed_boundary_residual_iter requires JAX",
            lambda: solve.solve_fixed_boundary_residual_iter(state, static, indata=object(), signgs=1),
        ),
        (
            "first_step_diagnostics requires JAX",
            lambda: solve.first_step_diagnostics(state, static, indata=object(), signgs=1),
        ),
    ]

    for message, call in calls:
        with pytest.raises(ImportError, match=message):
            call()


@pytest.mark.skipif(not has_jax(), reason="toy solver branches require JAX")
def test_solve_lambda_gd_one_surface_defaults_spacing_and_breaks_on_tolerance(monkeypatch, capsys):
    _install_tiny_lambda_problem(monkeypatch)

    result = solve.solve_lambda_gd(
        _tiny_lambda_state(),
        _tiny_solver_static(ns=1),
        phipf=np.ones(1),
        chipf=np.ones(1),
        signgs=1,
        lamscale=np.ones(1),
        max_iter=2,
        grad_tol=1.0e9,
        verbose=True,
    )

    assert result.n_iter == 0
    assert result.grad_rms_history.shape == (1,)
    assert result.step_history.shape == (0,)
    assert "[solve_lambda_gd] iter=000" in capsys.readouterr().out


@pytest.mark.skipif(not has_jax(), reason="toy fixed-boundary branches require JAX")
def test_fixed_boundary_gd_jitted_differentiable_one_surface_uses_pressure_default(monkeypatch):
    _install_quadratic_geometry(monkeypatch)

    result = solve.solve_fixed_boundary_gd(
        _tiny_solver_state(ns=1),
        _tiny_solver_static(ns=1),
        phipf=np.ones(1),
        chipf=np.zeros(1),
        signgs=1,
        lamscale=1.0,
        pressure=None,
        max_iter=1,
        step_size=0.1,
        jit_grad=True,
        differentiable=True,
        stop_grad_in_update=True,
        verbose=False,
    )

    assert result.n_iter == 1
    assert result.w_history.shape == (1,)
    np.testing.assert_allclose(np.asarray(result.step_history), [0.1])


@pytest.mark.skipif(not has_jax(), reason="toy fixed-boundary branches require JAX")
def test_fixed_boundary_gd_falls_back_to_raw_gradient_when_preconditioner_rejects(monkeypatch, capsys):
    _install_quadratic_geometry(monkeypatch)

    def uphill_preconditioner(grad, *_args, **_kwargs):
        return _negate_state(grad)

    monkeypatch.setattr(solve, "_apply_preconditioner", uphill_preconditioner)

    result = solve.solve_fixed_boundary_gd(
        _tiny_solver_state(ns=3, interior=2.0),
        _tiny_solver_static(ns=3),
        phipf=np.ones(3),
        chipf=np.zeros(3),
        signgs=1,
        lamscale=1.0,
        pressure=np.zeros(3),
        max_iter=1,
        step_size=0.25,
        grad_tol=0.0,
        max_backtracks=0,
        preconditioner="mode_diag",
        verbose=True,
    )

    assert result.n_iter == 1
    assert result.w_history[-1] < result.w_history[0]
    assert "fallback to unpreconditioned gradient" in capsys.readouterr().out


@pytest.mark.skipif(not has_jax(), reason="toy L-BFGS branches require JAX")
def test_fixed_boundary_lbfgs_one_surface_jitted_convergence_break(monkeypatch, capsys):
    _install_quadratic_geometry(monkeypatch)

    result = solve.solve_fixed_boundary_lbfgs(
        _tiny_solver_state(ns=1),
        _tiny_solver_static(ns=1),
        phipf=np.ones(1),
        chipf=np.zeros(1),
        signgs=1,
        lamscale=1.0,
        pressure=None,
        max_iter=1,
        grad_tol=1.0e9,
        jit_grad=True,
        verbose=True,
    )

    assert result.n_iter == 0
    assert result.grad_rms_history.shape == (1,)
    assert "[solve_fixed_boundary_lbfgs] iter=000" in capsys.readouterr().out


@pytest.mark.skipif(not has_jax(), reason="toy L-BFGS branches require JAX")
def test_fixed_boundary_lbfgs_failure_and_history_limit_branches(monkeypatch, capsys):
    _install_quadratic_geometry(monkeypatch)
    static = _tiny_solver_static(ns=3)

    failed = solve.solve_fixed_boundary_lbfgs(
        _tiny_solver_state(ns=3, interior=2.0),
        static,
        phipf=np.ones(3),
        chipf=np.zeros(3),
        signgs=1,
        lamscale=1.0,
        pressure=np.zeros(3),
        max_iter=1,
        step_size=1000.0,
        grad_tol=0.0,
        max_backtracks=0,
        verbose=True,
    )

    assert failed.n_iter == 0
    np.testing.assert_allclose(failed.step_history, [1000.0])
    assert "line search failed" in capsys.readouterr().out

    accepted = solve.solve_fixed_boundary_lbfgs(
        _tiny_solver_state(ns=3, interior=2.0),
        static,
        phipf=np.ones(3),
        chipf=np.zeros(3),
        signgs=1,
        lamscale=1.0,
        pressure=np.zeros(3),
        max_iter=3,
        step_size=0.25,
        history_size=1,
        grad_tol=0.0,
        max_backtracks=2,
        verbose=False,
    )

    assert accepted.n_iter >= 2
    assert np.all(np.diff(accepted.w_history) < 0.0)


@pytest.mark.skipif(not has_jax(), reason="toy residual optimizer branches require JAX")
def test_residual_lbfgs_accepts_best_finite_step_after_rejected_trial(monkeypatch, capsys):
    _install_fake_residual_physics(monkeypatch, _single_mode_residual)
    monkeypatch.setattr(solve, "_ensure_descent_direction", lambda g, _p: (jnp.zeros_like(g), 0.0, False))

    result = solve.solve_fixed_boundary_lbfgs_vmec_residual(
        _tiny_solver_state(ns=3, interior=2.0),
        _tiny_residual_static(),
        indata=_TinyResidualInData(),
        signgs=1,
        include_constraint_force=True,
        max_iter=1,
        max_backtracks=0,
        verbose=True,
    )

    assert result.n_iter == 1
    np.testing.assert_allclose(result.w_history, [1.0, 1.0])
    assert result.diagnostics["include_constraint_force"] is True
    assert "accepting best finite step" in capsys.readouterr().out


@pytest.mark.skipif(not has_jax(), reason="toy residual optimizer branches require JAX")
def test_residual_gn_rejects_nonfinite_initial_objective(monkeypatch):
    _install_fake_residual_physics(monkeypatch, _nan_residual)

    with pytest.raises(ValueError, match="non-finite residual objective"):
        solve.solve_fixed_boundary_gn_vmec_residual(
            _tiny_solver_state(ns=3, interior=2.0),
            _tiny_residual_static(),
            indata=_TinyResidualInData(),
            signgs=1,
            include_constraint_force=True,
            max_iter=1,
            jit_kernels=False,
            verbose=False,
        )


@pytest.mark.skipif(not has_jax(), reason="HLO dump branch coverage requires JAX")
def test_hlo_dump_key_fallback_as_text_and_verbose_error_paths(tmp_path, monkeypatch):
    jax_mod = pytest.importorskip("jax")
    import builtins

    class RaisingStatic:
        @property
        def cfg(self):
            raise RuntimeError("no cfg")

    class FakeTextIr:
        def as_text(self):
            return "fake-as-text"

    class FakeLowered:
        def compiler_ir(self, dialect):
            assert dialect == "hlo"
            return FakeTextIr()

    class FakeJitted:
        def lower(self, *_args, **_kwargs):
            return FakeLowered()

    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_DIR", str(tmp_path))
    monkeypatch.setattr(jax_mod, "jit", lambda _fn: FakeJitted())
    solve._HLO_DUMPED_KEYS.clear()

    solve._maybe_dump_hlo_kernel(
        label="tiny",
        fn=lambda x: x,
        args=(1,),
        kwargs={},
        static=RaisingStatic(),
        wout_like=SimpleNamespace(),
        force=True,
    )

    assert (tmp_path / "hlo_tiny_ns0_mpol0_ntor0.txt").read_text() == "fake-as-text"

    real_import = builtins.__import__

    def import_without_jax(name, *args, **kwargs):
        if name == "jax":
            raise ImportError("blocked jax import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_jax)
    solve._maybe_dump_hlo_kernel(
        label="importfail",
        fn=lambda x: x,
        args=(1,),
        kwargs={},
        static=SimpleNamespace(cfg=SimpleNamespace(ns=2, ntheta=4)),
        wout_like=SimpleNamespace(mpol=3, ntor=1, nfp=5, lasym=True),
        force=True,
    )
    monkeypatch.setattr(builtins, "__import__", real_import)

    class FailingJitted:
        def lower(self, *_args, **_kwargs):
            raise RuntimeError("lower boom")

    class StringOnlyIr:
        def __str__(self):
            return "xla-string"

    def string_xla_computation(_fn):
        def inner(*_args, **_kwargs):
            return StringOnlyIr()

        return inner

    monkeypatch.setattr(jax_mod, "jit", lambda _fn: FailingJitted())
    monkeypatch.setattr(jax_mod, "xla_computation", string_xla_computation, raising=False)
    solve._HLO_DUMPED_KEYS.clear()

    solve._maybe_dump_hlo_kernel(
        label="xla",
        fn=lambda x: x,
        args=(1,),
        kwargs={},
        static=SimpleNamespace(cfg=SimpleNamespace(ns=2, ntheta=4)),
        wout_like=SimpleNamespace(mpol=3, ntor=1, nfp=5, lasym=True),
        force=True,
    )

    assert (tmp_path / "hlo_xla_ns2_mpol3_ntor1.txt").read_text() == "xla-string"

    def failing_xla_computation(_fn):
        def inner(*_args, **_kwargs):
            raise RuntimeError("xla boom")

        return inner

    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_VERBOSE", "1")
    monkeypatch.setattr(jax_mod, "jit", lambda _fn: FailingJitted())
    monkeypatch.setattr(jax_mod, "xla_computation", failing_xla_computation, raising=False)
    solve._HLO_DUMPED_KEYS.clear()

    solve._maybe_dump_hlo_kernel(
        label="fail",
        fn=lambda x: x,
        args=(1,),
        kwargs={},
        static=SimpleNamespace(cfg=SimpleNamespace(ns=2, ntheta=4)),
        wout_like=SimpleNamespace(mpol=3, ntor=1, nfp=5, lasym=True),
        force=True,
    )

    error_text = (tmp_path / "hlo_fail_error_ns2_mpol3_ntor1.txt").read_text()
    assert "lower boom" in error_text
    assert "xla boom" in error_text


def test_disabled_lambda_dump_helpers_return_without_outputs(tmp_path, monkeypatch):
    monkeypatch.delenv("VMEC_JAX_DUMP_LAM", raising=False)
    monkeypatch.delenv("VMEC_JAX_DUMP_LAMCAL", raising=False)
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    static = _static(ns=2, mpol=2, ntor=0)

    solve._maybe_dump_lam_fsql1(fsql1_pre=1.0, fsql1_post=2.0, static=static, iter_idx=7)
    solve._maybe_dump_lamcal(lam_debug={"blam_pre": np.asarray([1.0])}, static=static, iter_idx=7)

    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not has_jax(), reason="JAX-backed helper branches require JAX")
def test_radial_mesh_and_axis_reset_helpers_cover_small_mesh_edges():
    rhs = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert solve._radial_tridi_smooth_dirichlet(rhs, alpha=0.0, skip_nonpositive=True) is rhs

    with pytest.raises(ValueError, match="ndim>=2"):
        solve._radial_tridi_smooth_dirichlet(np.asarray([1.0]), alpha=1.0)
    smoothed_4d = solve._radial_tridi_smooth_dirichlet(np.ones((3, 1, 1, 1)), alpha=1.0)
    assert np.asarray(smoothed_4d).shape == (3, 1, 1, 1)
    with pytest.raises(ValueError, match="ndim>=2"):
        solve._radial_tridi_smooth_dirichlet(np.ones((3, 1, 1)), alpha=1.0, allow_3d=False)

    unchanged_short = solve._radial_tridi_smooth_dirichlet(np.ones((2, 2, 1)), alpha=1.0)
    np.testing.assert_allclose(np.asarray(unchanged_short), 1.0)

    smoothed_3d = solve._radial_tridi_smooth_dirichlet(np.arange(4.0).reshape(4, 1, 1), alpha=0.5)
    assert np.asarray(smoothed_3d).shape == (4, 1, 1)
    np.testing.assert_allclose(np.asarray(smoothed_3d)[[0, -1], 0, 0], [0.0, 3.0])

    np.testing.assert_allclose(solve._pshalf_from_s_np(np.asarray([0.25])), [0.5])
    np.testing.assert_allclose(np.asarray(solve._pshalf_from_s_jax(np.asarray([0.25]), float)), [0.5])
    sm, sp = solve._sm_sp_from_s_np(np.asarray([0.0]))
    np.testing.assert_allclose(sm, [0.0, 0.0])
    np.testing.assert_allclose(sp, [0.0, 0.0])

    st = _tiny_solver_state(ns=3)
    st_axis = _tiny_solver_state(ns=3, interior=7.0)
    assert solve._merge_axis_reset_state(st=st, st_axis=st_axis, static=_static(m=(0, 1)), full_reset=True) is st_axis
    merged = solve._merge_axis_reset_state(
        st=st,
        st_axis=st_axis,
        static=SimpleNamespace(modes=SimpleNamespace(m=np.asarray([0, 1]))),
        full_reset=False,
    )
    np.testing.assert_allclose(np.asarray(merged.Rcos)[:, 0], np.asarray(st_axis.Rcos)[:, 0])
    np.testing.assert_allclose(np.asarray(merged.Rcos)[:, 1], np.asarray(st.Rcos)[:, 1])


@pytest.mark.skipif(not has_jax(), reason="JAX-backed helper branches require JAX")
def test_mode_gauge_and_constraint_helpers_cover_boundary_cases():
    assert solve._mode00_index(SimpleNamespace(m=np.asarray([1, 2]), n=np.asarray([0, 0]))) is None
    assert solve._mode00_index(SimpleNamespace(m=np.asarray([1, 0]), n=np.asarray([0, 0]))) == 1

    arr = jnp.asarray([[1.0, 2.0, 3.0]])
    same_lcos, same_lsin = solve._enforce_lambda_gauge(arr, arr + 10.0, idx00=None)
    assert same_lcos is arr
    assert same_lsin is not None
    np.testing.assert_allclose(np.asarray(solve._zero_coeff_column(arr[:, :1], idx=0)), [[0.0]])
    np.testing.assert_allclose(np.asarray(solve._zero_coeff_column(arr, idx=-1)), [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(np.asarray(solve._zero_coeff_column(arr, idx=2)), [[1.0, 2.0, 0.0]])
    np.testing.assert_allclose(np.asarray(solve._zero_coeff_column(arr, idx=1)), [[1.0, 0.0, 3.0]])

    mode_arr = jnp.asarray(np.arange(12.0).reshape(2, 3, 2))
    replacement = jnp.asarray([[100.0, 101.0], [102.0, 103.0]])
    assert solve._replace_mode_slice(None, mode_idx=0, replacement=replacement) is None
    np.testing.assert_allclose(np.asarray(solve._replace_mode_slice(mode_arr, mode_idx=-1, replacement=replacement)), np.asarray(mode_arr))
    np.testing.assert_allclose(np.asarray(solve._replace_mode_slice(mode_arr, mode_idx=0, replacement=replacement))[:, 0, :], replacement)
    np.testing.assert_allclose(np.asarray(solve._replace_mode_slice(mode_arr, mode_idx=2, replacement=replacement))[:, 2, :], replacement)
    np.testing.assert_allclose(np.asarray(solve._replace_mode_slice(mode_arr, mode_idx=1, replacement=replacement))[:, 1, :], replacement)
    assert solve._scale_mode_slice(None, mode_idx=1, scale=np.asarray([2.0, 3.0])) is None
    np.testing.assert_allclose(np.asarray(solve._scale_mode_slice(mode_arr, mode_idx=5, scale=np.asarray([2.0, 3.0]))), np.asarray(mode_arr))
    scaled = solve._scale_mode_slice(mode_arr, mode_idx=1, scale=np.asarray([2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(scaled)[:, 1, :], np.asarray(mode_arr)[:, 1, :] * np.asarray([[2.0], [3.0]]))

    np.testing.assert_allclose(solve._zero_coeff_column_np([[1.0, 2.0]], idx=5), [[1.0, 2.0]])
    assert solve._replace_mode_slice_np(None, mode_idx=0, replacement=replacement) is None
    np.testing.assert_allclose(solve._replace_mode_slice_np(mode_arr, mode_idx=5, replacement=replacement), np.asarray(mode_arr))
    assert solve._scale_mode_slice_np(None, mode_idx=0, scale=np.asarray([1.0])) is None
    np.testing.assert_allclose(solve._scale_mode_slice_np(mode_arr, mode_idx=5, scale=np.asarray([2.0, 3.0])), np.asarray(mode_arr))

    state = _state_from_array(np.arange(6.0).reshape(2, 3))
    assert (
        solve._apply_vmec_lambda_axis_rules_to_state(
            state,
            enforce_vmec_lambda_axis=False,
            host_update_assembly=True,
            idx00=1,
        )
        is state
    )
    host = solve._apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=True,
        idx00=1,
    )
    np.testing.assert_allclose(np.asarray(host.Lcos)[:, 1], 0.0)
    device = solve._apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=False,
        idx00=2,
    )
    np.testing.assert_allclose(np.asarray(device.Lsin)[:, 2], 0.0)

    full = solve._enforce_field_rows(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        axis_mask=np.asarray([1.0, 0.0]),
        edge_row=np.asarray([5.0, 6.0]),
    )
    np.testing.assert_allclose(np.asarray(full), [[1.0, 0.0], [5.0, 6.0]])
    single = solve._enforce_field_rows(
        np.asarray([[1.0, 2.0]]),
        axis_mask=np.asarray([1.0, 0.0]),
        edge_row=np.asarray([5.0, 6.0]),
    )
    np.testing.assert_allclose(np.asarray(single), [[5.0, 0.0]])


@pytest.mark.skipif(not has_jax(), reason="M=1 scaling helper branches require JAX")
def test_m1_preconditioner_scaling_host_and_device_branches():
    arr = np.ones((3, 2, 1), dtype=float)
    frzl = TomnspsRZL(
        frcc=arr,
        frss=2.0 * arr,
        fzsc=3.0 * arr,
        fzcs=4.0 * arr,
        flsc=5.0 * arr,
        flcs=6.0 * arr,
        frsc=7.0 * arr,
        frcs=8.0 * arr,
        fzcc=9.0 * arr,
        fzss=10.0 * arr,
        flcc=11.0 * arr,
        flss=12.0 * arr,
    )
    mats = {
        "dr": np.asarray([[0.0], [-2.0], [0.0], [-1.0]]).reshape(2, 2, 1),
        "dz": np.asarray([[0.0], [-6.0], [0.0], [-1.0]]).reshape(2, 2, 1),
    }

    assert solve._scale_m1_precond_rhs_from_mats(frzl, mats, lconm1=False, mpol=3, host_update_assembly=True) is frzl
    assert solve._scale_m1_precond_rhs_from_mats(frzl, mats, lconm1=True, mpol=1, host_update_assembly=True) is frzl

    host = solve._scale_m1_precond_rhs_from_mats(frzl, mats, lconm1=True, mpol=3, host_update_assembly=True)
    np.testing.assert_allclose(np.asarray(host.frss)[:, 1, 0], [0.5, 1.0, 2.0])
    np.testing.assert_allclose(np.asarray(host.fzcs)[:, 1, 0], [3.0, 2.0, 4.0])
    np.testing.assert_allclose(np.asarray(host.frsc)[:, 1, 0], [1.75, 3.5, 7.0])
    np.testing.assert_allclose(np.asarray(host.fzcc)[:, 1, 0], [6.75, 4.5, 9.0])

    device = solve._scale_m1_precond_rhs_from_mats(frzl, mats, lconm1=True, mpol=3, host_update_assembly=False)
    np.testing.assert_allclose(np.asarray(device.frss)[:, 1, 0], [0.5, 1.0, 2.0])
    np.testing.assert_allclose(np.asarray(device.fzcs)[:, 1, 0], [3.0, 2.0, 4.0])

    empty_mats = {"dr": np.empty((0, 2, 1)), "dz": np.empty((0, 2, 1))}
    assert (
        solve._scale_m1_precond_rhs_from_mats(
            frzl,
            empty_mats,
            lconm1=True,
            mpol=3,
            host_update_assembly=True,
        )
        is frzl
    )
    assert (
        solve._scale_m1_precond_rhs_from_mats(
            frzl,
            empty_mats,
            lconm1=True,
            mpol=3,
            host_update_assembly=False,
        )
        is frzl
    )


def test_dump_helpers_cover_enabled_iteration_and_directory_fallbacks(tmp_path, monkeypatch):
    static = _static(ns=2, mpol=2, ntor=1, lasym=True, lthreed=True)
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "4")

    monkeypatch.setenv("VMEC_JAX_DUMP_PRECOND_MATS", "1")
    solve._maybe_dump_precond_mats(
        mats={"ar": np.asarray([1.0]), "dz": np.asarray([2.0]), "ignored": np.asarray([3.0])},
        static=static,
        iter_idx=4,
        jmax=9,
        used_cache=True,
    )
    with np.load(tmp_path / "precond_mats_ns2_iter4.npz") as data:
        assert bool(data["used_cache"])
        assert set(data.files) >= {"ar", "dz", "jmax"}
        assert "ignored" not in data.files

    monkeypatch.setenv("VMEC_JAX_DUMP_LAM", "1")
    monkeypatch.delenv("VMEC_JAX_DUMP_LAM_ITER", raising=False)
    frzl_pre = _filled_forces(shape=(2, 2, 1))
    frzl_post = _filled_forces(shape=(2, 2, 1))
    frzl_post.flcs = 2.0 * np.asarray(frzl_post.flcs)
    solve._maybe_dump_lam_gcl(frzl_pre=frzl_pre, frzl_post=frzl_post, static=static, iter_idx=4, delta_s=0.5)
    assert (tmp_path / "lam_gcl_ns2_iter4.npz").exists()
    text = (tmp_path / "lam_fsql1_ns2_iter4.dat").read_text()
    assert "lambda fsql1" in text

    monkeypatch.setenv("VMEC_JAX_DUMP_SCALARS", "1")
    norms = SimpleNamespace(wb=1.0, wp=2.0, volume=3.0, r2=4.0, fnorm=5.0, fnormL=6.0)
    solve._maybe_dump_scalars(norms=norms, iter_idx=4, ns=2)
    assert "bcovar scalars" in (tmp_path / "scalars_ns2_iter4.dat").read_text()

    monkeypatch.setenv("VMEC_JAX_DUMP_GCX2", "1")
    solve._maybe_dump_gcx2(gcr2=1.0, gcz2=2.0, gcl2=3.0, iter_idx=4, include_edge=True, ns=2)
    assert "gcx2 dump" in (tmp_path / "gcx2_ns2_iter4.dat").read_text()
