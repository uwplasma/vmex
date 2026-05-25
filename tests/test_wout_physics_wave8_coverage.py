from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import numpy as np

import vmec_jax.finite_beta as finite_beta
import vmec_jax.wout as wout_module
from vmec_jax._compat import jnp
from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.modes import ModeTable, vmec_mode_table
from vmec_jax.namelist import InData
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.vmec_bcovar import VmecHalfMeshBcovar
from vmec_jax.vmec_jacobian import VmecHalfMeshJacobian
from vmec_jax.vmec_tomnsp import vmec_trig_tables


def _axisym_state(ns: int = 3) -> VMECState:
    modes = vmec_mode_table(2, 0)
    layout = StateLayout(ns=ns, K=modes.K, lasym=False)
    radial = np.linspace(0.0, 1.0, ns)[:, None]
    return VMECState(
        layout=layout,
        Rcos=np.concatenate([2.0 + 0.0 * radial, 0.08 + 0.02 * radial], axis=1),
        Rsin=np.zeros((ns, modes.K)),
        Zcos=np.zeros((ns, modes.K)),
        Zsin=np.concatenate([0.0 * radial, 0.20 + 0.03 * radial], axis=1),
        Lcos=np.zeros((ns, modes.K)),
        Lsin=np.zeros((ns, modes.K)),
    )


def _fake_bcovar(ns: int, trig) -> SimpleNamespace:
    shape = (ns, int(trig.ntheta2), int(np.asarray(trig.cosnv).shape[0]))
    base = np.ones(shape, dtype=float)
    jac = SimpleNamespace(
        sqrtg=1.1 * base,
        r12=2.0 * base,
        tau=0.55 * base,
        rs=np.zeros(shape),
        zs=np.zeros(shape),
        ru12=0.1 * base,
        zu12=0.2 * base,
    )
    return SimpleNamespace(
        jac=jac,
        bsupu=0.12 * base,
        bsupv=0.18 * base,
        bsubu=0.25 * base,
        bsubv=0.35 * base,
        bsubu_e=0.45 * base,
        bsubv_e=0.55 * base,
        bsubu_e_scaled=0.65 * base,
        bsubv_e_scaled=0.75 * base,
        bsubu_parity_even=0.25 * base,
        bsubu_parity_odd=0.05 * base,
        bsubv_parity_even=0.35 * base,
        bsubv_parity_odd=0.07 * base,
        bsubv_preblend=None,
        bsq=2.0 * base,
    )


def test_wout_minimal_constructor_threads_synthetic_physics_diagnostics(monkeypatch, tmp_path):
    ns = 3
    state = _axisym_state(ns)
    modes = vmec_mode_table(2, 0)
    static = SimpleNamespace(
        s=np.asarray([0.0, 0.5, 1.0]),
        modes=modes,
        cfg=SimpleNamespace(ns=ns, mpol=2, ntor=0, nfp=1, ntheta=4, nzeta=1, lasym=False, lconm1=False),
    )
    indata = InData(
        scalars={"NCURR": 0, "GAMMA": 0.0, "LBSUBS": True, "AC": 1.25, "PCURR_TYPE": "power_series"},
        indexed={},
        source_path=None,
    )
    trig = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=3, nmax=0, lasym=False, cache=False)
    bc = _fake_bcovar(ns, trig)
    mercier_calls: list[dict[str, object]] = []

    monkeypatch.setenv("VMEC_JAX_WOUT_FAST_BCOVAR", "1")
    monkeypatch.setenv("VMEC_JAX_SKIP_BSUB_FILTER", "1")
    monkeypatch.setattr(
        "vmec_jax.energy.flux_profiles_from_indata",
        lambda *_args, **_kwargs: SimpleNamespace(
            phipf=np.asarray([0.0, 1.0, 1.2]),
            phips=np.asarray([0.0, 0.8, 1.0]),
            chipf=np.asarray([0.0, 0.2, 0.4]),
        ),
    )
    monkeypatch.setattr(
        "vmec_jax.profiles.eval_profiles",
        lambda *_args, **_kwargs: {
            "pressure": np.asarray([9.0, 0.6, 0.2]),
            "iota": np.asarray([0.0, 0.31, 0.43]),
        },
    )
    monkeypatch.setattr(
        "vmec_jax.boundary.boundary_from_indata",
        lambda *_args, **_kwargs: BoundaryCoeffs(
            R_cos=np.asarray([2.0, 0.0]),
            R_sin=np.zeros(2),
            Z_cos=np.zeros(2),
            Z_sin=np.zeros(2),
        ),
    )
    monkeypatch.setattr("vmec_jax.vmec_bcovar.vmec_bcovar_half_mesh_from_wout", lambda **_kwargs: bc)
    monkeypatch.setattr(
        "vmec_jax.vmec_residue.vmec_force_norms_from_bcovar_dynamic",
        lambda **_kwargs: SimpleNamespace(
            vp=np.asarray([0.0, 1.5, 2.0]),
            wb=np.asarray(4.0),
            wp=np.asarray(1.0),
            volume=np.asarray(3.0),
        ),
    )
    monkeypatch.setattr(wout_module, "_compute_bsubs_half_mesh", lambda **_kwargs: np.zeros_like(bc.bsubu))
    monkeypatch.setattr(
        wout_module,
        "_compute_equif_wout",
        lambda **_kwargs: (
            np.asarray([0.0, 0.1, 0.2]),
            np.asarray([0.0, 0.3, 0.4]),
            np.asarray([0.0, 0.5, 0.6]),
            np.asarray([0.0, 0.7, 0.8]),
            np.asarray([0.0, 0.9, 1.0]),
        ),
    )
    monkeypatch.setattr(
        wout_module,
        "_compute_eqfor_beta",
        lambda **_kwargs: (0.11, 0.22, 0.33, 0.44),
    )
    monkeypatch.setattr(wout_module, "_compute_eqfor_betaxis", lambda **_kwargs: 0.44)

    def fake_mercier(**kwargs):
        mercier_calls.append(kwargs)
        return tuple(np.full(ns, i + 1.0) for i in range(8))

    monkeypatch.setattr(wout_module, "_compute_mercier", fake_mercier)

    generated = wout_module.wout_minimal_from_fixed_boundary(
        path=tmp_path / "wout_synthetic.nc",
        state=state,
        static=static,
        indata=indata,
        signgs=1,
        fsqr=1.0e-3,
        fsqz=2.0e-3,
        fsql=3.0e-3,
        converged=False,
    )

    assert generated.path == tmp_path / "wout_synthetic.nc"
    assert generated.mnmax == modes.K
    assert generated.mnmax_nyq > 0
    assert generated.fsqt.shape == (100,)
    assert generated.ac[0] == 1.25
    assert generated.pcurr_type == "power_series"
    assert generated.piota_type == "power_series"
    assert generated.betatotal == 0.33
    assert generated.betapol == 0.11
    assert generated.betator == 0.22
    np.testing.assert_allclose(generated.DMerc, np.ones(ns))
    np.testing.assert_allclose(generated.bdotgradv, np.full(ns, 8.0))
    assert mercier_calls and mercier_calls[0]["lbsubs"] is True
    np.testing.assert_allclose(np.asarray(mercier_calls[0]["pres"]), [0.0, 0.6, 0.2])


def test_compute_mercier_lasym_lbsubs_branch_with_reduced_bsub_inputs(monkeypatch):
    ns = 3
    modes = ModeTable(m=np.asarray([0, 1], dtype=int), n=np.asarray([0, 0], dtype=int))
    layout = StateLayout(ns=ns, K=2, lasym=True)
    radial = np.linspace(0.0, 1.0, ns)[:, None]
    state = VMECState(
        layout=layout,
        Rcos=np.concatenate([2.0 + 0.0 * radial, 0.12 + 0.0 * radial], axis=1),
        Rsin=np.concatenate([0.0 * radial, 0.03 + 0.0 * radial], axis=1),
        Zcos=np.concatenate([0.0 * radial, -0.02 + 0.0 * radial], axis=1),
        Zsin=np.concatenate([0.0 * radial, 0.25 + 0.0 * radial], axis=1),
        Lcos=np.zeros((ns, 2)),
        Lsin=np.zeros((ns, 2)),
    )
    trig = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=1, nmax=0, lasym=True, cache=False)
    nt2 = int(trig.ntheta2)
    nt3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    reduced_shape = (ns, nt2, nzeta)
    full_shape = (ns, nt3, nzeta)
    geom = {
        "R": 2.0 * np.ones(full_shape),
        "Z": 0.1 * np.ones(full_shape),
        "Ru": 0.08 * np.ones(full_shape),
        "Zu": 0.22 * np.ones(full_shape),
        "Rv": np.zeros(full_shape),
        "Zv": np.zeros(full_shape),
    }
    bsubs_seed = 0.01 * np.arange(np.prod(full_shape), dtype=float).reshape(full_shape)
    correction_seen = {"called": False}

    monkeypatch.setenv("VMEC_JAX_MERCIER_LASYM_FILTER", "0")
    monkeypatch.setattr(wout_module, "_compute_bsubs_half_mesh", lambda **_kwargs: bsubs_seed.copy())

    def fake_lasym_correction(**kwargs):
        correction_seen["called"] = True
        assert kwargs["bsubu"].shape == full_shape
        assert kwargs["bsubv"].shape == full_shape
        return kwargs["bsubs"] + 0.02, kwargs["bsubsu"] + 0.03, kwargs["bsubsv"] + 0.04

    monkeypatch.setattr(wout_module, "_jxbforce_apply_bsubs_correction_lasym_true", fake_lasym_correction)

    outputs = wout_module._compute_mercier(
        state=state,
        geom_modes=modes,
        s=np.asarray([0.0, 0.5, 1.0]),
        lconm1=False,
        lthreed=False,
        lasym=True,
        nfp=1,
        lbsubs=True,
        mmax_force=1,
        nmax_force=0,
        pres=np.asarray([0.0, 0.2, 0.1]),
        vp=np.asarray([1.0, 1.3, 1.7]),
        phips=np.asarray([0.0, 0.8, 1.0]),
        iotas=np.asarray([0.0, 0.22, 0.35]),
        bsq=2.0 * np.ones(full_shape),
        sqrtg=np.ones(full_shape),
        bsubu=0.4 * np.ones(reduced_shape),
        bsubv=0.5 * np.ones(reduced_shape),
        bsupu=0.1 * np.ones(full_shape),
        bsupv=0.2 * np.ones(full_shape),
        trig=trig,
        geom=geom,
        jac_half=SimpleNamespace(),
        signgs=-1,
    )

    assert correction_seen["called"] is True
    assert len(outputs) == 8
    assert all(arr.shape == (ns,) for arr in outputs)
    assert all(np.all(np.isfinite(arr)) for arr in outputs)
    assert np.any(np.asarray(outputs[5]) != 0.0)


def test_compute_mercier_short_mesh_returns_all_zero_component_profiles():
    ns = 2
    modes = ModeTable(m=np.asarray([0], dtype=int), n=np.asarray([0], dtype=int))
    layout = StateLayout(ns=ns, K=1, lasym=False)
    state = VMECState(
        layout=layout,
        Rcos=np.ones((ns, 1)),
        Rsin=np.zeros((ns, 1)),
        Zcos=np.zeros((ns, 1)),
        Zsin=np.zeros((ns, 1)),
        Lcos=np.zeros((ns, 1)),
        Lsin=np.zeros((ns, 1)),
    )
    trig = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=0, nmax=0, lasym=False, cache=False)
    shape = (ns, int(trig.ntheta2), int(np.asarray(trig.cosnv).shape[0]))

    outputs = wout_module._compute_mercier(
        state=state,
        geom_modes=modes,
        s=np.asarray([0.0, 1.0]),
        lconm1=False,
        lthreed=False,
        lasym=False,
        nfp=1,
        lbsubs=False,
        mmax_force=0,
        nmax_force=0,
        pres=np.asarray([0.0, 0.1]),
        vp=np.asarray([1.0, 1.2]),
        phips=np.asarray([0.0, 1.0]),
        iotas=np.asarray([0.0, 0.3]),
        bsq=np.ones(shape),
        sqrtg=np.ones(shape),
        bsubu=np.ones(shape),
        bsubv=np.ones(shape),
        bsupu=np.ones(shape),
        bsupv=np.ones(shape),
        trig=trig,
        geom={},
    )

    assert len(outputs) == 8
    assert all(arr.shape == (ns,) for arr in outputs)
    for profile in outputs:
        np.testing.assert_allclose(profile, 0.0, atol=0.0)


def test_finite_beta_zero_field_scalars_and_bcovar_tree_roundtrip(monkeypatch):
    static = SimpleNamespace(s=jnp.asarray([0.0]), trig_vmec=object())
    monkeypatch.setattr(finite_beta, "equilibrium_aspect_ratio_from_state", lambda **_kwargs: jnp.asarray(4.0))
    monkeypatch.setattr(
        finite_beta,
        "equilibrium_iota_profiles_from_state",
        lambda **_kwargs: (jnp.asarray([0.0]), jnp.asarray([0.0]), jnp.asarray([0.0])),
    )
    monkeypatch.setattr(finite_beta, "_wout_like_for_state", lambda **_kwargs: (SimpleNamespace(), jnp.asarray([0.0])))
    monkeypatch.setattr(finite_beta, "vmec_bcovar_half_mesh_from_wout", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        finite_beta,
        "vmec_force_norms_from_bcovar_dynamic",
        lambda **_kwargs: SimpleNamespace(
            wb=jnp.asarray(0.0),
            wp=jnp.asarray(3.0),
            volume=jnp.asarray(0.0),
            vp=jnp.asarray([0.0]),
        ),
    )

    scalars = finite_beta.finite_beta_scalars_from_state(
        state=object(),
        static=static,
        indata=object(),
        signgs=1,
    )

    assert float(scalars["mean_iota"]) == 0.0
    assert float(scalars["min_iota"]) == 0.0
    assert float(scalars["max_iota"]) == 0.0
    assert float(scalars["betatotal"]) == 0.0
    assert float(scalars["volavgB"]) == 0.0

    arr = jnp.arange(3.0).reshape(1, 3, 1)
    jac = VmecHalfMeshJacobian(
        r12=arr,
        rs=arr + 1.0,
        zs=arr + 2.0,
        ru12=arr + 3.0,
        zu12=arr + 4.0,
        tau=arr + 5.0,
        sqrtg=arr + 6.0,
    )
    payload = {
        field.name: (jac if field.name == "jac" else (jnp.asarray(2.0) if field.name == "lamscale" else arr))
        for field in fields(VmecHalfMeshBcovar)
    }
    bc = VmecHalfMeshBcovar(**payload)
    children, aux = bc.tree_flatten()
    restored = VmecHalfMeshBcovar.tree_unflatten(aux, children)

    assert len(children) == len(fields(VmecHalfMeshBcovar))
    assert restored.jac is jac
    np.testing.assert_allclose(np.asarray(restored.guu), np.asarray(arr))
    assert float(np.asarray(restored.lamscale)) == 2.0
