from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import jnp
import vmec_jax.solve as solve_mod
from vmec_jax.field import TWOPI
from vmec_jax.state import StateLayout, VMECState


def test_wout_like_vmec_forces_pytree_preserves_static_metadata_and_dynamic_profiles():
    jax = pytest.importorskip("jax")

    profiles = {
        "phipf": jnp.asarray([1.0, 2.0, 3.0]),
        "phips": jnp.asarray([4.0, 5.0, 6.0]),
        "chipf": jnp.asarray([0.0, 1.0, 3.0]),
        "pres": jnp.asarray([7.0, 8.0, 9.0]),
        "mass": jnp.asarray([0.0, 10.0, 11.0]),
        "icurv": jnp.asarray([0.0, 12.0, 13.0]),
        "phipf_internal": jnp.asarray([0.5, 1.0, 1.5]),
        "chipf_internal": jnp.asarray([0.0, 0.5, 1.5]),
        "chips_eff": jnp.asarray([0.0, 0.25, 1.0]),
    }
    forces = solve_mod._WoutLikeVmecForces(
        nfp=np.int64(5),
        mpol=np.int64(3),
        ntor=np.int64(2),
        lasym=np.bool_(True),
        signgs=np.int64(-1),
        gamma=np.float64(5.0 / 3.0),
        ncurr=np.int64(1),
        lcurrent=np.bool_(False),
        flux_is_internal=np.bool_(False),
        **profiles,
    )

    children, aux = forces.tree_flatten()
    rebuilt = solve_mod._WoutLikeVmecForces.tree_unflatten(aux, children)
    leaves, treedef = jax.tree_util.tree_flatten(forces)
    roundtrip = jax.tree_util.tree_unflatten(treedef, leaves)

    assert aux == (5, 3, 2, True, -1, 5.0 / 3.0, 1, False, False)
    assert rebuilt.nfp == 5
    assert rebuilt.lasym
    assert not rebuilt.lcurrent
    assert not rebuilt.flux_is_internal
    np.testing.assert_allclose(np.asarray(roundtrip.phipf), profiles["phipf"])
    np.testing.assert_allclose(np.asarray(roundtrip.chips_eff), profiles["chips_eff"])


def test_flux_profile_external_iotas_fallback_uses_internal_phip_and_sign():
    phipf_physical = -TWOPI * np.asarray([1.0, 2.0, 4.0])
    phipf_internal, chipf_internal, chips_eff = solve_mod._vmec_force_flux_profiles(
        phipf=phipf_physical,
        chipf=None,
        signgs=-1,
        flux_is_internal=False,
        iotas=np.asarray([0.25, 0.5, 0.75]),
    )

    np.testing.assert_allclose(np.asarray(phipf_internal), [1.0, 2.0, 4.0])
    assert chipf_internal is None
    np.testing.assert_allclose(np.asarray(chips_eff), [0.25, 1.0, 3.0])


def test_profile_helpers_cover_missing_pressure_phips_mass_and_zero_edge_current(monkeypatch):
    import vmec_jax.profiles as profiles

    calls = []

    def fake_eval_profiles(_indata, s):
        s_arr = jnp.asarray(s)
        calls.append(np.asarray(s_arr))
        if int(s_arr.shape[0]) == 1:
            return {"current": jnp.asarray([0.0], dtype=s_arr.dtype)}
        return {"current": 2.0 * s_arr}

    monkeypatch.setattr(profiles, "eval_profiles", fake_eval_profiles)
    indata = SimpleNamespace(
        get_int=lambda name, default=0: 1 if name == "NCURR" else default,
        get_float=lambda name, default=0.0: 5.0 if name == "CURTOR" else default,
    )
    s_full = jnp.asarray([0.0, 0.25, 1.0])

    pressure = solve_mod._pressure_half_mesh_from_indata(indata=indata, s_full=s_full)
    mass = solve_mod._mass_half_mesh_from_indata(
        indata=indata,
        s_full=s_full,
        phips=jnp.asarray([2.0, -3.0, 4.0]),
        chips=None,
        r00=2.0,
        gamma=2.0,
        lrfp=True,
    )
    icurv = solve_mod._icurv_full_mesh_from_indata(indata=indata, s_full=s_full, signgs=1)

    np.testing.assert_allclose(np.asarray(pressure), 0.0)
    np.testing.assert_allclose(np.asarray(mass), 0.0)
    np.testing.assert_allclose(np.asarray(icurv), 0.0)
    np.testing.assert_allclose(calls[0], [0.0, 0.125, 0.625])
    np.testing.assert_allclose(calls[-1], [1.0])


def test_sample_free_boundary_external_field_forwards_default_plascur(monkeypatch):
    import vmec_jax.free_boundary as free_boundary

    seen = {}

    def fake_sample_external_vacuum_diagnostics(*, state, static, plascur):
        seen["state"] = state
        seen["static"] = static
        seen["plascur"] = plascur
        return {
            "enabled": True,
            "available": True,
            "plascur": plascur,
            "bnormal_rms": 0.125,
        }

    monkeypatch.setattr(
        free_boundary,
        "sample_external_vacuum_diagnostics",
        fake_sample_external_vacuum_diagnostics,
    )
    layout = StateLayout(ns=1, K=1, lasym=False)
    state = VMECState(
        layout=layout,
        Rcos=np.asarray([[1.0]]),
        Rsin=np.asarray([[0.0]]),
        Zcos=np.asarray([[0.0]]),
        Zsin=np.asarray([[0.0]]),
        Lcos=np.asarray([[0.0]]),
        Lsin=np.asarray([[0.0]]),
    )
    static = SimpleNamespace(free_boundary_plascur=np.asarray(3.5))

    diagnostics = solve_mod._sample_free_boundary_external_field(state=state, static=static)

    assert diagnostics["enabled"]
    assert diagnostics["available"]
    assert diagnostics["bnormal_rms"] == 0.125
    assert seen["state"] is state
    assert seen["static"] is static
    assert seen["plascur"] == 3.5


def test_hlo_dump_label_specific_env_writes_once(tmp_path, monkeypatch):
    jnp_local = pytest.importorskip("jax.numpy")

    solve_mod._HLO_DUMPED_KEYS.clear()
    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_TINY_LABEL", "1")
    static = SimpleNamespace(cfg=SimpleNamespace(ns=2, ntheta=4))
    wout_like = SimpleNamespace(mpol=1, ntor=0, nfp=1, lasym=False)

    solve_mod._maybe_dump_hlo_kernel(
        label="tiny_label",
        fn=lambda x: x * 2.0,
        args=(jnp_local.asarray([1.0]),),
        kwargs={},
        static=static,
        wout_like=wout_like,
    )
    solve_mod._maybe_dump_hlo_kernel(
        label="tiny_label",
        fn=lambda x: x * 3.0,
        args=(jnp_local.asarray([1.0]),),
        kwargs={},
        static=static,
        wout_like=wout_like,
    )

    hlo_paths = list(tmp_path.glob("hlo_tiny_label_ns2_mpol1_ntor0.txt"))
    assert len(hlo_paths) == 1
    assert hlo_paths[0].read_text()
