from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.coords as coords_mod
import vmec_jax.fourier as fourier_mod
import vmec_jax.geom as geom_mod
import vmec_jax.nyquist as nyquist_mod
import vmec_jax.profiles as profiles_mod
from vmec_jax.geom import Geom
from vmec_jax.performance_hotspot_helpers import replay_timing_breakdown
from vmec_jax.solve_residual_objective_helpers import _sum_square_blocks, zero_edge_rz_force_block


def _duplicate_register(_cls):
    raise ValueError("Duplicate custom PyTreeDef type registration")


@pytest.mark.parametrize("module", [coords_mod, fourier_mod, geom_mod])
def test_duplicate_pytree_registration_helpers_return_class(monkeypatch, module) -> None:
    class LocalNode:
        pass

    monkeypatch.setattr(module, "_register_pytree_node_class", _duplicate_register)

    assert module.register_pytree_node_class(LocalNode) is LocalNode


def test_geom_alias_properties_cover_all_vmec_names() -> None:
    geom = Geom(
        R=0,
        Z=1,
        L=2,
        Rs=3,
        Zs=4,
        Ls=5,
        Rt=6,
        Zt=7,
        Lt=8,
        Rp=9,
        Zp=10,
        Lp=11,
        sqrtg=12,
        g_ss=13,
        g_st=14,
        g_sp=15,
        g_tt=16,
        g_tp=17,
        g_pp=18,
    )

    assert geom.R_s == 3
    assert geom.Z_s == 4
    assert geom.L_s == 5
    assert geom.R_theta == 6
    assert geom.Z_theta == 7
    assert geom.L_theta == 8
    assert geom.R_phi == 9
    assert geom.Z_phi == 10
    assert geom.L_phi == 11


def test_nyquist_cache_allowed_fallback_paths(monkeypatch) -> None:
    monkeypatch.setattr(nyquist_mod, "has_jax", lambda: False)
    assert nyquist_mod._cache_allowed()

    class BoomTraceContext:
        @staticmethod
        def is_top_level():
            raise RuntimeError("trace context unavailable")

    import jax

    monkeypatch.setattr(nyquist_mod, "has_jax", lambda: True)
    monkeypatch.setattr(jax.core, "trace_ctx", BoomTraceContext(), raising=False)
    assert not nyquist_mod._cache_allowed()


def test_zero_edge_rz_force_block_numpy_short_and_copy_paths() -> None:
    one_row = np.asarray([[1.0, 2.0]])
    assert zero_edge_rz_force_block(one_row) is one_row

    block = np.arange(6.0).reshape(3, 2)
    masked = zero_edge_rz_force_block(block)

    assert masked is not block
    np.testing.assert_allclose(masked[:-1], block[:-1])
    np.testing.assert_allclose(masked[-1], 0.0)
    np.testing.assert_allclose(block[-1], [4.0, 5.0])

    short_device_path = zero_edge_rz_force_block(one_row, preserve_numpy=False)
    np.testing.assert_allclose(np.asarray(short_device_path), one_row)


def test_sum_square_blocks_returns_zero_with_reference_dtype() -> None:
    frzl = SimpleNamespace(frcc=np.ones((2, 3), dtype=np.float32), empty=None)

    total = _sum_square_blocks(frzl, ("empty", "missing"), radial_stop=None)

    assert np.asarray(total).dtype == np.dtype(np.float32)
    assert float(np.asarray(total)) == 0.0


def test_replay_timing_breakdown_ignores_malformed_records() -> None:
    profile = {
        "jacobian_tape_replay": {"count": "bad", "wall_time_s": "bad"},
        "jacobian_tape_replay_dispatch": {"count": object(), "wall_time_s": object()},
        "jacobian_tape_replay_ready": {"count": 3, "wall_time_s": 0.25},
    }

    breakdown = replay_timing_breakdown(profile, prefix="jacobian")

    assert breakdown["total_s"] == pytest.approx(0.25)
    assert breakdown["dispatch_s"] == 0.0
    assert breakdown["ready_s"] == pytest.approx(0.25)
    assert breakdown["count"] == 3


def test_profile_scalar_conversion_fallbacks_and_power_series_pedestal(monkeypatch) -> None:
    sentinel = object()
    assert profiles_mod._as_float_list(sentinel) is sentinel

    object_array = np.asarray([object()], dtype=object)
    assert profiles_mod._as_float_list(object_array) is object_array

    class NumpyProxy:
        float64 = np.float64

        @staticmethod
        def asarray(*_args, **_kwargs):
            raise TypeError("force fallback")

    monkeypatch.setattr(profiles_mod, "np", NumpyProxy)
    s_aux, f_aux = profiles_mod._aux_profile_arrays(
        SimpleNamespace(get=lambda key, default=None: {"AC_AUX_S": [0.0, 1.0], "AC_AUX_F": [2.0, 3.0]}.get(key, default)),
        "AC",
    )
    np.testing.assert_allclose(np.asarray(s_aux), [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(f_aux), [2.0, 3.0])

    cfg = profiles_mod.ProfileInputs(
        pmass_type="power_series",
        piota_type="power_series",
        pcurr_type="power_series",
        am=profiles_mod.jnp.asarray([10.0, -4.0]),
        ai=profiles_mod.jnp.asarray([]),
        ac=profiles_mod.jnp.asarray([]),
        ac_aux_s=profiles_mod.jnp.asarray([]),
        ac_aux_f=profiles_mod.jnp.asarray([]),
        pres_scale=2.0,
        bloat=1.0,
        spres_ped=0.5,
        lrfp=False,
        ncurr=0,
    )
    pressure_pa = np.asarray(profiles_mod.eval_profiles(cfg, [0.25, 0.75])["pressure_pa"])
    np.testing.assert_allclose(pressure_pa, [18.0, 16.0])
