from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import vmec_jax.static as static_mod
from vmec_jax.config import FreeBoundaryConfig, VMECConfig


def _cfg(*, ns: int = 3, ntor: int = 0, lfreeb: bool = False) -> VMECConfig:
    return VMECConfig(
        mpol=2,
        ntor=int(ntor),
        ns=int(ns),
        nfp=1,
        lasym=False,
        lthreed=bool(ntor),
        lconm1=True,
        ntheta=4,
        nzeta=2 if ntor else 1,
        free_boundary=FreeBoundaryConfig(enabled=bool(lfreeb), mgrid_file="mgrid.nc" if lfreeb else "NONE"),
    )


def test_build_static_single_surface_and_free_boundary_runtime_state(monkeypatch) -> None:
    sentinel_state = SimpleNamespace(kind="freeb")
    monkeypatch.setattr(static_mod, "initial_free_boundary_state", lambda cfg: sentinel_state)

    out = static_mod.build_static(
        _cfg(ns=1, lfreeb=True),
        mgrid_metadata=SimpleNamespace(name="mgrid"),
        free_boundary_extcur=(1.0, 2.0),
    )

    np.testing.assert_allclose(out.s, [0.0])
    assert out.free_boundary_state0 is sentinel_state
    assert out.mgrid_metadata.name == "mgrid"
    assert out.free_boundary_extcur == (1.0, 2.0)


def test_build_static_tomnsp_import_failure_keeps_fallback_fields(monkeypatch) -> None:
    def fail_trig(**_kwargs):
        raise RuntimeError("synthetic trig failure")

    monkeypatch.setattr("vmec_jax.vmec_tomnsp.vmec_trig_tables", fail_trig)

    out = static_mod.build_static(_cfg(ns=3))

    assert out.trig_vmec is None
    assert out.tomnsps_masks is None
    assert out.tomnsps_masks_edge is None
    assert out.mode_scale_internal is None


def test_build_static_signed_maps_failure_keeps_mode_arrays(monkeypatch) -> None:
    def fail_signed_maps(_modes):
        raise RuntimeError("synthetic signed-map failure")

    monkeypatch.setattr("vmec_jax.vmec_parity.signed_maps_from_modes", fail_signed_maps)

    out = static_mod.build_static(_cfg(ns=4, ntor=1))

    assert out.signed_maps is None
    assert out.mn_idx_m is None
    assert out.mn_idx_n is None
    assert out.mn_idx_kp is None
    assert out.mn_idx_kn is None
    assert out.mn_has_kn is None
    assert out.m_np.shape == out.n_np.shape
    assert out.lambda_axis_copy_mask.shape == out.m_np.shape


def test_build_static_signed_maps_skip_missing_positive_modes(monkeypatch) -> None:
    def sparse_signed_maps(_modes):
        idx_pos = np.asarray([[0, -1], [1, 2]], dtype=np.int32)
        idx_neg = np.asarray([[0, -1], [1, -1]], dtype=np.int32)
        return SimpleNamespace(idx_pos=idx_pos, idx_neg=idx_neg)

    monkeypatch.setattr("vmec_jax.vmec_parity.signed_maps_from_modes", sparse_signed_maps)

    out = static_mod.build_static(_cfg(ns=4, ntor=1))

    assert out.signed_maps is not None
    assert (0, 1) not in set(zip(np.asarray(out.mn_idx_m), np.asarray(out.mn_idx_n)))
    assert (0, 0) in set(zip(np.asarray(out.mn_idx_m), np.asarray(out.mn_idx_n)))


def test_build_static_can_disable_vmec_phase_cache(monkeypatch) -> None:
    monkeypatch.setenv("VMEC_JAX_CACHE_VMEC_PHASE", "0")

    out = static_mod.build_static(_cfg(ns=3, ntor=1))

    assert out.trig_vmec is not None
    assert getattr(out.trig_vmec, "phase_stack", None) is None
    assert out.tomnsps_masks is not None
    assert out.tomnsps_masks_edge is not None


def test_build_static_phase_cache_failure_keeps_trig_tables(monkeypatch) -> None:
    original_concatenate = static_mod.np.concatenate

    def fail_concatenate(*_args, **_kwargs):
        raise RuntimeError("synthetic phase-cache failure")

    monkeypatch.setattr(static_mod.np, "concatenate", fail_concatenate)

    out = static_mod.build_static(_cfg(ns=3, ntor=1))

    monkeypatch.setattr(static_mod.np, "concatenate", original_concatenate)
    assert out.trig_vmec is not None
    assert getattr(out.trig_vmec, "phase_stack", None) is None
    assert getattr(out.trig_vmec, "phase_dtheta_stack", None) is None
    assert getattr(out.trig_vmec, "phase_dzeta_stack", None) is None
    assert out.tomnsps_masks is not None
    assert out.tomnsps_masks_edge is not None


def test_build_static_populates_vmec_phase_cache_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VMEC_JAX_CACHE_VMEC_PHASE", raising=False)

    out = static_mod.build_static(_cfg(ns=3, ntor=1))

    assert out.trig_vmec is not None
    assert out.trig_vmec.phase_stack is not None
    assert out.trig_vmec.phase_dtheta_stack is not None
    assert out.trig_vmec.phase_dzeta_stack is not None
    assert out.trig_vmec.phase_stack.shape[0] == 2 * out.modes.m.size
    assert out.trig_vmec.phase_stack_m is out.modes.m
    assert out.trig_vmec.phase_stack_n is out.modes.n
