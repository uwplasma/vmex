from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import vmec_jax.static as static_mod
from vmec_jax.config import FreeBoundaryConfig, VMECConfig, config_from_indata, load_config
from vmec_jax.namelist import InData


def _static_cfg(*, ns: int = 3, ntor: int = 0, lfreeb: bool = False) -> VMECConfig:
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


def test_config_from_indata_scalar_extcur_and_quoted_mgrid() -> None:
    indata = InData(
        scalars={
            "MPOL": 2,
            "NTOR": 0,
            "NS_ARRAY": 5,
            "NFP": 3,
            "LFREEB": True,
            "MGRID_FILE": "'mgrid_rel.nc'",
            "EXTCUR": 3.5,
            "NVACSKIP": 0,
        },
        indexed={},
    )

    cfg = config_from_indata(indata)

    assert cfg.lfreeb is True
    assert cfg.mgrid_file == "mgrid_rel.nc"
    assert cfg.extcur == (3.5,)
    assert cfg.nvacskip == 3


def test_config_from_indata_indexed_extcur_ignores_invalid_indices() -> None:
    indata = InData(
        scalars={
            "MPOL": 2,
            "NTOR": 0,
            "NS_ARRAY": 5,
            "NFP": 1,
            "LFREEB": True,
            "MGRID_FILE": "mgrid.nc",
        },
        indexed={
            "EXTCUR": {
                (0,): 99.0,
                (2,): 4.0,
                (2, 1): 88.0,
            }
        },
    )

    cfg = config_from_indata(indata)

    assert cfg.extcur == (0.0, 4.0)


def test_load_config_resolves_relative_mgrid_paths(tmp_path: Path) -> None:
    input_path = tmp_path / "input.freeb"
    input_path.write_text(
        """
&INDATA
  MPOL = 2
  NTOR = 0
  NS_ARRAY = 5
  NFP = 1
  LFREEB = T
  MGRID_FILE = 'subdir/mgrid.nc'
/
"""
    )

    cfg, _indata = load_config(input_path)

    assert cfg.lfreeb is True
    assert cfg.mgrid_file == str((tmp_path / "subdir" / "mgrid.nc").resolve())


def test_build_static_single_surface_and_free_boundary_runtime_state(monkeypatch) -> None:
    sentinel_state = SimpleNamespace(kind="freeb")
    monkeypatch.setattr(static_mod, "initial_free_boundary_state", lambda cfg: sentinel_state)

    out = static_mod.build_static(
        _static_cfg(ns=1, lfreeb=True),
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

    monkeypatch.setattr("vmec_jax.kernels.tomnsp.vmec_trig_tables", fail_trig)

    out = static_mod.build_static(_static_cfg(ns=3))

    assert out.trig_vmec is None
    assert out.tomnsps_masks is None
    assert out.tomnsps_masks_edge is None
    assert out.mode_scale_internal is None


def test_build_static_signed_maps_failure_keeps_mode_arrays(monkeypatch) -> None:
    def fail_signed_maps(_modes):
        raise RuntimeError("synthetic signed-map failure")

    monkeypatch.setattr("vmec_jax.kernels.parity.signed_maps_from_modes", fail_signed_maps)

    out = static_mod.build_static(_static_cfg(ns=4, ntor=1))

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

    monkeypatch.setattr("vmec_jax.kernels.parity.signed_maps_from_modes", sparse_signed_maps)

    out = static_mod.build_static(_static_cfg(ns=4, ntor=1))

    assert out.signed_maps is not None
    assert (0, 1) not in set(zip(np.asarray(out.mn_idx_m), np.asarray(out.mn_idx_n)))
    assert (0, 0) in set(zip(np.asarray(out.mn_idx_m), np.asarray(out.mn_idx_n)))


def test_build_static_caches_mode_transform_host_projection() -> None:
    out = static_mod.build_static(_static_cfg(ns=4, ntor=1))

    assert out.mode_transform_host_projection is not None
    assert out.mode_transform_host_projection.ncoeff == out.modes.m.size


def test_build_static_can_disable_vmec_phase_cache(monkeypatch) -> None:
    monkeypatch.setenv("VMEC_JAX_CACHE_VMEC_PHASE", "0")

    out = static_mod.build_static(_static_cfg(ns=3, ntor=1))

    assert out.trig_vmec is not None
    assert getattr(out.trig_vmec, "phase_stack", None) is None
    assert out.tomnsps_masks is not None
    assert out.tomnsps_masks_edge is not None


def test_build_static_phase_cache_failure_keeps_trig_tables(monkeypatch) -> None:
    class PhaseCacheNumpyProxy:
        def __getattr__(self, name):
            return getattr(np, name)

        def concatenate(self, *_args, **_kwargs):
            raise RuntimeError("synthetic phase-cache failure")

    monkeypatch.setattr(static_mod, "np", PhaseCacheNumpyProxy())

    out = static_mod.build_static(_static_cfg(ns=3, ntor=1))

    assert out.trig_vmec is not None
    assert getattr(out.trig_vmec, "phase_stack", None) is None
    assert getattr(out.trig_vmec, "phase_dtheta_stack", None) is None
    assert getattr(out.trig_vmec, "phase_dzeta_stack", None) is None
    assert out.tomnsps_masks is not None
    assert out.tomnsps_masks_edge is not None


def test_build_static_populates_vmec_phase_cache_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VMEC_JAX_CACHE_VMEC_PHASE", raising=False)

    out = static_mod.build_static(_static_cfg(ns=3, ntor=1))

    assert out.trig_vmec is not None
    assert out.trig_vmec.phase_stack is not None
    assert out.trig_vmec.phase_dtheta_stack is not None
    assert out.trig_vmec.phase_dzeta_stack is not None
    assert out.trig_vmec.phase_stack.shape[0] == 2 * out.modes.m.size
    assert out.trig_vmec.phase_stack_m is out.modes.m
    assert out.trig_vmec.phase_stack_n is out.modes.n
