from __future__ import annotations

from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.booz_input import booz_xform_inputs_from_state
from vmec_jax._compat import enable_x64
from vmec_jax.driver import example_paths
from vmec_jax.profiles import eval_profiles
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout
from vmec_jax.config import load_config


BOOZ_INPUT_PARITY_CASES = (
    pytest.param(
        "axisym_pressure",
        "input.shaped_tokamak_pressure",
        "wout_shaped_tokamak_pressure.nc",
        {
            "bmnc": (2.0e-6, 2.0e-5),
            "bsubumnc": (5.0e-12, 1.0e-12),
            "bsubvmnc": (2.0e-5, 3.0e-3),
        },
        id="axisym_pressure",
    ),
    pytest.param(
        "qa",
        "input.LandremanPaul2021_QA_lowres",
        "wout_LandremanPaul2021_QA_lowres.nc",
        {
            "bmnc": (1.0e-5, 1.0e-5),
            "bsubumnc": (5.0e-12, 1.0e-12),
            "bsubvmnc": (1.0e-5, 3.0e-5),
        },
        id="qa",
    ),
    pytest.param(
        "qh",
        "input.nfp4_QH_warm_start",
        "wout_nfp4_QH_warm_start.nc",
        {
            "bmnc": (8.0e-4, 1.0e-3),
            "bsubumnc": (5.0e-12, 1.0e-12),
            "bsubvmnc": (3.0e-5, 2.0e-4),
        },
        id="qh",
    ),
)


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data"


def _relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    scale = max(float(np.sqrt(np.mean(np.asarray(b, dtype=float) ** 2))), 1.0e-30)
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)) / scale)


def _assert_spectral_field_close(
    *,
    case_name: str,
    field_name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    rel_rms_limit: float,
    max_abs_limit: float,
) -> None:
    np.testing.assert_equal(actual.shape, expected.shape)
    rel_rms = _relative_rms(actual, expected)
    max_abs = float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))
    errors = {"rel_rms": rel_rms, "max_abs": max_abs}
    assert rel_rms < rel_rms_limit and max_abs < max_abs_limit, f"{case_name}.{field_name}: {errors}"


def test_booz_xform_inputs_from_state_shapes():
    pytest.importorskip("netCDF4")

    input_path, wout_path = example_paths("circular_tokamak")
    if wout_path is None:
        pytest.skip("No reference wout file available for circular_tokamak")

    cfg, indata = load_config(str(input_path))
    static = build_static(cfg)
    wout = read_wout(wout_path)
    state = state_from_wout(wout)

    inputs = booz_xform_inputs_from_state(state=state, static=static, indata=indata, signgs=wout.signgs)

    assert inputs.rmnc.shape[0] == cfg.ns - 1
    assert inputs.zmns.shape[0] == cfg.ns - 1
    assert inputs.lmns.shape[0] == cfg.ns - 1
    assert inputs.bmnc.shape[0] == cfg.ns - 1
    assert inputs.bsubumnc.shape[0] == cfg.ns - 1
    assert inputs.bsubvmnc.shape[0] == cfg.ns - 1
    assert inputs.iota.shape[0] == cfg.ns - 1
    assert inputs.rmns is None
    assert inputs.zmnc is None
    assert inputs.lmnc is None


def test_booz_xform_inputs_merge_partial_profile_overrides():
    pytest.importorskip("netCDF4")

    import numpy as np

    input_path, wout_path = example_paths("circular_tokamak")
    if wout_path is None:
        pytest.skip("No reference wout file available for circular_tokamak")

    cfg, indata = load_config(str(input_path))
    static = build_static(cfg)
    wout = read_wout(wout_path)
    state = state_from_wout(wout)

    s_full = np.asarray(static.s)
    s_half = np.concatenate([s_full[:1], 0.5 * (s_full[1:] + s_full[:-1])], axis=0)
    pressure = eval_profiles(indata, s_half)["pressure"]
    default = booz_xform_inputs_from_state(state=state, static=static, indata=indata, signgs=wout.signgs)
    overridden = booz_xform_inputs_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=wout.signgs,
        profiles_half={"pressure": pressure},
    )

    np.testing.assert_allclose(np.asarray(overridden.iota), np.asarray(default.iota), rtol=0.0, atol=0.0)
    assert np.linalg.norm(np.asarray(overridden.iota)) > 0.0


def test_booz_xform_inputs_from_state_jit_tracer_safe():
    pytest.importorskip("netCDF4")
    pytest.importorskip("jax")

    from vmec_jax._compat import jax, jnp

    enable_x64(True)

    input_path, wout_path = example_paths("circular_tokamak")
    if wout_path is None:
        pytest.skip("No reference wout file available for circular_tokamak")

    cfg, indata = load_config(str(input_path))
    static = build_static(cfg)
    wout = read_wout(wout_path)
    state = state_from_wout(wout)

    @jax.jit
    def _bmnc_from_rcos(rcos):
        traced_state = dc_replace(state, Rcos=rcos)
        return booz_xform_inputs_from_state(
            state=traced_state,
            static=static,
            indata=indata,
            signgs=wout.signgs,
        ).bmnc

    bmnc = _bmnc_from_rcos(jnp.asarray(state.Rcos))
    assert bmnc.shape[0] == cfg.ns - 1


def test_booz_xform_inputs_lasym_exports_asymmetric_geometry_channels():
    """LASYM Boozer pipelines must pass geometry sine/cosine channels through."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    enable_x64(True)
    data_dir = _data_dir()
    cfg, indata = load_config(str(data_dir / "input.basic_non_stellsym_simsopt"))
    wout = read_wout(data_dir / "wout_basic_non_stellsym_simsopt.nc")
    cfg = dc_replace(
        cfg,
        ns=int(wout.ns),
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        lthreed=bool(int(wout.ntor) > 0),
    )
    static = build_static(cfg)

    inputs = booz_xform_inputs_from_state(
        state=state_from_wout(wout),
        static=static,
        indata=indata,
        signgs=wout.signgs,
        use_nyq_from_grid=True,
    )

    assert bool(wout.lasym) is True
    np.testing.assert_array_equal(np.asarray(inputs.xm), np.asarray(wout.xm))
    np.testing.assert_array_equal(np.asarray(inputs.xn), np.asarray(wout.xn))
    np.testing.assert_array_equal(np.asarray(inputs.xm_nyq), np.asarray(wout.xm_nyq))
    np.testing.assert_array_equal(np.asarray(inputs.xn_nyq), np.asarray(wout.xn_nyq))

    for field_name in ("rmns", "zmnc", "lmnc"):
        field = getattr(inputs, field_name)
        assert field is not None, f"{field_name} must be exported for LASYM Boozer runs"
        np.testing.assert_equal(np.asarray(field).shape, np.asarray(inputs.rmnc).shape)
        assert float(np.linalg.norm(np.asarray(field))) > 0.0

    # Lambda channels are stored on the same half-mesh convention consumed by
    # booz_xform_jax; exact parity here protects the LASYM Boozer-angle map.
    np.testing.assert_allclose(np.asarray(inputs.lmns), np.asarray(wout.lmns)[1:], rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(inputs.lmnc), np.asarray(wout.lmnc)[1:], rtol=1.0e-13, atol=1.0e-13)

    for field_name in ("bmns", "bsubumns", "bsubvmns"):
        field = getattr(inputs, field_name)
        assert field is not None, f"{field_name} must be exported for LASYM Boozer runs"
        np.testing.assert_equal(np.asarray(field).shape, np.asarray(inputs.bmnc).shape)
        assert float(np.linalg.norm(np.asarray(field))) > 0.0


def test_booz_xform_inputs_lasym_feed_asymmetric_boozer_geometry():
    """The optional LASYM channels must reach booz_xform_jax, not only exist."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    booz_xform_jax = pytest.importorskip("booz_xform_jax")

    enable_x64(True)
    data_dir = _data_dir()
    cfg, indata = load_config(str(data_dir / "input.basic_non_stellsym_simsopt"))
    wout = read_wout(data_dir / "wout_basic_non_stellsym_simsopt.nc")
    cfg = dc_replace(
        cfg,
        ns=int(wout.ns),
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        lthreed=bool(int(wout.ntor) > 0),
    )
    static = build_static(cfg)

    inputs = booz_xform_inputs_from_state(
        state=state_from_wout(wout),
        static=static,
        indata=indata,
        signgs=wout.signgs,
        use_nyq_from_grid=True,
    )
    surface_index = int(np.asarray(inputs.rmnc).shape[0]) - 1

    out = booz_xform_jax.booz_xform_jax(
        rmnc=inputs.rmnc,
        zmns=inputs.zmns,
        lmns=inputs.lmns,
        bmnc=inputs.bmnc,
        bsubumnc=inputs.bsubumnc,
        bsubvmnc=inputs.bsubvmnc,
        iota=inputs.iota,
        xm=inputs.xm,
        xn=inputs.xn,
        xm_nyq=inputs.xm_nyq,
        xn_nyq=inputs.xn_nyq,
        nfp=inputs.nfp,
        mboz=4,
        nboz=4,
        asym=True,
        rmns=inputs.rmns,
        zmnc=inputs.zmnc,
        lmnc=inputs.lmnc,
        bmns=inputs.bmns,
        bsubumns=inputs.bsubumns,
        bsubvmns=inputs.bsubvmns,
        surface_indices=[surface_index],
    )

    for field_name in ("rmns_b", "zmnc_b", "pmnc_b", "pmns_b", "bmns_b"):
        field = np.asarray(out[field_name])
        assert field.shape[0] == 1
        assert float(np.linalg.norm(field)) > 0.0


@pytest.mark.parametrize(
    ("case_name", "input_name", "wout_name", "field_tolerances"),
    BOOZ_INPUT_PARITY_CASES,
)
def test_booz_xform_inputs_match_bundled_vmec2000_spectral_fields(
    case_name: str,
    input_name: str,
    wout_name: str,
    field_tolerances: dict[str, tuple[float, float]],
) -> None:
    """Boozer inputs should preserve VMEC2000 half-mesh spectral field conventions."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    enable_x64(True)
    data_dir = _data_dir()
    cfg, indata = load_config(str(data_dir / input_name))
    wout = read_wout(data_dir / wout_name)
    cfg = dc_replace(
        cfg,
        ns=int(wout.ns),
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        lthreed=bool(int(wout.ntor) > 0),
        ntheta=2 * int(wout.mpol) + 6,
        nzeta=1 if int(wout.ntor) == 0 else 2 * int(wout.ntor) + 4,
    )
    static = build_static(cfg)

    inputs = booz_xform_inputs_from_state(
        state=state_from_wout(wout),
        static=static,
        indata=indata,
        signgs=wout.signgs,
        use_nyq_from_grid=False,
    )

    np.testing.assert_array_equal(np.asarray(inputs.xm), np.asarray(wout.xm))
    np.testing.assert_array_equal(np.asarray(inputs.xn), np.asarray(wout.xn))
    np.testing.assert_array_equal(np.asarray(inputs.xm_nyq), np.asarray(wout.xm_nyq))
    np.testing.assert_array_equal(np.asarray(inputs.xn_nyq), np.asarray(wout.xn_nyq))

    np.testing.assert_allclose(np.asarray(inputs.lmns), np.asarray(wout.lmns)[1:], rtol=5.0e-13, atol=5.0e-13)
    np.testing.assert_allclose(np.asarray(inputs.iota), np.asarray(wout.iotas)[1:], rtol=5.0e-12, atol=5.0e-12)
    assert inputs.bmns is None
    assert inputs.bsubumns is None
    assert inputs.bsubvmns is None

    for field_name, (rel_rms_limit, max_abs_limit) in field_tolerances.items():
        _assert_spectral_field_close(
            case_name=case_name,
            field_name=field_name,
            actual=np.asarray(getattr(inputs, field_name)),
            expected=np.asarray(getattr(wout, field_name))[1:],
            rel_rms_limit=rel_rms_limit,
            max_abs_limit=max_abs_limit,
        )
