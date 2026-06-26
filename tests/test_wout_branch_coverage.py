from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.wout as wout_module
from vmec_jax.modes import ModeTable
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.kernels.tomnsp import vmec_trig_tables
from vmec_jax.wout import (
    _apply_nyquist_half_weight,
    _bsubuv_parity_from_state,
    _compute_bsubs_half_mesh,
    _filter_bsubuv_jxbforce_lasym_loop,
    _filter_bsubuv_jxbforce_loop,
    _filter_bsubuv_jxbforce_parity_loop,
    _jxbforce_apply_bsubs_correction_lasym_false,
    _jxbforce_apply_bsubs_correction_lasym_true,
    _jxbforce_bsubsu_bsubsv_loop,
    _jxbforce_filter_with_bsubs_derivs_loop,
    _jxbforce_getbsubs_coeffs_lasym_false,
    _jxbforce_getbsubs_coeffs_lasym_true,
    _jxbforce_nyquist_limits,
    _vmec_wrout_nyquist_lasym_loop,
)


def _state_and_modes(ns: int = 3) -> tuple[VMECState, ModeTable]:
    modes = ModeTable(m=np.asarray([0, 1], dtype=int), n=np.asarray([0, 0], dtype=int))
    layout = StateLayout(ns=ns, K=modes.K, lasym=False)
    radial = np.linspace(0.0, 1.0, ns)[:, None]
    Rcos = np.concatenate([1.0 + 0.2 * radial, 0.15 + 0.05 * radial], axis=1)
    Zsin = np.concatenate([0.2 + 0.1 * radial, 0.3 + 0.07 * radial], axis=1)
    zeros = np.zeros_like(Rcos)
    state = VMECState(layout=layout, Rcos=Rcos, Rsin=zeros.copy(), Zcos=zeros.copy(), Zsin=Zsin, Lcos=zeros, Lsin=zeros)
    return state, modes


def _grid_field(shape: tuple[int, int, int], offset: float, scale: float = 0.01) -> np.ndarray:
    return offset + scale * np.arange(int(np.prod(shape)), dtype=float).reshape(shape)


def test_compute_bsubs_half_mesh_covers_parity_force_jacobian_and_dump_paths(monkeypatch, tmp_path) -> None:
    state, modes = _state_and_modes()
    trig = vmec_trig_tables(ntheta=4, nzeta=2, nfp=1, mmax=1, nmax=0, lasym=False, cache=False)
    s = np.asarray([0.0, 0.25, 1.0])
    shape = (s.size, int(trig.ntheta2), int(np.asarray(trig.cosnv).shape[0]))
    bsupu = _grid_field(shape, 0.4)
    bsupv = _grid_field(shape, -0.2)

    parity_geom = {
        "pr1_even": _grid_field(shape, 1.0),
        "pr1_odd": _grid_field(shape, 0.2),
        "pz1_even": _grid_field(shape, -0.3),
        "pz1_odd": _grid_field(shape, 0.4),
        "pru_even": _grid_field(shape, 0.5),
        "pru_odd": _grid_field(shape, -0.1),
        "pzu_even": _grid_field(shape, 0.7),
        "pzu_odd": _grid_field(shape, 0.3),
        "prv_even": _grid_field(shape, -0.5),
        "prv_odd": _grid_field(shape, 0.6),
        "pzv_even": _grid_field(shape, 0.8),
        "pzv_odd": _grid_field(shape, -0.4),
    }

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_TAG", "tiny")
    monkeypatch.setenv("VMEC_JAX_DUMP_BSS_INPUTS", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_BSS_TERMS", "1")
    parity_out = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=modes,
        s=s,
        lconm1=False,
        lthreed=False,
        lasym=False,
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom=parity_geom,
    )
    assert parity_out.shape == shape
    assert np.all(np.isfinite(parity_out))
    assert (tmp_path / "bss_inputs_jax_tiny.dat").exists()
    with np.load(tmp_path / "bss_terms_jax_tiny.npz") as dump:
        assert dump["bsubs"].shape == shape
        assert dump["gsu"].shape == shape

    class BadR12:
        @property
        def r12(self):
            raise RuntimeError("synthetic missing r12")

    monkeypatch.delenv("VMEC_JAX_DUMP_BSS_TERMS", raising=False)
    monkeypatch.setenv("VMEC_JAX_DUMP_TAG", "bad_r12")
    bad_r12_out = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=modes,
        s=s,
        lconm1=False,
        lthreed=False,
        lasym=False,
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom=parity_geom,
        jac_half=BadR12(),
    )
    assert bad_r12_out.shape == shape
    assert (tmp_path / "bss_inputs_jax_bad_r12.dat").exists()

    monkeypatch.delenv("VMEC_JAX_DUMP_BSS_INPUTS", raising=False)

    force_out = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=modes,
        s=s,
        lconm1=False,
        lthreed=False,
        lasym=False,
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom={},
        force_rs=_grid_field(shape, 0.11),
        force_zs=_grid_field(shape, 0.13),
        force_ru12=_grid_field(shape, 0.17),
        force_zu12=_grid_field(shape, 0.19),
    )
    assert force_out.shape == shape
    assert float(np.linalg.norm(force_out)) > 0.0

    jac = SimpleNamespace(
        ru12=_grid_field(shape, 0.21),
        zu12=_grid_field(shape, 0.23),
        rs=_grid_field(shape, 0.29),
        zs=_grid_field(shape, 0.31),
    )
    jac_out = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=modes,
        s=s,
        lconm1=False,
        lthreed=False,
        lasym=False,
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom={},
        jac_half=jac,
    )
    assert jac_out.shape == shape
    assert float(np.linalg.norm(jac_out)) > 0.0

    fallback_out = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=modes,
        s=s,
        lconm1=False,
        lthreed=False,
        lasym=False,
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom={},
    )
    assert fallback_out.shape == shape
    assert np.all(np.isfinite(fallback_out))

    single_surface_shape = (1, int(trig.ntheta2), int(np.asarray(trig.cosnv).shape[0]))
    single_out = _compute_bsubs_half_mesh(
        state=_state_and_modes(ns=1)[0],
        geom_modes=modes,
        s=np.asarray([0.0]),
        lconm1=False,
        lthreed=False,
        lasym=False,
        bsupu=np.ones(single_surface_shape),
        bsupv=np.ones(single_surface_shape),
        trig=trig,
        geom={},
    )
    np.testing.assert_allclose(single_out, 0.0)


def test_state_geometry_helpers_cover_inferred_lasym_and_generated_trig_paths() -> None:
    state, modes = _state_and_modes()
    trig = vmec_trig_tables(ntheta=4, nzeta=2, nfp=1, mmax=1, nmax=0, lasym=False, cache=False)

    inferred = wout_module._vmec_realspace_geom_light_from_state(
        state=state,
        modes=modes,
        trig=trig,
        lasym=None,
    )
    assert set(inferred) == {"R", "Z", "Zu"}
    assert inferred["R"].shape == (1, int(trig.ntheta2), int(np.asarray(trig.cosnv).shape[0]))

    static = SimpleNamespace(
        cfg=SimpleNamespace(ntheta=4, nzeta=2, nfp=1, mpol=1, ntor=0, lasym=False),
        modes=modes,
    )
    aspect = wout_module.equilibrium_aspect_ratio_from_state(state=state, static=static)
    assert np.isfinite(float(np.asarray(aspect)))


def test_vmec_wrout_nyquist_lasym_loop_covers_symmetric_asymmetric_channels() -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=True, cache=False)
    modes = ModeTable(m=np.asarray([0, 1, 1, 2], dtype=int), n=np.asarray([0, -1, 1, 0], dtype=int))
    ns = 3
    shape = (ns, int(trig.ntheta2), int(np.asarray(trig.cosnv).shape[0]))
    base = _grid_field(shape, -0.4, scale=0.03)

    coeffs = _vmec_wrout_nyquist_lasym_loop(
        bsq=1.5 + base,
        gsqrt=2.0 + base,
        bsubu=0.2 + base,
        bsubv=-0.1 + base,
        bsubs=0.3 + base,
        bsupu=0.4 + base,
        bsupv=0.5 + base,
        modes=modes,
        trig=trig,
    )

    expected_keys = {
        "gmnc",
        "bmnc",
        "bsubumnc",
        "bsubvmnc",
        "bsubsmns",
        "bsupumnc",
        "bsupvmnc",
        "gmns",
        "bmns",
        "bsubumns",
        "bsubvmns",
        "bsubsmnc",
        "bsupumns",
        "bsupvmns",
    }
    assert set(coeffs) == expected_keys
    for value in coeffs.values():
        assert value.shape == (ns, modes.K)
        assert np.all(np.isfinite(value))
    np.testing.assert_allclose(coeffs["gmnc"][0], 0.0)
    np.testing.assert_allclose(coeffs["gmns"][0], 0.0)
    np.testing.assert_allclose(coeffs["bsubsmns"][0], 2.0 * coeffs["bsubsmns"][1] - coeffs["bsubsmns"][2])
    np.testing.assert_allclose(coeffs["bsubsmnc"][0], 2.0 * coeffs["bsubsmnc"][1] - coeffs["bsubsmnc"][2])

    empty = _vmec_wrout_nyquist_lasym_loop(
        bsq=1.5 + base,
        gsqrt=2.0 + base,
        bsubu=0.2 + base,
        bsubv=-0.1 + base,
        bsubs=0.3 + base,
        bsupu=0.4 + base,
        bsupv=0.5 + base,
        modes=ModeTable(m=np.asarray([], dtype=int), n=np.asarray([], dtype=int)),
        trig=trig,
    )
    assert all(value.shape == (ns, 0) for value in empty.values())


def test_bsubuv_parity_from_state_flips_only_odd_channel_when_odd_m_geometry_changes_sign() -> None:
    modes = ModeTable(m=np.asarray([1, 2], dtype=int), n=np.asarray([0, 0], dtype=int))
    ns = 3
    layout = StateLayout(ns=ns, K=modes.K, lasym=False)
    s = np.asarray([0.0, 0.25, 1.0])
    radial = s[:, None]
    rcos = np.concatenate([0.2 + 0.05 * radial, 0.1 + 0.03 * radial], axis=1)
    zsin = np.concatenate([0.3 + 0.04 * radial, 0.15 + 0.02 * radial], axis=1)
    zeros = np.zeros_like(rcos)
    state = VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zsin,
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )
    flipped_state = VMECState(
        layout=layout,
        Rcos=rcos * np.asarray([-1.0, 1.0])[None, :],
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zsin * np.asarray([-1.0, 1.0])[None, :],
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )
    trig = vmec_trig_tables(ntheta=8, nzeta=1, nfp=1, mmax=2, nmax=0, lasym=False, cache=False)
    shape = (ns, int(trig.ntheta2), int(np.asarray(trig.cosnv).shape[0]))
    bsupu = np.full(shape, 0.7)
    bsupv = np.zeros(shape)
    lambda_u = np.zeros(shape)
    lambda_v = np.zeros(shape)
    sqrtg = np.ones(shape)

    bsubu_even, bsubu_odd, bsubv_even, bsubv_odd = _bsubuv_parity_from_state(
        state=state,
        geom_modes=modes,
        trig=trig,
        s=s,
        lconm1=False,
        lthreed=False,
        lasym=False,
        bsupu=bsupu,
        bsupv=bsupv,
        lu1_full=lambda_u,
        lv1_full=lambda_v,
        sqrtg=sqrtg,
    )
    flipped_even, flipped_odd, flipped_v_even, flipped_v_odd = _bsubuv_parity_from_state(
        state=flipped_state,
        geom_modes=modes,
        trig=trig,
        s=s,
        lconm1=False,
        lthreed=False,
        lasym=False,
        bsupu=bsupu,
        bsupv=bsupv,
        lu1_full=lambda_u,
        lv1_full=lambda_v,
        sqrtg=sqrtg,
    )

    # Metric products quadratic in odd-m geometry remain in the even channel,
    # while even/odd cross-products change sign. This is the parity split VMEC
    # later filters separately before writing bsubu/bsubv.
    np.testing.assert_allclose(flipped_even, bsubu_even, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(flipped_odd, -bsubu_odd, rtol=1.0e-13, atol=1.0e-13)
    assert float(np.linalg.norm(bsubu_even[1:])) > 0.0
    assert float(np.linalg.norm(bsubu_odd[1:])) > 0.0
    np.testing.assert_allclose(flipped_v_even, bsubv_even, atol=1.0e-14)
    np.testing.assert_allclose(flipped_v_odd, bsubv_odd, atol=1.0e-14)

    constrained = _bsubuv_parity_from_state(
        state=state,
        geom_modes=modes,
        trig=trig,
        s=s,
        lconm1=True,
        lthreed=True,
        lasym=False,
        bsupu=bsupu,
        bsupv=bsupv,
        lu1_full=lambda_u,
        lv1_full=lambda_v,
        sqrtg=sqrtg,
    )
    assert [arr.shape for arr in constrained] == [shape] * 4
    assert all(np.all(np.isfinite(arr)) for arr in constrained)


def test_jxbforce_loop_filters_cover_single_surface_and_nyquist_weight_branches() -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=5, nfp=1, mmax=2, nmax=2, lasym=False, cache=False)
    ns = 1
    nt2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    mnyq, _ = _jxbforce_nyquist_limits(trig)
    shape = (ns, nt2, nzeta)
    base = _grid_field(shape, -0.25, scale=0.07)
    s_single = np.asarray([0.25])

    parity = _filter_bsubuv_jxbforce_parity_loop(
        bsubu_even=base,
        bsubu_odd=0.5 * base,
        bsubv_even=np.cos(base),
        bsubv_odd=0.25 * np.cos(base),
        trig=trig,
        mmax_force=mnyq,
        nmax_force=1,
        s=s_single,
    )
    assert [arr.shape for arr in parity] == [shape] * 2
    assert all(np.all(np.isfinite(arr)) for arr in parity)

    coupled = _jxbforce_filter_with_bsubs_derivs_loop(
        bsubs=np.sin(base),
        bsubu_even=base,
        bsubu_odd=0.5 * base,
        bsubv_even=np.cos(base),
        bsubv_odd=0.25 * np.cos(base),
        trig=trig,
        mmax_force=mnyq,
        nmax_force=1,
        s=s_single,
    )
    assert [arr.shape for arr in coupled] == [shape] * 4
    assert all(np.all(np.isfinite(arr)) for arr in coupled)

    loop_u, loop_v = _filter_bsubuv_jxbforce_loop(
        bsubu=base,
        bsubv=np.cos(base),
        trig=trig,
        mmax_force=mnyq,
        nmax_force=1,
        s=s_single,
    )
    assert loop_u.shape == shape
    assert loop_v.shape == shape
    assert np.all(np.isfinite(loop_u))
    assert np.all(np.isfinite(loop_v))

    deriv_u, deriv_v = _jxbforce_bsubsu_bsubsv_loop(
        bsubs=np.sin(base),
        trig=trig,
        mmax_force=mnyq,
        nmax_force=1,
    )
    assert deriv_u.shape == shape
    assert deriv_v.shape == shape
    assert np.all(np.isfinite(deriv_u))
    assert np.all(np.isfinite(deriv_v))


def test_jxbforce_filters_validate_shapes_and_preserve_identity_when_filter_disabled() -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=1, nmax=1, lasym=False, cache=False)
    nt2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    shape = (2, nt2, nzeta)
    base = _grid_field(shape, 0.1, scale=0.05)

    with pytest.raises(ValueError, match="Parity bsubu/bsubv shape mismatch"):
        _filter_bsubuv_jxbforce_parity_loop(
            bsubu_even=base,
            bsubu_odd=base[:, :1, :],
            bsubv_even=base,
            bsubv_odd=base,
            trig=trig,
            mmax_force=1,
            nmax_force=1,
        )
    with pytest.raises(ValueError, match="bsubu grid smaller than ntheta2"):
        _filter_bsubuv_jxbforce_parity_loop(
            bsubu_even=base[:, :1, :],
            bsubu_odd=base[:, :1, :],
            bsubv_even=base[:, :1, :],
            bsubv_odd=base[:, :1, :],
            trig=trig,
            mmax_force=1,
            nmax_force=1,
        )
    with pytest.raises(ValueError, match="bsubu grid smaller than ntheta2"):
        _filter_bsubuv_jxbforce_loop(
            bsubu=base[:, :1, :],
            bsubv=base[:, :1, :],
            trig=trig,
            mmax_force=1,
            nmax_force=1,
        )

    parity_u, parity_v = _filter_bsubuv_jxbforce_parity_loop(
        bsubu_even=base,
        bsubu_odd=2.0 * base,
        bsubv_even=-base,
        bsubv_odd=-2.0 * base,
        trig=trig,
        mmax_force=-1,
        nmax_force=1,
    )
    np.testing.assert_allclose(parity_u, base)
    np.testing.assert_allclose(parity_v, -base)

    loop_u, loop_v = _filter_bsubuv_jxbforce_loop(
        bsubu=base,
        bsubv=-base,
        trig=trig,
        mmax_force=-1,
        nmax_force=1,
    )
    np.testing.assert_allclose(loop_u, base)
    np.testing.assert_allclose(loop_v, -base)


def test_lasym_filter_loop_covers_fallback_channels_and_nyquist_weight_branch() -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=5, nfp=1, mmax=2, nmax=2, lasym=True, cache=False)
    ns = 1
    nt3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    mnyq, _ = _jxbforce_nyquist_limits(trig)
    shape = (ns, nt3, nzeta)
    bsubu = _grid_field(shape, 0.15, scale=0.04)
    bsubv = np.cos(bsubu)

    filtered_u, filtered_v = _filter_bsubuv_jxbforce_lasym_loop(
        bsubu=bsubu,
        bsubv=bsubv,
        trig=trig,
        mmax_force=mnyq,
        nmax_force=1,
        s=np.asarray([0.25]),
    )
    assert filtered_u.shape == shape
    assert filtered_v.shape == shape
    assert np.all(np.isfinite(filtered_u))
    assert np.all(np.isfinite(filtered_v))


def test_lasym_filter_parity_channels_zero_s_uses_unscaled_axis_branch() -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=1, nmax=1, lasym=True, cache=False)
    nt3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    shape = (1, nt3, nzeta)
    bsubu = _grid_field(shape, 0.2, scale=0.03)
    bsubv = _grid_field(shape, -0.1, scale=0.02)
    even_u = 0.5 * bsubu
    odd_u = -0.25 * bsubu
    even_v = 0.75 * bsubv
    odd_v = 0.40 * bsubv

    from_zero_s = _filter_bsubuv_jxbforce_lasym_loop(
        bsubu=bsubu,
        bsubv=bsubv,
        trig=trig,
        mmax_force=1,
        nmax_force=1,
        s=np.asarray([0.0]),
        bsubu_even=even_u,
        bsubu_odd=odd_u,
        bsubv_even=even_v,
        bsubv_odd=odd_v,
    )
    from_no_s = _filter_bsubuv_jxbforce_lasym_loop(
        bsubu=bsubu,
        bsubv=bsubv,
        trig=trig,
        mmax_force=1,
        nmax_force=1,
        s=None,
        bsubu_even=even_u,
        bsubu_odd=odd_u,
        bsubv_even=even_v,
        bsubv_odd=odd_v,
    )

    for actual, expected in zip(from_zero_s, from_no_s, strict=True):
        assert actual.shape == shape
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)


def test_nyquist_half_weight_validates_shapes_and_preserves_unweighted_cases() -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=1, nmax=1, lasym=False, cache=False)
    modes_empty = ModeTable(m=np.asarray([], dtype=int), n=np.asarray([], dtype=int))
    coeff_empty = np.zeros((2, 0))
    got_cos, got_sin = _apply_nyquist_half_weight(
        coeff_cos=coeff_empty,
        coeff_sin=coeff_empty,
        modes=modes_empty,
        trig=trig,
    )
    assert got_cos is coeff_empty
    assert got_sin is coeff_empty

    modes_unweighted = ModeTable(m=np.asarray([0], dtype=int), n=np.asarray([0], dtype=int))
    coeff = np.asarray([[2.0], [4.0]])
    got_cos, got_sin = _apply_nyquist_half_weight(
        coeff_cos=coeff,
        coeff_sin=-coeff,
        modes=modes_unweighted,
        trig=trig,
    )
    assert got_cos is coeff
    np.testing.assert_allclose(got_sin, -coeff)

    with pytest.raises(ValueError, match="Expected coeff arrays"):
        _apply_nyquist_half_weight(
            coeff_cos=np.ones((1, 1, 1)),
            coeff_sin=np.ones((1, 1)),
            modes=modes_unweighted,
            trig=trig,
        )


def test_read_and_write_wout_report_missing_netcdf_dependency(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "netCDF4", None)

    with pytest.raises(ImportError, match=r"netCDF4 is required to read wout files"):
        wout_module.read_wout(tmp_path / "wout_missing_dependency.nc")

    with pytest.raises(ImportError, match=r"netCDF4 is required to write wout files"):
        wout_module.write_wout(tmp_path / "wout_missing_dependency.nc", SimpleNamespace())


def test_getbsubs_coefficients_use_lstsq_when_direct_solve_fails(monkeypatch) -> None:
    trig_false = vmec_trig_tables(ntheta=4, nzeta=4, nfp=1, mmax=2, nmax=2, lasym=False, cache=False)
    nt2 = int(trig_false.ntheta2)
    nzeta = int(np.asarray(trig_false.cosnv).shape[0])
    grid = np.arange(nt2 * nzeta, dtype=float).reshape(nt2, nzeta)

    trig_true = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=2, nmax=0, lasym=True, cache=False)
    nt3 = int(trig_true.ntheta3)
    grid_true = np.arange(nt3, dtype=float).reshape(nt3, 1)

    def fail_solve(*_args, **_kwargs):
        raise np.linalg.LinAlgError("force least-squares fallback")

    monkeypatch.setattr(wout_module.np.linalg, "solve", fail_solve)

    coeff_false = _jxbforce_getbsubs_coeffs_lasym_false(
        frho=0.1 + np.sin(grid),
        bsupu=1.0 + 0.2 * np.cos(grid),
        bsupv=0.7 + 0.1 * np.sin(2.0 * grid),
        trig=trig_false,
        nfp=1,
    )
    assert coeff_false is not None
    assert coeff_false.shape == (nt2, 2 * (nzeta // 2) + 1)
    assert np.all(np.isfinite(coeff_false))

    coeff_true = _jxbforce_getbsubs_coeffs_lasym_true(
        frho=0.1 + np.sin(grid_true),
        bsupu=1.0 + 0.2 * np.cos(grid_true),
        bsupv=0.7 + 0.1 * np.sin(2.0 * grid_true),
        trig=trig_true,
        nfp=1,
    )
    assert coeff_true is not None
    assert coeff_true.shape == (int(trig_true.ntheta2), 1, 2)
    assert np.all(np.isfinite(coeff_true))


def test_bsubs_correction_helpers_keep_inputs_when_coefficients_are_unavailable(monkeypatch) -> None:
    trig_false = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=False, cache=False)
    ns = 3
    nt2 = int(trig_false.ntheta2)
    nzeta = int(np.asarray(trig_false.cosnv).shape[0])
    shape = (ns, nt2, nzeta)
    ones = np.ones(shape)
    bsubs = _grid_field(shape, 0.2, scale=0.03)
    bsubsu = _grid_field(shape, -0.1, scale=0.02)
    bsubsv = _grid_field(shape, 0.4, scale=0.01)

    monkeypatch.setattr(wout_module, "_jxbforce_getbsubs_coeffs_lasym_false", lambda **_: None)
    out_false = _jxbforce_apply_bsubs_correction_lasym_false(
        bsubu=ones,
        bsubv=2.0 * ones,
        bsubs=bsubs.copy(),
        bsubsu=bsubsu.copy(),
        bsubsv=bsubsv.copy(),
        bsupu=0.3 * ones,
        bsupv=0.2 * ones,
        sqrtg=ones,
        pres=np.asarray([0.0, 0.1, 0.2]),
        vp=np.asarray([1.0, 1.2, 1.4]),
        hs=0.5,
        signgs=1.0,
        trig=trig_false,
        nfp=1,
        sum_w=lambda arr: float(np.sum(arr)),
    )
    np.testing.assert_allclose(out_false[0][1], bsubs[1])
    np.testing.assert_allclose(out_false[1], bsubsu)
    np.testing.assert_allclose(out_false[2], bsubsv)

    early_false = _jxbforce_apply_bsubs_correction_lasym_false(
        bsubu=ones,
        bsubv=2.0 * ones,
        bsubs=bsubs.copy(),
        bsubsu=bsubsu.copy(),
        bsubsv=bsubsv.copy(),
        bsupu=0.3 * ones,
        bsupv=0.2 * ones,
        sqrtg=ones,
        pres=np.asarray([0.0, 0.1, 0.2]),
        vp=np.asarray([1.0, 1.2, 1.4]),
        hs=0.0,
        signgs=1.0,
        trig=trig_false,
        nfp=1,
        sum_w=lambda arr: float(np.sum(arr)),
    )
    np.testing.assert_allclose(early_false[0], bsubs)

    trig_true = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=2, nmax=0, lasym=True, cache=False)
    nt3 = int(trig_true.ntheta3)
    full_shape = (ns, nt3, 1)
    full_ones = np.ones(full_shape)
    full_bsubs = _grid_field(full_shape, 0.1, scale=0.05)
    monkeypatch.setattr(wout_module, "_jxbforce_getbsubs_coeffs_lasym_true", lambda **_: None)
    out_true = _jxbforce_apply_bsubs_correction_lasym_true(
        bsubu=full_ones,
        bsubv=2.0 * full_ones,
        bsubs=full_bsubs.copy(),
        bsubsu=np.zeros(full_shape),
        bsubsv=np.zeros(full_shape),
        bsupu=0.3 * full_ones,
        bsupv=0.2 * full_ones,
        sqrtg=full_ones,
        pres=np.asarray([0.0, 0.1, 0.2]),
        vp=np.asarray([1.0, 1.2, 1.4]),
        hs=0.5,
        signgs=1.0,
        trig=trig_true,
        nfp=1,
        sum_w=lambda arr: float(np.sum(arr)),
    )
    np.testing.assert_allclose(out_true[0][1], full_bsubs[1])
    np.testing.assert_allclose(out_true[1], 0.0)
    np.testing.assert_allclose(out_true[2], 0.0)

    short_trig = SimpleNamespace(ntheta1=1, ntheta2=2, ntheta3=1, cosnv=np.ones((1, 1)))
    early_true = _jxbforce_apply_bsubs_correction_lasym_true(
        bsubu=ones,
        bsubv=2.0 * ones,
        bsubs=bsubs.copy(),
        bsubsu=bsubsu.copy(),
        bsubsv=bsubsv.copy(),
        bsupu=0.3 * ones,
        bsupv=0.2 * ones,
        sqrtg=ones,
        pres=np.asarray([0.0, 0.1, 0.2]),
        vp=np.asarray([1.0, 1.2, 1.4]),
        hs=0.0,
        signgs=1.0,
        trig=short_trig,
        nfp=1,
        sum_w=lambda arr: float(np.sum(arr)),
    )
    np.testing.assert_allclose(early_true[0], bsubs)


def test_lasym_bsubs_correction_reconstructs_available_coefficients_on_full_grid(monkeypatch) -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=2, nmax=0, lasym=True, cache=False)
    ns = 4
    nt2 = int(trig.ntheta2)
    nt3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    reduced_shape = (ns, nt2, nzeta)
    surface = np.arange(np.prod(reduced_shape), dtype=float).reshape(reduced_shape)

    coeff = np.zeros((nt2, 1, 2), dtype=float)
    coeff[1, 0, 0] = 0.25
    coeff[2, 0, 1] = -0.125
    coeff_calls = []

    def fake_getbsubs_coeffs(**kwargs):
        coeff_calls.append(kwargs)
        assert kwargs["frho"].shape == (nt3, nzeta)
        assert kwargs["bsupu"].shape == (nt3, nzeta)
        assert kwargs["bsupv"].shape == (nt3, nzeta)
        return coeff

    monkeypatch.setattr(wout_module, "_jxbforce_getbsubs_coeffs_lasym_true", fake_getbsubs_coeffs)

    bsubs, bsubsu, bsubsv = _jxbforce_apply_bsubs_correction_lasym_true(
        bsubu=0.2 + 0.01 * surface,
        bsubv=0.3 + 0.02 * surface,
        bsubs=np.zeros(reduced_shape),
        bsubsu=np.zeros(reduced_shape),
        bsubsv=np.zeros(reduced_shape),
        bsupu=0.4 + 0.03 * surface,
        bsupv=0.5 + 0.04 * surface,
        sqrtg=1.0 + 0.01 * surface,
        pres=np.asarray([0.0, 0.3, 0.2, 0.1]),
        vp=np.asarray([1.0, 1.2, 1.4, 1.6]),
        hs=1.0 / 3.0,
        signgs=1.0,
        trig=trig,
        nfp=1,
        sum_w=lambda arr: float(np.sum(arr)),
    )

    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : coeff.shape[0]]
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : coeff.shape[0]]
    sinmum = np.asarray(trig.sinmum, dtype=float)[:nt2, : coeff.shape[0]]
    cosmum = np.asarray(trig.cosmum, dtype=float)[:nt2, : coeff.shape[0]]
    bsubs_s = np.einsum("jm,m->j", sinmu, coeff[:, 0, 0])[:, None]
    bsubs_a = np.einsum("jm,m->j", cosmu, coeff[:, 0, 1])[:, None]
    bsubsu_s = np.einsum("jm,m->j", cosmum, coeff[:, 0, 0])[:, None]
    bsubsu_a = np.einsum("jm,m->j", sinmum, coeff[:, 0, 1])[:, None]

    def extend(par0: np.ndarray, par1: np.ndarray) -> np.ndarray:
        full = np.zeros((nt3, nzeta), dtype=float)
        full[:nt2] = par0 + par1
        for theta_idx in range(nt2):
            mirror = 0 if theta_idx == 0 else int(trig.ntheta1) - theta_idx
            if mirror >= nt2:
                full[mirror] = par0[theta_idx] - par1[theta_idx]
        return full

    expected_bsubs = extend(bsubs_a, bsubs_s)
    expected_bsubsu = extend(bsubsu_s, bsubsu_a)

    assert len(coeff_calls) == ns - 2
    assert bsubs.shape == (ns, nt3, nzeta)
    assert bsubsu.shape == (ns, nt3, nzeta)
    assert bsubsv.shape == (ns, nt3, nzeta)
    np.testing.assert_allclose(bsubs[1], expected_bsubs, atol=1.0e-15)
    np.testing.assert_allclose(bsubs[2], expected_bsubs, atol=1.0e-15)
    np.testing.assert_allclose(bsubsu[1], expected_bsubsu, atol=1.0e-15)
    np.testing.assert_allclose(bsubsu[2], expected_bsubsu, atol=1.0e-15)
    np.testing.assert_allclose(bsubsv, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(bsubs[0], 2.0 * bsubs[1] - bsubs[2], atol=1.0e-15)
    np.testing.assert_allclose(bsubs[-1], -bsubs[-2], atol=1.0e-15)
