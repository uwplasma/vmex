from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.solve import (
    _HLO_DUMPED_KEYS,
    _dump_array,
    _dump_freeb_axis_trace_record,
    _dump_freeb_control_trace_record,
    _dump_time_control_trace_record,
    _maybe_dump_bsube,
    _maybe_dump_bsube_terms,
    _maybe_dump_bsubh,
    _maybe_dump_bsubs,
    _maybe_dump_checkpoint_record,
    _maybe_dump_evolve_trace_record,
    _maybe_dump_force_kernels,
    _maybe_dump_gcx2,
    _maybe_dump_gmetric,
    _maybe_dump_hlo_kernel,
    _maybe_dump_jacobian_terms_record,
    _maybe_dump_lam_fsql1,
    _maybe_dump_lam_gcl,
    _maybe_dump_lam_prec,
    _maybe_dump_lamcal,
    _maybe_dump_lulv,
    _maybe_dump_precond_inputs,
    _maybe_dump_precond_mats,
    _maybe_dump_scalars,
    _maybe_dump_time_control_record,
    _maybe_dump_tomnsps,
    _maybe_dump_xc,
)
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.kernels.tomnsp import TomnspsRZL


def _static(*, ns=2, mpol=2, ntor=1, lasym=False, lthreed=True, ntheta=2, nzeta=1):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            ns=ns,
            mpol=mpol,
            ntor=ntor,
            lasym=lasym,
            lthreed=lthreed,
            ntheta=ntheta,
            nzeta=nzeta,
        )
    )


def _arr(value, shape=(2, 2, 2)):
    return np.full(shape, float(value))


def test_dump_array_preserves_payloads_and_uses_empty_float_for_missing():
    missing = _dump_array(None)
    assert missing.shape == (0,)
    assert missing.dtype == np.dtype(float)

    arr = np.asarray([[1, 2]], dtype=np.int32)
    out = _dump_array(arr)
    np.testing.assert_array_equal(out, arr)
    assert out.dtype == arr.dtype


def _frzl(*, shape=(2, 2, 2), include_asym=True):
    kwargs = {}
    if include_asym:
        kwargs = {
            "frsc": _arr(7, shape),
            "frcs": _arr(8, shape),
            "fzcc": _arr(9, shape),
            "fzss": _arr(10, shape),
            "flcc": _arr(11, shape),
            "flss": _arr(12, shape),
        }
    return TomnspsRZL(
        frcc=_arr(1, shape),
        frss=_arr(2, shape),
        fzsc=_arr(3, shape),
        fzcs=_arr(4, shape),
        flsc=_arr(5, shape),
        flcs=_arr(6, shape),
        **kwargs,
    )


def _bc(shape=(2, 2, 1)):
    base = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.0
    return SimpleNamespace(
        bsubu_e_scaled=base,
        bsubv_e_scaled=base + 10.0,
        lamscale=2.5,
        lvv_sh=base + 20.0,
        lu0_force=base + 30.0,
        lu1_full=base + 40.0,
        phip_internal=np.linspace(0.0, 1.0, shape[0]),
        bsubu_tmp=base + 50.0,
        bsubv_preblend=base + 60.0,
        bsubu=base + 70.0,
        bsubv=base + 80.0,
        lu0_full=base + 90.0,
        lv0_full=base + 100.0,
        lv1_full=base + 110.0,
        guu=base + 120.0,
        guv=base + 130.0,
        gvv=base + 140.0,
        bsq=base + 150.0,
        jac=SimpleNamespace(
            r12=np.ones(shape),
            sqrtg=base + 160.0,
            ru12=base + 170.0,
            zu12=base + 180.0,
            tau=base + 190.0,
            rs=base + 200.0,
            zs=base + 210.0,
        ),
    )


def test_scalar_and_gcx2_dumps_respect_guards_and_write_rows(tmp_path, monkeypatch):
    norms = SimpleNamespace(wb=1.0, wp=2.0, volume=3.0, r2=4.0, fnorm=5.0, fnormL=6.0)

    _maybe_dump_scalars(norms=norms, iter_idx=2, ns=3)
    assert not (tmp_path / "scalars_ns3_iter2.dat").exists()

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_SCALARS", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "2")
    _maybe_dump_scalars(norms=norms, iter_idx=1, ns=3)
    assert not (tmp_path / "scalars_ns3_iter1.dat").exists()

    _maybe_dump_scalars(norms=norms, iter_idx=2, ns=3)
    scalars = (tmp_path / "scalars_ns3_iter2.dat").read_text()
    assert "# bcovar scalars dump" in scalars
    assert "5.0000000000000000e+00" in scalars

    monkeypatch.setenv("VMEC_JAX_DUMP_GCX2", "1")
    _maybe_dump_gcx2(gcr2=1.25, gcz2=2.5, gcl2=3.75, iter_idx=2, include_edge=True, ns=3)
    gcx2 = (tmp_path / "gcx2_ns3_iter2.dat").read_text()
    assert "columns: iter include_edge gcr2 gcz2 gcl2" in gcx2
    assert " 1.2500000000000000e+00" in gcx2


def test_host_control_dump_records_write_logs_and_axis_payload(tmp_path, monkeypatch):
    _maybe_dump_time_control_record(iter_idx=1, fsq=1.0, fsq0=2.0, res0=3.0, res1=4.0, time_step=0.5)
    assert not (tmp_path / "time_control.log").exists()

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_TIMECONTROL", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_CHECKPOINT", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_FREEB_CONTROL", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_FREEB_AXIS", "1")

    _maybe_dump_time_control_record(iter_idx=2, fsq=1.0, fsq0=2.0, res0=3.0, res1=4.0, time_step=0.5)
    _dump_time_control_trace_record(
        stage="restart",
        iter2=2,
        iter1=1,
        fsq=1.0,
        fsq0=2.0,
        res0=3.0,
        res1=4.0,
        time_step=0.5,
        irst=2,
    )
    _maybe_dump_checkpoint_record(iter_idx=2, fsq=1.0, fsq0=2.0, res0=3.0, res1=4.0)
    _dump_freeb_control_trace_record(
        iter2=2,
        iter1=1,
        ivac=3,
        ivacskip=4,
        nvacskip=5,
        fsq_rz_prev=0.25,
        cached=True,
    )
    _dump_freeb_axis_trace_record(iter2=2, axis_r=np.array([[1.0, 2.0]]), axis_z=np.array([[3.0, 4.0]]))

    assert "time_step=5.000000e-01" in (tmp_path / "time_control.log").read_text()
    assert (tmp_path / "time_control_trace.log").read_text().endswith("   2 restart\n")
    assert "iter=2 fsq=1.000000e+00" in (tmp_path / "checkpoint.log").read_text()
    assert (tmp_path / "freeb_control_trace.log").read_text().endswith(" 1\n")
    with np.load(tmp_path / "freeb_axis_iter2.npz") as data:
        np.testing.assert_allclose(data["axis_r"], [1.0, 2.0])
        np.testing.assert_allclose(data["axis_z"], [3.0, 4.0])


def test_evolve_trace_record_includes_asymmetric_velocity_and_force_norms(tmp_path, monkeypatch):
    import vmec_jax.diagnostics as diagnostics

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_EVOLVE", "1")

    layout = StateLayout(ns=2, K=2, lasym=True)
    state = VMECState(
        layout=layout,
        Rcos=np.ones((2, 2)),
        Rsin=2.0 * np.ones((2, 2)),
        Zcos=3.0 * np.ones((2, 2)),
        Zsin=4.0 * np.ones((2, 2)),
        Lcos=5.0 * np.ones((2, 2)),
        Lsin=6.0 * np.ones((2, 2)),
    )
    static = _static(ns=2, mpol=2, ntor=1, lasym=True)

    def fake_internal_mn_from_state(*args, **kwargs):
        return {
            "rcc": np.array([1.0]),
            "rss": np.array([2.0]),
            "zsc": np.array([3.0]),
            "zcs": np.array([4.0]),
            "lsc": np.array([5.0]),
            "lcs": np.array([6.0]),
            "rsc": np.array([7.0]),
            "rcs": np.array([8.0]),
            "zcc": np.array([9.0]),
            "zss": np.array([10.0]),
            "lcc": np.array([11.0]),
            "lss": np.array([12.0]),
        }

    def fake_xc_from_mn_blocks(*, cfg, **kwargs):
        assert cfg is static.cfg
        return np.concatenate([np.asarray(value, dtype=float).reshape(-1) for value in kwargs.values() if value is not None])

    monkeypatch.setattr(diagnostics, "vmec_internal_mn_from_state", fake_internal_mn_from_state)
    monkeypatch.setattr(diagnostics, "vmec_xc_from_mn_blocks", fake_xc_from_mn_blocks)

    _maybe_dump_evolve_trace_record(
        static=static,
        iter2=4,
        iter1=1,
        stage="post",
        fsq1_val=1.0,
        fsq_prev_val=2.0,
        time_step_val=0.5,
        dtau_val=0.25,
        b1_val=0.75,
        fac_val=0.5,
        state_val=state,
        vRcc_val=np.array([1.0]),
        vRss_val=np.array([2.0]),
        vZsc_val=np.array([3.0]),
        vZcs_val=np.array([4.0]),
        vLsc_val=np.array([5.0]),
        vLcs_val=np.array([6.0]),
        vRsc_val=np.array([7.0]),
        vRcs_val=np.array([8.0]),
        vZcc_val=np.array([9.0]),
        vZss_val=np.array([10.0]),
        vLcc_val=np.array([11.0]),
        vLss_val=np.array([12.0]),
        frcc_val=np.array([2.0]),
        frss_val=np.array([3.0]),
        fzsc_val=np.array([4.0]),
        fzcs_val=np.array([5.0]),
        flsc_val=np.array([6.0]),
        flcs_val=np.array([7.0]),
        frsc_val=np.array([8.0]),
        frcs_val=np.array([9.0]),
        fzcc_val=np.array([10.0]),
        fzss_val=np.array([11.0]),
        flcc_val=np.array([12.0]),
        flss_val=np.array([13.0]),
    )

    fields = (tmp_path / "evolve_trace.log").read_text().split()
    assert fields[:4] == ["4", "1", "2", "post"]
    assert float(fields[-1]) == pytest.approx(np.linalg.norm(np.arange(2.0, 14.0)))


def test_jacobian_terms_dump_record_respects_iter_guard_and_writes_ptau_rows(tmp_path, monkeypatch):
    arr = np.arange(3.0).reshape(3, 1, 1) + 1.0
    k = SimpleNamespace(
        pr1_even=arr,
        pr1_odd=arr + 1.0,
        pz1_even=arr + 2.0,
        pz1_odd=arr + 3.0,
        pru_even=arr + 4.0,
        pru_odd=arr + 5.0,
        pzu_even=arr + 6.0,
        pzu_odd=arr + 7.0,
        prv_even=arr + 8.0,
        prv_odd=arr + 9.0,
        pzv_even=arr + 10.0,
        pzv_odd=arr + 11.0,
    )

    _maybe_dump_jacobian_terms_record(k=k, s=np.array([0.0, 0.5, 1.0]), iter_idx=3)
    assert not (tmp_path / "jacobian_terms_iter3.dat").exists()

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_JACOBIAN_TERMS", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "4")
    _maybe_dump_jacobian_terms_record(k=k, s=np.array([0.0, 0.5, 1.0]), iter_idx=3)
    assert not (tmp_path / "jacobian_terms_iter3.dat").exists()

    _maybe_dump_jacobian_terms_record(k=k, s=np.array([0.0, 0.5, 1.0]), iter_idx=4)

    text = (tmp_path / "jacobian_terms_iter4.dat").read_text()
    assert "ru12 pzs pzu12 prs pr12 ptau" in text
    data_rows = [line for line in text.splitlines() if line.startswith("     ")]
    assert len(data_rows) == 2


def test_npz_dump_helpers_write_selected_payloads(tmp_path, monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "4")
    static = _static()
    frzl = _frzl()

    monkeypatch.setenv("VMEC_JAX_DUMP_TOMNSPS", "1")
    _maybe_dump_tomnsps(frzl=frzl, static=static, iter_idx=4, label="raw")
    with np.load(tmp_path / "tomnsps_raw_ns2_iter4.npz") as data:
        np.testing.assert_allclose(data["frcc"], 1.0)
        assert bool(data["lasym"]) is False

    monkeypatch.setenv("VMEC_JAX_DUMP_FORCE_KERNELS", "1")
    kernels = SimpleNamespace(armn_e=_arr(1), bc=SimpleNamespace(blmn_even=_arr(2), bsubu=_arr(3)))
    _maybe_dump_force_kernels(k=kernels, static=static, iter_idx=4, label="raw")
    with np.load(tmp_path / "force_kernels_raw_ns2_iter4.npz") as data:
        np.testing.assert_allclose(data["armn_e"], 1.0)
        np.testing.assert_allclose(data["blmn_e"], 2.0)
        np.testing.assert_allclose(data["bsubu"], 3.0)

    monkeypatch.setenv("VMEC_JAX_DUMP_PRECOND_MATS", "1")
    _maybe_dump_precond_mats(mats={"ar": _arr(4), "dz": _arr(5)}, static=static, iter_idx=4, jmax=7, used_cache=True)
    with np.load(tmp_path / "precond_mats_ns2_iter4.npz") as data:
        assert int(data["jmax"]) == 7
        assert bool(data["used_cache"]) is True
        np.testing.assert_allclose(data["dz"], 5.0)

    monkeypatch.setenv("VMEC_JAX_DUMP_LAM", "1")
    _maybe_dump_lam_fsql1(fsql1_pre=1.5, fsql1_post=0.5, static=static, iter_idx=4)
    assert "1.5000000000000000e+00" in (tmp_path / "lam_fsql1_ns2_iter4.dat").read_text()

    monkeypatch.setenv("VMEC_JAX_DUMP_LAMCAL", "1")
    lam_debug = {
        "blam_pre": _arr(1),
        "clam_pre": _arr(2),
        "dlam_pre": _arr(3),
        "blam_post": _arr(4),
        "clam_post": _arr(5),
        "dlam_post": _arr(6),
    }
    _maybe_dump_lamcal(lam_debug=lam_debug, static=static, iter_idx=4)
    with np.load(tmp_path / "lamcal_ns2_iter4.npz") as data:
        np.testing.assert_allclose(data["dlam_post"], 6.0)


def test_lambda_gcl_dump_writes_full_channel_layout_and_fsql(tmp_path, monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM_ITER", "5")

    static = _static(lasym=True, lthreed=False)
    pre = _frzl()
    post = _frzl()
    post = TomnspsRZL(**{**pre.__dict__, "flsc": 2.0 * pre.flsc, "flcc": 3.0 * pre.flcc})

    _maybe_dump_lam_gcl(frzl_pre=pre, frzl_post=post, static=static, iter_idx=5, delta_s=0.25)

    with np.load(tmp_path / "lam_gcl_ns2_iter5.npz") as data:
        assert data["gcl_pre"].shape == (2, 2, 2, 2)
        assert float(data["fsql1_post"]) > float(data["fsql1_pre"])
        assert float(data["delta_s"]) == 0.25
    assert (tmp_path / "lam_fsql1_ns2_iter5.dat").exists()


def test_bcovar_text_dump_helpers_write_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "6")
    static = _static()
    bc = _bc()

    monkeypatch.setenv("VMEC_JAX_DUMP_BSUBE", "1")
    _maybe_dump_bsube(bc=bc, static=static, iter_idx=6)
    assert "lamscale=2.5000000000000000e+00" in (tmp_path / "bsube_ns2_iter6.dat").read_text()

    monkeypatch.setenv("VMEC_JAX_DUMP_BSUBE_TERMS", "1")
    _maybe_dump_bsube_terms(bc=bc, static=static, iter_idx=6)
    assert "bsubu_tmp" in (tmp_path / "bsube_terms_ns2_iter6.dat").read_text()

    monkeypatch.setenv("VMEC_JAX_DUMP_BSUBH", "1")
    _maybe_dump_bsubh(bc=bc, static=static, iter_idx=6)
    assert "bsubuh bsubvh" in (tmp_path / "bsubh_ns2_iter6.dat").read_text()

    monkeypatch.setenv("VMEC_JAX_DUMP_LULV", "1")
    _maybe_dump_lulv(bc=bc, static=static, iter_idx=6)
    with np.load(tmp_path / "lulv_ns2_iter6.npz") as data:
        np.testing.assert_allclose(data["lv1_full"], bc.lv1_full)

    monkeypatch.setenv("VMEC_JAX_DUMP_GMETRIC", "1")
    _maybe_dump_gmetric(bc=bc, static=static, iter_idx=6)
    gmetric = (tmp_path / "gmetric_iter6.dat").read_text()
    assert "pguu pguv pgvv" in gmetric
    assert "0.0000000000000000e+00" in gmetric


def test_precond_inputs_dump_uses_weight_fallback_and_hidden_npz(tmp_path, monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "7")
    monkeypatch.setenv("VMEC_JAX_DUMP_PRECOND", "1")

    shape = (2, 2, 1)
    bc = _bc(shape)
    kernels = SimpleNamespace(
        pru_even=_arr(1, shape),
        pru_odd=_arr(2, shape),
        pzu_even=_arr(3, shape),
        pzu_odd=_arr(4, shape),
        pr1_odd=_arr(5, shape),
        pz1_odd=_arr(6, shape),
    )

    _maybe_dump_precond_inputs(bc=bc, trig=None, static=_static(), iter_idx=7, kernels=kernels)

    text = (tmp_path / "precond_inputs_iter7.dat").read_text()
    assert "ru12 zu12 wint" in text
    assert "1.0000000000000000E+00" in text
    with np.load(tmp_path / "precond_hidden_iter7.npz") as data:
        np.testing.assert_allclose(data["tau"], bc.jac.tau)
        np.testing.assert_allclose(data["pr1_odd"], 5.0)


def test_hlo_and_lam_prec_dump_helpers_cover_layout_branches(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    import jax.numpy as jnp

    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_DIR", str(tmp_path))
    _HLO_DUMPED_KEYS.clear()
    static = _static(ns=3, mpol=2, ntor=1, lasym=True)
    wout_like = SimpleNamespace(mpol=2, ntor=1, nfp=5, lasym=True)

    _maybe_dump_hlo_kernel(
        label="tiny",
        fn=lambda x: x + 1.0,
        args=(jnp.ones((1,)),),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )
    hlo_path = tmp_path / "hlo_tiny_ns3_mpol2_ntor1.txt"
    assert hlo_path.exists()
    assert hlo_path.read_text()

    _maybe_dump_hlo_kernel(
        label="tiny",
        fn=lambda x: x + 2.0,
        args=(jnp.ones((1,)),),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )
    assert len(list(tmp_path.glob("hlo_tiny_ns3_mpol2_ntor1.txt"))) == 1

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM_ITER", "8,9")
    lam_prec = np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2) + 1.0
    faclam = np.full_like(lam_prec, 2.0)
    _maybe_dump_lam_prec(lam_prec=lam_prec, faclam=faclam, static=static, iter_idx=8)
    with np.load(tmp_path / "lam_prec_ns3_iter8.npz") as data:
        assert data["pfaclam"].shape == (3, 2, 2, 4)
        assert data["faclam"].shape == (3, 2, 2, 4)
        np.testing.assert_allclose(data["pfaclam"][:, 0, 0, 1:], 0.0)
        np.testing.assert_allclose(data["pfaclam"][:, :, :, 0], np.transpose(lam_prec, (0, 2, 1)))

    _maybe_dump_lam_prec(lam_prec=lam_prec, faclam=np.array([1.0, 2.0, 3.0]), static=static, iter_idx=9)
    with np.load(tmp_path / "lam_prec_ns3_iter9.npz") as data:
        np.testing.assert_allclose(data["faclam"], np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="lam_prec expected 3D"):
        _maybe_dump_lam_prec(lam_prec=np.ones((3, 2)), faclam=None, static=static, iter_idx=8)


def test_hlo_dump_guard_and_error_branches(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    import jax.numpy as jnp

    import vmec_jax.solve as solve_mod

    _HLO_DUMPED_KEYS.clear()
    static = _static(ns=2, mpol=1, ntor=0)
    wout_like = SimpleNamespace(mpol=1, ntor=0, nfp=1, lasym=False)

    _maybe_dump_hlo_kernel(
        label="guard",
        fn=lambda x: x,
        args=(jnp.ones((1,)),),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )
    assert not list(tmp_path.glob("hlo_guard*"))

    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_DIR", str(tmp_path))
    _maybe_dump_hlo_kernel(
        label="disabled",
        fn=lambda x: x,
        args=(jnp.ones((1,)),),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=False,
    )
    assert not list(tmp_path.glob("hlo_disabled*"))

    monkeypatch.setattr(solve_mod, "has_jax", lambda: False)
    _maybe_dump_hlo_kernel(
        label="nojax",
        fn=lambda x: x,
        args=(jnp.ones((1,)),),
        kwargs={},
        static=static,
        wout_like=wout_like,
        force=True,
    )
    assert not list(tmp_path.glob("hlo_nojax*"))
    monkeypatch.setattr(solve_mod, "has_jax", lambda: True)

    class BadStatic:
        @property
        def cfg(self):
            raise RuntimeError("bad cfg")

    def bad_fn(x):
        raise RuntimeError("cannot lower")

    monkeypatch.setenv("VMEC_JAX_DUMP_HLO_VERBOSE", "1")
    _maybe_dump_hlo_kernel(
        label="bad",
        fn=bad_fn,
        args=(jnp.ones((1,)),),
        kwargs={},
        static=BadStatic(),
        wout_like=wout_like,
        force=True,
    )
    err_path = tmp_path / "hlo_bad_error_ns0_mpol0_ntor0.txt"
    assert err_path.exists()
    assert "cannot lower" in err_path.read_text()


def test_bsubs_dump_uses_force_kernel_payload_without_real_wout(tmp_path, monkeypatch):
    import vmec_jax.wout as wout

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "12")
    monkeypatch.setenv("VMEC_JAX_DUMP_BSUBS", "1")
    monkeypatch.setenv("VMEC_JAX_WOUT_FORCE_BSS", "1")

    shape = (3, 2, 1)
    bc = _bc(shape)
    bc.bsupu = _arr(20, shape)
    bc.bsupv = _arr(21, shape)
    static = _static(ns=3, mpol=2, ntor=1, lasym=True, ntheta=2, nzeta=1)
    static.s = np.array([0.0, 0.5, 1.0])
    static.modes = SimpleNamespace(m=np.array([0, 1]), n=np.array([0, 0]))
    kernels = SimpleNamespace(
        crmn_e=_arr(1, shape),
        czmn_e=_arr(2, shape),
        bzmn_e=_arr(3, shape),
        brmn_e=_arr(4, shape),
        azmn_e=_arr(5, shape),
        armn_e=_arr(6, shape),
        pr1_even=_arr(7, shape),
        pr1_odd=_arr(8, shape),
        pz1_even=_arr(9, shape),
        pz1_odd=_arr(10, shape),
    )

    def fake_symforce_apply(*, f, trig, kind):
        assert trig == "trig"
        assert kind
        return -np.asarray(f)

    def fake_compute_bsubs_half_mesh(**kwargs):
        np.testing.assert_allclose(kwargs["bsupu"], -kernels.crmn_e)
        np.testing.assert_allclose(kwargs["bsupv"], -kernels.czmn_e)
        np.testing.assert_allclose(kwargs["force_rs"], -kernels.bzmn_e)
        assert "pr1_even" in kwargs["geom"]
        return np.arange(np.prod(shape), dtype=float).reshape(shape)

    monkeypatch.setattr(wout, "_vmec_symforce_apply", fake_symforce_apply)
    monkeypatch.setattr(wout, "_compute_bsubs_half_mesh", fake_compute_bsubs_half_mesh)

    _maybe_dump_bsubs(bc=bc, state=SimpleNamespace(), static=static, trig="trig", iter_idx=12, kernels=kernels)

    with np.load(tmp_path / "bsubs_ns3_iter12.npz") as data:
        np.testing.assert_allclose(data["bsupu"], -kernels.crmn_e)
        np.testing.assert_allclose(data["bsubs_full"][0], 0.0)
        np.testing.assert_allclose(data["bsubs_full"][-1], 0.0)
        np.testing.assert_allclose(data["bsubs_full"][1], 0.5 * (data["bsubs_half"][1] + data["bsubs_half"][2]))


def test_bsubs_and_precond_input_guard_branches(tmp_path, monkeypatch):
    import vmec_jax.wout as wout

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "20")
    monkeypatch.setenv("VMEC_JAX_DUMP_BSUBS", "1")
    shape = (2, 1, 1)
    bc = _bc(shape)
    bc.bsupu = _arr(2, shape)
    bc.bsupv = _arr(3, shape)
    static = _static(ns=2, mpol=1, ntor=0, lasym=False, ntheta=1, nzeta=1)
    static.s = np.array([0.0, 1.0])
    static.modes = SimpleNamespace(m=np.array([0]), n=np.array([0]))

    _maybe_dump_bsubs(bc=bc, state=SimpleNamespace(), static=static, trig=None, iter_idx=19, kernels=None)
    assert not (tmp_path / "bsubs_ns2_iter19.npz").exists()

    monkeypatch.delenv("VMEC_JAX_WOUT_FORCE_BSS", raising=False)
    monkeypatch.setattr(wout, "_compute_bsubs_half_mesh", lambda **kwargs: np.ones(shape))
    monkeypatch.setattr(wout, "_vmec_symforce_apply", lambda **kwargs: (_ for _ in ()).throw(AssertionError("unused")))
    _maybe_dump_bsubs(bc=bc, state=SimpleNamespace(), static=static, trig=None, iter_idx=20, kernels=SimpleNamespace(crmn_e=_arr(5, shape)))
    with np.load(tmp_path / "bsubs_ns2_iter20.npz") as data:
        np.testing.assert_allclose(data["bsupu"], 5.0)

    monkeypatch.setenv("VMEC_JAX_DUMP_PRECOND", "1")
    _maybe_dump_precond_inputs(bc=SimpleNamespace(jac=SimpleNamespace()), trig=None, static=static, iter_idx=20)
    assert not (tmp_path / "precond_inputs_iter20.dat").exists()

    bc = _bc((3, 1, 1))
    _maybe_dump_precond_inputs(bc=bc, trig=SimpleNamespace(wint3_precond=np.ones((2,))), static=static, iter_idx=20)
    text_path = tmp_path / "precond_inputs_iter20.dat"
    assert not text_path.exists()

    _maybe_dump_precond_inputs(bc=bc, trig=SimpleNamespace(wint3_precond=np.ones((3, 1, 1))), static=static, iter_idx=20)
    assert text_path.exists()
    text_path.unlink()

    _maybe_dump_precond_inputs(bc=bc, trig=SimpleNamespace(wint3_precond=np.ones((2, 1, 1))), static=static, iter_idx=20)
    assert text_path.exists()


def test_xc_and_lulv_dump_helpers_cover_asymmetric_payloads(tmp_path, monkeypatch):
    import vmec_jax.diagnostics as diagnostics
    import vmec_jax.kernels.realspace as realspace

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "13")
    monkeypatch.setenv("VMEC_JAX_DUMP_XC", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_LULV", "1")

    layout = StateLayout(ns=2, K=3, lasym=True)
    coeffs = np.arange(6, dtype=float).reshape(2, 3)
    state = VMECState(
        layout=layout,
        Rcos=coeffs,
        Rsin=coeffs + 10.0,
        Zcos=coeffs + 20.0,
        Zsin=coeffs + 30.0,
        Lcos=coeffs + 40.0,
        Lsin=coeffs + 50.0,
    )
    static = _static(ns=2, mpol=2, ntor=1, lasym=True)
    static.modes = SimpleNamespace(m=np.array([0, 1, 3]), n=np.array([0, 0, 0]))
    static.s = np.array([0.0, 1.0])

    def fake_internal_mn_from_state(*args, **kwargs):
        return {
            "rcc": np.array([1.0]),
            "rss": np.array([2.0]),
            "zsc": np.array([3.0]),
            "zcs": np.array([4.0]),
            "lsc": np.array([5.0]),
            "lcs": np.array([6.0]),
            "rsc": np.array([7.0]),
            "rcs": np.array([8.0]),
            "zcc": np.array([9.0]),
            "zss": np.array([10.0]),
            "lcc": np.array([11.0]),
            "lss": np.array([12.0]),
        }

    def fake_xc_from_mn_blocks(*, cfg, **kwargs):
        assert cfg is static.cfg
        return np.concatenate([np.asarray(value).reshape(-1) for value in kwargs.values() if value is not None])

    monkeypatch.setattr(diagnostics, "vmec_internal_mn_from_state", fake_internal_mn_from_state)
    monkeypatch.setattr(diagnostics, "vmec_xc_from_mn_blocks", fake_xc_from_mn_blocks)

    _maybe_dump_xc(
        state=state,
        vRcc=np.array([1.0]),
        vRss=np.array([2.0]),
        vZsc=np.array([3.0]),
        vZcs=np.array([4.0]),
        vLsc=np.array([5.0]),
        vLcs=np.array([6.0]),
        vRsc=np.array([7.0]),
        vRcs=np.array([8.0]),
        vZcc=np.array([9.0]),
        vZss=np.array([10.0]),
        vLcc=np.array([11.0]),
        vLss=np.array([12.0]),
        static=static,
        iter_idx=13,
    )
    with np.load(tmp_path / "xc_ns2_iter13.npz") as data:
        assert data["xc"].shape == (12,)
        np.testing.assert_allclose(data["xcdot"], np.arange(1.0, 13.0))
        np.testing.assert_allclose(data["v"], data["xcdot"])

    def fake_dtheta(*, coeff_cos, coeff_sin, **kwargs):
        return np.asarray(coeff_cos) + np.asarray(coeff_sin)

    def fake_dzeta_phys(*, coeff_cos, coeff_sin, **kwargs):
        return 2.0 * (np.asarray(coeff_cos) + np.asarray(coeff_sin))

    monkeypatch.setattr(realspace, "vmec_realspace_synthesis_dtheta", fake_dtheta)
    monkeypatch.setattr(realspace, "vmec_realspace_synthesis_dzeta_phys", fake_dzeta_phys)
    _maybe_dump_lulv(bc=_bc(), static=static, iter_idx=13, state=state, trig=SimpleNamespace())
    with np.load(tmp_path / "lulv_ns2_iter13.npz") as data:
        assert "lu_phys_m1" in data
        assert "lv_phys_rest" in data
        np.testing.assert_allclose(data["m_modes"], np.array([0, 1, 3]))
