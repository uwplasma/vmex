from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.config import VMECConfig
from vmec_jax.namelist import InData
from vmec_jax.static import build_static
import vmec_jax.vmec_bcovar as vb
import vmec_jax.vmec_forces as vf
import vmec_jax.vmec_realspace as vr
from vmec_jax.vmec_tomnsp import vmec_trig_tables
from vmec_jax.wout import equilibrium_iota_profiles_from_state


def _small_modes_and_trig(*, lasym: bool = False):
    cfg = VMECConfig(
        ns=2,
        mpol=2,
        ntor=1,
        nfp=2,
        lasym=lasym,
        lthreed=True,
        lconm1=True,
        ntheta=6,
        nzeta=3,
    )
    static = build_static(cfg)
    trig = vmec_trig_tables(
        ntheta=cfg.ntheta,
        nzeta=cfg.nzeta,
        nfp=cfg.nfp,
        mmax=cfg.mpol - 1,
        nmax=cfg.ntor,
        lasym=lasym,
        cache=False,
    )
    return static.modes, trig


def test_realspace_cache_guards_reject_stale_or_malformed_phase_payloads() -> None:
    modes, trig = _small_modes_and_trig(lasym=True)
    k = int(modes.K)
    valid_pair = (np.zeros((k, trig.ntheta3, trig.cosnv.shape[0])), np.ones((k, trig.ntheta3, trig.cosnv.shape[0])))
    valid_stack = np.zeros((2 * k, trig.ntheta3, trig.cosnv.shape[0]))

    assert vr._phase_cache_valid(valid_pair, modes, trig) is True
    assert vr._phase_stack_cache_valid(valid_stack, modes, trig) is True
    assert vr._phase_cache_valid(object(), modes, trig) is False
    assert vr._phase_cache_valid((np.zeros((k + 1, trig.ntheta3, trig.cosnv.shape[0])), valid_pair[1]), modes, trig) is False
    assert vr._phase_stack_cache_valid(object(), modes, trig) is False
    assert vr._phase_stack_cache_valid(np.zeros((2 * k + 1, trig.ntheta3, trig.cosnv.shape[0])), modes, trig) is False

    phase = np.zeros((2 * k, trig.ntheta3, trig.cosnv.shape[0]))
    trig_with_stack = SimpleNamespace(
        phase_stack=phase,
        phase_stack_m=modes.m,
        phase_stack_n=modes.n,
    )
    assert vr._phase_stack_from_trig(modes, trig_with_stack, "missing") is None
    assert vr._phase_stack_from_trig(modes, trig_with_stack, "phase_stack") is phase
    copied_modes = SimpleNamespace(m=modes.m.copy(), n=modes.n.copy())
    assert vr._phase_stack_from_trig(copied_modes, trig_with_stack, "phase_stack") is phase
    wrong_value_modes = SimpleNamespace(m=modes.m + 1, n=modes.n)
    assert vr._phase_stack_from_trig(wrong_value_modes, trig_with_stack, "phase_stack") is None


def test_realspace_scalxc_mode_cache_reuses_static_table() -> None:
    modes, _trig = _small_modes_and_trig(lasym=False)
    s = np.linspace(0.0, 1.0, 4)

    vr._SCALXC_MN_CACHE.clear()
    first = vr._scalxc_mn_for_s(s=s, modes=modes, m_np=np.asarray(modes.m), dtype=np.float64)
    second = vr._scalxc_mn_for_s(s=s, modes=modes, m_np=np.asarray(modes.m), dtype=np.float64)

    assert second is first
    assert np.asarray(first).shape == (4, int(modes.K))
    vr._SCALXC_MN_CACHE.clear()


@pytest.mark.parametrize(
    "fn",
    [
        vr.vmec_realspace_synthesis,
        vr.vmec_realspace_synthesis_multi,
        vr.vmec_realspace_synthesis_dtheta,
        vr.vmec_realspace_synthesis_dzeta_phys,
    ],
)
def test_realspace_synthesis_helpers_guard_coefficient_shapes(fn) -> None:
    modes, trig = _small_modes_and_trig(lasym=False)
    k = int(modes.K)
    good = np.ones((2, k))

    with pytest.raises(ValueError, match="Expected coeff arrays"):
        fn(coeff_cos=np.ones((k,)), coeff_sin=np.ones((k,)), modes=modes, trig=trig)
    with pytest.raises(ValueError, match="same shape"):
        fn(coeff_cos=good, coeff_sin=np.ones((2, k + 1)), modes=modes, trig=trig)
    with pytest.raises(ValueError, match="Mode count mismatch"):
        fn(coeff_cos=np.ones((2, k + 1)), coeff_sin=np.ones((2, k + 1)), modes=modes, trig=trig)

    out = fn(
        coeff_cos=np.ones((1, k)),
        coeff_sin=np.zeros((1, k)),
        modes=modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
    )
    if isinstance(out, tuple):
        assert len(out) == 3
        assert all(np.asarray(item).shape == (1, trig.ntheta3, trig.cosnv.shape[0]) for item in out)
    else:
        assert np.asarray(out).shape == (1, trig.ntheta3, trig.cosnv.shape[0])


def test_realspace_multi_and_analysis_reject_invalid_requests() -> None:
    modes, trig = _small_modes_and_trig(lasym=False)
    k = int(modes.K)

    with pytest.raises(ValueError, match="Unknown deriv"):
        vr.vmec_realspace_synthesis_multi(
            coeff_cos=np.ones((2, k)),
            coeff_sin=np.zeros((2, k)),
            modes=modes,
            trig=trig,
            derivs=("base", "bad"),
        )

    with pytest.raises(ValueError, match="Expected f"):
        vr.vmec_realspace_analysis(f=np.ones((2, 2)), modes=modes, trig=trig)
    with pytest.raises(ValueError, match="theta grid"):
        vr.vmec_realspace_analysis(f=np.ones((2, trig.ntheta2 - 1, trig.cosnv.shape[0])), modes=modes, trig=trig)
    with pytest.raises(ValueError, match="zeta grid"):
        vr.vmec_realspace_analysis(f=np.ones((2, trig.ntheta2, trig.cosnv.shape[0] + 1)), modes=modes, trig=trig)
    with pytest.raises(ValueError, match="parity"):
        vr.vmec_realspace_analysis(f=np.ones((2, trig.ntheta2, trig.cosnv.shape[0])), modes=modes, trig=trig, parity="bad")

    _, full_trig = _small_modes_and_trig(lasym=True)
    with pytest.raises(ValueError, match="theta grid"):
        vr.vmec_realspace_analysis(
            f=np.ones((2, full_trig.ntheta3 - 1, full_trig.cosnv.shape[0])),
            modes=modes,
            trig=full_trig,
        )


def test_bcovar_profile_log_and_lambda_axis_noop_mask(capsys, monkeypatch) -> None:
    monkeypatch.delenv("VMEC_JAX_PROFILE_BCOVAR", raising=False)
    assert vb._vmec_bcovar_profile_enabled() is False
    vb._vmec_bcovar_profile_log("hidden", extra=1)
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("VMEC_JAX_PROFILE_BCOVAR", "yes")
    assert vb._vmec_bcovar_profile_enabled() is True
    vb._vmec_bcovar_profile_log("visible", start=0.0, extra=2)
    out = capsys.readouterr().out
    assert "[vmec_jax bcovar]" in out
    assert "'stage': 'visible'" in out
    assert "'extra': 2" in out

    lsin = np.asarray([[1.0, 2.0], [10.0, 20.0]])
    closed = vb._apply_vmec_lambda_axis_closure(
        Lsin=lsin,
        m_modes=np.asarray([0, 0]),
        n_modes=np.asarray([1, 2]),
        axis_copy_mask=np.asarray([False, False]),
        lthreed=True,
        ntor=2,
    )
    np.testing.assert_allclose(np.asarray(closed), lsin)


def test_force_profile_and_iter_list_empty_chunks(capsys, monkeypatch) -> None:
    monkeypatch.delenv("VMEC_JAX_PROFILE_FORCE", raising=False)
    assert vf._vmec_force_profile_enabled() is False
    vf._vmec_force_profile_log("hidden", extra=1)
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("VMEC_JAX_PROFILE_FORCE", "true")
    assert vf._vmec_force_profile_enabled() is True
    vf._vmec_force_profile_log("visible", start=0.0, extra=2)
    out = capsys.readouterr().out
    assert "[vmec_jax force]" in out
    assert "'stage': 'visible'" in out
    assert "'extra': 2" in out
    assert vf._parse_iter_list("1,,3-2,bad-range,5") == {1, 2, 3, 5}


def test_equilibrium_iota_profile_single_surface_iota_driven_branch() -> None:
    cfg = VMECConfig(
        ns=1,
        mpol=2,
        ntor=0,
        nfp=1,
        lasym=False,
        lthreed=False,
        lconm1=False,
        ntheta=6,
        nzeta=1,
    )
    static = build_static(cfg)
    shape = (cfg.ns, int(static.modes.K))
    state = SimpleNamespace(
        Rcos=np.zeros(shape),
        Rsin=np.zeros(shape),
        Zcos=np.zeros(shape),
        Zsin=np.zeros(shape),
        Lcos=np.zeros(shape),
        Lsin=np.zeros(shape),
    )
    idx00 = int(np.flatnonzero((np.asarray(static.modes.m) == 0) & (np.asarray(static.modes.n) == 0))[0])
    state.Rcos[:, idx00] = 1.0
    indata = InData(
        scalars={
            "NCURR": 0,
            "PHIEDGE": 2.0 * np.pi,
            "PIOTA_TYPE": "power_series",
            "AI": [0.4],
        },
        indexed={},
    )

    chips, iotas, iotaf = equilibrium_iota_profiles_from_state(state=state, static=static, indata=indata, signgs=1)

    np.testing.assert_allclose(np.asarray(chips), [0.0])
    np.testing.assert_allclose(np.asarray(iotas), [0.0])
    np.testing.assert_allclose(np.asarray(iotaf), [0.0])
