from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import jnp
from vmec_jax.solvers.fixed_boundary.scan.math import (
    _hold_step,
    _no_restart_updates,
    _ptau_minmax_from_k_host,
    _ptau_minmax_from_k_jax,
    _restart_updates,
    _state_jacobian,
)


def _pshalf_from_s_jax(s_arr, dtype):
    s_arr = jnp.asarray(s_arr, dtype=dtype)
    sh = 0.5 * (s_arr[1:] + s_arr[:-1])
    return jnp.sqrt(jnp.maximum(jnp.concatenate([sh[:1], sh], axis=0), jnp.asarray(0.0, dtype=dtype)))


def _kernel(ns: int = 3, *, nan: bool = False) -> SimpleNamespace:
    zeros = np.zeros((ns, 1, 1))
    pz1_even = np.asarray([0.0, 1.0, 3.0], dtype=float)[:ns, None, None]
    if nan and ns > 1:
        pz1_even = pz1_even.copy()
        pz1_even[1, 0, 0] = np.nan
    return SimpleNamespace(
        pru_even=np.full((ns, 1, 1), 2.0),
        pru_odd=zeros,
        pzu_even=zeros,
        pzu_odd=zeros,
        pr1_even=zeros,
        pr1_odd=zeros,
        pz1_even=pz1_even,
        pz1_odd=zeros,
    )


def test_ptau_minmax_host_computes_normal_missing_short_and_nan_paths():
    pshalf = np.ones(3)
    assert _ptau_minmax_from_k_host(_kernel(), pshalf=pshalf, ohs=2.0) == pytest.approx((4.0, 8.0))
    assert _ptau_minmax_from_k_host(SimpleNamespace(), pshalf=pshalf, ohs=2.0) == (None, None)
    assert _ptau_minmax_from_k_host(_kernel(1), pshalf=np.ones(1), ohs=2.0) == (None, None)

    ptau_min, ptau_max = _ptau_minmax_from_k_host(_kernel(nan=True), pshalf=pshalf, ohs=2.0)
    assert np.isnan(ptau_min)
    assert np.isnan(ptau_max)


def test_ptau_minmax_jax_matches_host_and_returns_nan_for_missing_or_short_kernel():
    pytest.importorskip("jax")
    s = jnp.asarray([0.0, 0.5, 1.0])
    ptau_min, ptau_max = _ptau_minmax_from_k_jax(_kernel(), s=s, pshalf_from_s_jax=_pshalf_from_s_jax)
    np.testing.assert_allclose(np.asarray((ptau_min, ptau_max)), np.asarray((4.0, 8.0)))

    missing = _ptau_minmax_from_k_jax(SimpleNamespace(), s=s, pshalf_from_s_jax=_pshalf_from_s_jax)
    assert np.isnan(float(np.asarray(missing[0])))
    short = _ptau_minmax_from_k_jax(_kernel(1), s=jnp.asarray([0.0]), pshalf_from_s_jax=_pshalf_from_s_jax)
    assert np.isnan(float(np.asarray(short[0])))


def test_state_jacobian_ignores_axis_point_and_handles_vmec2000_relative_and_nan_paths():
    decision = _state_jacobian(np.asarray([99.0, -1.0, 2.0]), vmec2000_control=False, ptau_tol=1.0e-9)
    assert bool(np.asarray(decision.bad_jacobian))
    assert float(np.asarray(decision.min_tau)) == pytest.approx(-1.0)
    assert float(np.asarray(decision.max_tau)) == pytest.approx(2.0)

    loose = _state_jacobian(
        np.asarray([99.0, -1.0e-3, 2.0]),
        vmec2000_control=True,
        ptau_tol=1.0e-9,
        relative_tol=1.0e-2,
    )
    assert not bool(np.asarray(loose.bad_jacobian))

    with_nan = _state_jacobian(np.asarray([0.0, np.nan, 2.0]), vmec2000_control=False, ptau_tol=0.0)
    assert not bool(np.asarray(with_nan.bad_jacobian))
    empty = _state_jacobian(np.asarray([]), vmec2000_control=False, ptau_tol=0.0)
    assert not bool(np.asarray(empty.bad_jacobian))
    assert np.isnan(float(np.asarray(empty.min_tau)))


def _carry() -> SimpleNamespace:
    one = np.asarray([1.0, 2.0])
    return SimpleNamespace(
        state="state",
        time_step=np.asarray(2.0),
        inv_tau=np.asarray([0.1, 0.2]),
        fsq_prev=np.asarray(3.0),
        fsqr_prev_phys=np.asarray(4.0),
        fsqz_prev_phys=np.asarray(5.0),
        fsql_prev_phys=np.asarray(6.0),
        fsqr1_prev=np.asarray(7.0),
        fsqz1_prev=np.asarray(8.0),
        fsql1_prev=np.asarray(9.0),
        r00_prev=np.asarray(10.0),
        z00_prev=np.asarray(11.0),
        w_mhd_prev=np.asarray(12.0),
        res0=np.asarray(13.0),
        res1=np.asarray(14.0),
        iter1=np.asarray(15),
        iter_offset=np.asarray(16),
        ijacob=np.asarray(17),
        bad_resets=np.asarray(18),
        bad_growth=np.asarray(19),
        vRcc=one,
        vRss=one + 1,
        vZsc=one + 2,
        vZcs=one + 3,
        vLsc=one + 4,
        vLcs=one + 5,
        vRsc=one + 6,
        vRcs=one + 7,
        vZcc=one + 8,
        vZss=one + 9,
        vLcc=one + 10,
        vLss=one + 11,
    )


def test_hold_step_emits_minimal_light_and_full_rejected_histories():
    carry = _carry()
    hist_min = lambda r, z, l: ("min", r, z, l)
    hist_light = lambda *args: ("light",) + args

    assert _hold_step(
        carry,
        dtype=jnp.float64,
        state_only_scan=True,
        scan_minimal=False,
        scan_light=False,
        scan_hist_min=hist_min,
        scan_hist_light=hist_light,
    ) == (carry, ())
    assert _hold_step(
        carry,
        dtype=jnp.float64,
        state_only_scan=False,
        scan_minimal=True,
        scan_light=False,
        scan_hist_min=hist_min,
        scan_hist_light=hist_light,
    )[1][0] == "min"
    light = _hold_step(
        carry,
        dtype=jnp.float64,
        state_only_scan=False,
        scan_minimal=False,
        scan_light=True,
        scan_hist_min=hist_min,
        scan_hist_light=hist_light,
    )[1]
    assert light[0] == "light"
    assert not bool(np.asarray(light[4]))

    full = _hold_step(
        carry,
        dtype=jnp.float64,
        state_only_scan=False,
        scan_minimal=False,
        scan_light=False,
        scan_hist_min=hist_min,
        scan_hist_light=hist_light,
    )[1]
    assert len(full) == 25
    assert not bool(np.asarray(full[6]))
    assert np.isnan(float(np.asarray(full[19])))


def test_restart_updates_zero_all_velocity_blocks_including_lasym_and_no_restart_preserves_them():
    carry = _carry()

    def transition_fn(**_kwargs):
        return SimpleNamespace(
            time_step=np.asarray(0.5),
            damping_time_step=np.asarray(0.25),
            iter_offset=np.asarray(20),
            iter1=np.asarray(21),
            ijacob=np.asarray(22),
            bad_resets=np.asarray(23),
            bad_growth=np.asarray(0),
            force_bcovar_update=np.asarray(True),
        )

    updated = _restart_updates(
        carry_adv=carry,
        state_checkpoint="checkpoint",
        fsq_prev_before=np.asarray(99.0),
        iter2=np.asarray(3),
        restart_reason=np.asarray(1),
        vmec2000_control=True,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        stage_transition_scale=0.5,
        step_size=0.1,
        k_ndamp=3,
        dtype=jnp.float64,
        scan_restart_transition_fn=transition_fn,
    )
    assert updated.state == "checkpoint"
    np.testing.assert_allclose(np.asarray(updated.inv_tau), np.full(3, 0.6))
    for block in (updated.vRcc, updated.vRss, updated.vRsc, updated.vRcs, updated.vZcc, updated.vLss):
        np.testing.assert_allclose(np.asarray(block), 0.0)
    assert bool(np.asarray(updated.force_bcovar_update))

    preserved = _no_restart_updates(carry)
    np.testing.assert_allclose(np.asarray(preserved.vRsc), np.asarray(carry.vRsc))
    assert not bool(np.asarray(preserved.force_bcovar_update))
