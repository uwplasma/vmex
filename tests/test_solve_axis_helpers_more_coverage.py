from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import has_jax
from vmec_jax.solvers.fixed_boundary.diagnostics.axis_reset import (
    bad_jacobian_from_tau_range,
    bad_jacobian_ptau_from_minmax,
    initial_force_physical_fsq,
    initial_axis_reset_runtime_decision,
    reset_axis_from_boundary,
)
from vmec_jax.solve import (
    _apply_vmec_lambda_axis_rules_to_state,
    _enforce_field_rows,
    _enforce_field_rows_np,
    _enforce_fixed_boundary_and_axis_np,
    _initial_axis_reset_decision,
    _merge_axis_reset_state,
    _write_axis_reset_dump,
)
from vmec_jax.state import StateLayout, VMECState


pytestmark = pytest.mark.skipif(not has_jax(), reason="solve axis helpers use JAX array operations")


def _state_from_base(base: np.ndarray, *, lasym: bool = False) -> VMECState:
    arr = np.asarray(base, dtype=float)
    layout = StateLayout(ns=int(arr.shape[0]), K=int(arr.shape[1]), lasym=lasym)
    return VMECState(
        layout=layout,
        Rcos=arr + 10.0,
        Rsin=arr + 20.0,
        Zcos=arr + 30.0,
        Zsin=arr + 40.0,
        Lcos=arr + 50.0,
        Lsin=arr + 60.0,
    )


def test_merge_axis_reset_state_uses_full_reset_fallback_and_cached_masks():
    state = _state_from_base(np.arange(6.0).reshape(2, 3))
    axis_state = _state_from_base(np.full((2, 3), 100.0))

    assert _merge_axis_reset_state(st=state, st_axis=axis_state, static=SimpleNamespace(), full_reset=True) is axis_state

    static_from_modes = SimpleNamespace(modes=SimpleNamespace(m=np.asarray([0, 1, 0])))
    merged = _merge_axis_reset_state(st=state, st_axis=axis_state, static=static_from_modes, full_reset=False)
    np.testing.assert_allclose(np.asarray(merged.Rcos)[:, [0, 2]], np.asarray(axis_state.Rcos)[:, [0, 2]])
    np.testing.assert_allclose(np.asarray(merged.Rcos)[:, 1], np.asarray(state.Rcos)[:, 1])
    np.testing.assert_allclose(np.asarray(merged.Lcos), np.asarray(state.Lcos))

    static_cached_mask = SimpleNamespace(
        modes=SimpleNamespace(m=np.asarray([9, 9, 9])),
        m_is_m0=np.asarray([0.0, 1.0, 0.0]),
    )
    cached = _merge_axis_reset_state(st=state, st_axis=axis_state, static=static_cached_mask, full_reset=False)
    np.testing.assert_allclose(np.asarray(cached.Zsin)[:, 1], np.asarray(axis_state.Zsin)[:, 1])
    np.testing.assert_allclose(np.asarray(cached.Zsin)[:, [0, 2]], np.asarray(state.Zsin)[:, [0, 2]])


def test_reset_axis_from_boundary_fallback_coefficients_preserve_non_axis_modes():
    class FakeIndata:
        def __init__(self, *, scalars, indexed):
            self.scalars = scalars
            self.indexed = indexed

    state = _state_from_base(np.arange(4.0).reshape(2, 2))
    static = SimpleNamespace(
        cfg=SimpleNamespace(ntor=1, ns=2),
        modes=SimpleNamespace(m=np.asarray([0, 1])),
    )
    indata = FakeIndata(scalars={"RAXIS_CC": [1.0], "ZAXIS_CS": [0.25]}, indexed={})

    def initial_guess_from_boundary(_static, _boundary, indata_local, *, dtype, infer_axis_if_missing):
        assert not infer_axis_if_missing
        raxis = float(indata_local.scalars["RAXIS_CC"][0])
        zaxis = float(indata_local.scalars["ZAXIS_CS"][0])
        layout = StateLayout(ns=2, K=2, lasym=False)
        rcos = np.full((2, 2), 110.0, dtype=dtype)
        zsin = np.full((2, 2), 140.0, dtype=dtype)
        rcos[0, 0] = raxis
        zsin[0, 0] = zaxis
        return VMECState(
            layout=layout,
            Rcos=rcos,
            Rsin=np.full((2, 2), 120.0, dtype=dtype),
            Zcos=np.full((2, 2), 130.0, dtype=dtype),
            Zsin=zsin,
            Lcos=np.full((2, 2), 150.0, dtype=dtype),
            Lsin=np.full((2, 2), 160.0, dtype=dtype),
        )

    def read_axis_coeffs(_indata):
        return {"RAXIS_CC": [1.0], "ZAXIS_CS": [0.25]}

    def recompute_axis_from_boundary(_static, _boundary, *, raxis_cc, zaxis_cs, signgs):
        assert signgs == 1
        return np.asarray([2.0, 3.0]), np.asarray([4.0, 5.0])

    out, coeffs = reset_axis_from_boundary(
        state,
        boundary_for_axis=object(),
        static=static,
        indata=indata,
        signgs=1,
        trig=object(),
        full_reset=False,
        zero_precond_diag=None,
        zero_tcon=None,
        constraint_active_false=False,
        compute_forces_iter_func=lambda *args, **kwargs: None,
        apply_vmec_lambda_axis_rules_func=lambda st: st,
        initial_guess_from_boundary_func=initial_guess_from_boundary,
        read_axis_coeffs_func=read_axis_coeffs,
        recompute_axis_from_state_vmec_func=lambda *args, **kwargs: None,
        recompute_axis_from_boundary_func=recompute_axis_from_boundary,
        axis_dump_dir="",
    )

    assert coeffs is not None
    np.testing.assert_allclose(coeffs[0], [2.0, 3.0])
    np.testing.assert_allclose(coeffs[3], [4.0, 5.0])
    np.testing.assert_allclose(np.asarray(out.Rcos)[:, 0], [2.0, 110.0])
    np.testing.assert_allclose(np.asarray(out.Rcos)[:, 1], np.asarray(state.Rcos)[:, 1])
    np.testing.assert_allclose(np.asarray(out.Lcos), np.asarray(state.Lcos))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                bad_jacobian_ptau=None,
                bad_jacobian_state=True,
                badjac_use_state=False,
                fsq_phys=2.0,
                axis_reset_fsq_min=0.0,
                force_axis_reset=False,
                axis_reset_always_3d=False,
                lthreed=False,
            ),
            (True, False, True),
        ),
        (
            dict(
                bad_jacobian_ptau=True,
                bad_jacobian_state=False,
                badjac_use_state=True,
                fsq_phys=2.0,
                axis_reset_fsq_min=0.0,
                force_axis_reset=False,
                axis_reset_always_3d=False,
                lthreed=True,
            ),
            (False, False, False),
        ),
        (
            dict(
                bad_jacobian_ptau=True,
                bad_jacobian_state=True,
                badjac_use_state=False,
                fsq_phys=0.25,
                axis_reset_fsq_min=1.0,
                force_axis_reset=False,
                axis_reset_always_3d=False,
                lthreed=True,
            ),
            (False, False, False),
        ),
        (
            dict(
                bad_jacobian_ptau=False,
                bad_jacobian_state=False,
                badjac_use_state=False,
                fsq_phys=None,
                axis_reset_fsq_min=1.0,
                force_axis_reset=False,
                axis_reset_always_3d=True,
                lthreed=True,
            ),
            (False, True, True),
        ),
        (
            dict(
                bad_jacobian_ptau=True,
                bad_jacobian_state=True,
                badjac_use_state=False,
                fsq_phys=np.inf,
                axis_reset_fsq_min=1.0,
                force_axis_reset=True,
                axis_reset_always_3d=False,
                lthreed=True,
                axis_reset_enabled=False,
            ),
            (False, True, False),
        ),
    ],
)
def test_initial_axis_reset_decision_branch_matrix(kwargs, expected):
    decision = _initial_axis_reset_decision(**kwargs)
    assert (decision.bad_jacobian, decision.force_reset, decision.reset) == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                bad_jacobian=True,
                fsq_phys=0.25,
                axis_reset_fsq_min=1.0,
                force_axis_reset=False,
                axis_reset_always_3d=False,
                lthreed=True,
            ),
            (False, False, False, False),
        ),
        (
            dict(
                bad_jacobian=False,
                fsq_phys=np.inf,
                axis_reset_fsq_min=1.0,
                force_axis_reset=False,
                axis_reset_always_3d=False,
                lthreed=True,
            ),
            (False, True, False, True),
        ),
        (
            dict(
                bad_jacobian=False,
                fsq_phys=0.25,
                axis_reset_fsq_min=1.0,
                force_axis_reset=False,
                axis_reset_always_3d=True,
                lthreed=True,
            ),
            (False, False, True, True),
        ),
        (
            dict(
                bad_jacobian=True,
                fsq_phys=2.0,
                axis_reset_fsq_min=1.0,
                force_axis_reset=False,
                axis_reset_always_3d=False,
                lthreed=True,
                axis_reset_enabled=False,
            ),
            (True, False, False, False),
        ),
    ],
)
def test_initial_axis_reset_runtime_decision_preserves_in_loop_gate(kwargs, expected):
    decision = initial_axis_reset_runtime_decision(**kwargs)
    assert (
        decision.bad_jacobian,
        decision.huge_initial_forces,
        decision.force_reset,
        decision.reset,
    ) == expected


def test_initial_axis_reset_shared_bad_jacobian_helpers():
    norms = SimpleNamespace(r1=2.0, fnorm=3.0, fnormL=5.0)
    assert initial_force_physical_fsq(norms=norms, gcr2=0.5, gcz2=0.25, gcl2=0.1) == pytest.approx(5.0)
    assert initial_force_physical_fsq(norms=object(), gcr2=0.5, gcz2=0.25, gcl2=0.1) is None

    assert bad_jacobian_from_tau_range(min_tau=-1.0, max_tau=2.0)
    assert not bad_jacobian_from_tau_range(min_tau=-1.0e-4, max_tau=2.0, abs_tol=1.0e-3)

    assert bad_jacobian_ptau_from_minmax(
        ptau_min=-1.0,
        ptau_max=2.0,
        ptau_tol=0.0,
        ptau_tol_rel=0.0,
    )
    assert not bad_jacobian_ptau_from_minmax(
        ptau_min=-1.0e-4,
        ptau_max=2.0,
        ptau_tol=0.0,
        ptau_tol_rel=1.0e-3,
    )
    assert bad_jacobian_ptau_from_minmax(
        ptau_min=None,
        ptau_max=2.0,
        ptau_tol=0.0,
        ptau_tol_rel=0.0,
    ) is None


def test_write_axis_reset_dump_disabled_short_and_success_paths(tmp_path):
    assert not _write_axis_reset_dump(
        axis_dump_dir=" ",
        ns=3,
        ntor=1,
        used_state_guess=False,
        raxis_cc=[1.0, 2.0],
        raxis_cs=[0.0, 0.0],
        zaxis_cc=[0.0, 0.0],
        zaxis_cs=[0.0, 0.0],
    )

    assert not _write_axis_reset_dump(
        axis_dump_dir=tmp_path,
        ns=3,
        ntor=2,
        used_state_guess=False,
        raxis_cc=[1.0, 2.0],
        raxis_cs=[0.0, 0.0],
        zaxis_cc=[0.0, 0.0],
        zaxis_cs=[0.0, 0.0],
    )

    assert _write_axis_reset_dump(
        axis_dump_dir=tmp_path,
        ns=5,
        ntor=1,
        used_state_guess=True,
        raxis_cc=[1.0, 2.0],
        raxis_cs=[0.1, 0.2],
        zaxis_cc=[3.0, 4.0],
        zaxis_cs=[0.3, 0.4],
    )
    text = (tmp_path / "axis_reset_ns5.dat").read_text(encoding="utf-8")
    assert "# used_state_guess=1" in text
    assert "0" in text and "1" in text


def test_apply_vmec_lambda_axis_rules_host_device_and_disabled_paths():
    state = _state_from_base(np.arange(9.0).reshape(3, 3))

    assert (
        _apply_vmec_lambda_axis_rules_to_state(
            state,
            enforce_vmec_lambda_axis=False,
            host_update_assembly=True,
            idx00=1,
        )
        is state
    )

    host_invalid = _apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=True,
        idx00=10,
    )
    np.testing.assert_allclose(np.asarray(host_invalid.Lcos), np.asarray(state.Lcos))

    host = _apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=True,
        idx00=1,
    )
    np.testing.assert_allclose(np.asarray(host.Lcos)[:, 1], 0.0)
    np.testing.assert_allclose(np.asarray(host.Lsin)[:, 1], 0.0)

    device = _apply_vmec_lambda_axis_rules_to_state(
        state,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=False,
        idx00=2,
    )
    np.testing.assert_allclose(np.asarray(device.Lcos)[:, 2], 0.0)
    np.testing.assert_allclose(np.asarray(device.Lsin)[:, 2], 0.0)


def test_enforce_field_rows_jax_handles_empty_single_and_combined_constraints():
    empty = _enforce_field_rows(np.empty((0, 2)), axis_mask=np.ones(2), edge_row=np.arange(2.0))
    assert np.asarray(empty).shape == (0, 2)

    single = _enforce_field_rows(
        np.asarray([[9.0, 9.0, 9.0]]),
        axis_mask=np.asarray([1.0, 0.0, 1.0]),
        edge_row=np.asarray([3.0, 4.0, 5.0]),
    )
    np.testing.assert_allclose(np.asarray(single), [[3.0, 0.0, 5.0]])

    single_zero = _enforce_field_rows(
        np.asarray([[9.0, 9.0, 9.0]]),
        axis_mask=np.asarray([1.0, 0.0, 1.0]),
        edge_row=np.asarray([3.0, 4.0, 5.0]),
        zero_axis=True,
    )
    np.testing.assert_allclose(np.asarray(single_zero), [[0.0, 0.0, 0.0]])

    base = np.arange(12.0).reshape(4, 3)
    combined = _enforce_field_rows(
        base,
        axis_mask=np.asarray([1.0, 0.0, 1.0]),
        edge_row=np.asarray([100.0, 101.0, 102.0]),
    )
    np.testing.assert_allclose(np.asarray(combined)[0], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(np.asarray(combined)[-1], [100.0, 101.0, 102.0])
    np.testing.assert_allclose(np.asarray(_enforce_field_rows(base, zero_axis=True))[0], 0.0)
    np.testing.assert_allclose(np.asarray(_enforce_field_rows(base, edge_row=[7.0, 8.0, 9.0]))[-1], [7.0, 8.0, 9.0])


def test_enforce_field_rows_np_handles_empty_single_and_combined_constraints():
    empty = _enforce_field_rows_np(np.empty((0, 2)), axis_mask=np.ones(2), edge_row=np.arange(2.0))
    assert empty.shape == (0, 2)

    single = _enforce_field_rows_np(
        np.asarray([[9.0, 9.0, 9.0]]),
        axis_mask=np.asarray([1.0, 0.0, 1.0]),
        edge_row=np.asarray([3.0, 4.0, 5.0]),
    )
    np.testing.assert_allclose(single, [[3.0, 0.0, 5.0]])

    single_zero = _enforce_field_rows_np(
        np.asarray([[9.0, 9.0, 9.0]]),
        edge_row=np.asarray([3.0, 4.0, 5.0]),
        zero_axis=True,
    )
    np.testing.assert_allclose(single_zero, [[0.0, 0.0, 0.0]])

    base = np.arange(12.0).reshape(4, 3)
    combined = _enforce_field_rows_np(
        base,
        axis_mask=np.asarray([1.0, 0.0, 1.0]),
        edge_row=np.asarray([100.0, 101.0, 102.0]),
    )
    np.testing.assert_allclose(combined[0], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(combined[-1], [100.0, 101.0, 102.0])
    np.testing.assert_allclose(_enforce_field_rows_np(base, zero_axis=True)[0], 0.0)


def test_enforce_fixed_boundary_and_axis_np_respects_toggles_and_precomputed_mask():
    state = _state_from_base(np.arange(12.0).reshape(4, 3))
    static = SimpleNamespace(modes=SimpleNamespace(m=np.asarray([9, 9, 9])))
    mask = np.asarray([1.0, 0.0, 1.0])

    constrained = _enforce_fixed_boundary_and_axis_np(
        state,
        static,
        edge_Rcos=np.asarray([100.0, 101.0, 102.0]),
        edge_Rsin=np.asarray([110.0, 111.0, 112.0]),
        edge_Zcos=np.asarray([120.0, 121.0, 122.0]),
        edge_Zsin=np.asarray([130.0, 131.0, 132.0]),
        enforce_axis=True,
        enforce_edge=True,
        enforce_lambda_axis=True,
        idx00=2,
        precomputed_axis_mask=mask,
    )
    np.testing.assert_allclose(constrained.Rcos[0], [10.0, 0.0, 12.0])
    np.testing.assert_allclose(constrained.Rsin[-1], [110.0, 111.0, 112.0])
    np.testing.assert_allclose(constrained.Zsin[-1], [130.0, 131.0, 132.0])
    np.testing.assert_allclose(constrained.Lcos[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(constrained.Lcos[:, 2], 0.0)
    np.testing.assert_allclose(constrained.Lsin[:, 2], 0.0)

    unconstrained = _enforce_fixed_boundary_and_axis_np(
        state,
        static,
        edge_Rcos=np.asarray([100.0, 101.0, 102.0]),
        edge_Rsin=np.asarray([110.0, 111.0, 112.0]),
        edge_Zcos=np.asarray([120.0, 121.0, 122.0]),
        edge_Zsin=np.asarray([130.0, 131.0, 132.0]),
        enforce_axis=False,
        enforce_edge=False,
        enforce_lambda_axis=False,
        idx00=None,
        precomputed_axis_mask=mask,
    )
    np.testing.assert_allclose(unconstrained.Rcos, np.asarray(state.Rcos))
    np.testing.assert_allclose(unconstrained.Lcos, np.asarray(state.Lcos))
