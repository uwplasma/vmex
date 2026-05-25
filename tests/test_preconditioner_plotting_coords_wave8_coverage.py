from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.config import VMECConfig
from vmec_jax.coords import Coords, eval_coords
from vmec_jax.modes import ModeTable, vmec_mode_table
from vmec_jax.plotting import (
    axis_rz_from_wout,
    select_zeta_slices,
    surface_data_from_wout,
    vmecplot2_cross_section_indices,
    zeta_grid_field_period,
)
from vmec_jax.preconditioner_1d import lambda_preconditioner
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.static import build_static
from vmec_jax.vmec_realspace import (
    _PHASE_CACHE,
    _phase_cache_key,
    _phase_stack_from_trig,
    _vmec_phase_tables_cached,
    vmec_realspace_synthesis,
    vmec_realspace_synthesis_multi,
)
from vmec_jax.vmec_tomnsp import vmec_trig_tables


def _cfg(*, mpol=3, ntor=1, ns=4, nfp=2, lasym=False, lthreed=True, ntheta=8, nzeta=5):
    return VMECConfig(
        mpol=mpol,
        ntor=ntor,
        ns=ns,
        nfp=nfp,
        lasym=lasym,
        lthreed=lthreed,
        lconm1=True,
        ntheta=ntheta,
        nzeta=nzeta,
    )


def _state_for_static(static) -> VMECState:
    ns = int(static.cfg.ns)
    K = int(static.modes.K)
    layout = StateLayout(ns=ns, K=K, lasym=bool(static.cfg.lasym))
    Rcos = np.zeros((ns, K), dtype=float)
    Rsin = np.zeros_like(Rcos)
    Zcos = np.zeros_like(Rcos)
    Zsin = np.zeros_like(Rcos)
    Lcos = np.zeros_like(Rcos)
    Lsin = np.zeros_like(Rcos)
    k00 = int(np.flatnonzero((static.modes.m == 0) & (static.modes.n == 0))[0])
    k10 = int(np.flatnonzero((static.modes.m == 1) & (static.modes.n == 0))[0])
    Rcos[:, k00] = 2.0 + 0.1 * np.arange(ns)
    Rcos[:, k10] = np.linspace(0.0, 0.3, ns)
    Zsin[:, k10] = np.linspace(0.0, 0.2, ns)
    Lsin[:, k10] = np.linspace(0.0, 0.05, ns)
    return VMECState(layout, Rcos, Rsin, Zcos, Zsin, Lcos, Lsin)


def test_static_can_skip_vmec_phase_cache_without_changing_mode_invariants(monkeypatch) -> None:
    monkeypatch.setenv("VMEC_JAX_CACHE_VMEC_PHASE", "0")
    static = build_static(_cfg(mpol=3, ntor=2, ns=3, nfp=3, lasym=True, ntheta=10, nzeta=7))

    assert static.trig_vmec is not None
    assert _phase_stack_from_trig(static.modes, static.trig_vmec, "phase_stack") is None
    np.testing.assert_array_equal(static.m_np, np.asarray(static.modes.m))
    np.testing.assert_array_equal(static.n_np, np.asarray(static.modes.n))
    np.testing.assert_array_equal(static.m_is_even, (static.m_np % 2) == 0)
    np.testing.assert_array_equal(static.m_is_odd, (static.m_np % 2) == 1)
    np.testing.assert_array_equal(static.lambda_axis_copy_mask, (static.m_np == 0) & (static.n_np > 0))

    for n in range(static.cfg.ntor + 1):
        k = int(static.m0_n_index[n])
        assert k >= 0
        assert int(static.m_np[k]) == 0
        assert int(static.n_np[k]) == n

    expected_scale = 1.0 / (
        np.asarray(static.trig_vmec.mscale)[static.m_np]
        * np.asarray(static.trig_vmec.nscale)[np.abs(static.n_np)]
    )
    np.testing.assert_allclose(static.mode_scale_internal, expected_scale)


def test_coords_tree_roundtrip_and_axisymmetric_derivative_identities() -> None:
    static = build_static(_cfg(mpol=2, ntor=0, ns=3, nfp=1, lasym=False, lthreed=False, ntheta=8, nzeta=1))
    state = _state_for_static(static)

    coords = eval_coords(state, static.basis)
    children, aux = coords.tree_flatten()
    rebuilt = Coords.tree_unflatten(aux, children)

    for original, roundtrip in zip(children, rebuilt.tree_flatten()[0], strict=True):
        np.testing.assert_allclose(np.asarray(roundtrip), np.asarray(original))

    # R = R0(s) + a(s) cos(theta), Z = b(s) sin(theta) has no toroidal variation.
    np.testing.assert_allclose(np.asarray(coords.R_phi), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(coords.Z_phi), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(coords.L_phi), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.mean(np.asarray(coords.R_theta), axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.mean(np.asarray(coords.Z_theta), axis=1), 0.0, atol=1e-14)


def test_lambda_preconditioner_uses_threed_cross_metric_and_axis_mask() -> None:
    cfg2d = SimpleNamespace(mpol=2, ntor=1, ntheta=4, nzeta=1, nfp=2, lasym=False, lthreed=False)
    cfg3d = SimpleNamespace(**{**cfg2d.__dict__, "lthreed": True})
    ns = 4
    shape = (ns, 3, 1)
    radial = np.arange(ns, dtype=float)[:, None, None]
    theta = np.arange(3, dtype=float)[None, :, None]
    bc = SimpleNamespace(
        guu=2.0 + 0.1 * radial + np.zeros(shape),
        guv=0.25 + 0.05 * radial + 0.01 * theta,
        gvv=3.0 + 0.2 * radial + np.zeros(shape),
        jac=SimpleNamespace(sqrtg=1.0 + 0.02 * radial + np.zeros(shape)),
        lamscale=1.5,
    )
    s = np.linspace(0.0, 1.0, ns)

    lam2d, debug2d = lambda_preconditioner(bc=bc, trig=None, s=s, cfg=cfg2d, r0scale=1.0, return_debug=True)
    lam3d, debug3d = lambda_preconditioner(bc=bc, trig=None, s=s, cfg=cfg3d, r0scale=1.0, return_debug=True)

    np.testing.assert_allclose(debug2d["dlam_pre"], 0.0)
    assert np.all(debug3d["dlam_pre"][1:] > 0.0)
    assert not np.allclose(lam2d[1:, 1, 1], lam3d[1:, 1, 1])
    np.testing.assert_allclose(lam3d[0, 1:, :], 0.0)
    np.testing.assert_allclose(lam3d[0, 0, 1:], 0.0)

    p_factor00 = 2.0 / (4.0 * bc.lamscale * bc.lamscale) * bc.lamscale * bc.lamscale
    expected_m00 = p_factor00 / np.where(debug3d["blam_post"] != 0.0, debug3d["blam_post"], -1.0e-10)
    np.testing.assert_allclose(lam3d[:, 0, 0], expected_m00)


def test_plotting_grids_and_surface_data_preserve_periodic_geometry() -> None:
    zeta = zeta_grid_field_period(4, nfp=2)
    np.testing.assert_allclose(zeta, [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0])
    np.testing.assert_allclose(zeta_grid_field_period(3, nfp=0), [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    np.testing.assert_array_equal(vmecplot2_cross_section_indices(8), [0, 2, 4, 6])
    with pytest.raises(ValueError, match="nzeta>=8"):
        vmecplot2_cross_section_indices(6)
    with pytest.raises(ValueError, match="positive"):
        select_zeta_slices(np.arange(3.0), n=0)

    wout = SimpleNamespace(
        nfp=2,
        lasym=False,
        xm=np.asarray([0, 1]),
        xn=np.asarray([0, 0]),
        xm_nyq=np.asarray([0, 1]),
        xn_nyq=np.asarray([0, 0]),
        rmnc=np.asarray([[1.5, 0.25]]),
        rmns=np.asarray([[100.0, 100.0]]),
        zmns=np.asarray([[0.0, 0.5]]),
        zmnc=np.asarray([[100.0, 100.0]]),
        bmnc=np.asarray([[2.0, 0.125]]),
        bmns=np.asarray([[100.0, 100.0]]),
        raxis_cc=np.asarray([1.0, 0.2]),
        raxis_cs=np.asarray([0.0, 0.3]),
        zaxis_cs=np.asarray([0.0, 0.4]),
        zaxis_cc=np.asarray([0.1, 0.5]),
    )
    theta = np.asarray([0.0, 0.5 * np.pi, np.pi])
    surf = surface_data_from_wout(wout, theta=theta, zeta=np.asarray([0.0]), s_index=0, with_bmag=True)
    np.testing.assert_allclose(surf.R[:, 0], [1.75, 1.5, 1.25], atol=1e-14)
    np.testing.assert_allclose(surf.Z[:, 0], [0.0, 0.5, 0.0], atol=1e-14)
    np.testing.assert_allclose(surf.B[:, 0], [2.125, 2.0, 1.875], atol=1e-14)

    R_axis, Z_axis = axis_rz_from_wout(wout, zeta=np.asarray([0.0, np.pi / 4.0]))
    np.testing.assert_allclose(R_axis, [1.2, 0.7], atol=1e-14)
    np.testing.assert_allclose(Z_axis, [0.6, -0.3], atol=1e-14)


def test_vmec_realspace_cache_validation_and_stacked_fallback_equivalence() -> None:
    modes = vmec_mode_table(3, 1)
    trig = vmec_trig_tables(ntheta=10, nzeta=5, nfp=2, mmax=2, nmax=1, lasym=True, dtype=np.float64, cache=False)
    ns = 2
    coeff_cos = np.arange(ns * modes.K, dtype=float).reshape(ns, modes.K) * 0.01
    coeff_sin = -0.5 * coeff_cos
    coeff_sin[:, 0] = 0.0

    key = _phase_cache_key(modes, trig)
    _PHASE_CACHE[key] = (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))
    cos_phase, sin_phase = _vmec_phase_tables_cached(modes=modes, trig=trig, cache=True)
    assert cos_phase.shape == (modes.K, trig.ntheta3, trig.cosnv.shape[0])
    assert sin_phase.shape == cos_phase.shape
    assert _PHASE_CACHE[key][0].shape == cos_phase.shape

    copied_modes = ModeTable(m=modes.m.copy(), n=modes.n.copy())
    phase_stack = np.ones((2 * modes.K, trig.ntheta3, trig.cosnv.shape[0]))
    trig_with_stack = replace(trig, phase_stack=phase_stack, phase_stack_m=modes.m, phase_stack_n=modes.n)
    assert _phase_stack_from_trig(modes, trig_with_stack, "phase_stack") is phase_stack
    assert _phase_stack_from_trig(copied_modes, trig_with_stack, "phase_stack") is phase_stack
    wrong_value_modes = ModeTable(m=modes.m + 1, n=modes.n)
    assert _phase_stack_from_trig(wrong_value_modes, trig_with_stack, "phase_stack") is None

    stacked = vmec_realspace_synthesis(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=modes,
        trig=trig,
        use_stacked_dot=True,
    )
    fallback = vmec_realspace_synthesis(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=modes,
        trig=trig,
        use_stacked_dot=False,
    )
    np.testing.assert_allclose(np.asarray(stacked), np.asarray(fallback), rtol=1e-13, atol=1e-13)

    with pytest.raises(ValueError, match="Unknown deriv"):
        vmec_realspace_synthesis_multi(
            coeff_cos=coeff_cos,
            coeff_sin=coeff_sin,
            modes=modes,
            trig=trig,
            derivs=("base", "bad"),
        )
