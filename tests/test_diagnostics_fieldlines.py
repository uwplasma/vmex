from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.diagnostics import (
    print_jacobian_stats,
    print_summary,
    slice_excluding_axis,
    summarize_array,
    summarize_many,
    vmec_internal_mn_from_state,
    vmec_xc_from_mn_blocks,
)
from vmec_jax.fieldlines import _bilinear_periodic, trace_fieldline_on_surface
from vmec_jax.modes import vmec_mode_table


def test_diagnostic_summaries_handle_empty_nonfinite_and_printing(capsys):
    empty = summarize_array("empty", np.asarray([]))
    assert empty.shape == (0,)
    assert np.isnan(empty.mean)
    assert all(np.isnan(v) for v in empty.q)

    summary = summarize_array("arr", np.asarray([0.0, 1.0, np.nan, np.inf, -2.0]))
    assert summary.n_nan == 1
    assert summary.n_inf == 1
    assert summary.n_zero == 1
    assert summary.n_neg == 1
    assert summary.min == -2.0
    assert summary.max == 1.0

    print_summary(summary, indent="  ")
    summarize_many([("one", [1.0, 2.0]), ("two", [-1.0, 0.0])], indent=">")
    print_jacobian_stats(np.asarray([[1.0, -2.0], [3.0, 4.0]]), indent="")
    captured = capsys.readouterr().out
    assert "arr: shape=(5,)" in captured
    assert "counts: nan=1 inf=1 zero=1 neg=1" in captured
    assert "sqrtg" in captured
    assert "|sqrtg|" in captured


def test_slice_excluding_axis_handles_scalars_and_axis_choice():
    assert slice_excluding_axis(np.asarray(3.0)).shape == ()
    arr = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(slice_excluding_axis(arr, axis_dim=0), arr[1:, :])
    np.testing.assert_array_equal(slice_excluding_axis(arr, axis_dim=1), arr[:, 1:])


def test_vmec_internal_mn_blocks_and_xc_order_for_lasym_threed():
    modes = vmec_mode_table(mpol=2, ntor=1)
    ns = 3
    ncoeff = len(modes.m)

    base = np.arange(ns * ncoeff, dtype=float).reshape(ns, ncoeff) / 10.0
    state = SimpleNamespace(
        Rcos=base + 1.0,
        Zsin=base + 2.0,
        Lsin=base + 3.0,
        Rsin=base + 4.0,
        Zcos=base + 5.0,
        Lcos=base + 6.0,
    )
    cfg = SimpleNamespace(mpol=2, ntor=1, lthreed=True, lasym=True, lconm1=True)
    static = SimpleNamespace(cfg=cfg, modes=modes)

    raw = vmec_internal_mn_from_state(
        state,
        static,
        apply_basis_norm=False,
        apply_m1_constraint=False,
    )
    constrained = vmec_internal_mn_from_state(
        state,
        static,
        apply_basis_norm=True,
        apply_m1_constraint=True,
    )

    assert set(constrained) == {"rcc", "rss", "rsc", "rcs", "zsc", "zcs", "zcc", "zss", "lsc", "lcs", "lcc", "lss"}
    for block in constrained.values():
        assert block.shape == (ns, 2, 2)
        assert np.all(np.isfinite(block))

    # The m=1 constrained basis should differ from the unconstrained raw blocks.
    assert not np.allclose(constrained["rss"][:, 1, :], raw["rss"][:, 1, :])
    assert not np.allclose(constrained["zcs"][:, 1, :], raw["zcs"][:, 1, :])

    xc = vmec_xc_from_mn_blocks(cfg=cfg, **constrained)
    assert xc.shape == (12 * ns * 2 * 2,)
    np.testing.assert_allclose(xc[: ns * 2 * 2], constrained["rcc"].reshape(ns, 4).T.reshape(-1))


def test_bilinear_periodic_wraps_and_rejects_bad_grids():
    grid = np.arange(12, dtype=float).reshape(3, 4)
    np.testing.assert_allclose(_bilinear_periodic(grid, 0.0, 0.0), grid[0, 0])
    np.testing.assert_allclose(_bilinear_periodic(grid, 2.0 * np.pi, 2.0 * np.pi), grid[0, 0])
    np.testing.assert_allclose(
        _bilinear_periodic(grid, 0.5 * np.pi, 0.5 * np.pi),
        _bilinear_periodic(grid, 0.5 * np.pi + 2.0 * np.pi, 0.5 * np.pi - 2.0 * np.pi),
    )

    with pytest.raises(ValueError, match="2D"):
        _bilinear_periodic(np.arange(3), 0.0, 0.0)
    with pytest.raises(ValueError, match="empty grid"):
        _bilinear_periodic(np.zeros((0, 3)), 0.0, 0.0)


def test_trace_fieldline_on_surface_constant_pitch():
    ntheta, nzeta = 8, 10
    radius = np.full((ntheta, nzeta), 2.0)
    z = np.zeros((ntheta, nzeta))
    bsupu = np.full((ntheta, nzeta), 0.5)
    bsupv = np.ones((ntheta, nzeta))
    bmag = np.full((ntheta, nzeta), 3.0)

    line = trace_fieldline_on_surface(
        R=radius,
        Z=z,
        bsupu=bsupu,
        bsupv=bsupv,
        Bmag=bmag,
        nfp=2,
        theta0=0.25,
        phi0=0.1,
        n_steps=6,
        dphi=0.2,
    )

    expected_phi = 0.1 + 0.2 * np.arange(6)
    expected_theta = 0.25 + 0.5 * (expected_phi - expected_phi[0])
    np.testing.assert_allclose(line.phi, expected_phi)
    np.testing.assert_allclose(line.theta, expected_theta)
    np.testing.assert_allclose(line.x, 2.0 * np.cos(np.mod(expected_phi, 2.0 * np.pi)))
    np.testing.assert_allclose(line.y, 2.0 * np.sin(np.mod(expected_phi, 2.0 * np.pi)))
    np.testing.assert_allclose(line.z, 0.0)
    np.testing.assert_allclose(line.Bmag, 3.0)


def test_trace_fieldline_zero_toroidal_component_holds_theta_constant():
    ntheta, nzeta = 6, 5
    radius = np.full((ntheta, nzeta), 1.5)
    z = np.zeros((ntheta, nzeta))
    bsupu = np.full((ntheta, nzeta), 2.0)
    bsupv = np.zeros((ntheta, nzeta))
    bmag = np.full((ntheta, nzeta), 4.0)

    line = trace_fieldline_on_surface(
        R=radius,
        Z=z,
        bsupu=bsupu,
        bsupv=bsupv,
        Bmag=bmag,
        nfp=1,
        theta0=0.7,
        phi0=0.2,
        n_steps=5,
        dphi=0.15,
    )

    np.testing.assert_allclose(line.theta, 0.7)
    np.testing.assert_allclose(line.phi, 0.2 + 0.15 * np.arange(5))
    np.testing.assert_allclose(line.Bmag, 4.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"nfp": 0}, "nfp must be positive"),
        ({"n_steps": 1}, "n_steps must be >= 2"),
        ({"dphi": 0.0}, "dphi must be nonzero"),
        ({"Z": np.zeros((2, 3))}, "same shape"),
    ],
)
def test_trace_fieldline_rejects_invalid_inputs(kwargs, message):
    base = {
        "R": np.ones((3, 3)),
        "Z": np.zeros((3, 3)),
        "bsupu": np.ones((3, 3)),
        "bsupv": np.ones((3, 3)),
        "Bmag": np.ones((3, 3)),
        "nfp": 1,
        "theta0": 0.0,
        "phi0": 0.0,
        "n_steps": 3,
        "dphi": 0.1,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        trace_fieldline_on_surface(**base)
