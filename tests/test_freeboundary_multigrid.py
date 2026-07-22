"""Free-boundary radial-ladder, continuation, and hot-restart regressions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vmex.core import freeboundary as FB  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.fourier import mode_table  # noqa: E402
from vmex.core.multigrid import (  # noqa: E402
    interpolate_state, solve_free_boundary_multigrid,
)
from vmex.core.preconditioner_2d import Prec2DConfig  # noqa: E402
from vmex.core.solver import SpectralState, resolution_from_input  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

REPO = Path(__file__).resolve().parents[1]
DECK = REPO / "examples" / "data" / "input.cth_like_free_bdy_lasym_small"
MGRID = REPO / "examples" / "data" / "mgrid_cth_like_lasym_small.nc"
CONV_DECK = REPO / "examples" / "data" / "input.cth_like_free_bdy"
CONV_MGRID = REPO / "examples" / "data" / "mgrid_cth_like.nc"
CONV_WOUT = REPO / "examples" / "data" / "single_grid" / "wout_cth_like_free_bdy.nc"


def _state(ns: int, mnmax: int, value: float) -> SpectralState:
    a = jnp.full((ns, mnmax), value, dtype=jnp.float64)
    return SpectralState(a, a, a, a, a, a)


def test_stage_transfer_carries_vacuum_and_interpolates_xstore(monkeypatch) -> None:
    """Increasing grids use xstore; equal grids use the current xc state."""
    inp = VmecInput.from_file(DECK)
    calls = []
    vacua = []

    def fake_stage(_inp, **kwargs):
        assert kwargs["time_step"] == 0.25
        assert kwargs["tcon0"] == 1.2
        assert kwargs["gamma"] == 0.1
        assert kwargs["nstep"] == 17
        assert kwargs["lconm1"] is False
        assert kwargs["precon_type"] == "NONE"
        assert kwargs["prec2d_threshold"] == 3e-7
        if len(calls) < 2:
            assert kwargs["reuse_vacuum_cache"] is False
        else:
            assert kwargs["reuse_vacuum_cache"] is True
            np.testing.assert_array_equal(
                np.asarray(kwargs["constraint_continuation"][0]), [2.0])
        ns = kwargs["resolution"].ns
        incoming = kwargs["initial_state"]
        continuation = kwargs["vacuum_continuation"]
        calls.append((ns, incoming, continuation))
        vacuum = FB.FreeBoundaryState(
            ivac=5 + len(calls), nvacskip=11 + len(calls), nvskip0=9,
            turned_on=True, delbsq=0.1 * len(calls),
        )
        vacua.append(vacuum)
        # Deliberately make current xc and xstore distinguishable.
        current = _state(ns, kwargs["resolution"].mnmax, 10.0 + len(calls))
        xstore = _state(ns, kwargs["resolution"].mnmax, 20.0 + len(calls))
        result = SimpleNamespace(state=current, marker=ns)
        return SimpleNamespace(
            result=result, continuation_state=xstore, vacuum=vacuum,
            rcon0=jnp.asarray([len(calls)], dtype=float),
            zcon0=jnp.asarray([-len(calls)], dtype=float),
        )

    monkeypatch.setattr(FB, "_solve_free_boundary_stage", fake_stage)
    field = object()
    result = solve_free_boundary_multigrid(
        inp, ns_array=[7, 15, 15], ftol_array=[1e-4], niter_array=[2],
        external_field=field, raise_on_max_iterations=False,
        time_step=0.25, tcon0=1.2, gamma=0.1, nstep=17, lconm1=False,
        precon_type="NONE", prec2d_threshold=3e-7,
    )

    assert result.marker == 15
    assert [c[0] for c in calls] == [7, 15, 15]
    assert calls[0][1] is None and calls[0][2] is None
    assert calls[1][1].R_cos.shape[0] == 15
    # The increasing-grid interpolation came from stage 1 xstore (=21), not
    # its current xc (=11).  Odd-m sqrt(s) scaling makes the expected radial
    # profile non-constant, so compare with the actual interp.f operator.
    modes = mode_table(int(inp.mpol), int(inp.ntor))
    expected = interpolate_state(
        _state(7, modes.mnmax, 21.0), ns_fine=15, modes=modes)
    np.testing.assert_allclose(
        np.asarray(calls[1][1].R_cos), np.asarray(expected.R_cos))
    assert calls[1][2] is vacua[0]
    # Equal NS returns early in initialize_radial.f and keeps current xc (=12).
    np.testing.assert_allclose(np.asarray(calls[2][1].R_cos), 12.0)
    assert calls[2][2] is vacua[1]
    assert calls[2][1].R_cos.shape[0] == 15


def test_public_two_stage_free_boundary_rebuilds_and_stays_finite() -> None:
    """The bundled non-confidential LASYM case crosses vacuum turn-on."""
    inp = VmecInput.from_file(DECK)
    lines: list[str] = []

    def emit(value="", end="\n"):
        lines.append(str(value) + end)

    result = solve_free_boundary_multigrid(
        inp, ns_array=[7, 15], ftol_array=[1e-10, 1e-10],
        niter_array=[60, 5], mgrid_path=MGRID, verbose=True, emit=emit,
        raise_on_max_iterations=False,
    )
    output = "".join(lines)
    assert output.count("VACUUM PRESSURE TURNED ON") == 1
    assert "NS =    7" in output and "NS =   15" in output
    assert result.state.R_cos.shape[0] == 15
    assert np.all(np.isfinite(result.fsq_history))
    assert np.all(np.isfinite(np.asarray(result.state.R_cos)))
    # Regression for the old turn-on-ordering blow-up (~5 km major radius).
    assert 0.5 < result.r00 < 1.0


def test_single_grid_hot_restart_preserves_free_edge() -> None:
    inp = VmecInput.from_file(DECK)
    first = FB.solve_free_boundary(
        inp, mgrid_path=MGRID, max_iterations=1,
        error_on_no_convergence=False,
    )
    seed = replace(
        first.state,
        R_cos=first.state.R_cos.at[-1, 0].add(1.0e-5),
        Z_sin=first.state.Z_sin.at[-1, 1].add(-1.0e-5),
    )
    restarted = FB.solve_free_boundary(
        inp, mgrid_path=MGRID, max_iterations=1, initial_state=seed,
        error_on_no_convergence=False,
    )
    # Vacuum activation repeats on a reset-style user hot restart, but the
    # evolved free boundary is not replaced by the deck's original edge.
    np.testing.assert_allclose(
        np.asarray(restarted.state.R_cos[-1]),
        np.asarray(seed.R_cos[-1]), rtol=0.0, atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(restarted.state.Z_sin[-1]),
        np.asarray(seed.Z_sin[-1]), rtol=0.0, atol=0.0,
    )


def test_rejects_fixed_boundary_input() -> None:
    inp = replace(VmecInput.from_file(DECK), lfreeb=False)
    with pytest.raises(ValueError, match="LFREEB"):
        solve_free_boundary_multigrid(inp, external_field=object())


@pytest.mark.full
def test_free_boundary_accepts_active_2d_preconditioner() -> None:
    """The fixed-boundary preconditioner controls remain live in free mode."""
    inp = VmecInput.from_file(DECK)
    cfg = Prec2DConfig(
        threshold=1e-2, start_iteration=45, step=0.05,
        gmres_restart=2, gmres_max_restarts=1, gmres_rtol=0.1,
    )
    result = FB.solve_free_boundary(
        inp, mgrid_path=MGRID, resolution=resolution_from_input(inp, ns=7),
        max_iterations=55, error_on_no_convergence=False, prec2d=cfg,
    )
    assert result.iterations == 55
    assert np.all(np.isfinite([result.fsqr, result.fsqz, result.fsql]))


@pytest.mark.full
def test_converged_multigrid_final_state_matches_vmec2000_wout() -> None:
    if not CONV_MGRID.exists() or not CONV_WOUT.exists():
        pytest.skip("converged CTH mgrid/wout assets unavailable")
    from vmex.core.wout import read_wout

    inp = VmecInput.from_file(CONV_DECK)
    result = solve_free_boundary_multigrid(
        inp, ns_array=[7, 15], ftol_array=[1e-8, 1e-10],
        niter_array=[1000, 2500], mgrid_path=CONV_MGRID,
    )
    reference = read_wout(CONV_WOUT)
    mine = {(int(m), int(n)): i for i, (m, n) in enumerate(zip(result.xm, result.xn))}
    idx = np.asarray([
        mine[(int(m), int(n))] for m, n in zip(reference.xm, reference.xn)
    ])
    rerr = np.max(np.abs(result.rmnc[-1, idx] - reference.rmnc[-1])) / np.max(np.abs(reference.rmnc[-1]))
    zerr = np.max(np.abs(result.zmns[-1, idx] - reference.zmns[-1])) / np.max(np.abs(reference.zmns[-1]))
    s_mine = np.linspace(0.0, 1.0, result.iotaf.size)
    s_ref = np.linspace(0.0, 1.0, reference.iotaf.size)
    iota_ref = np.interp(s_mine, s_ref, reference.iotaf)
    ierr = np.max(np.abs(result.iotaf - iota_ref)) / np.max(np.abs(iota_ref))
    assert result.converged
    # This packaged reference is ns=151, while the fast CI ladder ends at
    # ns=15; compare the common boundary and interpolated iota at the expected
    # radial-discretization scale.  The exact ns=15 VMEC2000 comparison is
    # recorded by benchmarks/run_freeboundary_multigrid.py (<1e-3 here).
    assert rerr < 1e-2
    assert zerr < 1e-2
    assert ierr < 1.5e-2
