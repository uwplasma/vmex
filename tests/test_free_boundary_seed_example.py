"""Pieces of ``examples/optimization/single_stage_free_boundary_seed.py`` that run without a GPU.

The full seed (fixed-boundary solves, virtual casing, the free boundary) is a GPU benchmark; these tests cover its
own numerics: the Levenberg-Marquardt solver of the coil fit, the offset start coils and the fit's per-point coil
rows.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("essos")

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "optimization"


@pytest.fixture(scope="module")
def seed():
    """The example module, for the qa4-beta case (its constants are read at import)."""
    previous = os.environ.get("COIL_CASE")
    os.environ["COIL_CASE"] = "qa4-beta"
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        for name in ("single_stage_free_boundary_seed", "single_stage_free_boundary_optimization_three_terms"):
            sys.modules.pop(name, None)
        yield importlib.import_module("single_stage_free_boundary_seed")
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        if previous is None:
            os.environ.pop("COIL_CASE", None)
        else:
            os.environ["COIL_CASE"] = previous


def _ellipse(nfp=2, aspect=4.0, ratio=0.5, ntor=4, mpol=5):
    """Rotating-ellipse boundary arrays as the three-terms example's ``seed_input`` builds them."""
    minor = 1.0 / aspect
    rbc, zbs = np.zeros((2 * ntor + 1, mpol)), np.zeros((2 * ntor + 1, mpol))
    rbc[ntor, 0] = 1.0
    rbc[ntor, 1] = zbs[ntor, 1] = minor * np.sqrt(1.0 + ratio**2)
    rbc[ntor - 1, 1], zbs[ntor - 1, 1] = -ratio * minor, ratio * minor
    return SimpleNamespace(nfp=nfp, ntor=ntor, rbc=rbc, zbs=zbs)


def test_levenberg_marquardt_converges_and_leaves_undetermined_directions(seed):
    """A nonlinear fit converges to its answer; a direction the rows do not see is not moved."""
    t = np.linspace(0.0, 2.0, 40)
    data = 1.5 * np.exp(-0.7 * t)

    def rows(x):  # x[2] enters no row: an undetermined direction, as a coil's parametrization phase
        return x[0] * np.exp(-x[1] * t) - data

    def jac(x):
        e = np.exp(-x[1] * t)
        return np.stack([e, -x[0] * t * e, np.zeros_like(t)], axis=1)

    x, converged, iterations = seed.levenberg_marquardt(rows, jac, np.array([1.0, 0.1, 0.3]), tol=1e-12,
                                                        max_iterations=200)
    assert converged and iterations < 50
    np.testing.assert_allclose(x[:2], [1.5, 0.7], rtol=1e-8)
    assert x[2] == pytest.approx(0.3, abs=1e-12)


def test_levenberg_marquardt_reports_the_iteration_cap(seed):
    x, converged, iterations = seed.levenberg_marquardt(lambda x: np.array([np.exp(x[0])]),
                                                        lambda x: np.array([[np.exp(x[0])]]),
                                                        np.array([0.0]), tol=1e-30, max_iterations=3)
    assert not converged and iterations == 3


def test_offset_coils_follow_the_boundary_at_the_offset(seed):
    """Every point of a start coil lies ``distance`` from the boundary's cross section in its plane."""
    import jax.numpy as jnp

    inp, distance = _ellipse(), 0.3
    coils = seed.offset_coils(inp, n_coils=4, order=12, distance=distance)
    gamma = np.asarray(coils.curves.gamma)[:4]  # the base coils
    theta = np.linspace(0.0, 2 * np.pi, 4000, endpoint=False)
    ms, ns = np.arange(inp.rbc.shape[1]), np.arange(-inp.ntor, inp.ntor + 1)
    for k, points in enumerate(gamma):
        phi = (k + 0.5) * np.pi / (inp.nfp * 4)
        assert np.allclose(np.arctan2(points[:, 1], points[:, 0]), phi, atol=1e-12)  # planar, in its plane
        angle = ms[None, :, None] * theta[None, None, :] - ns[:, None, None] * inp.nfp * phi
        R = np.einsum("nm,nmt->t", inp.rbc, np.cos(angle))
        Z = np.einsum("nm,nmt->t", inp.zbs, np.sin(angle))
        rc, zc = np.hypot(points[:, 0], points[:, 1]), points[:, 2]
        gap = np.min(np.hypot(rc[:, None] - R[None], zc[:, None] - Z[None]), axis=1)
        np.testing.assert_allclose(gap, distance, atol=2e-3)
    assert float(jnp.max(jnp.abs(jnp.asarray(coils.dofs_currents_raw) - 1.0))) == 0.0


def test_fit_penalty_rows_match_the_geometry(seed):
    """The per-point rows: one per coil (length, MSC) or per coil point, the clearance row the exact gap."""
    import jax.numpy as jnp

    tt = seed.tt
    inp, distance = _ellipse(), 0.3
    coils = seed.offset_coils(inp, n_coils=4, order=tt.COIL_ORDER, distance=distance)
    theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 64, endpoint=False),
                             np.linspace(0, 2 * np.pi, 128, endpoint=False), indexing="ij")
    ms, ns = np.arange(inp.rbc.shape[1]), np.arange(-inp.ntor, inp.ntor + 1)
    angle = ms[None, :, None, None] * theta[None, None] - ns[:, None, None, None] * inp.nfp * phi[None, None]
    R = np.einsum("nm,nmij->ij", inp.rbc, np.cos(angle))
    Z = np.einsum("nm,nmij->ij", inp.zbs, np.sin(angle))
    surface = jnp.asarray(np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=-1).reshape(-1, 3))
    rows = np.asarray(seed.fit_penalty_rows(coils, surface))
    n, p = 4, seed.FIT_POINTS
    assert rows.shape == (2 * n + 3 * n * p,) and np.all(np.isfinite(rows))
    # oracle: brute-force point distances of the same samples
    base = np.asarray(tt.resampled(coils, p, symmetric=False).gamma)
    every = np.asarray(tt.resampled(coils, p).gamma)
    gap = np.stack([np.min(np.linalg.norm(b[:, None] - np.asarray(surface)[None], axis=-1), axis=1) for b in base])
    others = np.stack([np.min(np.linalg.norm(base[i][:, None] - np.delete(every, i, axis=0).reshape(-1, 3)[None],
                                             axis=-1), axis=1) for i in range(n)])
    scaled = lambda d, limit: (d - limit - tt.DISTANCE_MARGIN) / limit  # noqa: E731
    np.testing.assert_allclose(rows[-n * p:], scaled(gap, tt.COIL_SURFACE_DISTANCE_LIMIT).ravel(), atol=1e-12)
    np.testing.assert_allclose(rows[2 * n + n * p:2 * n + 2 * n * p], scaled(others, tt.COIL_DISTANCE_LIMIT).ravel(),
                               atol=1e-12)
    assert np.min(gap) <= distance + 1e-3  # the offset is in-plane: the 3-D gap is at most the offset


def test_three_terms_chart_holds_no_seed_dependent_constant(seed):
    """The three-terms example's CoilChart keeps its nominal values in ``x0``: two coil sets lower to the same
    program (one persistent-cache entry for every seed), and ``x0`` is the nominal coils."""
    import jax
    import jax.numpy as jnp
    from essos.coils import Coils

    tt = seed.tt
    charts = []
    for distance in (0.3, 0.35):
        coils = seed.offset_coils(_ellipse(), n_coils=4, order=4, distance=distance)
        coils = Coils(coils.curves, jnp.full(4, 1.0e5 * (1 + distance)))
        chart = tt.CoilChart(coils, current_dofs=(0, 1, 2, 3), scales=np.full(4 + 4 * 3 * 9, 0.05))
        np.testing.assert_allclose(np.asarray(chart(jnp.asarray(chart.x0)).gamma), np.asarray(coils.gamma),
                                   atol=1e-14)
        np.testing.assert_allclose(np.asarray(chart.base_currents_at(chart.x0)), np.asarray(coils.dofs_currents_raw))
        charts.append(chart)
    texts = [jax.jit(lambda x, c=c: c(x).gamma).lower(jnp.asarray(c.x0)).as_text() for c in charts]
    assert texts[0] == texts[1]
    np.testing.assert_allclose(charts[0].scales[:4], 0.05 * 1.3e5)  # current scales in A: relative x nominal


def test_seed_interface_runs_the_virtual_casing_on_the_cpu(seed, monkeypatch):
    """One seed gives one set of start coils: on the GPU the virtual-casing B_plasma differs by ~1e-14 between two
    evaluations, which L-BFGS-B turned into start coils whose step-0 QS differed by 1 % (two builds of one Picard
    result).  ``seed_interface`` computes it under the CPU device and returns arrays on the default device."""
    from dataclasses import replace

    import jax
    import vmex as vj
    from vmex import optimize as opt

    deck = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.LandremanPaul2021_QA_beta2p5_bootstrap"
    inp = vj.VmecInput.from_file(str(deck)).change_resolution(mpol=3, ntor=3, ntheta=12, nzeta=16)
    inp = replace(inp, lfreeb=False, ns_array=np.array([11]), niter_array=np.array([3000]),
                  ftol_array=np.array([1e-10]))
    wout = opt.solve_equilibrium(inp).wout
    devices, from_wout = [], vj.PlasmaVacuumInterface.from_wout

    def recorded(*args, **kwargs):
        devices.append(jax.config.jax_default_device)
        return from_wout(*args, **kwargs)

    monkeypatch.setattr(vj.PlasmaVacuumInterface, "from_wout", staticmethod(recorded))
    interface = seed.tt.seed_interface(wout, nphi=8, ntheta=8)
    assert [getattr(d, "platform", None) for d in devices] == ["cpu"]
    assert interface.B_plasma.shape == (3, 8, 8) and interface.B_plasma.devices() == {jax.devices()[0]}
