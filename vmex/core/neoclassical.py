"""Effective ripple ``epsilon_eff`` of VMEX equilibria: Boozer tables to NEO_JAX bounce integrals.

The physics lives in :func:`neo_jax.effective_ripple_3_2` (Nemov et al., Phys. Plasmas 6, 4622
(1999), as bounce integrals along Boozer field lines; differentiable in forward and reverse mode).
This module is the adapter: solver state or wout to Boozer tables, linear interpolation in ``s``
between the two half-mesh rows that bracket each surface, and the NEO transport screen.
Needs ``neo-jax >= 1.2.0`` (``pip install "vmex[neoclassical]"``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "epsilon_effective_3_2_from_boozer",
    "epsilon_effective_3_2_from_state",
    "epsilon_effective_3_2_from_wout",
    "screen_transport",
    "transport_improves",
]


def _neo_imports():
    from .._compat import require_optional

    require_optional("neo-jax", "effective ripple")
    try:
        from neo_jax import NeoConfig, run_neo
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "effective ripple requires NEO_JAX; install VMEX with "
            "`pip install vmex[neoclassical]`") from exc
    return NeoConfig, run_neo


def _bracket(s_half, surfaces):
    """Static half-mesh rows ``rows`` and the weights that interpolate linearly in s to ``surfaces``."""
    s = np.atleast_1d(np.asarray(surfaces, float))
    hi = np.clip(np.searchsorted(s_half, s), 1, len(s_half) - 1)
    w = np.clip((s - s_half[hi - 1]) / (s_half[hi] - s_half[hi - 1]), 0.0, 1.0)
    rows = np.unique(np.concatenate([hi - 1, hi]))
    return rows, np.searchsorted(rows, hi - 1), np.searchsorted(rows, hi), w


def epsilon_effective_3_2_from_boozer(booz, budget=None, *, mode_first=None):
    """``(s, epsilon_eff**(3/2))`` of a NEO-style Boozer mapping (``nfp_b``, ``ixm_b``, ..., ``s_b``)."""
    from neo_jax import booz_xform_to_boozerdata_jax, effective_ripple_3_2

    data = booz_xform_to_boozerdata_jax(booz, mode_first=mode_first, mode_indices=np.arange(len(booz["ixm_b"])))
    return booz["s_b"], effective_ripple_3_2(data, budget)


def _ripple(booz, lo, hi, w, budget, mode_first):
    """``epsilon_eff**(3/2)`` of the Boozer rows, interpolated to the requested surfaces."""
    values = epsilon_effective_3_2_from_boozer(booz, budget, mode_first=mode_first)[1]
    return (1 - w) * values[lo] + w * values[hi]


def epsilon_effective_3_2_from_state(state, rt, surfaces: Sequence[float] = (0.25, 0.5, 0.75), *,
                                     mboz: int = 12, nboz: int = 9, budget=None):
    r"""``epsilon_eff**(3/2)`` of a solved VMEX state at ``surfaces``, traceable end to end.

    The state's Boozer tables (:func:`vmex.core.omnigenity.boozer_spectrum_state`, ``booz_xform_jax``'s
    jittable kernel) feed :func:`neo_jax.effective_ripple_3_2` on the two half-mesh rows that bracket
    each surface; the result is interpolated linearly in ``s``.  ``jax.grad``, ``jax.jvp`` and
    ``jax.jacrev`` pass through :func:`vmex.core.implicit.solve_implicit` and this function alike.
    ``budget`` is a :class:`neo_jax.RippleBudget`.  The effective ripple is ``value ** (2 / 3)``.
    """
    from .omnigenity import boozer_spectrum_state

    ns = int(np.shape(rt.setup.s_full)[0])
    s_half = (np.arange(ns - 1) + 0.5) / (ns - 1)
    rows, lo, hi, w = _bracket(s_half, surfaces)
    b = boozer_spectrum_state(state, rt, surfaces=s_half[rows], mboz=mboz, nboz=nboz, geometry=True)
    booz = {"nfp_b": b["nfp"], "ixm_b": b["xm_b"], "ixn_b": b["xn_b"], "iota_b": b["iota_b"],
            "buco_b": b["I_b"], "bvco_b": b["G_b"], "s_b": b["s_b"]}
    names = ("bmnc_b", "rmnc_b", "zmns_b", "pmns_b") + (
        ("bmns_b", "rmns_b", "zmnc_b", "pmnc_b") if bool(rt.setup.lasym) else ())
    booz |= {k: b[k] for k in names}
    return _ripple(booz, lo, hi, w, budget, mode_first=False)


def epsilon_effective_3_2_from_wout(wout, surfaces: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 0.95), *,
                                    mboz: int = 12, nboz: int = 9, budget=None):
    """``epsilon_eff**(3/2)`` at ``surfaces`` from a (VMEX or VMEC) ``wout``, interpolated like the state route."""
    from booz_xform_jax import Booz_xform

    bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
    bx.read_wout_data(wout)
    rows, lo, hi, w = _bracket(np.asarray(bx.s_in, float), surfaces)
    bx.compute_surfs = [int(r) for r in rows]
    bx.run()
    return _ripple(_neo_boozer_data(bx, rows), lo, hi, w, budget, mode_first=True)


def _neo_boozer_data(bx, indices):
    """Share one mode-first Boozer transform with NEO, including sine partners."""
    booz = {
        "nfp_b": int(bx.nfp), "ns_b": len(indices), "mode_first": True,
        "asym": bool(getattr(bx, "asym", False)),
        "ixm_b": np.asarray(bx.xm_b), "ixn_b": np.asarray(bx.xn_b),
        "iota_b": np.asarray(bx.iota)[indices],
        "buco_b": np.asarray(bx.Boozer_I), "bvco_b": np.asarray(bx.Boozer_G),
        "rmnc_b": np.asarray(bx.rmnc_b), "zmns_b": np.asarray(bx.zmns_b),
        "pmns_b": -np.asarray(bx.numns_b), "bmnc_b": np.asarray(bx.bmnc_b),
        "s_b": np.asarray(bx.s_b),
    }
    if booz["asym"]:
        booz.update(rmns_b=np.asarray(bx.rmns_b), zmnc_b=np.asarray(bx.zmnc_b),
                    pmnc_b=-np.asarray(bx.numnc_b), bmns_b=np.asarray(bx.bmns_b))
    return booz


def screen_transport(wout, surfaces, *, mboz=24, nboz=24, config=None):
    """Matched forward NEO screen; not a converged transport calculation."""
    from booz_xform_jax import Booz_xform
    NeoConfig, run_neo = _neo_imports()

    bx = Booz_xform(verbose=0, mboz=mboz, nboz=nboz)
    bx.read_wout_data(wout)
    indices = [int(np.argmin(abs(np.asarray(bx.s_in)-s))) for s in surfaces]
    if not np.allclose(np.asarray(bx.s_in)[indices], surfaces, rtol=0, atol=1e-12):
        raise ValueError("The NEO screen requires the requested half-mesh radii")
    bx.compute_surfs = indices
    bx.run()
    config = config if config is not None else NeoConfig(theta_n=192, phi_n=192, npart=80, multra=2, nstep_per=80,
                       nstep_min=800, nstep_max=2000, acc_req=.005, ref_swi=2,
                       max_rational_field_periods=100000, rational_surface_policy="error")
    results = run_neo(_neo_boozer_data(bx, indices), config=config, use_jax=True,
                      progress=False, jax_surface_scan=False)
    values = []
    for result in results:
        diag, value = result.diagnostics, float(result.epsilon_effective)
        if (not np.isfinite(value) or value <= 0 or not diag.get("rational_correction_valid")
                or diag.get("approximation_used") or diag.get("rational_work_limit_exceeded")):
            raise ValueError(f"Invalid NEO screen at s={float(result.s):.6g}, Q={value}; keep the incumbent")
        values.append(value)
    return np.asarray(values)


def transport_improves(baseline, trial):
    """Require a resolved mean gain and no radial tradeoff in the coarse screen."""
    a, b = np.asarray(baseline), np.asarray(trial)
    return bool(a.ndim == b.ndim == 1 and a.shape == b.shape and a.size > 0 and np.isfinite(a).all()
                and np.isfinite(b).all() and np.all(a > 0) and np.all(b > 0)
                and np.all(b <= a) and b.mean() < .99*a.mean())
