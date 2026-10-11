"""Effective-ripple diagnostics through the optional NEO_JAX backend."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "diagnostic_neo_config",
    "epsilon_effective_from_boozer",
    "epsilon_effective_from_wout",
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


def diagnostic_neo_config():
    """Return the bounded NEO resolution used by VMEX summary figures.

    On the bundled NFP=2 QA and NFP=4 QI finite-beta decks this stays within
    4% of NEO_JAX's default ``NeoConfig``.  A 16 x 16 spline grid aliases the
    summary Boozer spectrum (``mboz=16``) and fewer than 20 steps per field
    period or 200 field periods leave 7-30% errors.  Pass a
    ``neo_jax.NeoConfig`` to :func:`epsilon_effective_from_wout` for
    publication calculations.
    """
    NeoConfig, _ = _neo_imports()
    return NeoConfig(
        theta_n=32, phi_n=32, npart=24, multra=1, no_bins=50,
        nstep_per=20, nstep_min=200, nstep_max=500, acc_req=0.02,
        max_rational_field_periods=100000,
    )


def _require_asymmetric_neo():
    _neo_imports()
    from neo_jax import BoozerData

    if "bmns" not in BoozerData.__dataclass_fields__:
        raise NotImplementedError("LASYM requires NEO_JAX with sine spectra")


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


def epsilon_effective_from_boozer(booz: Any, *, config=None):
    r"""Return ``(s, epsilon_eff**(3/2))`` from Boozer-coordinate data.

    ``booz`` may be a NEO_JAX ``BoozerData`` or a booz_xform-style mapping.
    The returned NEO quantity is the effective-ripple transport measure
    :math:`\epsilon_\mathrm{eff}^{3/2}`, conventionally named ``epstot``.
    JAX arrays remain differentiable when the mapping and selected NEO path
    are JAX-native.
    """
    get = booz.get if isinstance(booz, dict) else lambda key, default=None: getattr(booz, key, default)
    if get("asym", get("lasym", get("bmns_b") is not None)):
        _require_asymmetric_neo()
    NeoConfig, run_neo = _neo_imports()
    cfg = NeoConfig() if config is None else config
    result = run_neo(
        booz, config=cfg, use_jax=True, progress=False, jax_surface_scan=True)
    values = getattr(result, "eps_eff", None)
    if values is None:
        values = result.epsilon_effective
    if hasattr(booz, "es"):
        surfaces = booz.es
    elif isinstance(booz, dict) and "s_b" in booz:
        surfaces = booz["s_b"]
    else:
        surfaces = np.arange(np.asarray(values).size, dtype=float)
    return surfaces, values


def epsilon_effective_from_wout(
    wout,
    *,
    surfaces: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 0.95),
    mboz: int = 16,
    nboz: int = 12,
    config=None,
    clear_jax_caches: bool = False,
):
    """Compute ``epsilon_eff**(3/2)`` directly from an in-memory VMEX wout.

    The same in-memory Boozer transform feeds NEO, including all LASYM
    partners. Caches remain available unless ``clear_jax_caches=True``.
    """
    if bool(getattr(wout, "lasym", False)):
        _require_asymmetric_neo()
    if clear_jax_caches:
        import gc

        import jax

        jax.clear_caches(); gc.collect()
    from booz_xform_jax import Booz_xform

    bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
    bx.read_wout_data(wout)
    s_in = np.asarray(bx.s_in, dtype=float)
    indices = sorted({int(np.argmin(np.abs(s_in - float(s)))) for s in surfaces})
    bx.compute_surfs = indices
    bx.run()
    booz = _neo_boozer_data(bx, indices)
    return epsilon_effective_from_boozer(booz, config=config)
