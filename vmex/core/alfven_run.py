"""``vmex --alfven``: shear-Alfven continuum gaps and the widest-gap global mode of a solved equilibrium.

Runs `FAX <https://github.com/uwplasma/FAX>`_ (``pip install 'vmex[alfven]'``) on a WOUT through ``fax.from_wout`` and
``fax.summary`` and writes ``<case>_alfven.png`` and ``<case>_alfven.json``. Frequencies scale as the inverse square root
of the mass density, so the assumed ion density and mass are printed in the JSON and the figure title.

FAX enables 64-bit floats when imported, so it is imported only here, after the equilibrium has been solved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib

__all__ = ["run_alfven"]


def run_alfven(wout_path, outdir, *, config=None, density=None, ion_mass=None, nodes=48, emit=print) -> dict:
    """Write ``<case>_alfven.png`` and ``.json`` for a WOUT. ``config`` is an optional ``fax run`` TOML case.

    A key given in the case file (``[modes]``, ``[plasma]``, ``[geometry]`` mboz/nboz/surfaces, ``[run]`` ntheta/nzeta/count)
    replaces the default; ``density`` [m^-3] and ``ion_mass`` [amu] override the case. Raises ImportError if FAX is missing.
    """
    from .._compat import require_optional

    require_optional("alfvex", "vmex --alfven")
    import fax  # after the solve: importing FAX enables x64

    raw = tomllib.loads(Path(config).read_text()) if config else {}
    case = fax.load(raw)  # validates the schema, rejecting unknown keys
    pick = lambda table, key, default: case[table][key] if key in raw.get(table, {}) else default  # noqa: E731
    mboz, nboz = pick("geometry", "mboz", 12), pick("geometry", "nboz", 12)
    s = np.linspace(0.0, 1.0, nodes + 2)[1:-1]  # radial finite-element nodes
    if "surfaces" in raw.get("geometry", {}):  # Boozer on the given surfaces, interpolated to the nodes
        geo = fax.from_wout(wout_path, mboz=mboz, nboz=nboz, surfaces=tuple(case["geometry"]["surfaces"])).interpolate(s)
    else:  # Boozer on every node: interpolating harmonics puts kinks in the metric derivatives
        geo = fax.from_wout(wout_path, mboz=mboz, nboz=nboz, surfaces=tuple(s))
        s = np.asarray(geo.s)
        if len(s) < 32:
            emit(f" WARNING: the WOUT has {len(s)} radial surfaces; gap modes need about 32 or more (raise NS_ARRAY)")
    m_max = pick("modes", "m_max", 12)
    if "n" in raw.get("modes", {}):  # an explicit family, as in ``fax run``
        harmonics = np.array([h for h in np.asarray(fax.mode_family(tuple(case["modes"]["n"]), m_max, pick("modes", "m_min", 0))) if h.any()])
    else:  # the resonant branches of every distinct family, both signs of n (fax.alfven_harmonics)
        harmonics = fax.alfven_harmonics(geo, m_max)
    dm, dn = np.ptp(harmonics[:, 0]), np.ptp(harmonics[:, 1])
    grid = dict(ntheta=pick("run", "ntheta", max(32, 2 * int(dm) + 2)), nzeta=pick("run", "nzeta", max(32, 2 * int(dn) + 2)))
    options = dict(**grid, count=pick("run", "count", 8))
    density = density or pick("plasma", "density", 1e19)
    ion_mass = ion_mass or pick("plasma", "ion_mass", 2.0)
    n_i = density * np.polyval(pick("plasma", "density_profile", [1.0])[::-1], s)
    with Dataset(wout_path) as w:  # pressure on the VMEC half mesh, s_j = (j - 1/2) / (ns - 1)
        pres = np.asarray(w["pres"][1:], dtype=float)
    pressure = np.interp(s, (np.arange(len(pres)) + 0.5) / len(pres), pres) if pres.max() > 0 else None
    emit(f" FAX shear-Alfven continuum: {len(harmonics)} harmonics, n_i(0) = {density:.3g} m^-3, {ion_mass:g} amu: {Path(wout_path).name}")
    rho = fax.mass_density(n_i, ion_mass)
    out = fax.summary(geo, harmonics, rho, pressure=pressure, ions=(n_i.tolist(), ion_mass), **options)
    stem = Path(outdir) / f"{Path(wout_path).stem.removeprefix('wout_')}_alfven"
    figure = fax.plot_summary(out, stem.with_suffix(".png"))
    mode = out["mode"] and {k: v for k, v in out["mode"].items() if k not in ("structure", "cross_section")}
    keys = ("harmonics", "density_m3", "ion_mass_amu", "density_kg_m3", "sound", "alfven_khz", "gaps", "mode_reason")
    stem.with_suffix(".json").write_text(json.dumps({**{k: out[k] for k in keys}, "mode": mode}, indent=2))
    for gap in out["gaps"][:3]:
        emit(f"   {gap['type']} gap {gap['center_khz']:.1f} kHz, width {gap['width_khz']:.1f} kHz, open over s = {gap['s_range'][0]:.2f}-{gap['s_range'][1]:.2f}")
    if not mode:
        emit(f"   no gap mode: {out['mode_reason']}")
    else:
        emit(f"   gap mode {mode['frequency_khz']:.1f} kHz peaked at s = {mode['peak_s']:.2f}, clear of the continuum, converged: {mode['converged']}")
    emit(f" Wrote Alfven figure: {figure}")
    return out
