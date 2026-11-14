"""``vmex --alfven``: shear-Alfven continuum gaps and the widest-gap global mode of a solved equilibrium.

Runs `FAX <https://github.com/uwplasma/FAX>`_ (``pip install 'vmex[alfven]'``) on a WOUT through ``fax.from_wout`` and
``fax.summary`` and writes ``<case>_alfven.png`` and ``<case>_alfven.json``. Frequencies scale as the inverse square root
of the mass density, so the assumed ion density and mass are printed in the JSON and the figure title.

FAX enables 64-bit floats when imported, so it is imported only here, after the equilibrium has been solved.
"""

from __future__ import annotations

import json
import sys
from math import ceil
from pathlib import Path

import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib

__all__ = ["default_harmonics", "run_alfven"]


def default_harmonics(nfp: int, iota) -> tuple[list[int], int]:
    """Toroidal families ``n = j nfp`` for j = 0, 1, 2 (plus ``n = +-1`` when iota < 1) and the poloidal cap.

    ``m_max = min(12, ceil(3 max(1, iota_max n_max) + 4))``: the cost grows as about H^3 per surface.
    """
    iota_max = float(np.max(np.abs(iota)))
    n = sorted({j * nfp for j in range(3)} | ({1, -1} if iota_max < 1 else set()))
    return n, min(12, ceil(3 * max(1.0, iota_max * max(map(abs, n))) + 4))


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
    mboz, nboz = pick("geometry", "mboz", 6), pick("geometry", "nboz", 6)
    surfaces = tuple(pick("geometry", "surfaces", [(i + 0.5) / 12 for i in range(12)]))
    geo = fax.from_wout(wout_path, mboz=mboz, nboz=nboz, surfaces=surfaces)
    if "n" in raw.get("modes", {}):
        n, m_max = case["modes"]["n"], case["modes"]["m_max"]
    else:
        n, m_max = default_harmonics(geo.nfp, geo.iota)
        m_max = pick("modes", "m_max", m_max)
    m_min = pick("modes", "m_min", 0)
    harmonics = np.array([h for h in np.asarray(fax.mode_family(tuple(n), m_max, m_min)) if h.any()])  # no (0, 0)
    nzeta = max(32, 2 * (max(n) - min(n)) + 2)
    options = dict(ntheta=pick("run", "ntheta", 32), nzeta=pick("run", "nzeta", nzeta), count=pick("run", "count", 4))
    density = density or pick("plasma", "density", 1e19)
    ion_mass = ion_mass or pick("plasma", "ion_mass", 2.0)
    emit(f" FAX shear-Alfven continuum: {len(harmonics)} harmonics, n_i = {density:.3g} m^-3, {ion_mass:g} amu: {Path(wout_path).name}")
    s = np.linspace(0.0, 1.0, nodes + 2)[1:-1]  # radial finite-element nodes
    out = fax.summary(geo.interpolate(s), harmonics, fax.mass_density(density, ion_mass), ions=(density, ion_mass), **options)
    stem = Path(outdir) / f"{Path(wout_path).stem.removeprefix('wout_')}_alfven"
    figure = fax.plot_summary(out, stem.with_suffix(".png"))
    mode = out["mode"] and {k: v for k, v in out["mode"].items() if k != "structure"}
    stem.with_suffix(".json").write_text(json.dumps({**{k: out[k] for k in ("harmonics", "density_m3", "ion_mass_amu", "density_kg_m3", "gaps", "mode_reason")}, "mode": mode}, indent=2))
    for gap in out["gaps"][:3]:
        emit(f"   {gap['type']} gap {gap['center_khz']:.1f} kHz, width {gap['width_khz']:.1f} kHz, closest at s = {gap['s']:.2f}")
    if not mode:
        emit(f"   no global mode: {out['mode_reason']}")
    else:
        emit(f"   global mode {mode['frequency_khz']:.1f} kHz at s = {mode['peak_s']:.2f}, in the gap, converged: {mode['converged']}")
    emit(f" Wrote Alfven figure: {figure}")
    return out
