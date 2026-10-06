#!/usr/bin/env python3
"""Regenerate ``readme_neoclassical_output.webp``, the README ``--neoclassical`` figure.

Solves the bundled Landreman-Paul QA beta = 2.5 % bootstrap deck, runs the
README command ``vmex --neoclassical --nc-profiles ...`` (default preset, DKX)
on its WOUT, and draws the numbers of the ``*_neoclassical.h5`` it writes at
README width, as an 1100 px WebP.  Needs ``pip install "vmex[neoclassical]"``.

Usage::

    python docs/_static/figures/sources/make_neoclassical_figure.py
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from vmex.core.cli import main as vmex  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
CASE = "LandremanPaul2021_QA_beta2p5_bootstrap"
DECK = REPO / "examples" / "data" / f"input.{CASE}"
PROFILES = REPO / "examples" / "data" / f"kinetic_profiles.{CASE}.json"
OUT = REPO / "docs" / "_static" / "figures" / "readme_neoclassical_output.webp"

d: dict[str, np.ndarray] = {}
with tempfile.TemporaryDirectory() as tmp:
    assert vmex([str(DECK), "--outdir", tmp, "--quiet"]) == 0
    assert vmex([f"{tmp}/wout_{CASE}.nc", "--neoclassical", "--nc-profiles", str(PROFILES),
                 "--outdir", tmp, "--quiet"]) == 0
    with h5py.File(Path(tmp) / f"{CASE}_neoclassical.h5") as h5:
        h5.visititems(lambda name, obj: d.update({name: np.asarray(obj)})
                      if isinstance(obj, h5py.Dataset) else None)

plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "legend.fontsize": 11,
                     "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(2, 4, figsize=(14, 7.4), layout="constrained")
nu, e_star = d["monoenergetic/nu_prime"], d["monoenergetic/e_star"]
for e in np.unique(e_star):
    for a, key in zip(ax[0, :3], ("D11", "D31", "D33")):
        a.plot(nu[e_star == e], np.abs(d[f"monoenergetic/{key}"][e_star == e]), "o-", ms=4,
               label=f"$E^*$ = {e:g}")
for a, key, what in zip(ax[0, :3], ("11", "31", "33"), ("radial", "bootstrap", "parallel")):
    a.set(xscale="log", yscale="linear" if key == "31" else "log", xlabel=r"collisionality $\nu'$",
          ylabel=rf"$|D_{{{key}}}^*|$", title=f"monoenergetic $D_{{{key}}}^*$ ({what})")
ax[0, 0].legend()
m = ax[0, 3].contourf(d["geometry/zeta"], d["geometry/theta"], d["geometry/BHat"], 24, cmap="viridis")
fig.colorbar(m, ax=ax[0, 3], label="|B| [T]")
ax[0, 3].set(xlabel=r"$\zeta$ [rad]", ylabel=r"$\theta$ [rad]", title="|B| on the r/a = 0.5 surface")
r = d["profiles/r"]
ax[1, 0].plot(r, d["profiles/er_evaluated"], "s-", color="C3", label="ion root")
ax[1, 0].set(xlabel="r/a", ylabel=r"$E_r$ [kV/m]", title=r"ambipolar $E_r$ ($\Sigma Z\Gamma$ = 0)")
ax[1, 0].legend()
for key, label, style in (("bootstrap_kA_m2", "DKX drift-kinetic", "o-"),
                          ("jdotb_redl_kA_m2", "Redl, same profiles", "v-."),
                          ("jdotb_vmec_kA_m2", "VMEX equilibrium", "^:")):
    ax[1, 1].plot(r, d[f"profiles/{key}"] / 1e3, style, label=label)
ax[1, 1].set(xlabel="r/a", ylabel=r"$\langle j\cdot B\rangle/\langle B^2\rangle^{1/2}$ [MA/m$^2$]",
             title="bootstrap current")
ax[1, 1].legend()
for a, key, unit, scale, title in (
        (ax[1, 2], "particle_flux_si", r"$\Gamma$ [$10^{20}$ m$^{-2}$ s$^{-1}$]", 1e20, "particle flux"),
        (ax[1, 3], "heat_flux_si", "Q [kW/m$^2$]", 1.0, "heat flux")):
    for s, (species, style) in enumerate((("ions", "o-"), ("electrons", "s--"))):
        a.plot(r, d[f"profiles/{key}"][:, s] / scale, style, label=species)
    a.set(xlabel="r/a", ylabel=unit, title=f"{title} at the root")
    a.legend()
fig.suptitle(r"QA, $\beta$ = 2.5 %: $n_e$ = 2.4e20 m$^{-3}$, $T_e = T_i$ = 9.45 keV on axis "
             r"(vmex --neoclassical, DKX)")
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100)
Image.open(buf).convert("RGB").resize((1100, round(1100 * 7.4 / 14)), Image.LANCZOS).save(
    OUT, "WEBP", quality=40, method=6)
print(f"wrote {OUT}")
