#!/usr/bin/env python
"""Remove the spectral-constraint bias of a free-boundary equilibrium.

Solves a vacuum free-boundary equilibrium in the Landreman & Paul (2021) QA coils (ESSOS),
relaxes it with ``vj.relax_free_boundary_constraint`` and prints the largest relative
``|B|`` error against the direct Biot-Savart field on three flux surfaces, before and after.
See docs/howto/free-boundary.md.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

import vmex as vj

DATA_DIR = Path(__file__).resolve().parent / "data"
COILS_FILE = DATA_DIR / "ESSOS_biot_savart_LandremanPaulQA.json"
INPUT_FILE = DATA_DIR / "input.LandremanPaul2021_QA_lowres"

# Cylindrical grid the coil field is tabulated onto (tricubic interpolation):
GRID = dict(rmin=0.45, rmax=1.55, zmin=-0.6, zmax=0.6, ir=96, jz=96, kp=32, order=3)

# Plasma resolution: NTOR = 12 resolves the coil ripple, NZETA = kp of the table.
MPOL, NTOR, NZETA, NS = 7, 12, 32, 16
NITER, FTOL = 20000, 1e-9
PHIEDGE = -0.025                  # toroidal flux matching the coil field [Wb]
RELAX_ITERATIONS = 2000
SURFACES = (0.25, 0.5, 0.9)       # normalized toroidal flux of the checks

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    NS, RELAX_ITERATIONS = 12, 1000

###############################################################################
# End of input parameters.
###############################################################################

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from essos.coils import Coils  # noqa: E402 (optional heavy import)
from essos.fields import BiotSavart  # noqa: E402

coils = Coils.from_json(str(COILS_FILE))
coil_B = jax.jit(jax.vmap(BiotSavart(coils).B))
field = vj.MgridField.from_coils(coils, **GRID)

inp = vj.VmecInput.from_file(INPUT_FILE)
k = inp.ntor - min(inp.ntor, NTOR)


def truncate(a):
    """The deck's boundary table on the (2 NTOR + 1, MPOL) grid of this run."""
    out = np.zeros((2 * NTOR + 1, MPOL))
    block = a[k:k + 2 * min(inp.ntor, NTOR) + 1, :MPOL]
    lo = NTOR - min(inp.ntor, NTOR)
    out[lo:lo + block.shape[0], :block.shape[1]] = block
    return out


def axis(a):
    return np.pad(np.asarray(a)[:NTOR + 1], (0, max(0, NTOR + 1 - len(a))))


inp = replace(
    inp, lfreeb=True, mgrid_file="essos_coils(direct)", mpol=MPOL, ntor=NTOR, nzeta=NZETA,
    rbc=truncate(inp.rbc), zbs=truncate(inp.zbs), rbs=truncate(inp.rbs), zbc=truncate(inp.zbc),
    raxis_c=axis(inp.raxis_c), zaxis_s=axis(inp.zaxis_s),
    raxis_s=axis(inp.raxis_s), zaxis_c=axis(inp.zaxis_c),
    phiedge=PHIEDGE, ns_array=[NS], niter_array=[NITER], ftol_array=[FTOL],
    pres_scale=0.0, curtor=0.0)


def b_error(res):
    """Largest relative |B| error against the coils on each surface in SURFACES."""
    w = vj.wout_from_result(inp, res)
    xm, xn = np.asarray(w.xm, float), np.asarray(w.xn, float)
    xmn, xnn = np.asarray(w.xm_nyq, float), np.asarray(w.xn_nyq, float)
    rmnc, zmns, bmnc = (np.asarray(a) for a in (w.rmnc, w.zmns, w.bmnc))
    s_half = (np.arange(1, w.ns) - 0.5) / (w.ns - 1)
    theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 64, endpoint=False),
                             np.linspace(0, np.pi, 33), indexing="ij")
    errors = []
    for s in SURFACES:
        j = int(np.argmin(abs(s_half - s))) + 1   # half-grid row between full rows j-1, j
        angle = theta[..., None] * xm - phi[..., None] * xn
        R = np.cos(angle) @ (0.5 * (rmnc[j] + rmnc[j - 1]))
        Z = np.sin(angle) @ (0.5 * (zmns[j] + zmns[j - 1]))
        B = np.cos(theta[..., None] * xmn - phi[..., None] * xnn) @ bmnc[j]
        xyz = np.stack([R * np.cos(phi), R * np.sin(phi), Z], -1).reshape(-1, 3)
        Bc = np.linalg.norm(np.asarray(coil_B(jnp.asarray(xyz))), axis=1).reshape(R.shape)
        errors.append(float(np.abs(B / Bc - 1.0).max()))
    return errors


converged = vj.solve_free_boundary(inp, external_field=field, error_on_no_convergence=False, verbose=True)
relaxed = vj.relax_free_boundary_constraint(inp, converged, external_field=field,
                                            iterations=RELAX_ITERATIONS, verbose=True)

print(f"\nmax |B|/|B_coils| - 1 (MPOL = {MPOL}, NTOR = {NTOR}, NS = {NS})")
print(f"{'':>10s} {'fsq':>8s} " + " ".join(f"{'s = ' + format(s, 'g'):>10s}" for s in SURFACES))
for label, res in (("converged", converged), ("relaxed", relaxed)):
    fsq = float(res.fsqr) + float(res.fsqz) + float(res.fsql)
    print(f"{label:>10s} {fsq:8.1e} " + " ".join(f"{e:10.1e}" for e in b_error(res)))
print(f"\nrelaxation kept iteration {relaxed.iterations} of {RELAX_ITERATIONS}")
