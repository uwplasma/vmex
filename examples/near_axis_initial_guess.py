"""Import a pyQSC/pyQIC boundary and seed VMEX's straight-field-line angle.

Install ``qsc`` or ``qicna`` (which imports as ``qic``) as selected below. Their existing
VMEC writer supplies the boundary, magnetic axis, pressure, current and flux.
The finite-radius solve remains VMEX's nested-surface equilibrium problem;
this is an asymptotic initial guess, not a near-axis-constrained equilibrium.

The near-axis foot point has Boozer angle varphi=phi0+nu(phi0). If readin
reverses theta, theta_V=pi-theta_B and iota_V=-iota_B. In either orientation,
lambda_V=-(dtheta_V/dtheta_B)*iota_B*(varphi-phi_cyl), so theta_V+lambda_V
is straight along a near-axis field line. The host conversion is not
differentiable; it does not differentiate through the near-axis generator.
"""

from dataclasses import replace
from importlib import import_module
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import vmex
from vmex.core.solver import _initial_state, prepare_runtime
from vmex.core.transforms import physical_to_internal_scale

PACKAGE = "qsc"                 # "qsc" or "qic"; pip install qsc / qicna
CONFIGURATION = "r1 section 5.1"
RADIUS = 0.025                  # near-axis minor radius [m]
NPHI = 61
NS, MPOL, NTOR = 17, 5, 6
NTHETA, NZETA = 24, 16
FTOL, MAX_ITERATIONS = 1e-12, 4000
USE_LAMBDA_SEED = True
OUTPUT_DIR = Path("output_near_axis_initial_guess")

###############################################################################
# End of input parameters.
###############################################################################

if PACKAGE not in ("qsc", "qic"):
    raise ValueError("PACKAGE must be 'qsc' or 'qic'")
try:
    near_axis_type = getattr(import_module(PACKAGE), "Qsc" if PACKAGE == "qsc" else "Qic")
except ImportError as error:
    distribution = "qsc" if PACKAGE == "qsc" else "qicna"
    raise ImportError(f"Install the optional near-axis package: pip install {distribution}") from error
near_axis = near_axis_type.from_paper(CONFIGURATION, nphi=NPHI)
if near_axis.lasym:
    raise ValueError("this example's lambda seed requires stellarator symmetry")
OUTPUT_DIR.mkdir(exist_ok=True)
input_path = OUTPUT_DIR / "input.near_axis"
near_axis.to_vmec(str(input_path), RADIUS, ntheta=NTHETA,
    params=dict(mpol=MPOL, ntor=NTOR, ns_array=[NS], ftol_array=[FTOL], niter_array=[MAX_ITERATIONS]))
inp = vmex.VmecInput.from_file(input_path).change_resolution(
    mpol=MPOL, ntor=NTOR, ntheta=NTHETA, nzeta=NZETA)
inp.to_indata(input_path)
runtime = prepare_runtime(inp)
initial = _initial_state(runtime.setup)

if USE_LAMBDA_SEED:
    theta = 2 * np.pi * np.arange(NTHETA) / NTHETA
    # Frenet_to_cylindrical returns an unshifted cylindrical grid, including
    # for Qic objects whose native phi nodes have an offset.
    nphi = near_axis.nphi
    phi = 2 * np.pi * np.arange(nphi) / (nphi * near_axis.nfp)
    m, n = np.asarray(runtime.modes.m), np.asarray(runtime.modes.n)
    theta_vmec = np.pi - theta if runtime.setup.lflip else theta
    phase = theta_vmec[:, None, None] * m - phi[None, :, None] * n * near_axis.nfp
    factor = np.where((m == 0) & (n == 0), 1.0, 2.0) / (NTHETA * nphi)
    theta_sign = -1 if runtime.setup.lflip else 1
    coefficients = []
    for rho in np.sqrt(np.asarray(runtime.setup.s_full)):
        _, _, foot = near_axis.Frenet_to_cylindrical(RADIUS * rho, NTHETA)
        lam = -theta_sign * near_axis.iota * (foot + near_axis.nu_spline(foot) - phi)
        coefficients.append(factor * np.einsum("tp,tpk->k", lam, np.sin(phase)))
    scale = physical_to_internal_scale(runtime.modes, runtime.trig)
    initial = replace(initial, L_sin=jnp.asarray(coefficients) * scale[None, :]
        * runtime.setup.phipf[:, None] / runtime.setup.lamscale)

result = vmex.solve(inp, initial_state=initial, polish=False, verbose=True)
wout = vmex.wout_from_result(inp, result)
path = vmex.write_wout(OUTPUT_DIR / "wout_near_axis.nc", wout)
expected_iota = (-1 if runtime.setup.lflip else 1) * near_axis.iota
print(f"Near-axis iota {expected_iota:.8f}; solved axis iota {wout.iotaf[0]:.8f}; wrote {path}")
