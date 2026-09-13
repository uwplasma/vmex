"""Independent QA pilot checks using fourth-order interior response-field curls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("--target-current", type=float, required=True)
a = p.parse_args()
root = a.run.resolve()
mu0 = 4e-7 * np.pi


def derivative(v, h, axis):
    q = (-np.roll(v, -2, axis) + 8*np.roll(v, -1, axis)
         - 8*np.roll(v, 1, axis) + np.roll(v, 2, axis)) / (12*h)
    # R/Z outer two layers are excluded by every reduction below.
    return q


def grid(f):
    """Native files store extents/dimensions; coordinate arrays are optional."""
    sizes = tuple(len(f.dimensions[k]) for k in ("phi", "Z", "R"))
    if min(sizes) < 5:
        raise ValueError(f"{f.filepath()}: fourth-order stencil needs >=5 points per axis")
    scalars = {}
    for key in ("mtor", "rminb", "rmaxb", "zminb", "zmaxb"):
        value = np.asarray(f[key][:])
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError(f"{f.filepath()}: invalid scalar {key}")
        scalars[key] = float(value.item())
    nfp = scalars["mtor"]
    if nfp < 1 or nfp != int(nfp):
        raise ValueError("mtor must be a positive integer")
    if not (0 < scalars["rminb"] < scalars["rmaxb"] and scalars["zminb"] < scalars["zmaxb"]):
        raise ValueError("Invalid cylindrical grid extents")
    coords = (np.linspace(scalars["rminb"], scalars["rmaxb"], sizes[2]),
              np.linspace(scalars["zminb"], scalars["zmaxb"], sizes[1]),
              np.arange(sizes[0]) * (2*np.pi/nfp/sizes[0]))
    for key, expected in zip(("R", "Z", "phi"), coords):
        if key in f.variables:
            actual = np.asarray(f[key][:])
            if f[key].dimensions != (key,) or actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"{f.filepath()}: {key} must match native uniform grid; phi starts at zero and excludes period endpoint")
    return sizes, scalars, coords


def matching_grid(f, reference):
    candidate = grid(f)
    if candidate[0] != reference[0] or any(not np.isclose(candidate[1][k], v, rtol=1e-12, atol=1e-12) for k, v in reference[1].items()):
        raise ValueError(f"{f.filepath()}: grid differs from prescribed vacuum")


def field_dimensions(f, names, time=False):
    expected = ("time", "phi", "Z", "R") if time else ("phi", "Z", "R")
    for name in names:
        if f[name].dimensions != expected:
            raise ValueError(f"{f.filepath()}: {name} dimensions must be {expected}")


with Dataset(root / "vacuum.nc") as f:
    reference_grid = grid(f)
    R, Z, phi = reference_grid[2]
    field_dimensions(f, ("Bvac_R", "Bvac_phi", "Bvac_Z"))
    vacuum = np.stack([f[k][:] for k in ("Bvac_R", "Bvac_phi", "Bvac_Z")], axis=-1)
    if not np.isfinite(vacuum).all():
        raise ValueError("Nonfinite prescribed vacuum field")
with Dataset(root / "limiter-12.nc") as f:
    matching_grid(f, reference_grid)
    field_dimensions(f, ("limiter",))
    limiter = np.asarray(f["limiter"][:])
    if not np.isin(limiter, (0., 1.)).all():
        raise ValueError("Expected native binary limiter mask")
    wall = limiter > .5
dr, dz, dp = R[1]-R[0], Z[1]-Z[0], phi[1]-phi[0]
rr = R[None, None, :]
interior = np.zeros(wall.shape, dtype=bool)
interior[:, 2:-2, 2:-2] = True
inside = interior & wall
dv = rr * dr * dz * 2*np.pi / len(phi)
rows = []
previous = None
with Dataset(root / "hint.nc") as f:
    matching_grid(f, reference_grid)
    field_dimensions(f, ("B_R", "B_phi", "B_Z", "P"), time=True)
    if not f["P"].shape[0]:
        raise ValueError("No HINT snapshots")
    for i in range(f["P"].shape[0]):
        B = np.stack([f[k][i] for k in ("B_R", "B_phi", "B_Z")], axis=-1)
        P = np.asarray(f["P"][i])
        response = B - vacuum
        br, bp, bz = [response[..., k] for k in range(3)]
        curl = np.stack((derivative(bz, dp, 0)/rr - derivative(bp, dz, 1),
                         derivative(br, dz, 1) - derivative(bz, dr, 2),
                         (derivative(rr*bp, dr, 2)-derivative(br, dp, 0))/rr), axis=-1)
        div = (derivative(rr*br, dr, 2) + derivative(bp, dp, 0))/rr + derivative(bz, dz, 1)
        grad = np.stack((derivative(P, dr, 2), derivative(P, dp, 0)/rr,
                         derivative(P, dz, 1)), axis=-1)
        jxb = np.cross(curl, B)
        force = jxb - grad
        mask = inside & (P > .01 * P.max())
        denominator = np.sum(np.sum(grad**2, axis=-1)*mask*dv)
        supports = {"inside_wall": inside, "positive_pressure": interior & (P > 0),
                    "pressure_above_one_percent": interior & (P > .01*P.max()),
                    "outside_wall": interior & ~wall, "all_interior": interior}
        currents = {}
        force_metrics = {}
        for name, support in supports.items():
            cuts = np.sum(np.where(support, curl[..., 1], 0.), axis=(1, 2))*dr*dz/mu0
            currents[name] = dict(current_A_by_cut=cuts.tolist(), mean_current_A=float(cuts.mean()),
                                 cells=int(support.sum()))
        for name, support in {"pressure_above_one_percent_inside_wall": mask,
                              "positive_pressure_interior": supports["positive_pressure"]}.items():
            numerator = float(np.sum(np.where(support, np.sum(force**2, axis=-1)*dv, 0.)))
            grad_sq = float(np.sum(np.where(support, np.sum(grad**2, axis=-1)*dv, 0.)))
            jxb_sq = float(np.sum(np.where(support, np.sum(jxb**2, axis=-1)*dv, 0.)))
            force_metrics[name] = dict(cells=int(support.sum()),
                force_squared_integral_T4_m=numerator, grad_pressure_squared_integral_T4_m=grad_sq,
                jxb_squared_integral_T4_m=jxb_sq,
                ratio_to_grad_pressure_squared=numerator/grad_sq if grad_sq>0 else None,
                ratio_to_grad_pressure_squared_plus_jxb_squared=numerator/(grad_sq+jxb_sq) if grad_sq+jxb_sq>0 else None)
        current = np.asarray(currents["inside_wall"]["current_A_by_cut"])
        row = dict(snapshot=i, time=float(np.asarray(f["t_snap"][i]).ravel()[0]),
                   finite=bool(np.isfinite(B).all() and np.isfinite(P).all()),
                   max_pressure_Pa=float(P.max()/mu0), pressure_integral_J=float(np.sum(P*dv)/mu0),
                   current_A_by_cut=current.tolist(), mean_current_A=float(current.mean()),
                   mean_current_fraction_of_target=float(current.mean()/a.target_current),
                   current_by_support=currents, force_by_support=force_metrics,
                   response_divergence_rms_T_per_m=float(np.sqrt(np.mean(div[inside]**2))),
                   force_ratio_squared_pressure_mask=float(np.sum(np.sum(force**2, axis=-1)*mask*dv)/denominator) if denominator>0 else None,
                   positive_pressure_cells_outside_wall=int(np.sum((P>1e-8*P.max()) & ~wall)),
                   pressure_fraction_outside_wall=float(np.sum(P*dv*~wall)/np.sum(P*dv)) if np.sum(P*dv)>0 else None)
        if previous is not None and np.linalg.norm(P)>0:
            row["relative_pressure_change_l2"] = float(np.linalg.norm(P-previous)/np.linalg.norm(P))
        previous = P
        rows.append(row)
    np.savez_compressed(root / "final-snapshot.npz", R=R, Z=Z, phi=phi, B_cyl=B, P_mu0_Pa=P)
report = dict(convergence_certified=False, target_current_A=a.target_current,
              grid_validation=dict(shape_phi_Z_R=reference_grid[0], **reference_grid[1],
                                   phi_convention="Uniform one field period from zero, endpoint excluded; volume integrals expanded to full torus"),
              current_method="Fourth-order curl of B-Bvac, rectangular R/Z integral inside numerical wall; excludes two outer grid layers",
              current_support_notes="Legacy current fields use inside_wall. Positive-pressure supports do not additionally restrict to the wall, matching native P>0 selection on retained interior cells. Native ghost-layer contributions are excluded; outside_wall and all_interior expose cancellation and leakage.",
              force_method="curl(B-Bvac) cross B - grad(mu0 p), volume-weighted squared ratio to grad(mu0 p), P>1% max inside wall",
              force_support_notes="Legacy force ratio uses P>1% max inside wall and grad(P)^2 denominator. Native printed <F>_r uses P>0 and grad(P)^2+|curl(response) cross B|^2; both ratios are reported separately. Interior-only finite differences do not reproduce native boundary/ghost contributions.",
              pressure_integral_convention="Full-grid integral of p dV in joules; not thermal energy (3/2 times this integral for isotropic monatomic convention)",
              limitation="Response subtraction removes prescribed-vacuum grid curl; not a force or topology convergence certificate",
              source_sha256=hashlib.sha256((root / "hint.nc").read_bytes()).hexdigest(), snapshots=rows)
(root / "pilot-summary.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(rows[-1], indent=2))
