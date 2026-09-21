#!/usr/bin/env python3
"""Evaluate an observational native-form force diagnostic for one VMEX root."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import re
import subprocess
import sys
import time

import numpy as np


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--samples-output", required=True, type=Path)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--expected-tree", required=True)
    p.add_argument("--expected-state", type=sha256_value)
    p.add_argument("--angular-count", type=angular_count, default=16)
    return p.parse_args()


def sha256_value(value):
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError("expected a lowercase 64-hex SHA-256")
    return value


def angular_count(value):
    count = int(value)
    if count < 4:
        raise argparse.ArgumentTypeError("angular count must be at least 4")
    return count


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True,
        stderr=subprocess.DEVNULL).strip()


def rel_l2(value, reference):
    return float(np.linalg.norm((value-reference).ravel()) /
                 max(np.linalg.norm(reference.ravel()), 1e-300))


def require_finite(value, label="result"):
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError(f"{label} is not finite")


def main():
    a = arguments()
    root = a.source_root.resolve()
    inp_path = a.input if a.input.is_absolute() else root / a.input
    outputs = (a.output.resolve(), a.samples_output.resolve())
    if outputs[0] == outputs[1]:
        raise ValueError("JSON and sample outputs must be distinct")
    if any(not p.parent.is_dir() for p in outputs):
        raise FileNotFoundError("output parent directory does not exist")
    if any(p.exists() for p in outputs):
        raise FileExistsError("output already exists")
    if git(root, "status", "--porcelain"):
        raise RuntimeError("source checkout is not clean")
    head, tree = git(root, "rev-parse", "HEAD"), git(root, "rev-parse", "HEAD^{tree}")
    if (head, tree) != (a.expected_head, a.expected_tree):
        raise RuntimeError("source revision does not match the requested identity")

    # X64 must be selected before importing JAX or VMEX.
    os.environ["JAX_ENABLE_X64"] = "1"
    sys.path.insert(0, str(root))
    import jax
    import jax.numpy as jnp
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core.profiles import MU0, pressure
    jax.config.update("jax_enable_x64", True)
    if not jax.config.x64_enabled:
        raise RuntimeError("JAX x64 mode is required")
    for module in (vj, opt):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("VMEX was not imported from --source-root")

    started = time.monotonic()
    inp = vj.VmecInput.from_file(inp_path)
    if float(inp.gamma) != 0.0:
        raise ValueError("this pressure-gradient diagnostic requires GAMMA=0")
    eq = opt.solve_equilibrium(inp, raise_on_max_iterations=True)
    jax.block_until_ready(eq.solution.R_cos)
    solve_seconds = time.monotonic()-started

    state_digest = hashlib.sha256()
    for leaf in jax.tree.leaves(eq.solution):
        state_digest.update(np.ascontiguousarray(leaf,dtype=np.float64).tobytes())
    state_sha256 = state_digest.hexdigest()
    if a.expected_state is not None and state_sha256 != a.expected_state:
        raise RuntimeError("solved state does not match --expected-state")

    # q=(s,theta,phi), with physical geometric phi over one field period.
    x, wx = np.polynomial.legendre.leggauss(5)
    s = .4*x+.5
    ws = .4*wx
    n = a.angular_count
    theta = (np.arange(n)+.5)*2*np.pi/n
    nfp = int(eq.solver_context.resolution.nfp)
    phi = (np.arange(n)+.375)*2*np.pi/(nfp*n)

    field, flux = eq.field, eq.field_in_flux_coordinates()
    def p_of_s(value):
        return pressure(inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f,
                        value, pres_scale=inp.pres_scale, bloat=inp.bloat,
                        spres_ped=inp.spres_ped)

    map_xyz = jax.jit(flux.to_xyz_batch)
    map_jacobian = jax.jit(jax.vmap(jax.jacfwd(flux.to_xyz)))
    profile_gradient = jax.jit(jax.vmap(jax.grad(p_of_s)))
    chunks = {name: [] for name in
              ("q", "xyz", "weights", "B", "gradB", "gradp", "det")}
    for sv, radial_weight in zip(s, ws, strict=True):
        tt, pp = np.meshgrid(theta, phi, indexing="ij")
        q_surface = np.stack((np.full_like(tt,sv),tt,pp),axis=-1).reshape(-1,3)
        qj = jnp.asarray(q_surface)
        xyz_surface = map_xyz(qj)
        eq.set_points_flux(qj)
        B_surface, gradB_surface = field.B(), field.gradB()
        dx_dq = map_jacobian(qj)
        grad_s = jnp.linalg.inv(dx_dq)[:,0,:]
        det_surface = jnp.linalg.det(dx_dq)
        gradp_surface = profile_gradient(qj[:,0])[:,None]*grad_s
        quadrature_surface = (np.full(n*n,radial_weight)*(2*np.pi/n)
                              *(2*np.pi/(nfp*n)))
        weights_surface = jnp.asarray(quadrature_surface)*jnp.abs(det_surface)
        jax.block_until_ready(gradp_surface)
        for name,value in (("q",q_surface),("xyz",xyz_surface),
                           ("weights",weights_surface),("B",B_surface),
                           ("gradB",gradB_surface),("gradp",gradp_surface),
                           ("det",det_surface)):
            chunks[name].append(value)
    q = np.concatenate(chunks["q"],axis=0)
    xyz = jnp.concatenate(chunks["xyz"],axis=0)
    weights = jnp.concatenate(chunks["weights"],axis=0)
    B = jnp.concatenate(chunks["B"],axis=0)
    gradB = jnp.concatenate(chunks["gradB"],axis=0)
    gradp = jnp.concatenate(chunks["gradp"],axis=0)
    det = jnp.concatenate(chunks["det"],axis=0)
    curl = jnp.stack((gradB[:,2,1]-gradB[:,1,2],
                      gradB[:,0,2]-gradB[:,2,0],
                      gradB[:,1,0]-gradB[:,0,1]), axis=-1)
    J = curl/MU0
    lorentz = jnp.cross(J, B)
    force = lorentz-gradp
    jax.block_until_ready(force)

    F = jnp.linalg.norm(force, axis=-1)
    Gp = jnp.linalg.norm(gradp, axis=-1)
    L = jnp.linalg.norm(lorentz, axis=-1)
    Gm = jnp.linalg.norm(jnp.einsum("ni,nij->nj", B, gradB)/MU0, axis=-1)
    divB = jnp.trace(gradB, axis1=1, axis2=2)
    signed_det = float(field.spectra["signgs"])*det

    def weighted_metrics(section):
        w = weights[section]
        def avg(value):
            return jnp.sum(w*value)/jnp.sum(w)
        f, gp = F[section], Gp[section]
        force_l2 = jnp.sqrt(avg(f**2))
        return {
            "absolute_l2_N_m3":float(force_l2),
            "volume_average_force_N_m3":float(avg(f)),
            "pointwise_normalized_l2":float(jnp.sqrt(avg(
                (2*f/(L[section]+gp+1e-12))**2))),
            "mean_force_over_mean_grad_p":float(avg(f)/avg(gp)),
            "mean_force_over_mean_magnetic_pressure_gradient":float(
                avg(f)/avg(Gm[section])),
            "magnetic_normalized_l2":float(force_l2/avg(Gm[section])),
            "B_rms_T":float(jnp.sqrt(avg(jnp.sum(B[section]**2,axis=-1)))),
            "divB_rms_T_m":float(jnp.sqrt(avg(divB[section]**2))),
            "divB_max_abs_T_m":float(jnp.max(jnp.abs(divB[section]))),
            "minimum_signed_coordinate_jacobian_m3":float(
                jnp.min(signed_det[section])),
            "nestedness_margin":float(jnp.min(signed_det[section])
                /avg(jnp.abs(signed_det[section]))),
        }

    stage1 = weighted_metrics(slice(None))
    stage1["sampled_window_full_torus_volume_m3"] = float(
        nfp*jnp.sum(weights))

    radial_bins = []
    block = n*n
    for radial_index, sv in enumerate(s):
        section = slice(radial_index*block,(radial_index+1)*block)
        row = weighted_metrics(section)
        row.update({"radial_index":radial_index,"s":float(sv),
            "full_torus_dV_ds_m3":float(
                nfp*jnp.sum(weights[section])/ws[radial_index]),
            "full_torus_window_volume_contribution_m3":float(
                nfp*jnp.sum(weights[section]))})
        radial_bins.append(row)

    # Keep the default subset exact and select the nearest cell centers on
    # other angular grids.
    theta_fraction = np.array((3,11,19,27))/32
    phi_fraction = (np.array((2,10))+.375)/16
    theta_subset = np.rint(theta_fraction*n-.5).astype(int) % n
    phi_subset = np.rint(phi_fraction*n-.375).astype(int) % n
    subset = np.array([np.ravel_multi_index((ir,it,ip),(5,n,n))
                       for ir in (0,1,3,4) for it in theta_subset
                       for ip in phi_subset], dtype=np.int64)
    xs, Bs = np.asarray(xyz)[subset], np.asarray(B)[subset]
    curl0, gp0, force0 = map(lambda value: np.asarray(value)[subset],
                             (curl, gradp, force))
    steps = (1e-3, 5e-4, 2.5e-4)
    levels, saved = [], {}
    for level, ratio in enumerate(steps):
        h = ratio*float(eq.wout.Aminor_p)
        eye = np.eye(3)
        stencil = np.stack([xs+k*h*eye[j] for j in range(3)
                            for k in (-2.,-1.,1.,2.)], axis=1)
        flat = jnp.asarray(stencil.reshape(-1,3))
        sb = np.asarray(field.B(flat)).reshape(32,12,3)
        sq = np.asarray(field.flux_coordinates(flat)).reshape(32,12,3)
        sp = np.asarray(p_of_s(jnp.asarray(sq[:,:,0])))
        gb, gp = np.empty((32,3,3)), np.empty((32,3))
        for j in range(3):
            m2,m1,p1,p2 = (sb[:,4*j+k] for k in range(4))
            gb[:,:,j] = (m2-8*m1+8*p1-p2)/(12*h)
            m2,m1,p1,p2 = (sp[:,4*j+k] for k in range(4))
            gp[:,j] = (m2-8*m1+8*p1-p2)/(12*h)
        cb = np.stack((gb[:,2,1]-gb[:,1,2], gb[:,0,2]-gb[:,2,0],
                       gb[:,1,0]-gb[:,0,1]), axis=-1)
        ff = np.cross(cb/MU0, Bs)-gp
        db = np.trace(gb, axis1=1, axis2=2)
        row = {"h_over_Aminor":ratio, "h_m":h,
               "curl_relative_l2_vs_analytic":rel_l2(cb,curl0),
               "pressure_gradient_relative_l2_vs_analytic":rel_l2(gp,gp0),
               "force_relative_l2_vs_analytic":rel_l2(ff,force0),
               "divB_rms_T_m":float(np.sqrt(np.mean(db**2)))}
        if level:
            row["force_relative_l2_vs_previous_h"] = rel_l2(ff,saved[f"fd_force_level_{level-1}"])
        levels.append(row)
        saved.update({f"fd_curl_level_{level}":cb,
                      f"fd_grad_p_level_{level}":gp,
                      f"fd_force_level_{level}":ff})

    versions = {name:importlib.metadata.version(name)
                for name in ("jax","jaxlib","numpy","vmex")}
    versions["python"] = sys.version.split()[0]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
      "scope":"observational native-form interpolation diagnostic; not a continuum certificate",
      "source":{"head":head,"tree":tree,"input_sha256":sha(inp_path),
                "native_state_sha256":state_sha256,
                "driver_sha256":sha(Path(__file__).resolve())},
      "versions":versions,
      "solve":{"converged":bool(eq.result.converged),
               "iterations":int(eq.result.iterations),"ns":int(eq.wout.ns),
               "mpol":int(eq.wout.mpol),"ntor":int(eq.wout.ntor),"nfp":nfp,
               "ntheta":int(inp.ntheta),"nzeta":int(inp.nzeta),
               "root_fsq":{"fsqr":float(eq.result.fsqr),
                           "fsqz":float(eq.result.fsqz),
                           "fsql":float(eq.result.fsql),
                           "sum":float(eq.result.fsqr+eq.result.fsqz+eq.result.fsql),
                           "max":float(max(eq.result.fsqr,eq.result.fsqz,eq.result.fsql))}},
      "coordinates":{"q_order":["normalized_toroidal_flux_s","poloidal_theta_rad","physical_geometric_phi_rad"],
        "s_interval":[.1,.9],"s_nodes":s.tolist(),"s_quadrature":"5-point Gauss-Legendre",
        "theta_count":n,"theta_shift_cells":.5,"phi_count_per_field_period":n,
        "phi_shift_cells":.375,"point_count":len(q),
        "common_points_sha256":hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),
        "evaluation":"mapped Cartesian points are inverted with exact q as native-field seeds"},
      "conventions":{"gradB":"gradB[i,j] = d B_i / d x_j in Cartesian coordinates",
        "curlB":"(dBz/dy-dBy/dz, dBx/dz-dBz/dx, dBy/dx-dBx/dy)",
        "current_density":"J = curl(B)/mu0 in A/m^2","force":"J cross B minus grad(p), N/m^3",
        "pressure_gradient":"dp/ds from the prescribed gamma=0 profile times row 0 of inverse dx/d(s,theta,phi)",
        "weights":"abs(det(dx/dq)) ds dtheta dphi; nfp is used only for the full-torus window volume",
        "pointwise_normalization":"2|F|/(|J cross B|+|grad p|+1e-12 N/m^3)"},
      "stage1":stage1,"radial_bins":radial_bins,
      "physical":{"prescribed_input":{"phiedge_Wb":float(inp.phiedge),
                                         "curtor_A":float(inp.curtor),
                                         "pressure_axis_Pa":float(p_of_s(jnp.asarray(0.0))),
                                         "pressure_edge_Pa":float(p_of_s(jnp.asarray(1.0)))},
                  "full_mesh_wout":{"phi_edge_Wb":float(eq.wout.phi[-1]),
                                     "toroidal_current_A":float(eq.wout.ctor),
                                     "pressure_axis_Pa":float(eq.wout.presf[0]),
                                     "pressure_edge_Pa":float(eq.wout.presf[-1]),
                                     "volume_m3":float(eq.wout.volume_p)}},
      "stage2":{"method":"fourth-order centered Cartesian differences using the same native B and reconstructed flux-coordinate pressure profile",
                "subset_count":len(subset),"subset_flat_indices":subset.tolist(),"levels":levels},
      "timing_seconds":{"solve":solve_seconds,"total":time.monotonic()-started},
      "resources":{"jax_platform":jax.default_backend(),"visible_device_count":len(jax.devices()),
                   "max_rss_bytes":int(rss*(1 if sys.platform=="darwin" else 1024))},
      "limitations":["the native interior field uses a continuous C2 radial interpolation of the finite-NS SpectralState",
        "the main evaluation performs a seeded Cartesian-to-flux inversion at mapped common points",
        "the Cartesian stencil checks derivative and inversion consistency on one state, not equilibrium radial convergence",
        "no NS, MPOL, or NTOR refinement is included"]}
    arrays = {"q_s_theta_phi":q,"xyz_m":np.asarray(xyz),"volume_weights_m3":np.asarray(weights),
              "B_T":np.asarray(B),"curlB_T_m":np.asarray(curl),"grad_p_N_m3":np.asarray(gradp),
              "force_N_m3":np.asarray(force),"subset_flat_indices":subset,**saved}
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"sample array {name} contains non-finite values")
    require_finite(result)
    with outputs[1].open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    result["samples_sha256"] = sha(outputs[1])
    with outputs[0].open("x") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False); stream.write("\n")
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__ == "__main__":
    main()
