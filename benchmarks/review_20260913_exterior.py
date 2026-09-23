#!/usr/bin/env python
"""Known-answer tests of the exterior (virtual-casing) field.

Feeds the ``exterior_field`` block of ``benchmarks/review_20260913.json``.
On a current-free equilibrium the plasma field outside the LCFS is exactly
zero, so every non-zero value returned there is quadrature or representation
error. ``--reference`` instead compares the default paths against a 256x256
direct evaluation on a finite-beta deck (self-convergence).

``--reference`` uses the near-surface continuation that 0.12.0 removed
(``with_near_surface_continuation``); run it at the record's revision,
``f09288b3``. The graded rule that replaced it is measured by
``benchmarks/extender_ab.py`` and ``tests/test_near_surface_quadrature.py``.

    PYTHONPATH=. python benchmarks/review_20260913_exterior.py wout.nc out.json
    PYTHONPATH=. python benchmarks/review_20260913_exterior.py wout.nc out.json --reference
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np
from matplotlib.path import Path

import vmex
from vmex.core.extender import VmecExtender
from vmex.core.virtual_casing import surface_field_data_from_wout


class Surface:
    """Boundary geometry, outward normals and an inside/outside test."""

    def __init__(self, wout):
        self.nfp = int(wout.nfp)
        self.xm = np.asarray(wout.xm)
        self.xn = np.asarray(wout.xn)
        self.rmnc = np.asarray(wout.rmnc)[-1]
        self.zmns = np.asarray(wout.zmns)[-1]
        self.a = float(wout.Aminor_p)

    def rz(self, theta, phi):
        angle = self.xm[None, :] * theta[:, None] - self.xn[None, :] * phi[:, None]
        return (self.rmnc[None, :] * np.cos(angle)).sum(1), (self.zmns[None, :] * np.sin(angle)).sum(1)

    def xyz(self, theta, phi):
        r, z = self.rz(theta, phi)
        return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)

    def outside(self, point):
        phi = np.arctan2(point[1], point[0])
        grid = np.linspace(0.0, 2.0 * np.pi, 721)
        r, z = self.rz(grid, np.full_like(grid, phi))
        return not Path(np.c_[r, z]).contains_point((np.hypot(point[0], point[1]), point[2]))

    def targets(self, count, seed=1):
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.0, 2.0 * np.pi, count)
        phi = rng.uniform(0.0, 2.0 * np.pi / self.nfp, count)
        step = 1e-5
        base = self.xyz(theta, phi)
        normal = np.cross(self.xyz(theta + step, phi) - self.xyz(theta - step, phi),
                          self.xyz(theta, phi + step) - self.xyz(theta, phi - step))
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        for i in range(count):
            if not self.outside(base[i] + 0.01 * self.a * normal[i]):
                normal[i] *= -1.0
        return base, normal


def trapezoid_field(wout, surface, nphi, points):
    """Full-torus trapezoid rule on VMEX's own surface data (independent of vc_jax)."""
    data = surface_field_data_from_wout(wout, nphi=nphi, ntheta=nphi)
    gamma = np.asarray(data.gamma).reshape(3, -1).T
    field = np.asarray(data.B_total).reshape(3, -1).T
    area_vector = np.asarray(data.area_vector).reshape(3, -1).T
    weight = (2.0 * np.pi / nphi) * (2.0 * np.pi / surface.nfp / nphi)

    def rotate(values, k):
        c, s = np.cos(2.0 * np.pi * k / surface.nfp), np.sin(2.0 * np.pi * k / surface.nfp)
        return values @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T

    sources = np.concatenate([rotate(gamma, k) for k in range(surface.nfp)])
    fields = np.concatenate([rotate(field, k) for k in range(surface.nfp)])
    areas = np.concatenate([rotate(area_vector, k) for k in range(surface.nfp)])
    norm = np.linalg.norm(areas, axis=1)
    current = np.cross(areas / norm[:, None], fields)
    out = []
    for x in points:
        r = x[None, :] - sources
        r3 = np.linalg.norm(r, axis=1) ** 3
        out.append((np.cross(current, r) / r3[:, None] * (norm * weight)[:, None]).sum(0) / (4.0 * np.pi))
    return np.array(out)


def vacuum_identity(wout, surface, out):
    base, normal = surface.targets(40)
    scale = float(wout.volavgB)
    rows = []
    for nphi in (32, 64, 128, 256):
        for d in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
            points = base + d * surface.a * normal
            keep = np.array([surface.outside(p) for p in points])
            error = np.linalg.norm(trapezoid_field(wout, surface, nphi, points[keep]), axis=1) / scale
            rows.append({"N": nphi, "d_over_a": d, "med": float(np.median(error)),
                         "max": float(error.max()), "targets": int(keep.sum())})
            print(json.dumps(rows[-1]), flush=True)
    out["vacuum_identity"] = rows


def finite_beta_reference(wout, surface, out):
    base, normal = surface.targets(40)
    scale = float(wout.volavgB)
    def build(n):
        return VmecExtender.from_wout(wout, external_field=None, plasma="include", nphi=n, ntheta=n)

    reference = build(256).plasma_field
    coarse = build(32)
    start = time.time()
    plan = coarse.with_near_surface_continuation().near_surface_plan
    out["taylor_plan_build_s"] = time.time() - start
    direct64 = build(64).plasma_field
    rows = []
    for d in (0.5, 0.2, 0.1, 0.05):
        x = jnp.asarray(base + d * surface.a * normal)
        truth = np.asarray(reference.B_plasma_xyz(x))

        def err(b):
            return np.linalg.norm(np.asarray(b) - truth, axis=1) / scale

        e32, e64 = err(coarse.plasma_field.B_plasma_xyz(x)), err(direct64.B_plasma_xyz(x))
        et = err(coarse.plasma_field.B_plasma_near_surface_xyz(x, plan))
        rows.append({"d_over_a": d, "Bplasma_over_B": float(np.linalg.norm(truth, axis=1).mean() / scale),
                     "direct32_med": float(np.median(e32)), "direct32_max": float(e32.max()),
                     "direct64_med": float(np.median(e64)), "direct64_max": float(e64.max()),
                     "taylor32_med": float(np.median(et)), "taylor32_max": float(et.max())})
        print(json.dumps(rows[-1]), flush=True)
    out["finite_beta_self_convergence"] = rows


def main():
    wout = vmex.read_wout(sys.argv[1])
    surface = Surface(wout)
    out = {"wout": os.path.basename(sys.argv[1]), "ctor": float(wout.ctor), "volavgB": float(wout.volavgB)}
    if "--reference" in sys.argv:
        finite_beta_reference(wout, surface, out)
    else:
        vacuum_identity(wout, surface, out)
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
