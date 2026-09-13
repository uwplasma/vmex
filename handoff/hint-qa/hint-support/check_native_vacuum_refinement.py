"""Compare native MAGVAL interpolation at fixed direct-coil targets across grids."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np
from netCDF4 import Dataset

p = argparse.ArgumentParser()
p.add_argument("--build", default="build-release")
p.add_argument("--grid-root", default="runs/qa-vacuum-grid/gpu")
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
build = root / a.build
config = json.loads((build / "build_config.json").read_text())
dest = build / "MAGVAL"
exe = dest / "magval_points.exe"
objects = sorted(x for x in dest.glob("*.o") if x.name not in ("main.o", "magout.o"))
subprocess.run(
    [config["compiler"], *shlex.split(config["flags"]), *shlex.split(config["includes"]),
     "-I.", str(root / "local-support/magval_points.f90"), *map(str, objects),
     *shlex.split(config["libraries"]), "-o", str(exe)], cwd=dest, check=True,
)
rows = []
for case in ("beta0p5", "beta2p5"):
    for level in ("n64t32", "n128t64"):
        source = root / a.grid_root / case / level
        work = root / "runs/native-vacuum-refinement" / case / level
        work.mkdir(parents=True, exist_ok=True)
        # MAGVAL consumes equilibrium-format fields, so wrap the prescribed
        # vacuum grid without changing its samples or native interpolation.
        temporary = work / "vacuum-as-field.nc"
        with Dataset(source / "vacuum.nc") as src, Dataset(temporary, "w") as dst:
            for key, dim in src.dimensions.items():
                dst.createDimension(key, len(dim))
            for key in ("mtor", "rminb", "rmaxb", "zminb", "zmaxb", "R", "phi", "Z"):
                value = src[key]
                dst.createVariable(key, value.dtype, value.dimensions)[:] = value[:]
            for key in ("B_R", "B_phi", "B_Z"):
                dst.createVariable(key, "f8", ("phi", "Z", "R"))[:] = src[key.replace("B_", "Bvac_")][:]
            for key in ("P", "v_R", "v_phi", "v_Z"):
                dst.createVariable(key, "f8", ("phi", "Z", "R"))[:] = 0.0
        target = np.load(source / "target-check.npz")
        xyz = target["xyz"]
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        points = np.column_stack((np.linalg.norm(xyz[:, :2], axis=1), phi, xyz[:, 2]))
        stdin = str(len(points)) + "\n" + "\n".join(" ".join(map(str, row)) for row in points)
        with (work / "native.log").open("w") as log:
            subprocess.run([str(exe), temporary.name], cwd=work, input=stdin, text=True,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        b = np.loadtxt(work / "native-points.csv", delimiter=",")
        cart = np.column_stack((b[:, 0]*np.cos(phi)-b[:, 1]*np.sin(phi),
                                b[:, 0]*np.sin(phi)+b[:, 1]*np.cos(phi), b[:, 2]))
        error = np.linalg.norm(cart - target["B_direct"], axis=1)
        assert np.isfinite(error).all()
        np.savez_compressed(work / "targets.npz", xyz=xyz, B_native=cart, B_direct=target["B_direct"])
        rows.append(dict(case=case, level=level, rms_error_T=float(np.sqrt(np.mean(error**2))),
                         max_error_T=float(error.max())))
        temporary.unlink()  # Only the generated format wrapper; source grids remain.
        print(rows[-1], flush=True)
(root / "local-support/native-vacuum-refinement.json").write_text(json.dumps(rows, indent=2)+"\n")
