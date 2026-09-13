"""Native flux-map refinement and imposed parallel-current cut diagnostics."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from netCDF4 import Dataset

root = Path(__file__).resolve().parents[1]
rows = []
for case in ("beta0p5", "beta2p5"):
    source = root / "qa-source" / case
    metadata = json.loads((source / "manifest.json").read_text())
    table = np.loadtxt(source / "current_mapping.csv", delimiter=",", skiprows=1)
    for nr, ntor in ((64, 32), (128, 64)):
        level = f"n{nr}t{ntor}"
        vacuum = root / "runs/qa-vacuum-grid/gpu" / case / level / "vacuum.nc"
        work = root / "runs/qa-current-refinement" / case / level
        work.mkdir(parents=True, exist_ok=True)
        flux = work / "flux.nc"
        with Dataset(vacuum) as f:
            R, Z = np.asarray(f["R"][:]), np.asarray(f["Z"][:])
            bphi = np.asarray(f["Bvac_phi"][:])
        if nr == 64:
            reference = root / "runs/qa-native-inputs" / case / "flux.nc"
            if not flux.exists():
                flux.symlink_to(reference)
        else:
            wout = work / "wout.nc"
            if not wout.exists():
                wout.symlink_to(next((source / "inputs").glob("wout*")))
            deck = (
                "&nlinp1 flx_form='netcdf',flx_file='flux.nc',file_format='netcdf',"
                f"wout_file='wout.nc',mtor=2,nr={nr},nz={nr},ntor={ntor},"
                f"rminb={R[0]},rmaxb={R[-1]},zminb={Z[0]},zmaxb={Z[-1]},"
                "ntheta=256,slimit=1.0 /\n"
            )
            (work / "mkflx.input").write_text(deck)
            with (work / "mkflx.log").open("w") as log:
                subprocess.run([root / "build-release/MKFLX/mkflx.exe"], cwd=work,
                               input=deck, text=True, stdout=log, stderr=subprocess.STDOUT,
                               timeout=180, check=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
        with Dataset(flux) as f:
            s = np.asarray(f["norm_s"][:])
        assert np.isfinite(s).all()
        lam = np.where(s < 1, np.interp(s, table[:, 0], table[:, 1]), 0.0)
        integral = np.sum(lam * bphi, axis=(1, 2)) * (R[1]-R[0]) * (Z[1]-Z[0])
        assert np.isfinite(integral).all() and integral[0] != 0
        current = metadata["source_current_A"] * integral / integral[0]
        row = dict(case=case, grid=[nr, nr, ntor], initial_current_A_by_cut=current.tolist(),
                   max_relative_cut_variation=float(np.max(np.abs(current/current[0]-1))),
                   normalization_integral_phi0=float(integral[0]),
                   meaning="Imposed lambda(s) B on the initial vacuum field; not relaxed total current")
        rows.append(row)
        print(case, level, row["max_relative_cut_variation"], flush=True)
(root / "local-support/qa-current-refinement.json").write_text(json.dumps(rows, indent=2)+"\n")
