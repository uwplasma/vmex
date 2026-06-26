"""Generate README fsq_total traces for representative single-grid cases.

This script is intended to support README figures that compare convergence
traces between VMEC2000 and vmec_jax using:

- NS_ARRAY = 151
- NITER_ARRAY = 5000
- FTOL_ARRAY = 1e-14

vmec_jax is run strictly via the CLI: `vmec_jax <inputfile>` (no extra flags).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Final

import netCDF4
import numpy as np

from vmec_jax.vmec2000_exec import _patch_indata, find_vmec2000_exec, run_xvmec2000

_C_VMEC2000: Final[str] = "#1f77b4"  # blue
_C_VMEC_JAX: Final[str] = "#ff7f0e"  # orange
_C_VMECPP: Final[str] = "#2ca02c"  # green


def _pyplot():
    """Import matplotlib only when a figure is actually requested."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _find_wout(workdir: Path, *, case: str) -> Path:
    # VMEC typically writes `wout_<case>.nc`.
    direct = workdir / f"wout_{case}.nc"
    if direct.exists():
        return direct
    matches = sorted(workdir.glob(f"wout_{case}*.nc"))
    if matches:
        return matches[0]
    matches = sorted(workdir.glob("wout_*.nc"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"no wout_*.nc found in {workdir}")


def _audit_wout(wout_path: Path) -> dict[str, object]:
    # Keep this summary safe to commit or paste: do not include absolute paths.
    out: dict[str, object] = {"file": wout_path.name}
    with netCDF4.Dataset(str(wout_path), mode="r") as ds:
        # Prefer dimension size when available (most robust across writers).
        if "radius" in ds.dimensions:
            out["radius_dim"] = int(ds.dimensions["radius"].size)
        if "ns" in ds.dimensions:
            out["ns_dim"] = int(ds.dimensions["ns"].size)
        if "ns" in ds.variables:
            try:
                out["ns"] = int(np.asarray(ds.variables["ns"][...]).item())
            except Exception:
                pass
        for key in ("nfp", "mpol", "ntor", "lasym", "lfreeb", "ier_flag"):
            if key in ds.variables:
                v = ds.variables[key][...]
                try:
                    out[key] = v.item()
                except Exception:
                    out[key] = np.asarray(v).tolist()
        for key in ("version_", "input_extension", "mgrid_file"):
            if key in ds.variables:
                v = ds.variables[key][...]
                try:
                    s = netCDF4.chartostring(v).tobytes().decode(errors="ignore")
                    out[key] = s.replace("\x00", "").strip()
                except Exception:
                    out[key] = str(v)
        # Common scalar diagnostics (not guaranteed).
        for key in ("wb", "wp", "w", "fsqr", "fsqz", "fsql"):
            if key in ds.variables:
                v = ds.variables[key][...]
                try:
                    out[key] = float(np.asarray(v).reshape(-1)[-1])
                except Exception:
                    pass
    return out


def _collect_vmec2000_trace(input_path: Path, *, ns: int, niter: int, ftol: float, workdir: Path):
    exe = find_vmec2000_exec()
    if exe is None:
        raise SystemExit("xvmec2000 executable not found")
    vmec = run_xvmec2000(
        input_path,
        exec_path=exe,
        workdir=workdir,
        timeout_s=3600.0,
        indata_updates={
            "NSTEP": "1",
            "NS_ARRAY": f"{int(ns)}",
            "NITER_ARRAY": f"{niter}",
            "FTOL_ARRAY": f"{float(ftol):.3e}",
        },
        keep_workdir=True,
    )
    fsq = []
    it = []
    for stage in vmec.stages:
        for row in stage.rows:
            it.append(int(row.it))
            fsq.append(float(row.fsqr + row.fsqz + row.fsql))
    case = input_path.name.replace("input.", "")
    # Archive threed1 before vmec_jax overwrites it in the same scratch dir.
    if vmec.threed1_path is not None and vmec.threed1_path.exists():
        archived_threed1 = workdir / f"threed1.{case}_VMEC2000"
        shutil.copy2(vmec.threed1_path, archived_threed1)
    wout_path = _find_wout(workdir, case=case)
    archived = workdir / f"wout_{case}_VMEC2000.nc"
    shutil.copy2(wout_path, archived)
    audit = _audit_wout(archived)
    it_arr = np.asarray(it, dtype=int)
    fsq_arr = np.asarray(fsq, dtype=float)
    np.savez(workdir / f"trace_{case}_VMEC2000.npz", it=it_arr, fsq_total=fsq_arr)
    return it_arr, fsq_arr, float(vmec.runtime_s), audit


def _parse_vmec_table_trace(stdout: str) -> tuple[np.ndarray, np.ndarray]:
    it: list[int] = []
    fsq: list[float] = []
    in_table = False

    def _f(tok: str) -> float:
        return float(tok.replace("D", "E").replace("d", "E"))

    for line in stdout.splitlines():
        if line.strip().startswith("ITER") and ("FSQR" in line) and ("FSQZ" in line):
            in_table = True
            continue
        if not in_table:
            continue
        toks = [tok.strip() for tok in line.split()]
        # VMEC++ legacy mode uses `|` separators in the table.
        toks = [tok for tok in toks if tok and tok != "|"]
        if len(toks) < 4 or (not toks[0].isdigit()):
            # End when VMEC prints post-table messages.
            if toks and toks[0].upper() in {"TRY", "EXECUTION", "FILE", "TOTAL", "TIME"}:
                break
            continue
        it.append(int(toks[0]))
        fsq.append(_f(toks[1]) + _f(toks[2]) + _f(toks[3]))

    return np.asarray(it, dtype=int), np.asarray(fsq, dtype=float)


def _collect_vmec_jax_trace(input_path: Path, *, ns: int, niter: int, ftol: float, workdir: Path):
    case = input_path.name.replace("input.", "")
    patched = _patch_indata(
        input_path.read_text(),
        updates={
            "NSTEP": "1",
            "NS_ARRAY": f"{int(ns)}",
            "NITER_ARRAY": f"{int(niter)}",
            "FTOL_ARRAY": f"{float(ftol):.3e}",
        },
    )
    tmp_input = workdir / f"input.{case}"
    tmp_input.write_text(patched)
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["vmec_jax", str(tmp_input)],
        cwd=str(workdir),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    runtime = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"vmec_jax failed for {input_path} (code={proc.returncode})\n{proc.stderr}")
    it, fsq = _parse_vmec_table_trace(proc.stdout)
    # Archive threed1 written by vmec_jax.
    threed1_path = workdir / f"threed1.{case}"
    if threed1_path.exists():
        shutil.copy2(threed1_path, workdir / f"threed1.{case}_vmec_jax")
    wout_path = _find_wout(workdir, case=case)
    archived = workdir / f"wout_{case}_vmec_jax.nc"
    shutil.copy2(wout_path, archived)
    audit = _audit_wout(archived)
    np.savez(workdir / f"trace_{case}_vmec_jax.npz", it=it, fsq_total=fsq)
    return it, fsq, float(runtime), audit


def _collect_vmecpp_trace(input_path: Path, *, ns: int, niter: int, ftol: float, workdir: Path):
    case = input_path.name.replace("input.", "")
    patched = _patch_indata(
        input_path.read_text(),
        updates={
            "NSTEP": "1",
            "NS_ARRAY": f"{int(ns)}",
            "NITER_ARRAY": f"{int(niter)}",
            "FTOL_ARRAY": f"{float(ftol):.3e}",
        },
    )
    tmp_input = workdir / f"input.{case}"
    tmp_input.write_text(patched)

    t0 = time.perf_counter()
    proc = subprocess.run(
        ["vmecpp", "--legacy", str(tmp_input)],
        cwd=str(workdir),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    runtime = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"vmecpp failed for {input_path} (code={proc.returncode})\n{proc.stderr}")

    it, fsq = _parse_vmec_table_trace(proc.stdout)
    wout_path = _find_wout(workdir, case=case)
    archived = workdir / f"wout_{case}_vmecpp.nc"
    shutil.copy2(wout_path, archived)
    audit = _audit_wout(archived)
    np.savez(workdir / f"trace_{case}_vmecpp.npz", it=it, fsq_total=fsq)
    return it, fsq, float(runtime), audit


def _trace_mismatch(it_a: np.ndarray, fsq_a: np.ndarray, it_b: np.ndarray, fsq_b: np.ndarray) -> dict[str, float]:
    # Align by explicit iteration numbers.
    a = {int(i): float(v) for i, v in zip(it_a.tolist(), fsq_a.tolist(), strict=False)}
    b = {int(i): float(v) for i, v in zip(it_b.tolist(), fsq_b.tolist(), strict=False)}
    common = sorted(set(a).intersection(b))
    if not common:
        return {"n_common": 0.0}
    rel = []
    for i in common:
        av = abs(a[i])
        bv = abs(b[i])
        denom = max(av, bv, 1e-300)
        rel.append(abs(a[i] - b[i]) / denom)
    rel = np.asarray(rel, dtype=float)
    return {
        "n_common": float(len(common)),
        "max_rel": float(np.nanmax(rel)),
        "p50_rel": float(np.nanmedian(rel)),
    }


def _plot_panel(
    ax,
    *,
    it_vmec: np.ndarray,
    fsq_vmec: np.ndarray,
    it_jax: np.ndarray,
    fsq_jax: np.ndarray,
    it_vmecpp: np.ndarray | None,
    fsq_vmecpp: np.ndarray | None,
    title: str,
    jax_label: str,
):
    # Many cases have very similar traces; use distinct markers and draw VMEC2000
    # on top with hollow markers so both curves remain visible even when they overlap.
    mark_vmec = max(1, int(max(it_vmec.size, 1) // 35))
    mark_jax = max(1, int(max(it_jax.size, 1) // 45))
    ax.plot(
        it_jax,
        fsq_jax,
        lw=2.6,
        linestyle="-",
        color=_C_VMEC_JAX,
        marker="o",
        ms=2.6,
        markevery=mark_jax,
        alpha=0.95,
        label=jax_label,
        zorder=1,
    )
    if it_vmecpp is not None and fsq_vmecpp is not None and it_vmecpp.size:
        mark_pp = max(1, int(max(it_vmecpp.size, 1) // 40))
        ax.plot(
            it_vmecpp,
            fsq_vmecpp,
            lw=2.6,
            linestyle="-",
            color=_C_VMECPP,
            marker="^",
            ms=3.0,
            markevery=mark_pp,
            mfc="none",
            mec=_C_VMECPP,
            mew=1.0,
            alpha=0.92,
            label="VMEC++",
            zorder=2,
        )
    ax.plot(
        it_vmec,
        fsq_vmec,
        lw=2.6,
        linestyle="--",
        color=_C_VMEC2000,
        marker="s",
        ms=3.6,
        markevery=mark_vmec,
        mfc="none",
        mec=_C_VMEC2000,
        mew=1.1,
        alpha=0.95,
        label="VMEC2000",
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("fsq_total")
    # Do not annotate runtimes on this figure: traces run with NSTEP=1 solely to
    # capture per-iteration values, and those runtimes are not representative.
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--axisym-input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "examples/data/single_grid/input.ITERModel"),
    )
    p.add_argument(
        "--stellarator-input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "examples/data/single_grid/input.LandremanPaul2021_QA_lowres"),
    )
    p.add_argument(
        "--qh-input",
        type=str,
        default="",
        help="Deprecated alias for --stellarator-input.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "docs/_static/figures"),
    )
    p.add_argument(
        "--workdir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "outputs" / "readme_fsq_trace_single_grid_work"),
        help="Scratch directory for VMEC2000/vmec_jax run artifacts (wouts, threed1, logs).",
    )
    p.add_argument(
        "--reuse-workdir",
        action="store_true",
        help="Reuse existing traces under --workdir (skip rerunning solvers).",
    )
    p.add_argument("--ns", type=int, default=151)
    p.add_argument("--niter", type=int, default=5000)
    p.add_argument("--ftol", type=float, default=1e-14)
    p.add_argument("--jax-label", type=str, default="vmec_jax")
    p.add_argument(
        "--skip-vmecpp",
        action="store_true",
        help="Skip VMEC++ trace collection (use when vmecpp is not available).",
    )
    args = p.parse_args()

    axisym_input = Path(args.axisym_input).expanduser().resolve()
    stellarator_input = Path(args.stellarator_input).expanduser().resolve()
    if args.qh_input:
        stellarator_input = Path(args.qh_input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    skip_vmecpp = bool(args.skip_vmecpp)

    axisym_work = workdir / "ITERModel"
    st_work = workdir / "LandremanPaul2021_QA_lowres"
    if not bool(args.reuse_workdir):
        shutil.rmtree(axisym_work, ignore_errors=True)
        shutil.rmtree(st_work, ignore_errors=True)
    axisym_work.mkdir(parents=True, exist_ok=True)
    st_work.mkdir(parents=True, exist_ok=True)

    _empty_arr = np.zeros(0, dtype=float)

    if bool(args.reuse_workdir):
        a_vmec = np.load(axisym_work / "trace_ITERModel_VMEC2000.npz")
        a_jax = np.load(axisym_work / "trace_ITERModel_vmec_jax.npz")
        it_vmec_a, fsq_vmec_a = a_vmec["it"], a_vmec["fsq_total"]
        it_jax_a, fsq_jax_a = a_jax["it"], a_jax["fsq_total"]
        if not skip_vmecpp and (axisym_work / "trace_ITERModel_vmecpp.npz").exists():
            a_pp = np.load(axisym_work / "trace_ITERModel_vmecpp.npz")
            it_pp_a, fsq_pp_a = a_pp["it"], a_pp["fsq_total"]
        else:
            it_pp_a, fsq_pp_a = None, None
        mismatch_a = _trace_mismatch(it_vmec_a, fsq_vmec_a, it_jax_a, fsq_jax_a)
        audit = json.loads((workdir / "readme_fsq_trace_single_grid_wout_audit.json").read_text())
        wout_vmec_a = audit["ITERModel"]["VMEC2000"]
        wout_jax_a = audit["ITERModel"]["vmec_jax"]
        wout_pp_a = audit["ITERModel"].get("vmecpp", {})

        s_vmec = np.load(st_work / "trace_LandremanPaul2021_QA_lowres_VMEC2000.npz")
        s_jax = np.load(st_work / "trace_LandremanPaul2021_QA_lowres_vmec_jax.npz")
        it_vmec_s, fsq_vmec_s = s_vmec["it"], s_vmec["fsq_total"]
        it_jax_s, fsq_jax_s = s_jax["it"], s_jax["fsq_total"]
        if not skip_vmecpp and (st_work / "trace_LandremanPaul2021_QA_lowres_vmecpp.npz").exists():
            s_pp = np.load(st_work / "trace_LandremanPaul2021_QA_lowres_vmecpp.npz")
            it_pp_s, fsq_pp_s = s_pp["it"], s_pp["fsq_total"]
        else:
            it_pp_s, fsq_pp_s = None, None
        mismatch_s = _trace_mismatch(it_vmec_s, fsq_vmec_s, it_jax_s, fsq_jax_s)
        audit_s = json.loads((workdir / "readme_fsq_trace_single_grid_wout_audit.json").read_text())
        wout_vmec_s = audit_s["LandremanPaul2021_QA_lowres"]["VMEC2000"]
        wout_jax_s = audit_s["LandremanPaul2021_QA_lowres"]["vmec_jax"]
        wout_pp_s = audit_s["LandremanPaul2021_QA_lowres"].get("vmecpp", {})
    else:
        it_vmec_a, fsq_vmec_a, _, wout_vmec_a = _collect_vmec2000_trace(
            axisym_input, ns=int(args.ns), niter=int(args.niter), ftol=float(args.ftol), workdir=axisym_work
        )
        it_jax_a, fsq_jax_a, _, wout_jax_a = _collect_vmec_jax_trace(
            axisym_input, ns=int(args.ns), niter=int(args.niter), ftol=float(args.ftol), workdir=axisym_work
        )
        if not skip_vmecpp:
            it_pp_a, fsq_pp_a, _, wout_pp_a = _collect_vmecpp_trace(
                axisym_input, ns=int(args.ns), niter=int(args.niter), ftol=float(args.ftol), workdir=axisym_work
            )
        else:
            it_pp_a, fsq_pp_a, wout_pp_a = None, None, {}
        mismatch_a = _trace_mismatch(it_vmec_a, fsq_vmec_a, it_jax_a, fsq_jax_a)

        it_vmec_s, fsq_vmec_s, _, wout_vmec_s = _collect_vmec2000_trace(
            stellarator_input, ns=int(args.ns), niter=int(args.niter), ftol=float(args.ftol), workdir=st_work
        )
        it_jax_s, fsq_jax_s, _, wout_jax_s = _collect_vmec_jax_trace(
            stellarator_input, ns=int(args.ns), niter=int(args.niter), ftol=float(args.ftol), workdir=st_work
        )
        if not skip_vmecpp:
            it_pp_s, fsq_pp_s, _, wout_pp_s = _collect_vmecpp_trace(
                stellarator_input, ns=int(args.ns), niter=int(args.niter), ftol=float(args.ftol), workdir=st_work
            )
        else:
            it_pp_s, fsq_pp_s, wout_pp_s = None, None, {}
        mismatch_s = _trace_mismatch(it_vmec_s, fsq_vmec_s, it_jax_s, fsq_jax_s)

    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    _plot_panel(
        axes[0],
        it_vmec=it_vmec_a,
        fsq_vmec=fsq_vmec_a,
        it_jax=it_jax_a,
        fsq_jax=fsq_jax_a,
        it_vmecpp=it_pp_a,
        fsq_vmecpp=fsq_pp_a,
        title="ITERModel fsq_total trace (single-grid)",
        jax_label=str(args.jax_label),
    )
    _plot_panel(
        axes[1],
        it_vmec=it_vmec_s,
        fsq_vmec=fsq_vmec_s,
        it_jax=it_jax_s,
        fsq_jax=fsq_jax_s,
        it_vmecpp=it_pp_s,
        fsq_vmecpp=fsq_pp_s,
        title="LandremanPaul2021_QA_lowres fsq_total trace (single-grid)",
        jax_label=str(args.jax_label),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    outpath = outdir / "readme_fsq_trace_single_grid.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    audit_out = workdir / "readme_fsq_trace_single_grid_wout_audit.json"
    audit_out.write_text(
        json.dumps(
            {
                "ITERModel": {"VMEC2000": wout_vmec_a, "vmec_jax": wout_jax_a, "vmecpp": wout_pp_a},
                "LandremanPaul2021_QA_lowres": {"VMEC2000": wout_vmec_s, "vmec_jax": wout_jax_s, "vmecpp": wout_pp_s},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    mismatch_out = workdir / "readme_fsq_trace_single_grid_mismatch.json"
    mismatch_out.write_text(
        json.dumps(
            {
                "ITERModel": {
                    "vmec_jax_vs_vmec2000": mismatch_a,
                    "vmecpp_vs_vmec2000": {} if (it_pp_a is None) else _trace_mismatch(it_vmec_a, fsq_vmec_a, it_pp_a, fsq_pp_a),
                },
                "LandremanPaul2021_QA_lowres": {
                    "vmec_jax_vs_vmec2000": mismatch_s,
                    "vmecpp_vs_vmec2000": {} if (it_pp_s is None) else _trace_mismatch(it_vmec_s, fsq_vmec_s, it_pp_s, fsq_pp_s),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {outpath}")
    print(f"Wrote {audit_out}")
    print(f"Wrote {mismatch_out}")
    print(f"Mismatch summary (relative): ITERModel max={mismatch_a.get('max_rel')} | QA max={mismatch_s.get('max_rel')}")


if __name__ == "__main__":
    main()
