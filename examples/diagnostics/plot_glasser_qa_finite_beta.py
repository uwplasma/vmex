"""Plot Mercier and Glasser resistive-interchange profiles for finite-beta QA.

This example runs the bundled finite-beta QA input file, writes a temporary
``wout`` under ``examples/diagnostics/results/``, and plots the ideal-MHD
Mercier diagnostic ``D_Merc`` together with the resistive-MHD Glasser
diagnostic ``D_R`` as functions of normalized toroidal flux ``s``.

Interpretation:
    * ideal Mercier stability: ``D_Merc >= 0``
    * Glasser resistive-interchange necessary condition: ``D_R <= 0``

The generated output directory is ignored by git, so running this script does
not add large solver artifacts to the repository.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

import vmec_jax as vj
from vmec_jax.wout import read_wout


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "examples" / "data" / "input.nfp2_QA_finite_beta"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "qa_finite_beta_glasser"
WOUT_FILE = OUTPUT_DIR / "wout_nfp2_QA_finite_beta.nc"
FIGURE_FILE = OUTPUT_DIR / "qa_finite_beta_dmerc_dr.png"

# Set to False to reuse an existing wout in OUTPUT_DIR while iterating on plots.
RUN_VMEC = True

# These defaults prioritize a faithful diagnostic example over aggressive speed.
# Set SOLVER_DEVICE = "gpu" to force a GPU backend when a JAX GPU install exists.
SOLVER_DEVICE: str | None = None
JIT_FORCES = True
WOUT_FAST_BCOVAR = False


def _display_path(path: Path) -> str:
    """Return a concise path for terminal output."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _import_matplotlib():
    """Import matplotlib with a writable cache for CI/HPC/sandboxed runs."""

    mpl_cache = Path(tempfile.gettempdir()) / "vmec_jax_mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def load_or_run_wout():
    """Return a finite-beta QA wout containing ``D_Merc`` and ``D_R`` profiles."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RUN_VMEC or not WOUT_FILE.exists():
        print(f"Running vmec_jax on {_display_path(INPUT_FILE)}")
        run = vj.run_fixed_boundary(
            INPUT_FILE,
            solver_device=SOLVER_DEVICE,
            jit_forces=JIT_FORCES,
            verbose=True,
        )
        wout = vj.write_wout_from_fixed_boundary_run(
            WOUT_FILE,
            run,
            include_fsq=True,
            fast_bcovar=WOUT_FAST_BCOVAR,
        )
    else:
        print(f"Reusing {_display_path(WOUT_FILE)}")
        wout = read_wout(WOUT_FILE)
    return wout


def glasser_profiles(wout) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract radial ``s``, ``D_Merc``, ``D_R``, and valid-shear mask arrays."""

    ns = int(wout.ns)
    s = np.linspace(0.0, 1.0, ns)
    dmerc = np.asarray(wout.DMerc, dtype=float)
    d_r = np.asarray(wout.D_R, dtype=float)
    valid = np.asarray(wout.glasser_shear_valid, dtype=bool)
    return s, dmerc, d_r, valid


def plot_glasser_profiles(wout, *, figure_path: Path = FIGURE_FILE) -> Path:
    """Write a publication-style profile plot for ``D_Merc`` and ``D_R``."""

    plt = _import_matplotlib()
    s, dmerc, d_r, valid = glasser_profiles(wout)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax.axhline(0.0, color="0.25", lw=0.9, ls="--", label="stability boundary")
    ax.plot(
        s,
        dmerc,
        marker="o",
        ms=3.2,
        lw=1.4,
        label=r"$D_{\mathrm{Merc}}$ (ideal stable if $\geq 0$)",
    )
    ax.plot(
        s,
        d_r,
        marker="s",
        ms=3.0,
        lw=1.4,
        label=r"$D_R$ (resistive stable if $\leq 0$)",
    )
    if valid.size == s.size and np.any(~valid):
        ax.scatter(
            s[~valid],
            d_r[~valid],
            marker="x",
            s=38,
            color="tab:red",
            label="near-zero-shear surface for $D_R$",
            zorder=5,
        )

    ax.set_xlabel(r"Normalized toroidal flux $s$")
    ax.set_ylabel("Stability diagnostic")
    ax.set_title("Finite-beta QA: Mercier and Glasser radial profiles")
    ax.set_yscale("symlog", linthresh=1.0e-4)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8.5)

    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def main() -> None:
    wout = load_or_run_wout()
    figure = plot_glasser_profiles(wout)

    s, dmerc, d_r, valid = glasser_profiles(wout)
    interior = slice(1, -1)
    print("\nFinite-beta QA stability diagnostics:")
    print(f"  wout:   {_display_path(WOUT_FILE)}")
    print(f"  figure: {_display_path(figure)}")
    print(f"  min D_Merc interior: {float(np.nanmin(dmerc[interior])):.6e}")
    print(f"  max D_R interior:    {float(np.nanmax(d_r[interior])):.6e}")
    print(f"  valid D_R surfaces:  {int(np.count_nonzero(valid))}/{valid.size}")
    print("\nSign convention:")
    print("  D_Merc >= 0 is ideal-MHD Mercier stable.")
    print("  D_R <= 0 satisfies the Glasser resistive-interchange necessary condition.")


if __name__ == "__main__":
    main()
