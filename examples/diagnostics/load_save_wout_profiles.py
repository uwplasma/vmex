"""Load, save, and inspect a VMEC ``wout`` file.

The example is intentionally small and explicit:

1. Load a ``wout`` if it exists, otherwise run the bundled QH input to create
   one under ``examples/diagnostics/results/``.
2. Save a round-trip copy of the ``wout``.
3. Print common scalar parameters, the radial iota profile, and simple
   flux-surface averages of ``|B|``.

Generated files live in an ignored results directory, so this example does not
add large netCDF artifacts to the repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax.wout import read_wout, write_wout


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "examples" / "data" / "input.nfp4_QH_warm_start"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "wout_profiles"
WOUT_FILE = OUTPUT_DIR / "wout_nfp4_QH_warm_start.nc"
ROUNDTRIP_FILE = OUTPUT_DIR / "wout_nfp4_QH_warm_start_roundtrip.nc"

# Set to "cpu" or "gpu" to force a backend.  None uses the default JAX backend.
SOLVER_DEVICE: str | None = None


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_or_create_wout():
    """Load ``WOUT_FILE`` or create it by running vmec_jax on ``INPUT_FILE``."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not WOUT_FILE.exists():
        print(f"Creating {_display_path(WOUT_FILE)} from {_display_path(INPUT_FILE)}")
        run = vj.run_fixed_boundary(INPUT_FILE, solver_device=SOLVER_DEVICE, verbose=True)
        wout = vj.write_wout_from_fixed_boundary_run(WOUT_FILE, run, include_fsq=True)
    else:
        print(f"Loading existing {_display_path(WOUT_FILE)}")
        wout = read_wout(WOUT_FILE)
    return wout


def radial_bmag_profile(wout, *, ntheta: int = 32, nzeta: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``s``, surface-mean ``|B|``, and surface-max ``|B|`` profiles."""

    ns = int(wout.ns)
    s = np.linspace(0.0, 1.0, ns)
    b_mean = np.full(ns, np.nan)
    b_max = np.full(ns, np.nan)
    for js in range(ns):
        if js == 0:
            # VMEC's axis row is not a regular flux surface for |B| contours.
            continue
        _theta, _zeta, bmag = vj.vmecplot2_bmag_grid(wout, s_index=js, ntheta=ntheta, nzeta=nzeta)
        b_mean[js] = float(np.mean(bmag))
        b_max[js] = float(np.max(bmag))
    return s, b_mean, b_max


def print_wout_summary(wout) -> None:
    """Print a compact summary and radial profiles useful for new users."""

    s = np.linspace(0.0, 1.0, int(wout.ns))
    iota = np.asarray(wout.iotaf, dtype=float)
    _s_b, b_mean, b_max = radial_bmag_profile(wout)

    print("\nWout summary:")
    print(f"  ns/mpol/ntor/nfp: {wout.ns}/{wout.mpol}/{wout.ntor}/{wout.nfp}")
    print(f"  lasym:            {bool(wout.lasym)}")
    print(f"  aspect:           {float(wout.aspect):.6g}")
    print(f"  volume:           {float(wout.volume_p):.6g}")
    print(f"  betatotal:        {float(wout.betatotal):.6g}")
    print(f"  force residuals:  fsqr={float(wout.fsqr):.3e}, fsqz={float(wout.fsqz):.3e}, fsql={float(wout.fsql):.3e}")

    print("\nRadial profiles:")
    print("      s        iota_f       <|B|>       max|B|")
    print("  # s=0 is the magnetic-axis row; |B| surface averages start at the first finite-radius surface.")
    for js in range(int(wout.ns)):
        print(f"  {s[js]:7.4f}  {iota[js]:11.6e}  {b_mean[js]:11.6e}  {b_max[js]:11.6e}")


def main() -> None:
    wout = load_or_create_wout()
    write_wout(ROUNDTRIP_FILE, wout, overwrite=True)
    print(f"Saved round-trip copy to {_display_path(ROUNDTRIP_FILE)}")
    print_wout_summary(wout)


if __name__ == "__main__":
    main()
