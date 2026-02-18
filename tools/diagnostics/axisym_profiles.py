"""Generate axisymmetric iota/pressure profile figure for README/docs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vmec_jax.plotting import profiles_from_wout
from vmec_jax.wout import read_wout


def _resolve_wout(input_path: Path, wout_path: Path | None) -> Path:
    if wout_path is not None:
        return wout_path
    data_dir = input_path.parent
    stem = input_path.name.split("input.", 1)[-1]
    cand = [
        data_dir / f"wout_{stem}_reference.nc",
        data_dir / f"wout_{stem}.nc",
    ]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing wout file for {input_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "examples/data/input.shaped_tokamak_pressure"),
    )
    p.add_argument("--wout", type=str, default=None)
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "docs/_static/figures"),
    )
    p.add_argument("--tag", type=str, default="shaped_tokamak_pressure")
    args = p.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    wout_path = Path(args.wout).expanduser().resolve() if args.wout else None
    wout_path = _resolve_wout(input_path, wout_path)

    wout = read_wout(wout_path)
    prof = profiles_from_wout(wout)
    s = np.asarray(prof["s"])
    s_half = np.asarray(prof["s_half"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    iotaf = np.asarray(prof["iotaf"])
    iotas = np.asarray(prof["iotas"])
    if iotas.shape[0] != s_half.shape[0]:
        iotas = iotas[: s_half.shape[0]]
    axes[0].plot(s, iotaf, lw=2.0, label="iota (full)")
    axes[0].plot(s_half, iotas, lw=1.5, linestyle="--", label="iota (half)")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("iota")
    axes[0].set_title("Rotational transform")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)

    presf = np.asarray(prof["presf"])
    pres = np.asarray(prof["pres"])
    if pres.shape[0] != s_half.shape[0]:
        pres = pres[: s_half.shape[0]]
    axes[1].plot(s, presf, lw=2.0, label="pressure (full)")
    axes[1].plot(s_half, pres, lw=1.5, linestyle="--", label="pressure (half)")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("pressure")
    axes[1].set_title("Pressure profile")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"showcase_{args.tag}_profiles.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
