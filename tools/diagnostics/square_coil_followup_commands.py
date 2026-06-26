#!/usr/bin/env python
"""Print strict square-coil follow-up profile commands.

This helper does not run VMEC.  It emits serial, copy-pasteable commands for
the next VMEC2000 reference scans after a strict square-coil row stalls above
the requested force tolerance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.toroidal_stellarator_mirror_hybrid_square_coils_free_boundary import ExampleConfig
from vmec_jax.toroidal_hybrid import recommended_square_axis_nzeta


DEFAULT_VMEC2000_EXEC = "/home/rjorge/miniforge3/envs/qh-gpu/bin/xvmec"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", default="python3")
    p.add_argument("--profile-script", default="tools/diagnostics/profile_square_coil_free_boundary.py")
    p.add_argument("--outdir-root", type=Path, default=Path("results"))
    p.add_argument("--vmec2000-exec", default=DEFAULT_VMEC2000_EXEC)
    p.add_argument("--vmec2000-timeout", type=int, default=21600)
    p.add_argument("--delt-values", default="0.015,0.02,0.025")
    p.add_argument("--ns-array", default="9,13,17")
    p.add_argument("--niter-array", default="8000,16000,32000")
    p.add_argument("--ftol-array", default="1e-8,1e-10,1e-12")
    p.add_argument("--mpol", type=int, default=ExampleConfig().mpol)
    p.add_argument("--ntor", type=int, default=ExampleConfig().ntor)
    p.add_argument("--nzeta", type=int, default=None)
    p.add_argument("--ns", type=int, default=ExampleConfig().ns)
    p.add_argument("--mgrid-nr", type=int, default=88)
    p.add_argument("--mgrid-nz", type=int, default=64)
    p.add_argument("--mgrid-nphi", type=int, default=None)
    p.add_argument("--axis-kind", default=ExampleConfig().plasma_axis_kind)
    p.add_argument("--side-power", type=float, default=ExampleConfig().side_power)
    p.add_argument("--corner-power", type=float, default=ExampleConfig().corner_power)
    p.add_argument("--coil-segments", type=int, default=64)
    p.add_argument("--coil-chunk-size", type=int, default=512)
    return p


def _parse_float_list(raw: str) -> list[float]:
    values = [float(tok.strip()) for tok in str(raw).replace(";", ",").split(",") if tok.strip()]
    if not values:
        raise ValueError("expected at least one float value")
    return values


def _last_int(raw: str) -> int:
    values = [int(tok.strip()) for tok in str(raw).replace(";", ",").split(",") if tok.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values[-1]


def _float_label(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _list_label(raw: str) -> str:
    return str(raw).replace(";", "_").replace(",", "_").replace(" ", "")


def _iter_label(value: int) -> str:
    if int(value) % 1000 == 0:
        return f"{int(value) // 1000}k"
    return str(int(value))


def _outdir_for(args: argparse.Namespace, *, delt: float, nzeta: int, mgrid_nphi: int) -> Path:
    return Path(args.outdir_root) / (
        "square_coil_freeb_backend_profile_vmec2000"
        f"_ns{_list_label(args.ns_array)}"
        f"_mpol{int(args.mpol)}_ntor{int(args.ntor)}_nzeta{int(nzeta)}"
        f"_mgrid{int(args.mgrid_nr)}x{int(args.mgrid_nz)}x{int(mgrid_nphi)}"
        f"_delt{_float_label(float(delt))}"
        f"_niter{_iter_label(_last_int(args.niter_array))}"
        f"_{str(args.axis_kind)}"
    )


def _command_for(args: argparse.Namespace, *, delt: float) -> list[str]:
    nzeta = int(args.nzeta or max(64, recommended_square_axis_nzeta(int(args.ntor))))
    mgrid_nphi = int(args.mgrid_nphi or nzeta)
    max_iter = _last_int(args.niter_array)
    return [
        str(args.python),
        str(args.profile_script),
        "--outdir",
        str(_outdir_for(args, delt=delt, nzeta=nzeta, mgrid_nphi=mgrid_nphi)),
        "--beta-percent",
        "0",
        "--mpol",
        str(int(args.mpol)),
        "--ntor",
        str(int(args.ntor)),
        "--ns",
        str(int(args.ns)),
        "--nzeta",
        str(nzeta),
        "--ns-array",
        str(args.ns_array),
        "--niter-array",
        str(args.niter_array),
        "--ftol-array",
        str(args.ftol_array),
        "--max-iter",
        str(max_iter),
        "--ftol",
        f"{ExampleConfig().ftol:.0e}",
        "--phiedge",
        f"{ExampleConfig().phiedge:.16g}",
        "--delt",
        f"{float(delt):.16g}",
        "--activate-fsq",
        f"{ExampleConfig().free_boundary_activate_fsq:.0e}",
        "--nvacskip",
        str(int(ExampleConfig().nvacskip)),
        "--nstep",
        "1",
        "--axis-kind",
        str(args.axis_kind),
        "--side-power",
        f"{float(args.side_power):.16g}",
        "--corner-power",
        f"{float(args.corner_power):.16g}",
        "--n-coils-per-side",
        str(int(ExampleConfig().n_coils_per_side)),
        "--coil-segments",
        str(int(args.coil_segments)),
        "--coil-chunk-size",
        str(int(args.coil_chunk_size)),
        "--mgrid-nr",
        str(int(args.mgrid_nr)),
        "--mgrid-nz",
        str(int(args.mgrid_nz)),
        "--mgrid-nphi",
        str(mgrid_nphi),
        "--mgrid-padding-fraction",
        "1.2",
        "--mgrid-min-padding",
        "0.5",
        "--max-boundary-projection-error",
        f"{ExampleConfig().max_boundary_projection_error:.0e}",
        "--skip-direct",
        "--skip-mgrid",
        "--skip-provider-parity",
        "--run-vmec2000",
        "--vmec2000-exec",
        str(args.vmec2000_exec),
        "--vmec2000-timeout",
        str(int(args.vmec2000_timeout)),
        "--solver-mode",
        "parity",
    ]


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    """Return one VMEC2000 profile command per requested ``DELT`` value."""

    return [_command_for(args, delt=delt) for delt in _parse_float_list(args.delt_values)]


def shell_join(command: list[str]) -> str:
    """Return a shell-quoted one-line command."""

    return " ".join(shlex.quote(part) for part in command)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for command in build_commands(args):
        print(shell_join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
