#!/usr/bin/env python
"""Print strict square-coil follow-up profile commands.

This helper does not run VMEC.  It emits serial, copy-pasteable commands for
the next strict square-coil scans after a row stalls above the requested force
tolerance.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.toroidal_stellarator_mirror_hybrid_square_coils_free_boundary import ExampleConfig
from vmec_jax.toroidal_hybrid import recommended_square_axis_ntheta, recommended_square_axis_nzeta


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
    p.add_argument(
        "--ntheta",
        type=int,
        default=None,
        help=(
            "Explicit VMEC NTHETA passed to the profile script. Omit to let "
            "the profile choose its square-axis recommendation."
        ),
    )
    p.add_argument("--nzeta", type=int, default=None)
    p.add_argument("--ns", type=int, default=ExampleConfig().ns)
    p.add_argument(
        "--phiedge",
        type=float,
        default=ExampleConfig().phiedge,
        help="VMEC PHIEDGE used by the generated follow-up profile commands.",
    )
    p.add_argument("--mgrid-nr", type=int, default=88)
    p.add_argument("--mgrid-nz", type=int, default=64)
    p.add_argument("--mgrid-nphi", type=int, default=None)
    p.add_argument("--axis-kind", default=ExampleConfig().plasma_axis_kind)
    p.add_argument("--side-power", type=float, default=ExampleConfig().side_power)
    p.add_argument("--corner-power", type=float, default=ExampleConfig().corner_power)
    p.add_argument("--coil-segments", type=int, default=64)
    p.add_argument("--coil-chunk-size", type=int, default=512)
    p.add_argument(
        "--profile-kind",
        choices=(
            "vmec2000",
            "resolution-preflight",
            "provider-parity",
            "full-backend",
            "direct-gpu",
            "direct-gpu-jax-nestor",
            "direct-gpu-edge-polish",
            "direct-gpu-edge-jax-nestor-polish",
        ),
        default="vmec2000",
        help=(
            "vmec2000 keeps the existing generated-mgrid reference command. "
            "resolution-preflight emits a cheap projection/NZETA/mgrid compatibility check. "
            "provider-parity runs JAX direct and generated-mgrid with initial and accepted-LCFS parity. "
            "full-backend adds VMEC2000 to that comparison. "
            "direct-gpu emits direct-only cached-JIT commands for GPU speed probes. "
            "direct-gpu-jax-nestor adds the experimental JAX NESTOR operator to that probe. "
            "direct-gpu-edge-polish constrains LCFS edge motion to the square-axis spline-control "
            "basis, enables pressure mixing and hot restarts by default, and keeps the direct-coil "
            "cached-JIT path. direct-gpu-edge-jax-nestor-polish adds the JAX NESTOR operator too."
        ),
    )
    p.add_argument(
        "--accepted-provider-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include accepted-LCFS direct/mgrid parity for provider-parity and full-backend commands.",
    )
    p.add_argument(
        "--freeb-anderson-pressure",
        action="store_true",
        help="Add pressure Anderson mixing to JAX backend commands.",
    )
    p.add_argument(
        "--freeb-jax-nestor-operator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the experimental JAX NESTOR operator flag to JAX backend commands.",
    )
    p.add_argument(
        "--freeb-jax-nestor-jit-operator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Control JIT caching for the experimental JAX NESTOR operator.",
    )
    p.add_argument(
        "--freeb-include-edge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add an explicit edge-residual override to JAX backend commands.",
    )
    p.add_argument(
        "--freeb-dense-solve-mode",
        choices=("mode", "grid"),
        default=None,
        help="Add a dense free-boundary solve-mode override to JAX backend commands.",
    )
    p.add_argument(
        "--freeb-experimental-fouri-matrix",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add a Fourier-matrix override for NESTOR operator profiling.",
    )
    p.add_argument(
        "--freeb-add-analytic-bvec",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add an analytic-B-vector override for NESTOR operator profiling.",
    )
    p.add_argument(
        "--freeb-edge-control-projection",
        choices=("none", "square", "stellarator"),
        default=None,
        help=(
            "Add the vmec_jax reduced edge-control projection to JAX commands. "
            "Polish profile kinds default to 'square'."
        ),
    )
    p.add_argument(
        "--freeb-edge-control-rcond",
        type=float,
        default=1.0e-12,
        help="Pseudo-inverse cutoff for the reduced edge-control projection.",
    )
    p.add_argument(
        "--freeb-edge-control-update-mode",
        choices=("projected_delta", "coordinate"),
        default=None,
        help=(
            "How the solver applies reduced edge-control updates. Edge-polish "
            "profile kinds default to 'coordinate'; other edge-projected JAX "
            "commands default to 'projected_delta'."
        ),
    )
    p.add_argument(
        "--jax-hot-restart-count",
        type=int,
        default=None,
        help="Hot-restart count for JAX commands. Polish profile kinds default to 2.",
    )
    p.add_argument(
        "--jax-hot-restart-iters",
        type=int,
        default=None,
        help="Per-hot-restart iteration budget for JAX commands. Omit to use the final stage budget.",
    )
    p.add_argument(
        "--jax-hot-restart-policy",
        choices=("state", "freeb", "full"),
        default="freeb",
        help="State carried into JAX hot restarts.",
    )
    p.add_argument(
        "--jax-hot-restart-always",
        action="store_true",
        help="Run all requested hot-restart passes even if strict convergence is reached early.",
    )
    p.add_argument(
        "--jax-initial-restart-wout",
        type=Path,
        default=None,
        help="Optional existing final-grid wout_*.nc used to seed the JAX command.",
    )
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
    kind = str(args.profile_kind).replace("-", "_")
    edge = _edge_control_projection(args)
    edge_label = (
        ""
        if edge in {None, "none"}
        else f"_edge_{edge}_{_edge_control_update_mode(args)}"
    )
    ntheta_label = "" if args.ntheta is None else f"_ntheta{int(args.ntheta)}"
    default_phiedge = float(ExampleConfig().phiedge)
    phiedge_label = (
        ""
        if math.isclose(float(args.phiedge), default_phiedge, rel_tol=0.0, abs_tol=1.0e-15)
        else f"_phiedge{_float_label(float(args.phiedge))}"
    )
    return Path(args.outdir_root) / (
        f"square_coil_freeb_backend_profile_{kind}"
        f"_ns{_list_label(args.ns_array)}"
        f"_mpol{int(args.mpol)}_ntor{int(args.ntor)}_nzeta{int(nzeta)}"
        f"{ntheta_label}"
        f"_mgrid{int(args.mgrid_nr)}x{int(args.mgrid_nz)}x{int(mgrid_nphi)}"
        f"{phiedge_label}"
        f"_delt{_float_label(float(delt))}"
        f"_niter{_iter_label(_last_int(args.niter_array))}"
        f"_{str(args.axis_kind)}"
        f"{edge_label}"
    )


def _is_direct_jax_kind(kind: str) -> bool:
    return kind in {
        "direct-gpu",
        "direct-gpu-jax-nestor",
        "direct-gpu-edge-polish",
        "direct-gpu-edge-jax-nestor-polish",
    }


def _is_polish_kind(kind: str) -> bool:
    return kind in {"direct-gpu-edge-polish", "direct-gpu-edge-jax-nestor-polish"}


def _edge_control_projection(args: argparse.Namespace) -> str | None:
    requested = args.freeb_edge_control_projection
    if requested is not None:
        return str(requested)
    return "square" if _is_polish_kind(str(args.profile_kind)) else None


def _edge_control_update_mode(args: argparse.Namespace) -> str:
    requested = args.freeb_edge_control_update_mode
    if requested is not None:
        return str(requested)
    return "coordinate" if _is_polish_kind(str(args.profile_kind)) else "projected_delta"


def _hot_restart_count(args: argparse.Namespace) -> int:
    if args.jax_hot_restart_count is not None:
        return max(0, int(args.jax_hot_restart_count))
    return 2 if _is_polish_kind(str(args.profile_kind)) else 0


def _command_for(args: argparse.Namespace, *, delt: float) -> list[str]:
    nzeta = int(args.nzeta or max(64, recommended_square_axis_nzeta(int(args.ntor))))
    if args.ntheta is not None and int(args.ntheta) < int(recommended_square_axis_ntheta(int(args.mpol))):
        raise ValueError("--ntheta is below the square-axis recommendation for the requested --mpol")
    mgrid_nphi = int(args.mgrid_nphi or nzeta)
    max_iter = _last_int(args.niter_array)
    kind = str(args.profile_kind)
    edge_projection = _edge_control_projection(args)
    edge_update_mode = _edge_control_update_mode(args)
    ntheta_args = [] if args.ntheta is None else ["--ntheta", str(int(args.ntheta))]
    command = [
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
        *ntheta_args,
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
        f"{float(args.phiedge):.16g}",
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
        "--solver-mode",
        "parity",
    ]
    jax_kind = kind in {"provider-parity", "full-backend"} or _is_direct_jax_kind(kind)
    if jax_kind and edge_projection not in {None, "none"}:
        command.extend(
            [
                "--freeb-edge-control-projection",
                str(edge_projection),
                "--freeb-edge-control-rcond",
                f"{float(args.freeb_edge_control_rcond):.16g}",
                "--freeb-edge-control-update-mode",
                str(edge_update_mode),
            ]
        )
    if (bool(args.freeb_anderson_pressure) or _is_polish_kind(kind)) and jax_kind:
        command.append("--freeb-anderson-pressure")
    hot_restart_count = _hot_restart_count(args) if jax_kind else 0
    if hot_restart_count > 0:
        command.extend(["--jax-hot-restart-count", str(int(hot_restart_count))])
        command.extend(
            [
                "--jax-hot-restart-iters",
                str(int(args.jax_hot_restart_iters or max_iter)),
                "--jax-hot-restart-policy",
                str(args.jax_hot_restart_policy),
            ]
        )
        if bool(args.jax_hot_restart_always):
            command.append("--jax-hot-restart-always")
    if jax_kind and args.jax_initial_restart_wout is not None:
        command.extend(["--jax-initial-restart-wout", str(args.jax_initial_restart_wout)])
    use_jax_nestor = bool(args.freeb_jax_nestor_operator) or kind in {
        "direct-gpu-jax-nestor",
        "direct-gpu-edge-jax-nestor-polish",
    }
    if jax_kind and use_jax_nestor:
        command.append("--freeb-jax-nestor-operator")
        if bool(args.freeb_jax_nestor_jit_operator):
            command.append("--freeb-jax-nestor-jit-operator")
        else:
            command.append("--no-freeb-jax-nestor-jit-operator")
    if jax_kind and args.freeb_include_edge is not None:
        command.append("--freeb-include-edge" if bool(args.freeb_include_edge) else "--no-freeb-include-edge")
    if jax_kind and args.freeb_dense_solve_mode is not None:
        command.extend(["--freeb-dense-solve-mode", str(args.freeb_dense_solve_mode)])
    if jax_kind and args.freeb_experimental_fouri_matrix is not None:
        command.append(
            "--freeb-experimental-fouri-matrix"
            if bool(args.freeb_experimental_fouri_matrix)
            else "--no-freeb-experimental-fouri-matrix"
        )
    if jax_kind and args.freeb_add_analytic_bvec is not None:
        command.append(
            "--freeb-add-analytic-bvec" if bool(args.freeb_add_analytic_bvec) else "--no-freeb-add-analytic-bvec"
        )
    if kind == "resolution-preflight":
        command.append("--resolution-diagnostics-only")
        return command
    if _is_direct_jax_kind(kind):
        command.extend(
            [
                "--skip-mgrid",
                "--skip-provider-parity",
                "--coil-chunk-size",
                "0",
                "--jit-forces",
                "--jit-direct-sampler",
                "--verbose-solver",
                "--return-best-scored-state",
            ]
        )
        return command
    command.extend(["--coil-chunk-size", str(int(args.coil_chunk_size))])
    if kind in {"provider-parity", "full-backend"} and bool(args.accepted_provider_parity):
        command.append("--accepted-provider-parity")
    if kind == "provider-parity":
        command.append("--return-best-scored-state")
        return command
    if kind == "full-backend":
        command.extend(
            [
                "--run-vmec2000",
                "--vmec2000-exec",
                str(args.vmec2000_exec),
                "--vmec2000-timeout",
                str(int(args.vmec2000_timeout)),
                "--return-best-scored-state",
            ]
        )
        return command
    command.extend(
        [
            "--skip-direct",
            "--skip-mgrid",
            "--skip-provider-parity",
            "--run-vmec2000",
            "--vmec2000-exec",
            str(args.vmec2000_exec),
            "--vmec2000-timeout",
            str(int(args.vmec2000_timeout)),
        ]
    )
    return command


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    """Return one VMEC2000 profile command per requested ``DELT`` value."""

    edge_projection = _edge_control_projection(args)
    if edge_projection not in {None, "none"} and str(args.axis_kind) != "control_spline":
        raise ValueError("--freeb-edge-control-projection requires --axis-kind control_spline")
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
