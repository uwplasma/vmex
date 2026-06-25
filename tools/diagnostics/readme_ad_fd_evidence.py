"""Render README-ready AD-vs-FD derivative evidence.

The rows here are intentionally small and deterministic.  They document the
derivative contract for public differentiable building blocks and branch-local
free-boundary evidence without claiming arbitrary adaptive branch derivatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
STRICT_DETERMINISTIC_TOL = 1.0e-9
REQUIRED_BRANCH_LOCAL_SCALARS = ("aspect", "qs_total", "mean_iota", "lcfs_boundary_moment")


@dataclass(frozen=True)
class EvidenceRow:
    scalar: str
    scope: str
    method: str
    ad_slope: float
    fd_slope: float
    abs_error: float
    rel_error: float
    tolerance: float
    passed: bool
    note: str


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _relative_error(ad: float, fd: float) -> float:
    scale = max(abs(ad), abs(fd), 1.0)
    return abs(ad - fd) / scale


def _ad_fd_row(
    *,
    scalar: str,
    scope: str,
    method: str,
    objective: Callable[[Any], Any],
    alpha0: float = 0.0,
    eps: float = 1.0e-5,
    tolerance: float = 1.0e-6,
    note: str = "",
) -> EvidenceRow:
    import jax
    import jax.numpy as jnp

    alpha = jnp.asarray(float(alpha0), dtype=jnp.float64)
    step = jnp.asarray(float(eps), dtype=jnp.float64)
    ad = float(np.asarray(jax.grad(objective)(alpha)))
    fd = float(np.asarray((objective(alpha + step) - objective(alpha - step)) / (2.0 * step)))
    abs_error = abs(ad - fd)
    rel_error = _relative_error(ad, fd)
    return EvidenceRow(
        scalar=scalar,
        scope=scope,
        method=method,
        ad_slope=ad,
        fd_slope=fd,
        abs_error=abs_error,
        rel_error=rel_error,
        tolerance=float(tolerance),
        passed=bool(math.isfinite(ad) and math.isfinite(fd) and rel_error <= float(tolerance)),
        note=note,
    )


def _aspect_row() -> EvidenceRow:
    import jax.numpy as jnp

    from vmec_jax.modes import ModeTable
    from vmec_jax.state import StateLayout, VMECState
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    modes = ModeTable(m=np.asarray([0, 1], dtype=int), n=np.asarray([0, 0], dtype=int))
    layout = StateLayout(ns=3, K=2, lasym=False)
    static = SimpleNamespace(cfg=SimpleNamespace(ntheta=10, nzeta=1, nfp=1, mpol=1, ntor=0, lasym=False), modes=modes)
    radial = jnp.linspace(0.0, 1.0, 3, dtype=jnp.float64)[:, None]

    def objective(alpha):
        rcos = jnp.concatenate([2.0 + 0.0 * radial, (0.15 + 0.03 * alpha) * radial], axis=1)
        zsin = jnp.concatenate([0.0 * radial, (0.25 - 0.02 * alpha) * radial], axis=1)
        zeros = jnp.zeros((3, 2), dtype=jnp.float64)
        state = VMECState(layout=layout, Rcos=rcos, Rsin=zeros, Zcos=zeros, Zsin=zsin, Lcos=zeros, Lsin=zeros)
        return equilibrium_aspect_ratio_from_state(state=state, static=static)

    return _ad_fd_row(
        scalar="aspect ratio",
        scope="fixed-boundary geometry",
        method="unrolled JAX AD",
        objective=objective,
        alpha0=0.2,
        tolerance=STRICT_DETERMINISTIC_TOL,
        note="equilibrium_aspect_ratio_from_state on a tiny stellarator-symmetric state",
    )


def _iota_profile_row() -> EvidenceRow:
    import jax.numpy as jnp

    from vmec_jax.profiles import ProfilePolynomial

    s = jnp.linspace(0.0, 1.0, 9, dtype=jnp.float64)

    def objective(alpha):
        profile = ProfilePolynomial(jnp.asarray([0.36 + 0.02 * alpha, 0.08 - 0.01 * alpha, 0.03], dtype=jnp.float64))
        return jnp.mean(profile.f(s))

    return _ad_fd_row(
        scalar="iota profile",
        scope="fixed-boundary radial profile",
        method="JAX profile AD",
        objective=objective,
        alpha0=0.4,
        tolerance=STRICT_DETERMINISTIC_TOL,
        note="ProfilePolynomial iota-like radial profile",
    )


def _qs_row() -> EvidenceRow:
    import jax.numpy as jnp

    from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_wout

    constant_mode = jnp.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)

    def objective(alpha):
        bmnc = constant_mode.at[:, 1].set(0.04 + alpha * jnp.asarray([0.0, 0.03, 0.06], dtype=jnp.float64))
        wout_like = SimpleNamespace(
            nfp=2,
            lasym=False,
            iotas=jnp.asarray([0.0, 0.4, 0.5], dtype=jnp.float64),
            buco=jnp.asarray([0.0, 0.2, 0.25], dtype=jnp.float64),
            bvco=jnp.asarray([0.0, 1.0, 1.1], dtype=jnp.float64),
            gmnc=constant_mode,
            bmnc=bmnc,
            bsubumnc=0.2 * constant_mode,
            bsubvmnc=0.3 * constant_mode,
            bsupumnc=0.4 * constant_mode,
            bsupvmnc=0.5 * constant_mode,
            xm_nyq=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
            xn_nyq=jnp.asarray([0.0, 2.0], dtype=jnp.float64),
            phi=jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64),
        )
        return quasisymmetry_ratio_residual_from_wout(
            wout_like,
            surfaces=[0.5],
            helicity_m=1,
            helicity_n=-1,
            ntheta=9,
            nphi=10,
        )["total"]

    return _ad_fd_row(
        scalar="QS residual",
        scope="fixed-boundary Boozer/VMEC diagnostic",
        method="JAX residual AD",
        objective=objective,
        alpha0=0.08,
        tolerance=STRICT_DETERMINISTIC_TOL,
        note="quasisymmetry_ratio_residual_from_wout",
    )


def _qi_row() -> EvidenceRow:
    import jax.numpy as jnp

    from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_boozer_output

    xm = jnp.asarray([0.0, 1.0, 1.0, 2.0], dtype=jnp.float64)
    xn = jnp.asarray([0.0, 0.0, 2.0, 2.0], dtype=jnp.float64)

    def objective(alpha):
        coeffs = jnp.asarray([[1.0, 0.08 + 0.02 * alpha, 0.12 - 0.01 * alpha, 0.03 + 0.015 * alpha]], dtype=jnp.float64)
        booz = {
            "bmnc_b": coeffs,
            "ixm_b": xm,
            "ixn_b": xn,
            "iota_b": jnp.asarray([0.43 + 0.01 * alpha], dtype=jnp.float64),
            "nfp_b": jnp.asarray(2),
        }
        return quasi_isodynamic_residual_from_boozer_output(
            booz,
            nphi=25,
            nalpha=9,
            n_bounce=9,
            branch_width_weight=0.25,
            shuffle_profile_weight=0.25,
            profile_weight=0.05,
        )["total"]

    return _ad_fd_row(
        scalar="smooth QI residual",
        scope="Boozer-space omnigenity diagnostic",
        method="smooth JAX AD",
        objective=objective,
        alpha0=0.12,
        tolerance=STRICT_DETERMINISTIC_TOL,
        note="quasi_isodynamic_residual_from_boozer_output",
    )


def _mercier_profile_row(field: str) -> EvidenceRow:
    import jax.numpy as jnp

    import vmec_jax as vj

    base = {
        "s": jnp.linspace(0.0, 1.0, 6, dtype=jnp.float64),
        "phips": jnp.ones(6, dtype=jnp.float64),
        "iotas": jnp.asarray([0.0, 0.18, 0.28, 0.43, 0.61, 0.72], dtype=jnp.float64),
        "vp": jnp.asarray([0.0, 0.90, 1.05, 1.18, 1.32, 1.50], dtype=jnp.float64),
        "pres": jnp.asarray([0.0, 0.050, 0.041, 0.030, 0.018, 0.0], dtype=jnp.float64),
        "torcur": jnp.asarray([0.0, 0.030, 0.055, 0.082, 0.110, 0.135], dtype=jnp.float64),
        "tpp": jnp.asarray([0.0, 1.35, 1.42, 1.55, 1.70, 0.0], dtype=jnp.float64),
        "tbb": jnp.asarray([0.0, 0.82, 0.91, 1.03, 1.15, 0.0], dtype=jnp.float64),
        "tjb": jnp.asarray([0.0, 0.18, 0.16, 0.14, 0.12, 0.0], dtype=jnp.float64),
        "tjj": jnp.asarray([0.0, 0.035, 0.045, 0.055, 0.068, 0.0], dtype=jnp.float64),
        "jdotb": jnp.asarray([0.0, 0.40, 0.36, 0.30, 0.24, 0.0], dtype=jnp.float64),
        "bdotb": jnp.asarray([0.0, 1.90, 1.85, 1.70, 1.55, 0.0], dtype=jnp.float64),
    }
    iota_direction = jnp.asarray([0.0, 0.02, -0.01, 0.03, -0.02, 0.0], dtype=jnp.float64)
    pressure_direction = jnp.asarray([0.0, -0.010, 0.015, -0.005, 0.012, 0.0], dtype=jnp.float64)

    def objective(alpha):
        trial = dict(base)
        trial["iotas"] = base["iotas"] + alpha * iota_direction
        trial["pres"] = base["pres"] + alpha * pressure_direction
        terms = vj.mercier_terms_from_profile_integrals(**trial, shear_epsilon=1.0e-3)
        return jnp.sum(jnp.asarray(terms[field], dtype=jnp.float64)[1:-1])

    return _ad_fd_row(
        scalar=field,
        scope="finite-beta profile stability",
        method="JAX profile-integral AD",
        objective=objective,
        alpha0=0.10,
        tolerance=STRICT_DETERMINISTIC_TOL,
        note="mercier_terms_from_profile_integrals",
    )


def _branch_local_rows(report_path: Path | None) -> list[EvidenceRow]:
    if report_path is None or not report_path.exists():
        return []
    payload = json.loads(report_path.read_text())
    rows: list[EvidenceRow] = []
    gate = (
        payload.get("physical_scalar_gate")
        or payload.get("physical_gate")
        or payload.get("branch_local_vector_gate", {}).get("physical_scalar_gate")
        or payload.get("branch_local_vector_gate", {}).get("physical_gate")
        or {}
    )
    scalars = gate.get("scalars", {}) if isinstance(gate, dict) else {}
    if not isinstance(gate, dict) or not scalars:
        raise ValueError(f"{report_path} does not contain a branch-local physical scalar gate")
    if not bool(gate.get("passed", False)):
        raise ValueError(f"{report_path} branch-local physical scalar gate did not pass")
    missing = [key for key in REQUIRED_BRANCH_LOCAL_SCALARS if key not in scalars]
    if missing:
        raise ValueError(
            f"{report_path} is missing required branch-local scalar(s): {', '.join(missing)}"
        )
    for key, scalar in scalars.items():
        ad = float(scalar.get("exact_directional", np.nan))
        fd = float(scalar.get("complete_fd_directional", np.nan))
        rel = float(scalar.get("rel_error", _relative_error(ad, fd)))
        rows.append(
            EvidenceRow(
                scalar=f"free-boundary {key}",
                scope="direct-coil free-boundary",
                method="branch-local replay JVP",
                ad_slope=ad,
                fd_slope=fd,
                abs_error=float(scalar.get("abs_error", abs(ad - fd))),
                rel_error=rel,
                tolerance=STRICT_DETERMINISTIC_TOL,
                passed=bool(scalar.get("passed", False)) and rel <= STRICT_DETERMINISTIC_TOL,
                note="same-branch/fingerprint-gated; not arbitrary adaptive branch differentiation",
            )
        )
    return rows


def _write_csv(rows: list[EvidenceRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(rows[0]).keys())
            if rows
            else list(asdict(EvidenceRow("", "", "", 0, 0, 0, 0, 0, False, "")).keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_json(rows: list[EvidenceRow], path: Path, *, branch_report: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "branch_local_report": None if branch_report is None else str(branch_report),
            "contract": "AD-vs-central-FD evidence; free-boundary rows are same-branch/fingerprint-gated only",
        },
        "records": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_figure(rows: list[EvidenceRow], path: Path) -> None:
    plt = _pyplot()
    labels = [row.scalar for row in rows]
    rel = np.asarray([max(row.rel_error, 1.0e-16) for row in rows], dtype=float)
    tol = np.asarray([row.tolerance for row in rows], dtype=float)
    colors = ["#2ca02c" if row.passed else "#d62728" for row in rows]
    y = np.arange(len(rows), dtype=float)
    fig, ax = plt.subplots(figsize=(11.5, max(4.5, 0.42 * len(rows) + 1.5)), constrained_layout=True)
    ax.barh(y, rel, color=colors, height=0.62, label="relative error")
    ax.scatter(tol, y, color="black", marker="|", s=140, label="tolerance")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("relative AD-vs-FD slope error")
    ax.set_title("Differentiation evidence: automatic differentiation vs central finite differences")
    ax.grid(axis="x", which="both", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-local-report", type=Path, default=None)
    parser.add_argument("--figure-out", type=Path, default=REPO_ROOT / "docs/_static/figures/readme_ad_fd_evidence.png")
    parser.add_argument("--csv-out", type=Path, default=REPO_ROOT / "docs/_static/figures/readme_ad_fd_evidence.csv")
    parser.add_argument("--json-out", type=Path, default=REPO_ROOT / "docs/_static/figures/readme_ad_fd_evidence.json")
    args = parser.parse_args()

    from vmec_jax._compat import enable_x64

    enable_x64(True)
    rows = [
        _aspect_row(),
        _iota_profile_row(),
        _qs_row(),
        _qi_row(),
        _mercier_profile_row("DMerc"),
        _mercier_profile_row("D_R"),
    ]
    rows.extend(_branch_local_rows(args.branch_local_report))
    _write_csv(rows, args.csv_out)
    _write_json(rows, args.json_out, branch_report=args.branch_local_report)
    _write_figure(rows, args.figure_out)
    failed = [row.scalar for row in rows if not row.passed]
    print(f"rows={len(rows)} figure={args.figure_out}")
    if failed:
        raise SystemExit(f"AD-vs-FD evidence failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
