"""Run the independent shifted point-oracle certificate on a native checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from polish_recovery_refine import _load_state
from vmex.core.strong_force import certify_strong_force


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = {
        "schema": "vmex.polish-recovery-certificate/2",
        "status": "running",
        "complete": False,
        "product_qualified": False,
        "method": "independent shifted point oracle with bounded point sweeps",
        "source": {
            "native_state": str(args.input_state),
            "native_state_sha256": _sha256(args.input_state),
            "generator": str(Path(__file__).resolve().relative_to(Path.cwd())),
            "generator_sha256": _sha256(Path(__file__)),
        },
        "model": {
            "declared_scope": "axisymmetric fixed-boundary prescribed-pressure/iota",
            "gamma": 0.0,
            "ncurr": 0,
            "lasym": False,
            "metadata_complete": False,
        },
        "certificate": {
            "complete": False,
            "force_target": 1.0e-5,
            "force_pass": False,
            "quadrature_pass": False,
            "geometry_pass": False,
            "profile_boundary_pass": False,
        },
        "stationarity": {"complete": False, "pass": False},
        "derivative": {"complete": False, "pass": False},
        "execution": {"phase": "load", "outcome": "running"},
        "work": {"total_process_seconds": 0.0},
        "limitations": [
            "This certificate qualifies the axisymmetric fixed-profile native state only.",
            "It does not qualify stationarity, 3-D, prescribed-current, LASYM, implicit derivatives, or the public polish driver.",
        ],
    }
    _write_json_atomic(args.output, payload)
    try:
        state = _load_state(args.input_state, prefix="accepted")
        spans = int(state.radial_basis.breakpoints.size - 1)
        order = int(state.radial_basis.degree + 3)
        coarse_order = max(int(state.radial_basis.degree + 1), order - 2)
        max_m = int(np.max(np.asarray(state.m), initial=0))
        max_n = int(np.max(np.abs(np.asarray(state.n)), initial=0))
        ntheta = max(8, 4 * (max_m + 1))
        nzeta = max(4, 4 * (max_n + 1))
        payload["certificate"].update(
            expected_fine_points=spans * order * ntheta * nzeta,
            expected_coarse_points=spans * coarse_order * ntheta * nzeta,
            completed_fine_points=0,
            completed_coarse_points=0,
            shifted_theta_fraction=0.5,
            shifted_zeta_fraction=0.375,
        )
        payload["execution"]["phase"] = "independent_certificate"
        _write_json_atomic(args.output, payload)
        report = certify_strong_force(state)
        force_pass = float(report.absolute_l2 / 5915447.712414409) <= 1.0e-5
        payload["certificate"].update(
            {
                "complete": True,
                "completed_fine_points": payload["certificate"]["expected_fine_points"],
                "completed_coarse_points": payload["certificate"]["expected_coarse_points"],
                "force_rms_N_per_m3": float(report.absolute_l2),
                "epsilon_B": float(report.absolute_l2 / 5915447.712414409),
                "normalized_l2": float(report.normalized_l2),
                "near_axis_force_rms_N_per_m3": float(report.near_axis_l2),
                "bulk_force_rms_N_per_m3": float(report.bulk_l2),
                "edge_force_rms_N_per_m3": float(report.edge_l2),
                "radial_refinement_difference": float(report.radial_refinement_difference),
                "angular_spectral_tail": float(report.angular_spectral_tail),
                "minimum_signed_jacobian": float(report.minimum_signed_jacobian),
                "nestedness_margin": float(report.nestedness_margin),
                "boundary_residual": float(report.boundary_residual),
                "lambda_gauge_residual": float(report.gauge_residual),
                "target_epsilon_B": 1.0e-5,
                "passes_force_target": force_pass,
                "force_pass": force_pass,
                "quadrature_pass": float(report.radial_refinement_difference) <= 1.0e-4,
                "geometry_pass": float(report.minimum_signed_jacobian) > 0.0,
                "profile_boundary_pass": (
                    float(report.boundary_residual) <= 1.0e-10 and float(report.gauge_residual) <= 1.0e-10
                ),
            }
        )
        payload["status"] = "measured-not-product-qualified"
        payload["complete"] = True
        payload["execution"].update(phase="complete", outcome="completed")
    except Exception as error:
        payload["status"] = "execution-failed"
        payload["execution"].update(outcome="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        payload["work"]["total_process_seconds"] = time.perf_counter() - started
        _write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
