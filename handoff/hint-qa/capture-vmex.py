#!/usr/bin/env python3
"""Capture the exact beta0p5 exterior field, derivatives, and parameter VJPs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=True,
        help="clean VMEX Git checkout to import and evaluate",
    )
    parser.add_argument(
        "--output-prefix", type=Path, required=True,
        help="write PREFIX.npz and PREFIX.json; existing files are refused",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="verify source placement, Git identity, x64, and dependencies without computing",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _reserve(paths: tuple[Path, ...]) -> dict[Path, int]:
    descriptors: dict[Path, int] = {}
    try:
        for path in paths:
            descriptors[path] = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644,
            )
    except Exception:
        for path, descriptor in descriptors.items():
            os.close(descriptor)
            path.unlink(missing_ok=True)
        raise
    return descriptors


args = _arguments()
source = args.source_root.expanduser().resolve(strict=True)
prefix = args.output_prefix.expanduser().resolve()
if not prefix.parent.is_dir():
    raise FileNotFoundError(f"output directory does not exist: {prefix.parent}")
input_path = source / "examples/data/input.LandremanPaul2021_QA_beta0p5_bootstrap"
coil_path = source / "examples/data/ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json"
for required in (source / "pyproject.toml", input_path, coil_path):
    if not required.is_file():
        raise FileNotFoundError(f"required repository file is absent: {required.relative_to(source)}")
head = _git(source, "rev-parse", "HEAD")
tree = _git(source, "rev-parse", "HEAD^{tree}")
if _git(source, "status", "--porcelain=v1"):
    raise RuntimeError("source checkout must be clean")
sys.path.insert(0, str(source))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import vmex as vj  # noqa: E402
from essos.coils import Coils  # noqa: E402
from essos.fields import BiotSavart  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402

vmex_file = Path(vj.__file__).resolve()
vc_file = Path(vc.__file__).resolve()
if not vmex_file.is_relative_to(source) or not vc_file.is_relative_to(source):
    raise RuntimeError("vmex imports do not resolve inside --source-root")
if not bool(jax.config.jax_enable_x64):
    raise RuntimeError("x64 is required; set JAX_ENABLE_X64=1 before launch")
if not vc.have_virtual_casing_jax():
    raise RuntimeError("virtual_casing_jax is not importable")

packages = (
    "vmex", "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt",
    "solvax", "booz-xform-jax", "virtual-casing-jax", "essos",
    "numpy", "scipy", "netCDF4", "h5py",
)
versions = {}
for package in packages:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
preflight = {
    "source_git_head": head,
    "source_git_tree": tree,
    "source_clean": True,
    "input_repository_path": str(input_path.relative_to(source)),
    "input_sha256": _sha256(input_path),
    "coil_repository_path": str(coil_path.relative_to(source)),
    "coil_sha256": _sha256(coil_path),
    "driver_sha256": _sha256(Path(__file__).resolve()),
    "python_version": sys.version.split()[0],
    "package_versions": versions,
    "x64_enabled": True,
}
if args.preflight_only:
    print(json.dumps(preflight, indent=2, sort_keys=True))
    raise SystemExit(0)

npz_path = Path(f"{prefix}.npz")
json_path = Path(f"{prefix}.json")
reserved = _reserve((npz_path, json_path))
success = False
try:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {
        **preflight,
        "default_backend": jax.default_backend(),
        "default_devices": [str(device) for device in jax.devices()],
        "cpu_devices": [str(device) for device in jax.devices("cpu")],
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
        "timings_seconds": {},
        "arrays": {},
        "estimate_semantics": {
            "quantity": "order-0 virtual-casing plasma-field quadrature estimate",
            "normalization": "RMS magnitude of total B on the source surface",
            "dimensionless": True,
            "domain": "one exterior Cartesian target 0.25 m beyond the LCFS along the axis-edge ray at theta=phi=0",
            "exclusions": [
                "not an absolute error in tesla",
                "not the error of the combined coil-plus-plasma total field",
                "not a spatial-derivative accuracy estimate",
            ],
        },
    }
    try:
        metadata["gpu_devices"] = [str(device) for device in jax.devices("gpu")]
    except RuntimeError:
        metadata["gpu_devices"] = []

    def timed(name, function):
        start = time.monotonic()
        value = function()
        jax.block_until_ready(value)
        metadata["timings_seconds"][name] = time.monotonic() - start
        return value

    def record(name, value):
        device_method = getattr(value, "devices", None)
        devices = sorted(str(device) for device in device_method()) if callable(device_method) else []
        array = np.asarray(jax.device_get(value))
        arrays[name] = array
        metadata["arrays"][name] = {
            "shape": list(array.shape), "dtype": str(array.dtype), "devices": devices,
            "finite": bool(np.all(np.isfinite(array))),
            "min": float(np.min(array)), "max": float(np.max(array)),
            "l2": float(np.linalg.norm(array.ravel())),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        }
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"{name} is non-finite")
        return array

    total_start = time.monotonic()
    inp = vj.VmecInput.from_file(input_path)
    start = time.monotonic()
    problem = opt.VmecProblem.from_input(inp, max_mode=1, use_ess=True, progress=True)
    metadata["timings_seconds"]["problem"] = time.monotonic() - start
    start = time.monotonic()
    equilibrium = problem.equilibrium_from_x(problem.x0)
    metadata["timings_seconds"]["equilibrium"] = time.monotonic() - start
    result = equilibrium.result
    metadata["root"] = {
        "converged": bool(result.converged), "iterations": int(result.iterations),
        "fsqr": float(result.fsqr), "fsqz": float(result.fsqz), "fsql": float(result.fsql),
    }
    coils = Coils.from_json(str(coil_path))

    def coil_field_from_dofs(dofs):
        field = BiotSavart(coils.with_dofs(dofs))
        return lambda points: jax.vmap(field.B)(points)

    equilibrium.set_points_flux([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    axis, edge = equilibrium.field.get_points_cart()
    xyz = edge + 0.25 * (edge - axis) / jnp.linalg.norm(edge - axis)
    record("xyz", xyz)
    start = time.monotonic()
    outside = equilibrium.exterior_field(
        external_parameters=coils.dofs,
        external_field_from_parameters=coil_field_from_dofs,
        external_dof_names=coils.dof_names,
        digits=4,
        accuracy_check="raise",
    ).set_points_xyz(xyz[None])
    metadata["timings_seconds"]["exterior_constructor"] = time.monotonic() - start
    metadata["uses_virtual_casing"] = bool(outside.uses_virtual_casing)
    metadata["dof_names"] = list(outside.dof_names)
    B = timed("B", outside.B)
    absB = timed("absB", outside.absB)
    gradB = timed("gradB", outside.gradB)
    gradgradB = timed("gradgradB", outside.gradgradB)
    gradgradgradB = timed("gradgradgradB", outside.gradgradgradB)
    error_estimate = timed("B_error_estimate", outside.B_error_estimate)
    B_cotangent = jnp.ones_like(B)
    gradB_cotangent = jnp.ones_like(gradB)
    gradgradB_cotangent = jnp.ones_like(gradgradB)
    gradgradgradB_cotangent = jnp.ones_like(gradgradgradB)
    dBdx = timed("B_vjp", lambda: outside.B_vjp(B_cotangent))
    dgradBdx = timed("gradB_vjp", lambda: outside.gradB_vjp(gradB_cotangent))
    d2Bdx = timed("gradgradB_vjp", lambda: outside.gradgradB_vjp(gradgradB_cotangent))
    d3Bdx = timed("gradgradgradB_vjp", lambda: outside.gradgradgradB_vjp(gradgradgradB_cotangent))
    for name, value in (
        ("B", B), ("absB", absB), ("gradB", gradB),
        ("gradgradB", gradgradB), ("gradgradgradB", gradgradgradB),
        ("B_error_estimate", error_estimate),
        ("B_cotangent", B_cotangent), ("gradB_cotangent", gradB_cotangent),
        ("gradgradB_cotangent", gradgradB_cotangent),
        ("gradgradgradB_cotangent", gradgradgradB_cotangent),
        ("B_vjp", dBdx), ("gradB_vjp", dgradBdx),
        ("gradgradB_vjp", d2Bdx), ("gradgradgradB_vjp", d3Bdx),
    ):
        record(name, value)
    metadata["vjp_contract"] = {
        "parameter_order_key": "dof_names", "parameter_count": len(outside.dof_names),
        "B_vjp": {"output": "B", "cotangent": "B_cotangent"},
        "gradB_vjp": {"output": "gradB", "cotangent": "gradB_cotangent"},
        "gradgradB_vjp": {"output": "gradgradB", "cotangent": "gradgradB_cotangent"},
        "gradgradgradB_vjp": {"output": "gradgradgradB", "cotangent": "gradgradgradB_cotangent"},
    }
    metadata["timings_seconds"]["total"] = time.monotonic() - total_start
    metadata["all_arrays_finite"] = all(row["finite"] for row in metadata["arrays"].values())
    metadata["numerical_assertions"] = {
        "B_nonzero": bool(np.any(np.abs(arrays["B"]) > 0)),
        "absB_nonzero": bool(np.any(np.abs(arrays["absB"]) > 0)),
        "all_vjps_nonzero": bool(all(np.any(np.abs(arrays[name]) > 0) for name in
            ("B_vjp", "gradB_vjp", "gradgradB_vjp", "gradgradgradB_vjp"))),
        "order0_plasma_quadrature_estimate_within_requested_digits": bool(
            float(np.max(arrays["B_error_estimate"])) <= 1.0e-4),
    }
    if not all(metadata["numerical_assertions"].values()):
        raise AssertionError(metadata["numerical_assertions"])

    with os.fdopen(reserved.pop(npz_path), "wb") as output:
        np.savez_compressed(output, **arrays)
        output.flush()
        os.fsync(output.fileno())
    with os.fdopen(reserved.pop(json_path), "w") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    success = True
    print(json.dumps(metadata, indent=2, sort_keys=True))
finally:
    for path, descriptor in reserved.items():
        os.close(descriptor)
        path.unlink(missing_ok=True)
    if not success:
        npz_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
