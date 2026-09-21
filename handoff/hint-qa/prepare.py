#!/usr/bin/env python3
# ruff: noqa: D103
"""Prepare finite-beta QA inputs with ESSOS and the native HINT preprocessors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import time

import f90nml
import jax

import jax.numpy as jnp
import numpy as np
from essos.coils import Coils
from essos.fields import BiotSavart
from netCDF4 import Dataset
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient
import vmex
from vmex import VmecInput
from vmex.core.wout import read_wout

jax.config.update("jax_enable_x64", True)


MU0 = 4.0e-7 * np.pi
CASES = ("beta0p5", "beta2p5")
COIL_SEGMENTS = 75
SAFE_BUILD_FLAGS = {
    "-cpp", "-DNETCDF", "-ffree-line-length-none", "-fopenmp", "-O0", "-O1", "-O2", "-O3", "-g", "-Wall", "-Wextra",
    "-fcheck=all", "-fbacktrace", "-ffpe-trap=invalid,zero,overflow",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode())
    digest.update(json.dumps(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def verify_member(root: Path, manifest: dict[str, object], recorded: str) -> tuple[Path, dict[str, object]]:
    if recorded not in manifest:
        raise ValueError(f"SHA256.json does not register {recorded}")
    path = (root / recorded).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"manifest member is absent: {recorded}")
    expected = manifest[recorded]
    expected_hash = expected["sha256"] if isinstance(expected, dict) else expected
    expected_bytes = expected.get("bytes") if isinstance(expected, dict) else None
    actual = sha256(path)
    if actual != expected_hash or (expected_bytes is not None and path.stat().st_size != expected_bytes):
        raise ValueError(f"input checksum mismatch: {recorded}")
    public_name = str(Path(*(part for part in Path(recorded).parts if part != "..")))
    return path, {"name": public_name, "bytes": path.stat().st_size, "sha256": actual}


def current_profile(wout) -> np.ndarray:
    """Reproduce the published lambda(s)=<J.B>/<B^2> table."""
    theta, phi = np.meshgrid(
        np.arange(64) * 2 * np.pi / 64,
        np.arange(64) * 2 * np.pi / (int(wout.nfp) * 64),
        indexing="ij",
    )
    angle = theta.ravel()[:, None] * np.asarray(wout.xm_nyq) - phi.ravel()[:, None] * np.asarray(wout.xn_nyq)
    cosine = np.cos(angle)
    bmag = np.asarray(wout.bmnc) @ cosine.T
    jacobian = np.abs(np.asarray(wout.gmnc) @ cosine.T)
    b2 = np.sum(jacobian[1:] * bmag[1:] ** 2, axis=1) / np.sum(jacobian[1:], axis=1)
    half_s = (np.arange(1, int(wout.ns)) - 0.5) / (int(wout.ns) - 1)
    full_s = np.linspace(0.0, 1.0, int(wout.ns))
    profile = np.interp(half_s, full_s, np.asarray(wout.jdotb)) / b2
    profile = np.interp(full_s, half_s, profile)
    profile /= np.max(np.abs(profile))
    return np.column_stack((full_s, profile))


def validate_case_contract(wout, source_input: Path, deck: Path, table: np.ndarray) -> dict[str, object]:
    vmec_input = VmecInput.from_file(source_input)
    parsed = f90nml.read(deck)
    n2 = parsed["nlinp2"]
    expected_beta = 2 * MU0 * float(vmec_input.am[0]) / abs(float(wout.b0)) ** 2
    comparisons = {
        "beta": (float(n2["beta"]), expected_beta),
        "bt0_T": (float(n2["bt0"]), abs(float(wout.b0))),
        "axis_pressure_Pa": (float(n2["am0"][0]), float(vmec_input.am[0])),
        "source_current_A": (float(n2["inet0"]), float(wout.ctor)),
    }
    for name, (actual, expected) in comparisons.items():
        if not np.isclose(actual, expected, rtol=2e-14, atol=1e-12):
            raise ValueError(f"deck {name} violates source contract: {actual} != {expected}")
    if not np.allclose(np.asarray(n2["am0"], dtype=float), np.asarray(vmec_input.am), rtol=0, atol=5e-10):
        raise ValueError("deck pressure polynomial differs from the VMEX source deck")
    if int(n2["nprofj"]) != len(table):
        raise ValueError("deck nprofj differs from the derived current table")
    if not np.array_equal(np.asarray(n2["torfj"], dtype=float), table[:, 0]):
        raise ValueError("deck torfj differs from the published current table")
    if not np.array_equal(np.asarray(n2["profj"], dtype=float), table[:, 1]):
        raise ValueError("deck profj differs from the published current table")
    return {
        "pressure_conversion": "beta=2*mu0*p_axis/B_reference^2",
        "axis_pressure_Pa": float(vmec_input.am[0]),
        "hint_axis_beta": expected_beta,
        "source_current_A": float(wout.ctor),
        "current_definition": "normalized lambda(s)=<J.B>/<B^2>, 64x64 theta/phi quadrature",
        "current_table_sha256": array_sha256(table),
        "deck_file_sha256": sha256(deck),
    }


def boundary(wout, nphi: int = 128, ntheta: int = 256) -> tuple[np.ndarray, np.ndarray]:
    theta = np.arange(ntheta) * 2 * np.pi / ntheta
    phi = np.arange(nphi) * 2 * np.pi / (int(wout.nfp) * nphi)
    phase = theta[None, :, None] * np.asarray(wout.xm)[None, None, :] - phi[:, None, None] * np.asarray(wout.xn)[None, None, :]
    radius = np.cos(phase) @ np.asarray(wout.rmnc[-1])
    height = np.sin(phase) @ np.asarray(wout.zmns[-1])
    return np.stack((radius, height), axis=-1), phi


def write_wall(path: Path, edge: np.ndarray, nfp: int, buffer_m: float) -> np.ndarray:
    rings = []
    count = edge.shape[1]
    for section in edge:
        polygon = orient(Polygon(section).buffer(buffer_m, quad_segs=32), sign=1)
        if not polygon.is_valid or polygon.geom_type != "Polygon":
            raise ValueError("wall buffer did not produce one valid polygon")
        line = polygon.exterior
        start = line.project(Point(section[0] + np.array((buffer_m, 0.0))))
        ring = np.asarray([line.interpolate((start + u * line.length / count) % line.length).coords[0] for u in range(count)])
        if not Polygon(ring).covers(Polygon(section)):
            raise ValueError("resampled wall does not contain the source boundary")
        rings.append(ring)
    wall = np.asarray(rings)
    with path.open("w") as stream:
        stream.write(f"{nfp}, {count + 1}, {len(wall)},\n")
        for ring in wall:
            np.savetxt(stream, np.vstack((ring, ring[0])), fmt="%.16e", delimiter=", ")
    return wall


def write_vacuum(path: Path, coil_path: Path, box: list[float], shape: tuple[int, int, int], nfp: int) -> dict[str, object]:
    nr, nz, ntor = shape
    coils = Coils.from_json(str(coil_path))
    coils.n_segments = COIL_SEGMENTS
    field = jax.jit(jax.vmap(BiotSavart(coils).B))
    radius = np.linspace(box[0], box[1], nr)
    height = np.linspace(box[2], box[3], nz)
    phi = np.arange(ntor) * 2 * np.pi / (nfp * ntor)
    pp, zz, rr = np.meshgrid(phi, height, radius, indexing="ij")
    xyz = np.stack((rr * np.cos(pp), rr * np.sin(pp), zz), axis=-1)
    flat = xyz.reshape(-1, 3)
    cartesian = np.empty_like(flat)
    for start in range(0, len(flat), 512):
        cartesian[start : start + 512] = np.asarray(field(jnp.asarray(flat[start : start + 512])))
    cartesian = cartesian.reshape(xyz.shape)
    cylindrical = np.stack(
        (
            cartesian[..., 0] * np.cos(pp) + cartesian[..., 1] * np.sin(pp),
            -cartesian[..., 0] * np.sin(pp) + cartesian[..., 1] * np.cos(pp),
            cartesian[..., 2],
        ),
        axis=-1,
    )
    if not np.isfinite(cylindrical).all():
        raise ValueError("nonfinite ESSOS vacuum field")
    with Dataset(path, "w") as dataset:
        for name, values in (("R", radius), ("Z", height), ("phi", phi)):
            dataset.createDimension(name, len(values))
            dataset.createVariable(name, "f8", (name,))[:] = values
        for name, value in (("mtor", nfp), ("rminb", radius[0]), ("rmaxb", radius[-1]), ("zminb", height[0]), ("zmaxb", height[-1])):
            dataset.createVariable(name, "i4" if name == "mtor" else "f8").assignValue(value)
        for index, name in enumerate(("Bvac_R", "Bvac_phi", "Bvac_Z")):
            dataset.createVariable(name, "f8", ("phi", "Z", "R"))[:] = cylindrical[..., index]
    coil_arrays = {"curves": np.asarray(coils.curves.curves), "currents": np.asarray(coils.currents), "gamma": np.asarray(coils.gamma)}
    return {
        "n_segments": int(coils.n_segments),
        "arrays": {name: {"shape": list(values.shape), "dtype": str(values.dtype), "array_sha256": array_sha256(values)} for name, values in coil_arrays.items()},
    }


def render_deck(source: Path, destination: Path, shape: tuple[int, int, int], box: list[float], wout, vmec_input, table: np.ndarray) -> None:
    nfp = int(wout.nfp)
    patch = {
        "nlinp2": {
            "beta": 2 * MU0 * float(vmec_input.am[0]) / abs(float(wout.b0)) ** 2,
            "bt0": abs(float(wout.b0)),
            "am0": np.asarray(vmec_input.am).tolist(),
            "inet0": float(wout.ctor),
            "inet1": 0.0,
            "jcuts": 1.0,
            "nprofj": len(table),
            "torfj": table[:, 0].tolist(),
            "profj": table[:, 1].tolist(),
        },
        "nlinp3": {
            "nr": shape[0],
            "nz": shape[1],
            "ntor": shape[2],
            "mtor": nfp,
            "rminb": box[0],
            "rmaxb": box[1],
            "zminb": box[2],
            "zmaxb": box[3],
        },
        "nlinp4": {"rstart": float(np.sum(np.asarray(wout.rmnc[0]))), "zstart": 0.0},
    }
    current = f90nml.read(source)
    wanted = f90nml.Namelist(patch)
    matches = True
    for group, values in wanted.items():
        for name, expected in values.items():
            actual = current[group][name]
            if isinstance(expected, list):
                matches &= np.array_equal(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float))
            elif isinstance(expected, float):
                matches &= float(actual) == expected
            else:
                matches &= actual == expected
    if matches:
        shutil.copy2(source, destination)
    else:
        for group, values in patch.items():
            for name, value in values.items():
                current[group][name] = value
        current.write(destination, force=True)


def native_decks(case_dir: Path, shape: tuple[int, int, int], nfp: int, box: list[float], flux_name: str, limiter_name: str) -> tuple[str, str]:
    nr, nz, ntor = shape
    grid = f"mtor={nfp},nr={nr},nz={nz},ntor={ntor},rminb={box[0]:.17e},rmaxb={box[1]:.17e},zminb={box[2]:.17e},zmaxb={box[3]:.17e}"
    flux = f"&nlinp1 flx_form='netcdf',flx_file='{flux_name}',file_format='netcdf',wout_file='wout.nc',{grid},ntheta=256,slimit=1.0 /\n"
    limiter = f"&nlinp lim_form='netcdf',lim_file='{limiter_name}',vessel_model='3D',vessel_file='../common/wall-12cm.dat',{grid} /\n"
    (case_dir / "mkflx.input").write_text(flux)
    (case_dir / "mklim.input").write_text(limiter)
    return flux, limiter


def run_native(executable: Path, deck: str, cwd: Path, log_name: str) -> None:
    environment = {**os.environ, "OMP_NUM_THREADS": "1"}
    with (cwd / log_name).open("w") as log:
        subprocess.run([executable], cwd=cwd, input=deck, text=True, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=180, check=True)


def grid_summary(path: Path, field_names: tuple[str, ...]) -> dict[str, object]:
    with Dataset(path) as dataset:
        fields = {name: np.asarray(dataset[name][:]) for name in field_names}
        nfp = int(dataset["mtor"].getValue())
        if all(name in dataset.variables for name in ("R", "Z", "phi")):
            coordinates = {name: np.asarray(dataset[name][:]) for name in ("R", "Z", "phi")}
        else:
            shape = next(iter(fields.values())).shape
            coordinates = {
                "R": np.linspace(float(dataset["rminb"].getValue()), float(dataset["rmaxb"].getValue()), shape[2]),
                "Z": np.linspace(float(dataset["zminb"].getValue()), float(dataset["zmaxb"].getValue()), shape[1]),
                "phi": np.arange(shape[0]) * 2 * np.pi / (nfp * shape[0]),
            }
    for name, values in fields.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name}:{name} contains nonfinite values")
    return {
        "file_sha256": sha256(path),
        "nfp": nfp,
        "coordinates": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "array_sha256": array_sha256(value)} for name, value in coordinates.items()},
        "fields": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "array_sha256": array_sha256(value), "min": float(value.min()), "max": float(value.max()), "unique_values": np.unique(value).tolist() if name == "limiter" else None} for name, value in fields.items()},
        "_coordinates": coordinates,
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def imported_source_identity() -> dict[str, object]:
    module = Path(vmex.__file__).resolve()
    identity: dict[str, object] = {"module": "vmex/__init__.py", "module_sha256": sha256(module)}
    try:
        root = Path(subprocess.check_output(["git", "-C", str(module.parent), "rev-parse", "--show-toplevel"], text=True).strip())
        relative_package = module.parent.relative_to(root)
        identity.update({
            "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "tree": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True).strip(),
            "checkout_clean": not bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1"], text=True).strip()),
            "package_clean": not bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1", "--", str(relative_package)], text=True).strip()),
        })
    except (subprocess.CalledProcessError, ValueError):
        identity.update({"commit": None, "tree": None, "checkout_clean": None, "package_clean": None})
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="public handoff data directory containing SHA256.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-bin", type=Path, required=True, help="build root containing MKFLX/mkflx.exe and MKLIM/mklim.exe")
    parser.add_argument("--source-commit", required=True, help="HINT source commit used for the native binaries")
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--case", choices=CASES, action="append", dest="cases")
    parser.add_argument("--grid", default="64,64,32", help="nr,nz,ntor")
    parser.add_argument("--deck-name", help="protocol deck in --inputs; defaults to coarse.in or fine.in for a matching grid")
    parser.add_argument("--prepare-only", action="store_true", help="write native decks without executing MKFLX/MKLIM")
    args = parser.parse_args()

    cases = tuple(args.cases or CASES)
    if len(set(cases)) != len(cases):
        parser.error("each case may be selected once")
    try:
        shape = tuple(int(value) for value in args.grid.split(","))
    except ValueError:
        parser.error("--grid must be nr,nz,ntor")
    if len(shape) != 3 or min(shape) < 4:
        parser.error("--grid must contain three dimensions >=4")
    deck_name = args.deck_name
    if deck_name is None:
        deck_name = {(64, 64, 32): "coarse.in", (128, 128, 64): "fine.in"}.get(shape)
        if deck_name is None:
            parser.error("select --deck-name explicitly for a grid other than the published coarse/fine grids")
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", args.source_commit):
        parser.error("--source-commit must be a hexadecimal revision")
    if args.output.exists():
        parser.error("--output already exists; refusing to overwrite or merge results")
    if jax.default_backend() != args.device:
        parser.error(f"requested {args.device}, but JAX selected {jax.default_backend()}")

    inputs = args.inputs.resolve()
    native_bin = args.native_bin.resolve()
    executables = {"mkflx": native_bin / "MKFLX" / "mkflx.exe", "mklim": native_bin / "MKLIM" / "mklim.exe"}
    for name, executable in executables.items():
        if not executable.is_file() or not os.access(executable, os.X_OK):
            parser.error(f"missing executable for {name}: expected {name.upper()}/{name}.exe below --native-bin")
    build_config_path = native_bin / "build_config.json"
    if not build_config_path.is_file():
        parser.error("--native-bin must contain build_config.json from the public build helper")
    build_config = json.loads(build_config_path.read_text())
    if str(build_config.get("commit", "")).lower() != args.source_commit.lower():
        parser.error("--source-commit does not match native build_config.json")
    build_flags = shlex.split(str(build_config.get("flags", "")))
    unsupported_flags = sorted(set(build_flags) - SAFE_BUILD_FLAGS)
    if unsupported_flags:
        parser.error("native build_config.json contains unsupported or path-bearing compiler flags")
    manifest_path = inputs / "SHA256.json"
    if not manifest_path.is_file():
        parser.error("--inputs must contain the public handoff data/SHA256.json")
    input_manifest = json.loads(manifest_path.read_text())
    verified_members = {}
    deck_path, verified_members[deck_name] = verify_member(inputs, input_manifest, deck_name)

    loaded = {}
    for case in cases:
        label = "LandremanPaul2021_QA_" + case + "_bootstrap"
        wout_key = case + ".nc"
        vmec_key = "../../../examples/data/input." + label
        coil_key = "../../../examples/data/ESSOS_biot_savart_LandremanPaulQA_" + case + "_bootstrap.json"
        wout_path, verified_members[wout_key] = verify_member(inputs, input_manifest, wout_key)
        vmec_path, verified_members[vmec_key] = verify_member(inputs, input_manifest, vmec_key)
        coil_path, verified_members[coil_key] = verify_member(inputs, input_manifest, coil_key)
        wout = read_wout(wout_path)
        if int(wout.ier_flag) != 0:
            raise ValueError(f"{case} source WOUT is not converged")
        table = current_profile(wout)
        loaded[case] = {"wout_path": wout_path, "coil_path": coil_path, "vmec_path": vmec_path, "wout": wout, "table": table}

    first = loaded[cases[0]]["wout"]
    nfp = int(first.nfp)
    edge, phi = boundary(first)
    boundary_identity = {"shared": len(cases) > 1, "exact": True}
    for case in cases[1:]:
        other = loaded[case]["wout"]
        exact = (
            int(other.nfp) == nfp
            and np.array_equal(np.asarray(other.xm), np.asarray(first.xm))
            and np.array_equal(np.asarray(other.xn), np.asarray(first.xn))
            and np.array_equal(np.asarray(other.rmnc[-1]), np.asarray(first.rmnc[-1]))
            and np.array_equal(np.asarray(other.zmns[-1]), np.asarray(first.zmns[-1]))
        )
        if not exact:
            raise ValueError("selected cases do not have identical outer-boundary Fourier coefficients")
    box = [float(edge[..., 0].min() - 0.19), float(edge[..., 0].max() + 0.19), float(edge[..., 1].min() - 0.19), float(edge[..., 1].max() + 0.19)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=args.output.name + ".tmp-", dir=args.output.parent))
    started = time.monotonic()
    try:
        common = temporary / "common"
        common.mkdir()
        wall = write_wall(common / "wall-12cm.dat", edge, nfp, 0.12)
        np.savez_compressed(common / "wall-12cm.npz", RZ=wall, phi=phi)

        provenance: dict[str, object] = {
            "schema": 1,
            "preparation_driver": {"name": Path(__file__).name, "sha256": sha256(Path(__file__).resolve())},
            "vmex_source": imported_source_identity(),
            "status": "prepared" if args.prepare_only else "complete",
            "source_commit": args.source_commit.lower(),
            "device": args.device,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "jax_backend": jax.default_backend(),
                "jax_device_kind": jax.devices()[0].device_kind,
            },
            "grid": list(shape),
            "nfp_from_wout": nfp,
            "coil_segments": COIL_SEGMENTS,
            "box_m": box,
            "wall": {"definition": "0.12 m outward poloidal R/Z polygon buffer", "sections_per_period": int(edge.shape[0]), "vertices_per_section": int(edge.shape[1]), "boundary_identity": boundary_identity, "file_sha256": sha256(common / "wall-12cm.dat"), "array_sha256": array_sha256(wall)},
            "verified_input_manifest_sha256": sha256(manifest_path),
            "verified_input_members": verified_members,
            "native_build": {
                "build_config_sha256": sha256(build_config_path),
                "declared_source_commit": build_config.get("commit"),
                "mode": build_config.get("mode"),
                "compiler_basename": Path(str(build_config.get("compiler", "unknown"))).name,
                "flags": build_flags,
                "platform": build_config.get("platform"),
                "linked_source_certified": False,
                "limitation": "The legacy build config may describe stale objects and does not hash patched sources; executable hashes are authoritative artifacts, not source-build proof.",
            },
            "native_binaries": {name: {"sha256": sha256(path), "name": f"{name}.exe"} for name, path in executables.items()},
            "packages": {name: package_version(name) for name in ("jax", "jaxlib", "numpy", "scipy", "netCDF4", "h5py", "shapely", "vmex", "essos", "f90nml")},
            "cases": {},
        }
        coordinate_reference = None
        for case in cases:
            item = loaded[case]
            destination = temporary / case
            destination.mkdir()
            shutil.copy2(item["wout_path"], destination / "wout.nc")
            np.savetxt(destination / "current_mapping.csv", item["table"], delimiter=",", header="s,normalized_lambda", comments="")
            vmec_input = VmecInput.from_file(item["vmec_path"])
            render_deck(deck_path, destination / "hint.input", shape, box, item["wout"], vmec_input, item["table"])
            contract = validate_case_contract(item["wout"], item["vmec_path"], destination / "hint.input", item["table"])
            rendered = f90nml.read(destination / "hint.input")
            n3 = rendered["nlinp3"]
            actual_grid = (int(n3["nr"]), int(n3["nz"]), int(n3["ntor"]))
            if actual_grid != shape or int(n3["mtor"]) != nfp:
                raise ValueError("rendered HINT deck grid contract failed")
            open_group = rendered["fopen"]
            output_names = {
                "vacuum": str(open_group["vac_file"]),
                "flux": str(open_group["flx_file"]),
                "limiter": str(open_group["limiter_file"]),
            }
            if any(Path(name).name != name or not name.endswith(".nc") for name in output_names.values()):
                raise ValueError("HINT input field filenames must be local NetCDF basenames")
            coil_discretization = write_vacuum(destination / output_names["vacuum"], item["coil_path"], box, shape, nfp)
            flux_deck, limiter_deck = native_decks(destination, shape, nfp, box, output_names["flux"], output_names["limiter"])
            if not args.prepare_only:
                run_native(executables["mkflx"], flux_deck, destination, "mkflx.log")
                run_native(executables["mklim"], limiter_deck, destination, "mklim.log")

            summaries = {output_names["vacuum"]: grid_summary(destination / output_names["vacuum"], ("Bvac_R", "Bvac_phi", "Bvac_Z"))}
            if not args.prepare_only:
                summaries[output_names["flux"]] = grid_summary(destination / output_names["flux"], ("norm_s",))
                summaries[output_names["limiter"]] = grid_summary(destination / output_names["limiter"], ("limiter",))
                limiter = summaries[output_names["limiter"]]["fields"]["limiter"]
                if set(limiter["unique_values"]) != {0.0, 1.0}:
                    raise ValueError("native limiter must contain both binary values")
            for summary in summaries.values():
                coordinates = summary.pop("_coordinates")
                if coordinate_reference is None:
                    coordinate_reference = coordinates
                elif any(not np.array_equal(coordinates[name], coordinate_reference[name]) for name in coordinates):
                    raise ValueError("prepared files do not share identical coordinates")

            provenance["cases"][case] = {
                "source_files": {"wout": {"name": item["wout_path"].name, "sha256": sha256(item["wout_path"])}, "coil": {"name": item["coil_path"].name, "sha256": sha256(item["coil_path"])}, "vmex_input": {"name": item["vmec_path"].name, "sha256": sha256(item["vmec_path"])}, "protocol_deck": {"name": deck_name, "sha256": sha256(deck_path)}},
                "contract": contract,
                "coil_discretization": coil_discretization,
                "outputs": {"current_mapping.csv": sha256(destination / "current_mapping.csv"), "hint.input": sha256(destination / "hint.input"), "mkflx.input": sha256(destination / "mkflx.input"), "mklim.input": sha256(destination / "mklim.input"), **summaries},
                "native_commands": [{"program": "mkflx.exe", "stdin": "mkflx.input"}, {"program": "mklim.exe", "stdin": "mklim.input"}],
            }
        provenance["elapsed_seconds"] = time.monotonic() - started
        (temporary / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        temporary.rename(args.output)
        print(json.dumps({"output": args.output.name, "status": provenance["status"], "cases": list(cases), "grid": list(shape), "provenance_sha256": sha256(args.output / "provenance.json")}, sort_keys=True))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
