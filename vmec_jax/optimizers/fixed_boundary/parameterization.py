"""Boundary parameterization helpers for fixed-boundary optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.modes import ModeTable


@dataclass(frozen=True)
class BoundaryParamSpec:
    """Descriptor for a boundary Fourier coefficient parameter."""

    name: str
    kind: str
    index: int
    m: int
    n: int


def coeff_label(prefix: str, m: int, n: int) -> str:
    """Evaluate coeff label for fixed-boundary VMEC solve and implicit differentiation."""
    n_str = f"{n:+d}".replace("+", "")
    return f"{prefix}{m}{n_str}"


def rebuild_indata_with_resolution(indata, *, mpol: int, ntor: int):
    """Return a copy of ``indata`` with updated VMEC spectral resolution."""

    from vmec_jax.namelist import InData

    new_scalars = dict(indata.scalars)
    new_scalars["MPOL"] = int(mpol)
    new_scalars["NTOR"] = int(ntor)
    return InData(
        scalars=new_scalars,
        indexed=indata.indexed,
        source_path=indata.source_path,
    )


def extend_boundary_for_max_mode(
    indata,
    static,
    boundary,
    max_mode: int,
    *,
    active_max_m: int | None = None,
    active_max_n: int | None = None,
    min_mpol: int | None = None,
    min_ntor: int | None = None,
    required_mpol: int | None = None,
    required_ntor: int | None = None,
) -> tuple:
    """Extend ``indata``, ``static``, and ``boundary`` to support ``max_mode`` DOFs."""

    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.config import config_from_indata
    from vmec_jax.namelist import InData
    from vmec_jax.static import build_static

    cur_mpol = int(indata.get_int("MPOL", 6))
    cur_ntor = int(indata.get_int("NTOR", 0))
    active_m = int(max_mode if active_max_m is None else active_max_m)
    active_n = int(max_mode if active_max_n is None else active_max_n)
    mpol_floor = 5 if min_mpol is None else int(min_mpol)
    ntor_floor = 5 if min_ntor is None else int(min_ntor)
    need_mpol = max(mpol_floor, active_m + 2)
    need_ntor = max(ntor_floor, active_n + 2)
    if required_mpol is not None:
        need_mpol = max(need_mpol, int(required_mpol))
    if required_ntor is not None:
        need_ntor = max(need_ntor, int(required_ntor))

    if need_mpol <= cur_mpol and need_ntor <= cur_ntor:
        return indata, static, boundary

    new_mpol = max(cur_mpol, need_mpol)
    new_ntor = max(cur_ntor, need_ntor)
    new_scalars = dict(indata.scalars)
    new_scalars["MPOL"] = new_mpol
    new_scalars["NTOR"] = new_ntor
    new_indata = InData(
        scalars=new_scalars,
        indexed=indata.indexed,
        source_path=indata.source_path,
    )

    cfg = config_from_indata(new_indata)
    new_static = build_static(cfg)
    new_boundary = boundary_from_indata(new_indata, new_static.modes)

    print(
        f"  [extend_boundary_for_max_mode] extended mpol {cur_mpol}→{new_mpol}, "
        f"ntor {cur_ntor}→{new_ntor}  "
        f"(modes table size: {len(new_static.modes.m)})"
    )
    return new_indata, new_static, new_boundary


def truncate_indata_boundary_modes(
    indata,
    *,
    max_mode: int | None,
    max_m: int | None = None,
    max_n: int | None = None,
):
    """Return a copy of ``indata`` with inactive boundary modes zeroed."""

    from vmec_jax.namelist import InData

    if max_mode is None and max_m is None and max_n is None:
        return indata
    if max_m is None:
        max_m = max_mode
    if max_n is None:
        max_n = max_mode
    if max_m is None or max_n is None:
        raise ValueError("max_m and max_n must be finite when max_mode is omitted.")
    m_limit = int(max_m)
    n_limit = int(max_n)
    boundary_names = {"RBC", "RBS", "ZBC", "ZBS"}
    indexed = {}
    for name, values in indata.indexed.items():
        upper = str(name).upper()
        copied = dict(values)
        if upper in boundary_names:
            copied = {
                tuple(key): float(value)
                for key, value in copied.items()
                if len(tuple(key)) >= 2
                and abs(int(tuple(key)[0])) <= n_limit
                and abs(int(tuple(key)[1])) <= m_limit
            }
        indexed[name] = copied
    return InData(
        scalars=dict(indata.scalars),
        indexed=indexed,
        source_path=indata.source_path,
    )


def boundary_param_specs(
    boundary: BoundaryCoeffs,
    modes: ModeTable,
    *,
    max_mode: int | None = None,
    max_m: int | None = None,
    max_n: int | None = None,
    min_coeff: float = 0.0,
    include: Sequence[str] = ("rc", "zs"),
    fix: Sequence[str] = ("rc00",),
    include_axis: bool = False,
) -> list[BoundaryParamSpec]:
    """Build parameter specifications for boundary optimization."""

    max_m = max_m if max_m is not None else max_mode
    max_n = max_n if max_n is not None else max_mode
    include_set = {item.lower() for item in include}
    fix_set = {item.lower() for item in fix}

    r_cos = np.asarray(boundary.R_cos)
    r_sin = np.asarray(boundary.R_sin)
    z_cos = np.asarray(boundary.Z_cos)
    z_sin = np.asarray(boundary.Z_sin)

    specs: list[BoundaryParamSpec] = []
    for k, (m_i, n_i) in enumerate(zip(np.asarray(modes.m), np.asarray(modes.n))):
        m_i = int(m_i)
        n_i = int(n_i)
        if m_i < 0:
            continue
        if max_m is not None and abs(m_i) > int(max_m):
            continue
        if max_n is not None and abs(n_i) > int(max_n):
            continue
        if not include_axis and m_i == 0 and n_i == 0:
            continue

        if "rc" in include_set and abs(float(r_cos[k])) >= float(min_coeff):
            name = coeff_label("rc", m_i, n_i)
            if name.lower() not in fix_set:
                specs.append(BoundaryParamSpec(name, "rc", k, m_i, n_i))
        if "rs" in include_set and abs(float(r_sin[k])) >= float(min_coeff):
            name = coeff_label("rs", m_i, n_i)
            if name.lower() not in fix_set:
                specs.append(BoundaryParamSpec(name, "rs", k, m_i, n_i))
        if "zc" in include_set and abs(float(z_cos[k])) >= float(min_coeff):
            name = coeff_label("zc", m_i, n_i)
            if name.lower() not in fix_set:
                specs.append(BoundaryParamSpec(name, "zc", k, m_i, n_i))
        if "zs" in include_set and abs(float(z_sin[k])) >= float(min_coeff):
            name = coeff_label("zs", m_i, n_i)
            if name.lower() not in fix_set:
                specs.append(BoundaryParamSpec(name, "zs", k, m_i, n_i))

    return specs


def boundary_param_names(specs: Sequence[BoundaryParamSpec]) -> list[str]:
    """Return the parameter names for a list of specs."""

    return [spec.name for spec in specs]


def lift_boundary_params(
    source_specs: Sequence[BoundaryParamSpec],
    source_params,
    target_specs: Sequence[BoundaryParamSpec],
) -> np.ndarray:
    """Lift a parameter vector defined on one boundary basis to another."""

    source_vals = {spec.name: float(value) for spec, value in zip(source_specs, np.asarray(source_params, dtype=float))}
    return np.asarray([source_vals.get(spec.name, 0.0) for spec in target_specs], dtype=float)


def apply_boundary_params(
    boundary: BoundaryCoeffs,
    specs: Sequence[BoundaryParamSpec],
    params: jnp.ndarray,
) -> BoundaryCoeffs:
    """Apply parameter updates to a boundary coefficient set."""

    r_cos = jnp.asarray(boundary.R_cos)
    r_sin = jnp.asarray(boundary.R_sin)
    z_cos = jnp.asarray(boundary.Z_cos)
    z_sin = jnp.asarray(boundary.Z_sin)

    for idx, spec in enumerate(specs):
        if spec.kind == "rc":
            r_cos = r_cos.at[spec.index].add(params[idx])
        elif spec.kind == "rs":
            r_sin = r_sin.at[spec.index].add(params[idx])
        elif spec.kind == "zc":
            z_cos = z_cos.at[spec.index].add(params[idx])
        elif spec.kind == "zs":
            z_sin = z_sin.at[spec.index].add(params[idx])
        else:
            raise ValueError(f"Unknown boundary parameter kind '{spec.kind}'")

    return BoundaryCoeffs(R_cos=r_cos, R_sin=r_sin, Z_cos=z_cos, Z_sin=z_sin)


def apply_boundary_params_numpy(
    boundary: BoundaryCoeffs,
    specs: Sequence[BoundaryParamSpec],
    params: np.ndarray,
) -> BoundaryCoeffs:
    """Apply parameter updates on the host for branch/cache-key logic."""

    params = np.asarray(params, dtype=float).reshape(-1)
    r_cos = np.asarray(boundary.R_cos, dtype=float).copy()
    r_sin = np.asarray(boundary.R_sin, dtype=float).copy()
    z_cos = np.asarray(boundary.Z_cos, dtype=float).copy()
    z_sin = np.asarray(boundary.Z_sin, dtype=float).copy()

    for idx, spec in enumerate(specs):
        if idx >= int(params.size):
            break
        if spec.kind == "rc":
            r_cos[spec.index] += float(params[idx])
        elif spec.kind == "rs":
            r_sin[spec.index] += float(params[idx])
        elif spec.kind == "zc":
            z_cos[spec.index] += float(params[idx])
        elif spec.kind == "zs":
            z_sin[spec.index] += float(params[idx])
        else:
            raise ValueError(f"Unknown boundary parameter kind '{spec.kind}'")

    return BoundaryCoeffs(R_cos=r_cos, R_sin=r_sin, Z_cos=z_cos, Z_sin=z_sin)


def indexed_boundary_maps_from_boundary(
    boundary: BoundaryCoeffs,
    modes: ModeTable,
) -> dict[str, dict[tuple[int, int], float]]:
    """Build sparse VMEC namelist boundary maps from dense boundary coefficients."""

    maps = {"RBC": {}, "RBS": {}, "ZBC": {}, "ZBS": {}}
    seen: set[tuple[int, int]] = set()
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    r_cos = np.asarray(boundary.R_cos, dtype=float)
    r_sin = np.asarray(boundary.R_sin, dtype=float)
    z_cos = np.asarray(boundary.Z_cos, dtype=float)
    z_sin = np.asarray(boundary.Z_sin, dtype=float)
    for idx, (m_i, n_i) in enumerate(zip(m_arr, n_arr)):
        m_i = int(m_i)
        n_i = int(n_i)
        if m_i < 0:
            continue
        key = (n_i, m_i)
        if key in seen:
            continue
        seen.add(key)
        maps["RBC"][key] = float(r_cos[idx])
        maps["RBS"][key] = float(r_sin[idx])
        maps["ZBC"][key] = float(z_cos[idx])
        maps["ZBS"][key] = float(z_sin[idx])
    return maps


def create_x_scale(
    specs: Sequence[BoundaryParamSpec],
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute per-parameter exponential spectral scaling weights."""

    scales = np.empty(len(specs), dtype=float)
    norm = math.exp(-alpha) if alpha > 0.0 else 1.0
    for i, spec in enumerate(specs):
        level = max(abs(spec.m), abs(spec.n))
        scales[i] = math.exp(-alpha * level) / norm if alpha > 0.0 else 1.0
    return scales

