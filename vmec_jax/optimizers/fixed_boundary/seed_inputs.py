"""Input-deck seed helpers for fixed-boundary optimization examples."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Sequence

from vmec_jax.optimizers.fixed_boundary.parameterization import rebuild_indata_with_resolution


def rebuild_for_optimization_resolution(
    indata,
    *,
    max_mode: int,
    min_vmec_mode: int = 5,
    vmec_mpol: int | None = None,
    vmec_ntor: int | None = None,
):
    """Set VMEC spectral resolution for an optimization run.

    By default both VMEC ``MPOL`` and ``NTOR`` are set to at least
    ``max(min_vmec_mode, max_mode + 2)``.  ``vmec_mpol`` and ``vmec_ntor`` are
    explicit internal-resolution overrides for diagnostic sweeps that decouple
    the VMEC spectral grid from active boundary degrees of freedom.  Stage
    builders still extend a mode table if a later active stage actually needs
    more modes, so these overrides are safe for anisotropic probes such as
    ``max_m=1, max_n=5``.
    """

    floor = max(int(min_vmec_mode), int(max_mode) + 2)
    mpol = max(1, int(vmec_mpol)) if vmec_mpol is not None else floor
    ntor = max(0, int(vmec_ntor)) if vmec_ntor is not None else floor
    return rebuild_indata_with_resolution(indata, mpol=mpol, ntor=ntor)


def simple_omnigenity_seed_indata(
    indata,
    *,
    max_mode: int,
    include: Sequence[str] = ("rc", "zs"),
    fix: Sequence[str] = ("rc00",),
    perturbation: float = 1.0e-5,
    r0: float | None = None,
    rbc01: float | None = None,
    zbs01: float | None = None,
):
    """Return an input deck with a deterministic near-circular omnigenity seed.

    The seed keeps only the base shape ``RBC(0,0)``, ``RBC(0,1)``, and
    ``ZBS(0,1)`` from the source deck unless explicit values are supplied.
    Every other active optimizable boundary coefficient with
    ``max(abs(m), abs(n)) <= max_mode`` is set to a deterministic
    ``+/-perturbation`` value.  This avoids exactly-zero Jacobian columns when
    examples start far from a QA/QH/QP/QI warm-start boundary.
    """

    max_mode_i = int(max_mode)
    perturbation = float(perturbation)
    if max_mode_i < 0:
        raise ValueError("max_mode must be non-negative.")
    if not math.isfinite(perturbation) or perturbation < 0.0:
        raise ValueError("perturbation must be finite and non-negative.")

    boundary_keys = {"RBC", "RBS", "ZBC", "ZBS"}
    include_set = {str(item).lower() for item in include}
    fix_set = {str(item).lower() for item in fix}
    family_keys = {
        "rc": "RBC",
        "rs": "RBS",
        "zc": "ZBC",
        "zs": "ZBS",
    }
    kind_offsets = {"rc": 17, "rs": 29, "zc": 43, "zs": 59}

    def _source_value(key: str, index: tuple[int, int], default: float) -> float:
        return float(indata.indexed.get(key, {}).get(index, default))

    def _sign(kind: str, m_i: int, n_i: int) -> float:
        parity = (kind_offsets[kind] + 1009 * int(m_i) + 9176 * (int(n_i) + 8192)) % 2
        return 1.0 if parity == 0 else -1.0

    def _spec_name(kind: str, m_i: int, n_i: int) -> str:
        n_str = f"{int(n_i):+d}".replace("+", "")
        return f"{kind}{int(m_i)}{n_str}"

    out = copy.deepcopy(indata)
    out.indexed = {
        key: copy.deepcopy(values)
        for key, values in indata.indexed.items()
        if str(key).upper() not in boundary_keys
    }
    out.indexed["RBC"] = {
        (0, 0): _source_value("RBC", (0, 0), 1.0) if r0 is None else float(r0),
        (0, 1): _source_value("RBC", (0, 1), 0.2) if rbc01 is None else float(rbc01),
    }
    out.indexed["ZBS"] = {
        (0, 1): _source_value("ZBS", (0, 1), 0.2) if zbs01 is None else float(zbs01),
    }

    preserved = {("RBC", (0, 0)), ("RBC", (0, 1)), ("ZBS", (0, 1))}
    for m_i in range(0, max_mode_i + 1):
        n_values = range(0, max_mode_i + 1) if m_i == 0 else range(-max_mode_i, max_mode_i + 1)
        for n_i in n_values:
            if m_i == 0 and n_i == 0:
                continue
            for kind, key in family_keys.items():
                if kind not in include_set:
                    continue
                spec_name = _spec_name(kind, m_i, n_i)
                if spec_name.lower() in fix_set or (key, (n_i, m_i)) in preserved:
                    continue
                out.indexed.setdefault(key, {})[(n_i, m_i)] = _sign(kind, m_i, n_i) * perturbation

    return out


def prepare_simple_omnigenity_seed_input(
    input_file,
    output_dir,
    *,
    max_mode: int,
    min_vmec_mode: int = 5,
    vmec_mpol: int | None = None,
    vmec_ntor: int | None = None,
    enabled: bool = True,
    include: Sequence[str] = ("rc", "zs"),
    fix: Sequence[str] = ("rc00",),
    perturbation: float = 1.0e-5,
    filename: str = "input.simple_seed",
):
    """Write and return a simple omnigenity seed input path when enabled."""

    input_path = Path(input_file)
    if not bool(enabled):
        return input_path

    from vmec_jax.namelist import read_indata, write_indata

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indata = read_indata(input_path)
    indata = rebuild_for_optimization_resolution(
        indata,
        max_mode=max_mode,
        min_vmec_mode=min_vmec_mode,
        vmec_mpol=vmec_mpol,
        vmec_ntor=vmec_ntor,
    )
    seeded = simple_omnigenity_seed_indata(
        indata,
        max_mode=max_mode,
        include=include,
        fix=fix,
        perturbation=perturbation,
    )
    write_indata(output_path, seeded)
    return output_path


def interpolate_indata_boundary(
    seed_indata,
    reference_indata,
    lam: float,
    *,
    keys: Sequence[str] = ("RBC", "ZBS", "RBS", "ZBC"),
    scalar_keys: Sequence[str] = (),
    max_mode: int | None = None,
    require_same_nfp: bool = True,
    preserve_seed_scalars: Sequence[str] = ("NFP", "LASYM"),
):
    """Interpolate selected VMEC boundary Fourier coefficients and scalars.

    This helper implements a deterministic global-to-local preconditioner for
    far-seed QI optimization.  ``lam=0`` preserves the seed boundary for the
    selected keys, while ``lam=1`` uses the reference boundary.  Optional
    ``scalar_keys`` are interpolated by the same rule.  Scalar VMEC metadata
    remains seed-owned for entries in ``preserve_seed_scalars`` so a reference
    family can be used without accidentally changing the user's field-period
    count or symmetry flag.

    If ``max_mode`` is given, selected boundary coefficient dictionaries are
    projected to modes with ``abs(m) <= max_mode`` and ``abs(n) <= max_mode``.
    """

    lam = float(lam)
    if not math.isfinite(lam):
        raise ValueError("Boundary interpolation lambda must be finite.")
    if bool(require_same_nfp) and int(seed_indata.get_int("NFP", -1)) != int(reference_indata.get_int("NFP", -2)):
        raise ValueError(
            "Boundary interpolation requires same-NFP inputs; "
            f"got seed NFP={seed_indata.get_int('NFP')} and reference NFP={reference_indata.get_int('NFP')}."
        )

    out = copy.deepcopy(seed_indata)
    for key in preserve_seed_scalars:
        key = key.upper()
        if key in seed_indata.scalars:
            out.scalars[key] = copy.deepcopy(seed_indata.scalars[key])

    if "MPOL" in seed_indata.scalars or "MPOL" in reference_indata.scalars:
        out.scalars["MPOL"] = max(int(seed_indata.get_int("MPOL", 0)), int(reference_indata.get_int("MPOL", 0)))
    if "NTOR" in seed_indata.scalars or "NTOR" in reference_indata.scalars:
        out.scalars["NTOR"] = max(int(seed_indata.get_int("NTOR", 0)), int(reference_indata.get_int("NTOR", 0)))

    def _interpolate_scalar_value(seed_value, reference_value, key: str):
        if isinstance(seed_value, bool) or isinstance(reference_value, bool):
            raise TypeError(f"Cannot interpolate boolean VMEC scalar {key!r}.")
        if isinstance(seed_value, (int, float)) and isinstance(reference_value, (int, float)):
            return (1.0 - lam) * float(seed_value) + lam * float(reference_value)
        if isinstance(seed_value, (list, tuple)) and isinstance(reference_value, (list, tuple)):
            if len(seed_value) != len(reference_value):
                raise ValueError(
                    f"Cannot interpolate VMEC scalar list {key!r} with lengths "
                    f"{len(seed_value)} and {len(reference_value)}."
                )
            return [
                _interpolate_scalar_value(seed_item, reference_item, key)
                for seed_item, reference_item in zip(seed_value, reference_value, strict=True)
            ]
        raise TypeError(
            f"Cannot interpolate VMEC scalar {key!r} with values {seed_value!r} and {reference_value!r}."
        )

    preserved = {str(key).upper() for key in preserve_seed_scalars}
    for key in tuple(str(item).upper() for item in scalar_keys):
        if key in preserved:
            raise ValueError(f"Cannot interpolate preserved seed scalar {key!r}.")
        if key not in seed_indata.scalars or key not in reference_indata.scalars:
            continue
        out.scalars[key] = _interpolate_scalar_value(seed_indata.scalars[key], reference_indata.scalars[key], key)

    max_mode_i = None if max_mode is None else int(max_mode)
    for key in tuple(item.upper() for item in keys):
        seed_coeffs = seed_indata.indexed.get(key, {})
        ref_coeffs = reference_indata.indexed.get(key, {})
        indices = set(seed_coeffs) | set(ref_coeffs)
        interpolated = {}
        for idx in sorted(indices):
            if max_mode_i is not None and any(abs(int(i)) > max_mode_i for i in idx[:2]):
                continue
            seed_value = float(seed_coeffs.get(idx, 0.0))
            reference_value = float(ref_coeffs.get(idx, 0.0))
            interpolated[idx] = (1.0 - lam) * seed_value + lam * reference_value
        if interpolated or key in out.indexed:
            out.indexed[key] = interpolated
    return out
