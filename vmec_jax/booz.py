"""Boozer-transform helpers used by the CLI and examples."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .namelist import (
    _ASSIGN_RE,
    _expand_fortran_repeats,
    _parse_key,
    _parse_scalar,
    _strip_fortran_comments,
    _tokenize_value_chunk,
)


@dataclass(frozen=True)
class BoozConfig:
    """Configuration for a ``booz_xform_jax`` run."""

    enabled: bool = False
    mbooz: int = 32
    nbooz: int = 32
    surfaces: tuple[float, ...] | None = None
    jit: bool = False


def _truthy(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("", "0", "f", ".f.", "false", ".false.", "no", "off"):
        return False
    if text in ("1", "t", ".t.", "true", ".true.", "yes", "on"):
        return True
    return default


def _as_first(value: object, default: object = None) -> object:
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _extract_namelist(path: Path, name: str) -> dict[str, object]:
    text = Path(path).read_text()
    m_start = re.search(rf"&\s*{re.escape(name)}", text, flags=re.IGNORECASE)
    if not m_start:
        return {}
    m_end = re.search(r"\n\s*/\s*\n|\n\s*/\s*$", text[m_start.end() :], flags=re.MULTILINE)
    if not m_end:
        raise ValueError(f"No terminating '/' for &{name}")
    block = text[m_start.end() : m_start.end() + m_end.start()]
    cleaned = "\n".join(_strip_fortran_comments(line) for line in block.splitlines())
    matches = list(_ASSIGN_RE.finditer(cleaned))
    values: dict[str, object] = {}
    for i, match in enumerate(matches):
        key_raw = match.group("key")
        key_base, idx = _parse_key(key_raw)
        if idx is not None:
            continue
        val_start = match.end()
        val_end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        chunk = re.sub(r",\s*$", "", cleaned[val_start:val_end].strip())
        tokens = _expand_fortran_repeats(_tokenize_value_chunk(chunk))
        parsed = [_parse_scalar(token) for token in tokens]
        if not parsed:
            continue
        values[key_base] = parsed[0] if len(parsed) == 1 else parsed
    return values


def parse_booz_surfaces(value: object) -> tuple[float, ...] | None:
    """Parse ``all`` or a comma/space-separated list of surface values."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("", "all", "*"):
            return None
        parts = [part for part in re.split(r"[\s,]+", text) if part]
    elif isinstance(value, Iterable):
        parts = list(value)
    else:
        parts = [value]
    out = tuple(float(part) for part in parts)
    return out or None


def read_booz_config(path: str | Path) -> BoozConfig:
    """Read optional ``&BOOZ_XFORM_JAX`` settings from a VMEC input file.

    The group is intentionally separate from ``&INDATA`` so conventional VMEC
    inputs remain readable by VMEC2000 while vmec_jax can carry Boozer defaults
    in the same text file.
    """

    values = _extract_namelist(Path(path), "BOOZ_XFORM_JAX")
    if not values:
        return BoozConfig()
    return BoozConfig(
        enabled=_truthy(_as_first(values.get("LBOOZ")), default=False),
        mbooz=int(_as_first(values.get("MBOOZ"), 32)),
        nbooz=int(_as_first(values.get("NBOOZ"), 32)),
        surfaces=parse_booz_surfaces(values.get("BOOZ_SURFACES")),
        jit=_truthy(_as_first(values.get("JIT_BOOZ")), default=False),
    )


def _case_from_vmec_or_wout(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if lower.startswith("wout_") and lower.endswith(".nc"):
        return path.stem.split("wout_", 1)[-1]
    if lower.startswith("input."):
        return name.split("input.", 1)[-1]
    if lower.startswith("input_"):
        return name.split("input_", 1)[-1]
    return path.stem


def resolve_boozmn_path(*, source_path: Path, outdir: Path | None = None, output: Path | None = None) -> Path:
    """Return the default ``boozmn_*.nc`` path for a VMEC input or WOUT file."""

    if output is not None:
        return Path(output)
    base_dir = Path(outdir) if outdir is not None else Path(source_path).parent
    return base_dir / f"boozmn_{_case_from_vmec_or_wout(Path(source_path))}.nc"


def _surface_indices_from_values(bx, surfaces: tuple[float, ...] | None) -> list[int] | None:
    if surfaces is None:
        return None
    ns_in = int(getattr(bx, "ns_in", 0) or 0)
    if ns_in <= 0:
        raise ValueError("Cannot select Boozer surfaces before reading a WOUT file")
    values = np.asarray(surfaces, dtype=float)
    if np.all((0.0 <= values) & (values <= 1.0)):
        s_in = np.asarray(getattr(bx, "s_in", None), dtype=float)
        if s_in.size != ns_in:
            full_grid = np.linspace(0.0, 1.0, ns_in + 1)
            s_in = 0.5 * (full_grid[:-1] + full_grid[1:])
        return [int(np.argmin(np.abs(s_in - value))) for value in values]
    indices = [int(round(value)) for value in values]
    for index in indices:
        if index < 0 or index >= ns_in:
            raise ValueError(f"Boozer surface index {index} is outside 0..{ns_in - 1}")
    return indices


def run_booz_xform(
    wout_path: str | Path,
    *,
    output_path: str | Path | None = None,
    outdir: str | Path | None = None,
    mbooz: int = 32,
    nbooz: int = 32,
    surfaces: tuple[float, ...] | None = None,
    jit: bool = False,
    verbose: bool = True,
) -> Path:
    """Run ``booz_xform_jax`` from a WOUT file and write ``boozmn_*.nc``."""

    try:
        from booz_xform_jax import Booz_xform
    except Exception as exc:  # pragma: no cover - dependency is optional at import time
        raise ImportError(
            "Boozer transforms require booz_xform_jax. Install vmec-jax from the "
            "current package metadata or run `pip install booz_xform_jax`."
        ) from exc

    wout = Path(wout_path).expanduser().resolve()
    if not wout.exists():
        raise FileNotFoundError(f"WOUT file not found: {wout}")
    out = resolve_boozmn_path(
        source_path=wout,
        outdir=Path(outdir).expanduser().resolve() if outdir is not None else None,
        output=Path(output_path).expanduser().resolve() if output_path is not None else None,
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    bx = Booz_xform(verbose=1 if verbose else 0, mboz=int(mbooz), nboz=int(nbooz))
    bx.read_wout(str(wout), flux=False)
    bx.compute_surfs = _surface_indices_from_values(bx, surfaces)
    bx.run(jit=bool(jit))
    bx.write_boozmn(str(out))
    return out


__all__ = [
    "BoozConfig",
    "parse_booz_surfaces",
    "read_booz_config",
    "resolve_boozmn_path",
    "run_booz_xform",
]
