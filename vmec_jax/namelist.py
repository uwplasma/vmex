"""Minimal parser for VMEC-style Fortran namelist (&INDATA).

Goals:
- **No third-party dependency** (no f90nml)
- Works on VMEC2000 input files used in the included python/tests
- Handles indexed coefficient syntax: RBC(m,n)=..., including negative indices
- Handles arrays split across multiple lines

This is not a full Fortran namelist implementation, but is intentionally targeted to VMEC inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

Number = Union[int, float]
Scalar = Union[str, bool, Number]
Value = Union[Scalar, List[Scalar]]


def _strip_fortran_comments(line: str) -> str:
    """Remove '!' comments, respecting single-quoted strings."""
    out = []
    in_quote = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "'":
            in_quote = not in_quote
            out.append(ch)
        elif ch == "!" and not in_quote:
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out)


_ASSIGN_RE = re.compile(r"(?P<key>[A-Za-z_]\w*(?:\([^\)]*\))?)\s*=", re.MULTILINE)


def _tokenize_value_chunk(chunk: str) -> List[str]:
    """Tokenize a value chunk into tokens, keeping quoted strings intact."""
    tokens: List[str] = []
    buf: List[str] = []
    in_quote = False
    chunk = chunk.strip()
    i = 0
    while i < len(chunk):
        ch = chunk[i]
        if ch == "'":
            in_quote = not in_quote
            buf.append(ch)
        elif not in_quote and ch in [",", "\n", "\t", " ", "\r"]:
            if buf:
                tok = "".join(buf).strip()
                if tok:
                    tokens.append(tok)
                buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        tok = "".join(buf).strip()
        if tok:
            tokens.append(tok)
    return tokens


_BOOL_TRUE = {"T", ".T.", ".TRUE.", "TRUE"}
_BOOL_FALSE = {"F", ".F.", ".FALSE.", "FALSE"}


_REPEAT_RE = re.compile(r"^(?P<count>\d+)\*(?P<value>.+)$")


def _expand_fortran_repeats(tokens: List[str]) -> List[str]:
    """Expand Fortran repeat-count syntax like `11*0.0` into explicit tokens.

    VMEC inputs often use this compact form for arrays (e.g. `AI = 11*0.0, 0.8`).
    We support only the common scalar case: `N*<scalar>`.
    """
    out: List[str] = []
    for tok in tokens:
        m = _REPEAT_RE.match(tok.strip())
        if not m:
            out.append(tok)
            continue
        n = int(m.group("count"))
        v = m.group("value").strip()
        # If parsing fails or count is weird, fall back to keeping the token.
        if n <= 0 or not v:
            out.append(tok)
            continue
        out.extend([v] * n)
    return out


def _parse_scalar(tok: str) -> Scalar:
    tok = tok.strip()
    # strings
    if len(tok) >= 2 and tok[0] == "'" and tok[-1] == "'":
        return tok[1:-1]
    up = tok.upper()
    if up in _BOOL_TRUE:
        return True
    if up in _BOOL_FALSE:
        return False
    # integers (including leading zeros)
    if re.fullmatch(r"[+-]?\d+", tok):
        try:
            return int(tok)
        except Exception:
            pass
    # floats (Fortran exponent forms)
    # Accept D or E exponent.
    f = tok.replace("D", "E").replace("d", "E")
    try:
        return float(f)
    except Exception:
        # fall back to raw string
        return tok


def _parse_key(key: str) -> Tuple[str, Tuple[int, ...] | None]:
    """Split KEY or KEY(i,j) into (base, indices)."""
    key = key.strip()
    if "(" not in key:
        return key.upper(), None
    base, rest = key.split("(", 1)
    rest = rest.rstrip(")")
    # VMEC and related tools sometimes use full-slice notation like `RAXIS_CC(:) = ...`.
    # Treat this as a non-indexed assignment with a list value.
    if ":" in rest:
        return base.upper(), None
    idx = tuple(int(x.strip()) for x in rest.split(",") if x.strip() != "")
    return base.upper(), idx


@dataclass
class InData:
    """Parsed VMEC ``&INDATA`` namelist.

    ``scalars`` stores ordinary assignments while ``indexed`` stores VMEC-style
    indexed arrays such as ``RBC(n,m)`` and ``ZBS(n,m)``.  The helper accessors
    intentionally keep VMEC's permissive input behavior: malformed or missing
    values fall back to caller-provided defaults.
    """

    scalars: Dict[str, Value]
    indexed: Dict[str, Dict[Tuple[int, ...], Scalar]]
    source_path: str | None = None

    def get(self, name: str, default: Value | None = None) -> Value | None:
        """Return get for VMEC-JAX numerical workflow."""
        return self.scalars.get(name.upper(), default)

    def get_bool(self, name: str, default: bool = False) -> bool:
        """Return get bool for VMEC-JAX numerical workflow."""
        v = self.get(name, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, list) and v and isinstance(v[0], bool):
            return bool(v[0])
        return bool(v)

    def get_int(self, name: str, default: int = 0) -> int:
        """Return get int for VMEC-JAX numerical workflow."""
        v = self.get(name, default)
        if isinstance(v, list):
            v = v[0] if v else default
        try:
            return int(v)  # type: ignore[arg-type]
        except Exception:
            return default

    def get_float(self, name: str, default: float = 0.0) -> float:
        """Return get float for VMEC-JAX numerical workflow."""
        v = self.get(name, default)
        if isinstance(v, list):
            v = v[0] if v else default
        try:
            return float(v)  # type: ignore[arg-type]
        except Exception:
            return default


def read_indata(path: str | Path) -> InData:
    """Read &INDATA from a VMEC input file."""
    path = Path(path)
    text = path.read_text()
    # isolate &INDATA block
    m_start = re.search(r"&\s*INDATA", text, flags=re.IGNORECASE)
    if not m_start:
        raise ValueError("No &INDATA found")
    # naive end: first '/' after start
    m_end = re.search(r"\n\s*/\s*\n|\n\s*/\s*$", text[m_start.end() :], flags=re.MULTILINE)
    if not m_end:
        raise ValueError("No terminating '/' for &INDATA")
    block = text[m_start.end() : m_start.end() + m_end.start()]

    # strip comments line-by-line
    lines = [_strip_fortran_comments(ln) for ln in block.splitlines()]
    cleaned = "\n".join(lines)

    scalars: Dict[str, Value] = {}
    indexed: Dict[str, Dict[Tuple[int, ...], Scalar]] = {}

    matches = list(_ASSIGN_RE.finditer(cleaned))
    if not matches:
        return InData(scalars=scalars, indexed=indexed, source_path=str(path))

    for i, m in enumerate(matches):
        key_raw = m.group("key")
        key_base, idx = _parse_key(key_raw)
        val_start = m.end()
        val_end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
        chunk = cleaned[val_start:val_end].strip()
        # remove trailing commas
        chunk = re.sub(r",\s*$", "", chunk)
        toks = _expand_fortran_repeats(_tokenize_value_chunk(chunk))
        parsed = [_parse_scalar(t) for t in toks]
        value: Value
        if len(parsed) == 0:
            continue
        elif len(parsed) == 1:
            value = parsed[0]
        else:
            value = parsed

        if idx is None:
            scalars[key_base] = value
        else:
            if key_base not in indexed:
                indexed[key_base] = {}
            # indexed assignments in VMEC are scalar
            if isinstance(value, list):
                if len(value) != 1:
                    raise ValueError(f"Indexed assignment {key_raw} has multiple values")
                indexed[key_base][idx] = value[0]
            else:
                indexed[key_base][idx] = value

    return InData(scalars=scalars, indexed=indexed, source_path=str(path))


def _format_scalar(value: Scalar) -> str:
    """Format a scalar in VMEC-friendly namelist syntax."""
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.16E}"
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _format_value(value: Value) -> str:
    """Format a scalar or list value for namelist output."""
    if isinstance(value, list):
        return ", ".join(_format_scalar(item) for item in value)
    return _format_scalar(value)


def write_indata(path: str | Path, indata: InData) -> None:
    """Write a VMEC ``&INDATA`` namelist block.

    The output is intended for reproducible round-tripping through
    :func:`read_indata`, not for preserving the exact original whitespace or
    comments from the source file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = ["&INDATA"]
    for key, value in indata.scalars.items():
        lines.append(f"  {key} = {_format_value(value)}")

    for key in sorted(indata.indexed):
        coeffs = indata.indexed[key]
        for idx in sorted(coeffs):
            idx_text = ",".join(str(int(i)) for i in idx)
            lines.append(f"  {key}({idx_text}) = {_format_scalar(coeffs[idx])}")

    lines.append("/")
    path.write_text("\n".join(lines) + "\n")


def minimal_fixed_boundary_indata(
    *,
    nfp: int,
    r0: float = 1.0,
    rbc01: float = 0.2,
    zbs01: float = 0.2,
    mpol: int = 5,
    ntor: int = 5,
    ns_array: int | list[int] = 35,
    niter_array: int | list[int] = 1500,
    ftol_array: float | list[float] = 1.0e-13,
    phiedge: float = 0.083,
) -> InData:
    """Return a minimal fixed-boundary VMEC seed used by optimization examples.

    The boundary has only the circular/elliptic seed coefficients
    ``RBC(0,0)``, ``RBC(0,1)``, and ``ZBS(0,1)``.  Optimization examples can
    then activate higher Fourier coefficients through their selected
    ``max_mode`` and continuation policy, so the same simple template can be
    used to demonstrate QA, QH, QP, and QI optimization from a seed far from the
    target magnetic-field structure.
    """

    def _as_list(value):
        return list(value) if isinstance(value, list) else value

    scalars: Dict[str, Value] = {
        "DELT": 0.9,
        "NITER": 10000,
        "NSTEP": 200,
        "TCON0": 2.0,
        "NS_ARRAY": _as_list(ns_array),
        "NITER_ARRAY": _as_list(niter_array),
        "FTOL_ARRAY": _as_list(ftol_array),
        "PRECON_TYPE": "none",
        "PREC2D_THRESHOLD": 1.0e-19,
        "LASYM": False,
        "NFP": int(nfp),
        "MPOL": int(mpol),
        "NTOR": int(ntor),
        "PHIEDGE": float(phiedge),
        "LFREEB": False,
        "NVACSKIP": 6,
        "GAMMA": 0.0,
        "BLOAT": 1.0,
        "SPRES_PED": 1.0,
        "PRES_SCALE": 1.0,
        "PMASS_TYPE": "power_series",
        "AM": 0.0,
        "CURTOR": 0,
        "NCURR": 1,
        "PIOTA_TYPE": "power_series",
        "PCURR_TYPE": "power_series",
    }
    indexed: Dict[str, Dict[Tuple[int, ...], Scalar]] = {
        "RBC": {
            (0, 0): float(r0),
            (0, 1): float(rbc01),
        },
        "ZBS": {
            (0, 1): float(zbs01),
        },
    }
    return InData(scalars=scalars, indexed=indexed)
