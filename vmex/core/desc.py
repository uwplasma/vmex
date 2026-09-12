"""Optional DESC input bridge; WOUT is produced by the ordinary VMEX solve."""

from dataclasses import replace
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

from .errors import VmecInputError
from .input import VmecInput

_FORMATS = {".h5": "hdf5", ".hdf5": "hdf5", ".pkl": "pickle", ".pickle": "pickle"}


def is_desc_file(path: Path) -> bool:
    """Recognize DESC outputs or native text decks without importing DESC."""
    if path.suffix.lower() in _FORMATS:
        return True
    try:
        with path.open("rb") as stream:
            text = stream.read(65536).decode("utf-8", errors="replace")
    except OSError as err:
        raise VmecInputError(f"Cannot read input file: {path}", hint=str(err)) from err
    text = re.sub(r"[!#].*", "", text)
    return not re.search(r"&indata|^\s*\{", text, re.I) and bool(
        re.search(r"^\s*(?:sym|M_pol|L_rad|N_tor|Psi)\s*=", text, re.M | re.I)
    )


def _compact_boundary(inp, tolerance):
    # Triangle inequality bounds the displacement everywhere, relative to the
    # RMS nonconstant R/Z geometry (not the much larger major radius). Apply
    # the same bound to both angular derivatives to retain small sharp modes.
    coefficients = np.stack([getattr(inp, key) for key in ("rbc", "zbs", "rbs", "zbc")])
    m = np.arange(inp.mpol)[None, :]
    n = np.arange(-inp.ntor, inp.ntor + 1)[:, None]
    amplitude = np.sum(np.abs(coefficients), axis=0)
    weights = np.stack(np.broadcast_arrays(np.ones_like(amplitude), m, abs(n)))
    scales = np.sqrt(np.sum((coefficients[None] * weights[:, None]) ** 2, axis=(1, 2, 3)) / 2)
    scales[0] = np.sqrt(np.sum(coefficients[:, (n != 0) | (m != 0)] ** 2) / 2)
    candidates = []
    for mp in range(2, inp.mpol + 1):
        for nt in range(inp.ntor + 1):
            omitted = (m >= mp) | (abs(n) > nt)
            tail = np.sum(weights * amplitude * omitted, axis=(1, 2))
            if np.all(tail <= tolerance * scales):
                candidates.append((mp * (2 * nt + 1), mp, nt))
    _, mp, nt = min(candidates)
    cut = slice(inp.ntor - nt, inp.ntor + nt + 1)
    return replace(inp, mpol=mp, ntor=nt, **{
        key: getattr(inp, key)[cut, :mp] for key in ("rbc", "zbs", "rbs", "zbc")
    })


def write_desc_input(source: Path, outdir: Path | None = None, *, tolerance: float = 0.01) -> Path:
    """Convert the final DESC equilibrium/deck stage to a compact VMEC input.

    ``tolerance`` bounds discarded boundary position and angular derivatives;
    it is not a bound on the solved magnetic field. Zero retains all nonzero
    boundary modes. DESC is imported only for this optional conversion.
    """
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 0.01:
        raise VmecInputError("DESC boundary tolerance must lie between 0 and 0.01")
    try:
        from desc.equilibrium import EquilibriaFamily, Equilibrium
        from desc.io import load
        from desc.profiles import PowerSeriesProfile
        from desc.vmec import VMECIO
    except ImportError as err:
        raise VmecInputError("DESC conversion requires desc-opt", hint="Install with pip install 'vmex[desc]'") from err
    try:
        file_format = _FORMATS.get(source.suffix.lower())
        eq = (load(str(source), file_format=file_format) if file_format
              else Equilibrium.from_input_file(str(source)))
        if isinstance(eq, EquilibriaFamily):
            eq = eq[-1]
        if not isinstance(eq, Equilibrium):
            raise ValueError("expected a DESC Equilibrium or nonempty EquilibriaFamily")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.desc"
            VMECIO.write_vmec_input(eq, str(path), NS_ARRAY=[17, 33, 65],
                                    NITER_ARRAY=[2000, 4000, 8000], FTOL_ARRAY=[1e-8, 1e-10, 1e-12])
            inp = _compact_boundary(VmecInput.from_file(path), tolerance)
        # Splines in rho are not splines in s=rho**2. Sample the actual
        # profiles, especially enclosed current: DESC's derivative exporter
        # sets dI/ds=0 on axis, which distorts otherwise regular currents.
        rho = np.linspace(0, 1, 101)
        for profile, prefix, kind in ((eq.pressure, "am", "pmass"), (eq.iota, "ai", "piota"),
                                      (eq.current, "ac", "pcurr")):
            if profile is not None and not (isinstance(profile, PowerSeriesProfile) and profile.sym):
                inp = replace(inp, **{f"{kind}_type": "cubic_spline_I" if kind == "pcurr" else "cubic_spline",
                                      f"{prefix}_aux_s": rho**2, f"{prefix}_aux_f": np.asarray(profile(rho))})
        name = source.stem if file_format else source.name
        target = (outdir or source.parent) / f"input.{name}"
        return inp.to_indata(target)
    except Exception as err:
        raise VmecInputError(f"Cannot convert DESC file: {err}", hint=str(source)) from err
