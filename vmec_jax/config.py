"""High-level configuration extracted from VMEC input (&INDATA).

For now we only extract what we need for geometry/basis:
- mpol, ntor, ns, nfp, lasym
- ntheta/nzeta defaults (VMEC conventions)

This config can be extended to include profiles, iteration controls, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .namelist import InData, read_indata
from .modes import default_grid_sizes


@dataclass(frozen=True)
class FreeBoundaryConfig:
    """Free-boundary runtime inputs from &INDATA.

    This mirrors VMEC2000's top-level indata behavior:
    - `LFREEB=T` is ignored when `MGRID_FILE='NONE'`.
    - `NVACSKIP<=0` falls back to `NFP` in readin.f.
    """

    enabled: bool = False
    mgrid_file: str = "NONE"
    extcur: tuple[float, ...] = ()
    nvacskip: int = 1


@dataclass(frozen=True)
class VMECConfig:
    """Resolved VMEC discretization and boundary-mode configuration.

    This is the lightweight, typed view of ``&INDATA`` used by setup and solver
    kernels.  It stores only values that define array shapes, Fourier grids,
    symmetry, and free-boundary runtime policy; profile data and iteration
    controls remain in ``InData`` so namelist semantics stay VMEC-compatible.
    """

    mpol: int
    ntor: int
    ns: int
    nfp: int
    lasym: bool
    lthreed: bool
    lconm1: bool
    ntheta: int
    nzeta: int
    free_boundary: FreeBoundaryConfig = field(default_factory=FreeBoundaryConfig)

    @property
    def lfreeb(self) -> bool:
        return bool(self.free_boundary.enabled)

    @property
    def mgrid_file(self) -> str:
        return str(self.free_boundary.mgrid_file)

    @property
    def extcur(self) -> tuple[float, ...]:
        return tuple(self.free_boundary.extcur)

    @property
    def nvacskip(self) -> int:
        return int(self.free_boundary.nvacskip)


def _as_float_sequence(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(value)]


def _extcur_from_indata(indata: InData) -> tuple[float, ...]:
    # EXTCUR can appear as EXTCUR = ... or indexed EXTCUR(i) = ...
    indexed = indata.indexed.get("EXTCUR")
    if indexed:
        max_i = 0
        for idx in indexed:
            if len(idx) == 1 and idx[0] > max_i:
                max_i = int(idx[0])
        if max_i > 0:
            vals = [0.0] * max_i
            for idx, value in indexed.items():
                if len(idx) != 1:
                    continue
                i = int(idx[0])
                if i <= 0:
                    continue
                vals[i - 1] = float(value)
            return tuple(vals)
    return tuple(_as_float_sequence(indata.get("EXTCUR", None)))


def config_from_indata(indata: InData) -> VMECConfig:
    """Build a ``VMECConfig`` from parsed ``&INDATA`` values."""

    mpol = indata.get_int("MPOL", 6)
    ntor = indata.get_int("NTOR", 0)
    # VMEC commonly uses NS_ARRAY = [coarse, ..., fine]. For setup we want the *finest*.
    ns_array = indata.get("NS_ARRAY", 0)
    if isinstance(ns_array, list) and ns_array:
        ns = int(ns_array[-1])
    else:
        ns = indata.get_int("NS_ARRAY", 0)
    if ns == 0:
        ns = indata.get_int("NS", 31)  # fallback (some inputs use NS instead)
    nfp = indata.get_int("NFP", 1)
    lasym = indata.get_bool("LASYM", False)
    # VMEC convention: lthreed is derived from toroidal modes (ntor>0).
    lthreed = bool(ntor > 0)
    lconm1 = indata.get_bool("LCONM1", True)
    ntheta_in = indata.get_int("NTHETA", 0)
    nzeta_in = indata.get_int("NZETA", 0)
    ntheta, nzeta = default_grid_sizes(mpol=mpol, ntor=ntor, ntheta=ntheta_in, nzeta=nzeta_in)
    mgrid_file = str(indata.get("MGRID_FILE", "NONE")).strip()
    if (mgrid_file.startswith("'") and mgrid_file.endswith("'")) or (
        mgrid_file.startswith('"') and mgrid_file.endswith('"')
    ):
        mgrid_file = mgrid_file[1:-1].strip()
    lfreeb_req = bool(indata.get_bool("LFREEB", False))
    # VMEC2000 read_indata.f: IF (lfreeb .and. mgrid_file.eq.'NONE') lfreeb = .false.
    lfreeb = bool(lfreeb_req and mgrid_file.upper() != "NONE")
    nvacskip = int(indata.get_int("NVACSKIP", 0))
    # VMEC2000 readin.f: IF (nvacskip .LE. 0) nvacskip = nfp
    if nvacskip <= 0:
        nvacskip = nfp
    free_boundary = FreeBoundaryConfig(
        enabled=lfreeb,
        mgrid_file=mgrid_file,
        extcur=_extcur_from_indata(indata),
        nvacskip=nvacskip,
    )
    return VMECConfig(
        mpol=mpol,
        ntor=ntor,
        ns=ns,
        nfp=nfp,
        lasym=lasym,
        lthreed=lthreed,
        lconm1=lconm1,
        ntheta=ntheta,
        nzeta=nzeta,
        free_boundary=free_boundary,
    )


def _resolve_input_relative_paths(cfg: VMECConfig, *, input_path: str | Path) -> VMECConfig:
    if not bool(cfg.lfreeb):
        return cfg
    mgrid_file = str(cfg.mgrid_file).strip()
    if (not mgrid_file) or mgrid_file.upper() == "NONE":
        return cfg
    mgrid_path = Path(mgrid_file).expanduser()
    if mgrid_path.is_absolute():
        return cfg
    input_dir = Path(input_path).expanduser().resolve().parent
    resolved = str((input_dir / mgrid_path).resolve())
    return replace(cfg, free_boundary=replace(cfg.free_boundary, mgrid_file=resolved))


def load_config(path: str | Path) -> tuple[VMECConfig, InData]:
    """Read an input deck and return both resolved config and raw namelist."""

    indata = read_indata(path)
    cfg = _resolve_input_relative_paths(config_from_indata(indata), input_path=path)
    return cfg, indata
