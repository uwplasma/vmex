"""Mode tables matching VMEC's conventions.

VMEC uses a slightly special convention:
- m ranges from 0..mpol-1 (mpol1), but many arrays allocate up to mpol (mpol)
- n ranges from -ntor..ntor, except for m=0 where negative n are omitted
- VMEC internally uses the *field-period* toroidal angle zeta in [0,2pi) and stores xn = n*nfp

For the JAX port we keep explicit (m, n) pairs and apply nfp scaling when needed (e.g. physical toroidal derivatives).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ModeTable:
    """VMEC Fourier mode ordering for arrays indexed by ``(m, n)``.

    ``m`` and ``n`` are integer arrays with the same length.  The toroidal
    number is stored before multiplying by ``nfp`` so the same table can be
    reused for input parsing, geometry evaluation, and VMEC ``xm/xn`` output.
    """

    m: np.ndarray  # (K,)
    n: np.ndarray  # (K,)  integer toroidal mode number (before *nfp)

    @property
    def K(self) -> int:
        """Evaluate K for VMEC-JAX numerical workflow."""
        return int(self.m.size)


@lru_cache(maxsize=64)
def _vmec_mode_table_cached(mpol: int, ntor: int) -> ModeTable:
    mpol = int(abs(mpol))
    ntor = int(abs(ntor))

    ms = []
    ns = []
    for m in range(0, mpol):
        nmin = 0 if m == 0 else -ntor
        for n in range(nmin, ntor + 1):
            ms.append(m)
            ns.append(n)
    return ModeTable(m=np.asarray(ms, dtype=int), n=np.asarray(ns, dtype=int))


def vmec_mode_table(mpol: int, ntor: int) -> ModeTable:
    """Create VMEC-like (m,n) pairs.

    Includes all (m,n) with m=0..mpol-1 and n in [-ntor..ntor], but for m=0 uses n>=0.

    This matches the xm/xn tables built in VMEC's `fixaray.f` (see nmin0 logic).
    """
    return _vmec_mode_table_cached(int(abs(mpol)), int(abs(ntor)))


@lru_cache(maxsize=64)
def _nyquist_mode_table_cached(mpol: int, ntor: int) -> ModeTable:
    """Create a VMEC-style Nyquist mode table.

    VMEC's Nyquist grids extend the Fourier bandwidth by a small padding. Empirically
    (and consistent with `fixaray` defaults), the Nyquist mode limits are:

    - mmax_nyq = mpol + 3
    - nmax_nyq = ntor + 2  (but 0 when ntor==0)

    This matches the bundled VMEC2000 `wout_*.nc` files used for parity checks.
    """
    mpol = int(abs(mpol))
    ntor = int(abs(ntor))

    mmax = mpol + 3
    nmax = 0 if ntor == 0 else ntor + 2

    ms = []
    ns = []
    for m in range(0, mmax + 1):
        nmin = 0 if m == 0 else -nmax
        for n in range(nmin, nmax + 1):
            ms.append(m)
            ns.append(n)
    return ModeTable(m=np.asarray(ms, dtype=int), n=np.asarray(ns, dtype=int))


def nyquist_mode_table(mpol: int, ntor: int) -> ModeTable:
    """Create a cached VMEC-style Nyquist mode table from spectral limits."""

    return _nyquist_mode_table_cached(int(abs(mpol)), int(abs(ntor)))


def nyquist_mode_table_from_grid(*, mpol: int, ntor: int, ntheta: int, nzeta: int) -> ModeTable:
    """Create a VMEC-style Nyquist mode table using the angular grid sizes.

    VMEC derives Nyquist limits from the angular grid (fixaray.f):

    - mnyq = max(ntheta1/2, mpol-1)
    - nnyq = max(nzeta/2, ntor)

    where ntheta1 = 2*(ntheta//2).
    """
    mpol = int(abs(mpol))
    ntor = int(abs(ntor))
    ntheta1 = 2 * (int(ntheta) // 2)
    mnyq = max(ntheta1 // 2, max(mpol - 1, 0))
    nnyq = max(int(nzeta) // 2, max(ntor, 0))

    ms = []
    ns = []
    for m in range(0, mnyq + 1):
        nmin = 0 if m == 0 else -nnyq
        for n in range(nmin, nnyq + 1):
            ms.append(m)
            ns.append(n)
    return ModeTable(m=np.asarray(ms, dtype=int), n=np.asarray(ns, dtype=int))


def default_grid_sizes(mpol: int, ntor: int, ntheta: int = 0, nzeta: int = 0) -> Tuple[int, int]:
    """Match VMEC defaults in initialize_vmec_arrays.f."""
    mpol = int(abs(mpol))
    ntor = int(abs(ntor))
    if ntheta <= 0:
        ntheta = 2 * mpol + 6
    # VMEC makes ntheta1 = even
    ntheta1 = 2 * (ntheta // 2)

    if ntor == 0:
        # Axisymmetric: solution is constant in ζ, so NZETA=1 is always sufficient.
        # VMEC2000 convention: user-specified NZETA is ignored when NTOR=0 (lthreed=False).
        # This gives a large speedup when users specify NZETA>1 for historical reasons
        # (e.g. cth_like with NZETA=36 but NTOR=0 runs ~26× faster with NZETA=1).
        nzeta = 1
    elif nzeta <= 0:
        nzeta = 2 * ntor + 4

    return ntheta1, int(nzeta)
