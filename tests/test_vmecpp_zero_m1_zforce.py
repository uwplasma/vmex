from __future__ import annotations

import numpy as np

from vmec_jax.vmec_residue import vmec_zero_m1_zforce
from vmec_jax.vmec_tomnsp import TomnspsRZL


def test_vmec_zero_m1_zforce_matches_vmecpp_semantics():
    # VMEC++ `zeroZForceForM1` zeros only the Z-force blocks that would drive the
    # constrained m=1 Z coefficients away from zero. It does NOT touch `fzsc`.
    ns = 5
    mpol = 4
    nrange = 3

    fzsc = np.random.default_rng(0).normal(size=(ns, mpol, nrange))
    fzcs = np.random.default_rng(1).normal(size=(ns, mpol, nrange))
    fzcc = np.random.default_rng(2).normal(size=(ns, mpol, nrange))

    frzl = TomnspsRZL(
        frcc=np.zeros((ns, mpol, nrange)),
        frss=None,
        fzsc=fzsc.copy(),
        fzcs=fzcs.copy(),
        flsc=np.zeros((ns, mpol, nrange)),
        flcs=None,
        frsc=None,
        frcs=None,
        fzcc=fzcc.copy(),
        fzss=None,
        flcc=None,
        flss=None,
    )

    out = vmec_zero_m1_zforce(frzl=frzl, enabled=True)

    np.testing.assert_allclose(np.asarray(out.fzsc), fzsc, rtol=0, atol=0)

    expected_fzcs = fzcs.copy()
    expected_fzcs[:, 1, :] = 0.0
    np.testing.assert_allclose(np.asarray(out.fzcs), expected_fzcs, rtol=0, atol=0)

    expected_fzcc = fzcc.copy()
    expected_fzcc[:, 1, :] = 0.0
    np.testing.assert_allclose(np.asarray(out.fzcc), expected_fzcc, rtol=0, atol=0)

