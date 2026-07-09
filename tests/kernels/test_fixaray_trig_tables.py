from __future__ import annotations

import numpy as np


def test_vmec_trig_tables_match_fixaray_endpoint_weights():
    # VMEC fixaray.f uses endpoint half-weights for cosmui at i=1 and i=ntheta2,
    # but does NOT apply endpoint half-weights to sinmui (because sin(0)=sin(pi)=0).
    from vmec_jax.kernels.tomnsp import vmec_theta_sizes, vmec_trig_tables

    ntheta = 8
    nzeta = 5
    nfp = 3
    mmax = 6
    nmax = 4

    for lasym in [False]:
        ntheta1, ntheta2, ntheta3 = vmec_theta_sizes(ntheta, lasym=lasym)
        trig = vmec_trig_tables(ntheta=ntheta, nzeta=nzeta, nfp=nfp, mmax=mmax, nmax=nmax, lasym=lasym)

        assert int(trig.ntheta1) == int(ntheta1)
        assert int(trig.ntheta2) == int(ntheta2)
        assert int(trig.ntheta3) == int(ntheta3)

        # Check a representative m>=1 column (m=2) so scaling is non-trivial.
        m = 2
        assert m <= mmax

        # i indices in python are 0-based; VMEC's i=1 corresponds to i0=0.
        i0 = 0
        ipi = int(ntheta2 - 1)

        dnorm = float(trig.dnorm)
        theta0 = 0.0
        thetapi = np.pi

        mscale = float(np.asarray(trig.mscale[m]))

        # cosmui should be dnorm*cos(mu)*mscale, with endpoint half-weights.
        expected_cos_i0 = 0.5 * dnorm * (np.cos(m * theta0) * mscale)
        expected_cos_ipi = 0.5 * dnorm * (np.cos(m * thetapi) * mscale)
        assert np.allclose(float(np.asarray(trig.cosmui[i0, m])), expected_cos_i0)
        assert np.allclose(float(np.asarray(trig.cosmui[ipi, m])), expected_cos_ipi)

        # sinmui should be dnorm*sin(mu)*mscale, with NO endpoint half-weights.
        expected_sin_i0 = dnorm * (np.sin(m * theta0) * mscale)
        expected_sin_ipi = dnorm * (np.sin(m * thetapi) * mscale)
        assert np.allclose(float(np.asarray(trig.sinmui[i0, m])), expected_sin_i0)
        assert np.allclose(float(np.asarray(trig.sinmui[ipi, m])), expected_sin_ipi)


def test_vmec_trig_tables_lasym_full_theta_grid():
    from vmec_jax.kernels.tomnsp import vmec_theta_sizes, vmec_trig_tables

    ntheta = 16
    nzeta = 5
    nfp = 3
    mmax = 6
    nmax = 4
    lasym = True

    ntheta1, ntheta2, ntheta3 = vmec_theta_sizes(ntheta, lasym=lasym)
    trig = vmec_trig_tables(ntheta=ntheta, nzeta=nzeta, nfp=nfp, mmax=mmax, nmax=nmax, lasym=lasym)

    assert int(trig.ntheta1) == int(ntheta1)
    assert int(trig.ntheta2) == int(ntheta2)
    assert int(trig.ntheta3) == int(ntheta3)
    assert int(trig.ntheta3) == int(trig.ntheta1)

    # Pick a point strictly beyond the pi index to ensure we are on the full grid.
    i = int(ntheta2)
    m = 3
    theta = (2.0 * np.pi) * float(i) / float(ntheta1)
    mscale = float(np.asarray(trig.mscale[m]))

    expected_cos = np.cos(m * theta) * mscale
    expected_sin = np.sin(m * theta) * mscale

    assert np.allclose(float(np.asarray(trig.cosmu[i, m])), expected_cos)
    assert np.allclose(float(np.asarray(trig.sinmu[i, m])), expected_sin)

    # VMEC fixaray.f uses dnorm = 1/(nzeta*(ntheta2-1)) UNCONDITIONALLY for
    # the reduced [0, pi] force projections (lasym kernels are symmetrized by
    # symforce.f first); only the surface-average normalization dnorm3
    # switches to the full grid 1/(nzeta*ntheta3) for LASYM=T (SPH012314).
    # This test previously pinned the ported defect (full-grid dnorm for
    # lasym), which halved every lasym force projection vs VMEC2000; fixed
    # alongside vmec_jax/core/fourier.py in vmec_jax/kernels/tomnsp.py.
    assert np.allclose(float(trig.dnorm), 1.0 / float(nzeta * (ntheta2 - 1)))
    assert np.allclose(float(trig.dnorm3), 1.0 / float(nzeta * ntheta3))
