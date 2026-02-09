from __future__ import annotations

import numpy as np

from vmec_jax.multigrid import interp_vmec_radial_coeffs


def _scalxc_ref(*, ns: int, m: np.ndarray) -> np.ndarray:
    ns = int(ns)
    if ns <= 0:
        return np.zeros((0, int(m.size)), dtype=float)
    s = np.linspace(0.0, 1.0, ns, dtype=float)
    sqrts = np.sqrt(np.maximum(s, 0.0))
    sqrts[-1] = 1.0
    sq2 = sqrts[1] if ns >= 2 else 1.0
    scal_odd = 1.0 / np.maximum(sqrts, sq2)
    is_odd = (m.astype(int) % 2) == 1
    out = np.ones((ns, int(m.size)), dtype=float)
    out[:, is_odd] = scal_odd[:, None]
    return out


def _interp_ref(x_old: np.ndarray, *, m: np.ndarray, ns_new: int) -> np.ndarray:
    x_old = np.asarray(x_old, dtype=float)
    ns_old, K = x_old.shape
    ns_new = int(ns_new)
    if ns_old <= 0 or ns_new <= 0:
        return np.zeros((max(ns_new, 0), K), dtype=float)
    if ns_old == ns_new:
        return x_old
    if ns_new == 1:
        return x_old[:1]
    if ns_old == 1:
        return np.repeat(x_old[:1], ns_new, axis=0)

    m = np.asarray(m, dtype=int)
    if m.shape != (K,):
        raise ValueError("m shape mismatch")

    scal_old = _scalxc_ref(ns=ns_old, m=m)
    scal_new = _scalxc_ref(ns=ns_new, m=m)

    x_scaled = x_old * scal_old
    odd = (m % 2) == 1
    if ns_old >= 3:
        x_scaled[0, odd] = 2.0 * x_scaled[1, odd] - x_scaled[2, odd]

    out = np.zeros((ns_new, K), dtype=float)
    hs_old = 1.0 / float(ns_old - 1)
    for js in range(ns_new):
        sj = float(js) / float(ns_new - 1)
        js1 = int((js * (ns_old - 1)) // (ns_new - 1))
        js2 = min(js1 + 1, ns_old - 1)
        s1 = float(js1) * hs_old
        xint = (sj - s1) / hs_old
        xint = min(1.0, max(0.0, xint))
        out[js] = ((1.0 - xint) * x_scaled[js1] + xint * x_scaled[js2]) / scal_new[js]

    out[0, odd] = 0.0
    return out


def test_interp_vmec_radial_coeffs_matches_reference():
    rng = np.random.default_rng(0)
    ns_old = 11
    ns_new = 23
    K = 37
    m = rng.integers(0, 9, size=(K,), dtype=np.int32)
    x_old = rng.standard_normal((ns_old, K))

    ref = _interp_ref(x_old, m=m, ns_new=ns_new)
    out = np.asarray(interp_vmec_radial_coeffs(x_old, m=m, ns_new=ns_new))

    np.testing.assert_allclose(out, ref, rtol=0.0, atol=1e-14)


def test_interp_preserves_boundary_and_zeros_odd_axis():
    rng = np.random.default_rng(1)
    ns_old = 9
    ns_new = 17
    K = 20
    m = rng.integers(0, 7, size=(K,), dtype=np.int32)
    x_old = rng.standard_normal((ns_old, K))

    out = np.asarray(interp_vmec_radial_coeffs(x_old, m=m, ns_new=ns_new))
    np.testing.assert_allclose(out[-1], x_old[-1], rtol=0.0, atol=1e-14)

    odd = (m.astype(int) % 2) == 1
    np.testing.assert_allclose(out[0, odd], 0.0, rtol=0.0, atol=0.0)

