"""A/B equivalence tests: new ``vmec_jax.core`` vs the legacy (parity-proven) kernels.

Old implementations under test (left untouched):

- ``vmec_jax.kernels.tomnsp.vmec_trig_tables``  (fixaray.f tables)
- ``vmec_jax.kernels.tomnsp.tomnsps_rzl`` / ``tomnspa_rzl``  (tomnsp_mod.f)
- ``vmec_jax.kernels.realspace.vmec_realspace_synthesis*`` / ``vmec_realspace_analysis``
  (totzsp_mod.f-equivalent synthesis on the VMEC internal grid)

New implementations:

- ``vmec_jax.core.fourier``  (Resolution / mode_table / trig_tables / angle_grids)
- ``vmec_jax.core.transforms``  (fourier_to_real / real_to_fourier / tomnsps /
  tomnspa / symforce_split)
"""

from __future__ import annotations

import os

# Pin the legacy kernels to their CPU DFT lane (deterministic A/B reference).
os.environ.setdefault("VMEC_JAX_TOMNSPS_FFT", "0")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from vmec_jax.kernels.realspace import (
    vmec_realspace_analysis,
    vmec_realspace_synthesis,
    vmec_realspace_synthesis_dtheta,
    vmec_realspace_synthesis_dzeta_phys,
)
from vmec_jax.kernels.tomnsp import (
    tomnspa_rzl,
    tomnsps_rzl,
    vmec_angle_grid,
    vmec_trig_tables,
)
from vmec_jax.modes import vmec_mode_table

from vmec_jax.core.fourier import Resolution, angle_grids, mode_table, trig_tables
from vmec_jax.core.transforms import (
    fourier_to_real,
    real_to_fourier,
    symforce_split,
    tomnspa,
    tomnsps,
)

# (mpol, ntor, ntheta, nzeta, nfp, lasym)
CASES = [
    (6, 0, 18, 1, 1, False),
    (6, 3, 22, 16, 3, False),
    (9, 6, 28, 24, 4, False),
    (5, 2, 20, 18, 2, True),
]
CASE_IDS = ["axisym", "nfp3", "nfp4", "nfp2-lasym"]
NS = 7

RTOL = 1e-12
ATOL = 1e-13


def _resolution(case) -> Resolution:
    mpol, ntor, ntheta, nzeta, nfp, lasym = case
    return Resolution(
        mpol=mpol, ntor=ntor, ntheta=ntheta, nzeta=nzeta, nfp=nfp, lasym=lasym, ns=NS
    )


def _old_trig(case):
    # The legacy tables are Fortran-faithful since the fixaray.f lasym dnorm
    # fix landed in vmec_jax/kernels/tomnsp.py (matching the core fix in
    # vmec_jax/core/fourier.py); no rescaling compensation is needed.
    mpol, ntor, ntheta, nzeta, nfp, lasym = case
    return vmec_trig_tables(
        ntheta=ntheta, nzeta=nzeta, nfp=nfp, mmax=mpol - 1, nmax=ntor, lasym=lasym,
    )


def _seed(case) -> int:
    return abs(hash(case)) % (2**31)


# ---------------------------------------------------------------------------
# Trig tables, mode table, angle grids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_trig_tables_match_old(case):
    res = _resolution(case)
    new = trig_tables(res)
    old = _old_trig(case)

    assert new.ntheta1 == old.ntheta1
    assert new.ntheta2 == old.ntheta2
    assert new.ntheta3 == old.ntheta3
    assert new.dnorm == pytest.approx(old.dnorm, abs=0.0)
    assert new.dnorm3 == pytest.approx(old.dnorm3, abs=0.0)

    pairs = [
        ("mscale", old.mscale),
        ("nscale", old.nscale),
        ("cosmu", old.cosmu),
        ("sinmu", old.sinmu),
        ("cosmum", old.cosmum),
        ("sinmum", old.sinmum),
        ("cosmui", old.cosmui),
        ("sinmui", old.sinmui),
        ("cosmumi", old.cosmumi),
        ("sinmumi", old.sinmumi),
        ("cosmui3", old.cosmui3),
        ("cosmumi3", old.cosmumi3),
        ("cosnv", old.cosnv),
        ("sinnv", old.sinnv),
        ("cosnvn", old.cosnvn),
        ("sinnvn", old.sinnvn),
    ]
    for name, old_table in pairs:
        np.testing.assert_allclose(
            np.asarray(getattr(new, name)),
            np.asarray(old_table),
            rtol=0.0,
            atol=1e-15,
            err_msg=f"trig table {name!r} mismatch",
        )
    # Integration weights: old code stores them with a leading radial axis.
    np.testing.assert_allclose(
        np.asarray(new.wint),
        np.asarray(old.wint3_precond)[0],
        rtol=0.0,
        atol=1e-15,
        err_msg="wint mismatch",
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_mode_table_and_sizes_match_old(case):
    mpol, ntor, *_ = case
    res = _resolution(case)
    new = mode_table(mpol, ntor)
    old = vmec_mode_table(mpol, ntor)
    np.testing.assert_array_equal(new.m, old.m)
    np.testing.assert_array_equal(new.n, old.n)
    assert new.mnmax == old.K == res.mnmax
    np.testing.assert_array_equal(new.m_is_even, (old.m % 2) == 0)
    np.testing.assert_array_equal(new.m_is_zero, old.m == 0)
    np.testing.assert_array_equal(new.m_is_one, old.m == 1)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_angle_grids_match_old(case):
    _, _, ntheta, nzeta, nfp, lasym = case
    res = _resolution(case)
    theta, zeta = angle_grids(res)
    old = vmec_angle_grid(ntheta=ntheta, nzeta=nzeta, nfp=nfp, lasym=lasym)
    np.testing.assert_allclose(theta, np.asarray(old.theta), rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(zeta, np.asarray(old.zeta), rtol=0.0, atol=1e-15)
    assert res.nznt == res.nzeta * res.ntheta3


# ---------------------------------------------------------------------------
# Fourier -> real (totzsps/totzspa equivalents)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("internal", [False, True], ids=["physical", "internal"])
def test_fourier_to_real_matches_old(case, internal):
    mpol, ntor, *_ = case
    res = _resolution(case)
    new_trig = trig_tables(res)
    old_trig = _old_trig(case)
    new_modes = mode_table(mpol, ntor)
    old_modes = vmec_mode_table(mpol, ntor)

    rng = np.random.default_rng(_seed(case))
    coeff_cos = rng.standard_normal((NS, new_modes.mnmax))
    coeff_sin = rng.standard_normal((NS, new_modes.mnmax))

    value, dtheta, dzeta = fourier_to_real(
        coeff_cos,
        coeff_sin,
        modes=new_modes,
        trig=new_trig,
        internal_coeffs=internal,
    )
    kwargs = dict(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=old_modes,
        trig=old_trig,
        coeffs_internal=internal,
    )
    old_value = vmec_realspace_synthesis(**kwargs)
    old_dtheta = vmec_realspace_synthesis_dtheta(**kwargs)
    old_dzeta = vmec_realspace_synthesis_dzeta_phys(**kwargs)

    np.testing.assert_allclose(np.asarray(value), np.asarray(old_value), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(dtheta), np.asarray(old_dtheta), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(dzeta), np.asarray(old_dzeta), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_fourier_to_real_odd_m_sqrt_s_matches_old(case):
    """The odd-m 1/sqrt(s) (profil3d.f scalxc) lane matches the legacy kernel."""
    mpol, ntor, *_ = case
    res = _resolution(case)
    new_trig = trig_tables(res)
    old_trig = _old_trig(case)
    new_modes = mode_table(mpol, ntor)
    old_modes = vmec_mode_table(mpol, ntor)

    rng = np.random.default_rng(_seed(case) + 1)
    coeff_cos = rng.standard_normal((NS, new_modes.mnmax))
    coeff_sin = rng.standard_normal((NS, new_modes.mnmax))
    s = np.linspace(0.0, 1.0, NS)

    (value,) = fourier_to_real(
        coeff_cos,
        coeff_sin,
        modes=new_modes,
        trig=new_trig,
        derivatives=("value",),
        internal_coeffs=True,
        odd_m_sqrt_s=True,
        s=s,
    )
    old_value = vmec_realspace_synthesis(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=old_modes,
        trig=old_trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )
    np.testing.assert_allclose(np.asarray(value), np.asarray(old_value), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("parity", ["both", "cos", "sin"])
def test_real_to_fourier_matches_old(case, parity):
    mpol, ntor, *_ = case
    res = _resolution(case)
    new_trig = trig_tables(res)
    old_trig = _old_trig(case)
    new_modes = mode_table(mpol, ntor)
    old_modes = vmec_mode_table(mpol, ntor)

    rng = np.random.default_rng(_seed(case) + 2)
    field = rng.standard_normal((NS, res.ntheta3, res.nzeta))

    new_cos, new_sin = real_to_fourier(field, modes=new_modes, trig=new_trig, parity=parity)
    old_cos, old_sin = vmec_realspace_analysis(
        f=field, modes=old_modes, trig=old_trig, parity=parity
    )
    np.testing.assert_allclose(np.asarray(new_cos), np.asarray(old_cos), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(new_sin), np.asarray(old_sin), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_synthesis_analysis_roundtrip(case):
    """real_to_fourier(fourier_to_real(x)) == x on band-limited data.

    On the reduced symmetric theta grid the cos and sin families are each
    internally orthogonal (but not mutually), so the round trip is exact per
    parity block; on the full lasym grid it is exact for both blocks jointly.
    """
    mpol, ntor, *_ = case
    lasym = case[5]
    res = _resolution(case)
    trig = trig_tables(res)
    modes = mode_table(mpol, ntor)

    rng = np.random.default_rng(_seed(case) + 3)
    coeff_cos = rng.standard_normal((NS, modes.mnmax))
    coeff_sin = rng.standard_normal((NS, modes.mnmax))
    # sin(m*theta - n*zeta) is identically zero for (m, n) = (0, 0).
    coeff_sin[:, (modes.m == 0) & (modes.n == 0)] = 0.0

    if lasym:
        (field,) = fourier_to_real(
            coeff_cos, coeff_sin, modes=modes, trig=trig, derivatives=("value",)
        )
        back_cos, back_sin = real_to_fourier(field, modes=modes, trig=trig, parity="both")
        np.testing.assert_allclose(np.asarray(back_cos), coeff_cos, rtol=RTOL, atol=1e-12)
        np.testing.assert_allclose(np.asarray(back_sin), coeff_sin, rtol=RTOL, atol=1e-12)
    else:
        zero = np.zeros_like(coeff_cos)
        (field_cos,) = fourier_to_real(
            coeff_cos, zero, modes=modes, trig=trig, derivatives=("value",)
        )
        back_cos, back_sin = real_to_fourier(field_cos, modes=modes, trig=trig, parity="cos")
        np.testing.assert_allclose(np.asarray(back_cos), coeff_cos, rtol=RTOL, atol=1e-12)
        assert not np.any(np.asarray(back_sin))

        (field_sin,) = fourier_to_real(
            zero, coeff_sin, modes=modes, trig=trig, derivatives=("value",)
        )
        back_cos, back_sin = real_to_fourier(field_sin, modes=modes, trig=trig, parity="sin")
        np.testing.assert_allclose(np.asarray(back_sin), coeff_sin, rtol=RTOL, atol=1e-12)
        assert not np.any(np.asarray(back_cos))


# ---------------------------------------------------------------------------
# Real -> Fourier force projections (tomnsps/tomnspa)
# ---------------------------------------------------------------------------


_CHANNELS = (
    ("force_R", "armn"),
    ("force_R_du", "brmn"),
    ("force_R_dv", "crmn"),
    ("force_Z", "azmn"),
    ("force_Z_du", "bzmn"),
    ("force_Z_dv", "czmn"),
    ("force_lambda_du", "blmn"),
    ("force_lambda_dv", "clmn"),
    ("constraint_R", "arcon"),
    ("constraint_Z", "azcon"),
)


def _random_force_kernels(case, seed_offset: int):
    res = _resolution(case)
    rng = np.random.default_rng(_seed(case) + seed_offset)
    new_kwargs = {}
    old_kwargs = {}
    for new_name, old_name in _CHANNELS:
        for parity in ("even", "odd"):
            data = rng.standard_normal((NS, res.ntheta3, res.nzeta))
            new_kwargs[f"{new_name}_{parity}"] = data
            old_kwargs[f"{old_name}_{parity}"] = data
    return new_kwargs, old_kwargs


_SYM_BLOCKS = [
    ("force_R_cc", "frcc"),
    ("force_R_ss", "frss"),
    ("force_Z_sc", "fzsc"),
    ("force_Z_cs", "fzcs"),
    ("force_lambda_sc", "flsc"),
    ("force_lambda_cs", "flcs"),
]
_ASYM_BLOCKS = [
    ("force_R_sc", "frsc"),
    ("force_R_cs", "frcs"),
    ("force_Z_cc", "fzcc"),
    ("force_Z_ss", "fzss"),
    ("force_lambda_cc", "flcc"),
    ("force_lambda_ss", "flss"),
]


def _assert_blocks_match(new_out, old_out, blocks, *, lthreed: bool):
    for new_name, old_name in blocks:
        new_block = getattr(new_out, new_name)
        old_block = getattr(old_out, old_name)
        if not lthreed and new_name.endswith(("_ss", "_cs")):
            assert new_block is None, f"{new_name} should be None for ntor=0"
            assert old_block is None, f"{old_name} should be None for ntor=0"
            continue
        assert new_block is not None, f"{new_name} missing"
        assert old_block is not None, f"{old_name} missing"
        np.testing.assert_allclose(
            np.asarray(new_block),
            np.asarray(old_block),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"block {new_name} ({old_name}) mismatch",
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("include_edge", [False, True], ids=["noedge", "edge"])
def test_tomnsps_matches_old(case, include_edge):
    mpol, ntor, _, _, nfp, lasym = case
    res = _resolution(case)
    new_trig = trig_tables(res)
    old_trig = _old_trig(case)
    new_kwargs, old_kwargs = _random_force_kernels(case, seed_offset=4)

    new_out = tomnsps(
        **new_kwargs, mpol=mpol, ntor=ntor, trig=new_trig, include_edge=include_edge
    )
    old_out = tomnsps_rzl(
        **old_kwargs,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        lasym=lasym,
        trig=old_trig,
        include_edge=include_edge,
        masks=None,
    )
    _assert_blocks_match(new_out, old_out, _SYM_BLOCKS, lthreed=ntor > 0)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("include_edge", [False, True], ids=["noedge", "edge"])
def test_tomnspa_matches_old(case, include_edge):
    mpol, ntor, _, _, nfp, lasym = case
    res = _resolution(case)
    new_trig = trig_tables(res)
    old_trig = _old_trig(case)
    new_kwargs, old_kwargs = _random_force_kernels(case, seed_offset=5)

    new_out = tomnspa(
        **new_kwargs, mpol=mpol, ntor=ntor, trig=new_trig, include_edge=include_edge
    )
    old_out = tomnspa_rzl(
        **old_kwargs,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        lasym=lasym,
        trig=old_trig,
        include_edge=include_edge,
        masks=None,
    )
    _assert_blocks_match(new_out, old_out, _ASYM_BLOCKS, lthreed=ntor > 0)


# ---------------------------------------------------------------------------
# tomnsp( totzsp(x) ) round trip
# ---------------------------------------------------------------------------


def _rz_mask(ns: int, mpol: int) -> np.ndarray:
    """Expected R/Z radial mask (jmin2 start indices, edge excluded)."""
    js = np.arange(ns) + 1
    m = np.arange(mpol)
    jmin2 = np.where(m == 0, 1, 2)[None, :]
    mask = (js[:, None] >= jmin2) & (js[:, None] <= ns - 1)
    return mask.astype(float)[:, :, None]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_tomnsps_recovers_band_limited_coefficients(case):
    """tomnsps is the exact inverse of the totzsps synthesis on band-limited data.

    Feed ``tomnsps`` a pure geometry-style field through the undifferentiated
    kernels (``armn``/``azmn``) built from internal coefficients ``x`` with the
    scaled trig tables.  The mscale/nscale normalization makes the projection
    recover ``x`` exactly (factor 1) in both symmetry modes: for
    ``lasym=True`` the reduced-interval integration carries the fixaray.f
    weight ``dnorm = 1/(nzeta*(ntheta2-1)) = 2/(nzeta*ntheta1)`` with
    endpoint half-weights, which equals the full-grid average for the
    reflection-symmetric basis products fed here.  (Before the core lasym
    dnorm fix this recovered ``x/2`` — the inherited legacy defect that
    halved every lasym force projection.)
    """
    mpol, ntor, _, _, _, lasym = case
    res = _resolution(case)
    trig = trig_tables(res)

    rng = np.random.default_rng(_seed(case) + 6)
    m_grid = np.arange(mpol)[:, None]
    n_grid = np.arange(ntor + 1)[None, :]

    coeff_R_cc = rng.standard_normal((NS, mpol, ntor + 1))
    coeff_R_ss = rng.standard_normal((NS, mpol, ntor + 1))
    coeff_R_ss *= (m_grid > 0) & (n_grid > 0)  # sin*sin basis vanishes otherwise
    coeff_Z_sc = rng.standard_normal((NS, mpol, ntor + 1))
    coeff_Z_sc *= m_grid > 0
    coeff_Z_cs = rng.standard_normal((NS, mpol, ntor + 1))
    coeff_Z_cs *= n_grid > 0

    cosmu, sinmu = np.asarray(trig.cosmu), np.asarray(trig.sinmu)
    cosnv, sinnv = np.asarray(trig.cosnv), np.asarray(trig.sinnv)
    field_R = np.einsum("smn,im,kn->sik", coeff_R_cc, cosmu, cosnv) + np.einsum(
        "smn,im,kn->sik", coeff_R_ss, sinmu, sinnv
    )
    field_Z = np.einsum("smn,im,kn->sik", coeff_Z_sc, sinmu, cosnv) + np.einsum(
        "smn,im,kn->sik", coeff_Z_cs, cosmu, sinnv
    )

    zeros = np.zeros_like(field_R)
    out = tomnsps(
        force_R_even=field_R,
        force_R_odd=field_R,
        force_R_du_even=zeros,
        force_R_du_odd=zeros,
        force_R_dv_even=zeros,
        force_R_dv_odd=zeros,
        force_Z_even=field_Z,
        force_Z_odd=field_Z,
        force_Z_du_even=zeros,
        force_Z_du_odd=zeros,
        force_Z_dv_even=zeros,
        force_Z_dv_odd=zeros,
        mpol=mpol,
        ntor=ntor,
        trig=trig,
        include_edge=False,
    )

    # Known normalization: exact recovery in both symmetry modes (fixaray.f
    # dnorm; see the docstring).
    factor = 1.0
    mask = _rz_mask(NS, mpol)

    np.testing.assert_allclose(
        np.asarray(out.force_R_cc), factor * coeff_R_cc * mask, rtol=RTOL, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(out.force_Z_sc), factor * coeff_Z_sc * mask, rtol=RTOL, atol=1e-12
    )
    if ntor > 0:
        np.testing.assert_allclose(
            np.asarray(out.force_R_ss), factor * coeff_R_ss * mask, rtol=RTOL, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(out.force_Z_cs), factor * coeff_Z_cs * mask, rtol=RTOL, atol=1e-12
        )
    else:
        assert out.force_R_ss is None
        assert out.force_Z_cs is None


# ---------------------------------------------------------------------------
# symforce split (symforce.f)
# ---------------------------------------------------------------------------


def test_symforce_split_recovers_pure_parities():
    """A reflection-even field has no antisymmetric part (and vice versa)."""
    case = CASES[3]  # the lasym case: full theta grid
    mpol, ntor, *_ = case
    res = _resolution(case)
    trig = trig_tables(res)
    rng = np.random.default_rng(_seed(case) + 7)

    cosmu, sinmu = np.asarray(trig.cosmu), np.asarray(trig.sinmu)
    cosnv, sinnv = np.asarray(trig.cosnv), np.asarray(trig.sinnv)
    coeff = rng.standard_normal((NS, mpol, ntor + 1))

    # Even under (theta, zeta) -> (-theta, -zeta): cos*cos (and sin*sin).
    field_even = np.einsum("smn,im,kn->sik", coeff, cosmu, cosnv)
    sym, asym = symforce_split(field_even, trig=trig, reflect_even=True)
    n_theta2 = trig.ntheta2
    np.testing.assert_allclose(
        np.asarray(sym)[:, :n_theta2], field_even[:, :n_theta2], rtol=RTOL, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(asym), 0.0, atol=1e-12)

    # Odd under the reflection: sin*cos.
    field_odd = np.einsum("smn,im,kn->sik", coeff, sinmu, cosnv)
    sym, asym = symforce_split(field_odd, trig=trig, reflect_even=False)
    np.testing.assert_allclose(
        np.asarray(sym)[:, :n_theta2], field_odd[:, :n_theta2], rtol=RTOL, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(asym), 0.0, atol=1e-12)
