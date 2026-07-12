"""Validation gates for :mod:`vmec_jax.core.bootstrap` (R26.g steps 1-2).

Spec: ``notes_r26g_redl_spec.md`` sections 7-8 — the CI-sized subset:

- analytic ``B0(1 + eps cos theta)`` trapped-fraction model (physics limits);
- V1 formula parity vs a pasted simsopt ``j_dot_B_Redl`` row at Zeff > 1;
- V2/V3 Zenodo cross-check on the precise QA/QH wouts (skips when the local
  Zenodo dataset is absent): <= 1% vs the stored simsopt Redl curves, <= 10%
  RMS vs the verbatim SFINCS arrays (interior), trapped-fraction parity;
- differentiability (double-where guards): finite grads through
  ``compute_trapped_fraction`` and ``j_dot_B_redl``;
- traceable lane vs wout lane agreement on the solovev equilibrium.

Reference arrays were generated offline with simsopt master (2026-07-11,
``RedlGeomVmec`` + ``j_dot_B_Redl``) and are pasted to 8-9 significant
digits, which bounds the achievable fsa_* comparison at ~1e-7 relative.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from vmec_jax.core import bootstrap as bs
from vmec_jax.core import optimize as opt
from vmec_jax.core.input import VmecInput

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
ZENODO = Path("/Users/rogerio/local/"
              "20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current")
BENCH = ZENODO / "calculations" / "20211226-01-sfincs_for_precise_QS_for_Redl_benchmark"
needs_zenodo = pytest.mark.skipif(not BENCH.is_dir(),
                                  reason="local Zenodo bootstrap dataset unavailable")

# ---------------------------------------------------------------------------
# Reference data (pasted; see module docstring)
# ---------------------------------------------------------------------------

S_SFINCS = np.linspace(0.025, 0.975, 39)

# simsopt j_dot_B_Redl on the Zenodo precise QA/QH wouts,
# ne = 4.13e20*(1 - s^5) [1/m^3], Te = Ti = 12e3*(1 - s) [eV], Zeff = 1.
JDOTB_REDL_SIMSOPT = {
    "QA": np.array([
        -1868115.50914418, -2775270.70114074, -3400094.21620049, -3880512.31715286,
        -4272034.67163305, -4605282.25672966, -4896479.78903347, -5157675.93723554,
        -5397094.84478662, -5618707.75979406, -5827400.31924707, -6025929.51261721,
        -6215782.69081739, -6397832.79335228, -6571947.11515539, -6736901.83260687,
        -6891331.25914811, -7032101.26701129, -7155564.84327067, -7257229.20300984,
        -7331885.26186289, -7373664.19662929, -7376108.14533720, -7330926.04206708,
        -7234397.68356614, -7078113.42942039, -6855803.10470368, -6562155.56769031,
        -6193373.61844689, -5747552.60591943, -5225635.72365078, -4632083.87643703,
        -3975865.84387911, -3271621.43868459, -2540943.41497415, -1814240.10476981,
        -1132776.03221678, -552055.86258937, -144539.65649977]),
    "QH": np.array([
        -1114383.94355856, -1373023.69582629, -1542603.30028131, -1676362.30947490,
        -1791000.00783239, -1894271.74679599, -1990675.50944924, -2083236.98537477,
        -2174204.90634980, -2265352.89617580, -2358121.13015272, -2453665.25046156,
        -2552886.08377357, -2656413.57142992, -2764590.32718924, -2877450.68484820,
        -2994683.19684573, -3115588.73291005, -3239052.06331003, -3363502.24519278,
        -3486864.21166135, -3606526.20398868, -3719305.48942384, -3821412.65437074,
        -3908428.21550472, -3975279.56550552, -4018947.98067198, -4028019.17313547,
        -3998307.79747368, -3921706.52977070, -3790517.92599779, -3596760.80035085,
        -3332796.41832887, -2992073.09947603, -2570475.11848448, -2068726.58564085,
        -1497198.45896363, -886259.45170203, -312647.08120789]),
}

# SFINCS drift-kinetic benchmark (Zenodo figure01 script, verbatim).
JDOTB_SFINCS = {
    "QA": np.array([
        -2164875.78234086, -3010997.00425800, -3586912.40439179, -4025873.78974165,
        -4384855.40656673, -4692191.91608418, -4964099.33007648, -5210508.61474677,
        -5442946.68999908, -5657799.82786579, -5856450.57370037, -6055808.19817868,
        -6247562.80014873, -6431841.43078959, -6615361.81912527, -6793994.01503932,
        -6964965.34953497, -7127267.47873969, -7276777.92844458, -7409074.62499181,
        -7518722.07866914, -7599581.37772525, -7644509.67670812, -7645760.36382036,
        -7594037.38147436, -7481588.70786642, -7299166.08742784, -7038404.20002745,
        -6691596.45173419, -6253955.52847633, -5722419.58059673, -5098474.47777983,
        -4390147.20699043, -3612989.71633149, -2793173.34162084, -1967138.17518374,
        -1192903.42248978, -539990.08867700, -115053.37380415]),
    "QH": np.array([
        -1086092.95617750, -1327299.73501589, -1490400.04894085, -1626634.32037339,
        -1736643.64671843, -1836285.33939607, -1935027.30993120, -2024949.13178129,
        -2112581.50178861, -2200196.92359437, -2289400.72956248, -2381072.32897262,
        -2476829.87345286, -2575019.97938908, -2677288.45525839, -2783750.09013764,
        -2894174.68898196, -3007944.74771214, -3123697.37793226, -3240571.57445779,
        -3356384.98579004, -3468756.64908024, -3574785.02500657, -3671007.37469685,
        -3753155.07811322, -3816354.48636373, -3856198.22429860, -3866041.76391937,
        -3839795.40512069, -3770065.26594065, -3649660.76253605, -3471383.50141700,
        -3228174.23182819, -2914278.54799143, -2525391.54652021, -2058913.26485519,
        -1516843.60879267, -912123.39517400, -315980.89711036]),
}

# simsopt compute_trapped_fraction (adaptive quad + spline extrema) at
# S_SFINCS indices [0, 9, 19, 29, 38] on the same wouts.
_TF_IDX = [0, 9, 19, 29, 38]
TRAPPED_FRACTION_SIMSOPT = {
    "QA": {
        "Bmin": np.array([5.80559891, 5.58429171, 5.45092648, 5.34876316, 5.27117014]),
        "Bmax": np.array([6.01231260, 6.23771294, 6.37470656, 6.47978641, 6.56027572]),
        "epsilon": np.array([0.01749156, 0.05527161, 0.07811676, 0.09561808, 0.10895588]),
        "fsa_B2": np.array([34.88941027, 34.65544608, 34.39571806, 34.13689887, 33.90473201]),
        "fsa_1overB": np.array([0.16933717, 0.17025640, 0.17128461, 0.17231836, 0.17325348]),
        "f_t": np.array([0.19356866, 0.34425562, 0.40845759, 0.45085385, 0.48035648]),
    },
    "QH": {
        "Bmin": np.array([5.73151049, 5.42043251, 5.24200669, 5.10865432, 5.00735325]),
        "Bmax": np.array([6.04156548, 6.40498829, 6.64014441, 6.83072542, 6.98268602]),
        "epsilon": np.array([0.02633594, 0.08325757, 0.11766705, 0.14423455, 0.16474781]),
        "fsa_B2": np.array([34.60667062, 34.51123710, 34.39838777, 34.27929551, 34.16726013]),
        "fsa_1overB": np.array([0.17007541, 0.17109505, 0.17225033, 0.17342657, 0.17450143]),
        "f_t": np.array([0.23659462, 0.41517507, 0.48875846, 0.53685600, 0.56995364]),
    },
}

WOUT = {"QA": "wout_new_QA_aScaling.nc", "QH": "wout_new_QH_aScaling.nc"}
HELICITY_N = {"QA": 0, "QH": -1}

PAPER_PROFILES = bs.KineticProfiles(
    ne_coeffs=4.13e20 * np.array([1, 0, 0, 0, 0, -1.0]),
    Te_coeffs=12.0e3 * np.array([1, -1.0]),
    Ti_coeffs=12.0e3 * np.array([1, -1.0]),
)


def _analytic_model(eps, B0=0.8, ntheta=128, nzeta=4):
    """B = B0*(1 + eps*cos(theta)), constant sqrt(g), shape (neps, ntheta, nzeta)."""
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    modB = (B0 * (1.0 + eps[:, None, None] * np.cos(theta)[None, :, None])
            * np.ones((1, 1, nzeta)))
    sqrtg = 1.3 * np.ones_like(modB)
    return jnp.asarray(modB), jnp.asarray(sqrtg)


# ---------------------------------------------------------------------------
# Trapped fraction: analytic model (spec section 8 physics limits)
# ---------------------------------------------------------------------------


def test_trapped_fraction_analytic_model():
    B0 = 0.8
    eps = np.array([0.01, 0.03, 0.1, 0.3])
    modB, sqrtg = _analytic_model(eps, B0=B0)
    Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t = bs.compute_trapped_fraction(modB, sqrtg)
    # Grid extrema are exact for this model (theta = 0, pi are grid points).
    np.testing.assert_allclose(np.asarray(Bmin), B0 * (1 - eps), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(Bmax), B0 * (1 + eps), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(epsilon), eps, rtol=1e-13)
    # <B^2> = B0^2 (1 + eps^2/2); <1/B> = 1/(B0 sqrt(1 - eps^2)) (exact FSAs;
    # the uniform periodic grid integrates both spectrally).
    np.testing.assert_allclose(np.asarray(fsa_B2), B0 ** 2 * (1 + 0.5 * eps ** 2), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(fsa_1overB),
                               1.0 / (B0 * np.sqrt(1 - eps ** 2)), rtol=1e-12)
    # Large-aspect-ratio asymptote f_t ~ 1.46 sqrt(eps) at small eps.
    ratio = np.asarray(f_t)[:2] / (1.46 * np.sqrt(eps[:2]))
    np.testing.assert_allclose(ratio, 1.0, atol=0.05)
    # f_t increases with eps and stays in (0, 1).
    ft = np.asarray(f_t)
    assert np.all(np.diff(ft) > 0) and np.all((ft > 0) & (ft < 1))
    # Fixed-order quadrature self-check: 64 vs 256 nodes.
    *_, f_t_hi = bs.compute_trapped_fraction(modB, sqrtg, n_lambda=256)
    np.testing.assert_allclose(ft, np.asarray(f_t_hi), rtol=1e-4)


def test_trapped_fraction_grad_finite():
    """Reverse-mode AD through the sqrt(1 - lambda*B) double-where guard."""
    modB, sqrtg = _analytic_model([0.1], ntheta=32, nzeta=2)

    def scalar(b):
        return jnp.sum(bs.compute_trapped_fraction(b, sqrtg)[5])

    grad = jax.grad(scalar)(modB)
    assert np.all(np.isfinite(np.asarray(grad)))


# ---------------------------------------------------------------------------
# V1: formula parity vs simsopt at Zeff > 1 (pasted single reference rows)
# ---------------------------------------------------------------------------


def test_j_dot_B_redl_formula_parity_zeff():
    """simsopt j_dot_B_Redl with explicit arrays: ne=4.13e20(1-s^5),
    Te=12e3(1-s), Ti=11e3(1-0.9s), Zeff=1.8+0.2s, helicity_n=0, nfp=2."""
    profiles = bs.KineticProfiles(
        ne_coeffs=4.13e20 * np.array([1, 0, 0, 0, 0, -1.0]),
        Te_coeffs=12.0e3 * np.array([1, -1.0]),
        Ti_coeffs=11.0e3 * np.array([1, -0.9]),
        Zeff_coeffs=np.array([1.8, 0.2]),
    )
    geom = bs.RedlGeometry(
        surfaces=jnp.array([0.3, 0.6]), iota=jnp.array([0.42, 0.44]),
        G=jnp.array([35.1, 34.9]), I=jnp.zeros(2), R=jnp.array([5.92, 5.95]),
        epsilon=jnp.array([0.08, 0.11]), f_t=jnp.array([0.41, 0.48]),
        fsa_B2=jnp.zeros(2), fsa_1overB=jnp.zeros(2),
        Bmin=jnp.zeros(2), Bmax=jnp.zeros(2), psi_edge=jnp.asarray(-8.2), nfp=2)
    jdotB, det = bs.j_dot_B_redl(profiles, geom, 0)
    expected = {
        "nu_e_star": [0.7773562452672003, 1.3094624601213856],
        "nu_i_star": [2.2842024881863416, 3.3405656273652236],
        "L31": [0.2717120099828814, 0.26724483107140384],
        "L32": [0.05313742248893882, 0.09760343266080607],
        "alpha": [0.4543400490751495, 0.814992230099803],
        "jdotB": [-4131811.6745255208, -4884860.735215111],
    }
    for name, ref in expected.items():
        np.testing.assert_allclose(np.asarray(det[name]), ref, rtol=1e-12, err_msg=name)
    np.testing.assert_allclose(np.asarray(jdotB), expected["jdotB"], rtol=1e-12)


# ---------------------------------------------------------------------------
# V2 + V3: Zenodo precise QA/QH cross-check (paper Fig. 1 / figure01)
# ---------------------------------------------------------------------------


@needs_zenodo
@pytest.mark.parametrize("config", ["QA", "QH"])
def test_zenodo_figure01_redl_vs_simsopt_and_sfincs(config):
    geom = bs.redl_geometry_from_wout(BENCH / WOUT[config], S_SFINCS)
    jdotB, _ = bs.j_dot_B_redl(PAPER_PROFILES, geom, HELICITY_N[config])
    jdotB = np.asarray(jdotB)

    # <= 1% vs the stored simsopt Redl curve (same formula, GL-quadrature vs
    # adaptive-quad f_t numerics; observed <= 5.1e-4).
    ref = JDOTB_REDL_SIMSOPT[config]
    np.testing.assert_allclose(jdotB, ref, rtol=1e-2)

    # <= 10% RMS vs the SFINCS drift-kinetic benchmark, interior s in [0.1, 0.9]
    # (paper-reported agreement; observed 5.0% QA / 3.4% QH).
    interior = (S_SFINCS >= 0.1) & (S_SFINCS <= 0.9)
    sfincs = JDOTB_SFINCS[config][interior]
    rel = (jdotB[interior] - sfincs) / np.abs(sfincs)
    assert np.sqrt(np.mean(rel ** 2)) <= 0.10

    # V3 trapped-fraction parity vs simsopt (spline extrema + adaptive quad):
    # f_t <= 0.3%, epsilon <= 0.5% relative; fsa_* limited by the 8-digit
    # pasted references (~1e-7), not by the quadrature (~1e-10 when compared
    # at full precision offline).
    tf = TRAPPED_FRACTION_SIMSOPT[config]
    np.testing.assert_allclose(np.asarray(geom.f_t)[_TF_IDX], tf["f_t"], rtol=3e-3)
    np.testing.assert_allclose(np.asarray(geom.epsilon)[_TF_IDX], tf["epsilon"], rtol=5e-3)
    np.testing.assert_allclose(np.asarray(geom.Bmin)[_TF_IDX], tf["Bmin"], rtol=1e-3)
    np.testing.assert_allclose(np.asarray(geom.Bmax)[_TF_IDX], tf["Bmax"], rtol=1e-3)
    np.testing.assert_allclose(np.asarray(geom.fsa_B2)[_TF_IDX], tf["fsa_B2"], rtol=1e-7)
    np.testing.assert_allclose(np.asarray(geom.fsa_1overB)[_TF_IDX], tf["fsa_1overB"],
                               rtol=1e-6)


# ---------------------------------------------------------------------------
# Physics limits and the isomorphism (spec section 8)
# ---------------------------------------------------------------------------


def _toy_geometry():
    return bs.RedlGeometry(
        surfaces=jnp.array([0.25, 0.5, 0.75]), iota=jnp.array([1.05, 1.1, 1.15]),
        G=jnp.array([35.0, 34.9, 34.8]), I=jnp.array([-0.1, -0.2, -0.3]),
        R=jnp.array([5.9, 5.9, 5.9]), epsilon=jnp.array([0.06, 0.09, 0.12]),
        f_t=jnp.array([0.35, 0.43, 0.5]), fsa_B2=jnp.array([34.0, 34.0, 34.0]),
        fsa_1overB=jnp.array([0.17, 0.17, 0.17]),
        Bmin=jnp.array([5.5, 5.4, 5.3]), Bmax=jnp.array([6.2, 6.3, 6.4]),
        psi_edge=jnp.asarray(-8.2), nfp=4)


def test_flat_profiles_give_zero_jdotB():
    profiles = bs.KineticProfiles(ne_coeffs=4.0e20, Te_coeffs=1.0e4, Ti_coeffs=1.0e4)
    jdotB, _ = bs.j_dot_B_redl(profiles, _toy_geometry(), -1)
    np.testing.assert_array_equal(np.asarray(jdotB), 0.0)


def test_isomorphism_enters_only_through_iota_minus_N():
    """(iota, N = nfp*helicity_n) == (iota - N, 0) manually shifted."""
    geom = _toy_geometry()
    N = geom.nfp * (-1)
    shifted = dataclasses.replace(geom, iota=geom.iota - N)
    j1, _ = bs.j_dot_B_redl(PAPER_PROFILES, geom, -1)
    j2, _ = bs.j_dot_B_redl(PAPER_PROFILES, shifted, 0)
    np.testing.assert_allclose(np.asarray(j1), np.asarray(j2), rtol=1e-14)


# ---------------------------------------------------------------------------
# Differentiability (validation gate 3)
# ---------------------------------------------------------------------------


def test_j_dot_B_redl_grad_finite():
    """grad of a scalar of j_dot_B_redl w.r.t. a geometry input and a profile
    coefficient is finite (exercises the Zeff = 1 sqrt(Zeff - 1) guard)."""
    geom = _toy_geometry()

    def scalar_of_ft(f_t):
        j, _ = bs.j_dot_B_redl(PAPER_PROFILES, dataclasses.replace(geom, f_t=f_t), -1)
        return jnp.sum(j)

    g_ft = jax.grad(scalar_of_ft)(geom.f_t)
    assert np.all(np.isfinite(np.asarray(g_ft)))
    assert np.any(np.asarray(g_ft) != 0.0)

    def scalar_of_n0(n0):
        profiles = bs.KineticProfiles(
            ne_coeffs=n0 * jnp.array([1, 0, 0, 0, 0, -1.0]),
            Te_coeffs=12.0e3 * jnp.array([1, -1.0]),
            Ti_coeffs=12.0e3 * jnp.array([1, -1.0]))
        j, _ = bs.j_dot_B_redl(profiles, geom, -1)
        return jnp.sum(j)

    g_n0 = jax.grad(scalar_of_n0)(4.13e20)
    assert np.isfinite(float(g_n0)) and float(g_n0) != 0.0


def test_full_chain_grad_through_trapped_fraction_finite():
    """grad through compute_trapped_fraction -> j_dot_B_redl w.r.t. |B|."""
    modB, sqrtg = _analytic_model([0.1], B0=5.7, ntheta=32, nzeta=2)

    def scalar(b):
        Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t = bs.compute_trapped_fraction(b, sqrtg)
        iota = jnp.array([0.42])
        G = jnp.array([35.0])
        I = jnp.array([-0.1])  # noqa: E741 - Boozer I
        geom = bs.RedlGeometry(
            surfaces=jnp.array([0.5]), iota=iota, G=G, I=I,
            R=(G + iota * I) * fsa_1overB, epsilon=epsilon, f_t=f_t,
            fsa_B2=fsa_B2, fsa_1overB=fsa_1overB, Bmin=Bmin, Bmax=Bmax,
            psi_edge=jnp.asarray(-8.2), nfp=2)
        j, _ = bs.j_dot_B_redl(PAPER_PROFILES, geom, 0)
        return jnp.sum(j)

    grad = jax.grad(scalar)(modB)
    arr = np.asarray(grad)
    assert np.all(np.isfinite(arr)) and np.any(arr != 0.0)


# ---------------------------------------------------------------------------
# Step 2: traceable lane vs wout lane (solovev, small)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eq():
    equilibrium = opt.solve_equilibrium(VmecInput.from_file(DATA_DIR / "input.solovev"))
    assert equilibrium.result.converged
    return equilibrium


def test_state_lane_matches_wout_lane(eq):
    """redl_geometry_from_state agrees with redl_geometry_from_wout at
    discretization level (solver internal grid vs 64x65 wout synthesis)."""
    surfaces = np.array([0.3, 0.5, 0.7])
    g_state = bs.redl_geometry_from_state(eq.state, eq.runtime, surfaces=surfaces)
    g_wout = bs.redl_geometry_from_wout(eq.wout, surfaces)
    assert g_state.nfp == g_wout.nfp
    for name, rtol in [("iota", 1e-8), ("G", 1e-8), ("I", 2e-2), ("psi_edge", 1e-8),
                       ("fsa_B2", 2e-2), ("fsa_1overB", 2e-2), ("epsilon", 2e-2),
                       ("f_t", 2e-2), ("R", 2e-2)]:
        a = np.asarray(getattr(g_state, name))
        b = np.asarray(getattr(g_wout, name))
        np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-4, err_msg=name)
    # Tokamak: the Redl <J.B> from both lanes agrees at the same level.
    j_state, _ = bs.j_dot_B_redl(PAPER_PROFILES, g_state, 0)
    j_wout, _ = bs.j_dot_B_redl(PAPER_PROFILES, g_wout, 0)
    assert np.all(np.isfinite(np.asarray(j_state)))
    np.testing.assert_allclose(np.asarray(j_state), np.asarray(j_wout), rtol=5e-2)


def test_state_lane_default_surfaces_and_pytree(eq):
    geom = bs.redl_geometry_from_state(eq.state, eq.runtime)
    assert geom.surfaces.shape == (16,)
    leaves = jax.tree_util.tree_leaves(geom)
    assert leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    assert 0.0 < float(np.min(np.asarray(geom.f_t))) < float(np.max(np.asarray(geom.f_t))) < 1.0
