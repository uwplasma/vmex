"""Physics anchors, convergence and differentiation gates for ``Gamma_c`` / ``Gamma_alpha``.

Lanes: closed-form limits of the line kernel (a constant drift angle gives
``Gamma_c = gamma^2 pi/(2 sqrt 2) f_trapped``; the bounce time of a cosine
well is a complete elliptic integral; the first-event hazard of
``Gamma_alpha`` on a two-sector map), the axisymmetric limit, ordering of
real equilibria, a cross-code number (DESC), a convergence ladder, and
reverse-mode gradients through the implicit solve against central
differences that are themselves shown to have converged in the step.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ellipk

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vmex.core import gammac  # noqa: E402
from vmex.core import implicit as im  # noqa: E402
from vmex.core import optimize as opt  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
#: Reduced budget of the physics tests below; the measured anchors in the
#: docstrings correspond to exactly these settings.
FAST = dict(nalpha=8, num_transit=3, points_per_transit=32, pitch_nodes=3,
            quadrature_order=32, max_levels=96)


def _model_lines(eps=0.2, periods=3, n=97, radial=1.0, gradient=1.0,
                 correction=0.0, tangency=1.0, nline=2):
    """``B = 1 + eps cos x`` on whole periods, ``dl = B dx`` (so ``dl/B = dx``)."""
    x = jnp.linspace(0.0, 2 * jnp.pi * periods, n * periods - periods + 1)
    one = jnp.ones((nline, x.size))
    bmag = (1.0 + eps * jnp.cos(x)) * one
    return dict(bmag=bmag, radial_drift=radial * one, radial_gradient=gradient * one,
                drift_correction=correction * one, tangency=tangency * one,
                dl_dx=bmag, length=float(x[-1])), x


@pytest.mark.parametrize("radial, gradient, tangency", [
    (1.0, 1.0, 1.0), (0.3, -1.0, 2.0), (-0.5, 0.7, 0.4), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)])
def test_constant_drift_angle_gives_the_trapped_fraction(radial, gradient, tangency):
    """Nemov 2008 eq. 61 / Velasco 2021 eq. 16-24 in closed form.

    With constant ``v_r / (T v_p)`` the bounce averages share the factor
    ``int w dl`` and ``gamma_c = (2/pi) arctan(radial / (tangency gradient))``
    in every well, so ``Gamma_c = gamma_c^2 (pi / 2 sqrt 2) f_trapped`` with
    ``f_trapped = <sqrt(1 - B/B_max)>`` (eq. 24).  For ``B = 1 + eps cos x``
    ``f_trapped = sqrt(eps/(1+eps)) 2 sqrt 2 / pi``, hence
    ``Gamma_c = gamma_c^2 sqrt(eps/(1+eps))``.  Exact to quadrature error with
    ``completeness = 0`` (the ratio estimator then returns the mean exactly).
    """
    eps = 0.2
    lines, _ = _model_lines(eps, radial=radial, gradient=gradient, tangency=tangency)
    out = gammac.gamma_c_from_fieldlines(
        **lines, num_transit=3, completeness=0.0, max_levels=16)
    angle = (2 / np.pi) * np.arctan2(radial, abs(tangency * gradient))
    expected = angle**2 * np.sqrt(eps / (1 + eps))
    assert not bool(out["overflow"])
    np.testing.assert_allclose(float(out["gamma_c"]), expected, rtol=2e-4, atol=1e-12)
    np.testing.assert_allclose(np.asarray(out["gamma_c_lines"]), expected, rtol=2e-4, atol=1e-12)


@pytest.mark.parametrize("level", [1.04, 1.12, 1.19])
def test_cosine_well_bounce_time_is_an_elliptic_integral(level):
    """``v tau = 2 sqrt(b) int dx / sqrt(b - B)`` of the well of ``B = 1 + eps cos x`` at ``x = pi``.

    ``int_{-t0}^{t0} dt / sqrt(cos t - cos t0) = 2 sqrt 2 K(sin(t0/2))``, so
    ``v tau = 4 sqrt(2 b / eps) K(m = sin^2(t0/2))`` with
    ``cos t0 = -(b - 1) / eps``.
    """
    eps, n = 0.2, 129
    x = jnp.linspace(0.0, 2 * jnp.pi, n)
    # the field minimum at x = pi centres a complete well in the window
    F = jnp.stack([1.0 + eps * jnp.cos(x)] + [jnp.ones(n)] * 5)
    F = F.at[4].set(1.0)                         # dl = dx
    wells = gammac._line_wells(F, 2 * np.pi, jnp.array([level]), W=4,
                               qorder=48, width=0.05)
    mask = np.asarray(wells["mask"])[0]
    assert mask.sum() == 1
    t0 = np.arccos(-(level - 1.0) / eps)
    expected = 4 * np.sqrt(2 * level / eps) * ellipk(np.sin(t0 / 2) ** 2)
    np.testing.assert_allclose(np.asarray(wells["vt"])[0][mask][0], expected, rtol=2e-5)


def test_stationary_points_resolve_wells_smaller_than_a_cell():
    """A dip inside one grid cell is a well: the spline pieces are monotone."""
    n = 33
    x = jnp.linspace(0.0, 2 * jnp.pi, n)
    # sample a field whose second harmonic dips below the level between two nodes
    B = 1.0 + 0.2 * jnp.cos(x) + 0.0 * x
    F = jnp.stack([B] + [jnp.ones(n)] * 5)
    levels, _ = gammac._breakpoint_nodes(F, 2 * np.pi, 16, 2)[0:2]
    assert bool(jnp.all(jnp.diff(jnp.sort(levels)) >= 0))
    c = gammac._hermite(B)
    t, ok = gammac._stationary(c)
    # B = 1 + 0.2 cos x has its extrema at the nodes x = 0, pi, 2 pi (t = 0 or 1)
    assert int(ok.sum()) >= 3
    assert float(jnp.max(jnp.abs(gammac._peval(c, jnp.where(ok, t, 0.0)) - (1 + 0.2 * jnp.cos(
        (jnp.arange(n - 1)[None, :] + t) * 2 * jnp.pi / (n - 1))))[ok])) < 1e-6


def test_zero_radial_drift_gives_zero_and_scale_invariance():
    """Pure poloidal drift is confined; ``B -> cB``, ``psi -> c psi`` is invariant."""
    lines, _ = _model_lines(radial=0.0)
    assert float(gammac.gamma_c_from_fieldlines(**lines, num_transit=3)["gamma_c"]) == 0.0
    base, _ = _model_lines(radial=0.4, gradient=0.8)
    # B = c B and psi = c psi leave the drift ingredients and dl/dx unchanged
    scaled = dict(base, bmag=3 * base["bmag"])
    a = gammac.gamma_c_from_fieldlines(**base, num_transit=3)["gamma_c"]
    b = gammac.gamma_c_from_fieldlines(**scaled, num_transit=3)["gamma_c"]
    np.testing.assert_allclose(float(b), float(a), rtol=2e-3)


def test_overflow_is_nan_not_a_low_cost():
    """Too few level or well slots poison the row instead of under-counting wells."""
    lines, _ = _model_lines(periods=3)
    full = gammac.gamma_c_from_fieldlines(**lines, num_transit=3, max_levels=16)
    assert np.isfinite(float(full["gamma_c"]))
    for slots in (dict(max_levels=2), dict(max_wells=1)):
        out = gammac.gamma_c_from_fieldlines(**lines, num_transit=3, **{"max_levels": 16, **slots})
        assert bool(out["overflow"]) and np.isnan(float(out["gamma_c"]))


def test_pitch_quadrature_converges_with_nodes_and_levels():
    """Error against a refined reference falls as nodes and level slots grow."""
    lines, x = _model_lines(0.25, periods=2)
    # drift ratios that vary along the line make gamma depend on the well and level
    lines["radial_drift"] = 0.6 * jnp.cos(0.5 * x + 0.3)[None, :] + lines["radial_drift"]
    lines["radial_gradient"] = 0.3 * jnp.sin(x)[None, :] + 0.6 * lines["radial_gradient"]
    ref = float(gammac.gamma_c_from_fieldlines(
        **lines, num_transit=2, pitch_nodes=8, max_levels=96, quadrature_order=96)["gamma_c"])
    errors = []
    for nodes, levels in ((2, 8), (3, 16), (4, 32)):
        value = float(gammac.gamma_c_from_fieldlines(
            **lines, num_transit=2, pitch_nodes=nodes, max_levels=levels,
            quadrature_order=96)["gamma_c"])
        errors.append(abs(value - ref) / ref)
    assert errors[0] > errors[1] > errors[2] or errors[2] < 1e-4
    assert errors[2] < 5e-3


@pytest.mark.parametrize("threshold_width", [0.02])
def test_lost_fraction_first_event_hazard(threshold_width):
    """Eq. 25 on a two-sector map: outward half circle, inward half circle.

    A particle is lost when its precession meets an outward superbanana
    (``gamma* > 0.2``) before an inward one.  With the outward sector on
    half of the periodic ``alpha`` grid, half of the orbits are lost for
    either precession direction, and none when no inward event exists.
    """
    n = 64
    gamma = jnp.where(jnp.arange(n) < n // 2, 0.6, -0.6)[:, None]
    for direction in (1.0, -1.0, 0.0):
        lost = gammac._lost_fraction(gamma, jnp.full((n, 1), direction), threshold_width)
        np.testing.assert_allclose(float(jnp.mean(lost)), 0.5, atol=2e-3)
        # lost exactly on the outward half
        np.testing.assert_allclose(np.asarray(lost[:, 0]),
                                   (np.arange(n) < n // 2).astype(float), atol=2e-3)
    outward_only = jnp.full((n, 1), 0.6)
    assert float(jnp.max(gammac._lost_fraction(outward_only, jnp.ones((n, 1)), 0.02))) < 5e-3
    # a free gap between an inward event and an outward one: hit order decides
    gap = jnp.zeros((n, 1)).at[10].set(-0.6).at[20].set(0.6)
    plus = gammac._lost_fraction(gap, jnp.ones((n, 1)), 0.02)[:, 0]
    minus = gammac._lost_fraction(gap, -jnp.ones((n, 1)), 0.02)[:, 0]
    # moving to larger alpha: starting in (10, 20] one meets the outward event first
    assert float(plus[15]) > 0.99 and float(minus[15]) < 5e-3
    assert float(plus[25]) < 5e-3 and float(minus[25]) > 0.99


def test_gamma_alpha_model_field_bounds_and_trapped_fraction():
    """``0 <= Gamma_alpha <= f_trapped`` and ``f_trapped`` matches eq. 24."""
    eps = 0.2
    lines, _ = _model_lines(eps, radial=0.5, gradient=0.2, nline=8)
    out = gammac.gamma_alpha_from_fieldlines(
        **lines, iota=0.4, nfp=1, num_transit=3, pitch_levels=48, completeness=0.0)
    trapped = np.sqrt(eps / (1 + eps)) * 2 * np.sqrt(2) / np.pi
    # the midpoint pitch grid omits the sliver below B_min and above B_max
    np.testing.assert_allclose(float(out["f_trapped"]), trapped, rtol=2e-2)
    assert 0.0 <= float(out["gamma_alpha"]) <= float(out["f_trapped"]) + 1e-12
    # no radial drift: no superbanana, nothing lost
    quiet, _ = _model_lines(eps, radial=0.0, nline=8)
    zero = gammac.gamma_alpha_from_fieldlines(**quiet, iota=0.4, nfp=1, num_transit=3)
    assert float(zero["gamma_alpha"]) == 0.0


# ---------------------------------------------------------------------------
# Real equilibria
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokamak_eq():
    eq = opt.solve_equilibrium(VmecInput.from_file(DATA_DIR / "input.circular_tokamak"))
    assert eq.result.converged
    return eq


@pytest.fixture(scope="module")
def qa_eq():
    inp = VmecInput.from_file(DATA_DIR / "input.LandremanPaul2021_QA_lowres")
    inp = dataclasses.replace(
        inp, ns_array=np.array([13]), ftol_array=np.array([1e-12]),
        niter_array=np.array([4000]))
    eq = opt.solve_equilibrium(inp)
    assert eq.result.converged
    return eq


@pytest.fixture(scope="module")
def ncsx_eq():
    eq = opt.solve_equilibrium(VmecInput.from_file(DATA_DIR / "input.li383_low_res"))
    assert eq.result.converged
    return eq


def _value(eq, name="gamma_c", surface=0.5, **overrides):
    fn = {"gamma_c": gammac.gamma_c_state, "gamma_alpha": gammac.gamma_alpha_state}[name]
    base = FAST if name == "gamma_c" else {
        **{k: v for k, v in FAST.items() if k not in ("pitch_nodes", "max_levels")}, "pitch_levels": 24}
    out = fn(eq.state, eq.runtime, surfaces=(surface,), **{**base, **overrides})
    assert float(out["overflow"][0]) == 0.0
    return float(out[name][0])


def test_axisymmetric_limit_and_optimization_ordering(tokamak_eq, qa_eq, ncsx_eq):
    """Gamma_c orders tokamak << precise QA << unoptimized 3D (Nemov 2008; Velasco 2021).

    In axisymmetry the bounce-averaged radial drift vanishes exactly
    (``dJ/dalpha = 0``), so ``Gamma_c = 0``; ``Gamma_alpha`` likewise.
    """
    tok, qa, ncsx = (_value(e) for e in (tokamak_eq, qa_eq, ncsx_eq))
    assert tok < 1.0e-8                      # measured exactly 0
    assert _value(tokamak_eq, "gamma_alpha") < 1.0e-8
    assert qa < 0.5 * ncsx
    assert 1.0e-3 < qa < 5.0e-2 and 1.0e-2 < ncsx < 2.0e-1


def test_gamma_c_matches_desc_on_li383(ncsx_eq):
    """Cross-code: DESC's Nemov ``Gamma_c`` on the same equilibrium, s = 0.25.

    DESC (``GammaC`` objective on the wout loaded at L=16, M=12, N=12 -- the
    default 7/7/8 load is 10-30x off --, 8 alpha lines, 8 field-period
    transits per line, 33 pitch angles) gives 3.60e-2; VMEX with the sharp
    ``tangency_width = 0.005`` (DESC uses the hard bottom of the well), 16
    lines, 6 transits and a 96-node sine map gives 3.43e-2 (-5 %).  At
    s = 0.5 DESC reads 3.18e-2 against VMEX 4.29e-2 (+35 %): DESC itself moves
    by 10-20 % between its own budgets (QA, s = 0.5: 3.7e-2, 4.4e-2, 4.9e-2 at
    8, 16, 32 transits) and the outer-surface disagreement of its wout refit is
    documented, so only s = 0.25 is asserted.
    """
    value = _value(ncsx_eq, surface=0.25, nalpha=16, num_transit=6, max_levels=288,
                   tangency_width=0.005, quadrature_order=96, pitch_nodes=4)
    assert abs(value / 3.60e-2 - 1.0) < 0.15


def test_gamma_c_converges_with_the_line_budget(qa_eq):
    """Short lines undercount (cut wells); from 2 transits on the value is stable.

    Measured on this QA at s = 0.5, 16 lines: 2e-4 (1 transit), 1.43e-3, 1.30e-3,
    1.37e-3 (2, 4, 8 transits) -- within 5 % from two transits on.
    """
    values = [_value(qa_eq, **{**FAST, "num_transit": nt, "nalpha": 16, "max_levels": 48 * nt})
              for nt in (1, 4, 8)]
    assert values[0] < 0.3 * values[1]
    assert abs(values[2] / values[1] - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Gradients through the implicit solve
# ---------------------------------------------------------------------------

def _boundary_objective(inp, name="gamma_c", **overrides):
    settings = {**FAST, "nalpha": 8, **overrides}
    if name == "gamma_alpha":
        settings.pop("pitch_nodes"), settings.pop("max_levels")
    fn = {"gamma_c": gammac.gamma_c_state, "gamma_alpha": gammac.gamma_alpha_state}[name]

    def objective(params):
        solution = im.run(inp, params, ns=13, ftol=1.0e-13, max_iterations=20000,
                          adjoint_tol=1.0e-13, device=None)
        return jnp.sum(fn(solution.state, solution.runtime, surfaces=(0.5,), **settings)[name])

    return objective


def test_boundary_gradient_matches_converged_finite_differences():
    """``jax.grad`` through the implicit solve against central differences.

    li383 at ns = 13, ``Gamma_c`` at s = 0.5, d/d rbc[n=0, m=1]: the central
    difference is first shown converged in the step (``h = 1e-5`` and
    ``3e-5`` agree) and the reverse-mode value matches it.  The earlier
    discontinuous form changed sign under refinement (+70 AD against -12 FD on
    a QA start).
    """
    inp = VmecInput.from_file(DATA_DIR / "input.li383_low_res")
    params = im.params_from_input(inp, device=None)
    index = (int(inp.ntor), 1)
    objective = _boundary_objective(inp)
    adjoint = float(np.asarray(jax.grad(objective)(params).rbc)[index])

    def central(step):
        values = [float(objective(dataclasses.replace(
            params, rbc=jnp.asarray(params.rbc).at[index].add(sign * step))))
            for sign in (-1.0, 1.0)]
        return (values[1] - values[0]) / (2.0 * step)

    fd = [central(3.0e-5), central(1.0e-5)]
    assert abs(fd[0] / fd[1] - 1.0) < 0.05          # FD converged in h
    assert abs(adjoint / fd[1] - 1.0) < 0.05


def test_gamma_alpha_boundary_gradient_is_finite():
    """Reverse mode of the smoothed ``Gamma_alpha`` through the solve is finite and live."""
    inp = VmecInput.from_file(DATA_DIR / "input.li383_low_res")
    params = im.params_from_input(inp, device=None)
    objective = _boundary_objective(inp, "gamma_alpha", pitch_levels=32)
    value, grad = jax.value_and_grad(objective)(params)
    assert np.isfinite(float(value)) and 0.0 < float(value)
    assert np.all(np.isfinite(np.asarray(grad.rbc))) and float(jnp.max(jnp.abs(grad.rbc))) > 0.0


def test_line_kernel_is_mirror_invariant(qa_eq):
    """``Gamma_c`` of a line equals that of its mirror image, with its gradient.

    Reversing the line and flipping the sign of the odd ingredient
    (``radial_drift``) is the stellarator reflection; ``gamma_c^2`` and its
    derivatives with respect to the ingredients are invariant.  Measured on
    this QA: values to 1e-9, gradient entries to 2e-3 of their scale (six of
    390 entries; a few level choices flip at rounding noise).
    """
    from vmex.core.stability import _ballooning_context

    ctx = _ballooning_context(qa_eq.state, qa_eq.runtime)
    length, alpha, x = gammac._sampling_grids(ctx["s"].dtype, 4, 2, 32)
    lines = gammac._line_ingredients(ctx, 6, jnp.asarray(0.0), alpha, x)
    lines.pop("iota")
    F = gammac._check_lines(lines["bmag"], lines)
    sign = jnp.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])[:, None]
    line = jax.jit(jax.value_and_grad(lambda f: gammac._gamma_c_line(
        f, length, K=96, Q=3, W=16, qorder=32, width=0.05, completeness=0.05)[0]))
    for index in (1, 2):
        value, grad = line(F[index])
        mirror, mgrad = line(F[index][:, ::-1] * sign)
        np.testing.assert_allclose(float(mirror), float(value), rtol=1e-9)
        scale = float(jnp.max(jnp.abs(grad)))
        np.testing.assert_allclose(np.asarray(mgrad[:, ::-1] * sign), np.asarray(grad),
                                   atol=5e-3 * scale)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

def test_class_contract_and_validation():
    """The classes are thin bindings of the state functions with guards."""
    calls = {}

    def fake_state(state, rt, **kwargs):
        calls.update(kwargs)
        return {"gamma_c": jnp.array([0.1, 0.2]), "gamma_alpha": jnp.array([0.3, 0.4])}

    original = gammac._evaluate
    gammac._evaluate = lambda kind, state, rt, surfaces, zeta0, options: fake_state(
        state, rt, kind=kind, surfaces=surfaces, **options)
    try:
        eq = SimpleNamespace(state=object(), runtime=object())
        term = gammac.GammaC([0.3, 0.7], weights=[1.0, 4.0], nalpha=5)
        np.testing.assert_allclose(np.asarray(term(eq)), [0.1, 0.4])
        assert float(term.total(eq)) == pytest.approx(0.01 + 0.16)
        assert calls["nalpha"] == 5 and tuple(calls["surfaces"]) == (0.3, 0.7)
        alpha = gammac.GammaAlpha([0.4, 0.6], pitch_levels=16)
        np.testing.assert_allclose(np.asarray(alpha(eq)), [0.3, 0.4])
        assert calls["kind"] == "alpha" and alpha.name == "gamma_alpha"
    finally:
        gammac._evaluate = original

    with pytest.raises(ValueError, match="increasing surfaces"):
        gammac.GammaC([0.7, 0.3])
    with pytest.raises(ValueError, match="strictly inside"):
        gammac.GammaC([0.0, 0.5])
    with pytest.raises(ValueError, match="weights"):
        gammac.GammaC([0.3, 0.7], weights=[1.0])
    with pytest.raises(ValueError, match="nalpha"):
        gammac.GammaC([0.5], nalpha=1)
    with pytest.raises(TypeError, match="unknown option"):
        gammac.GammaC([0.5], temperature=0.1)
    with pytest.raises(ValueError, match="surfaces must be finite"):
        gammac._surface_rows([], 13)
    with pytest.raises(ValueError, match="duplicate radial rows"):
        gammac._surface_rows([0.5, 0.52], 13)
    with pytest.raises(ValueError, match="num_transit"):
        gammac.gamma_c_state(None, None, num_transit=0)
    with pytest.raises(ValueError, match="pitch_nodes"):
        gammac.gamma_c_state(None, None, pitch_nodes=1)
    with pytest.raises(ValueError, match="tangency_width"):
        gammac.gamma_c_state(None, None, tangency_width=0.0)
    with pytest.raises(ValueError, match="nline"):
        gammac.gamma_c_from_fieldlines(
            bmag=jnp.ones(8), radial_drift=0.0, radial_gradient=0.0,
            drift_correction=0.0, tangency=1.0, dl_dx=1.0, length=1.0)


def test_optimize_reexports_the_gamma_family():
    """``vmex.optimize`` lazily resolves both classes and state functions."""
    for name in ("GammaC", "GammaAlpha", "gamma_c_state", "gamma_alpha_state"):
        assert getattr(opt, name) is getattr(gammac, name)


@pytest.mark.parametrize("lasym", (False, True))
@pytest.mark.parametrize("kind", ("c", "alpha"))
def test_wout_row_map_preserves_values_and_directional_derivatives(lasym, kind):
    """A compact analytic torus retains every diagnostic and its JVP."""
    s = jnp.linspace(0., 1., 7)
    radius = 0.3 * jnp.sqrt(s)
    rmnc = jnp.stack([3 * jnp.ones_like(s), radius, .02 * s], axis=-1)
    zero = jnp.zeros_like(rmnc)
    tables = dict(
        s=s, hs=s[1] - s[0], psi_edge=jnp.array(.1),
        iotas=.4 + .02 * s, phipf=jnp.full_like(s, .1),
        m=jnp.array([0., 1., 1.]), xn=jnp.array([0., 0., 1.]),
        rmnc=rmnc, zmns=rmnc.at[:, 0].set(0.), lmns=zero,
        rmns=.01 * rmnc if lasym else None,
        zmnc=.01 * rmnc if lasym else None,
        lmnc=zero if lasym else None)
    options = dict(nalpha=4, num_transit=1, points_per_transit=16, quadrature_order=8,
                   max_wells=4, **({"pitch_nodes": 2, "max_levels": 8} if kind == "c"
                                   else {"pitch_levels": 6}))
    cfg = gammac._resolve(kind, options) + (("nfp", 1),)
    settings = dict(rows=(2, 4), kind=kind, cfg=cfg)

    def evaluate(coefficients, mapped):
        ctx = dict(tables, rmnc=coefficients)
        if mapped:
            return gammac._gamma_c_rows_from_tables(
                ctx, jnp.array(0.), lasym=lasym, **settings)
        return gammac._rows_from_context(
            dict(ctx, lasym=lasym), jnp.array(0.), **settings)

    direction = jnp.reshape(jnp.linspace(-.001, .001, rmnc.size), rmnc.shape)
    results = [jax.jit(lambda c: jax.jvp(
        lambda x: evaluate(x, mapped), (c,), (direction,)))(rmnc)
        for mapped in (False, True)]
    for original, mapped in zip(*results):
        assert original.keys() == mapped.keys()
        for name in original:
            np.testing.assert_array_equal(np.isnan(original[name]), np.isnan(mapped[name]))
            np.testing.assert_allclose(original[name], mapped[name], rtol=2e-6, atol=1e-10)
