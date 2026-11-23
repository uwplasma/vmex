"""Panel-inventory and style smoke tests for the ``--plot`` summary figure.

One in-process solve of the bundled ``cth_like_fixed_bdy`` deck feeds every
check, so the module needs no golden fixtures and stays network-free:

- the summary figure carries the full required panel set (iota full-mesh
  with ``<J.B>`` on its right axis, pressure with the
  ``eps_eff^(3/2)``/``Gamma_c`` confinement axis, relative radial force
  balance, combined Mercier/Glasser/well profiles, 3-D LCFS,
  polar ``J(alpha, s)``, two Boozer ``|B|`` panels, scalar card);
- the confinement bundle is cached per in-memory WOUT (bounded, weakly
  keyed), reuses one Boozer transform and one ``Gamma_c`` executable, and
  drops an unavailable diagnostic with a stated reason instead of a zero;
- style invariants are pinned: every ``|B|`` contour set is non-filled and
  jet-mapped, the 3-D surface colormap constant is jet, all text is >= 11 pt,
  every drawn text artist stays inside the canvas, saved PNGs are >= 200 dpi;
- the wout-based Glasser ``D_R`` reconstruction and frozen-pressure response
  must recover the traceable equilibrium values at the stored beta.
"""

from __future__ import annotations

import dataclasses
import inspect
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("booz_xform_jax")
jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

from vmex.core import optimize as opt  # noqa: E402
from vmex.core import plotting  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.wout import wout_from_state  # noqa: E402

try:
    import neo_jax.fastion  # noqa: F401
    HAS_FASTION = True
except ImportError:
    HAS_FASTION = False
needs_fastion = pytest.mark.skipif(not HAS_FASTION, reason="needs neo_jax.fastion")
DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
DECK = "cth_like_fixed_bdy"

EXPECTED_PANELS = {
    "iota", "profiles", "force_balance", "stability", "boundary_3d",
    "j_invariant", "card", "boozer_mid", "boozer_lcfs",
}


@pytest.fixture(scope="module")
def solved_case():
    """Solve the deck once; return ``(eq, WoutData)``."""
    inp = VmecInput.from_file(DATA_DIR / f"input.{DECK}")
    eq = opt.solve_equilibrium(inp)
    wout = wout_from_state(inp=inp, state=eq.state, fsqr=0.0, fsqz=0.0, fsql=0.0)
    return eq, wout


@pytest.fixture(scope="module")
def summary_figure(solved_case):
    """Rendered summary figure + meta; closed after the module finishes."""
    import matplotlib.pyplot as plt

    _, wout = solved_case
    fig, meta = plotting._summary_figure(wout)
    fig.canvas.draw()
    yield fig, meta
    plt.close(fig)


def _drawn_tick_labels(axis):
    """Tick labels matplotlib actually draws (tick within the view interval)."""
    lo, hi = sorted(axis.get_view_interval())
    for tick in axis.get_major_ticks():
        if lo <= tick.get_loc() <= hi:
            yield tick.label1


def _text_artists(fig):
    for ax in fig.axes:
        items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        items += list(_drawn_tick_labels(ax.xaxis))
        items += list(_drawn_tick_labels(ax.yaxis))
        items += list(ax.texts)
        legend = ax.get_legend()
        if legend is not None:
            items += list(legend.get_texts())
        for text in items:
            if text.get_visible() and text.get_text().strip():
                yield text


def _contour_sets(ax):
    from matplotlib.contour import QuadContourSet

    return [c for c in getattr(ax, "collections", []) if isinstance(c, QuadContourSet)]


def test_summary_panel_inventory(summary_figure):
    """All nine required panels exist and are populated."""
    _, meta = summary_figure
    assert set(meta["axes"]) == EXPECTED_PANELS
    for name, ax in meta["axes"].items():
        assert ax.lines or ax.collections or ax.texts, f"panel {name!r} is empty"
        if name not in ("card", "boundary_3d"):
            assert ax.get_xlabel().strip(), f"panel {name!r} lacks an x label"
            assert ax.get_ylabel().strip(), f"panel {name!r} lacks a y label"


def test_summary_contours_and_colormaps(summary_figure):
    """J is filled with line contours on top; Boozer |B| is unfilled and jet-mapped."""
    _, meta = summary_figure
    j_sets = _contour_sets(meta["axes"]["j_invariant"])
    assert any(cs.filled for cs in j_sets) and any(not cs.filled for cs in j_sets)
    for name in ("boozer_mid", "boozer_lcfs"):
        sets = _contour_sets(meta["axes"][name])
        assert sets, f"panel {name!r} has no contour set"
        for cs in sets:
            assert cs.filled is False, f"filled contour in {name!r}"
            assert cs.get_cmap().name == "jet"


def test_summary_typography_and_no_clipping(summary_figure):
    """Text >= 11 pt and every drawn text artist inside the canvas."""
    fig, _meta = summary_figure
    renderer = fig.canvas.get_renderer()
    bbox = fig.bbox
    for text in _text_artists(fig):
        assert text.get_fontsize() >= 10.9, f"{text.get_text()!r} is {text.get_fontsize()} pt"
        extent = text.get_window_extent(renderer=renderer)
        assert extent.x0 >= bbox.x0 - 2 and extent.x1 <= bbox.x1 + 2, text.get_text()
        assert extent.y0 >= bbox.y0 - 2 and extent.y1 <= bbox.y1 + 2, text.get_text()


def test_summary_field_line_and_j_map_present(summary_figure):
    """Boozer panels carry the iota field line; the J map spans surfaces."""
    _, meta = summary_figure
    for name in ("boozer_mid", "boozer_lcfs"):
        labels = [line.get_label() for line in meta["axes"][name].lines]
        assert any("field line" in label for label in labels), name
    j_map = meta["j_map"]["j_map"]
    assert np.isfinite(j_map).any()
    assert j_map.shape[0] >= 5  # radial spread of Boozer surfaces
    j_axis = meta["axes"]["j_invariant"]
    assert j_axis.get_xlabel() == r"$s\,\cos\alpha$"
    assert j_axis.get_ylabel() == r"$s\,\sin\alpha$"
    assert j_axis.get_aspect() == 1.0


def test_summary_combines_stability_and_well(summary_figure):
    """DMerc, dashed D_R, and dash-dot V'' share zeroes; legend stays below."""
    fig, meta = summary_figure
    stability = meta["axes"]["stability"]
    well = meta["well_axis"]
    assert {line.get_linestyle() for line in stability.lines} >= {"-", "--"}
    assert any(line.get_linestyle() == "-." for line in well.lines)
    assert well.yaxis.label.get_color() == plotting._LINE_COLORS[2]
    assert "V''" in well.get_ylabel()
    labels = [text.get_text() for text in stability.get_legend().get_texts()]
    assert any("V''" in label and "well" in label for label in labels)
    fig.canvas.draw()
    assert stability.transData.transform((0.0, 0.0))[1] == pytest.approx(
        well.transData.transform((0.0, 0.0))[1])
    renderer = fig.canvas.get_renderer()
    assert stability.get_legend().get_window_extent(renderer).y1 <= (
        stability.get_window_extent(renderer).y0 + 2)


def test_summary_combines_force_current_and_confinement(
    solved_case, summary_figure,
):
    """Force balance carries the equilibrium <J.B>; iota stands alone; pressure carries confinement."""
    _, meta = summary_figure
    iota = meta["axes"]["iota"]
    assert len(iota.lines) == 1 and iota.get_legend() is None
    assert iota.get_title() == "rotational transform"

    force = meta["axes"]["force_balance"]
    current = meta["current_axis"]
    assert len(current.lines) == 1
    assert current.get_shared_x_axes().joined(current, force)
    assert current.get_ylabel().startswith(r"$\langle \mathbf{J}")   # total, not bootstrap
    assert current.get_ylabel().endswith("[kA T/m$^2$]")
    labels = [text.get_text() for text in force.get_legend().get_texts()]
    assert "force error" in labels
    assert any(r"\mathbf{J}" in label for label in labels)

    profiles = meta["axes"]["profiles"]
    assert len(profiles.lines) == 1
    assert "pressure and confinement" in profiles.get_title()
    labels = [text.get_text() for text in profiles.get_legend().get_texts()]
    assert any(label == r"$p$" for label in labels)


@needs_fastion
def test_summary_confinement_axis_draws_only_valid_profiles(summary_figure):
    """The right confinement axis carries exactly the valid diagnostics.

    On this deck ``Gamma_c`` always evaluates; the effective ripple rides
    along when NEO_JAX is installed and is dropped **with its reason
    recorded** when not — an invalid diagnostic must never appear as a
    plausible zero curve.
    """
    _, meta = summary_figure
    conf = meta["confinement"]
    axis = meta["confinement_axis"]
    assert conf.validity["gamma_c"], conf.notes["gamma_c"]
    labels = {line.get_label(): line for line in axis.lines}
    assert r"$\Gamma_c$" in labels
    gamma_line = labels[r"$\Gamma_c$"]
    values = np.asarray(gamma_line.get_ydata(), dtype=float)
    assert np.all(np.isfinite(values)) and np.all(values != 0.0)
    np.testing.assert_allclose(values, conf.gamma_c)
    assert conf.timing["gamma_c"] > 0.0
    if conf.validity["epsilon_effective"]:
        assert r"$\epsilon_{\mathrm{eff}}^{3/2}$" in labels
    else:
        assert conf.notes["epsilon_effective"]
        assert r"$\epsilon_{\mathrm{eff}}^{3/2}$" not in labels
    # distinct styles on the shared axis
    styles = {(line.get_linestyle(), line.get_marker()) for line in axis.lines}
    assert len(styles) == len(axis.lines)


def test_summary_reports_force_error(solved_case, summary_figure):
    """The force panel states its normalization; the card gives its volume average."""
    _, meta = summary_figure
    force = meta["axes"]["force_balance"]
    assert force.get_yscale() == "log"
    assert r"\rho=\sqrt{s}" in force.get_xlabel()
    assert r"s=\psi/\psi_B" in force.get_xlabel()
    assert r"\nabla p" in force.get_ylabel() and r"\nabla(B^2/2\mu_0)" in force.get_ylabel()
    assert r"0.1\leq s\leq 0.99" in force.get_ylabel()
    _, wout = solved_case
    assert meta["force_error"] == plotting._relative_force_error_profile(wout)[2]
    # Converged finite-beta deck: 1.3e-3 measured, while equif reaches 0.94.
    assert meta["force_error"] < 1.0e-2
    card_text = " ".join(text.get_text() for text in meta["axes"]["card"].texts)
    assert r"\nabla B^2/2\mu_0" in card_text


@pytest.mark.parametrize("niter,low,high", [(3000, 0.0, 1.0e-2), (40, 5.0e-2, np.inf)])
def test_force_error_resolves_vacuum_convergence(niter, low, high):
    """equif is 1 on a currentless vacuum at any residual; the plotted error is not."""
    inp = dataclasses.replace(
        VmecInput.from_file(DATA_DIR / "input.LandremanPaul2021_QA_lowres"),
        ns_array=[16], niter_array=[niter], ftol_array=[1e-12],
    )
    wout = wout_from_state(inp=inp, state=opt.solve_equilibrium(inp).state, fsqr=0.0, fsqz=0.0, fsql=0.0)
    np.testing.assert_allclose(np.abs(wout.equif[1:-1]), 1.0, atol=1e-6)
    rho, profile, average = plotting._relative_force_error_profile(wout)
    assert rho.size == 14 and np.all(profile > 0.0)
    # Measured 1.6e-3 converged and 0.14 after 40 iterations.
    assert low < average < high


def test_summary_style_constants():
    """CLI plots keep publication resolution, smooth 3-D grids, and jet |B|."""
    assert plotting._DPI >= 200
    assert plotting._CMAP_3D == "jet"
    assert plotting._CMAP_MODB == "jet"
    signature = inspect.signature(plotting.plot_boundary_3d)
    assert signature.parameters["ntheta"].default >= 120


@pytest.mark.parametrize("zsort", ("average", "min", "max"))
def test_surface_projection_cache_preserves_geometry_colors_and_camera(monkeypatch, zsort):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def rz(wout, *, s_index, theta, phi):
        shape = (len(theta), len(phi))
        return (np.broadcast_to(2 + .2 * np.cos(theta)[:, None], shape),
                np.broadcast_to(.2 * np.sin(theta)[:, None], shape))

    monkeypatch.setattr(plotting, "surface_rz", rz)
    monkeypatch.setattr(plotting, "surface_modB", lambda wout, **kwargs: rz(wout, **kwargs)[0])
    fig = plt.figure(figsize=(3, 3), dpi=60)
    ax = fig.add_subplot(projection="3d")
    plotting._boundary_3d_panel(ax, SimpleNamespace(ns=2), ntheta=8, nzeta=12)
    artist = ax.collections[0]
    artist.set_zsort(zsort)
    depths = np.array([[-0.0] * 4, [1e16, 1.0, -1e16, 1.0]])
    for row in (*depths, depths[0, :3]):
        expected = Poly3DCollection._zsort_functions[zsort](row)
        assert artist._zsortfunc(row).view(np.uint64) == expected.view(np.uint64)
    masked = np.ma.array(depths, mask=depths < 0)
    for values, axis in ((depths, -1), (depths, 0), (np.tile(depths, (2, 1)), None),
                         (depths[0], -1), (depths[:, :3], -1), (depths[:0], -1),
                         (masked, -1), (masked[1], None),
                         (np.full(4, np.iinfo(np.int64).max, dtype=np.int64), None),
                         (np.array([True, True, True, False]), None),
                         (np.array([np.finfo(np.float32).smallest_subnormal, 0, 0, 0], dtype=np.float32), None)):
        try:
            expected = Poly3DCollection._zsort_functions[zsort](values, axis=axis)
        except (ValueError, ZeroDivisionError) as exc:
            with pytest.raises(type(exc)):
                artist._zsortfunc(values, axis=axis)
        else:
            actual = artist._zsortfunc(values, axis=axis)
            np.testing.assert_array_equal(actual, expected)
            assert np.asarray(actual).dtype == np.asarray(expected).dtype
    original, calls = Poly3DCollection.do_3d_projection, []

    def project(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(Poly3DCollection, "do_3d_projection", project)
    try:
        for update in (lambda: None, lambda: ax.view_init(20, 40),
                       lambda: artist.set_facecolor("red"),
                       lambda: artist.set_verts([[[0, 0, 0], [1, 0, 0], [0, 1, 1]]]),
                       lambda: ax.set_xlim(-1, 3)):
            update()
            fig.canvas.draw()
            count = len(calls)
            fig.canvas.draw()
            assert len(calls) == count
            pixels = np.asarray(fig.canvas.buffer_rgba()).copy()
            with monkeypatch.context() as reference:
                reference.setattr(type(artist), "do_3d_projection", original)
                reference.setattr(artist, "_zsortfunc", Poly3DCollection._zsort_functions[zsort])
                fig.canvas.draw()
            np.testing.assert_array_equal(fig.canvas.buffer_rgba(), pixels)
    finally:
        plt.close(fig)


def test_force_panel_reports_missing_data_deliberately():
    import matplotlib.pyplot as plt

    assert plotting._fmt_compact(float("nan")) == "unavailable"
    fig, ax = plt.subplots()
    try:
        maximum = plotting._relative_force_error_panel(
            ax, SimpleNamespace(ns=3, equif=np.full((3,), np.nan))
        )
        assert np.isnan(maximum)
        assert [text.get_text() for text in ax.texts] == ["force error unavailable"]
    finally:
        plt.close(fig)


# ==========================================================================
# Confinement summary: shared work, cache, and unavailability semantics
# ==========================================================================

def _fresh_conf(wout):
    """confinement_summary through a cleared cache (fresh computation)."""
    plotting._CONFINEMENT_CACHE.clear()
    return plotting.confinement_summary(wout, None, booz_note="not sampled")


@needs_fastion
def test_confinement_summary_is_cached_and_never_recompiles(solved_case):
    """Same WOUT + settings: one computation, one XLA executable.

    The second call must be a pure cache hit (same object, effectively
    instant — the bounded runtime gate for repeated summary generation), and
    a from-scratch recomputation for the same shapes must reuse the jitted
    ``Gamma_c`` executable rather than compile a second one.
    """
    _, wout = solved_case
    plotting._CONFINEMENT_CACHE.clear()
    first = plotting.confinement_summary(wout, None, booz_note="not sampled")
    start = time.perf_counter()
    again = plotting.confinement_summary(wout, None, booz_note="not sampled")
    elapsed = time.perf_counter() - start
    assert again is first
    assert elapsed < 0.5
    recomputed = _fresh_conf(wout)
    assert recomputed is not first
    np.testing.assert_allclose(recomputed.gamma_c, first.gamma_c)


def test_confinement_cache_is_bounded_and_weakly_keyed(solved_case, monkeypatch):
    """At most ``_CONFINEMENT_CACHE_SIZE`` entries; dead WOUTs are evicted."""
    import dataclasses as dc

    monkeypatch.setattr(
        plotting, "_gamma_c_profile",
        lambda wout: (np.array([0.5]), np.array([1.0e-3]), ""))
    monkeypatch.setattr(
        plotting, "_epsilon_effective_profile",
        lambda booz, note: (None, None, "skipped"))
    _, wout = solved_case
    plotting._CONFINEMENT_CACHE.clear()
    clones = [dc.replace(wout) for _ in range(plotting._CONFINEMENT_CACHE_SIZE + 2)]
    for clone in clones:
        plotting.confinement_summary(clone, None, booz_note="x")
    assert len(plotting._CONFINEMENT_CACHE) == plotting._CONFINEMENT_CACHE_SIZE
    del clones, clone
    import gc

    gc.collect()
    plotting.confinement_summary(wout, None, booz_note="x")
    assert len(plotting._CONFINEMENT_CACHE) == 1


@needs_fastion
def test_confinement_missing_neo_and_lasym_notes(solved_case, monkeypatch):
    """Missing NEO_JAX or Boozer tables give stated reasons."""
    from vmex.core import neoclassical

    _, wout = solved_case

    def _no_neo():
        raise ImportError("effective ripple requires NEO_JAX")

    monkeypatch.setattr(neoclassical, "diagnostic_neo_config", _no_neo)
    booz = {"mboz": 4, "nboz": 4, "s_b": np.array([0.5]), "neo_booz": {}}
    plotting._CONFINEMENT_CACHE.clear()
    conf = plotting.confinement_summary(wout, booz)
    assert not conf.validity["epsilon_effective"]
    assert "NEO_JAX" in conf.notes["epsilon_effective"]
    assert conf.validity["gamma_c"]

    plotting._CONFINEMENT_CACHE.clear()
    conf = plotting.confinement_summary(wout, {**booz, "neo_booz": None})
    assert not conf.validity["epsilon_effective"]
    assert "tables unavailable" in conf.notes["epsilon_effective"]


def test_confinement_one_valid_one_invalid_and_never_zero(
    solved_case, monkeypatch,
):
    """A failing Gamma_c is dropped with a reason while eps still plots.

    A failed or nonconverged diagnostic must never be drawn as zero: the
    panel keeps only the valid curve and the note carries the cause.
    """
    import matplotlib.pyplot as plt

    from vmex.core import fastion, neoclassical

    _, wout = solved_case

    def _fake_eps(booz, *, config=None):
        return np.array([0.3, 0.6, 0.9]), np.array([1e-4, 2e-4, 4e-4])

    def _broken_gamma(*args, **kwargs):
        raise RuntimeError("synthetic gamma failure")

    monkeypatch.setattr(neoclassical, "diagnostic_neo_config", lambda: None)
    monkeypatch.setattr(
        neoclassical, "epsilon_effective_from_boozer", _fake_eps)
    monkeypatch.setattr(fastion, "fast_ion_from_wout", _broken_gamma)
    plotting._CONFINEMENT_CACHE.clear()
    booz = {"mboz": 4, "nboz": 4, "s_b": np.array([0.5]), "neo_booz": {}}
    conf = plotting.confinement_summary(wout, booz)
    assert conf.validity["epsilon_effective"]
    assert not conf.validity["gamma_c"]
    assert "RuntimeError" in conf.notes["gamma_c"]

    fig, ax = plt.subplots()
    try:
        axis, lines = plotting._confinement_panel(ax, conf)
        assert [line.get_label() for line in lines] == [
            r"$\epsilon_{\mathrm{eff}}^{3/2}$"]
        assert axis.get_yscale() == "linear"  # 4x dynamic range: no log
    finally:
        plt.close(fig)


@needs_fastion
def test_confinement_guards_report_failures_and_skip_cache(
    solved_case, monkeypatch,
):
    """Every defensive branch reports its reason; no silent zeros anywhere.

    NEO raising mid-evaluation, NEO returning nothing positive, an all-NaN
    ``Gamma_c`` (iota ~ 0 poison on every surface), and a wout stand-in that
    cannot be weak-referenced (computed uncached rather than crashing).
    """
    from vmex.core import fastion, neoclassical

    _, wout = solved_case
    booz = {"mboz": 4, "nboz": 4, "s_b": np.array([0.5]), "neo_booz": {}}
    monkeypatch.setattr(neoclassical, "diagnostic_neo_config", lambda: None)

    def _raises(_booz, *, config=None):
        raise RuntimeError("synthetic NEO failure")

    monkeypatch.setattr(neoclassical, "epsilon_effective_from_boozer", _raises)
    plotting._CONFINEMENT_CACHE.clear()
    conf = plotting.confinement_summary(wout, booz)
    assert conf.notes["epsilon_effective"] == "NEO evaluation failed: RuntimeError: synthetic NEO failure"

    monkeypatch.setattr(
        neoclassical, "epsilon_effective_from_boozer",
        lambda _booz, *, config=None: (np.array([0.5]), np.array([0.0])))
    monkeypatch.setattr(
        fastion, "fast_ion_from_wout",
        lambda w, **kw: {"s": np.array([0.5]), "gamma_c": np.array([np.nan])})
    plotting._CONFINEMENT_CACHE.clear()
    conf = plotting.confinement_summary(wout, booz)
    assert conf.notes["epsilon_effective"] == "NEO returned no finite positive values"
    assert conf.notes["gamma_c"] == "no surface returned a finite Gamma_c"
    assert not conf.validity["gamma_c"] and conf.gamma_c is None

    class SlotsWout:
        __slots__ = ("ns",)                   # no __weakref__: uncacheable

    monkeypatch.setattr(
        plotting, "_gamma_c_profile",
        lambda w: (np.array([0.5]), np.array([1e-3]), ""))
    plotting._CONFINEMENT_CACHE.clear()
    conf = plotting.confinement_summary(SlotsWout(), booz)
    assert conf.validity["gamma_c"]
    assert len(plotting._CONFINEMENT_CACHE) == 0  # computed, not cached


def test_confinement_panel_annotates_when_nothing_is_valid():
    """Both diagnostics invalid: an explicit note, no fabricated curves."""
    import matplotlib.pyplot as plt

    conf = plotting.ConfinementSummary(
        surfaces={}, epsilon_effective=None, gamma_c=None,
        validity={"epsilon_effective": False, "gamma_c": False},
        notes={"epsilon_effective": "a", "gamma_c": "b"}, timing={})
    fig, ax = plt.subplots()
    try:
        axis, lines = plotting._confinement_panel(ax, conf)
        assert lines == []
        assert any(
            "confinement diagnostics unavailable" in t.get_text()
            for t in axis.texts)
    finally:
        plt.close(fig)


def test_confinement_log_scale_needs_positive_wide_range():
    """Log scale only for all-positive data spanning >= two decades."""
    import matplotlib.pyplot as plt

    base = dict(
        epsilon_effective=None, gamma_c=np.array([1e-6, 5e-4, 2e-3]),
        validity={"epsilon_effective": False, "gamma_c": True},
        notes={"epsilon_effective": "no", "gamma_c": ""}, timing={})
    wide = plotting.ConfinementSummary(
        surfaces={"gamma_c": np.array([0.3, 0.6, 0.9])}, **base)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    try:
        axis, _ = plotting._confinement_panel(ax1, wide)
        assert axis.get_yscale() == "log"
        base["gamma_c"] = np.array([0.0, 5e-4, 2e-3])   # zero: no log
        zero = plotting.ConfinementSummary(
            surfaces={"gamma_c": np.array([0.3, 0.6, 0.9])}, **base)
        axis, _ = plotting._confinement_panel(ax2, zero)
        assert axis.get_yscale() == "linear"
    finally:
        plt.close(fig)


@needs_fastion
def test_fast_ion_from_wout_matches_live_state(solved_case):
    """The plot route reproduces the live-state Gamma_c (same Boozer rows)."""
    from neo_jax.fastion import FastIonBudget

    from vmex.core import fastion

    eq, wout = solved_case
    budget = FastIonBudget(**plotting._GAMMA_C_DIAGNOSTIC)
    live = fastion.fast_ion_from_state(eq.state, eq.runtime, (0.3, 0.6), budget=budget)
    from_wout = fastion.fast_ion_from_wout(wout, (0.3, 0.6), budget=budget)
    np.testing.assert_allclose(np.asarray(from_wout["gamma_c"]), np.asarray(live["gamma_c"]),
                               rtol=1e-4, atol=1e-8)
