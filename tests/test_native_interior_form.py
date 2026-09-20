"""The interior field built from the native VMEC form.

``B`` used to be synthesized from the Nyquist ``B^u``/``B^v`` tables: a Fourier
fit of a rational function, interpolated radially on the half mesh, against
tangent vectors taken from a different (full-mesh) geometry.  The angular parts
of that are exact and the radial parts are not, and the two do not agree well
enough for the result to be solenoidal.  The native form instead evaluates

    B^theta = (chi' - lambda_zeta) / sqrt(g),
    B^zeta  = (phi' + lambda_theta) / sqrt(g),

with ``sqrt(g)`` built from the same ``R`` and ``Z`` series the position uses,
so only the radial profiles are interpolated and ``div B`` vanishes in the
angles identically rather than approximately.

The conventions it has to honour, read off ``vmex.core.fields.magnetic_fields``
and ``lambda_scale``:

===========================  ============  ==================================
quantity                     mesh          definition
===========================  ============  ==================================
``phip``                     full          ``signgs * phipf_wout / (2 pi)``
``chip`` (``chips``)         half          ``d(chi)/ds``, VMEC-internal
``lamscale``                 scalar        ``sqrt(hs * sum phips**2)``
``lamscale * lambda``        full          evolved internally, sine for
                                           stellarator symmetry
``sqrt(g)``                  half          ``R (R_theta Z_s - R_s Z_theta)``,
                                           the ``ru12*zs - rs*zu12`` ordering
                                           of ``jacobian.f``
===========================  ============  ==================================

The ``+`` on the ``dlambda/dzeta`` term in the ``magnetic_fields`` class
docstring disagrees with its own ``lv = -lamscale * dlambda/dzeta`` and with
the code; the code is authoritative.

Three tests, because the change has three independent ways to be wrong.
:func:`test_native_form_matches_the_fitted_field_on_a_solved_equilibrium` pins
every convention in that table at once, by requiring the two paths to agree on
a real equilibrium to the accuracy the fitted one has; any convention error
moves that by far more than the tolerance.
:func:`test_native_form_reaches_the_second_derivative_gate` measures what the
change was for, against a field known in closed form.
:func:`test_fitted_fallback_keeps_its_measured_accuracy` measures the path
callers still get when their spectra predate this change.

The oracles came from PR #382, which committed them on their own, before the
fix existed, as bands asserting how wrong the code was.  That is a fair thing
to decline, and it was declined.  What they asserted has changed with them: the
breathing circle now asserts the gates the native form meets, and the purely
toroidal one documents a fallback's accuracy rather than pinning a defect
awaiting repair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vmex.core import extender as ext  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"

#: Circle parameters, flux derivatives and angular truncation of the oracle.
MAJOR, MINOR, TOROIDAL, POLOIDAL, MODES = 1.0, 0.25, 0.7, 0.35, 28

#: Reviewer L-EXT's R6 gates for the interior field.
SECOND_DERIVATIVE_GATE = 1.0e-3
THIRD_DERIVATIVE_GATE = 1.0e-2
DIVERGENCE_GATE = 1.0e-12

#: Flux labels the oracles are sampled at, and the poloidal angle they use.
SAMPLES = np.array([0.3, 0.55, 0.8, 0.95])
ANGLE = 0.7

#: Purely-toroidal oracle: concentric circles of radius ``a rho`` carrying
#: ``B = F/R phi_hat``.  Ceilings on what the fitted fallback achieves on it,
#: measured at ``ns = 41, 81, 161`` (see the test's docstring for the table).
#: They are upper bounds, not bands: an improved fallback passes unchanged.
FALLBACK_MINOR, FALLBACK_FLUX = 0.1, 2.0
#: Applied at the FINEST resolution tested; the coarse end is documented in
#: the test's table rather than asserted, because it is not usable there.
FALLBACK_CEILINGS = {"B": 3.0e-6, "gradB": 1.0e-4,
                     "gradgradB": 5.0e-3, "gradgradgradB": 1.0e0}


def _radius(s):
    """Minor radius of the breathing circle; it has real radial curvature."""
    return MINOR * jnp.sqrt(jnp.maximum(s, 1e-300)) * jnp.exp(0.4 * s)


def _radius_np(s):
    return MINOR * np.sqrt(np.maximum(s, 1e-300)) * np.exp(0.4 * s)


def _exact_B(xyz):
    """``B`` of the oracle in closed form, from the native form itself.

    With ``lambda = 0`` the Jacobian is ``-R A A'``, so
    ``B = -(phi'/(A A')) phi_hat - (chi'/(R A')) theta_hat``.  The flux label of
    a Cartesian point follows from ``A(s)^2 = (R - R0)^2 + z^2``.
    """
    x, y, z = xyz
    big = jnp.sqrt(x**2 + y**2)
    target = (big - MAJOR) ** 2 + z**2
    s = jax.lax.custom_root(
        lambda q: _radius(q) ** 2 - target, jnp.asarray(0.5),
        lambda _, start: jax.lax.fori_loop(
            0, 60, lambda _, q: jnp.clip(
                q - (_radius(q) ** 2 - target)
                / jax.grad(lambda v: _radius(v) ** 2)(q), 1e-12, 1.5), start),
        lambda linear, rhs: rhs / linear(1.0))
    minor, slope = _radius(s), jax.grad(_radius)(s)
    angle = jnp.arctan2(z, big - MAJOR)
    cosine, sine = x / big, y / big
    toroidal = -(TOROIDAL / (minor * slope)) * jnp.stack(
        (-y, x, jnp.zeros_like(z))) / big
    poloidal = -(POLOIDAL * s / (big * slope)) * jnp.stack(
        (-jnp.sin(angle) * cosine, -jnp.sin(angle) * sine, jnp.cos(angle)))
    return toroidal + poloidal


def _oracle_spectra(ns, native):
    """Spectra of the breathing circle, with or without the native fields."""
    s = np.linspace(0.0, 1.0, ns)
    # Row 0 of a half-mesh table is VMEC's unused axis row.
    half = np.concatenate((s[1:2] / 2.0, 0.5 * (s[:-1] + s[1:])))
    minor_half = _radius_np(half)[:, None]
    slope_half = np.asarray(jax.vmap(jax.grad(_radius))(
        jnp.asarray(np.maximum(half, 1e-14))))[:, None]
    angles = 2 * np.pi * np.arange(4 * MODES) / (4 * MODES)
    big = MAJOR + minor_half * np.cos(angles)[None, :]
    jacobian = -(big * minor_half * slope_half)
    modes = np.arange(MODES, dtype=float)
    weight = np.where(modes == 0, 1.0, 2.0) / len(angles)
    basis = np.cos(modes[None, :] * angles[:, None])
    spectra = {
        "nfp": 1, "ns": ns,
        "xm": jnp.array([0.0, 1.0]), "xn": jnp.zeros(2),
        "xmn": jnp.asarray(modes), "xnn": jnp.zeros(MODES),
        "rmnc": jnp.asarray(np.stack((np.full(ns, MAJOR), _radius_np(s)), axis=1)),
        "zmns": jnp.asarray(np.stack((np.zeros(ns), _radius_np(s)), axis=1)),
        "rmns": None, "zmnc": None,
        "bsupu": jnp.asarray(((POLOIDAL * half[:, None] / jacobian) @ basis) * weight),
        "bsupv": jnp.asarray(((TOROIDAL / jacobian) @ basis) * weight),
        "bsupu_s": None, "bsupv_s": None, "lasym": False, "signgs": -1,
    }
    if native:
        spectra |= {"lmns": jnp.zeros((ns, 2)),
                    "phipf": jnp.full(ns, TOROIDAL),
                    "chipf": jnp.asarray(POLOIDAL * s)}
    return spectra


def _toroidal_spectra(ns):
    """Concentric circles carrying ``B = F/R phi_hat``, no native fields.

    Linear interpolation is exact for this geometry, so the fitted path's error
    here is the radial interpolation of its contravariant tables and nothing
    else.  The field cannot be expressed in the native form on this geometry —
    it would need a ``phi'`` that varies with ``theta`` — which is why this
    oracle measures the fallback and only the fallback.
    """
    s = np.linspace(0.0, 1.0, ns)
    half = np.concatenate((s[1:2] / 2.0, 0.5 * (s[:-1] + s[1:])))
    angles = 2 * np.pi * np.arange(4 * MODES) / (4 * MODES)
    big = MAJOR + FALLBACK_MINOR * np.sqrt(half)[:, None] * np.cos(angles)[None, :]
    modes = np.arange(MODES, dtype=float)
    weight = np.where(modes == 0, 1.0, 2.0) / len(angles)
    basis = np.cos(modes[None, :] * angles[:, None])
    minor = FALLBACK_MINOR * np.sqrt(s)
    return {
        "nfp": 1, "ns": ns,
        "xm": jnp.array([0.0, 1.0]), "xn": jnp.zeros(2),
        "xmn": jnp.asarray(modes), "xnn": jnp.zeros(MODES),
        "rmnc": jnp.asarray(np.stack((np.full(ns, MAJOR), minor), axis=1)),
        "zmns": jnp.asarray(np.stack((np.zeros(ns), minor), axis=1)),
        "rmns": None, "zmnc": None,
        "bsupu": jnp.zeros((ns, MODES)),
        "bsupv": jnp.asarray(((FALLBACK_FLUX / big**2) @ basis) * weight),
        "bsupu_s": None, "bsupv_s": None, "lasym": False, "signgs": -1,
    }


def _toroidal_exact(xyz):
    """``B = F/R phi_hat`` in Cartesian components."""
    x, y, z = xyz
    return FALLBACK_FLUX * jnp.stack(
        (-y, x, jnp.zeros_like(z))) / (x**2 + y**2)


def _errors(field, points, exact, names):
    """Relative error of each derivative order against a closed-form field."""
    out = []
    for order, name in enumerate(names):
        got = np.asarray(getattr(field, name)(points))
        derivative = exact
        for _ in range(order):
            derivative = jax.jacfwd(derivative)
        want = np.stack([np.asarray(jax.jit(derivative)(p)) for p in points])
        out.append(float(np.max(np.abs(got - want)) / np.max(np.abs(want))))
    return out


def _oracle_points():
    minor = _radius_np(SAMPLES)
    return jnp.asarray(np.stack(
        (MAJOR + minor * np.cos(ANGLE), np.zeros_like(minor),
         minor * np.sin(ANGLE)), axis=1))


@pytest.mark.full  # nightly: one solovev solve plus two field evaluations
def test_native_form_matches_the_fitted_field_on_a_solved_equilibrium():
    """Every VMEC convention the native form depends on, pinned at once.

    The fitted path is independent of all of them: it reads ``B^u``/``B^v``
    straight out of the equilibrium.  Requiring the two to agree therefore
    checks the Jacobian's sign and ordering, VMEC's ``lamscale``, the internal
    ``phip = signgs * phipf_wout / (2 pi)``, and that ``chips`` was moved off
    the half mesh — together, in one number.  They agree to 5.8e-4 here, which
    is the fitted path's own radial accuracy; a wrong convention is a factor,
    not a fraction of a per cent.
    """
    import vmex as vj
    from vmex import optimize as opt

    equilibrium = opt.solve_equilibrium(
        vj.VmecInput.from_file(DATA / "input.solovev"), verbose=False)
    spectra = equilibrium.field.spectra
    assert ext._has_native_form(spectra), sorted(spectra)

    fitted = {key: value for key, value in spectra.items()
              if key not in ("lmns", "phipf", "chipf")}
    points = ext._flux_coordinates_to_xyz(
        spectra, jnp.array([[0.25, 0.6, 0.0], [0.5, 2.0, 0.4], [0.8, 4.0, 1.1]]))
    native_B = np.asarray(ext.VmecInteriorField(spectra).B(points))
    fitted_B = np.asarray(ext.VmecInteriorField(fitted).B(points))
    difference = np.max(np.abs(native_B - fitted_B)) / np.max(np.abs(fitted_B))
    assert difference < 5.0e-3, (
        f"native and fitted B differ by {difference:.2e}, which is far more "
        "than the fitted path's own radial accuracy: a VMEC convention "
        "(Jacobian sign, lamscale, phip, or chips's mesh) is wrong.\n"
        f"native {native_B}\nfitted {fitted_B}")


@pytest.mark.full  # nightly: four derivative orders at three resolutions
def test_native_form_reaches_the_second_derivative_gate():
    """What the change is for, against a field known in closed form.

    The geometry breathes with ``s``, so the radial interpolant is exercised
    rather than being exact by construction, and a non-zero ``chi'`` makes
    ``div B`` a real test rather than a symmetry.  These are gates, not pins:
    this PR is what makes them pass, so they assert the target directly.

    Measured, native against fitted at ``ns = 41, 81, 161``:

    =============  ==========================  ==========================
    quantity       native                      fitted
    =============  ==========================  ==========================
    ``gradgradB``  2.2e-4, 2.5e-5, 1.2e-5      2.9e-2, 2.1e-3, 5.2e-5
    ``d3B``        7.8e-4, 6.0e-4, 6.0e-4      2.1e-1, 2.9e-2, 1.7e-3
    ``div B/|B'|`` 8.5e-18, 4.5e-18, 8.3e-18   1.1e-6, 2.8e-7, 7.1e-8
    =============  ==========================  ==========================

    So ``gradgradB`` clears its 1e-3 gate by a factor of five at the coarsest
    resolution, and ``div B`` is at round-off at every ``ns`` rather than
    converging towards zero.
    """
    points = _oracle_points()
    names = ("B", "gradB", "gradgradB", "gradgradgradB")
    table, divergence = [], []
    for ns in (41, 81, 161):
        field = ext.VmecInteriorField(_oracle_spectra(ns, native=True))
        table.append(_errors(field, points, _exact_B, names))
        gradient = np.asarray(field.gradB(points))
        divergence.append(float(
            np.abs(np.trace(gradient, axis1=1, axis2=2)).max()
            / np.abs(gradient).max()))
    value, first, second, third = (list(column) for column in zip(*table))

    assert max(second) < SECOND_DERIVATIVE_GATE, second
    assert max(third) < THIRD_DERIVATIVE_GATE, third
    assert max(divergence) < DIVERGENCE_GATE, divergence
    # B and its first derivative converge; the higher two are limited by the
    # inversion tolerance rather than by ns, which is why they flatten.
    assert value[0] > value[-1] and first[0] > first[-1], (value, first)


@pytest.mark.full  # nightly: four derivative orders at three resolutions
def test_fitted_fallback_keeps_its_measured_accuracy():
    """What callers get without ``lmns``/``phipf``/``chipf`` in their spectra.

    The native form is the accurate path and every equilibrium built by
    ``_state_field_spectra`` now takes it.  The fitted path remains reachable
    for spectra assembled by hand or carried over from before this change, so
    what it achieves is worth stating rather than leaving to be rediscovered.
    A caller who wants the accurate path supplies the three native fields; on
    a live equilibrium that is automatic.

    Measured against the exact ``B = F/R phi_hat``:

    =================  ========  ========  ========  ========
    quantity           ns = 41   ns = 81   ns = 161  ns = 321
    =================  ========  ========  ========  ========
    ``B``              2.0e-5    4.9e-6    1.2e-6    3.1e-7
    ``gradB``          8.8e-4    2.4e-4    6.0e-5    1.5e-5
    ``gradgradB``      6.8e-1    4.8e-2    1.9e-3    4.6e-4
    ``gradgradgradB``  1.9e+2    2.6e+1    2.6e-1    1.2e-2
    =================  ========  ========  ========  ========

    Two things to read off that, and the second is a cost of this PR.  ``B``
    and its first derivative converge at second order and are what the
    fallback can be trusted for.  The second and third now *converge*, where
    before this PR they were flat at 4.5e-2 and 5.8e-1 at every ``ns`` — but
    they are far worse at coarse resolution, because the C2 interpolant that
    the native form needs differentiates the half-mesh conversion's error
    twice, and the fitted path has no native form to protect it.  A caller
    reading ``gradgradgradB`` off the fallback at ``ns = 41`` gets noise; the
    honest answer for them is to supply the native fields.

    The assertions are ceilings and convergence rather than bands, so an
    improved fallback passes them unchanged and only a regression fails.
    """
    names = ("B", "gradB", "gradgradB", "gradgradgradB")
    points = jnp.asarray(np.stack(
        (MAJOR + FALLBACK_MINOR * np.sqrt(SAMPLES) * np.cos(ANGLE),
         np.zeros_like(SAMPLES),
         FALLBACK_MINOR * np.sqrt(SAMPLES) * np.sin(ANGLE)), axis=1))
    resolutions = (41, 81, 161)
    table = [_errors(ext.VmecInteriorField(_toroidal_spectra(ns)), points,
                     _toroidal_exact, names) for ns in resolutions]
    columns = {name: list(column) for name, column in zip(names, zip(*table))}

    for name, ceiling in FALLBACK_CEILINGS.items():
        worst = columns[name][-1]
        assert worst < ceiling, (
            f"the fitted fallback's {name} error at the finest resolution "
            f"reached {worst:.2e}, above its measured ceiling {ceiling:.0e}: "
            f"{[f'{v:.2e}' for v in columns[name]]} at ns = {resolutions}. "
            "These are ceilings, so they need no editing when the fallback "
            "improves - only a regression trips them.")
    for name in names:
        first, last = columns[name][0], columns[name][-1]
        assert first > 3.0 * last, (
            f"the fitted fallback's {name} error stopped converging with ns: "
            f"{[f'{v:.2e}' for v in columns[name]]}. Before this PR the second "
            "and third derivatives were flat; losing that again would mean the "
            "radial interpolant regressed.")
