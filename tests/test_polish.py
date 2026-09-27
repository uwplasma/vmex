"""Force-balance polish driver: scope, configuration, and WOUT export."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmex.core import implicit, solver
from vmex.core.errors import StrongForceCertificationError, VmecInputError
from vmex.core.input import VmecInput
from vmex.core import polish
from vmex.core.polish import (
    PolishConfig,
    polish_legacy_solution,
    polished_wout_input,
    polished_wout_ns,
    polished_wout_state,
    sample_high_order_state,
)
from vmex.core.strong_force import append_high_order_state_modes, certify_strong_force, lift_high_order_state

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _small_solovev_input():
    inp = VmecInput.from_file(DATA / "input.solovev").change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    return dataclasses.replace(
        inp, ns_array=np.asarray([5]), ftol_array=np.asarray([1.0e-10]), niter_array=np.asarray([1000])
    )


@pytest.fixture(scope="module")
def small_lift():
    inp = _small_solovev_input()
    config = implicit.make_config(inp, ftol=1.0e-10, max_iterations=1000)
    params = implicit.params_from_input(inp)
    state, _ = implicit.solve_implicit_with_aux(params, config)
    runtime = implicit.runtime_from_params(params, config)
    return lift_high_order_state(state, runtime, degree=3), state


def test_sample_high_order_state_inverts_the_lift_on_any_mesh(small_lift):
    native, state = small_lift
    inp = _small_solovev_input()
    for ns in (5, 11):
        runtime = solver.prepare_runtime(inp, solver.resolution_from_input(inp, ns=ns))
        sampled = sample_high_order_state(native, runtime)
        assert np.shape(np.asarray(sampled.R_cos)) == (ns, native.m.size)
        relift = lift_high_order_state(sampled, runtime, radial_basis=native.radial_basis, degree=3)
        for name in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"):
            np.testing.assert_allclose(np.asarray(getattr(relift, name)), np.asarray(getattr(native, name)),
                                       rtol=0.0, atol=1.0e-11)
    # The lift pins the edge coefficients, so the solve-mesh sample keeps the boundary.
    runtime = solver.prepare_runtime(inp, solver.resolution_from_input(inp, ns=5))
    sampled = sample_high_order_state(native, runtime)
    np.testing.assert_allclose(np.asarray(sampled.R_cos[-1]), np.asarray(state.R_cos[-1]), rtol=0.0, atol=1.0e-13)


def test_sample_high_order_state_requires_matching_mode_tables(small_lift):
    native, _ = small_lift
    other = VmecInput.from_file(DATA / "input.solovev").change_resolution(mpol=4, ntor=0, ntheta=12, nzeta=4)
    runtime = solver.prepare_runtime(other, solver.resolution_from_input(other, ns=5))
    with pytest.raises(ValueError, match="mode tables"):
        sample_high_order_state(native, runtime)


def test_polished_wout_mesh_and_modes_carry_the_native_state():
    def native(size, breakpoints=None, m_max=3):
        spans = size - 3 if breakpoints is None else len(breakpoints) - 1
        breaks = np.linspace(0.0, 1.0, spans + 1) if breakpoints is None else np.asarray(breakpoints)
        return SimpleNamespace(radial_basis=SimpleNamespace(size=size, breakpoints=breaks), m=np.arange(m_max + 1))

    # Four samples per default reconstruction span, never coarser than the solve.
    assert polished_wout_ns(native(17), solve_ns=31) == 129
    assert polished_wout_ns(native(17), solve_ns=201) == 201
    # Two samples per coefficient, and two uniform-s samples in the narrowest span.
    assert polished_wout_ns(native(90), solve_ns=31) == 181
    assert polished_wout_ns(native(20, [0.0, 0.01, 0.02, 0.5, 1.0]), solve_ns=31) == 201
    # MPOL grows only when padded modes exceed it.
    deck = _small_solovev_input()
    assert polished_wout_input(native(17, m_max=2), deck) is deck
    assert int(polished_wout_input(native(17, m_max=8), deck).mpol) == 9


def test_polish_scope_and_configuration():
    deck = dataclasses.replace(_small_solovev_input(), ncurr=1)
    # Outside the scope: AUTO leaves the solve unpolished, an explicit request refuses.
    assert polish_legacy_solution(deck, None, None, auto=True) is None
    with pytest.raises(VmecInputError, match="axisymmetric"):
        polish_legacy_solution(deck, None, None)
    for updates, message in (({"fail_policy": "maybe"}, "fail_policy"), ({"degree": 4}, "degree"),
                             ({"force_tolerance": 0.0}, "positive")):
        with pytest.raises(ValueError, match=message):
            PolishConfig(**updates)


def test_polish_driver_reports_fails_and_exports_with_a_stub_solver(monkeypatch, small_lift):
    """The driver, the solver hook, and the export around a stubbed polish (the
    real polish runs in the full-polish-gn lane)."""

    native, state = small_lift
    inp = _small_solovev_input()
    window = SimpleNamespace(volume_average_force=2.0, magnetic_relative_force_error=1.0e-3, s_min=0.0, s_max=1.0)
    certificate = SimpleNamespace(absolute_l2=1.0, normalized_l2=0.5, minimum_signed_jacobian=1.0,
                                  window_normalizations=window)
    stub = polish.NativePolishResult(native, 1.0, 1.0e-3, False, 1, 2, 1, 0.1)
    monkeypatch.setattr(polish, "certify_strong_force", lambda _state: certificate)
    monkeypatch.setattr(polish, "force_error_measures", lambda *_: ())
    monkeypatch.setattr(polish, "physical_scales", lambda _state: (1.0, 1.0))
    monkeypatch.setattr(polish, "polish_native", lambda *_, **__: stub)
    resolution = solver.resolution_from_input(inp, ns=5)
    lines = []
    fallback = polish_legacy_solution(inp, resolution, state, verbose=True,
                                      config=PolishConfig(degree=3, fail_policy="return_unpolished"),
                                      emit=lambda *parts, **_: lines.append(" ".join(map(str, parts))))
    report = fallback.polish_report
    assert not report.converged and report.nonlinear_iterations == 3 and report.normalization_window == (0.0, 1.0)
    assert "native polish: |F|_rms" in "".join(lines) and "FAILED" in "".join(lines)
    with pytest.raises(StrongForceCertificationError, match="did not converge"):
        polish_legacy_solution(inp, resolution, state, config=PolishConfig(degree=3))
    exported = polished_wout_state(fallback.native_equilibrium, inp, solve_ns=5)
    assert np.shape(np.asarray(exported.R_cos))[0] == polished_wout_ns(fallback.native_equilibrium, solve_ns=5)
    # The solver hook: an out-of-scope AUTO keeps the result, a polish replaces it.
    base = SimpleNamespace(state=state)
    monkeypatch.setattr(solver, "replace", lambda result, **fields: {**vars(result), **fields})
    for returned, check in ((None, lambda out: out is base), (fallback, lambda out: out["polish_report"] is report)):
        monkeypatch.setattr(polish, "polish_legacy_solution", lambda *_, _r=returned, **__: _r)
        assert check(solver._polish_solve_result(inp, resolution, base, polish="auto", polish_config=None,
                                                 lconm1=True, verbose=True, emit=lambda *_, **__: None))


def test_polish_building_blocks_reject_malformed_requests(small_lift):
    native, _ = small_lift
    with pytest.raises(ValueError, match="radial_order"):
        polish.make_variational_plan(native, radial_order=1)
    with pytest.raises(ValueError, match="angular grid"):
        polish.make_variational_plan(native, ntheta=0)
    with pytest.raises(ValueError, match="force coordinates"):
        polish._chart(native, 1.0, 1.0).force(np.zeros(3))
    with pytest.raises(ValueError, match="nonzero length"):
        append_high_order_state_modes(native, [], [])
    with pytest.raises(ValueError, match="angular shifts"):
        certify_strong_force(native, theta_shift=1.0)


def test_public_solver_rejects_unknown_polish_mode_before_solving():
    inp = VmecInput.from_file(DATA / "input.solovev")
    with pytest.raises(ValueError, match="False, True, or 'auto'"):
        solver.solve(inp, polish="unknown")


def test_public_solver_resolves_polish_keywords_only():
    """Directives live in run_options; the solver sees only its keywords."""
    inp = VmecInput.from_file(DATA / "input.solovev")
    assert solver._resolve_force_balance_polish(inp, None, None) is False
    assert solver._resolve_force_balance_polish(inp, True, None) is True
    assert solver._resolve_force_balance_polish(inp, "auto", None) == "auto"
    with pytest.raises(ValueError, match="either polish or polish_force_balance"):
        solver._resolve_force_balance_polish(inp, False, True)
