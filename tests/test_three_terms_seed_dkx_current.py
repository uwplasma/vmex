"""The three-terms driver's --seed-run DKX current check (``seed_dkx_current``), with the solve and DKX stubbed."""
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "optimization"


@pytest.fixture
def driver():
    previous = os.environ.get("COIL_CASE")
    os.environ["COIL_CASE"] = "qi6-beta"
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        sys.modules.pop("single_stage_free_boundary_optimization_three_terms", None)
        yield importlib.import_module("single_stage_free_boundary_optimization_three_terms")
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("single_stage_free_boundary_optimization_three_terms", None)
        if previous is None:
            os.environ.pop("COIL_CASE", None)
        else:
            os.environ["COIL_CASE"] = previous


def _stub(monkeypatch, driver, mismatches):
    """Equilibria tagged by the deck's CURTOR; DKX's mismatch looked up by that tag; the Picard current doubles it."""
    from vmex import optimize as opt

    calls = []
    monkeypatch.setattr(opt, "solve_equilibrium", lambda inp, **kw: SimpleNamespace(
        state=float(inp.curtor), solver_context=None))
    monkeypatch.setattr(driver, "dkx_mismatch", lambda inp: lambda state, runtime: mismatches[state])

    def picard(inp, fixed, device):
        calls.append(float(inp.curtor))
        return driver.replace(inp, curtor=2 * float(inp.curtor))

    monkeypatch.setattr(driver, "dkx_current", picard)
    return calls


def _deck(driver):
    import vmex as vj

    deck = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.rotating_ellipse_nfp2"
    return driver.replace(vj.VmecInput.from_file(str(deck)), curtor=1.0e4, lfreeb=False)


def test_seed_current_within_tolerance_is_kept(monkeypatch, driver):
    calls = _stub(monkeypatch, driver, {1.0e4: 0.5 * driver.REDL_TOLERANCE})
    inp = _deck(driver)
    out, refit = driver.seed_dkx_current(inp, None)
    assert out is inp and refit is None
    assert calls == []


def test_seed_current_over_tolerance_is_replaced_by_dkx_picard(monkeypatch, driver):
    # the QI A8 campaign seed: 3.2e-2 against REDL_TOLERANCE 1e-3 (DKX 2.9.0); the Picard current is DKX's own
    calls = _stub(monkeypatch, driver, {1.0e4: 3.2e-2, 2.0e4: 8.4e-5})
    out, refit = driver.seed_dkx_current(_deck(driver), None)
    assert calls == [1.0e4]
    assert refit.state == 2.0e4  # the new current's fixed-boundary equilibrium, for the coil refit
    assert float(out.curtor) == 2.0e4
    assert out.lfreeb is False
