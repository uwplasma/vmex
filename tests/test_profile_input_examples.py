from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from vmec_jax.namelist import read_indata
from vmec_jax.profiles import eval_profiles


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "profile_input_examples.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("profile_input_examples", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_input_example_writes_polynomial_and_spline_decks(tmp_path: Path, capsys) -> None:
    module = _load_example_module()

    assert module.main(["--outdir", str(tmp_path)]) == 0

    polynomial_path = tmp_path / "input.profile_polynomial_pressure_current"
    spline_path = tmp_path / "input.profile_spline_pressure_current"
    assert polynomial_path.exists()
    assert spline_path.exists()

    polynomial = read_indata(polynomial_path)
    spline = read_indata(spline_path)

    assert polynomial.get("PMASS_TYPE") == "power_series"
    assert polynomial.get("PCURR_TYPE") == "power_series_i"
    assert polynomial.get_int("NCURR") == 1
    assert spline.get("PMASS_TYPE") == "cubic_spline"
    assert spline.get("PCURR_TYPE") == "akima_spline_i"
    assert spline.get_int("NCURR") == 1

    s = np.linspace(0.0, 1.0, 5)
    polynomial_profiles = eval_profiles(polynomial, s)
    spline_profiles = eval_profiles(spline, s)
    assert np.all(np.asarray(polynomial_profiles["pressure_pa"]) >= 0.0)
    assert np.all(np.asarray(spline_profiles["pressure_pa"]) >= 0.0)
    assert np.asarray(polynomial_profiles["current"])[0] == 0.0
    assert np.asarray(spline_profiles["current"])[0] == 0.0

    out = capsys.readouterr().out
    assert "Run with: vmec " in out
    assert "pressure[Pa]" in out


def test_profile_input_example_decks_evaluate_declared_pressure_and_current_values() -> None:
    module = _load_example_module()
    base = read_indata(module.BASE_INPUT)

    polynomial = module.polynomial_pressure_current_indata(base)
    s = np.asarray([0.0, 0.5, 1.0])
    polynomial_profiles = eval_profiles(polynomial, s)
    pressure_c0, pressure_c1, pressure_c2 = module.POLYNOMIAL_PRESSURE_AM
    current_c0, current_c1 = module.POLYNOMIAL_CURRENT_AC

    np.testing.assert_allclose(
        np.asarray(polynomial_profiles["pressure_pa"]),
        pressure_c0 + pressure_c1 * s + pressure_c2 * s * s,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(polynomial_profiles["current"]),
        current_c0 * s + current_c1 * s * s,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    spline = module.spline_pressure_current_indata(base)
    knots = np.asarray(module.S_KNOTS)
    spline_profiles = eval_profiles(spline, knots)
    np.testing.assert_allclose(
        np.asarray(spline_profiles["pressure_pa"]),
        module.SPLINE_PRESSURE_VALUES,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(spline_profiles["current"]),
        module.SPLINE_CURRENT_VALUES,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
