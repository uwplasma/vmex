from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.driver import example_paths, load_example, run_fixed_boundary, save_npz
from vmec_jax.vmec_tomnsp import vmec_angle_grid


ROOT = Path(__file__).resolve().parents[1]


def test_example_paths_reports_missing_wout_as_none(tmp_path):
    data_dir = tmp_path / "examples" / "data"
    data_dir.mkdir(parents=True)
    input_path = data_dir / "input.synthetic"
    input_path.write_text("&INDATA\n/\n")

    actual_input, actual_wout = example_paths("synthetic", root=tmp_path)

    assert actual_input == input_path
    assert actual_wout is None


def test_load_example_without_wout_skips_optional_wout_read():
    example = load_example("circular_tokamak", root=ROOT, with_wout=False)

    assert example.input_path.exists()
    if example.wout_path is not None:
        assert example.wout_path.name == "wout_circular_tokamak.nc"
    assert example.wout is None
    assert example.state is None
    assert example.static.cfg.ns == example.cfg.ns


def test_save_npz_creates_parent_and_preserves_arrays(tmp_path):
    path = save_npz(tmp_path / "nested" / "demo.npz", a=np.asarray([1, 2, 3]), b=np.asarray([[4.0], [5.0]]))

    with np.load(path) as data:
        np.testing.assert_array_equal(data["a"], np.asarray([1, 2, 3]))
        np.testing.assert_allclose(data["b"], np.asarray([[4.0], [5.0]]))


def test_run_fixed_boundary_initial_guess_verbose_vmec2000_mode(capsys):
    grid = vmec_angle_grid(ntheta=8, nzeta=1, nfp=1, lasym=False)

    run = run_fixed_boundary(
        ROOT / "examples" / "data" / "input.circular_tokamak",
        solver="vmec2000_iter",
        max_iter=1,
        use_initial_guess=True,
        vmec_project=False,
        verbose=True,
        grid=grid,
    )

    out = capsys.readouterr().out
    assert "fixed-boundary run (initial guess)" in out
    assert "max_iter=" not in out
    assert run.result is None
    assert run.state is not None


def test_run_fixed_boundary_unknown_solver_raises_before_solver_dispatch():
    grid = vmec_angle_grid(ntheta=8, nzeta=1, nfp=1, lasym=False)

    with pytest.raises(ValueError, match="Unknown solver"):
        run_fixed_boundary(
            ROOT / "examples" / "data" / "input.circular_tokamak",
            solver="bogus",
            max_iter=1,
            ns_override=3,
            vmec_project=False,
            verbose=False,
            grid=grid,
        )
