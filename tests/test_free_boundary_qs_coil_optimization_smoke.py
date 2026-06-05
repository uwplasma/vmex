from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "examples" / "optimization" / "free_boundary_QS_coil_optimization.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("free_boundary_qs_coil_optimization_example", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_objective_terms_report_weighted_proxy_components():
    module = _load_example_module()

    summary = {
        "residual_proxy": 2.0,
        "qs_total": 0.25,
        "qs_helicity_m": 1,
        "qs_helicity_n": 0,
        "qs_surfaces": [0.25, 0.5],
        "aspect": 5.5,
        "target_aspect": 6.0,
        "mean_iota": 0.3,
        "target_iota": 0.4,
    }

    terms = module.objective_terms_from_summary(
        summary,
        residual_weight=3.0,
        qs_weight=4.0,
        aspect_weight=0.5,
        iota_weight=10.0,
    )

    assert terms["residual"]["contribution"] == pytest.approx(6.0)
    assert terms["quasisymmetry"]["contribution"] == pytest.approx(1.0)
    assert terms["quasisymmetry"]["helicity_m"] == 1
    assert terms["quasisymmetry"]["surfaces"] == [0.25, 0.5]
    assert terms["aspect"]["error"] == pytest.approx(-0.5)
    assert terms["aspect"]["contribution"] == pytest.approx(0.125)
    assert terms["mean_iota"]["contribution"] == pytest.approx(0.1)
    assert terms["total"] == pytest.approx(7.225)
    assert module.objective_from_summary(
        summary,
        residual_weight=3.0,
        qs_weight=4.0,
        aspect_weight=0.5,
        iota_weight=10.0,
    ) == pytest.approx(terms["total"])


def test_objective_terms_report_missing_unweighted_proxy_components():
    module = _load_example_module()

    terms = module.objective_terms_from_summary(
        {
            "residual_proxy": 2.0,
            "qs_total": None,
            "aspect": None,
            "target_aspect": 6.0,
            "mean_iota": None,
            "target_iota": 0.4,
        },
        residual_weight=3.0,
        qs_weight=4.0,
        aspect_weight=0.5,
        iota_weight=10.0,
    )

    assert terms["total"] == pytest.approx(6.0)
    assert terms["missing_unweighted_terms"] == ["qs_total", "aspect", "mean_iota"]
    assert terms["quasisymmetry"]["contribution"] == pytest.approx(0.0)
    assert terms["aspect"]["contribution"] == pytest.approx(0.0)
    assert terms["mean_iota"]["contribution"] == pytest.approx(0.0)


def test_circle_variable_manifest_and_apply_are_coil_only():
    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)
    _x0, variables = module.select_coil_variables(
        base_params,
        max_current_vars=1,
        max_fourier_vars=1,
    )

    manifest = module.variable_records(
        variables,
        base_params,
        current_step=0.1,
        dof_step=0.5,
    )
    perturbed = module.apply_coil_variables(
        base_params,
        np.asarray([1.0, -2.0]),
        variables,
        current_step=0.1,
        dof_step=0.5,
    )

    assert [record["kind"] for record in manifest] == ["current", "fourier_dof"]
    assert all(record["kind"] in {"current", "fourier_dof"} for record in manifest)
    assert manifest[0]["parameterization"] == "multiplicative"
    assert manifest[0]["unit_x_delta"] == pytest.approx(0.2)
    assert manifest[1]["parameterization"] == "additive"
    assert manifest[1]["unit_x_delta"] == pytest.approx(0.5)
    assert float(np.asarray(perturbed.base_currents)[0]) == pytest.approx(2.2)
    assert float(np.asarray(base_params.base_currents)[0]) == pytest.approx(2.0)
    assert float(np.asarray(perturbed.base_curve_dofs)[variables[1][1]]) == pytest.approx(0.4)
    assert float(np.asarray(base_params.base_curve_dofs)[variables[1][1]]) == pytest.approx(1.4)
    assert perturbed.n_segments == base_params.n_segments
    assert perturbed.nfp == base_params.nfp
    assert perturbed.stellsym == base_params.stellsym


def test_same_branch_direction_selects_current_and_fourier_variables():
    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)

    direction = module.same_branch_direction_from_variables(
        [
            ("current", (0,)),
            ("fourier_dof", (0, 0, 2)),
            ("fourier_dof", (0, 1, 1)),
        ]
    )

    np.testing.assert_array_equal(direction, np.asarray([1.0, 1.0, 0.0]))
    tangent = module.coil_param_direction_from_variables(
        base_params,
        direction,
        [
            ("current", (0,)),
            ("fourier_dof", (0, 0, 2)),
            ("fourier_dof", (0, 1, 1)),
        ],
        current_step=0.02,
        dof_step=1.0e-3,
    )
    np.testing.assert_allclose(np.asarray(tangent.base_currents), np.asarray([0.04]))
    assert float(np.asarray(tangent.base_curve_dofs)[0, 0, 2]) == pytest.approx(1.0e-3)
    assert float(np.asarray(tangent.base_curve_dofs)[0, 1, 1]) == pytest.approx(0.0)


def test_same_branch_report_writer_uses_source_helper(tmp_path, monkeypatch):
    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)
    _x0, variables = module.select_coil_variables(
        base_params,
        max_current_vars=1,
        max_fourier_vars=1,
    )
    args = SimpleNamespace(
        current_step=0.02,
        dof_step=1.0e-3,
        target_aspect=6.0,
        target_iota=0.4,
        helicity_m=1,
        helicity_n=0,
        qs_surfaces="0.25,0.5",
        qs_ntheta=15,
        qs_nphi=16,
        residual_weight=1.0,
        qs_weight=2.0,
        aspect_weight=1.0e-2,
        iota_weight=1.0,
        same_branch_report_eps=1.0e-4,
        same_branch_report_max_iter=3,
        vmec_max_iter=2,
        ftol=1.0e-8,
        jit_forces=False,
        activate_fsq=1.0e99,
    )
    calls = []

    def fake_report(input_path, params, *, params_for, objective_fn, eps, solve_kwargs):
        calls.append(
            {
                "input_path": input_path,
                "params": params,
                "plus_current": float(np.asarray(params_for(eps).base_currents)[0]),
                "minus_current": float(np.asarray(params_for(-eps).base_currents)[0]),
                "eps": eps,
                "solve_kwargs": solve_kwargs,
            }
        )
        return {
            "branch_compatibility": {
                "same_branch": True,
                "plus": {
                    "changed_fields": (),
                    "max_abs_scalar_delta": 0.0,
                    "max_rel_scalar_delta": 0.0,
                },
                "minus": {
                    "changed_fields": (),
                    "max_abs_scalar_delta": 0.0,
                    "max_rel_scalar_delta": 0.0,
                },
            },
            "values": {
                "base": 1.0,
                "plus": 1.1,
                "minus": 0.9,
                "central_fd_directional": 1000.0,
            },
            "objective_values": {
                "objective": {
                    "base": 1.0,
                    "plus": 1.1,
                    "minus": 0.9,
                    "central_fd_directional": 1000.0,
                },
                "qs_total": {
                    "base": 0.5,
                    "plus": 0.6,
                    "minus": 0.4,
                    "central_fd_directional": 1000.0,
                },
                "aspect": {
                    "base": 6.0,
                    "plus": 6.1,
                    "minus": 5.9,
                    "central_fd_directional": 1000.0,
                },
            },
            "primary_objective": "objective",
        }

    import vmec_jax.free_boundary_adjoint as freeb_adj

    monkeypatch.setattr(freeb_adj, "direct_coil_same_branch_complete_solve_fd_report", fake_report)
    path = module.write_same_branch_validation_report(
        input_path=tmp_path / "input.direct",
        base_params=base_params,
        variables=variables,
        args=args,
        outdir=tmp_path,
    )

    assert path == tmp_path / "same_branch_complete_solve_report.json"
    assert calls
    assert calls[0]["solve_kwargs"]["max_iter"] == 3
    assert calls[0]["solve_kwargs"]["jit_forces"] is False
    assert calls[0]["plus_current"] > float(np.asarray(base_params.base_currents)[0])
    assert calls[0]["minus_current"] < float(np.asarray(base_params.base_currents)[0])
    report = json.loads(path.read_text())
    assert report["branch_compatibility"]["same_branch"] is True
    assert report["values"]["central_fd_directional"] == pytest.approx(1000.0)
    assert set(report["objective_values"]) == {"objective", "qs_total", "aspect"}
    assert report["primary_objective"] == "objective"
    assert report["branch_local_vector_jacobian"]["available"] is False
    assert "adaptive host branch" in report["branch_local_vector_jacobian"]["scope"]
    assert [record["kind"] for record in report["direction_variables"]] == ["current", "fourier_dof"]


def test_same_branch_report_writer_records_branch_local_vector_jacobian(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)
    _x0, variables = module.select_coil_variables(
        base_params,
        max_current_vars=1,
        max_fourier_vars=1,
    )
    args = SimpleNamespace(
        current_step=0.02,
        dof_step=1.0e-3,
        target_aspect=6.0,
        target_iota=0.4,
        helicity_m=1,
        helicity_n=0,
        qs_surfaces="0.25,0.5",
        qs_ntheta=15,
        qs_nphi=16,
        residual_weight=1.0,
        qs_weight=2.0,
        aspect_weight=1.0e-2,
        iota_weight=1.0,
        same_branch_report_eps=1.0e-4,
        same_branch_report_max_iter=3,
        vmec_max_iter=2,
        ftol=1.0e-8,
        jit_forces=False,
        activate_fsq=1.0e99,
    )

    def fake_report(*_args, **_kwargs):
        return {
            "base": {"traces": ("synthetic-trace",)},
            "branch_compatibility": {
                "same_branch": True,
                "plus": {"changed_fields": (), "max_abs_scalar_delta": 0.0, "max_rel_scalar_delta": 0.0},
                "minus": {"changed_fields": (), "max_abs_scalar_delta": 0.0, "max_rel_scalar_delta": 0.0},
            },
            "values": {"base": 1.0, "plus": 1.1, "minus": 0.9, "central_fd_directional": 1000.0},
            "objective_values": {
                "objective": {"base": 1.0, "plus": 1.1, "minus": 0.9, "central_fd_directional": 1000.0},
                "aspect": {"base": 6.0, "plus": 6.1, "minus": 5.9, "central_fd_directional": 0.1},
                "qs_total": {"base": 0.4, "plus": 0.42, "minus": 0.38, "central_fd_directional": 0.4},
                "lcfs_boundary_moment": {
                    "base": 0.2,
                    "plus": 0.21,
                    "minus": 0.19,
                    "central_fd_directional": 0.2,
                },
                "accepted_bnormal_rms": {
                    "base": 0.3,
                    "plus": 0.31,
                    "minus": 0.29,
                    "central_fd_directional": 0.3,
                },
            },
            "primary_objective": "objective",
        }

    direction_x = module.same_branch_direction_from_variables(variables)
    direction_params = module.coil_param_direction_from_variables(
        base_params,
        direction_x,
        variables,
        current_step=args.current_step,
        dof_step=args.dof_step,
    )
    output_count = 4
    jacobian = jax.tree_util.tree_map(
        lambda leaf: jnp.zeros((output_count,) + jnp.asarray(leaf).shape),
        direction_params,
    )

    def fake_branch_local_vector(*_args, **_kwargs):
        return {
            "uses_production_forward": True,
            "differentiates_adaptive_controller": False,
            "differentiates_run_free_boundary": False,
            "differentiates_fixed_accepted_branch": True,
            "scalar_keys": ("aspect", "qs_total", "lcfs_boundary_moment", "accepted_bnormal_rms"),
            "replay_option_flags": {"use_stacked_step_controls": True},
            "max_base_abs_delta": 0.0,
            "values": {
                "aspect": 6.0,
                "qs_total": 0.4,
                "lcfs_boundary_moment": 0.2,
                "accepted_bnormal_rms": 0.3,
            },
            "replay_value_map": {
                "aspect": jnp.asarray(6.0),
                "qs_total": jnp.asarray(0.4),
                "lcfs_boundary_moment": jnp.asarray(0.2),
                "accepted_bnormal_rms": jnp.asarray(0.3),
            },
            "base_abs_delta": {
                "aspect": 0.0,
                "qs_total": 0.0,
                "lcfs_boundary_moment": 0.0,
                "accepted_bnormal_rms": 0.0,
            },
            "jacobian": jacobian,
        }

    import vmec_jax.free_boundary_adjoint as freeb_adj

    monkeypatch.setattr(freeb_adj, "direct_coil_same_branch_complete_solve_fd_report", fake_report)
    monkeypatch.setattr(
        freeb_adj,
        "direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax",
        fake_branch_local_vector,
    )

    path = module.write_same_branch_validation_report(
        input_path=tmp_path / "input.direct",
        base_params=base_params,
        variables=variables,
        args=args,
        outdir=tmp_path,
    )

    report = json.loads(path.read_text())
    vector = report["branch_local_vector_jacobian"]
    assert vector["available"] is True
    assert vector["uses_production_forward"] is True
    assert vector["differentiates_adaptive_controller"] is False
    assert vector["differentiates_run_free_boundary"] is False
    assert vector["differentiates_fixed_accepted_branch"] is True
    assert vector["scalar_keys"] == ["aspect", "qs_total", "lcfs_boundary_moment", "accepted_bnormal_rms"]
    assert vector["replay_option_flags"]["use_stacked_step_controls"] is True
    assert vector["max_base_abs_delta"] == pytest.approx(0.0)
    assert vector["scalars"]["aspect"]["complete_fd_directional"] == pytest.approx(0.1)
    assert vector["scalars"]["qs_total"]["complete_fd_directional"] == pytest.approx(0.4)


def test_circle_dry_run_writes_configuration_without_solves(tmp_path, monkeypatch):
    module = _load_example_module()

    def fake_make_free_boundary_indata(_input_path, output_path, **_kwargs):
        output_path.write_text("&INDATA\n/\n")
        return output_path

    def fail_run_direct_free_boundary(*_args, **_kwargs):
        raise AssertionError("dry-run must not call run_direct_free_boundary")

    def fail_minimize(*_args, **_kwargs):
        raise AssertionError("dry-run must not call scipy.optimize.minimize")

    def fail_write_wout(*_args, **_kwargs):
        raise AssertionError("dry-run must not write a best wout")

    monkeypatch.setattr(module, "make_free_boundary_indata", fake_make_free_boundary_indata)
    monkeypatch.setattr(module, "run_direct_free_boundary", fail_run_direct_free_boundary)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", fail_write_wout)
    fake_scipy_optimize = ModuleType("scipy.optimize")
    fake_scipy_optimize.minimize = fail_minimize
    monkeypatch.setitem(sys.modules, "scipy.optimize", fake_scipy_optimize)

    exit_code = module.main(
        [
            "--smoke",
            "--dry-run",
            "--provider",
            "circle",
            "--helicity-n",
            "-1",
            "--qs-surfaces",
            "0.3,0.7",
            "--outdir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "history.json").exists()
    assert not (tmp_path / "wout_best_direct_coil_qs.nc").exists()
    assert (tmp_path / "input.direct_coil_qs").read_text() == "&INDATA\n/\n"

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "optimizer" not in summary
    assert "best" not in summary
    assert summary["scope"] == "deterministic coil-only direct-coil free-boundary QS optimization example"
    assert summary["dry_run"] is True
    assert summary["plasma_boundary_optimized"] is False
    assert any("Boozer-space" in limitation for limitation in summary["single_stage_limitations"])
    assert any("full-loop" in limitation for limitation in summary["single_stage_limitations"])
    assert summary["provider"]["provider"] == "circle"
    assert summary["baseline_coils"]["n_base_coils"] == 1
    assert summary["vmec_config"]["external_field_provider_kind"] == "direct_coils"
    assert summary["vmec_config"]["mgrid_file"] == "DIRECT_COILS"
    assert summary["vmec_config"]["uses_generated_mgrid"] is False
    assert summary["vmec_config"]["vmec_max_iter"] == 2
    assert summary["vmec_config"]["jit_forces"] is True
    assert [record["kind"] for record in summary["optimized_variables"]] == ["current", "fourier_dof"]
    assert summary["optimized_variables"][0]["parameterization"] == "multiplicative"
    assert summary["optimized_variables"][1]["parameterization"] == "additive"
    assert summary["objective_model"]["target_aspect"] == pytest.approx(6.0)
    assert summary["objective_model"]["helicity_n"] == -1
    assert summary["objective_model"]["qs_surfaces"] == [0.3, 0.7]


def test_essos_provider_skip_returns_code_77_without_solves(tmp_path, monkeypatch, capsys):
    module = _load_example_module()

    def fake_load_essos_provider(*_args, **_kwargs):
        raise module.SkipExample("synthetic missing ESSOS assets")

    def fail_make_free_boundary_indata(*_args, **_kwargs):
        raise AssertionError("ESSOS skip must happen before input generation")

    monkeypatch.setattr(module, "load_essos_provider", fake_load_essos_provider)
    monkeypatch.setattr(module, "make_free_boundary_indata", fail_make_free_boundary_indata)

    exit_code = module.main(
        [
            "--smoke",
            "--dry-run",
            "--provider",
            "essos",
            "--outdir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == module.SKIP_EXIT_CODE
    assert "SKIP: synthetic missing ESSOS assets" in captured.err
    assert not (tmp_path / "summary.json").exists()


def test_essos_dry_run_writes_direct_coil_configuration_without_mgrid(tmp_path, monkeypatch):
    module = _load_example_module()
    synthetic_params, _metadata = module.make_circle_provider(current_scale=1.0)

    def fake_load_essos_provider(coils_json, *, chunk_size, current_scale):
        assert coils_json is None
        assert chunk_size == 128
        assert current_scale == pytest.approx(1.25)
        return synthetic_params, {
            "provider": "essos",
            "coils_json": "/synthetic/ESSOS_biot_savart_LandremanPaulQA.json",
            "n_base_coils": 1,
            "n_segments": int(synthetic_params.n_segments),
            "nfp": int(synthetic_params.nfp),
            "stellsym": bool(synthetic_params.stellsym),
            "current_scale_multiplier": 1.25,
        }

    def fake_make_free_boundary_indata(_input_path, output_path, **kwargs):
        output_path.write_text("&INDATA\n  LFREEB = T\n  MGRID_FILE = 'DIRECT_COILS'\n/\n")
        assert kwargs["vmec_max_iter"] == 2
        assert kwargs["ftol"] == pytest.approx(1.0e-8)
        return output_path

    def fail_run_direct_free_boundary(*_args, **_kwargs):
        raise AssertionError("dry-run must not call run_direct_free_boundary")

    def fail_minimize(*_args, **_kwargs):
        raise AssertionError("dry-run must not call scipy.optimize.minimize")

    def fail_write_wout(*_args, **_kwargs):
        raise AssertionError("dry-run must not write a best wout")

    monkeypatch.setattr(module, "load_essos_provider", fake_load_essos_provider)
    monkeypatch.setattr(module, "make_free_boundary_indata", fake_make_free_boundary_indata)
    monkeypatch.setattr(module, "run_direct_free_boundary", fail_run_direct_free_boundary)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", fail_write_wout)
    fake_scipy_optimize = ModuleType("scipy.optimize")
    fake_scipy_optimize.minimize = fail_minimize
    monkeypatch.setitem(sys.modules, "scipy.optimize", fake_scipy_optimize)

    exit_code = module.main(
        [
            "--smoke",
            "--dry-run",
            "--provider",
            "essos",
            "--chunk-size",
            "128",
            "--current-scale",
            "1.25",
            "--max-current-vars",
            "1",
            "--max-fourier-vars",
            "1",
            "--outdir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "history.json").exists()
    assert not (tmp_path / "wout_best_direct_coil_qs.nc").exists()
    generated_input = tmp_path / "input.direct_coil_qs"
    assert "DIRECT_COILS" in generated_input.read_text()

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["dry_run"] is True
    assert summary["plasma_boundary_optimized"] is False
    assert summary["provider"]["provider"] == "essos"
    assert summary["provider"]["coils_json"].endswith("ESSOS_biot_savart_LandremanPaulQA.json")
    assert summary["baseline_coils"]["n_base_coils"] == 1
    assert summary["vmec_config"]["generated_input"].endswith("input.direct_coil_qs")
    assert summary["vmec_config"]["generated_input"] == str(generated_input)
    assert summary["vmec_config"]["external_field_provider_kind"] == "direct_coils"
    assert summary["vmec_config"]["mgrid_file"] == "DIRECT_COILS"
    assert summary["vmec_config"]["uses_generated_mgrid"] is False
    assert "generated_mgrid" not in summary["vmec_config"]
    assert [record["kind"] for record in summary["optimized_variables"]] == ["current", "fourier_dof"]
    assert all(record["kind"] != "boundary" for record in summary["optimized_variables"])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Full coil -> direct-coil free-boundary solve -> Boozer/QS exact "
        "gradient validation is phase 2; current tests validate provider, "
        "projection, dense-vacuum, and dense mode-space adjoint pieces."
    ),
)
def test_full_free_boundary_qs_exact_gradient_validation_phase2_marker():
    raise NotImplementedError("production NESTOR/QS exact-gradient validation is not promoted yet")


def test_deterministic_circle_smoke_records_qs_terms(tmp_path, monkeypatch):
    module = _load_example_module()
    calls = []

    def fake_make_free_boundary_indata(_input_path, output_path, **_kwargs):
        output_path.write_text("&INDATA\n/\n")
        return output_path

    def fake_run_direct_free_boundary(input_path, params, *, vmec_max_iter, activate_fsq, jit_forces=True):
        calls.append(
            {
                "input_path": input_path,
                "current": float(np.asarray(params.base_currents)[0]),
                "vmec_max_iter": vmec_max_iter,
                "activate_fsq": activate_fsq,
                "jit_forces": bool(jit_forces),
            }
        )
        return SimpleNamespace(), 0.01

    def fake_summarize_run(
        _run,
        params,
        *,
        objective,
        wall_s,
        target_aspect,
        target_iota,
        helicity_m,
        helicity_n,
        qs_surfaces,
        qs_ntheta,
        qs_nphi,
    ):
        current = float(np.asarray(params.base_currents)[0])
        return {
            "objective": objective,
            "wall_s": wall_s,
            "vmec_n_iter": 1,
            "fsqr": current,
            "fsqz": 0.0,
            "fsql": 0.0,
            "residual_proxy": current,
            "qs_total": 0.25,
            "qs_helicity_m": helicity_m,
            "qs_helicity_n": helicity_n,
            "qs_surfaces": qs_surfaces,
            "qs_ntheta": qs_ntheta,
            "qs_nphi": qs_nphi,
            "aspect": target_aspect,
            "target_aspect": target_aspect,
            "mean_iota": target_iota,
            "target_iota": target_iota,
            "coil_current_norm": abs(current),
            "mean_coil_length": 1.0,
            "vmec_history": {"w": [], "fsqr2": [], "fsqz2": [], "fsql2": []},
        }

    def fake_write_wout(path, _run, *, include_fsq):
        path.write_text(f"include_fsq={include_fsq}\n")

    monkeypatch.setattr(module, "make_free_boundary_indata", fake_make_free_boundary_indata)
    monkeypatch.setattr(module, "run_direct_free_boundary", fake_run_direct_free_boundary)
    monkeypatch.setattr(module, "summarize_run", fake_summarize_run)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", fake_write_wout)

    exit_code = module.main(
        [
            "--smoke",
            "--provider",
            "circle",
            "--max-evals",
            "1",
            "--max-iter",
            "1",
            "--qs-weight",
            "4.0",
            "--helicity-n",
            "-1",
            "--outdir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert all(call["jit_forces"] is True for call in calls)
    assert calls[0]["current"] == pytest.approx(2.0)

    history = json.loads((tmp_path / "history.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert len(history) == 1
    assert history[0]["variables"][0]["parameterization"] == "multiplicative"
    assert history[0]["coil_diagnostics"]["n_base_coils"] == 1
    assert history[0]["summary"]["objective_terms"]["residual"]["contribution"] == pytest.approx(2.0)
    assert history[0]["summary"]["objective_terms"]["quasisymmetry"]["contribution"] == pytest.approx(1.0)
    assert history[0]["summary"]["objective_terms"]["total"] == pytest.approx(3.0)
    assert summary["dry_run"] is False
    assert summary["scope"] == "deterministic coil-only direct-coil free-boundary QS optimization example"
    assert summary["baseline_coils"]["n_base_coils"] == 1
    assert summary["optimized_variables"][0]["unit_x_delta"] == pytest.approx(0.04)
    assert summary["objective_model"]["qs_weight"] == pytest.approx(4.0)
    assert summary["objective_model"]["helicity_n"] == -1
    assert (tmp_path / "wout_best_direct_coil_qs.nc").read_text() == "include_fsq=True\n"
