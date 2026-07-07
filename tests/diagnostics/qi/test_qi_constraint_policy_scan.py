from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "diagnostics"
    / "qi"
    / "qi_constraint_policy_scan.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("qi_constraint_policy_scan", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_policy_matrix_covers_requested_axes():
    mod = _load_module()

    policies = {policy.name: policy for policy in mod.default_policies(max_nfev=1)}

    assert {
        "large_mirror_weights",
        "staged_mirror_relax_tight",
        "softplus_barriers",
        "scalar_trust_qi_iota",
        "scipy_qi_iota",
        "matrix_free_qi_iota",
        "mode_continuation_repeat",
        "augmented_lagrangian_mirror",
    } <= set(policies)
    assert policies["large_mirror_weights"].stages[0].mirror_weight >= 50.0
    staged = policies["staged_mirror_relax_tight"].stages
    assert len(staged) == 2
    assert staged[0].mirror_threshold > staged[1].mirror_threshold
    assert policies["softplus_barriers"].stages[0].qi_ceiling_weight > 0.0
    assert policies["scalar_trust_qi_iota"].stages[0].method == "scalar_trust"
    assert policies["scipy_qi_iota"].stages[0].method == "scipy"
    assert policies["matrix_free_qi_iota"].stages[0].method == "scipy_matrix_free"
    assert policies["mode_continuation_repeat"].stages[0].stage_modes == (2, 3, 3)
    al_stage = policies["augmented_lagrangian_mirror"].stages[0]
    assert al_stage.use_augmented_lagrangian is True
    assert al_stage.al_mirror_penalty > 1.0
    assert al_stage.al_elongation_penalty > 1.0
    assert al_stage.iota_weight > al_stage.qi_weight


def test_cli_dry_run_writes_bounded_plan(tmp_path):
    mod = _load_module()

    assert mod.main(["--out-root", str(tmp_path), "--policy", "scipy_qi_iota", "--max-nfev", "1"]) == 0

    plan = (tmp_path / "plan.json").read_text()
    assert "input.QI_stel_seed_3127" in plan
    assert "scipy_qi_iota" in plan
    assert '"max_nfev": 1' in plan
