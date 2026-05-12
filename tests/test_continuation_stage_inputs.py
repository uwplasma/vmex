from __future__ import annotations

import numpy as np

from vmec_jax.config import config_from_indata
from vmec_jax.namelist import InData
from vmec_jax.optimization_workflow import build_fixed_boundary_objective_stage


def _with_extra_high_mode(indata):
    indexed = {name: dict(values) for name, values in indata.indexed.items()}
    indexed.setdefault("RBC", {})[(2, 0)] = 0.123
    indexed.setdefault("ZBS", {})[(2, 0)] = 0.456
    return InData(
        scalars=dict(indata.scalars),
        indexed=indexed,
        source_path=indata.source_path,
    )


def test_projected_continuation_stage_keeps_high_modes_zero(load_case_circular_tokamak):
    _cfg, base_indata, _static, _boundary, _state0 = load_case_circular_tokamak
    indata = _with_extra_high_mode(base_indata)

    stage1 = build_fixed_boundary_objective_stage(
        config_from_indata(indata),
        indata,
        stage_mode=1,
        objectives=[],
        project_input_boundary_to_max_mode=True,
        inner_max_iter=1,
        trial_max_iter=1,
    )

    assert (2, 0) not in stage1.ctx.indata.indexed.get("RBC", {})
    assert (2, 0) not in stage1.ctx.indata.indexed.get("ZBS", {})

    next_indata = stage1.optimizer._indata_from_params(np.zeros(len(stage1.specs)))
    assert next_indata.indexed["RBC"][(2, 0)] == 0.0
    assert next_indata.indexed["ZBS"][(2, 0)] == 0.0

    stage2 = build_fixed_boundary_objective_stage(
        config_from_indata(next_indata),
        next_indata,
        stage_mode=2,
        objectives=[],
        project_input_boundary_to_max_mode=True,
        inner_max_iter=1,
        trial_max_iter=1,
    )

    assert stage2.ctx.indata.indexed["RBC"][(2, 0)] == 0.0
    assert stage2.ctx.indata.indexed["ZBS"][(2, 0)] == 0.0

