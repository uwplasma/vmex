from __future__ import annotations

import pytest

from vmec_jax.booz_input import booz_xform_inputs_from_state
from vmec_jax.driver import example_paths
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout
from vmec_jax.config import load_config


def test_booz_xform_inputs_from_state_shapes():
    pytest.importorskip("netCDF4")

    input_path, wout_path = example_paths("circular_tokamak")
    if wout_path is None:
        pytest.skip("No reference wout file available for circular_tokamak")

    cfg, indata = load_config(str(input_path))
    static = build_static(cfg)
    wout = read_wout(wout_path)
    state = state_from_wout(wout)

    inputs = booz_xform_inputs_from_state(state=state, static=static, indata=indata, signgs=wout.signgs)

    assert inputs.rmnc.shape[0] == cfg.ns - 1
    assert inputs.zmns.shape[0] == cfg.ns - 1
    assert inputs.lmns.shape[0] == cfg.ns - 1
    assert inputs.bmnc.shape[0] == cfg.ns - 1
    assert inputs.bsubumnc.shape[0] == cfg.ns - 1
    assert inputs.bsubvmnc.shape[0] == cfg.ns - 1
    assert inputs.iota.shape[0] == cfg.ns - 1
