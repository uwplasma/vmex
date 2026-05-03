from __future__ import annotations

from dataclasses import replace as dc_replace

import pytest

from vmec_jax.booz_input import booz_xform_inputs_from_state
from vmec_jax._compat import enable_x64
from vmec_jax.driver import example_paths
from vmec_jax.profiles import eval_profiles
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


def test_booz_xform_inputs_merge_partial_profile_overrides():
    pytest.importorskip("netCDF4")

    import numpy as np

    input_path, wout_path = example_paths("circular_tokamak")
    if wout_path is None:
        pytest.skip("No reference wout file available for circular_tokamak")

    cfg, indata = load_config(str(input_path))
    static = build_static(cfg)
    wout = read_wout(wout_path)
    state = state_from_wout(wout)

    s_full = np.asarray(static.s)
    s_half = np.concatenate([s_full[:1], 0.5 * (s_full[1:] + s_full[:-1])], axis=0)
    pressure = eval_profiles(indata, s_half)["pressure"]
    default = booz_xform_inputs_from_state(state=state, static=static, indata=indata, signgs=wout.signgs)
    overridden = booz_xform_inputs_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=wout.signgs,
        profiles_half={"pressure": pressure},
    )

    np.testing.assert_allclose(np.asarray(overridden.iota), np.asarray(default.iota), rtol=0.0, atol=0.0)
    assert np.linalg.norm(np.asarray(overridden.iota)) > 0.0


def test_booz_xform_inputs_from_state_jit_tracer_safe():
    pytest.importorskip("netCDF4")
    pytest.importorskip("jax")

    from vmec_jax._compat import jax, jnp

    enable_x64(True)

    input_path, wout_path = example_paths("circular_tokamak")
    if wout_path is None:
        pytest.skip("No reference wout file available for circular_tokamak")

    cfg, indata = load_config(str(input_path))
    static = build_static(cfg)
    wout = read_wout(wout_path)
    state = state_from_wout(wout)

    @jax.jit
    def _bmnc_from_rcos(rcos):
        traced_state = dc_replace(state, Rcos=rcos)
        return booz_xform_inputs_from_state(
            state=traced_state,
            static=static,
            indata=indata,
            signgs=wout.signgs,
        ).bmnc

    bmnc = _bmnc_from_rcos(jnp.asarray(state.Rcos))
    assert bmnc.shape[0] == cfg.ns - 1
