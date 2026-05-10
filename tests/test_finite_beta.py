from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vmec_jax as vj
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.static import build_static


pytestmark = pytest.mark.full


def test_finite_beta_scalars_are_finite_on_bundled_qi_input():
    path = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.nfp4_QI_finite_beta"
    cfg, indata = load_config(str(path))
    static = build_static(cfg)
    boundary = boundary_from_indata(indata, static.modes)
    state = initial_guess_from_boundary(static, boundary, indata)
    geom = eval_geom(state, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))

    scalars = vj.finite_beta_scalars_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
    )

    for key in ("aspect", "mean_iota", "volavgB", "betatotal", "wb", "wp", "volume"):
        assert np.isfinite(float(np.asarray(scalars[key])))
    assert float(np.asarray(scalars["betatotal"])) > 0.0


def test_mercier_terms_are_finite_on_bundled_qi_input():
    path = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.nfp4_QI_finite_beta"
    cfg, indata = load_config(str(path))
    static = build_static(cfg)
    boundary = boundary_from_indata(indata, static.modes)
    state = initial_guess_from_boundary(static, boundary, indata)
    geom = eval_geom(state, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))

    terms = vj.mercier_terms_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
    )

    for key in ("DMerc", "Dshear", "Dcurr", "Dwell", "Dgeod", "torcur", "vp"):
        arr = np.asarray(terms[key])
        assert arr.shape == np.asarray(static.s).shape
        assert np.all(np.isfinite(arr))
