"""Small analytic and finite-difference checks of the additive NEOPAX APIs."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmex
from vmex.core import implicit, omnigenity, stability


def test_raw_block_views_preserve_tangent_transpose_and_certificate(monkeypatch):
    matrix = jnp.array([[2.0, -0.4], [0.3, 1.5]])
    tangent = {"boundary": jnp.array([0.2, -0.1])}
    params, cfg, state, mask = (object() for _ in range(4))
    certificates = []

    def forward(p, c, x, m, batch, **options):
        assert (p, c, x, m) == (params, cfg, state, mask)
        assert options == {"probe_chunk_size": 2, "response_chunk_size": 1}
        assert batch["boundary"].shape == (1, 2)
        return {"state": batch["boundary"] @ matrix.T}, "certificate"

    def reverse(p, c, x, m, bars, **options):
        assert (p, c, x, m) == (params, cfg, state, mask)
        assert options == {"solver": "block", "probe_chunk_size": 2, "response_chunk_size": 1}
        return {"boundary": bars["state"] @ matrix}

    monkeypatch.setattr(implicit, "implicit_state_tangent_multi_rhs", forward)
    monkeypatch.setattr(implicit, "implicit_state_pullback_multi_rhs", reverse)
    monkeypatch.setattr(implicit, "_raise_unconverged_rows", lambda *args: certificates.append(args))
    response = implicit.implicit_state_tangent_raw_block(
        params, cfg, state, mask, tangent, probe_chunk_size=2)
    bars = {"state": jnp.array([[0.8, -0.6], [0.2, 0.9]])}
    gradient = implicit.implicit_state_pullback_multi_rhs_raw_block_transpose(
        params, cfg, state, mask, bars, probe_chunk_size=2)
    np.testing.assert_allclose(response["state"], matrix @ tangent["boundary"])
    np.testing.assert_allclose(
        bars["state"] @ response["state"], gradient["boundary"] @ tangent["boundary"])
    assert certificates == [(cfg, "certificate", "block")]


def test_constructed_j_surrogate_values_directions_and_edge_cases():
    helper = vmex.j_invariant_qi_maxj_residual_from_boozer
    assert helper is omnigenity.j_invariant_qi_maxj_residual_from_boozer
    bmnc = jnp.array([[1.4, 0.12, 0.03], [1.5, 0.14, 0.04]])
    options = dict(xm_b=jnp.array([0, 1, 1]), xn_b=jnp.array([0, 0, 2]),
                   iota_b=jnp.array([0.4, 0.45]), gi_b=jnp.array([1.0, 1.1]),
                   s_b=jnp.array([0.3, 0.6]), nfp=2, nphi=16, nalpha=3,
                   n_bounce=4, nphi_int=12)
    baseline = helper(bmnc_b=bmnc, **options)
    np.testing.assert_allclose(
        baseline["total"], baseline["qi_objective"] + baseline["maxj_objective"])
    direction = jnp.array([[0.0, 0.02, -0.01], [0.0, -0.01, 0.02]])
    def objective(values):
        return helper(bmnc_b=values, **options)["total"]
    ad = jnp.vdot(jax.grad(objective)(bmnc), direction)
    step = 1.0e-5
    fd = (objective(bmnc + step * direction) - objective(bmnc - step * direction)) / (2 * step)
    np.testing.assert_allclose(ad, fd, rtol=2e-4, atol=1e-7)
    for pairing in ("same_alpha", "all_to_all", "soft_local"):
        result = helper(bmnc_b=bmnc, **options, maxj_pairing=pairing,
                        maxj_sigma_alpha=0.3, weights=(1.0, 1.0))
        assert np.isfinite(result["total"])
        np.testing.assert_allclose(result["maxj_pair_weights"].sum(axis=1), 1.0)
    only_qi = helper(bmnc_b=bmnc, **options, include_maxj=False)
    only_maxj = helper(bmnc_b=bmnc, **options, include_qi=False)
    np.testing.assert_allclose(only_qi["total"], baseline["qi_objective"])
    np.testing.assert_allclose(only_maxj["total"], baseline["maxj_objective"])
    one_surface = dict(options, iota_b=options["iota_b"][:1], gi_b=options["gi_b"][:1],
                       s_b=options["s_b"][:1])
    assert helper(bmnc_b=bmnc[:1], **one_surface, include_qi=False)["total"] == 0.0
    for invalid in ({"nphi": 7}, {"nalpha": 1}, {"n_bounce": 1},
                    {"maxj_pairing": "invalid"}, {"weights": (1.0,)},
                    {"include_qi": False, "include_maxj": False}):
        with pytest.raises(ValueError):
            helper(bmnc_b=bmnc, **dict(options, **invalid))
    with pytest.raises(ValueError, match="non-empty"):
        helper(bmnc_b=bmnc[:0], **options)


def test_mercier_aggregate_preserves_weighted_residual_and_gradient(monkeypatch):
    options = SimpleNamespace(margin=0.02, smoothing=0.1)

    def residual(state, runtime, *, margin, smoothing):
        assert runtime is options
        return smoothing * jax.nn.softplus((margin - state) / smoothing)

    monkeypatch.setattr(stability, "mercier_stability_residual", residual)
    values = jnp.array([-0.2, 0.05, 0.3])
    def fn(state):
        return stability.mercier_stability_softmax(
            state, options, margin=options.margin, smoothing=options.smoothing, temperature=0.2)
    vector = np.asarray(residual(values, options, margin=0.02, smoothing=0.1))
    weights = np.exp((vector - vector.max()) / 0.2)
    np.testing.assert_allclose(fn(values), np.dot(vector, weights / weights.sum()))
    direction = jnp.array([0.2, -0.1, 0.4])
    fd = (fn(values + 1e-5 * direction) - fn(values - 1e-5 * direction)) / 2e-5
    np.testing.assert_allclose(jnp.vdot(jax.grad(fn)(values), direction), fd, rtol=1e-8)
    with pytest.raises(ValueError, match="positive"):
        stability.mercier_stability_softmax(values, options, temperature=0.0)
