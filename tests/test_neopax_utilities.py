"""Small analytic and finite-difference checks of the additive NEOPAX APIs."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmex
from vmex.core import implicit, omnigenity, stability


@pytest.mark.parametrize("chunk", [1, 2])
def test_raw_block_views_preserve_legacy_inverse_and_assembly_transpose(monkeypatch, chunk):
    """Four scalar blocks exercise the real direct solve, never a VMEX solve."""
    fields = implicit._STATE_FIELDS
    def make(**values):
        return implicit.SpectralState(**{
            name: values.get(name, jnp.zeros((4, 1))) for name in fields
        })
    matrix = jnp.diag(jnp.array([2., 3., 2.5, 4.]))
    matrix += jnp.diag(jnp.array([.2, -.1, .3]), -1)
    matrix += jnp.diag(jnp.array([-.4, .3, .2]), 1)
    forcing = jnp.array([[.2, -.3], [.3, -.1], [.4, .2], [.5, .4]])
    direct_r = jnp.tile(jnp.array([.2, -.3]), (4, 1))
    direct_z = jnp.tile(jnp.array([-.1, .4]), (4, 1))
    params, tangent = jnp.array([.1, .2]), jnp.array([.2, -.1])
    state, mask = make(), make(R_cos=jnp.ones((4, 1)))
    cfg = SimpleNamespace(resolution=SimpleNamespace(ns=4))
    def project(tree):
        return jax.tree.map(jnp.multiply, tree, mask)

    def residual(_cfg, frozen, dofs, *, formulation):
        assert formulation == "raw" and dofs is mask
        return lambda z, p: make(R_cos=(matrix @ z.R_cos[:, 0] + forcing @ p)[:, None])

    def assemble(z, p, frozen, proj, edge):
        return make(R_cos=z.R_cos + (direct_r @ p)[:, None],
                    Z_sin=2*z.R_cos + (direct_z @ p)[:, None])

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy raw-block helpers must not use refined/certified response lanes")

    monkeypatch.setattr(implicit, "_dof_projector", lambda *args: project)
    monkeypatch.setattr(implicit, "_edge_mask", lambda cfg: make())
    monkeypatch.setattr(implicit, "residual_fn", residual)
    monkeypatch.setattr(implicit, "runtime_from_params", lambda p, cfg: p)
    monkeypatch.setattr(implicit, "_assemble", assemble)
    for name in ("implicit_state_tangent_multi_rhs", "implicit_state_pullback_multi_rhs",
                 "_raw_block_system", "_block_factor", "_raise_unconverged_rows"):
        monkeypatch.setattr(implicit, name, forbidden)

    response_r = -np.linalg.solve(np.asarray(matrix), np.asarray(forcing))
    expected_r, expected_z = response_r + direct_r, 2*response_r + direct_z
    response = implicit.implicit_state_tangent_raw_block(
        params, cfg, state, mask, tangent, probe_chunk_size=chunk)
    np.testing.assert_allclose(response.R_cos[:, 0], expected_r @ tangent, rtol=1e-12)
    np.testing.assert_allclose(response.Z_sin[:, 0], expected_z @ tangent, rtol=1e-12)
    compiled_response = jax.jit(lambda direction: implicit.implicit_state_tangent_raw_block(
        params, cfg, state, mask, direction, probe_chunk_size=chunk))(tangent)
    np.testing.assert_allclose(compiled_response.R_cos, response.R_cos, rtol=1e-12)
    bars = jax.tree.map(lambda value: jnp.stack([value+.3, value-.2]), make())
    gradient = implicit.implicit_state_pullback_multi_rhs_raw_block_transpose(
        params, cfg, state, mask, bars, probe_chunk_size=chunk)
    expected = bars.R_cos[..., 0] @ expected_r + bars.Z_sin[..., 0] @ expected_z
    np.testing.assert_allclose(gradient, expected, rtol=1e-12, atol=1e-13)
    compiled_gradient = jax.jit(lambda cotangents: implicit.implicit_state_pullback_multi_rhs_raw_block_transpose(
        params, cfg, state, mask, cotangents, probe_chunk_size=chunk))(bars)
    np.testing.assert_allclose(compiled_gradient, expected, rtol=1e-12, atol=1e-13)
    products = sum(jnp.sum(b*t[None], axis=(1, 2)) for b, t in zip(
        jax.tree.leaves(bars), jax.tree.leaves(response), strict=True))
    np.testing.assert_allclose(products, gradient @ tangent, rtol=1e-12, atol=1e-13)


def test_raw_block_empty_mask_has_zero_inverse(monkeypatch):
    state = implicit.SpectralState(**{name: jnp.zeros((2, 1)) for name in implicit._STATE_FIELDS})
    cfg = SimpleNamespace(resolution=SimpleNamespace(ns=2))
    monkeypatch.setattr(implicit, "_dof_projector", lambda *args: lambda tree: tree)
    monkeypatch.setattr(implicit, "residual_fn", lambda *args, **kwargs: None)
    for transpose in (False, True):
        result = implicit._neopax_raw_block_solve(cfg, cfg, state, state, state, transpose=transpose)
        assert all(np.count_nonzero(value) == 0 for value in jax.tree.leaves(result))


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
