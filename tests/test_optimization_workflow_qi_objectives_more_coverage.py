from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.optimizers.fixed_boundary import qi_objectives


def test_smooth_positive_part_hard_and_soft_branches() -> None:
    import vmec_jax.optimization_workflow as workflow

    hard = workflow._smooth_positive_part(np.asarray([-2.0, 0.0, 3.0]), softness=0.0)
    np.testing.assert_allclose(np.asarray(hard), [0.0, 0.0, 3.0])

    soft = workflow._smooth_positive_part(np.asarray([-1.0, 0.0, 1.0]), softness=0.5)
    assert float(soft[0]) > 0.0
    assert float(soft[1]) == pytest.approx(0.5 * np.log(2.0))
    assert float(soft[2]) > 1.0


def test_qi_residual_ceiling_objective_uses_hard_and_smooth_excess() -> None:
    import vmec_jax.optimization_workflow as workflow

    hard_term = workflow.qi_residual_ceiling_objective(maximum=1.5, weight=2.0, smooth_penalty=0.0)
    residuals, total = hard_term.residual_and_total(None, None, {"total": np.asarray([1.0, 2.0])})
    np.testing.assert_allclose(np.asarray(residuals), [0.0, 1.0])
    assert float(total) == pytest.approx(1.0)

    smooth_term = workflow.qi_residual_ceiling_objective(maximum=1.5, weight=3.0, smooth_penalty=0.25)
    smooth_residuals, smooth_total = smooth_term.residual_and_total(None, None, {"total": np.asarray([1.5])})
    assert float(smooth_residuals[0]) == pytest.approx(3.0 * 0.25 * np.log(2.0))
    assert float(smooth_total) == pytest.approx(float(smooth_residuals[0]) ** 2)


def test_qi_boozer_b_target_objective_normalizes_and_excludes_b00() -> None:
    import vmec_jax.optimization_workflow as workflow

    term = workflow.qi_boozer_b_target_objective(
        target_bmnc=np.asarray([[1.0, 1.0, 3.0]]),
        target_bmns=np.asarray([[0.5, 1.0, 1.5]]),
        weight=2.0,
        normalize=True,
        include_b00=False,
    )
    field = {
        "booz": {
            "bmnc_b": np.asarray([[2.0, 4.0, 6.0]]),
            "bmns_b": np.asarray([[1.0, 2.0, 3.0]]),
        }
    }

    residuals, total = term.residual_and_total(None, None, field)

    expected = np.asarray([0.0, 2.0 / np.sqrt(6.0), 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(residuals), expected)
    assert float(total) == pytest.approx(float(np.sum(expected * expected)))


def test_qi_boozer_b_target_objective_non_normalized_includes_b00_and_missing_sines() -> None:
    import vmec_jax.optimization_workflow as workflow

    term = workflow.qi_boozer_b_target_objective(
        target_bmnc=np.asarray([[1.0, 0.5]]),
        weight=4.0,
        normalize=False,
        include_b00=True,
    )
    field = {"booz": {"bmnc_b": np.asarray([[2.0, 1.5]])}}

    residuals, total = term.residual_and_total(None, None, field)

    expected = np.asarray([2.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(residuals), expected)
    assert float(total) == pytest.approx(8.0)


def test_qi_mirror_ratio_constraint_normalizes_surfaces(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    ctx = SimpleNamespace(static=SimpleNamespace(cfg=SimpleNamespace(nfp=5)))
    seen = {}

    def fake_mirror_ratio_penalty_from_boozer_output(booz, **kwargs):
        seen["booz"] = booz
        seen["kwargs"] = kwargs
        return {
            "mirror_ratio": np.asarray([0.2, 0.5]),
            "residuals1d": np.asarray([0.0, 0.2]),
            "total": 0.04,
        }

    monkeypatch.setattr(qi_objectives, "mirror_ratio_penalty_from_boozer_output", fake_mirror_ratio_penalty_from_boozer_output)

    term = workflow.qi_mirror_ratio_constraint(
        threshold=0.3,
        ntheta=7,
        nphi=9,
        phimin=0.125,
        smooth_extrema=0.01,
        normalize_surfaces=True,
    )
    field = {"booz": {"bmnc_b": np.ones((2, 3)), "bmns_b": np.zeros((2, 3))}}

    residuals, total = term.residual_and_total(ctx, None, field)

    np.testing.assert_allclose(np.asarray(residuals), [-0.1 / np.sqrt(2.0), 0.2 / np.sqrt(2.0)])
    assert float(total) == pytest.approx(0.02)
    assert seen["kwargs"] == {
        "nfp": 5,
        "threshold": 0.3,
        "weights": [0.5, 0.5],
        "ntheta": 7,
        "nphi": 9,
        "phimin": 0.125,
        "smooth_extrema": 0.01,
        "smooth_penalty": 0.0,
    }
    np.testing.assert_allclose(seen["booz"]["bmnc_b"], np.ones((2, 3)))


def test_qi_mirror_ratio_constraint_slices_one_surface_without_normalization(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    ctx = SimpleNamespace(static=SimpleNamespace(cfg=SimpleNamespace(nfp=3)))
    seen = {}

    def fake_mirror_ratio_penalty_from_boozer_output(booz, **kwargs):
        seen["booz"] = booz
        seen["kwargs"] = kwargs
        return {"mirror_ratio": np.asarray([0.45]), "residuals1d": np.asarray([0.1]), "total": 0.01}

    monkeypatch.setattr(qi_objectives, "mirror_ratio_penalty_from_boozer_output", fake_mirror_ratio_penalty_from_boozer_output)

    term = workflow.qi_mirror_ratio_constraint(threshold=0.4, surface_index=1, normalize_surfaces=True)
    field = {
        "booz": {
            "bmnc_b": np.asarray([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]]),
            "bmns_b": np.asarray([[0.0, 0.4], [0.0, 0.5], [0.0, 0.6]]),
            "s_b": np.asarray([0.25, 0.5, 0.75]),
        }
    }

    residuals, total = term.residual_and_total(ctx, None, field)

    np.testing.assert_allclose(np.asarray(residuals), [0.05])
    assert float(total) == pytest.approx(0.0025)
    assert seen["kwargs"]["weights"] is None
    np.testing.assert_allclose(np.asarray(seen["booz"]["bmnc_b"]), [[2.0, 0.2]])
    np.testing.assert_allclose(np.asarray(seen["booz"]["s_b"]), [0.5])


def test_qi_max_elongation_constraint_monkeypatches_geometry_helper(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    ctx = SimpleNamespace(static="static")
    calls = []

    def fake_max_elongation_penalty_from_state(**kwargs):
        calls.append(kwargs)
        return {"max_elongation": kwargs["state"], "residuals1d": np.asarray([99.0]), "total": 99.0}

    monkeypatch.setattr(qi_objectives, "max_elongation_penalty_from_state", fake_max_elongation_penalty_from_state)

    term = workflow.qi_max_elongation_constraint(threshold=4.0, ntheta=5, nphi=6, smooth_extrema=0.02)
    high_residuals, high_total = term.residual_and_total(ctx, 5.5, {})
    low_residuals, low_total = term.residual_and_total(ctx, 3.5, {})

    np.testing.assert_allclose(np.asarray(high_residuals), [1.5])
    assert float(high_total) == pytest.approx(2.25)
    np.testing.assert_allclose(np.asarray(low_residuals), [-0.5])
    assert float(low_total) == pytest.approx(0.0)
    assert calls == [
        {
            "state": 5.5,
            "static": "static",
            "threshold": 4.0,
            "ntheta": 5,
            "nphi": 6,
            "smooth_extrema": 0.02,
            "smooth_penalty": 0.0,
        },
        {
            "state": 3.5,
            "static": "static",
            "threshold": 4.0,
            "ntheta": 5,
            "nphi": 6,
            "smooth_extrema": 0.02,
            "smooth_penalty": 0.0,
        },
    ]
