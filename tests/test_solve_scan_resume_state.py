import numpy as np
import pytest

from vmec_jax._compat import has_jax, jnp
from vmec_jax.solve import _initialize_scan_resume_state
from vmec_jax.solvers.fixed_boundary.scan.resume import build_traced_scan_resume_state


pytestmark = pytest.mark.skipif(not has_jax(), reason="scan resume-state helper requires JAX arrays")


_VELOCITY_NAMES = (
    "vRcc",
    "vRss",
    "vZsc",
    "vZcs",
    "vLsc",
    "vLcs",
    "vRsc",
    "vRcs",
    "vZcc",
    "vZss",
    "vLcc",
    "vLss",
)


def _init(resume_state, *, checkpoint="checkpoint", time_step=0.2, flip_sign=1.0, shape=(2, 2, 1)):
    return _initialize_scan_resume_state(
        resume_state,
        dtype=jnp.float32,
        velocity_shape=shape,
        k_ndamp=10,
        time_step_default=jnp.asarray(time_step, dtype=jnp.float32),
        flip_sign_default=jnp.asarray(flip_sign, dtype=jnp.float32),
        state_checkpoint_default=checkpoint,
    )


def _scalar(value):
    return float(np.asarray(value))


def _int_scalar(value):
    return int(np.asarray(value))


def test_build_traced_scan_resume_state_keeps_arrays_and_advances_iter_offset():
    shape = (1, 1, 1)
    carry = type(
        "Carry",
        (),
        {
            "time_step": jnp.asarray(0.25),
            "inv_tau": jnp.asarray([0.1, 0.2]),
            "fsq_prev": jnp.asarray(1.0),
            "fsq0_prev": jnp.asarray(2.0),
            "flip_sign": jnp.asarray(-1.0),
            "iter1": jnp.asarray(3, dtype=jnp.int32),
            "iter_offset": jnp.asarray(4, dtype=jnp.int32),
            "res0": jnp.asarray(5.0),
            "res1": jnp.asarray(6.0),
            "ijacob": jnp.asarray(7, dtype=jnp.int32),
            "bad_resets": jnp.asarray(8, dtype=jnp.int32),
            "bad_growth": jnp.asarray(9, dtype=jnp.int32),
            "fsqz_prev": jnp.asarray(10.0),
            "state_checkpoint": {"state": "checkpoint"},
            "cache_valid": jnp.asarray(True),
            "force_bcovar_update": jnp.asarray(False),
            **{name: jnp.full(shape, float(i + 1)) for i, name in enumerate(_VELOCITY_NAMES)},
        },
    )()

    payload = build_traced_scan_resume_state(carry, max_iter=6)

    assert _int_scalar(payload["iter_offset"]) == 10
    assert payload["state_checkpoint"] == {"state": "checkpoint"}
    assert bool(np.asarray(payload["vmec2000_cache_valid"]))
    assert not bool(np.asarray(payload["force_bcovar_update"]))
    np.testing.assert_allclose(np.asarray(payload["vLss"]), np.full(shape, 12.0))


def test_initialize_scan_resume_state_accepts_valid_resume_state():
    shape = (2, 2, 1)
    velocities = {name: np.full(shape, float(idx + 1)) for idx, name in enumerate(_VELOCITY_NAMES)}
    checkpoint = object()
    out = _init(
        {
            "time_step": "0.25",
            "flip_sign": "-1.0",
            "inv_tau": np.arange(10, dtype=float) + 0.5,
            "fsq_prev": "2.0",
            "fsq0_prev": "3.0",
            "res0": "4.0",
            "res1": "5.0",
            "iter1": "6",
            "ijacob": "7",
            "bad_resets": "8",
            "bad_growth_streak": "9",
            "fsqz_prev": "10.0",
            "state_checkpoint": checkpoint,
            **velocities,
        },
        shape=shape,
    )

    assert _scalar(out.time_step) == pytest.approx(0.25)
    assert _scalar(out.flip_sign) == pytest.approx(-1.0)
    np.testing.assert_allclose(np.asarray(out.inv_tau), np.arange(10, dtype=np.float32) + 0.5)
    assert _scalar(out.fsq_prev) == pytest.approx(2.0)
    assert _scalar(out.fsq0_prev) == pytest.approx(3.0)
    assert _scalar(out.res0) == pytest.approx(4.0)
    assert _scalar(out.res1) == pytest.approx(5.0)
    assert _int_scalar(out.iter1) == 6
    assert _int_scalar(out.ijacob) == 7
    assert _int_scalar(out.bad_resets) == 8
    assert _int_scalar(out.bad_growth) == 9
    assert _scalar(out.fsqz_prev) == pytest.approx(10.0)
    assert out.state_checkpoint is checkpoint
    for idx, name in enumerate(_VELOCITY_NAMES):
        np.testing.assert_allclose(np.asarray(getattr(out, name)), np.full(shape, float(idx + 1)))


def test_initialize_scan_resume_state_keeps_defaults_on_bad_conversions():
    class BadFloat:
        def __float__(self):
            raise TypeError("bad float")

    class BadInt:
        def __int__(self):
            raise TypeError("bad int")

    class BadBool:
        def __bool__(self):
            raise TypeError("bad bool")

    out = _init(
        {
            "time_step": BadFloat(),
            "flip_sign": BadFloat(),
            "fsq_prev": BadFloat(),
            "fsq0_prev": BadFloat(),
            "res0": BadFloat(),
            "res1": 5.0,
            "iter1": BadInt(),
            "ijacob": BadInt(),
            "bad_resets": BadInt(),
            "bad_growth_streak": BadInt(),
            "fsqz_prev": BadFloat(),
            "force_bcovar_update": BadBool(),
        },
        time_step=0.2,
        flip_sign=2.0,
    )

    assert _scalar(out.time_step) == pytest.approx(0.2)
    assert _scalar(out.flip_sign) == pytest.approx(2.0)
    np.testing.assert_allclose(np.asarray(out.inv_tau), np.full((10,), 0.15 / 0.2, dtype=np.float32))
    assert _scalar(out.fsq_prev) == pytest.approx(1.0)
    assert _scalar(out.fsq0_prev) == pytest.approx(1.0)
    assert _scalar(out.res0) == pytest.approx(-1.0)
    assert _scalar(out.res1) == pytest.approx(-1.0)
    assert _int_scalar(out.iter1) == 1
    assert _int_scalar(out.ijacob) == 0
    assert _int_scalar(out.bad_resets) == 0
    assert _int_scalar(out.bad_growth) == 0
    assert _scalar(out.fsqz_prev) == pytest.approx(1.0)
    assert bool(np.asarray(out.force_bcovar_update)) is False


def test_initialize_scan_resume_state_handles_missing_velocity_blocks():
    shape = (2, 2, 1)
    vRcc = np.full(shape, 3.0)
    out = _init({"vRcc": vRcc}, shape=shape)

    np.testing.assert_allclose(np.asarray(out.vRcc), vRcc)
    for name in _VELOCITY_NAMES[1:]:
        np.testing.assert_allclose(np.asarray(getattr(out, name)), np.zeros(shape, dtype=np.float32))

    ignored = _init({"vRss": np.full(shape, 4.0)}, shape=shape)
    for name in _VELOCITY_NAMES:
        np.testing.assert_allclose(np.asarray(getattr(ignored, name)), np.zeros(shape, dtype=np.float32))


def test_initialize_scan_resume_state_optional_payloads_default_and_override():
    default = _init({})

    assert bool(np.asarray(default.force_bcovar_update)) is False
    assert _scalar(default.r00_prev) == pytest.approx(0.0)
    assert _scalar(default.z00_prev) == pytest.approx(0.0)
    assert _scalar(default.w_mhd_prev) == pytest.approx(0.0)

    out = _init(
        {
            "force_bcovar_update": True,
            "r00_prev": "1.25",
            "z00_prev": "-2.5",
            "w_mhd_prev": "3.75",
        }
    )

    assert bool(np.asarray(out.force_bcovar_update)) is True
    assert _scalar(out.r00_prev) == pytest.approx(1.25)
    assert _scalar(out.z00_prev) == pytest.approx(-2.5)
    assert _scalar(out.w_mhd_prev) == pytest.approx(3.75)
