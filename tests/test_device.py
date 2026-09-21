"""Unit tests for :mod:`vmex.core.device` — the CPU/GPU placement policy.

Most tests exercise host-side policy.  One tiny solve verifies that final
arrays stay on a selected nondefault CPU or GPU.  Accelerator-only branches
skip when the required hardware is unavailable.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from vmex.core import device as dev
from vmex.core import freeboundary, multigrid, solver, wout as wout_module
from vmex.core.fourier import Resolution
from vmex.core.input import VmecInput

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _res(ns: int, mpol: int, ntor: int, nfp: int = 1) -> Resolution:
    return Resolution(mpol=mpol, ntor=ntor, ntheta=2 * mpol + 6,
                      nzeta=(2 * ntor + 4) if ntor else 1, nfp=nfp,
                      lasym=False, ns=ns)


def test_iteration_work_and_recommendation_threshold():
    small = _res(ns=11, mpol=6, ntor=0)          # tiny tokamak-like
    big = _res(ns=101, mpol=12, ntor=12, nfp=4)  # reactor-scale 3D
    high_mode = _res(ns=101, mpol=18, ntor=24, nfp=4)
    assert dev.iteration_work(small) < dev.GPU_MIN_ITERATION_WORK
    assert dev.iteration_work(big) >= dev.GPU_MIN_ITERATION_WORK
    assert high_mode.mnmax > dev.GPU_MAX_SPECTRAL_MODES
    assert dev.recommended_device(small) == "cpu"
    assert dev.recommended_device(big) == "gpu"
    assert dev.recommended_device(high_mode) == "cpu"


@pytest.mark.parametrize("key", ["jax_platforms", "jax_platform_name"])
def test_selected_jax_platform_is_never_overridden(key):
    previous = jax.config.values.get(key)
    try:
        jax.config.update(key, "cpu")
        # Even a GPU-recommended resolution is left alone when JAX is pinned.
        assert dev.resolve_device(
            dev.AUTO, _res(ns=201, mpol=16, ntor=16, nfp=5)
        ) is None
    finally:
        jax.config.update(key, previous)


def test_none_leaves_placement_to_jax(monkeypatch):
    monkeypatch.setattr(dev, "recommended_device", lambda _: pytest.fail("auto policy ran"))
    assert dev.resolve_device(None, _res(ns=11, mpol=6, ntor=0)) is None
    assert dev.resolve_implicit_device(None, None) is None
    assert dev.resolve_mirror_device(None) is None


def test_omitted_device_uses_auto_policy(monkeypatch):
    seen = []
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setattr(dev, "recommended_device", lambda res: seen.append(res) or "cpu")
    dev.resolve_device(resolution=_res(ns=11, mpol=6, ntor=0))
    assert len(seen) == 1


def test_auto_without_resolution_has_a_clear_error():
    with pytest.raises(ValueError, match="resolution is required"):
        dev.resolve_device()
    with pytest.raises(ValueError, match="resolution is required"):
        dev.device_context()


def test_default_device_context_is_never_overridden(monkeypatch):
    # Pretend the process default backend is an accelerator: absent the active
    # CPU context, AUTO would explicitly return a CPU device for this deck.
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with jax.default_device(jax.devices("cpu")[0]):
        assert dev.resolve_device(dev.AUTO, _res(ns=11, mpol=6, ntor=0)) is None
        assert dev.resolve_implicit_device(dev.AUTO, None) is None
        assert dev.resolve_mirror_device(dev.AUTO) is None


def test_auto_cpu_recommendation_is_a_noop_on_cpu(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    # CPU-recommended + default backend already CPU -> nothing to place.
    if jax.default_backend() == "cpu":
        assert dev.resolve_device(dev.AUTO, _res(ns=11, mpol=6, ntor=0)) is None


def test_synthesis_policy_uses_measured_hardware_defaults(monkeypatch):
    resolution = _res(ns=101, mpol=18, ntor=24, nfp=4)
    target = SimpleNamespace(platform="cpu")
    monkeypatch.setattr(solver, "_placement_device", lambda *_: target)
    monkeypatch.setattr(solver.platform, "machine", lambda: "x86_64")
    assert not solver._resolve_use_fft(None, "cpu", resolution)
    monkeypatch.setattr(solver.platform, "machine", lambda: "arm64")
    assert solver._resolve_use_fft(None, "cpu", resolution)
    target.platform = "gpu"
    assert solver._resolve_use_fft(None, "gpu", resolution)
    assert not solver._resolve_use_fft(
        None, "gpu", _res(ns=50, mpol=8, ntor=8, nfp=2)
    )
    assert solver._resolve_use_fft(False, "gpu", resolution) is False
    assert solver._resolve_use_fft(True, "cpu", resolution) is True


def test_synthesis_policy_respects_active_default_device(monkeypatch):
    monkeypatch.setattr(solver.platform, "machine", lambda: "x86_64")
    resolution = _res(ns=101, mpol=18, ntor=24, nfp=4)
    with jax.default_device(jax.devices("cpu")[0]):
        assert not solver._resolve_use_fft(None, None, resolution)


def test_auto_gpu_recommendation_without_accelerator(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    # GPU-recommended but CPU-only machine -> None (nothing to do).
    if jax.default_backend() == "cpu":
        assert dev.resolve_device(dev.AUTO, _res(ns=201, mpol=16, ntor=16, nfp=5)) is None


def test_explicit_cpu_returns_a_cpu_device():
    resolved = dev.resolve_device("cpu", _res(ns=11, mpol=6, ntor=0))
    assert resolved is not None and resolved.platform == "cpu"


def test_explicit_jax_device_is_returned_unchanged():
    cpu0 = jax.devices("cpu")[0]
    assert dev.resolve_device(cpu0, _res(ns=11, mpol=6, ntor=0)) is cpu0


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="unknown device"):
        dev.resolve_device("quantum", _res(ns=11, mpol=6, ntor=0))


def test_gpu_request_on_cpu_machine_raises():
    # explicit accelerator request is honored -> missing hardware raises.
    if jax.default_backend() == "cpu":
        with pytest.raises(RuntimeError):
            dev.resolve_device("gpu", _res(ns=11, mpol=6, ntor=0))


def test_multigrid_moves_residual_continuation_with_the_state():
    """A rung placed on a new device must not leave the previous rung's
    residual scalars behind.

    ``solve_multigrid`` moves the state to each rung's placement device but
    carried ``(fsqr, fsqz, fsql)`` from the rung before straight through, so
    an AUTO ladder that crossed the CPU/GPU work threshold built a carry with
    the state on one device and its residual scalars on another; the jitted
    while lane rejects that.  Forcing consecutive rungs onto two devices
    reproduces it without needing a GPU.
    """
    devices = []
    for platform in ("gpu", "cpu"):
        try:
            devices = jax.devices(platform)
        except RuntimeError:
            pass
        if len(devices) >= 2:
            break
    if len(devices) < 2:
        pytest.skip("two devices unavailable")
    # Two short rungs: the failure is in how the second rung's carry is
    # assembled, so it needs a rung boundary, not convergence.
    inp = replace(
        VmecInput.from_file(DATA / "input.solovev"),
        ns_array=[3, 5], niter_array=[2, 2], ftol_array=[1.0, 1.0],
    )
    placements = []

    def alternating(_device, _resolution):
        target = devices[min(len(placements), 1)]
        placements.append(target)
        return target

    original = multigrid._placement_device
    multigrid._placement_device = alternating
    try:
        result = multigrid.solve_multigrid(
            inp, mode="jit", device="auto", verbose=False,
            prefetch_compile=False,
        )
    finally:
        multigrid._placement_device = original

    assert placements[:2] == [devices[0], devices[1]]
    assert np.all(np.isfinite(np.asarray(result.rmnc)))
    assert np.asarray(result.rmnc).shape[0] == 5  # the fine rung really ran


def test_fixed_boundary_honors_second_device_without_outer_context():
    devices = []
    for platform in ("gpu", "cpu"):
        try:
            devices = jax.devices(platform)
        except RuntimeError:
            pass
        if len(devices) >= 2:
            break
    if len(devices) < 2:
        pytest.skip("two devices unavailable")
    inp = replace(
        VmecInput.from_file(DATA / "input.solovev"),
        ns_array=[3, 5], niter_array=[2, 2], ftol_array=[1.0, 1.0],
    )
    reference = multigrid.solve_multigrid(
        inp, device=devices[0], verbose=False, prefetch_compile=False,
    )
    result = multigrid.solve_multigrid(
        inp,
        device=devices[1],
        verbose=False,
        prefetch_compile=False,
    )
    assert result.converged
    assert {
        leaf.device
        for leaf in jax.tree.leaves(result.state)
        if hasattr(leaf, "device")
    } == {devices[1]}
    for name in ("rmnc", "zmns", "iotaf", "fsq_history"):
        np.testing.assert_allclose(
            getattr(result, name), getattr(reference, name),
            rtol=5e-9, atol=1e-12,
        )

    single = solver.solve(
        inp,
        resolution=solver.resolution_from_input(inp, ns=5),
        initial_state=reference.state,
        device=devices[1],
        verbose=False,
    )
    for name in ("rmnc", "zmns", "iotaf"):
        np.testing.assert_allclose(
            getattr(single, name), getattr(reference, name),
            rtol=5e-9, atol=1e-12,
        )
    bad_state = replace(reference.state, R_cos=reference.state.R_cos[:-1])
    with pytest.raises(ValueError, match="interpolate_state"):
        solver.solve(inp, initial_state=bad_state, device=devices[1], verbose=False)


def test_fixed_boundary_builds_runtime_in_device_context(monkeypatch):
    active = False
    seen = []
    seed = SimpleNamespace(R_cos=np.zeros((2, 1)))
    runtime = SimpleNamespace(modes=object())
    carry = SimpleNamespace(
        ier=multigrid.MORE_ITER_FLAG,
        state=SimpleNamespace(R_cos=np.zeros((3, 1))),
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
    )
    result = object()

    @contextlib.contextmanager
    def context(*_):
        nonlocal active
        active = True
        yield
        active = False

    def prepare(*_, **__):
        assert active
        seen.append("runtime")
        return runtime

    def interpolate(*_, **__):
        assert active
        seen.append("interpolate")
        return carry.state

    def hot_restart(*_):
        assert active
        seen.append("hot restart")
        return carry.state

    def baselines(*_, **__):
        assert active
        seen.append("baselines")
        return runtime

    def solve(*_, **__):
        assert active
        return carry

    def unconverged(*_):
        assert active
        return result

    monkeypatch.setattr(multigrid, "device_context", context)
    monkeypatch.setattr(multigrid, "prepare_runtime", prepare)
    monkeypatch.setattr(multigrid, "interpolate_state", interpolate)
    monkeypatch.setattr(multigrid, "hot_restart_state", hot_restart)
    monkeypatch.setattr(multigrid, "runtime_with_baselines", baselines)
    monkeypatch.setattr(multigrid, "_solve_stage", solve)
    monkeypatch.setattr(multigrid, "_result_from_carry", unconverged)
    actual = multigrid.solve_multigrid(
        VmecInput(ns_array=[3]),
        initial_state=seed,
        device="cpu",
        raise_on_max_iterations=False,
    )
    assert actual is result
    assert seen == ["runtime", "interpolate", "hot restart", "baselines"]


@pytest.mark.parametrize(
    ("device", "expected_active"), [("cpu", True), (dev.AUTO, False)]
)
def test_free_boundary_loads_mgrid_in_explicit_device_context(
    monkeypatch, device, expected_active
):
    active = False

    @contextlib.contextmanager
    def context(*_):
        nonlocal active
        active = True
        yield
        active = False

    def load(*_):
        assert active is expected_active
        raise RuntimeError("mgrid reached")

    monkeypatch.setattr(multigrid, "device_context", context)
    monkeypatch.setattr(freeboundary, "_external_field_from_input", load)
    with pytest.raises(RuntimeError, match="mgrid reached"):
        multigrid.solve_free_boundary_multigrid(
            VmecInput(lfreeb=True, mgrid_file="fake", ns_array=[3]), device=device
        )


def test_resolve_implicit_device_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    res = _res(ns=35, mpol=6, ntor=6, nfp=4)  # a QH-class stage
    resolved = dev.resolve_implicit_device(dev.AUTO, res)
    if jax.default_backend() == "cpu":
        # already on CPU -> nothing to place (the implicit path stays put).
        assert resolved is None
    else:
        # on an accelerator the launch-bound Jacobian is pinned to the CPU.
        assert resolved is not None and resolved.platform == "cpu"


def test_resolve_implicit_device_honors_explicit_and_pin():
    res = _res(ns=35, mpol=6, ntor=6, nfp=4)
    # explicit device is honored (delegated to resolve_device).
    explicit = dev.resolve_implicit_device("cpu", res)
    assert explicit is not None and explicit.platform == "cpu"
    # A JAX platform pin stands the auto policy down.
    previous = jax.config.jax_platforms
    try:
        jax.config.update("jax_platforms", "cpu")
        assert dev.resolve_implicit_device(dev.AUTO, res) is None
    finally:
        jax.config.update("jax_platforms", previous)


def test_resolve_mirror_device_defaults_to_cpu_and_honors_explicit():
    resolved = dev.resolve_mirror_device(dev.AUTO)
    if jax.default_backend() == "cpu":
        assert resolved is None
    else:
        assert resolved is not None and resolved.platform == "cpu"
    explicit = dev.resolve_mirror_device("cpu")
    assert explicit is not None and explicit.platform == "cpu"


def test_device_context_is_a_nullcontext_when_placement_untouched(monkeypatch):
    monkeypatch.setattr(dev, "_user_selected_placement", lambda: True)
    res = _res(ns=11, mpol=6, ntor=0)
    for choice in (None, dev.AUTO):
        ctx = dev.device_context(choice, res)
        assert isinstance(ctx, contextlib.nullcontext)
        with ctx:
            pass


def test_device_context_wraps_default_device_for_explicit_cpu():
    ctx = dev.device_context("cpu", _res(ns=11, mpol=6, ntor=0))
    # jax.default_device(...) returns a context manager that is not nullcontext.
    assert not isinstance(ctx, contextlib.nullcontext)
    with ctx:
        pass


def test_wout_builder_uses_state_device_context(monkeypatch):
    active = False
    target = SimpleNamespace(platform="gpu")
    state = SimpleNamespace(R_cos=SimpleNamespace(device=target))

    @contextlib.contextmanager
    def context(device):
        nonlocal active
        assert device is target
        active = True
        yield
        active = False

    def build(**_):
        assert active
        return "wout"

    monkeypatch.setattr(jax, "default_device", context)
    wrapped = wout_module._on_state_device(build)
    assert wrapped(state=state) == "wout"
    state.R_cos.device = None
    assert wout_module._on_state_device(lambda **_: "wout")(state=state) == "wout"


def test_put_numeric_leaves_preserves_metadata_and_none_contract():
    resolution = _res(ns=11, mpol=6, ntor=0)
    assert dev._placement_device(None, resolution) is None
    cpu = jax.devices("cpu")[0]
    resident = jax.device_put(np.ones(2), cpu)
    assert dev._put_numeric_leaves(resident, cpu) is resident
    moved = dev._put_numeric_leaves(
        {"array": np.ones(2), "metadata": "kept"}, cpu,
    )
    assert moved["array"].device.platform == "cpu"
    assert moved["metadata"] == "kept"


def test_put_numeric_leaves_moves_registered_partial_captures():
    target = jax.devices("cpu")[0]
    field = jax.tree_util.Partial(
        lambda scale, points: scale * points,
        jax.numpy.asarray(2.0),
    )
    moved = dev._put_numeric_leaves(field, target)
    assert jax.tree.leaves(moved)[0].device.platform == "cpu"
    np.testing.assert_allclose(moved(jax.numpy.asarray([3.0])), 6.0)


def test_put_numeric_leaves_host_stages_cross_accelerator_copy(monkeypatch):
    class Device:
        platform = "gpu"

    class Array:
        def devices(self):
            return {Device()}

        def __array__(self, dtype=None):
            return np.asarray([1.0, 2.0], dtype=dtype)

    target = Device()
    monkeypatch.setattr(dev.jax, "Array", Array)
    monkeypatch.setattr(
        dev.jax, "device_put", lambda value, device: (np.asarray(value), device)
    )
    moved, placed = dev._put_numeric_leaves(Array(), target)
    np.testing.assert_array_equal(moved, [1.0, 2.0])
    assert placed is target


def test_put_numeric_leaves_does_not_inspect_tracer_devices(monkeypatch):
    target = SimpleNamespace(platform="gpu")
    monkeypatch.setattr(dev.jax, "device_put", lambda value, _: value)
    gradient = jax.grad(
        lambda x: jax.numpy.sum(dev._put_numeric_leaves(x, target))
    )(jax.numpy.ones(2))
    np.testing.assert_array_equal(gradient, np.ones(2))


def test_free_boundary_uses_shared_device_context(monkeypatch):
    resolution = _res(ns=11, mpol=6, ntor=0)
    seen = {}

    @contextlib.contextmanager
    def fake_context(device, resolved):
        seen["context"] = (device, resolved)
        yield

    def fake_solve(inp, **kwargs):
        seen["solve"] = (inp, kwargs)
        return SimpleNamespace(result="result")

    monkeypatch.setattr(freeboundary, "device_context", fake_context)
    monkeypatch.setattr(freeboundary, "_solve_free_boundary_stage", fake_solve)
    inp = object()
    result = freeboundary.solve_free_boundary(
        inp, resolution=resolution, device="cpu", max_iterations=3
    )

    assert result == "result"
    assert seen["context"] == ("cpu", resolution)
    assert seen["solve"][0] is inp
    assert seen["solve"][1]["resolution"] is resolution
    assert seen["solve"][1]["max_iterations"] == 3
    assert freeboundary.solve_free_boundary.__kwdefaults__["device"] == dev.AUTO
    assert multigrid.solve_free_boundary_multigrid.__kwdefaults__["device"] == dev.AUTO


def test_commit_to_single_device_only_normalizes_one_shared_placement():
    loose = jax.numpy.arange(3.0)
    target = next(iter(loose.devices()))
    pinned = jax.device_put(np.ones(2), target)
    assert not loose._committed and pinned._committed
    host = np.zeros(1)
    tree = {"loose": loose, "pinned": pinned, "label": "kept", "host": host, "none": None}

    out = dev.commit_to_single_device(tree)

    assert out["loose"]._committed and out["pinned"]._committed
    assert out["loose"].sharding == loose.sharding
    np.testing.assert_array_equal(out["loose"], loose)
    np.testing.assert_array_equal(out["pinned"], pinned)
    assert out["label"] == "kept" and out["host"] is host and out["none"] is None

    scalars = {"x": 1.0, "host": host}
    assert dev.commit_to_single_device(scalars) is scalars
    mesh = jax.sharding.Mesh(np.asarray([target]), ("device",))
    named = jax.device_put(
        np.ones(2), jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    )
    mixed = (loose, named)
    assert dev.commit_to_single_device(mixed) is mixed

    traced = []
    jax.make_jaxpr(
        lambda x: traced.append(dev.commit_to_single_device((x, pinned))[0] is x) or x
    )(loose)
    assert traced == [True]


def test_placement_neutral_clears_the_context_only_for_committed_jit_arguments():
    loose = jax.numpy.arange(3.0)
    target = next(iter(loose.devices()))
    pinned = jax.device_put(np.ones(2), target)

    def clears(tree):
        with jax.default_device(target):
            with dev.placement_neutral(tree):
                return jax.config.jax_default_device is None

    with jax.disable_jit(False):
        assert clears((pinned, "label"))
        assert not clears((loose, pinned))
        assert not clears({"x": 1.0})
        assert isinstance(dev.placement_neutral((loose,)), contextlib.nullcontext)
    with jax.disable_jit(True):
        assert not clears((pinned,))


def test_host_callback_is_not_pinned_across_platforms():
    """A CPU pin inside an accelerator computation is not expressible in JAX.

    ``resolve_implicit_device`` stands the implicit-gradient path down to the
    CPU on an accelerator backend.  Pinning the host callback there while the
    enclosing jit compiles for the accelerator made JAX's lowering raise
    ``tuple.index(x): x not in tuple`` -- every jitted optimization gradient on
    a GPU machine.  Same-platform pins, including the two-accelerator case the
    pin exists for, are kept.
    """
    import dataclasses

    from vmex.core import implicit as im

    class _FakeDevice:
        def __init__(self, platform: str) -> None:
            self.platform = platform

    cfg = im.make_config(VmecInput.from_file(DATA / "input.solovev"), multigrid=True)
    here = jax.default_backend()
    other = "gpu" if here != "gpu" else "tpu"

    assert im._callback_sharding(dataclasses.replace(cfg, device=None)) is None
    assert im._callback_sharding(
        dataclasses.replace(cfg, device=_FakeDevice(other))) is None
    same = im._callback_sharding(
        dataclasses.replace(cfg, device=jax.devices(here)[0]))
    assert isinstance(same, jax.sharding.SingleDeviceSharding)


@pytest.mark.parametrize("root_index, template_index, explicit_index", [
    (0, 0, None), (1, 1, None), (1, 0, None), (0, 1, None),
    (0, 1, 0), (1, 0, 1),
])
def test_host_callback_refinement_follows_solved_cpu_device(
        monkeypatch, root_index, template_index, explicit_index):
    """Callback inputs meet on the root or already-cached template device."""
    from vmex.core import implicit as im

    cfg = im.make_config(VmecInput.from_file(DATA / "input.solovev"))
    devices = jax.devices("cpu")
    if len(devices) <= max(root_index, template_index):
        pytest.skip("requires two forced host devices")
    root_device = devices[root_index]
    template_device = devices[template_index]
    expected_device = (devices[explicit_index] if explicit_index is not None
                       else template_device)
    cfg = replace(cfg, device=(devices[explicit_index]
                              if explicit_index is not None else None))
    params = im.params_from_input(cfg.inp)
    params_np = jax.tree.map(np.asarray, params)
    leaf = jax.device_put(np.ones((2, 2)), root_device)
    state = im.SpectralState(*(leaf,) * 6)
    host_mask = jax.tree.map(lambda value: np.ones_like(value), state)
    original_template_runtime = im._template_runtime
    original_template_runtime.cache_clear()
    with jax.default_device(template_device):
        template = im._put_numeric_leaves(
            original_template_runtime(cfg), template_device)

    monkeypatch.setattr(
        im, "_host_solve", lambda *_: SimpleNamespace(state=state))
    monkeypatch.setattr(im, "_template_runtime", lambda *_: template)
    monkeypatch.setattr(im, "_boundary_pack_tables", lambda *_: None)
    key = ("callback-alignment-test",)
    monkeypatch.setattr(
        im, "_mask_cache_key", lambda *_: key)
    monkeypatch.setitem(im._MASK_CACHE, key, host_mask)

    def refine(_cfg, aligned_params, solved, aligned_mask):
        for tree in (aligned_params, solved, aligned_mask):
            assert all(
                value.devices() == {expected_device}
                for value in jax.tree.leaves(tree)
            )
        return solved

    monkeypatch.setattr(im, "_refine_fixed_point", refine)
    returned, returned_mask = im._host_solve_and_mask(cfg, params_np)

    assert cfg.device == (None if explicit_index is None else expected_device)
    assert all(isinstance(value, np.ndarray)
               for value in jax.tree.leaves(returned))
    assert all(isinstance(value, np.ndarray)
               for value in jax.tree.leaves(returned_mask))


@pytest.mark.parametrize("caller_index, device_index, explicit_index", [
    (0, 0, None), (0, 1, None), (1, 0, None), (0, 1, 0), (1, 0, 1),
])
def test_status_certificate_preserves_returned_state_on_runtime_device(
        monkeypatch, caller_index, device_index, explicit_index):
    """Host conversion preserves coefficients and the runtime's placement."""
    from vmex.core import implicit as im

    devices = jax.devices("cpu")
    if len(devices) <= max(caller_index, device_index):
        pytest.skip("requires two forced host devices")
    device = devices[device_index]
    expected_device = (devices[explicit_index] if explicit_index is not None
                       else device)
    inp = VmecInput.from_file(DATA / "input.solovev").change_resolution(
        mpol=2, ntor=0, ntheta=8, nzeta=4)
    inp = replace(inp, ns_array=np.asarray([3]))
    cfg = im.make_config(inp, device=(devices[explicit_index]
                                     if explicit_index is not None else None))
    params = im.params_from_input(inp)
    template = im._put_numeric_leaves(im._template_runtime(cfg), device)
    monkeypatch.setattr(im, "_template_runtime", lambda _: template)
    returned = im._initial_state(template.setup)
    returned_np = jax.tree.map(np.asarray, returned)
    mask_np = jax.tree.map(np.ones_like, returned_np)
    key = im._params_key(params)
    result = SimpleNamespace(fsqr=1e-14, fsqz=0.0, fsql=0.0)
    monkeypatch.setitem(im._LAST_SOLVE, cfg, (key, result))
    # A memo's coefficients must never replace the actual callback output.
    different = jax.tree.map(lambda x: x + 1.0, returned)
    monkeypatch.setitem(im._LAST_REFINED, cfg, (key, different))
    monkeypatch.setattr(im, "_host_solve_and_mask_impl",
                        lambda *_: (returned_np, mask_np))

    def measurements(state, aligned_params, mask, config):
        assert config is cfg
        # Materialize host values as the compiled measurement would.
        state, aligned_params, mask = jax.tree.map(
            jax.numpy.asarray, (state, aligned_params, mask))
        for tree in (state, aligned_params, mask):
            assert all(isinstance(x, jax.Array) and x.devices() == {expected_device}
                       for x in jax.tree.leaves(tree))
        for actual, expected in zip(jax.tree.leaves(state),
                                    jax.tree.leaves(returned_np)):
            np.testing.assert_array_equal(actual, expected)
        return 1e-8, 1e-8, 1e-14, True

    monkeypatch.setattr(im, "_primal_measurements", measurements)
    with jax.default_device(devices[caller_index]):
        _, _, status, _, _ = im._host_solve_and_mask_status(
            cfg, jax.tree.map(np.asarray, params))
    assert status == 0
    assert cfg.device == (None if explicit_index is None else expected_device)
    assert im._LAST_PRIMAL_CERTIFICATE[cfg][1] == im._primal_state_key(returned_np)
