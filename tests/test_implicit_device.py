"""Focused placement tests for the public implicit-differentiation API."""

from __future__ import annotations

import jax
import pytest

from vmex.core import implicit as im
from vmex.core import optimize as opt
from vmex.core.device import AUTO
from vmex.core.input import VmecInput


class _Stop(Exception):
    pass


def _device(kind):
    try:
        return jax.devices(kind)[0]
    except RuntimeError:
        pytest.skip(f"{kind.upper()} unavailable")


def _platforms(params):
    def platform(array):
        device = array.device
        return (device() if callable(device) else device).platform

    return {platform(leaf) for leaf in jax.tree.leaves(params)}


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_params_from_input_honors_explicit_device(kind):
    params = im.params_from_input(VmecInput(), device=_device(kind))
    assert _platforms(params) == {kind}


@pytest.mark.parametrize("requested", [AUTO, None, "cpu"])
def test_run_forwards_device_when_constructing_params(monkeypatch, requested):
    seen = []

    def params_from_input(inp, *, device=None):
        seen.append(device)
        raise _Stop

    monkeypatch.setattr(im, "params_from_input", params_from_input)
    with pytest.raises(_Stop):
        im.run(VmecInput(), device=requested)
    assert seen == [requested]


def test_run_preserves_supplied_params_for_auto_and_none(monkeypatch):
    params = im.params_from_input(VmecInput(), device="cpu")

    def solve_implicit(got, cfg):
        assert got is params
        raise _Stop

    monkeypatch.setattr(im, "solve_implicit", solve_implicit)
    for requested in (AUTO, None):
        with pytest.raises(_Stop):
            im.run(VmecInput(), params, device=requested)


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_run_places_supplied_params_on_requested_device(monkeypatch, kind):
    requested = _device(kind)
    inp = VmecInput()
    params = im.params_from_input(inp, device="cpu")

    def solve_implicit(got, cfg):
        assert _platforms(got) == {kind}
        raise _Stop

    monkeypatch.setattr(im, "solve_implicit", solve_implicit)
    with pytest.raises(_Stop):
        im.run(inp, params, device=requested)


def test_least_squares_places_params_on_jacobian_device(monkeypatch):
    requested = _device("cpu")
    seen = []

    def params_from_input(inp, *, device=None):
        seen.append(device)
        raise _Stop

    monkeypatch.setattr(im, "params_from_input", params_from_input)
    with pytest.raises(_Stop):
        opt.least_squares(
            [(opt.aspect_ratio, 4.0, 1.0)], VmecInput(), jac="implicit",
            device=requested,
        )
    assert seen == [requested]
