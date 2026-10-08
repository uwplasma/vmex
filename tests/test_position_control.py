"""Position control for free-boundary solves."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vmex.core import freeboundary as FB  # noqa: E402
from vmex.core.fourier import mode_table  # noqa: E402
from vmex.core.mgrid import MgridField  # noqa: E402
from vmex.core.multigrid import solve_free_boundary_multigrid  # noqa: E402
from vmex.core.position_control import (  # noqa: E402
    ControlledField, PositionControl, PositionControlResult, initial_control,
    measure_position, update_control, wrap_field,
)
from vmex.core.solver import SpectralState  # noqa: E402

from tests.test_lasym_free_case import lasym_free_field, lasym_free_input  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

REPO = Path(__file__).resolve().parents[1]


def _field(nfp: int = 2) -> MgridField:
    z = jnp.zeros((1, 2, 2, 2))
    return MgridField(br=z, bp=z, bz=z, extcur=jnp.ones(1),
                      rmin=0.5, rmax=1.5, zmin=-0.5, zmax=0.5, nfp=nfp)


def _control(nmax: int, **amplitudes) -> PositionControlResult:
    return PositionControlResult(
        bz=np.asarray(amplitudes.get("bz", np.zeros(nmax + 1)), dtype=float),
        br=np.asarray(amplitudes.get("br", np.zeros(nmax)), dtype=float),
        target=np.zeros(2 * nmax + 1), measured=np.zeros(2 * nmax + 1),
        integral=np.zeros(2 * nmax + 1), error=np.zeros(2 * nmax + 1), r0=1.0)


def test_uniform_vertical_field_and_delegation() -> None:
    base = _field()
    field = wrap_field(base, _control(0, bz=[0.25]), stop=7)
    r, p, z = jnp.array([1.0]), jnp.array([0.1]), jnp.array([0.2])
    out = field.b_cyl(r, p, z)
    np.testing.assert_allclose(out[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(out[1], 0.0, atol=1e-14)
    np.testing.assert_allclose(out[2], 0.25)
    assert field.br is base.br and int(field.stop) == 7
    again = wrap_field(field, _control(0, bz=[-0.1]))
    assert not isinstance(again.base, ControlledField)   # re-wrapping does not nest
    leaves, tree = jax.tree_util.tree_flatten(again)
    assert isinstance(jax.tree_util.tree_unflatten(tree, leaves), ControlledField)


def test_harmonic_channels_are_curl_and_divergence_free() -> None:
    """The n >= 1 channels are exact vacuum fields (potential harmonics)."""
    field = wrap_field(_field(), _control(2, bz=[0.1, 0.2, -0.3], br=[0.4, -0.2]))

    def cart(x):
        r, p, z = np.hypot(x[0], x[1]), np.arctan2(x[1], x[0]), x[2]
        br, bp, bz = (np.asarray(v).item() for v in field.b_cyl(jnp.array(r), jnp.array(p), jnp.array(z)))
        return np.array([br * np.cos(p) - bp * np.sin(p), br * np.sin(p) + bp * np.cos(p), bz])

    x0, h = np.array([0.9, 0.3, 0.07]), 1e-5
    jac = np.stack([(cart(x0 + h * e) - cart(x0 - h * e)) / (2 * h) for e in np.eye(3)], axis=1)
    assert abs(np.trace(jac)) < 1e-7
    np.testing.assert_allclose(jac, jac.T, atol=1e-7)   # curl-free
    assert np.linalg.norm(jac) > 0.1


def test_update_law_sign_clip_deadband_and_history() -> None:
    cfg = PositionControl(gain=2.0, integral_gain=0.0, derivative_gain=0.0, max_field=1.0, max_step=1.0)
    ctl = _control(0)
    ctl = PositionControlResult(**{**ctl.__dict__, "target": np.array([1.0])})
    out = update_control(cfg, ctl, np.array([1.1]), 10, sign=+1.0, fsq=3.0)   # too far out: push inward
    np.testing.assert_allclose(out.bz, [-0.2])
    assert update_control(cfg, ctl, np.array([1.1]), 10, sign=-1.0).bz[0] > 0
    assert update_control(cfg, ctl, np.array([5.0]), 10, sign=+1.0).bz[0] == -1.0
    np.testing.assert_allclose(out.history[0, :3], [10, 3.0, -0.2])
    held = update_control(PositionControl(gain=2.0, deadband=0.5), ctl, np.array([1.1]), 10, sign=+1.0)
    assert held.bz[0] == 0.0                                                    # inside the dead band
    slewed = update_control(PositionControl(gain=2.0, max_step=0.05), ctl, np.array([1.1]), 10, sign=+1.0)
    assert slewed.bz[0] == pytest.approx(-0.05)


def test_measurement_reads_axis_and_edge_coefficients() -> None:
    modes = mode_table(2, 1)
    i00 = int(np.flatnonzero((modes.m == 0) & (modes.n == 0))[0])
    i01 = int(np.flatnonzero((modes.m == 0) & (modes.n == 1))[0])
    R, Z = np.zeros((3, modes.mnmax)), np.zeros((3, modes.mnmax))
    R[0, i00], R[-1, i00], R[0, i01], Z[0, i01] = 1.2, 1.5, 0.05, 0.02
    zero = jnp.zeros_like(R)
    state = SpectralState(jnp.asarray(R), zero, zero, jnp.asarray(Z), zero, zero)
    np.testing.assert_allclose(measure_position(state, modes, PositionControl(nmax=1)), [1.2, 0.05, 0.02])
    assert measure_position(state, modes, PositionControl(measure="boundary"))[0] == pytest.approx(1.5)
    ctl = initial_control(PositionControl(target=1.1), state, modes)
    assert ctl.target[0] == 1.1 and ctl.r0 == 1.1 and not ctl.bz.any()


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        PositionControl(measure="centroid")
    with pytest.raises(ValueError):
        PositionControl(interval=0)
    with pytest.raises(ValueError):
        PositionControl(nmax=-1)


def test_solver_converges_with_a_displaced_target_and_is_off_by_default() -> None:
    """Tokamak fixture: a 2 cm outward target converges with a few mT of vertical field."""
    inp = dataclasses.replace(
        lasym_free_input(REPO / "examples" / "data"), ns_array=np.asarray([16]),
        ftol_array=np.asarray([1.0e-10]), niter_array=np.asarray([2500]))
    field = lasym_free_field()
    plain = FB.solve_free_boundary(inp, external_field=field, max_iterations=2500)
    assert plain.position_control is None
    r_plain = measure_position(plain.state, mode_table(int(inp.mpol), int(inp.ntor)), PositionControl())[0]
    run = FB.solve_free_boundary(
        inp, external_field=field, max_iterations=2500,
        position_control=PositionControl(target=float(r_plain) + 0.02))
    ctl = run.position_control
    assert run.converged and ctl.history.shape[0] > 0
    assert 0.001 < ctl.vertical_field < 0.02          # outward target: positive B_Z, a few mT
    assert abs(ctl.measured[0] - ctl.target[0]) < 5e-3
    shift = measure_position(run.state, mode_table(int(inp.mpol), int(inp.ntor)), PositionControl())[0] - r_plain
    assert 0.01 < shift < 0.03


def test_multigrid_carries_the_control_across_rungs() -> None:
    """The target, amplitudes and history continue from one radial rung to the next."""
    inp = dataclasses.replace(
        lasym_free_input(REPO / "examples" / "data"), ns_array=np.asarray([8, 16]),
        ftol_array=np.asarray([1.0e-8, 1.0e-10]), niter_array=np.asarray([1500, 2500]))
    modes = mode_table(int(inp.mpol), int(inp.ntor))
    plain = solve_free_boundary_multigrid(inp, external_field=lasym_free_field())
    r_plain = measure_position(plain.state, modes, PositionControl())[0]
    with pytest.raises(ValueError, match="NTOR"):
        solve_free_boundary_multigrid(inp, external_field=lasym_free_field(),
                                      position_control=PositionControl(nmax=1))
    run = solve_free_boundary_multigrid(
        inp, external_field=lasym_free_field(),
        position_control=PositionControl(target=float(r_plain) + 0.01, interval=20),
        raise_on_max_iterations=False)
    ctl = run.position_control
    assert plain.position_control is None and ctl is not None
    assert ctl.target[0] == pytest.approx(r_plain + 0.01) and ctl.bz.shape == (1,) and ctl.br.shape == (0,)
    assert np.isfinite(ctl.history).all() and np.diff(ctl.history[:, 0]).min() < 0   # iteration restarts on rung 2


### Input-deck option (LPOSITION_CONTROL) ####################################

DATA = REPO / "examples" / "data"
CONTROL_DECK = DATA / "input.DIII-D_position_control"
CONTROL_MGRID = DATA / "mgrid_d3d_vertical_deficit.nc"


def test_deck_parameters_round_trip_and_default_off(tmp_path) -> None:
    from vmex.core.input import VmecInput

    deck = VmecInput.from_file(CONTROL_DECK)
    assert deck.lposition_control and deck.position_target == pytest.approx(1.7199)
    assert (deck.position_interval, deck.position_gain, deck.position_nmax) == (20, 4.0, 0)
    for writer, name in ((deck.to_indata, "input.rt"), (deck.to_json, "input.rt.json")):
        out = writer(tmp_path / name)
        again = VmecInput.from_file(out)
        assert again.lposition_control and again.position_target == deck.position_target
        assert again.position_interval == 20 and again.position_gain == 4.0
    ctl = PositionControl.from_input(deck)
    assert ctl.target == pytest.approx(1.7199) and ctl.gain == 4.0
    assert ctl.integral_gain == pytest.approx(0.01) and ctl.derivative_gain == pytest.approx(200.0)
    # a deck without the flag writes none of the keys and requests no control
    plain = VmecInput.from_file(DATA / "input.DIII-D_lasym_false")
    assert not plain.lposition_control and PositionControl.from_input(plain) is None
    plain.to_indata(tmp_path / "input.plain")
    assert "POSITION" not in (tmp_path / "input.plain").read_text()
    from vmex.core.position_control import resolve_position_control
    explicit = PositionControl(gain=1.0)
    assert resolve_position_control(explicit, deck) is explicit      # explicit argument wins
    assert resolve_position_control(False, deck) is None             # False forces it off
    assert resolve_position_control(None, plain) is None


@pytest.mark.parametrize("key, value", [("POSITION_INTERVAL", "0"), ("POSITION_GAIN", "-1.0"),
                                        ("POSITION_NMAX", "3")])
def test_deck_parameters_are_validated(tmp_path, key, value) -> None:
    from vmex.core.input import VmecInput

    text = CONTROL_DECK.read_text()
    bad = tmp_path / "input.bad"
    bad.write_text("\n".join(f"  {key} = {value}" if line.strip().startswith(key) else line
                             for line in text.splitlines()))
    with pytest.raises(ValueError, match="POSITION"):
        VmecInput.from_file(bad)


def test_deck_with_vertical_field_deficit_needs_and_gets_the_control() -> None:
    """The 40 mT deficit defeats the plain solve; the deck's control converges and recovers it."""
    from vmex.core.input import VmecInput

    inp = VmecInput.from_file(CONTROL_DECK)
    off = solve_free_boundary_multigrid(inp, mgrid_path=CONTROL_MGRID, position_control=False,
                                        raise_on_max_iterations=False)
    assert not off.converged and off.position_control is None
    on = solve_free_boundary_multigrid(inp, mgrid_path=CONTROL_MGRID)
    assert on.converged and on.position_control is not None
    assert on.position_control.vertical_field == pytest.approx(0.040, abs=0.002)
    assert abs(on.position_control.measured[0] - 1.7199) < 5e-3


def test_cli_runs_the_control_deck(tmp_path, capsys) -> None:
    import shutil
    from vmex.core import cli

    for src in (CONTROL_DECK, CONTROL_MGRID):
        shutil.copy(src, tmp_path / src.name)
    rc = cli.main([str(tmp_path / CONTROL_DECK.name)])
    out = capsys.readouterr().out
    assert rc == 0 and "POSITION CONTROL: B_Z^ctrl = 39." in out
