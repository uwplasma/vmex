"""Package-surface tests: lazy exports, ``python -m vmec_jax``, CLI helpers.

Covers the ``vmec_jax/__init__.py`` lazy-attribute machinery (every
documented public name resolves; unknown names raise), the source-tree
version resolution, the ``__main__`` module entry point, and the small
pure CLI helpers (``case_from_input``, ``--booz-surfaces`` parsing, and
ESSOS coils-file loading for ``--coils``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import vmec_jax
from vmec_jax.core import cli
from vmec_jax.core.errors import VmecInputError


# ---------------------------------------------------------------------------
# vmec_jax/__init__.py
# ---------------------------------------------------------------------------


def test_version_matches_pyproject():
    src = vmec_jax._source_tree_version()
    assert src is not None
    assert vmec_jax.__version__ == src


def test_source_tree_version_parses_pyproject():
    pyproject = Path(vmec_jax.__file__).resolve().parents[0] / "pyproject.toml"
    expected = None
    in_project = False
    for line in pyproject.read_text().splitlines():
        line = line.strip()
        if line == "[project]":
            in_project = True
        elif in_project and line.startswith("version"):
            expected = line.split("=", 1)[1].strip().strip('"')
            break
    assert vmec_jax._source_tree_version() == expected


def test_every_lazy_export_resolves():
    for name in vmec_jax._LAZY_ATTRS:
        value = getattr(vmec_jax, name)
        assert value is not None, name
    # a second access hits the cached globals() entry
    assert getattr(vmec_jax, "VmecInput") is vmec_jax.VmecInput


def test_unknown_attribute_raises_and_dir_lists_exports():
    with pytest.raises(AttributeError):
        vmec_jax.no_such_symbol  # noqa: B018
    listing = dir(vmec_jax)
    for name in ("VmecInput", "solve", "solve_multigrid", "read_wout", "optimize"):
        assert name in listing
    assert set(vmec_jax.__all__) >= set(vmec_jax._LAZY_ATTRS)


def test_python_dash_m_entrypoint_exposes_cli_main():
    import vmec_jax.__main__ as main_mod

    assert main_mod.main is cli.main


@pytest.mark.parametrize("argv", [["-m", "vmec_jax", "--version"]])
def test_python_dash_m_version(argv):
    proc = subprocess.run([sys.executable, *argv], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0
    assert vmec_jax.__version__ in proc.stdout


# ---------------------------------------------------------------------------
# CLI pure helpers
# ---------------------------------------------------------------------------


def test_case_from_input_naming_conventions(tmp_path):
    assert cli.case_from_input(Path("input.solovev")) == "solovev"
    assert cli.case_from_input(Path("input_solovev")) == "solovev"
    assert cli.case_from_input(Path("solovev.json")) == "solovev"
    assert cli.case_from_input(Path("input.cth.json")) == "cth"


def test_parse_booz_surfaces():
    assert cli._parse_booz_surfaces(None) is None
    assert cli._parse_booz_surfaces("all") is None
    assert cli._parse_booz_surfaces("") is None
    assert cli._parse_booz_surfaces("0.25, 0.5 0.75") == [0.25, 0.5, 0.75]
    with pytest.raises(VmecInputError):
        cli._parse_booz_surfaces("0.25, banana")


def _coils_payload():
    rng = np.random.default_rng(0)
    return {
        "dofs_curves": rng.normal(size=(2, 3, 5)).tolist(),
        "dofs_currents": [1.0e5, 2.0e5],
        "n_segments": 32,
        "nfp": 4,
        "stellsym": True,
        "currents_scale": 2.0,
    }


def test_load_coilset_json_and_npz(tmp_path):
    payload = _coils_payload()

    jpath = tmp_path / "coils.json"
    jpath.write_text(json.dumps(payload))
    coils_j = cli._load_coilset(jpath)
    assert coils_j.n_segments == 32
    assert coils_j.nfp == 4
    assert coils_j.stellsym is True

    npath = tmp_path / "coils.npz"
    np.savez(npath, **{k: np.asarray(v) for k, v in payload.items()})
    coils_n = cli._load_coilset(npath)
    np.testing.assert_allclose(np.asarray(coils_n.base_curve_dofs),
                               np.asarray(coils_j.base_curve_dofs))
    np.testing.assert_allclose(np.asarray(coils_n.base_currents), [1.0e5, 2.0e5])
    assert float(coils_n.current_scale) == 2.0


def test_load_coilset_rejects_malformed_file(tmp_path):
    bad = tmp_path / "coils.json"
    bad.write_text(json.dumps({"not_dofs": []}))
    with pytest.raises(VmecInputError):
        cli._load_coilset(bad)


# ---------------------------------------------------------------------------
# CLI dispatch: --doctor and the argparse usage-error contract
# ---------------------------------------------------------------------------


def test_cli_doctor_flag(capsys):
    assert cli.main(["--doctor"]) == 0
    assert "vmec_jax installation doctor" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    [],                                # no target at all
    ["--plot"],                        # --plot with no path anywhere
    ["--doctor", "input.solovev"],     # --doctor takes no input
    ["--test", "input.solovev"],       # --test takes no input
    ["--test", "--plot", "x.nc"],      # --test and --plot are exclusive
])
def test_cli_usage_errors_exit_2(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_cli_wout_and_boozmn_require_plot_or_booz(tmp_path):
    wout = tmp_path / "wout_x.nc"
    wout.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        cli.main([str(wout)])
    assert exc.value.code == 2

    boozmn = tmp_path / "boozmn_x.nc"
    boozmn.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        cli.main([str(boozmn)])
    assert exc.value.code == 2

    with pytest.raises(SystemExit) as exc:
        cli.main(["--plot", str(wout), str(boozmn)])  # both target forms
    assert exc.value.code == 2


def test_cli_coils_flag_requires_free_boundary_deck(capsys, tmp_path):
    coils = tmp_path / "coils.json"
    coils.write_text(json.dumps(_coils_payload()))
    deck = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.solovev"
    rc = cli.main([str(deck), "--coils", str(coils), "--outdir", str(tmp_path)])
    assert rc != 0
    assert "LFREEB = T" in capsys.readouterr().out


def test_cli_direct_coils_requires_coils_argument(capsys, tmp_path):
    src = (Path(__file__).resolve().parents[1] / "examples" / "data"
           / "input.cth_like_free_bdy_lasym_small")
    text = src.read_text()
    assert "MGRID_FILE" in text
    deck = tmp_path / "input.direct_coils_case"
    deck.write_text(
        "\n".join(
            "  MGRID_FILE = 'DIRECT_COILS'" if "MGRID_FILE" in ln else ln
            for ln in text.splitlines()
        )
    )

    rc = cli.main([str(deck), "--outdir", str(tmp_path)])
    assert rc != 0
    assert "DIRECT_COILS" in capsys.readouterr().out

    rc = cli.main([str(deck), "--coils", str(tmp_path / "missing.json"),
                   "--outdir", str(tmp_path)])
    assert rc != 0
    assert "coils file not found" in capsys.readouterr().out


def test_cli_missing_input_is_zero_crash(capsys):
    rc = cli.main(["/nonexistent/input.notthere"])
    assert rc != 0
    out = capsys.readouterr().out
    assert "HINT" in out and "not found" in out


@pytest.mark.usefixtures("_module_jit_enabled")
def test_cli_bundled_test_verbose(tmp_path, capsys):
    """--test without --quiet prints the guided walkthrough (README quickstart)."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("netCDF4")
    rc = cli.main(["--test", "--outdir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "vmec bundled test" in out
    assert "Equivalent manual command:" in out
    assert "Bundled test complete" in out
    assert (tmp_path / "figures").is_dir()
