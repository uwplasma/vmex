"""Input parsing and writer round-trip tests for ``vmex.core.input``.

For every bundled input deck, :class:`vmex.core.input.VmecInput` is
round-tripped through both writers (JSON and INDATA) and must reproduce every
field exactly.  A VMEC++ JSON example (``data/solovev.json``, copied verbatim
from the vmecpp repository) validates JSON-schema compatibility.  (Field-level
parity with the historical parser stack was proven by the A/B suite that
accompanied the port and retired with the legacy tree.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmex.core.input import VmecInput

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
DECKS = sorted(DATA.glob("input.*"))
FIXTURES = Path(__file__).parent / "data"

assert DECKS, f"no input decks found under {DATA}"


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.name)
def test_json_round_trip(deck: Path, tmp_path: Path) -> None:
    """VmecInput -> to_json -> from_file reproduces every field exactly."""
    original = VmecInput.from_file(deck)
    reread = VmecInput.from_file(original.to_json(tmp_path / "roundtrip.json"))
    assert reread == original


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.name)
def test_indata_round_trip(deck: Path, tmp_path: Path) -> None:
    """VmecInput -> to_indata -> from_file reproduces every field exactly."""
    original = VmecInput.from_file(deck)
    reread = VmecInput.from_file(original.to_indata(tmp_path / "input.roundtrip"))
    assert reread == original


def test_vmecpp_json_example() -> None:
    """The VMEC++ solovev.json example parses with VMEC++ schema semantics."""
    new = VmecInput.from_file(FIXTURES / "solovev.json")
    assert new.mpol == 6
    assert new.ntor == 0
    assert new.nfp == 1  # default
    assert new.lasym is False  # default
    assert new.ncurr == 0
    assert new.nstep == 250
    assert new.delt == 0.9
    np.testing.assert_array_equal(new.ns_array, [5, 11, 55])
    np.testing.assert_array_equal(new.niter_array, [1000, 2000, 2000])
    np.testing.assert_array_equal(new.ftol_array, [1e-16, 1e-16, 1e-16])
    np.testing.assert_array_equal(new.am[:3], [0.125, -0.125, 0.0])
    assert not np.any(new.am[3:])
    np.testing.assert_array_equal(new.ai[:2], [1.0, 0.0])
    np.testing.assert_array_equal(new.raxis_c, [4.0])
    np.testing.assert_array_equal(new.zaxis_s, [0.0])
    # Sparse boundary lists land at [n + ntor, m] = [0, m] for ntor = 0.
    np.testing.assert_array_equal(new.rbc[0, :3], [3.999, 1.026, -0.068])
    np.testing.assert_array_equal(new.zbs[0, :3], [0.0, 1.58, 0.01])
    assert not np.any(new.rbs) and not np.any(new.zbc)
    assert new.lfreeb is False  # VMEC++ default
    assert new.gamma == 0.0  # default (JSON alias: adiabatic_index)
    assert new.mgrid_file == "NONE"

    # And it survives both writers.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        assert VmecInput.from_file(new.to_json(Path(tmp) / "x.json")) == new
        assert VmecInput.from_file(new.to_indata(Path(tmp) / "input.x")) == new


def test_indexed_indata_vectors_overlay_vmec_defaults() -> None:
    """Indexed 1-D assignments follow VMEC2000 namelist lower bounds/defaults."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 3
        NTOR = 0
        APHI(2) = 0.25
        AM(0) = 1000.0
        AI(1) = -0.125
        AC(2) = 0.5
        NS_ARRAY(1) = 7
        NS_ARRAY(2) = 15
        FTOL_ARRAY(2) = 1e-12
        NITER_ARRAY(2) = 400
        EXTCUR(2) = -3.0
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )

    # vmec_input.f initializes APHI(1)=1 before applying APHI(2).
    np.testing.assert_array_equal(inp.aphi[:3], [1.0, 0.25, 0.0])
    np.testing.assert_array_equal(inp.am[:3], [1000.0, 0.0, 0.0])
    np.testing.assert_array_equal(inp.ai[:3], [0.0, -0.125, 0.0])
    np.testing.assert_array_equal(inp.ac[:3], [0.0, 0.0, 0.5])
    np.testing.assert_array_equal(inp.ns_array, [7, 15])
    np.testing.assert_array_equal(inp.ftol_array, [1e-10, 1e-12])
    # A partial NITER_ARRAY assignment prevents VMEC2000's all--1 fallback;
    # unassigned active entries therefore remain -1.
    np.testing.assert_array_equal(inp.niter_array, [-1, 400])
    np.testing.assert_array_equal(inp.extcur, [0.0, -3.0])


def test_indexed_aphi_can_override_identity_term() -> None:
    """APHI(1) is one-based and may explicitly replace the VMEC default."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 3
        NTOR = 0
        APHI(1) = 0.5
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )
    np.testing.assert_array_equal(inp.aphi[:3], [0.5, 0.0, 0.0])


def test_indexed_aphi_overlays_dense_assignment() -> None:
    """A sparse APHI term is not discarded after a dense leading zero."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 3
        NTOR = 0
        APHI = 0.0
        APHI(2) = 1.0
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )
    np.testing.assert_array_equal(inp.aphi[:3], [0.0, 1.0, 0.0])


def test_indexed_aphi_starting_element_consumes_following_values() -> None:
    """Fortran ``APHI(1)=0,1`` assigns two elements, not one scalar."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 3
        NTOR = 0
        APHI(1) = 0.0, 1.0
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )
    np.testing.assert_array_equal(inp.aphi[:3], [0.0, 1.0, 0.0])


def test_indexed_aphi_section_uses_fortran_inclusive_bounds() -> None:
    """A non-leading array section retains VMEC's initialized first term."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 3
        NTOR = 0
        APHI(2:3) = 0.25, 0.5
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )
    np.testing.assert_array_equal(inp.aphi[:4], [1.0, 0.25, 0.5, 0.0])


def test_boundary_section_accepts_toroidal_mode_range() -> None:
    """VMEC boundary rows accept compact ``RBC(nlo:nhi,m)`` assignments."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 2
        NTOR = 2
        RBC(-2:2,0) = 1.0, 2.0, 3.0, 4.0, 5.0
        ZBS(-2:2,1) = -1.0, -2.0, -3.0, -4.0, -5.0
        /
        """
    )
    np.testing.assert_array_equal(inp.rbc[:, 0], [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(inp.zbs[:, 1], [-1.0, -2.0, -3.0, -4.0, -5.0])


def test_boundary_section_uses_fortran_column_major_order() -> None:
    """The first boundary subscript varies fastest, as in Fortran namelists."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 2
        NTOR = 1
        RBC(-1:1,0:1) = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
        /
        """
    )
    np.testing.assert_array_equal(inp.rbc[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(inp.rbc[:, 1], [4.0, 5.0, 6.0])


def test_boundary_section_accepts_negative_stride() -> None:
    """Inclusive section bounds and negative strides follow Fortran syntax."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 2
        NTOR = 2
        RBC(2:-2:-2,0) = 1.0, 2.0, 3.0
        /
        """
    )
    np.testing.assert_array_equal(inp.rbc[:, 0], [3.0, 0.0, 2.0, 0.0, 1.0])


def test_indexed_legacy_axis_overlays_vmec_axis() -> None:
    """Indexed obsolete RAXIS/ZAXIS entries retain VMEC2000 compatibility."""
    inp = VmecInput.from_indata_text(
        """&INDATA
        MPOL = 3
        NTOR = 1
        RAXIS_CC(0) = 1.0
        RAXIS(0) = 2.0
        RAXIS(1) = 0.25
        ZAXIS(1) = -0.5
        RBC(0,0) = 1.0
        RBC(0,1) = 0.1
        ZBS(0,1) = 0.1
        /
        """
    )
    np.testing.assert_array_equal(inp.raxis_c, [2.0, 0.25])
    np.testing.assert_array_equal(inp.zaxis_s, [0.0, -0.5])
