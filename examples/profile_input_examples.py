"""Create VMEC inputs with polynomial and spline pressure/current profiles.

Run this script from a source checkout with:

    python examples/profile_input_examples.py

It writes two input decks under ``examples/outputs/profile_inputs`` and prints
the corresponding ``vmec`` commands.  The generated decks are intentionally
small and editable: use them as templates for finite-beta runs, or copy the
profile blocks into another VMEC input.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from vmec_jax.namelist import InData, read_indata, write_indata
from vmec_jax.profiles import eval_profiles


HERE = Path(__file__).resolve().parent
BASE_INPUT = HERE / "data" / "input.profile_splines"
DEFAULT_OUTDIR = HERE / "outputs" / "profile_inputs"

# Radial knots use normalized toroidal flux s in [0, 1].
S_KNOTS = [0.0, 0.25, 0.50, 0.75, 1.0]

# Pressure examples are in Pa before VMEC multiplies by PRES_SCALE.
POLYNOMIAL_PRESSURE_AM = [1.0e4, -7.5e3, -2.5e3]  # p(s) = 1e4 - 7.5e3*s - 2.5e3*s^2
SPLINE_PRESSURE_VALUES = [1.0e4, 8.8e3, 6.0e3, 2.5e3, 0.0]

# Current examples use the enclosed toroidal-current profile I(s).
# To prescribe I'(s) instead, use PCURR_TYPE="power_series" or "*_ip".
POLYNOMIAL_CURRENT_AC = [1.2e5, -3.0e4]  # I(s) = 1.2e5*s - 3e4*s^2
SPLINE_CURRENT_VALUES = [0.0, 3.0e4, 6.5e4, 8.5e4, 9.0e4]


def _drop_profile_keys(indata: InData) -> InData:
    """Remove profile keys that are not used by the constructed example deck."""

    out = deepcopy(indata)
    for key in (
        "AM",
        "AM_AUX_S",
        "AM_AUX_F",
        "AI",
        "AI_AUX_S",
        "AI_AUX_F",
        "PIOTA_TYPE",
        "AC",
        "AC_AUX_S",
        "AC_AUX_F",
        "PCURR_TYPE",
        "PMASS_TYPE",
        "PRES_SCALE",
    ):
        out.scalars.pop(key, None)
    return out


def polynomial_pressure_current_indata(base: InData) -> InData:
    """Return a current-driven deck using polynomial pressure and current."""

    indata = _drop_profile_keys(base)
    indata.scalars.update(
        {
            "NCURR": 1,
            "PMASS_TYPE": "power_series",
            "AM": POLYNOMIAL_PRESSURE_AM,
            "PRES_SCALE": 1.0,
            # power_series_i means AC gives enclosed current I(s).  Use
            # power_series to prescribe I'(s), matching VMEC's derivative form.
            "PCURR_TYPE": "power_series_i",
            "AC": POLYNOMIAL_CURRENT_AC,
        }
    )
    return indata


def spline_pressure_current_indata(base: InData) -> InData:
    """Return a current-driven deck using tabulated spline profiles."""

    indata = _drop_profile_keys(base)
    indata.scalars.update(
        {
            "NCURR": 1,
            "PMASS_TYPE": "cubic_spline",
            "AM_AUX_S": S_KNOTS,
            "AM_AUX_F": SPLINE_PRESSURE_VALUES,
            "PRES_SCALE": 1.0,
            # akima_spline_i means AC_AUX_F gives enclosed current I(s).  Use
            # akima_spline_ip or cubic_spline_ip to prescribe I'(s) instead.
            "PCURR_TYPE": "akima_spline_i",
            "AC_AUX_S": S_KNOTS,
            "AC_AUX_F": SPLINE_CURRENT_VALUES,
            # AC only selects that a current profile is present; the spline
            # values above carry the actual radial dependence.
            "AC": [0.0],
        }
    )
    return indata


def _profile_table(indata: InData) -> str:
    s = np.linspace(0.0, 1.0, 6)
    values = eval_profiles(indata, s)
    pressure = np.asarray(values["pressure_pa"], dtype=float)
    current = np.asarray(values["current"], dtype=float)

    lines = ["    s        pressure[Pa]        I(s)"]
    for sj, pj, ij in zip(s, pressure, current):
        lines.append(f"  {sj:5.2f}    {pj:14.6e}  {ij:14.6e}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for generated input decks (default: examples/outputs/profile_inputs).",
    )
    args = parser.parse_args(argv)

    base = read_indata(BASE_INPUT)
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    examples = {
        "polynomial": (
            outdir / "input.profile_polynomial_pressure_current",
            polynomial_pressure_current_indata(base),
        ),
        "spline": (
            outdir / "input.profile_spline_pressure_current",
            spline_pressure_current_indata(base),
        ),
    }

    for label, (path, indata) in examples.items():
        write_indata(path, indata)
        print(f"\n{label.capitalize()} profile deck: {path}")
        print(_profile_table(indata))
        print(f"Run with: vmec {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
