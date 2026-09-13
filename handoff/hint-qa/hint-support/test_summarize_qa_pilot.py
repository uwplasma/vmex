"""Temporary-fixture regression for the real QA summarizer (no solver/JAX jobs).

Run with the study Python environment:
    .venv/bin/python local-support/test_summarize_qa_pilot.py
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from netCDF4 import Dataset


SCRIPT = Path(__file__).with_name("summarize_qa_pilot.py")
MU0 = 4e-7 * np.pi


class SummarizerRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Unequal dimensions reveal accidental R/Z transposes.
        self.R = np.linspace(1., 2., 9)
        self.Z = np.linspace(-.5, .5, 7)
        self.phi = np.arange(8) * np.pi / 8

    def fixture(self, orientation=1., equilibrium=False, bad_dimensions=False):
        shape = (8, 7, 9)
        # Deliberately non-curl-free prescribed field: subtraction must occur
        # before computing the response current, regardless of vacuum label.
        vacuum = np.broadcast_to((0. if equilibrium else .3)*self.R**2, shape)
        for name in ("vacuum.nc", "limiter-12.nc", "hint.nc"):
            with Dataset(self.root/name, "w") as f:
                for k, values in (("phi", self.phi), ("Z", self.Z), ("R", self.R)):
                    f.createDimension(k, len(values))
                for k, value in dict(mtor=2, rminb=1, rmaxb=2, zminb=-.5, zmaxb=.5).items():
                    f.createVariable(k, "f8").assignValue(value)
                if name == "vacuum.nc":
                    for k, values in (("R", self.R), ("Z", self.Z), ("phi", self.phi)):
                        f.createVariable(k, "f8", (k,))[:] = values
                    for k in ("Bvac_R", "Bvac_phi", "Bvac_Z"):
                        f.createVariable(k, "f8", ("phi", "Z", "R"))[:] = vacuum if k == "Bvac_Z" else 0.
                elif name == "limiter-12.nc":
                    wall = np.zeros(shape)
                    wall[:, :, 4:] = 1.
                    f.createVariable("limiter", "f8", ("phi", "Z", "R"))[:] = wall
                else:
                    f.createDimension("time", 1)
                    f.createDimension("d001", 1)
                    f.createVariable("t_snap", "f8", ("time", "d001"))[:] = 1.
                    pressure = np.array([0., 0., 0., .001, 1., 0., 2., 0., 0.])
                    if equilibrium:
                        # Bz=±R², curl(B)phi=∓2R, (curl B x B)R=-2R³.
                        # This quartic pressure has exactly that gradient.
                        pressure = 16. - .5*self.R**4
                    for k in ("B_R", "B_phi", "B_Z", "P"):
                        values = np.zeros(shape)
                        if k == "B_Z":
                            values = orientation*np.broadcast_to(self.R**2, shape) + vacuum
                        elif k == "P":
                            values = np.broadcast_to(pressure, shape)
                        dims = ("time", "phi", "Z", "R")
                        if bad_dimensions and k == "B_R":
                            dims = ("time", "phi", "R", "Z")
                            values = values.transpose(0, 2, 1)
                        f.createVariable(k, "f8", dims)[0] = values

    def run_summary(self, error=None):
        result = subprocess.run([sys.executable, str(SCRIPT), str(self.root),
                                 "--target-current", "-100000"],
                                capture_output=True, text=True, timeout=20)
        if error:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertFalse((self.root/"pilot-summary.json").exists())
            return None
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads((self.root/"pilot-summary.json").read_text())["snapshots"][0]

    def test_analytic_response_current_both_orientations_and_masks(self):
        radial_indices = dict(inside_wall=[4, 5, 6], positive_pressure=[3, 4, 6],
                              pressure_above_one_percent=[4, 6], outside_wall=[2, 3],
                              all_interior=[2, 3, 4, 5, 6])
        for orientation in (1., -1.):
            with self.subTest(orientation=orientation):
                self.fixture(orientation)
                row = self.run_summary()
                for support, indices in radial_indices.items():
                    # Analytic Jphi=-2*orientation*R/mu0, rectangular cut
                    # weights and three retained Z samples; no FD oracle.
                    expected = -2*orientation*sum(self.R[indices])/MU0 * .125 * (.5/3) * 3
                    metrics = row["current_by_support"][support]
                    np.testing.assert_allclose(metrics["current_A_by_cut"], expected, rtol=2e-13)
                    self.assertEqual(metrics["cells"], 8*3*len(indices))
                np.testing.assert_allclose(row["current_A_by_cut"],
                    row["current_by_support"]["inside_wall"]["current_A_by_cut"])
                self.assertEqual(row["response_divergence_rms_T_per_m"], 0.)

    def test_exact_cylindrical_force_balance_and_pressure_units(self):
        self.fixture(orientation=-1., equilibrium=True)
        row = self.run_summary()
        self.assertAlmostEqual(row["max_pressure_Pa"], 15.5/MU0)
        expected_integral = 2*np.pi * .125 * (.5/3) * 7 * sum(self.R*(16.-.5*self.R**4))/MU0
        np.testing.assert_allclose(row["pressure_integral_J"], expected_integral, rtol=1e-14)
        for metrics in row["force_by_support"].values():
            self.assertGreater(metrics["grad_pressure_squared_integral_T4_m"], 0.)
            self.assertLess(metrics["ratio_to_grad_pressure_squared"], 1e-25)
            self.assertLess(metrics["ratio_to_grad_pressure_squared_plus_jxb_squared"], 1e-25)

    def test_rejects_toroidal_endpoint(self):
        self.fixture()
        with Dataset(self.root/"vacuum.nc", "a") as f:
            f["phi"][:] = np.linspace(0., np.pi, 8)
        self.run_summary("excludes period endpoint")

    def test_rejects_mismatched_field_period(self):
        self.fixture()
        with Dataset(self.root/"hint.nc", "a") as f:
            f["mtor"].assignValue(3)
        self.run_summary("grid differs from prescribed vacuum")

    def test_rejects_transposed_field_dimensions(self):
        self.fixture(bad_dimensions=True)
        self.run_summary("B_R dimensions must be")


if __name__ == "__main__":
    unittest.main(verbosity=2)
