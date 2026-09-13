"""Analytic array-only regression: no equilibrium jobs or persistent fixtures."""
import unittest
import numpy as np
from diagnose_qa_current_profile import diagnose, MU0


class CurrentProfileRegression(unittest.TestCase):
    def test_helical_field_current_and_drive_support(self):
        R=np.linspace(1.,2.,9);Z=np.linspace(-.5,.5,7);phi=np.arange(8)*np.pi/8
        shape=(8,7,9);B=np.zeros(shape+(3,))
        # Bphi=R, Bz=R² => Jphi=-2R/mu0, Jz=2/mu0, J.B=0.
        B[...,1]=R;B[...,2]=R**2
        P=np.broadcast_to(np.where(R>=1.75,0.,2.-R),shape).copy()
        wall=np.broadcast_to(R<1.5,shape).copy()
        report=diagnose(R,Z,phi,B,np.zeros_like(B),P,wall,np.array([0.,1.]),np.array([-1.,-1.]),-1000.)
        all_cells=report["supports"]["all_interior"]
        expected=-2*sum(R[2:7])*3*.125/6/MU0
        np.testing.assert_allclose(all_cells["attained_current_A_by_cut"],expected,rtol=2e-13)
        self.assertAlmostEqual(all_cells["candidate_drive_current_A_by_cut"][0],-1000.,places=10)
        self.assertLess(abs(all_cells["attained_JdotB_over_B2_A_per_T_m2"]),1e-8)
        zero=report["supports"]["zero_pressure"]
        self.assertLess(zero["attained_mean_current_A"],0.)
        self.assertEqual(zero["candidate_drive_mean_current_A"],0.)
        self.assertGreater(all_cells["attained_abs_toroidal_current_A_mean"],0.)
        self.assertEqual(sum(x["cells"] for x in report["profile_bins"])+report["supports"]["drive_excluded_s_ge_1"]["cells"],all_cells["cells"])
        self.assertFalse(report["drive_is_saved_native_field"])

    def test_rejects_zero_toroidal_flux(self):
        R=np.linspace(1.,2.,7);Z=np.linspace(-1.,1.,7);phi=np.arange(8)*np.pi/8
        B=np.zeros((8,7,7,3));B[...,2]=R**2
        with self.assertRaisesRegex(ValueError,"toroidal flux"):
            diagnose(R,Z,phi,B,np.zeros_like(B),np.ones(B.shape[:-1]),np.ones(B.shape[:-1],bool),
                     np.array([0.,1.]),np.array([-1.,-1.]),-1000.)


if __name__=="__main__":unittest.main(verbosity=2)
