from __future__ import annotations

import numpy as np

from vmec_jax.vmec_residue import (
    VmecForceNorms,
    VmecForceNormsDynamic,
    vmec_apply_m1_constraints,
    vmec_apply_scalxc_to_tomnsps,
    vmec_fsq_from_tomnsps,
    vmec_fsq_from_tomnsps_dynamic,
    vmec_fsq_sums_from_tomnsps,
    vmec_gcx2_from_tomnsps,
    vmec_gcx2_from_tomnsps_np,
    vmec_scalxc_from_s,
    vmec_zero_m1_zforce,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL


def _constant_tomnsps(*, ns: int = 3, mpol: int = 3, ntor1: int = 1) -> TomnspsRZL:
    shape = (ns, mpol, ntor1)

    def block(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=float)

    return TomnspsRZL(
        frcc=block(1.0),
        frss=block(4.0),
        fzsc=block(2.0),
        fzcs=block(5.0),
        flsc=block(3.0),
        flcs=block(6.0),
        frsc=block(7.0),
        frcs=block(10.0),
        fzcc=block(8.0),
        fzss=block(11.0),
        flcc=block(9.0),
        flss=block(12.0),
    )


def test_m1_constraint_rotation_and_zforce_zeroing_are_local_to_m1_z_blocks():
    frzl = _constant_tomnsps()
    constrained = vmec_apply_m1_constraints(frzl=frzl, lconm1=True)
    osqrt2 = 1.0 / np.sqrt(2.0)

    np.testing.assert_allclose(np.asarray(constrained.frss)[:, 1, :], (4.0 + 5.0) * osqrt2)
    np.testing.assert_allclose(np.asarray(constrained.fzcs)[:, 1, :], (4.0 - 5.0) * osqrt2)
    np.testing.assert_allclose(np.asarray(constrained.frsc)[:, 1, :], (7.0 + 8.0) * osqrt2)
    np.testing.assert_allclose(np.asarray(constrained.fzcc)[:, 1, :], (7.0 - 8.0) * osqrt2)

    np.testing.assert_allclose(np.asarray(constrained.frss)[:, [0, 2], :], 4.0)
    np.testing.assert_allclose(np.asarray(constrained.fzcs)[:, [0, 2], :], 5.0)
    np.testing.assert_allclose(np.asarray(constrained.fzsc), 2.0)

    zeroed = vmec_zero_m1_zforce(frzl=frzl, enabled=True)
    np.testing.assert_allclose(np.asarray(zeroed.fzcs)[:, 1, :], 0.0)
    np.testing.assert_allclose(np.asarray(zeroed.fzcc)[:, 1, :], 0.0)
    np.testing.assert_allclose(np.asarray(zeroed.fzsc), 2.0)
    np.testing.assert_allclose(np.asarray(zeroed.fzcs)[:, [0, 2], :], 5.0)


def test_scalxc_and_fsq_sums_apply_odd_m_scaling_and_edge_policy():
    s = np.array([0.0, 0.25, 1.0])
    scalxc = vmec_scalxc_from_s(s=s, mpol=3)
    np.testing.assert_allclose(np.asarray(scalxc), [[1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])

    frzl = _constant_tomnsps()
    scaled = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)
    np.testing.assert_allclose(np.asarray(scaled.frcc)[:, :, 0], np.asarray(scalxc) * 1.0)
    np.testing.assert_allclose(np.asarray(scaled.fzsc)[:, :, 0], np.asarray(scalxc) * 2.0)

    sums = vmec_fsq_sums_from_tomnsps(
        frzl=frzl,
        apply_m1_constraints=False,
        apply_scalxc=True,
        include_edge=False,
        s=s,
    )
    assert sums.gcr2 == (1.0**2 + 4.0**2 + 7.0**2 + 10.0**2) * 12.0
    assert sums.gcz2 == (2.0**2 + 5.0**2 + 8.0**2 + 11.0**2) * 12.0
    assert sums.gcl2 == (3.0**2 + 6.0**2 + 9.0**2 + 12.0**2) * 15.0

    gcx2_jax = vmec_gcx2_from_tomnsps(
        frzl=frzl,
        apply_m1_constraints=False,
        apply_scalxc=False,
        include_edge=False,
    )
    gcx2_np = vmec_gcx2_from_tomnsps_np(frzl=frzl, include_edge=False)
    np.testing.assert_allclose(np.asarray(gcx2_jax), np.asarray(gcx2_np))

    norms = VmecForceNorms(fnorm=2.0, fnormL=3.0, r1=5.0)
    dynamic_norms = VmecForceNormsDynamic(
        fnorm=2.0,
        fnormL=3.0,
        r1=5.0,
        r2=0.0,
        volume=0.0,
        wb=0.0,
        wp=0.0,
        vp=np.zeros(3),
    )
    fsq = vmec_fsq_from_tomnsps(
        frzl=frzl,
        norms=norms,
        apply_m1_constraints=False,
        apply_scalxc=False,
        include_edge=False,
    )
    fsq_dynamic = vmec_fsq_from_tomnsps_dynamic(
        frzl=frzl,
        norms=dynamic_norms,
        apply_m1_constraints=False,
        apply_scalxc=False,
        include_edge=False,
    )
    np.testing.assert_allclose([fsq.fsqr, fsq.fsqz, fsq.fsql], np.asarray([fsq_dynamic.fsqr, fsq_dynamic.fsqz, fsq_dynamic.fsql]))
