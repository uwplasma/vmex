from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.driver import run_fixed_boundary
from vmec_jax.vmec2000_exec import find_vmec2000_exec, flatten_threed1, run_xvmec2000
from vmec_jax.wout import read_wout, wout_minimal_from_fixed_boundary


@pytest.mark.vmec2000
def test_vmec2000_qa_signgs1_trace_and_wout_parity(tmp_path):
    if os.environ.get("VMEC2000_INTEGRATION", "0") != "1":
        pytest.skip("Set VMEC2000_INTEGRATION=1 to run VMEC2000 integration parity tests")

    exe = find_vmec2000_exec()
    if exe is None:
        pytest.skip("xvmec2000 executable not found")

    input_path = Path("/Users/rogeriojorge/local/test/input.qa_signgs1")
    if not input_path.exists():
        pytest.skip("QA input file not found")

    # VMEC2000 run (for threed1 + wout reference) with a capped iteration budget
    # to keep parity tests fast and aligned with vmec_jax's max_iter.
    niter_stage1 = 1
    niter_stage2 = 9
    niter_total = niter_stage1 + niter_stage2
    vmec = run_xvmec2000(
        input_path,
        exec_path=exe,
        workdir=tmp_path / "vmec2000",
        timeout_s=60.0,
        indata_updates={
            "NITER": str(niter_total),
            "NITER_ARRAY": f"{niter_stage1} {niter_stage2}",
        },
        keep_workdir=True,
    )

    case = input_path.name.split("input.", 1)[-1]
    wout_path = vmec.workdir / f"wout_{case}.nc"
    if not wout_path.exists():
        pytest.skip("VMEC2000 wout not produced")
    wref = read_wout(wout_path)

    # vmec_jax run with the same iteration budget (use input NITER/NS staging).
    res = run_fixed_boundary(
        str(input_path),
        solver="vmec2000_iter",
        max_iter=niter_total,
        multigrid_use_input_niter=True,
        verbose=False,
        jit_forces=False,
    )

    # Per-iteration trace parity (fsqr/fsqz/fsql).
    fsqr_jax = np.asarray(res.result.fsqr2_history, dtype=float)
    fsqz_jax = np.asarray(res.result.fsqz2_history, dtype=float)
    fsql_jax = np.asarray(res.result.fsql2_history, dtype=float)

    diag = res.result.diagnostics
    stage_offsets = np.asarray(diag.get("multigrid_stage_offsets", []), dtype=int)
    stage_ns = np.asarray(diag.get("multigrid_ns_stages", []), dtype=int)
    assert stage_offsets.size == len(vmec.stages)
    assert stage_ns.size == len(vmec.stages)

    for stage_idx, stage in enumerate(vmec.stages):
        assert int(stage.ns) == int(stage_ns[stage_idx])
        offset = int(stage_offsets[stage_idx])
        for row in stage.rows:
            j = offset + int(row.it) - 1
            assert 0 <= j < fsqr_jax.size
            np.testing.assert_allclose(fsqr_jax[j], row.fsqr, rtol=3e-3, atol=1e-10)
            np.testing.assert_allclose(fsqz_jax[j], row.fsqz, rtol=2e-3, atol=1e-10)
            np.testing.assert_allclose(fsql_jax[j], row.fsql, rtol=2e-3, atol=1e-10)

    # End-state wout parity against VMEC2000.
    wnew = wout_minimal_from_fixed_boundary(
        path=tmp_path / "wout_qa_signgs1_vmec_jax.nc",
        state=res.state,
        static=res.static,
        indata=res.indata,
        signgs=int(res.signgs),
        fsqr=float(fsqr_jax[-1]),
        fsqz=float(fsqz_jax[-1]),
        fsql=float(fsql_jax[-1]),
    )

    def _rel_rms(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        num = float(np.sqrt(np.mean((a - b) ** 2)))
        den = float(np.sqrt(np.mean(b**2)))
        return num / den if den != 0.0 else float("inf")

    # Current parity envelope (tighten as parity improves).
    assert _rel_rms(wnew.rmnc, wref.rmnc) < 1e-1
    assert _rel_rms(wnew.zmns, wref.zmns) < 3.5e-1
    assert _rel_rms(wnew.lmns, wref.lmns) < 1e-4

    fsq_vmec = float(wref.fsqr + wref.fsqz + wref.fsql)
    fsq_jax = float(wnew.fsqr + wnew.fsqz + wnew.fsql)
    denom = max(abs(fsq_vmec), 1e-20)
    assert abs(fsq_jax - fsq_vmec) / denom < 1e-6
