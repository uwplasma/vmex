from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.diagnostics.parity.vmec2000_exec_stage_trace_compare import _parse_vmec2000_threed1


def test_parse_bundled_threed1_trace_compares_physical_and_preconditioned_fsq() -> None:
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "vmec2000_threed1_short_trace.txt"
    stages = _parse_vmec2000_threed1(fixture)

    assert [stage.ns for stage in stages] == [13, 25]
    assert [stage.niter for stage in stages] == [2, 3]
    np.testing.assert_allclose([stage.ftolv for stage in stages], [1e-10, 1e-12])
    assert [len(stage.rows) for stage in stages] == [2, 2]

    rows = [row for stage in stages for row in stage.rows]
    fsq_physical = np.asarray([row.fsqr + row.fsqz + row.fsql for row in rows])
    fsq_preconditioned = np.asarray([row.fsqr1 + row.fsqz1 + row.fsql1 for row in rows])

    np.testing.assert_allclose(fsq_physical, [3.534e-2, 3.03e-3, 1.77e-3, 1.14e-3])
    np.testing.assert_allclose(fsq_preconditioned, [1.0167e-1, 9.06e-3, 1.104e-2, 5.01e-3])
    assert rows[0].r00 == 1.234
    assert rows[0].w == 7.89
    assert rows[2].r00 == 1.236
    assert rows[2].w == 7.7
    assert np.all(np.isfinite(fsq_physical))
    assert fsq_physical[1] < fsq_physical[0]
    assert fsq_physical[3] < fsq_physical[2]
