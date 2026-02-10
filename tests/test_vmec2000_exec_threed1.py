from __future__ import annotations

from pathlib import Path

import numpy as np

from vmec_jax.vmec2000_exec import flatten_threed1, threed1_fsq_total


def test_parse_threed1_trace_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    threed1_path = repo_root / "threed1.circular_tokamak"
    if not threed1_path.exists():
        return

    from vmec_jax.vmec2000_exec import _parse_vmec2000_threed1  # local import for test visibility

    import vmec_jax.vmec2000_exec as vx

    text = threed1_path.read_text()
    if not any(vx._RE_STAGE.match(line) for line in text.splitlines()):
        return
    stages = _parse_vmec2000_threed1(threed1_path)
    assert stages, "expected at least one stage in threed1"
    rows = flatten_threed1(stages)
    assert rows, "expected at least one iteration row"
    fsq = threed1_fsq_total(rows)
    assert fsq.size == len(rows)
    assert np.all(np.isfinite(fsq[:5]))
