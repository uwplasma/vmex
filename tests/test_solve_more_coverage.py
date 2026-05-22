import numpy as np

from vmec_jax import solve


def test_axis_reset_dump_returns_false_when_filesystem_write_fails(monkeypatch, tmp_path):
    def fail_mkdir(self, *args, **kwargs):
        raise OSError("synthetic mkdir failure")

    monkeypatch.setattr(solve.Path, "mkdir", fail_mkdir)

    assert not solve._write_axis_reset_dump(
        axis_dump_dir=tmp_path / "axis",
        ns=3,
        ntor=1,
        used_state_guess=True,
        raxis_cc=np.asarray([1.0, 0.1]),
        raxis_cs=np.asarray([0.0, 0.0]),
        zaxis_cc=np.asarray([0.0, 0.0]),
        zaxis_cs=np.asarray([0.0, 0.2]),
    )


def test_scan_chunk_settings_wrapper_uses_runtime_backend_and_env(monkeypatch):
    monkeypatch.setattr(solve, "_scan_backend_name", lambda: "gpu")
    monkeypatch.setenv("VMEC_JAX_SCAN_CHUNK_SIZE", "7")

    assert solve._scan_chunk_settings(
        max_iter_scan=40,
        nstep_screen=5,
        need_print=False,
        lthreed=True,
    ) == (7, True)


def test_axis_m0_mask_falls_back_to_modes_when_precomputed_mask_missing():
    static = type("Static", (), {"modes": type("Modes", (), {"m": np.asarray([0, 1, 0])})()})()

    np.testing.assert_allclose(np.asarray(solve._axis_m0_mask(static, dtype=np.float64)), [1.0, 0.0, 1.0])
