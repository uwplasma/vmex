from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.solvers.fixed_boundary.residual.setup import (
    build_residual_cache_keys,
    grid_matches_vmec_static_grid,
    resolve_free_boundary_setup_policy,
)


def _cfg(**updates):
    values = {
        "lfreeb": False,
        "nvacskip": 0,
        "nfp": 2,
        "ntor": 3,
        "mpol": 4,
        "ns": 8,
        "lasym": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_grid_matches_vmec_static_grid_requires_same_coordinates() -> None:
    grid = SimpleNamespace(nfp=2, theta=np.asarray([0.0, 0.5]), zeta=np.asarray([0.0, 0.25]))
    same = SimpleNamespace(nfp=2, theta=np.asarray([0.0, 0.5]), zeta=np.asarray([0.0, 0.25]))
    changed_nfp = SimpleNamespace(nfp=3, theta=same.theta, zeta=same.zeta)
    changed_theta = SimpleNamespace(nfp=2, theta=np.asarray([0.0, 0.6]), zeta=same.zeta)
    broken = SimpleNamespace(nfp=2, theta=object(), zeta=same.zeta)

    assert grid_matches_vmec_static_grid(grid, same)
    assert not grid_matches_vmec_static_grid(grid, changed_nfp)
    assert not grid_matches_vmec_static_grid(grid, changed_theta)
    assert not grid_matches_vmec_static_grid(broken, same)


def test_build_residual_cache_keys_delegates_hash_and_edge_signatures() -> None:
    static = SimpleNamespace(
        cfg=SimpleNamespace(mpol=4, ntor=2, ntheta=8, nzeta=6, nfp=3, ns=5, lasym=False),
        modes=SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, 1])),
        grid=SimpleNamespace(theta=np.asarray([0.0, 0.5]), zeta=np.asarray([0.0, 0.25])),
    )
    wout_like = SimpleNamespace(
        nfp=3,
        mpol=4,
        ntor=2,
        lasym=False,
        signgs=-1,
        phipf=np.asarray([1.0, 2.0]),
        phips=np.asarray([0.0, 1.0]),
        chipf=np.asarray([0.0, 0.5]),
        pres=np.asarray([0.0, 0.1]),
        icurv=np.asarray([0.0, 0.2]),
    )

    def fake_hash(value):
        return ("hash", tuple(np.asarray(value).shape), float(np.asarray(value).sum()))

    keys = build_residual_cache_keys(
        static=static,
        wout_like=wout_like,
        edge_Rcos=np.asarray([1.0]),
        edge_Rsin=np.asarray([2.0]),
        edge_Zcos=np.asarray([3.0]),
        edge_Zsin=np.asarray([4.0]),
        constraint_tcon0=1.25,
        hash_array_bytes_func=fake_hash,
        edge_signature_key_func=lambda *arrays: ("sig", len(arrays)),
        edge_value_key_func=lambda *arrays: ("val", sum(float(np.asarray(a).sum()) for a in arrays)),
    )

    assert keys.static_key[:7] == (4, 2, 8, 6, 3, 5, False)
    assert keys.wout_key[:5] == (3, 4, 2, False, -1)
    assert keys.wout_key[-1] == 1.25
    assert keys.edge_signature_key == ("sig", 4)
    assert keys.edge_value_key == ("val", 10.0)


def test_free_boundary_setup_policy_disables_scan_and_resolves_direct_provider() -> None:
    policy = resolve_free_boundary_setup_policy(
        _cfg(lfreeb=True, nvacskip=4),
        external_field_provider_kind="Coils",
        use_scan=True,
        freeb_couple_env="1",
        freeb_sample_env="yes",
        jit_strict_update_env="off",
        backend_name="cpu",
        host_update_assembly=True,
        cpu_work_limit_env="1000",
    )

    assert policy.free_boundary_enabled
    assert policy.free_boundary_provider_kind == "coils"
    assert policy.direct_free_boundary_provider
    assert policy.freeb_nvacskip == 4
    assert policy.freeb_nvskip0 == 4
    assert policy.freeb_couple_edge
    assert not policy.use_scan
    assert policy.freeb_sample_external
    assert not policy.jit_strict_update_enabled


def test_free_boundary_setup_policy_auto_strict_update_matches_cpu_gpu_defaults() -> None:
    small_cpu_host = resolve_free_boundary_setup_policy(
        _cfg(lfreeb=False, ns=8, mpol=4, ntor=3),
        external_field_provider_kind=None,
        use_scan=False,
        freeb_couple_env="0",
        freeb_sample_env="0",
        jit_strict_update_env="auto",
        backend_name="cpu",
        host_update_assembly=True,
        cpu_work_limit_env="100",
    )
    assert not small_cpu_host.free_boundary_enabled
    assert small_cpu_host.free_boundary_provider_kind == ""
    assert not small_cpu_host.direct_free_boundary_provider
    assert small_cpu_host.freeb_nvacskip == 1
    assert not small_cpu_host.freeb_couple_edge
    assert not small_cpu_host.freeb_sample_external
    assert not small_cpu_host.jit_strict_update_enabled
    assert small_cpu_host.update_work == 8 * 4 * 4

    large_cpu_device = resolve_free_boundary_setup_policy(
        _cfg(lfreeb=False, ns=20, mpol=6, ntor=5, lasym=True),
        external_field_provider_kind="mgrid",
        use_scan=False,
        freeb_couple_env="1",
        freeb_sample_env="1",
        jit_strict_update_env="auto",
        backend_name="cpu",
        host_update_assembly=False,
        cpu_work_limit_env="100",
    )
    assert large_cpu_device.update_work == 20 * 6 * 11
    assert large_cpu_device.jit_strict_update_enabled

    gpu = resolve_free_boundary_setup_policy(
        _cfg(lfreeb=False, ns=1, mpol=1, ntor=0),
        external_field_provider_kind="direct_coils",
        use_scan=False,
        freeb_couple_env="1",
        freeb_sample_env="1",
        jit_strict_update_env="auto",
        backend_name="gpu",
        host_update_assembly=True,
        cpu_work_limit_env="not-an-int",
    )
    assert gpu.jit_strict_update_enabled
    assert gpu.cpu_work_limit == 1000
