from __future__ import annotations

"""Pytest configuration.

Allows running tests directly from the repo without requiring an editable install.
"""


import sys
from pathlib import Path
from typing import Any

import pytest
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Keep the test suite fast: avoid JAX compilation in unit tests.
# Tests cover correctness on small arrays; compilation dominates runtime.
try:  # pragma: no cover
    import jax

    jax.config.update("jax_disable_jit", True)
except Exception:  # pragma: no cover
    pass


def require_slow() -> None:
    """Skip tests marked as slow unless RUN_SLOW=1 is set."""
    if os.environ.get("RUN_SLOW", "") != "1":
        pytest.skip("Set RUN_SLOW=1 to run slow gradient/implicit tests")


def circular_coil_dofs(
    *,
    radius: float,
    major_offset: float = 0.0,
    out_of_plane: float = 0.0,
    order: int = 1,
):
    """Return Fourier dofs for one circular direct-coil test fixture.

    The coefficient layout follows the ESSOS-compatible ``CoilFieldParams``
    convention: ``x`` gets the first cosine coefficient, ``y`` the first sine
    coefficient, and optional constant offsets move the coil center or add a
    vertical displacement for non-axisymmetric free-boundary tests.
    """

    from vmec_jax._compat import jnp

    if order < 1:
        raise ValueError("circular_coil_dofs requires order >= 1")
    dofs = jnp.zeros((1, 3, 2 * int(order) + 1), dtype=float)
    if major_offset != 0.0:
        dofs = dofs.at[0, 0, 0].set(float(major_offset))
    dofs = dofs.at[0, 0, 2].set(float(radius))
    dofs = dofs.at[0, 1, 1].set(float(radius))
    if out_of_plane != 0.0:
        dofs = dofs.at[0, 2, 0].set(float(out_of_plane))
    return dofs


def circular_coil_params(
    *,
    current: float,
    radius: float,
    n_segments: int,
    major_offset: float = 0.0,
    out_of_plane: float = 0.0,
    nfp: int = 1,
    stellsym: bool = False,
    current_scale: float = 1.0,
    regularization_epsilon: float = 0.0,
    chunk_size: int | None = None,
):
    """Build one circular Fourier coil as a differentiable test parameter set."""

    from vmec_jax._compat import jnp
    from vmec_jax.external_fields import CoilFieldParams

    return CoilFieldParams(
        base_curve_dofs=circular_coil_dofs(
            radius=radius,
            major_offset=major_offset,
            out_of_plane=out_of_plane,
        ),
        base_currents=jnp.asarray([current], dtype=float),
        n_segments=int(n_segments),
        nfp=int(nfp),
        stellsym=bool(stellsym),
        current_scale=float(current_scale),
        regularization_epsilon=float(regularization_epsilon),
        chunk_size=None if chunk_size is None else int(chunk_size),
    )


def off_axis_circular_coil_params(
    *,
    current: float = 2.1e5,
    radius: float = 0.22,
    major_offset: float = 1.65,
    out_of_plane: float = 0.08,
    n_segments: int = 96,
    nfp: int = 1,
    stellsym: bool = False,
):
    """Return an off-axis circular coil for mgrid/direct-provider parity tests."""

    return circular_coil_params(
        current=current,
        radius=radius,
        n_segments=n_segments,
        major_offset=major_offset,
        out_of_plane=out_of_plane,
        nfp=nfp,
        stellsym=stellsym,
    )


def tiny_direct_freeb_input(
    path: Path,
    *,
    lasym: bool = False,
    niter: int = 4,
    mpol: int = 4,
    ntheta: int = 8,
) -> Path:
    """Write a tiny finite-pressure direct-coil free-boundary VMEC input.

    The deck is intentionally small but physically active: pressure is nonzero,
    the vacuum field comes from the direct-coil provider, and the VMEC free-
    boundary cadence is active from the first iteration.  Tests use it for
    branch-local derivative and NESTOR replay gates without relying on large
    external mgrid files.
    """

    lasym_flag = "T" if bool(lasym) else "F"
    path.write_text(
        f"""
&INDATA
  LFREEB = T
  MGRID_FILE = 'DIRECT_COILS'
  EXTCUR = 1.0
  LASYM = {lasym_flag}
  NFP = 1
  MPOL = {int(mpol)}
  NTOR = 0
  NS = 7
  NZETA = 2
  NTHETA = {int(ntheta)}
  NS_ARRAY = 7
  FTOL_ARRAY = 1.0E-8
  NITER_ARRAY = {int(niter)}
  NITER = {int(niter)}
  FTOL = 1.0E-8
  NSTEP = 20
  NVACSKIP = 1
  GAMMA = 0.0
  PHIEDGE = 1.0
  CURTOR = 0.0
  SPRES_PED = 1.0
  NCURR = 0
  PRES_SCALE = 1.0E4
  AM = 1.0 -1.0
  AI = 0.4 0.0
  AC = 0.0
  RAXIS = 1.0
  ZAXIS = 0.0
  RBC(0,0) = 1.0  ZBS(0,0) = 0.0
  RBC(0,1) = 0.25 ZBS(0,1) = 0.25
  RBC(0,2) = 0.03 ZBS(0,2) = 0.00
/
""".lstrip()
    )
    return path


def direct_free_boundary_initial_guess(input_path: Path, params: Any):
    """Run only VMEC's initial free-boundary state for direct-coil tests."""

    from vmec_jax.driver import run_free_boundary

    return run_free_boundary(
        input_path,
        use_initial_guess=True,
        verbose=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )


def direct_free_boundary_solve(input_path: Path, params: Any):
    """Run a tiny direct-coil free-boundary solve in VMEC2000-parity mode."""

    from vmec_jax.driver import run_free_boundary

    return run_free_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="parity",
        multigrid_use_input_niter=True,
        verbose=False,
        jit_forces=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )


def direct_nestor_step(
    run: Any,
    params: Any,
    *,
    ivac: int = 1,
    ivacskip: int = 0,
    iter_idx: int = 1,
    runtime: Any = None,
):
    """Evaluate one direct-coil NESTOR/free-boundary sampling step."""

    from vmec_jax.free_boundary import nestor_external_only_step

    return nestor_external_only_step(
        state=run.state,
        static=run.static,
        ivac=int(ivac),
        ivacskip=int(ivacskip),
        iter_idx=int(iter_idx),
        runtime=runtime,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )



_ASSET_SENTINEL = _ROOT / "examples" / "data" / "wout_circular_tokamak_reference.nc"
_WOUT_FIXTURE_SENTINEL = _ROOT / "examples" / "data" / "wout_circular_tokamak.nc"

# Tests that exercise optional generated/reference WOUT fixtures.  The fixtures
# live in a release asset bundle, not in git, so default local test runs skip
# these modules until ``python tools/fetch_assets.py --bundle wout-fixtures`` is
# run.  CI fetches them before coverage runs.
_OPTIONAL_WOUT_FIXTURE_TEST_FILES = {
    "test_booz_input.py",
    "test_converged_wout_matrix_parity.py",
    "test_free_boundary_beta_response_validation.py",
    "test_physics_gate_wave13_coverage.py",
    "test_physics_parity_helper_gates.py",
    "test_qi_readme_cases.py",
    "test_qi_seed_suitability_audit.py",
    "test_qs_ess_render_smoke.py",
    "test_residue_getfsq_parity.py",
    "test_validation_gates_extra.py",
    "test_wout_beta_eqfor_bundled_parity.py",
    "test_wout_chipf_bundled_parity.py",
    "test_wout_contravariant_field_gate.py",
    "test_wout_fixture_inventory.py",
    "test_wout_family_converged_quantities.py",
    "test_wout_geometry_mercier_bundled_parity.py",
    "test_wout_lasym_bsubv_parity.py",
    "test_wout_physics_gates.py",
}


def _wout_fixtures_available() -> bool:
    return _WOUT_FIXTURE_SENTINEL.exists()


def _assets_available() -> bool:
    """Return True if full-test reference assets are available.

    Accepts either the downloaded reference NC (``wout_*_reference.nc``) OR the
    optional WOUT fixture bundle restored by ``tools/fetch_assets.py``.
    """
    return _ASSET_SENTINEL.exists() or _wout_fixtures_available()


def pytest_collection_modifyitems(config, items):
    has_assets = _assets_available()
    has_wout_fixtures = _wout_fixtures_available()
    run_full = os.environ.get("RUN_FULL", "") == "1"
    skip_py311_coverage_only = os.environ.get("VMEC_JAX_SKIP_PY311_COVERAGE_ONLY", "") == "1"
    if run_full and not has_assets:
        raise pytest.UsageError("RUN_FULL=1 but example assets are missing. Run tools/fetch_assets.py")
    for item in items:
        if skip_py311_coverage_only and (
            item.get_closest_marker("py311_coverage_only") is not None
            or item.get_closest_marker("py311_slow_coverage") is not None
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason="Skipped in compatibility-only CI lanes; covered by the Python 3.11 coverage lane."
                )
            )
        if not has_wout_fixtures and Path(str(item.fspath)).name in _OPTIONAL_WOUT_FIXTURE_TEST_FILES:
            item.add_marker(
                pytest.mark.skip(
                    reason="Optional WOUT fixtures are missing. Run tools/fetch_assets.py --bundle wout-fixtures."
                )
            )
        if item.get_closest_marker("full") is not None:
            if not run_full:
                item.add_marker(pytest.mark.skip(reason="Full tests disabled. Set RUN_FULL=1."))
            elif not has_assets:
                item.add_marker(pytest.mark.skip(reason="Missing example assets. Run tools/fetch_assets.py"))

@pytest.fixture(scope="session")
def load_case_qa_reactorscale_lowres():
    """Load the bundled QA reactor-scale low-res input used in examples."""
    from vmec_jax._compat import has_jax, enable_x64
    from vmec_jax.config import load_config
    from vmec_jax.static import build_static
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.init_guess import initial_guess_from_boundary

    if has_jax():
        enable_x64(True)

    inpath = _ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_reactorScale_lowres"
    cfg, indata = load_config(str(inpath))
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)
    return cfg, indata, static, bdy, st0


@pytest.fixture(scope="session")
def load_case_circular_tokamak():
    """Load the bundled circular tokamak (axisymmetric, lasym=False) input used in examples."""
    from vmec_jax._compat import has_jax, enable_x64
    from vmec_jax.config import load_config
    from vmec_jax.static import build_static
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.init_guess import initial_guess_from_boundary

    if has_jax():
        enable_x64(True)

    inpath = _ROOT / "examples" / "data" / "input.circular_tokamak"
    cfg, indata = load_config(str(inpath))
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)
    return cfg, indata, static, bdy, st0


@pytest.fixture(scope="session")
def load_case_qh_warm_start():
    """Load the bundled QH warm-start input used for the fixed-boundary benchmark."""
    from vmec_jax._compat import has_jax, enable_x64
    from vmec_jax.config import load_config
    from vmec_jax.static import build_static
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.init_guess import initial_guess_from_boundary

    if has_jax():
        enable_x64(True)

    inpath = _ROOT / "examples" / "data" / "input.nfp4_QH_warm_start"
    cfg, indata = load_config(str(inpath))
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)
    return cfg, indata, static, bdy, st0
