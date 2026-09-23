Contributing
============

Workflow
--------

1. Add or extend a kernel with a focused API in the matching
   :mod:`vmex.core` module; the module header docstring names the
   VMEC2000 counterpart file it ports — keep that cross-reference current.
2. Add a regression test under ``tests/`` (fast, offline where possible);
   parity-sensitive changes need a check against the golden VMEC2000
   fixtures. Add its module to ``tests/manifest.json``; the collection gate
   requires one owner, primary class, expected duration, device, asset bundle,
   oracle, and CI lane.
3. Keep JAX gotchas in mind:

   - jitted functions should only take arrays / pytrees (static config stays
     hashable and out of traced signatures),
   - solver functions are pure ``state -> state`` — no hidden host state,
   - float64 is mandatory (enforced at solver import).

Development install and checks::

  pip install -e .[dev]
  python tools/preflight.py --static   # lint, types, docs prose, guard tests
  python tools/preflight.py            # also the suites the diff affects
  python tools/preflight.py --docs     # also a warning-free Sphinx build
  python tools/test_manifest.py check

The workflows obtain their selectors from the manifest:

- ``CI`` is the stable pull-request gate. It runs fast API checks and
  representative fixed-boundary, free-boundary/NESTOR, mirror, device, and AD
  paths. Changed executable lines must be at least 95% covered.
- ``Nightly`` runs optimization and optional integrations, plus the retained
  nonlinear polish, homotopy, file-directive and free-boundary adjoint checks.
  A full-suite ownership entry alone does not schedule a test: its selector
  must also appear in a workflow.
- ``Weekly high resolution`` runs the selected high-resolution campaigns and
  the ``full``-marked tests of every ``full-*`` lane that no other workflow runs.
- ``Docs linkcheck`` audits external links weekly and on documentation pull
  requests; ``Publish to PyPI`` runs on ``v*`` release tags only.
- Every lane a manifest record names must be selected by one of these
  workflows or declared under ``local_only`` in ``tests/manifest.json`` with
  the reason and the command (``tests/test_test_manifest.py`` enforces it).
  ``gpu-smoke`` is the one local-only lane; see `GPU checks`_.

Use ``pytest --vmex-report=report.json`` to record the 50 slowest tests and all
skip reasons with the same metadata.

JAX compatibility and test tiers
--------------------------------

The stationarity contract is checked on Python 3.12 with JAX/JAXlib 0.9.2
and 0.11.1, using identical float64 tolerances. These are tested numerical
versions for the core install; optional integrations can require newer JAX.
The broader package dependency bounds do not certify every intermediate release. The PR matrix exercises eager and compiled linear
certificates; Nightly runs the tight real-MHD derivative and rejected-root
cases on both versions. Update the pair deliberately after reproducing failures,
with environment and attained residuals recorded in ``plan.md``.

PR physics and parity jobs have a 25-minute timeout. Actual successful job
times establish the runtime gate. ``full`` marks retain expensive checks for
scheduled runs; they do not weaken assertions. Run either tier locally::

  VMEX_COMPILATION_CACHE=disabled pytest -q -m "not full and not weekly" tests/test_polish_linear.py
  RUN_FULL=1 VMEX_COMPILATION_CACHE=disabled pytest -q tests/test_polish_linear.py

Use the restoring ``_module_jit_enabled`` fixture when a module needs compiled
solves; an unscoped ``jax.config.update`` can change later tests. Directive
parsing and precedence use a mocked driver; one real unpolished solve/export
check remains in PR CI and the polished file workflow runs in Nightly.

Reference assets
----------------

Keep input decks and analytic fixtures in git. Put generated WOUT, mgrid, and
benchmark archives in a GitHub Release and add their URL, byte size, SHA-256,
source, license, generator revision, and installed paths to
``assets/manifest.json``. Fetch repository fixtures with
``python tools/fetch_assets.py`` and VMEC2000 goldens with
``python tools/fetch_assets.py --bundle golden-v1``. CI rejects any tracked
file larger than 1 MiB.

**Pull-request lanes must not depend on a released bundle.** A release can be
deleted, and when ``assets-20260316-nc`` was, every pull request went red on
the download step while the lane it fed lost exactly one test. PR selectors run
on git-tracked decks, generated fixtures, and ``tests/golden_digests.json``;
only the nightly and weekly lanes fetch ``reference-nc``. Prefer a generated
fixture to a released one whenever the assertion is about solver behaviour
rather than a specific machine: ``tests/test_lasym_free_case.py`` (a converged
free-boundary case as 90 Chebyshev coefficients) and
``tests/test_qi_free_boundary_case.py`` (analytic modular coils tabulated with
``tabulate_cartesian_field``) are the two that PR lanes build on, and modules
that need neither declare ``"asset": "generated"`` in ``tests/manifest.json``.

Cut a new bundle with ``python tools/pack_reference_assets.py``, which packs
the git-ignored netCDF files under each bundle's roots into a byte-reproducible
tarball and prints the manifest fields. It applies two slimming rules, both
measured lossless: MAKEGRID's ``ar_``/``ap_``/``az_`` vector potential is
dropped (neither VMEX nor ``xvmec2000`` reads it back — half of
``mgrid_cth_like.nc``), and duplicate ``single_grid/`` copies are re-created on
extraction from the manifest's ``mirrors`` rule instead of shipped. Mirroring
is also what keeps the tracked ``mgrid_cth_like_lasym_small.nc`` and its
``single_grid/`` copy in step.

Documentation builds must pass strict mode::

  python -m sphinx -W -j auto -b html docs docs/_build/html

An installed VMEC2000 can be exercised live, outside ordinary offline CI::

  pytest -q tests/test_vmec2000_live.py --run-vmec2000 \
    --vmec2000-executable /path/to/xvmec2000

The test uses isolated output directories and compares WOUTs produced during
that invocation. It covers finite-beta current/Mercier profiles and a
converged, asymmetrically forced LASYM free-boundary case including NESTOR
potential and surface-field tables. Omitting ``--run-vmec2000`` skips it.

GPU checks
----------

No CI runner has a GPU. The former ``Trusted GPU physics`` workflow ran on a
self-hosted runner that was registered by hand for each use; none is
registered now, it last passed on 2026-07-31, and it was retired so that no
lane looks scheduled that cannot run. The GPU tests are the local-only
``gpu-smoke`` lane in ``tests/manifest.json``. Run them on a machine with an
NVIDIA driver 580 or newer, without ``JAX_PLATFORMS`` or
``JAX_PLATFORM_NAME`` set::

  pip install -U "jax[cuda13]"
  pip install -e ".[coils,turbulence,freeb]" pytest
  pytest -q $(python tools/test_manifest.py select gpu-smoke)
  python benchmarks/device_parity.py --quick --devices cpu,gpu --metrics quasisymmetry

With a GPU present, ``tests/test_gpu_ci.py`` fails if JAX does not select it
by ordinary discovery or a platform pin is set; without one its GPU tests
skip, so check the skip count (``--vmex-report``) before reading a local run
as a GPU result. The parity audit accepts the metrics ``mhd_energy``,
``magnetic_well``, ``dmerc_interior_mean``, ``jdotb_interior_mean``,
``glasser_d_r_interior_mean``, ``quasisymmetry`` and ``quasi_isodynamic``,
and ``--gpu-id 1`` repeats it on a second GPU. Timing in its JSON output is
informational. The CPU halves of these tests, and the two-device placement
checks on forced host devices, still run in the ``CI`` workflow.

Releasing
---------

Releases are cut from ``main``:

#. Bump ``version`` in ``pyproject.toml`` (semantic versioning).
#. Tag the commit (``vX.Y.Z``) and publish a GitHub Release. The
   ``publish-pypi`` workflow validates that the tag matches the project version
   and uploads the wheel and sdist to PyPI.

Release notes live in ``CHANGELOG.md`` and the GitHub releases.

Support and conduct
-------------------

Bug reports, feature requests, and usage questions all go to `GitHub issues
<https://github.com/uwplasma/vmex/issues>`_, which offers a form for each;
a bug report needs the input deck and the output of ``vmex --doctor``.
:doc:`/howto/troubleshoot` covers non-convergence, NaNs, device placement,
and cache problems first. ``CONTRIBUTING.md`` at the repository root is the
short version of this page; participation in the project is governed by
``CODE_OF_CONDUCT.md``, the Contributor Covenant 2.1.
