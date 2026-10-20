# Ecosystem

VMEX is one of several uwplasma JAX codes. Each capability has one owner;
the others import it.

## Who owns what

| Repository | Owns | VMEX uses it for |
| --- | --- | --- |
| VMEX | Equilibrium (fixed and free boundary), WOUT schema and writer, mgrid I/O, AD state-to-Boozer tables, field-line geometry (PEST), ballooning, bounce integrals, Gamma_c, maximum-J, QI residuals, bootstrap geometry | — |
| [SOLVAX](https://github.com/uwplasma/SOLVAX) | Generic numerics: tridiagonal and block-tridiagonal solves, Krylov (GMRES, GCROT, PCG), Newton-Krylov, Gauss-Newton/LM, continuation, chunked Jacobians | the radial preconditioner, implicit adjoints, polishing |
| [booz_xform_jax](https://github.com/uwplasma/booz_xform_jax) | Boozer transform, `boozmn` reader and writer, light WOUT reader for codes without VMEX | Boozer spectra (core dependency) |
| [NEO_JAX](https://github.com/uwplasma/NEO_JAX) | Effective ripple `epsilon_eff` | `vmex[neoclassical]` |
| [DKX](https://github.com/uwplasma/DKX) | Drift-kinetic neoclassical transport and bootstrap current | `vmex[neoclassical]` |
| [GKX](https://github.com/uwplasma/GKX) | Gyrokinetic flux-tube solver, GX-convention Boozer geometry | `vmex[turbulence]` |
| [virtual_casing_jax](https://github.com/uwplasma/virtual_casing_jax) | Virtual-casing exterior fields and their derivative kernels | `vmex[freeb]` |
| [ESSOS](https://github.com/uwplasma/ESSOS) | Coils, Biot-Savart, particle tracing | `vmex[coils]` |

Downstream codes (NEOPAX, NTX, DRBX, balloon_jax, FAX) import VMEX for
equilibria, WOUT files and Boozer tables. The PyPI name `vmec-jax` and the
import name `vmec_jax` are retired; use `vmex`.

## Rules

- Generic linear algebra and nonlinear solvers go to SOLVAX, not into a
  physics code. A physics code keeps a solver only when SOLVAX has no
  equivalent (VMEX: the constrained KKT polishing step and the bounded
  Newton-Krylov of the mirror solver).
- Read WOUT with `vmex.read_wout` and `boozmn` with booz_xform_jax.
- Each dependency floor in `pyproject.toml` is the first release with what
  VMEX calls; the comment next to it says why.

## JAX floor

The core accepts `jax>=0.9.2` and `vmex[turbulence]` (GKX) needs
`jax>=0.10.1`. The core floor stays at 0.9.2 because DESC (`desc-opt` 0.17)
caps `jax<0.10` and VMEX's DESC interface is tested against it. No floor
caps JAX from above, so a fresh environment resolves the newest JAX
(0.11 on Python 3.12+, 0.10 on 3.11). See {doc}`../installation`.

## Duplication still to remove

Ranked by value over risk.

1. Dead `vmec_jax` code paths: NEOPAX fallback imports and its `vmec_jax.api`
   shim, `virtual_casing_jax.vmec_jax_bridge`, and examples in NEO_JAX and
   booz_xform_jax. No numerical change.
2. Local Thomas, block-Thomas, GMRES and damped Gauss-Newton copies in DRBX,
   NEOPAX, DKX and GKX, replaced by SOLVAX. Needs parity tests.
3. Five `boozmn` readers (NEO_JAX, NTX, NEOPAX, balloon_jax) replaced by one
   stateless booz_xform_jax reader. Needs a booz_xform_jax release.
4. A public VMEX "state to Boozer spectrum" call to replace the hand-written
   `boozer_input_tables` glue in NTX, GKX, NEO_JAX, DKX and NEOPAX. Needs a
   VMEX release, then downstream bumps.
5. A batched symmetric tridiagonal eigenvalue solver in SOLVAX for the
   ballooning operator, shared by VMEX and balloon_jax (O(n) memory instead
   of a dense `eigvalsh`). Changes numerics; must match the dense result.

Kept apart on purpose: GKX and VMEX field-line geometry use different
coordinates (Boozer/GX against PEST); DKX keeps its own WOUT reader and
trapped fraction because it does not depend on VMEX.
