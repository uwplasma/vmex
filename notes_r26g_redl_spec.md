# R26g SPEC — Differentiable Redl (2021) bootstrap current + self-consistent-current optimization

Status: **specification only** (no code yet).  Target: plan.md R26.g — "QA (nfp 2) + QH (nfp 4)
optimization with SELF-CONSISTENT BOOTSTRAP CURRENT, reproducing arXiv:2205.02914
(Landreman–Buller–Drevlak) against the Zenodo data", requiring the **Redl (2021) bootstrap formula
in vmec_jax, DIFFERENTIABLE**, plus a loop iterating the current profile to self-consistency.

References
----------
- Landreman, Buller, Drevlak, *Optimization of quasisymmetric stellarators with self-consistent
  bootstrap current and energetic particle confinement*, Phys. Plasmas 29, 082501 (2022),
  arXiv:2205.02914.
- Redl, Angioni, Belli, Sauter, *A new set of analytical formulae for the computation of the
  bootstrap current and the neoclassical conductivity in tokamaks*, Phys. Plasmas 28, 022502 (2021).
- Sauter, Angioni, Lin-Liu, Phys. Plasmas 6, 2834 (1999) — collisionality/lnΛ definitions
  (eqs. 18b–18e) reused by Redl.
- Ground-truth implementation: simsopt `src/simsopt/mhd/bootstrap.py`
  (`compute_trapped_fraction`, `j_dot_B_Redl`, `RedlGeomVmec`, `RedlGeomBoozer`,
  `VmecRedlBootstrapMismatch`) — this exact code produced the paper's results.  A copy was reviewed
  on 2026-07-11 (master); equations transcribed below.
- Local validation data:
  `/Users/rogerio/local/20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current`
  (see §2).

---

## 1. What the paper does (method to reproduce)

1. **Redl formula as a drop-in for the drift-kinetic solve.** All neoclassical physics in perfect
   quasisymmetry is isomorphic to axisymmetry (Boozer isomorphism): to evaluate the bootstrap
   current on a QS surface with `(B, iota, G, I)`, evaluate the *tokamak* (Redl) formula with
   `(B, iota − N, G + N·I, I)`, where `N = nfp·helicity_n` (`N = 0` QA; `N = ±nfp` QH).
   In the simsopt implementation the substitution is applied as `iota → iota − N` everywhere
   `iota` appears (collisionality geometry factor and the `1/(psi_edge·(iota − N))` prefactor);
   `G` is used un-shifted (see §4 note).
2. **Inputs beyond VMEC**: prescribed kinetic profiles (NOT part of a VMEC input)
   `n_e(s) = n0·(1 − s^5)`, `T_e(s) = T_i(s) = T0·(1 − s)`, `Zeff = 1` (hydrogen, n_i = n_e).
   The VMEC pressure deck is *derived* from them: `p(s) = n_e T_e + n_i T_i = 2 n0 T0 e (1−s)(1−s^5)`
   — e.g. the QA deck's `AM = 720979.4853·(1, −1, 0, 0, 0, −1, 1)` is exactly
   `2 × 2.38e20 × 9.45e3 × 1.602177e-19 ≈ 7.2098e5` Pa times `(1−s)(1−s^5)` expanded.
3. **Self-consistency by penalty, not fixed-point iteration**: the equilibrium current profile
   (VMEC `AC` = dI_toroidal/ds, plus `CURTOR`) is a **free optimization variable** alongside the
   boundary Fourier coefficients, and the objective adds the mismatch term

       f_boot = ∫ ds [⟨J·B⟩_vmec − ⟨J·B⟩_Redl]² / ∫ ds [⟨J·B⟩_vmec + ⟨J·B⟩_Redl]²

   (simsopt `VmecRedlBootstrapMismatch`; residual vector
   `R_j = (Jv(s_j) − Jr(s_j)) / sqrt(Σ_k (Jv(s_k) + Jr(s_k))²)` over the geometry surfaces).
   Full objective: `f = f_QS + f_boot + w_A(A−A*)² + w_a(a−a*)² + w_B(B̄−B̄*)² [+ ⟨iota⟩ term (QA)
   + iota<1.03 barrier (high-β QH)]`, ARIES-CS scale `a* = 1.70 m`, `B̄* = 5.86 T`.
4. **Optional post-hoc refinement**: a few Picard (fixed-point) iterations of
   `AC ← current profile implied by the kinetic ⟨J·B⟩` at *fixed boundary* (paper does this with
   SFINCS; Zenodo `calculations/20220102-01-simsoptSelfConsistentBootstrapProfile/
   convertSfincsToVmecCurrentProfile` documents the exact conversion — see §6.3).
5. **Configurations produced** (all in the Zenodo `configurations/`):

   | name | nfp | A | β | n0 [m⁻³] | T0 [keV] | I_p |
   |---|---|---|---|---|---|---|
   | QA_aspect6_beta2.5  | 2 | 6.0 | 2.5% | 2.38e20 | 9.45 | −2.72 MA (CURTOR of deck) |
   | QH_aspect6.5_beta0   | 4 | 6.5 | 0    | –       | –    | 0 |
   | QH_aspect6.5_beta2.5 | 4 | 6.5 | 2.5% | 2.2e20  | 10   | −1.208 MA (wout ctor) |
   | QH_aspect6.5_beta5   | 4 | 6.5 | 5%   | 3e20    | 15   | (iterated with SFINCS) |

---

## 2. Zenodo data inventory (validation assets)

Root: `/Users/rogerio/local/20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current`

`configurations/` — each has `input.*` (INDATA deck), `wout_*.nc`, `boozmn_*.nc`:
- `QA_aspect6_beta2.5/wout_QA_beta0p025_iota0p42_dreopt_HIGHERRES_2022-04-15.nc` — ns=201, mpol=16,
  ntor=12, `NCURR=1`, `PCURR_TYPE="cubic_spline_ip"` with 50 `AC_AUX_S/AC_AUX_F` knots,
  `CURTOR=−2721013.45`, `PHIEDGE=51.868`.
- `QH_aspect6.5_beta2.5/wout_20220218-01-021_QH_A6.5_n0_2.2_T0_10_highResVmecForBestFrom020.nc` —
  verified readable with `vmec_jax.core.wout.read_wout`: ns=201, nfp=4, aspect=6.50005, b0=5.808 T,
  ctor=−1,207,674.85 A, betatotal=2.465e-2, full-grid `jdotb` present.
- `QH_aspect6.5_beta0/` (two wouts, incl. `_aScaling`), `QH_aspect6.5_beta5/
  wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs.nc`.

`calculations/` — figure-generating scripts with **verbatim SFINCS benchmark arrays**:
- `figure01/20220329-01-comparingRedlVsSfincsForPreciseQS` (Redl-vs-SFINCS for precise QA/QH):
  profiles `ne = 4.13e20·(1−s^5)`, `Te = Ti = 12e3·(1−s)` eV, `Zeff=1`;
  `s_sfincs = linspace(0.025, 0.975, 39)`; two 39-point `jdotB_sfincs` arrays
  (QA: −2.1649e6 … peak ≈ −7.6458e6 @ s≈0.60 … −1.1505e5; QH helicity_n=−1: −1.0861e6 … −3.1598e5).
  The matching equilibria `wout_new_QA_aScaling.nc` / `wout_new_QH_aScaling.nc` are in
  `calculations/20211226-01-sfincs_for_precise_QS_for_Redl_benchmark/` (also shipped as simsopt
  test files).  **These are the primary formula-validation targets.**
- `figure04/20220403-03-compareMethodsOfComputingJDotBForOptimizedSelfConsistentQH_beta2p5`:
  optimized QH β=2.5% config; `ne = 2.2e20·(1−s^5)`, `Te = Ti = 10e3·(1−s)`;
  `s = linspace(0.02, 0.98, 49)`; 49-point high-res SFINCS array
  (−3.8955e5 … −1.4338e6 @ s≈0.7 … −1.3823e5).  Demonstrates the three-way match
  VMEC `jdotb` ≈ Redl ≈ SFINCS on the *converged self-consistent* configuration.
- `figure09/`, `figure10/` — same comparison for QH β=5% before/after the SFINCS fixed point.
- `figure16/` — bootsj-vs-SFINCS for the new QA (context only).
- `20220102-01-simsoptSelfConsistentBootstrapProfile/convertSfincsToVmecCurrentProfile` —
  the ⟨J·B⟩ ↔ (AC, CURTOR) conversion used for Picard iteration; gives the **exact MHD identity**
  we will also use for the traceable ⟨J·B⟩_vmec (§6.2):

      ⟨J·B⟩(s) = [⟨B²⟩(s)·dI/ds + μ0 I(s)·dp/ds] / (2π ψ_a),
      ψ_a = Φ_edge/(2π),  I(s) = (2π/μ0)·buco(s)   [A],

  verified in that script against VMEC's `jdotb` output.

Repo hygiene: these files stay where they are (repo ≤ 10 MB rule, plan R26.j).  Tests hardcode the
small SFINCS arrays; wout-reading validations live in `benchmarks/` and skip when the Zenodo path
is absent.

---

## 3. Exact equations to implement (transcribed from simsopt `bootstrap.py`)

All quantities per flux surface `s` (1-D arrays over the requested surface grid).  Units:
`ne` [m⁻³], `Te, Ti` [eV], `G, I, R` [T·m], `psi_edge` [Wb/rad = Wb/2π], output ⟨J·B⟩ [A·T/m²].
`N ≡ helicity_N = nfp·helicity_n` (0 for QA; simsopt convention `helicity_n = −1` for the
QH configs here, so `N = −4`).

**Ion quantities (Zeff profile, =1 here):** `ni = ne/Zeff`, `pe = ne·Te`, `pi = ni·Ti`.

**Coulomb logarithms (Sauter 18d–e):**

    ln_Lambda_e  = 31.3 − ln( sqrt(ne) / Te )
    ln_Lambda_ii = 30.0 − ln( Zeff^3 · sqrt(ni) / Ti^1.5 )

**Collisionalities (Sauter 18b–c with the isomorphism substitution):**

    gf    = | R / (iota − N) |
    nu_e* = gf · 6.921e-18 · ne · Zeff   · ln_Lambda_e  / (Te² · ε^1.5)
    nu_i* = gf · 4.90e-18  · ni · Zeff^4 · ln_Lambda_ii / (Ti² · ε^1.5)

**Redl eq (11), (10) — L31:**

    X31 = f_t / ( 1 + (0.67(1 − 0.7 f_t)√ν_e*)/(0.56 + 0.44 Zeff)
                    + (0.52 + 0.086√ν_e*)(1 + 0.87 f_t)ν_e* / (1 + 1.13√(Zeff − 1)) )
    Zfac = Zeff^1.2 − 0.71
    L31 = (1 + 0.15/Zfac)·X31 − (0.22/Zfac)·X31² + (0.01/Zfac)·X31³ + (0.06/Zfac)·X31⁴

**Redl eqs (14), (13), (16), (15), (12) — L32:**

    X32e  = f_t / ( 1 + 0.23(1 − 0.96 f_t)√ν_e*/√Zeff
                     + 0.13(1 − 0.38 f_t)(ν_e*/Zeff²)·( √(1 + 2√(Zeff−1))
                       + f_t²·√((0.075 + 0.25(Zeff−1)²)·ν_e*) ) )
    F32ee = (0.1 + 0.6 Zeff)(X32e − X32e⁴)/(Zeff(0.77 + 0.63(1 + (Zeff−1)^1.1)))
            + 0.7/(1 + 0.2 Zeff)·(X32e² − X32e⁴ − 1.2(X32e³ − X32e⁴))
            + 1.3/(1 + 0.5 Zeff)·X32e⁴
    X32ei = f_t / ( 1 + 0.87(1 + 0.39 f_t)√ν_e*/(1 + 2.95(Zeff−1)²)
                     + 1.53(1 − 0.37 f_t)ν_e*(2 + 0.375(Zeff−1)) )
    F32ei = −(0.4 + 1.93 Zeff)/(Zeff(0.8 + 0.6 Zeff))·(X32ei − X32ei⁴)
            + 5.5/(1.5 + 2 Zeff)·(X32ei² − X32ei⁴ − 0.8(X32ei³ − X32ei⁴))
            − 1.3/(1 + 0.5 Zeff)·X32ei⁴
    L32 = F32ee + F32ei

**Redl eqs (19)–(21):** `L34 = L31` and

    alpha0 = −(0.62 + 0.055(Zeff−1))(1 − f_t)
             / ( (0.53 + 0.17(Zeff−1))·(1 − (0.31 − 0.065(Zeff−1)) f_t − 0.25 f_t²) )
    alpha  = ( (alpha0 + 0.7 Zeff √(f_t ν_i*))/(1 + 0.18√ν_i*) − 0.002 ν_i*² f_t⁶ )
             / ( 1 + 0.004 ν_i*² f_t⁶ )

**Assembly (e = 1.602176634e-19 converts eV → J):**

    pref = −G·e / (psi_edge · (iota − N))
    ⟨J·B⟩_Redl = pref · [ (ne Te + ni Ti)·L31·(dln ne/ds)
                          + pe·(L31 + L32)·(dln Te/ds)
                          + pi·(L31 + alpha·L34)·(dln Ti/ds) ]

**Geometry quantities (simsopt `RedlGeomVmec` semantics):** on each requested surface,

    ε      : (Bmax/Bmin − 1)/(Bmax/Bmin + 1) = (Bmax − Bmin)/(Bmax + Bmin)
    ⟨B²⟩, ⟨1/B⟩ : flux-surface averages with weight |sqrt(g)|
    f_t    = 1 − (3/4)·⟨B²⟩·∫₀^{1/Bmax} λ dλ / ⟨√(1 − λB)⟩
    G, I   : Boozer covariant averages = wout `bvco`, `buco` (T·m)
    R      = (G + iota·I)·⟨1/B⟩       (effective major radius for collisionality)
    psi_edge = −phi(s=1)/(2π)          (wout sign convention)

Note on the isomorphism: the paper (§ eq. isomorphism) prescribes `G → G + N·I` as well; simsopt's
code applies only `iota → iota − N` and keeps `G` (and `R = (G + iota I)⟨1/B⟩` with the *unshifted*
iota).  **We replicate simsopt exactly** (it generated the paper's own figures); a
`strict_isomorphism: bool = False` knob can expose the textbook variant later.  Practical effect is
small (|N·I| ≪ |G| for these configs).

---

## 4. What vmec_jax already provides (precise names)

- **Objective protocol** (`vmec_jax/core/optimize.py`): terms are `(callable, target, weight)`;
  `_call_term` dispatches two-positional callables as traceable `(SpectralState, SolverRuntime)`
  functions, one-positional as `Equilibrium` (wout-engine, FD-only).  `jac="implicit"` requires a
  traceable callable or a `residuals_state(state, rt)` method (`_traceable_term`);
  `QuasisymmetryRatioResidual` is the template for a class exposing both lanes
  (`compute(wout)`/`J(eq)` + `_pointwise_state`/`residuals_state`).
- **Field chain** (`optimize._field_chain`): geometry → `geometry.half_mesh_jacobian` (→
  `jacobian.sqrt_g`, half mesh, internal (ns, ntheta2-reduced, nzeta) grid) →
  `fields.metric_elements` → `fields.magnetic_fields` (→ `bsupu/bsupv/bsubu/bsubv`,
  `total_pressure` = |B|²/2 + p, `pressure` (internal μ0·Pa), `vp`, `chips`) →
  `fields.energies_and_force_norms`.
  `|B|² = 2·(total_pressure − pressure[:, None, None])` — exactly the recipe of
  `QuasisymmetryRatioResidual._pointwise_state` (optimize.py:414) including the reduced-[0, π]
  theta grid mirroring to the full circle (`i_src/k_src` maps, optimize.py:399–411).
- **iota, G, I traceably**: `optimize._iotas_half` (ncurr=1-aware, from `fields.chips/phips`);
  `fields.surface_currents` → `CurrentDiagnostics.buco = ⟨B_u⟩`, `bvco = ⟨B_v⟩` (T·m, half mesh)
  — these ARE the Redl `I`, `G`.  `psi_edge` traceably as in optimize.py:440:
  `d_psi_d_s = −signgs·hs·Σ phipf[1:]` (equals `−phi[-1]/2π` of the wout).
- **Half-grid interpolation helpers**: `optimize._half_grid`, `optimize._interp_half_grid`
  (linear interp of half-mesh samples onto arbitrary `surfaces`).
- **wout lane**: `wout.WoutData` / `wout.read_wout` (verified on the Zenodo files) exposes
  `bmnc, gmnc, xm_nyq, xn_nyq, iotas, bvco, buco, phi, jdotb, pres, ns, nfp, signgs` — everything
  `RedlGeomVmec` uses.  `wout_from_state` gives the same tables for a converged core state.
- **VMEC ⟨J·B⟩ (parity reference, host NumPy)**: `nyquist.mercier_and_jxb` (jxbforce.f port,
  full-grid `jdotb` in A·T/m²) — used via `Equilibrium.wout.jdotb`.  NOT traceable; §6.2 adds a
  traceable equivalent.
- **Profiles**: `profiles.current` evaluates every VMEC `pcurr_type` (incl. `power_series` = I'
  power series and `cubic_spline_ip`) in pure jnp; `setup.flux_profiles` builds `icurv` with the
  `Itor = signgs·μ0·curtor/(2π·pcurr(1))` scaling, traced in `curtor`/`ac`.
- **Differentiable parameters** (`implicit.ImplicitParams`): already includes `ac` (dense
  coefficients) and `curtor`; `runtime_from_params` rebuilds `icurv` traceably from them.
  **Gap**: `ac_aux_f` (spline knot values, used by the paper's `cubic_spline_ip` decks) is static
  (`inp.ac_aux_f` in the shim, implicit.py:388) — spline-knot current dofs need an
  `ImplicitParams.ac_aux_f` field (§6.4).
- **Driver gap**: `least_squares` packs only boundary dofs (`pack_boundary`); the implicit path's
  `params_of(x)` (optimize.py:1426) sets only `rbc/zbs`.  §6.4 extends both.
- **Gradient test harness**: `tests/test_implicit_grad.py` (implicit-vs-FD pattern to copy).

Nothing bootstrap/trapped-fraction-like exists yet (`grep -rn "trapped\|Redl"` → only plan.md and
this note's targets).

---

## 5. Module layout

New module **`vmec_jax/core/bootstrap.py`** (pure jnp; no scipy) exporting:

    KineticProfiles                  # frozen pytree: ne/Te/Ti/Zeff polynomial coefficients
    profile_value_and_dds(coeffs, s) # Horner value + analytic d/ds (traceable)
    compute_trapped_fraction(modB, sqrtg, *, n_lambda=64)   # §6.1
    RedlGeometry                     # frozen pytree: surfaces, iota, G, I, R, epsilon, f_t,
                                     #   fsa_B2, fsa_1overB, Bmin, Bmax, psi_edge, nfp
    redl_geometry_from_wout(wout, surfaces, *, ntheta=64, nphi=65)     # parity lane (§6.1a)
    redl_geometry_from_state(state, rt, *, surfaces)                   # traceable lane (§6.1b)
    j_dot_B_redl(profiles, geom, helicity_n) -> (jdotB, details_dict)  # §3 equations
    vmec_j_dot_B(state, rt, *, surfaces)                               # §6.2 traceable ⟨J·B⟩_vmec
    RedlBootstrapMismatch            # objective class (§6.3): J(eq) + residuals_state(state, rt)
    self_consistent_bootstrap(inp, profiles, helicity_n, *, n_iter=6, ...)  # §6.5 Picard loop

Wiring elsewhere:
- `core/optimize.py`: re-export `RedlBootstrapMismatch`; extend `least_squares`/
  `_least_squares_implicit` with current-profile dofs (§6.4).
- `core/implicit.py`: add `ac_aux_f` to `ImplicitParams` (+ `params_from_input`,
  `input_with_params`, `runtime_from_params` shim) — optional if the power-series AC lane is used.
- `vmec_jax/__init__` exports; `tests/test_bootstrap.py`; examples
  `examples/optimization/QA_bootstrap_selfconsistent.py`, `QH_bootstrap_selfconsistent.py`;
  `benchmarks/benchmark_redl_zenodo.py` (skips without the Zenodo dir).

---

## 6. Design

### 6.1 Trapped fraction and geometry (differentiable rewrite of `compute_trapped_fraction`)

`compute_trapped_fraction(modB, sqrtg)` with `modB, sqrtg` shaped `(nsurf, ntheta, nzeta)`
(leading surface axis — note simsopt uses trailing; we standardize on leading to match the rest of
the core).  Per surface, with `w = |sqrtg|` (FSA weight; simsopt uses signed `sqrtg` from wout
which has uniform sign — we take `signgs·sqrtg` on the internal grid):

    Vp        = mean(w)                        # ∝ dV/ds; normalization cancels in FSAs
    fsa_B2    = mean(B² w)/Vp
    fsa_1overB= mean(w/B)/Vp
    Bmax, Bmin= max/min over the angular grid   (hard extrema; see AD note)
    epsilon   = (Bmax − Bmin)/(Bmax + Bmin)
    f_t       = 1 − 0.75·fsa_B2 · Σ_k ω_k · λ_k / mean(sqrt(max(1 − λ_k B, 0))·w)/Vp

λ-integral: **fixed-order Gauss–Legendre** (default `n_lambda = 64`) mapped to `[0, 1/Bmax]`
(DESC does the same; replaces simsopt's adaptive `scipy.integrate.quad`).  GL nodes exclude the
endpoint `λ = 1/Bmax`, where the integrand is finite but `sqrt(1 − λB)` hits 0 at the B-max grid
point; still guard with the codebase's **double-where pattern** so reverse-mode AD never sees
`d/dx sqrt(0)`:

    arg  = 1 − λ_k·B
    safe = jnp.where(arg > 0, arg, 1.0)
    root = jnp.where(arg > 0, jnp.sqrt(safe), 0.0)

Bmax/Bmin: plain `jnp.max/min` over a *dense* angular grid.  Differences vs simsopt's
spline-refined extrema are O(grid²) in ε and f_t; §7 quantifies against simsopt on the Zenodo
wouts and picks the default grid (64×65 like simsopt's synthesis grid; the traceable lane uses the
solver grid, so validation must confirm the solver resolutions used in the examples suffice —
otherwise synthesize on a finer grid from the state's spectral coefficients).  Hard max ⇒
piecewise-smooth gradients (subgradient at argmax ties): acceptable for trust-region least squares
exactly as `mirror_ratio` already is (optimize.py:601 docstring).  Optional smooth-max knob
deferred.

**(a) `redl_geometry_from_wout`** (parity lane; consumes `WoutData` or any wout-like, works on the
Zenodo `.nc` files): mirror simsopt `RedlGeomVmec.__call__` verbatim — linear-interp
`iotas[1:], bvco[1:], buco[1:], gmnc[:,1:], bmnc[:,1:]` from the half grid onto `surfaces`
(use `optimize._interp_half_grid` + `_half_grid`), synthesize `modB/sqrtg` on
`theta = linspace(0, 2π, ntheta, False)`, `phi = linspace(0, 2π/nfp, nphi, False)` from the
`xm_nyq/xn_nyq` cosine series, then `compute_trapped_fraction`; `R = (G + iota·I)·fsa_1overB`;
`psi_edge = −phi_wout[-1]/(2π)`.  jnp arrays throughout (still jit-able given a WoutData whose
tables are arrays), but its role is validation/FD-lane, not the implicit path.

**(b) `redl_geometry_from_state`** (traceable lane, pure `(state, rt)`):
1. `_field_chain(state, rt)` → `jacobian.sqrt_g`, `fields` (import from `optimize` or hoist the
   helper into a shared private module to avoid a cycle — decision: hoist `_field_chain`,
   `_iotas_half`, `_half_grid`, `_interp_half_grid` into `core/_state_diag.py` and re-import from
   both; keeps `optimize.py` API unchanged).
2. `|B|` on the half-mesh internal grid: `bmag = sqrt(max(2·(total_pressure − pressure), tiny))`,
   mirrored from the reduced `[0, π]` theta grid to the full circle exactly as
   `QuasisymmetryRatioResidual._pointwise_state` does (reuse its `i_src/k_src` construction; drop
   the axis row `js=0` *before* dividing — the 0·inf AD note at optimize.py:408 applies).
3. Same mirroring for `sqrt_g`; weight `w = |sqrt_g|`.
4. `iota = _iotas_half(...)[1:]`, `I, G = surface_currents(...).buco[1:], .bvco[1:]` (half mesh).
5. Interpolate the *per-surface scalars* (iota, G, I) and the *angular fields* (`bmag`, `w`) onto
   the requested `surfaces` with `_interp_half_grid` (it already handles trailing angular dims).
   Then `compute_trapped_fraction` on the interpolated fields.
6. `psi_edge = −signgs·hs·Σ phipf[1:]` (optimize.py:440 recipe).
   Default `surfaces`: `linspace(0.05, 0.95, 16)` — interior only; `s → 1` is excluded because
   `Te, Ti → 0` makes `ln Λ` and `ν*` blow up (simsopt warns; we must *not* sample there or AD
   produces inf).  Clamp `Te, Ti` with `jnp.maximum(T, T_floor=1.0 eV)` and `ne` with
   `max(ne, 1e17)` as cheap belt-and-suspenders inside `j_dot_B_redl`.

### 6.2 Traceable ⟨J·B⟩_vmec — the MHD identity, not a jxbforce port

Porting the jxbforce `jdotb` block (nyquist.py:762–833) would drag `bsubs_half_mesh` +
angle-derivative filtering into jnp.  Unnecessary: for a VMEC equilibrium the flux-surface-averaged
parallel current obeys (validated in the Zenodo `convertSfincsToVmecCurrentProfile` against VMEC's
own `jdotb`, and plotted as one of the "methods" in figure04/09/10):

    ⟨J·B⟩(s) = [ ⟨B²⟩(s)·dI/ds + μ0·I(s)·dp/ds ] / (2π ψ_a)

with `I(s) = (2π/μ0)·buco(s)` [A] (script check: `2π/μ0·bsubumnc(0,0) = ctor` at the edge),
`p` in Pa, `ψ_a = Φ_edge/(2π)` [Wb/rad].  In core internal units (`pressure` is μ0·Pa,
`buco` is T·m):

    vmec_j_dot_B(state, rt, surfaces):
        I_int  = buco                        # T·m, half mesh (surface_currents)
        dI_ds  = d(buco)/ds  (half→full central differences, ends extrapolated)
        dp_ds  = d(fields.pressure)/ds       # internal μ0·Pa
        jdotB  = ( fsa_B2·dI_ds + I_int·dp_ds ) · (2π/μ0) / (2π ψ_a)
               = ( fsa_B2·dI_ds + I_int·dp_ds ) / (μ0 ψ_a)

    (dimension check: [T²·(T·m)] / ([T·m/A]·[T·m²]) = A·T/m² ✓)

then `_interp_half_grid` onto `surfaces`.  `fsa_B2` comes free from `compute_trapped_fraction`
(step 6.1b reuses one geometry evaluation for both Jr and Jv).  **Acceptance gate**: on the three
Zenodo optimized wouts, this identity evaluated from the wout tables must match `wout.jdotb`
within 2% (interior s ∈ [0.1, 0.9]) — exactly what figure04 shows.  The wout lane
(`RedlBootstrapMismatch.J(eq)`) instead uses `eq.wout.jdotb` directly (simsopt parity).

### 6.3 Objective term: `RedlBootstrapMismatch`

Mirrors `QuasisymmetryRatioResidual`'s dual-lane shape:

    RedlBootstrapMismatch(profiles, helicity_n, surfaces=None, *, ntheta=64, nphi=65,
                          n_lambda=64)

- `J(eq: Equilibrium) -> residual vector` (FD lane): `redl_geometry_from_wout(eq.wout, surfaces)`,
  `Jr = j_dot_B_redl(...)`; `Jv = interp(eq.wout.jdotb, full grid → surfaces)`;
  `R_j = (Jv_j − Jr_j)/sqrt(Σ_k (Jv_k + Jr_k)²)` — simsopt `VmecRedlBootstrapMismatch.residuals`
  verbatim, so `sum(R²) = f_boot` of the paper.
- `residuals_state(state, rt)` (implicit lane): same `R_j` from `redl_geometry_from_state` +
  `vmec_j_dot_B`.  Both lanes agree at discretization level (different Jv route), same as the QS
  residual's wout-vs-internal-grid situation.
- The denominator depends on the dofs (self-normalizing residual).  simsopt/paper do exactly this;
  keep it, note it makes the Gauss–Newton model slightly non-standard but bounded `f_boot ≤ 1`.
- `total_state`, `profile` conveniences as in the QS class.

Term usage (target 0, weight 1 — the normalization already scales it):

    terms = [(qs.residuals_state, 0.0, 1.0),
             (boot.residuals_state, 0.0, w_boot),
             (aspect_ratio, 6.5, 1.0), (mean_iota, 1.05, 10.0), ...]

### 6.4 Current-profile degrees of freedom

Two lanes, in order of delivery:

1. **Power-series AC lane (no `ImplicitParams` change; ships first).**  Re-parameterize the deck
   with `PCURR_TYPE="power_series"` (AC = I'(s) coefficients, degree ≲ 12; the Zenodo QA deck's
   50-knot spline is a *result* representation — a 12–19 degree polynomial reproduces the smooth
   dI/ds fine, cf. the `degree = 19` polyfit in `convertSfincsToVmecCurrentProfile`).
   `ImplicitParams.ac` and `curtor` are already differentiable and `runtime_from_params` already
   rebuilds `icurv` from them.  Extend the drivers:
   - `least_squares(..., current_dofs=k)`: appends `[ac_0..ac_{k−1}, curtor_scaled]` to the dof
     vector.  FD path: `unpack` → `dataclasses.replace(inp, ac=..., curtor=...)`.  Implicit path:
     `params_of(x)` additionally sets `params.ac`/`params.curtor`; the one-hot tangent stack gains
     `k+1` rows (`t_ac`, `t_curtor`) — mechanical extension of optimize.py:1455–1462.
   - Scaling: curtor dof stored as `curtor/1e6` (MA) and ac dofs as `ac/|curtor|` so the
     trust region sees O(1) numbers (add to `x_scale`).
2. **Spline-knot lane (optional, exactness for re-running the Zenodo decks unmodified).**  Add
   `ac_aux_f: Array` to `ImplicitParams` (+ the three constructors; `ac_aux_s` stays static);
   dofs = knot values.  Deferred until (1) works — it is not needed to reproduce the paper.

### 6.5 Self-consistency loop — penalty first, Picard as refinement

**Primary (paper-faithful): single least-squares run** — boundary dofs (staged `max_mode`
(1, 2, 3, ...)) + current dofs (§6.4) with the term stack of §6.3.  No outer loop: the optimizer
finds the (shape, AC) pair where ⟨J·B⟩_vmec ≡ ⟨J·B⟩_Redl and QS holds.  This is the deliverable
optimization example.

**Secondary: `self_consistent_bootstrap(inp, profiles, helicity_n, n_iter=6, tol=1e-3)`** —
fixed-boundary Picard iteration used to (a) initialize the AC dofs before the big optimization,
(b) post-refine, (c) validate against the Zenodo `iteratedWithSfincs` config.  Host-side loop
(NumPy fine; no AD through the loop needed):

    for it in range(n_iter):
        eq   = solve_equilibrium(inp)
        Jr   = j_dot_B_redl(profiles, redl_geometry_from_wout(eq.wout, s_half), helicity_n)
        # invert the §6.2 identity for I(s) (the Zenodo script's 'smooth method'):
        #   solve  [⟨B²⟩ d/ds + μ0 dp/ds] I = 2π ψ_a · Jr,   I(0) = 0   (dense (ns,ns) solve)
        #   dI/ds = (2π ψ_a Jr − μ0 I dp/ds)/⟨B²⟩
        inp  = replace(inp, ac=polyfit(s, dI/ds, deg), curtor=I(1))
        f_boot = mismatch(eq)  ;  break when < tol

Convergence expectation from the paper: a few (≤ 5) iterations at these betas; damping factor
knob `relax=1.0` (0.5 if β=5% oscillates).

### 6.6 Kinetic profile inputs

`KineticProfiles` frozen pytree: `ne_coeffs, Te_coeffs, Ti_coeffs, Zeff_coeffs` (polynomial in s,
value+derivative via Horner — `profile_value_and_dds`; matches simsopt `ProfilePolynomial` and
covers the paper: `ne = n0·[1,0,0,0,0,−1]`, `Te = T0·[1,−1]`).  These are *not* VMEC inputs and do
not enter `VmecInput`; they are captured by the objective/loop.  Helper
`pressure_am_from_profiles(profiles) -> am coefficients` (Pa) generates the consistent VMEC `AM`
deck (`p = e·(ne Te + ni Ti)` polynomial product), used by the examples so pressure and kinetic
profiles can't drift apart.  Coefficients are jnp arrays → optionally differentiable (future:
profile-shape optimization) but treated as constants here.

---

## 7. Validation plan (Zenodo-anchored)

V1 **Formula parity vs simsopt** (pure-function level, no simsopt dependency at test time):
   feed hardcoded geometry/profile arrays (one row per config: precise-QA s=0.5 etc., values
   generated once offline with simsopt and pasted) through `j_dot_B_redl`; assert ≤ 1e-12 relative
   on `L31, L32, alpha, nu_e*, nu_i*, jdotB`.  Covers every coefficient in §3.

V2 **Redl vs SFINCS, precise QA/QH (paper Fig. 1 / Zenodo figure01)**: load
   `wout_new_QA_aScaling.nc` / `wout_new_QH_aScaling.nc` (Zenodo, benchmarks-only), profiles
   `ne=4.13e20(1−s⁵), Te=Ti=12keV(1−s)`, `s = linspace(0.025, 0.975, 39)`;
   compare to the hardcoded 39-point SFINCS arrays: RMS relative deviation ≤ 10% interior
   (s ∈ [0.1, 0.9]), matching the paper's reported agreement, and ≤ 1% against a stored
   simsopt-`j_dot_B_Redl` reference curve (tighter — same formula, different f_t numerics).

V3 **Trapped-fraction numerics**: `compute_trapped_fraction` vs simsopt reference values (stored
   offline) on the same wouts: `f_t` ≤ 0.3% relative, `epsilon` ≤ 0.5%, `fsa_B2/fsa_1overB`
   ≤ 1e-10 (quadrature-limited vs extremum-limited errors separated).

V4 **⟨J·B⟩_vmec identity**: on the three optimized Zenodo wouts (QA β=2.5, QH β=2.5, QH β=5
   iterated), §6.2 identity vs `wout.jdotb`: ≤ 2% interior.

V5 **Self-consistency of the published optima (Zenodo figure04)**: for
   `wout_20220218-01-021_QH...`, with `ne=2.2e20(1−s⁵), Te=Ti=10keV(1−s)`: `f_boot ≤ 1e-3`, and
   Redl curve vs the 49-point high-res SFINCS array ≤ 10% interior.  Ditto QA (n0=2.38e20,
   T0=9.45 keV) and QH β=5 (n0=3e20, T0=15 keV, figure09/10 arrays).

V6 **Round-trip / Picard**: run `self_consistent_bootstrap` from the QH β=2.5 *boundary* with a
   flat initial AC; converged `f_boot ≤ 1e-3` and `ctor` within a few % of −1.208 MA; iota profile
   close to the wout's.

V7 **End-to-end reproduction (example scripts, not CI)**: QA nfp=2 A=6 β=2.5% and QH nfp=4 A=6.5
   β=2.5% optimizations from the paper's initial conditions; success = QS residual and `f_boot`
   comparable to the Zenodo configs, cross-checked by loading the optimized state and running V5
   on it.  (Compute-heavy — lives in `examples/` + a README section, mirroring the existing
   QA/QH examples and the office-deck campaign workflow.)

---

## 8. Test plan (`tests/test_bootstrap.py`, CI-sized)

Physics limits & units:
- **Trapped fraction, analytic model** (simsopt's `test_compute_trapped_fraction` analogue):
  `B = B0(1 + ε cos θ)` on a (θ, ζ) grid with constant `sqrtg` ⇒ `Bmin/Bmax = B0(1∓ε)` exact,
  `epsilon` recovers ε, `fsa_B2 = B0²(1 + ε²/2)`; f_t against the large-aspect asymptote
  `f_t ≈ 1.46√ε` at small ε (loose 5% tolerance) and against a high-order quadrature self-check.
- **Axisymmetric tokamak benchmark**: an `nfp=1`/axisymmetric VMEC deck (circular tokamak, the
  solver's existing test inventory or a tiny new deck), banana-regime profiles ⇒ compare
  `j_dot_B_redl` with `helicity_n=0` against the stored simsopt value AND check the sign/magnitude
  against the standard estimate `⟨J·B⟩ ≈ −L31 G p'/psi_a` at Zeff=1 to 30% (formula-structure
  sanity, not precision).
- **Isomorphism invariance**: evaluating the QH geometry with `(iota, N)` vs `(iota − N, 0)`
  manually shifted gives identical `jdotB` (the substitution enters only via `iota − N`).
- **V1 + V3 + V4 subsets** with hardcoded arrays (no .nc files in repo; tiny reference vectors
  inline).
- **Zero-limits**: flat profiles (`dn/ds = dT/ds = 0`) ⇒ `jdotB ≡ 0`;
  `f_t → 0` (ε → 0) ⇒ `X31 → 0` ⇒ `jdotB → 0`.

Differentiability:
- `jax.jit` + `jax.grad` of `sum(j_dot_B_redl(...))` w.r.t. geometry arrays and profile
  coefficients: finite, NaN-free (exercises the double-where guards; include a surface where
  `1 − λB` hits 0 on the grid).
- **FD validation of the full chain**: scalar `f_boot` through
  `solve_implicit`/`runtime_from_params` — `jax.grad` w.r.t. one boundary dof, one `ac` dof, and
  `curtor` vs central finite differences of re-solved equilibria (tolerance and structure copied
  from `tests/test_implicit_grad.py`); small deck (solovev-like ns, low mpol/ntor) to stay
  CI-sized.
- `residuals_state` vs `J(eq)` lane agreement at discretization level (pattern:
  `tests/test_optimize_traceable_qs.py`).
- Driver: `least_squares(..., current_dofs=k, jac="implicit")` smoke test — one Gauss–Newton step
  on a toy config reduces `f_boot`.

---

## 9. Delivery order

1. `core/bootstrap.py`: `compute_trapped_fraction` + `j_dot_B_redl` + wout geometry lane; tests V1,
   V3-subset, physics limits.  (Pure functions; no solver coupling.)
2. Traceable lanes: `redl_geometry_from_state`, `vmec_j_dot_B`, `RedlBootstrapMismatch`
   (+ `_state_diag.py` hoist); tests V4-subset, lane agreement, AD/NaN tests.
3. Driver `current_dofs` extension (power-series lane) + FD-vs-implicit gradient test.
4. `self_consistent_bootstrap` Picard loop; V6.
5. Benchmarks against the full Zenodo set (V2, V5); QA/QH examples + README (V7).

## 10. Main risks / open points

- **f_t gradient quality** (main risk, see summary): hard `max` for Bmax and the near-singular
  `sqrt(1 − λB)` make ∂f_t/∂(boundary) the noisiest link; if FD-vs-AD mismatch exceeds tolerance,
  fall back to (i) finer angular grid, (ii) tanh-smoothed extrema, (iii) treating `f_t` with a
  custom_jvp built from the analytic λ-integral derivative.
- **⟨J·B⟩_vmec identity vs jxbforce**: 2% agreement assumed from the Zenodo evidence; if the gate
  fails on some config, port the jxbforce block to jnp (heavier: traceable `bsubs_half_mesh`).
- **Internal-grid resolution**: the traceable lane samples |B| on the solver grid; low-resolution
  optimization stages may under-resolve Bmax.  Mitigation: synthesize |B| from the state's
  spectral tables on a fixed 64×65 grid instead (costlier but resolution-independent) — decide
  during step 2 based on V3 numbers at example resolutions.
- **Self-normalizing residual denominator** (dof-dependent) slightly distorts Gauss–Newton; if the
  trust region misbehaves, freeze the denominator per Jacobian evaluation (simsopt lives with it).
- **`iota − N` near-resonance**: QH decks keep `iota` well away from `N`; the β=5% case adds the
  paper's iota barrier term — include it in the QH β=5 example.
