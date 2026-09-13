> Archived context, 13 September 2026. The top-level handoff gives current status; older job statements below are historical. Machine locations were removed; unbundled raw data are not implied to be available.

# QA resistivity sensitivity at matched magnetic time

The completed trials show faster growth of negative current in positive-pressure cells with higher numerical resistivity, alongside a larger force residual. They do **not** select a converged or more accurate resistivity setting. Oppositely signed current persists in zero-pressure regions and almost cancels the pressure-region integral over the retained domain. The current-free exterior and attained-current gates remain unmet.

## Controlled comparison

The reference `qa-cadence-n1000-001` and trials `qa-resistivity-eta2e-3-001` and `qa-resistivity-eta5e-3-001` all restart from SHA-256 `b801cd321fce9473efd668143409f06efdc7166003fcd1a47e7b8ebb3a38ff62`. All finish at normalized magnetic time 1.0800000000000185, after two outer steps with 1000 magnetic substeps each and `dt_b=1e-4`. Input diffs change only `eta0`, from 0.001 to 0.002 or 0.005. Pressure/current/coil constraints are preserved; these are uniform numerical-resistivity variations, not different physical bootstrap models.

The profiles use the original `current_mapping.csv`; local and remote validation host SHA-256 agree at `4faf985eb768e14cd02f969a17b2404f3a696eb49310eb59b7581cba166e6621`. Trial provenance records binary SHA-256 `56faef4649083c9cb06c10764c2cd63bad78ce00d1576004f5a9fd3dcab18fb6`. Shared restart and matched final time isolate the input variation at this transient endpoint; they do not demonstrate a common asymptotic equilibrium.

## Attained current and force

Currents below are fourth-order `curl(B−Bvac)/mu0` rectangular cut integrals, averaged over toroidal cuts. Every reduction excludes two outer R/Z layers. Pressure support is `P>0`; wall and pressure masks are distinct.

| Quantity | eta0=0.001 | eta0=0.002 | eta0=0.005 |
|---|---:|---:|---:|
| Positive-pressure current, kA | −17.6924 | −19.8894 | −25.5209 |
| Inside-wall current, kA | −1.1101 | −1.4265 | −2.7189 |
| Zero-pressure current, kA | +17.8093 | +20.0478 | +25.6688 |
| All-interior current, kA | +0.1169 | +0.1584 | +0.1479 |
| Outside-wall current, kA | +1.2270 | +1.5849 | +2.8668 |
| All-interior mean integral of absolute toroidal current, kA | 64.6694 | 65.8052 | 71.6782 |
| Force squared / (pressure-gradient squared + Lorentz-force squared), P>0 | 0.0140061 | 0.0152315 | 0.0197138 |
| Recorded run time, seconds | 90.697 | 85.878 | 89.060 |

The imposed plasma-current target is −95.77835 kA. The positive-pressure-region current remains far from that target, and its mask still requires qualification as a plasma-current definition. The target must not be applied indiscriminately to wall or all-domain integrals, which can include return current. The all-interior integral being close to zero does not mean that local currents are small: the absolute-current integrals and signed mask decomposition demonstrate cancellation. Likewise, increasing the negative pressure-region current alone does not prove improvement in the full equilibrium. Runtime differences are single-run measurements, not a speedup claim.

## Field sensitivity

The same native MAGVAL spline sampled 192 fixed targets exterior to the original WOUT boundary. Those targets are not certified exterior to the relaxed current distribution.

| Difference from eta0=0.001 | eta0=0.002 | eta0=0.005 |
|---|---:|---:|
| Target RMS vector-field difference, mT | 1.5228 | 5.4080 |
| Target maximum vector-field difference, mT | 2.4288 | 8.2141 |
| Inside-wall volume-weighted RMS difference, mT | 1.1955 | 4.3109 |
| Outside-wall maximum grid difference, mT | 36.5378 | 66.5113 |
| Full-grid relative pressure L2 difference | 0.0015425 | 0.0035079 |

The large outside-wall maxima require spatial inspection and refinement before interpretation. They are not proof of an interpolation defect or a physical edge instability. These are differences among coarse transient HINT states, not HINT–VMEX errors.

## Parallel-current profile interpretation

The reviewed helper reconstructs 501 pressure-enclosed toroidal-flux levels on phi=0 and forms a candidate `C*lambda(s)*B`, normalizing that candidate's cut current to the requested target. Its all-cut mean is about −95.65 to −95.67 kA. This is an algebraic normalization check, not evidence that the attained current has reached the imposed drive.

The saved snapshots do not contain native historical `ss` or `Jnet`. Reconstruction here uses the snapshot maximum as a proxy for traced-axis pressure. Native labels are formed at the beginning of a magnetic step and held through its substeps, whereas these candidate labels use the final snapshot. Bin averages are finite-volume pressure-label shell averages, not exact flux-surface averages; geometry shifts also change bin membership.

Nevertheless, all three snapshots show a substantial qualitative mismatch in the retained moment `integral(J·B dV)/integral(B² dV)`: the first shell, s in [0,0.1), is positive (approximately +33.5 to +36.2 thousand A/(T m²)), while the candidate drive is negative (approximately −82.9 to −83.7 thousand). The last active shell, s in [0.9,1), also has opposite signs. Midradius attained moments become more negative with eta but remain distinct from the candidate. These observations motivate a saved-native-label/current audit; they do not prescribe a new current formula.

The candidate drive is zero in zero-pressure and outside-wall cells in all three reconstructions, while attained response current is nonzero. This agrees with the source audit: the strict `s<jcuts=1` drive cutoff is different from a constraint setting curl-derived current to zero. Nonzero `profj` at s=1 does not itself inject drive at the excluded endpoint.

**Full-vector current-minus-parallel-drive RMS is not a zero target.** It includes required pressure-driven perpendicular current and other equilibrium current structure absent from the imposed parallel drive. It must not serve as a standalone resistivity-selection or convergence criterion.

## Next acceptance evidence

The immediate spatial diagnostic is an independent boundary-circulation check. For a rectangle at fixed phi, using physical cylindrical response components,

\[
\mu_0 I_\phi=\int_{R_l}^{R_r}[\Delta B_R(R,Z_t)-\Delta B_R(R,Z_b)]\,dR
-\int_{Z_b}^{Z_t}[\Delta B_Z(R_r,Z)-\Delta B_Z(R_l,Z)]\,dZ.
\]

Compare this with the integrated response curl on successively expanded rectangles, with consistent quadrature domains. A point-weighted rectangular FD sum and a trapezoidal boundary integral need not coincide exactly at finite resolution; their discretization difference must be recorded. Track signed/absolute current in shells relative to positive pressure, the numerical wall and the grid boundary on each cut and across saved times. Near-zero total current can arise from circulation and return-current structure; it is not intrinsically a failed plasma-current normalization.

The source imposes zero normal-field increments on the R and Z faces while extrapolating tangential increments (stepb_mod.f90:1511 (source/evidence reference; see handoff index), stepb_mod.f90:1570 (source/evidence reference; see handoff index)). This does not alone fix the poloidal tangential circulation or prove a zero all-domain current constraint. The near-zero all-interior mean also hides cut variation: baseline cuts span −2.169 to +1.706 kA. Most signed zero-pressure return current lies inside the wall (16.582, 18.463 and 22.802 kA for the three settings). Its location relative to the plasma edge is the useful next discriminator, rather than assuming all return current is a wall artifact or coil incompatibility.

Retain all three settings as transient sensitivity evidence. Before selecting production controls, inspect current structure on common spatial sections and distinguish pressure-region, zero-pressure and boundary contributions. A longer bounded continuation should test whether the current and force histories approach a stationary state; a separate resolution/domain check must test whether exterior cancellation persists. These interventions answer different questions and should not be bundled into an uncontrolled scan.

Native saved `ss`, traced-axis pressure and imposed-current fields would enable a closer comparison with the reconstructed drive. Until those are available, preserve the helper's candidate label in every plot or table. The 0.5% equilibrium remains unqualified; the results do not justify promoting the 2.5% case or claiming quantitative exterior agreement.

## Retained evidence

- eta0=0.002 comparison (see bundled evidence index) and eta0=0.005 comparison (see bundled evidence index), including native target-sampler hash. Their generic helper's `cadence sensitivity` limitation text also applies as a transient limitation; the varied parameter here is resistivity.
- Current-profile records: reference (source/evidence reference; see handoff index), eta0=0.002 (source/evidence reference; see handoff index), eta0=0.005 (source/evidence reference; see handoff index).
- Each run directory retains compact pilot summary, restart provenance and native target samples. Full native fields remain on remote validation host. Postprocessing completed without launching an equilibrium solver.

Assessment prepared 13 September 2026; no scientific acceptance gate is promoted by this document.

## Completed spatial and circulation follow-up

The new return-current postprocessor (see bundled evidence index) was run on the t=0.88 follow-up, t=1.08 reference and t=1.08 eta0=0.005 trial. An analytic temporary-fixture check with Bz=R² verifies the real helper's signed circulation/current comparison to below 1e-7 A; no solver runs were added.

On the retained outer rectangle, the mean circulation-derived currents are −211.73 A, +138.94 A and +109.83 A respectively. The independent line quadrature and integrated fourth-order curl agree to approximately 5–6e-11 A RMS across cuts there. Inner rectangles have finite discretization mismatches of approximately 45–222 A RMS. Their enclosed-current integrals differ from the pilot's point-weighted domain sums because this comparison uses trapezoidal area quadrature. The small outer integral is consistent with the sampled boundary circulation; it is not a sign or mu0 error in the current postprocessor.

Distance shells measured in each R/Z cut reveal where the signed return current resides:

| Distance outside P>0 support | t=0.88, eta0=0.001, kA | t=1.08, eta0=0.001, kA | t=1.08, eta0=0.005, kA |
|---|---:|---:|---:|
| 1–2 cm | +6.028 | +6.198 | +6.730 |
| 2–4 cm | +5.387 | +5.873 | +7.334 |
| 4–8 cm | +3.660 | +4.496 | +8.066 |
| 8–16 cm | +0.252 | +0.825 | +2.934 |
| Beyond 16 cm | +0.353 | +0.417 | +0.605 |

There are no grid-cell centers in the 0–1 cm shell: grid spacings are approximately 1.46 cm in R and 1.82 cm in Z. The first shell is therefore only about one grid step wide. Approximately 93% of signed zero-pressure return current lies within 8 cm at t=1.08 for eta0=0.001, versus 86% for eta0=0.005. The greater farther-out contribution with eta is consistent with transient redistribution; these three snapshots do not establish a diffusion scaling or a physical return-current equilibrium.

The eta0=0.005 phi=0 section (source/evidence reference; see handoff index) shows opposite current bands around the pressure region. Black contours mark P=1% maximum and green contours the numerical wall. The strong edge gradients span few cells; both finite resolution and changing support must be assessed before interpreting their width. JSON files also retain a second shell decomposition using the 1% pressure threshold, so weak-pressure tails are not hidden by a single mask choice.

**Next-run implication:** prioritize a bounded grid/edge-resolution comparison at the same physical time, source constraints and moderate reference resistivity over another increase of eta. Reuse time snapshots for shell diagnostics before spending on longer relaxation. If a controlled fine-grid restart is unavailable, document interpolation/restart changes and use matched coarse/fine initialization; do not label an interpolated restart an isolated grid test without validating its transfer. A longer moderate-eta continuation remains useful for time convergence, but cannot resolve the one-cell edge-gradient question alone. No conclusion of coil/current incompatibility or a source-code defect follows from this cancellation diagnostic.

Compact records and plots: t=0.88 (source/evidence reference; see handoff index), reference t=1.08 (source/evidence reference; see handoff index), eta0=0.005 t=1.08 (source/evidence reference; see handoff index). Distances are per-cut distances to pressure-selected grid cells, not 3D normal distances.
