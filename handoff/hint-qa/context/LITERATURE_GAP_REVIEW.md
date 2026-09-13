> Archived context, 13 September 2026. The top-level handoff gives current status; older job statements below are historical. Machine locations were removed; unbundled raw data are not implied to be available.

# Literature gaps and acceptance evidence for the HINT–VMEX comparison

The remaining scientific bottleneck is matching the equilibrium constraints and demonstrating numerical convergence. Additional literature strengthens those requirements, but does not certify the two local QA equilibria or predict their island widths. This review supplements [LITERATURE_CONTEXT.md](LITERATURE_CONTEXT.md); it separates published evidence, maintained-code behavior and proposed tests.

## Foundation-paper access and scope

| Reference | Evidence available | What can be concluded |
|---|---|---|
| Harafuji, Hayashi and Sato (1989), JCP 81, 169–192 | Publisher abstract and bibliographic record; full article not obtained | The original method used time-dependent relaxation on a rotating, nonorthogonal helical Eulerian grid and considered finite-beta equilibria without net current. It is not a specification of the present cylindrical implementation. [1] |
| Suzuki et al. (2006), NF 46, L19–L24 | Bibliographic identification and institutional description of HINT2; full journal article not obtained | HINT2 separates pressure and magnetic relaxation and accommodates islands/stochastic fields. Historical implementation details still require the article or version-specific source. [2,3] |
| Suzuki (2017), PPCF 59, 054008 | Bibliographic identification; DOI resolution did not yield accessible full text | Retain this as foundational reading. No detailed current-closure or tokamak-response claim here depends on its inaccessible methods. [4] |
| Schmitt et al. (2025), Infinity Two equilibrium study | Full publisher HTML, especially Appendices B–D; Suzuki is a coauthor | Provides a directly inspectable description of HINT and an additional HINT–VMEC comparison, with explicit limitations. [5] |

An open-access flag or an indexed PDF link is not evidence that the full paper has been inspected. The three historical journal-paper access gaps remain open; the newer paper is a supplementary primary source, not a substitute edition of them. NIFS's institutional description independently identifies cylindrical dissipative MHD and alternating pressure/field evolution for HINT2. [3]

## A current-closure ambiguity that must stay explicit

Schmitt et al., Eq. C.3, prints

\[
\mathbf J_{\mathrm{net}}=\mathbf B\,\frac{\langle\mathbf J\cdot\mathbf B\rangle}{B^2},
\]

with a **local** denominator. Its Appendix D instead uses a surface-averaged denominator in the conversion between toroidal and parallel current. These expressions must not be silently identified. Appendix C also reports connection-length pressure averaging, resets at iterations 20/40/60, and artificial dissipative parameters. Its particular 64³ mesh was unreliable; 128³ versus 256³ preserved major results but changed edge-map fidelity. These are configuration-specific observations. The comparison adjusted VMEC `PHIEDGE` to align selected boundary extrema. [5, Appendices C–D]

**Maintained-code evidence.** In stepb_mod.f90 (source/evidence reference; see handoff index), `cal_volume` integrates the tabulated profile multiplied by total toroidal field over the selected cut, `kcros=1`. In cal_netj (source/evidence reference; see handoff index), the linear branch sets all three imposed-current components to total field times the same normalized profile. Thus the implemented input contract is

\[
\mathbf J_{\mathrm{imposed}}=C\lambda(s)\mathbf B,
\]

with a scalar normalization determined on one cut, subject to code units and current cutoff. There is no local division by field magnitude in this branch. The existing candidate profile is proportional to the WOUT ratio \(\langle\mathbf J\cdot\mathbf B\rangle/\langle B^2\rangle\). This is source inspection of the maintained worktree, not a newly established equivalence between the paper and the code.

**Analytical consequence.** If \(s\) is a true field-surface label and \(\nabla\cdot\mathbf B=0\), then \(\nabla\cdot[\lambda(s)\mathbf B]=0\). A frozen or imperfect initial flux map need not have this property for the evolving field. Conversely, a factor \(A(s)/B^2\) generally varies along a field line. This algebra shows why matching the net toroidal integral does not alone validate either spatial current distribution. It does not prove the printed equation is a typo: its physical definition and implementation context must be established before changing a solver.

The decisive validation should record four different objects: the input profile; the explicitly imposed current; the current from the curl of the relaxed plasma-response field; and the total current integrated on several toroidal cuts. On surviving surfaces, compare the final parallel-current moment and enclosed-current profile against WOUT. A pressure-surface label becomes ambiguous across flat-pressure islands, so any averaging there needs a declared geometric definition.

This is a targeted follow-up to the existing [QA input contract](QA_INPUT_CONTRACT.md), not a recommendation to alter the input formula based on a single printed equation.

## Finite-beta comparisons are constraint comparisons

Landreman, Buller and Drevlak optimize the equilibrium current profile together with geometry, penalizing disagreement between equilibrium and predicted bootstrap parallel current. The study explicitly couples equilibrium and kinetic consistency. Therefore retaining a source WOUT current profile while pressure or geometry changes is a prescribed-current experiment; bootstrap consistency must be reassessed separately. Its optimization framework does not establish that the two local files form a one-parameter beta scan. [6, Sections IV–V]

For these fixtures, preserve the original given-boundary application as one result. A separately matched free-boundary calculation can then ask whether the original coil fit or changed equilibrium constraints explain the discrepancy. Record every adjusted quantity, including toroidal flux, pressure profile, integrated pressure and current. Alignment obtained by changing enclosed flux is informative, but it answers a different question from keeping all original constraints fixed.

An accessible eleven-page preprint, arXiv:2004.07565v1, discusses pressure-profile control and loss of closed-surface volume in an LHD configuration. It explicitly excludes net toroidal current and distinguishes degradation of the beta response from eventual saturation. Its title and five-author list differ from the later three-author 2020 journal paper cited in the existing literature context; this review does **not** assert that it is the identical accepted manuscript. Its Section II changes an outboard pressure-profile anchor, illustrating why such controls must be recorded as changes to the experiment. [7, pp. 4–9]

The 2026 Civit-Bertran–Suzuki–Futatani preprint scans **axis beta** for three LHD vacuum-axis configurations, with initial pressure linear in normalized toroidal flux. It compares axis displacement, axis transform and closed-surface volume, and reports configuration-dependent divergence. The short full text does not provide the resolution/relaxation matrix needed to certify the local QA runs. Its thresholds should not be applied to the QA volume-beta labels. [8]

**Recommended pressure record:** retain dimensional peak pressure, pressure integral, the precise volume used in each average, final field normalization and both source/final beta definitions. For an isotropic monatomic plasma, the thermal-energy convention is \(W=3\int p\,dV/2\); a stored pressure integral is not numerically the same quantity. State the adopted convention when reproducing a published energy match.

## What the W7-X benchmark can and cannot establish

The retained Geiger manuscript, Section 3 and Figs. 8–10, maps the final HINT pressure to VMEC's radial coordinate and compares profiles with and without island flattening at matched energy. It reports smaller reconstructed internal 5/5 islands, while the edge 10/9 chain and global geometry agree better. Crucially, an internal-island discrepancy also appears in a vacuum reconstruction against Biot–Savart. These observations preclude attributing the entire finite-beta discrepancy to self-consistent plasma response. [9]

The supplied 2022 poster supplies the immediate experimental context but no transferable numerical QA island-width target. Exact poster reproduction still requires its configuration-specific machine/current files, equilibria, numerical settings and final fields. The available LHD example should proceed independently of that access gap. [10]

The historical EXTENDER documentation exposes separate coil/plasma/total-field output, fixed target points and adaptive integration controls. It also warns that coil currents are read per filament from the fourth coil-file column. Those capabilities suggest a useful independent control if that implementation is built later; they do not establish identical behavior in modern JAX VMEX. The documentation does not specify a verified near-boundary error for this study. [11]

The 2005 Drevlak–Monticello–Reiman institutional abstract establishes EXTENDER_P's historical relationship to a generalized virtual-casing procedure and to VMEC/PIES initial conditions. Full text was not obtained here, so no detailed interior-extension algorithm is inferred from that abstract. [12]

## Numerical evidence required before topology claims

These are proposed study tests, not tolerances asserted by a publication:

| Observable | Required separation | Efficient next test |
|---|---|---|
| Attained toroidal current | Imposed profile versus curl-derived current; cut normalization versus conservation | Evaluate several cuts after the pilot; repeat the flux/current map at the next grid level before a long run. |
| Pressure equilibrium | Initial profile, ramp, reset and final state | Compare histories sufficiently long after the last pressure intervention; evaluate field-parallel pressure variation independently. |
| Exterior response | Coil interpolation versus reconstructed plasma field | Keep Cartesian targets fixed, subtract the same coil definition, and publish errors in tesla and relative to the response. |
| Last closed surface | Reference WOUT, wall, pressure support and traced topology | Use a common tracer and fixed wall; bracket surviving surfaces with refined seeds and trace lengths. |
| Island width | Field-grid, quadrature and tracing errors | Refine the field source and tracer independently; report unresolved widths rather than assigning a code disagreement. |
| Relaxed equilibrium | Small time change versus physical force balance | Recompute curl, divergence and pressure gradients with declared masks and denominators; extend relaxation and repeat at higher resolution. |

For every comparison point, retain a classification relative to the original boundary, relaxed pressure/current region and wall. A point exterior to WOUT can enter the relaxed plasma. A low pressure threshold alone does not establish absence of current, particularly with imposed parallel current. Conversely, interpolation errors in an externally prescribed vacuum field can contaminate curl-based diagnostics; compute the response and vacuum contributions separately.

One common field-line tracer is especially valuable because native postprocessing and JAX tracing otherwise combine field and integration differences. First demonstrate a coil-only reference, then use identical launch points and toroidal sections for the response fields. An integration stopping at a finite maximum length must distinguish “did not reach the wall within the cap” from “closed.”

## Consequences for differentiability and optimization

The finite-beta optimization literature supplies a physically motivated objective and coupled-current constraint, not a validation of an automatic derivative through today's VMEX/Solvax implementation. The derivative of a smooth, uniquely selected converged equilibrium is a different object from the derivative of a fixed number of relaxation iterations. History-dependent solves and unresolved primal residuals must remain visible in derivative eligibility.

For the present development lane, use the source equilibrium constraints as the test contract. Re-solve perturbed states independently, verify the Taylor regime before solver noise dominates, and compare tangent and adjoint products on the same qualified root. A successful small differentiable PDE example validates infrastructure, not the QA equilibrium model. Topology diagnostics based on island appearance or wall-connection events may change discontinuously and should not be presented as ordinary smooth optimization objectives without a separately defined treatment.

The best literature-informed order remains: current/pressure contract, bounded 0.5% pilot, LHD and QA refinement, controlled 2.5% comparison, then quantitative islands and optimization results. The new source evidence strengthens these gates; it does not justify a broad parameter matrix before them.

## Sources and evidence status

1. K. Harafuji, T. Hayashi and T. Sato. “Computational study of three-dimensional magnetohydrodynamic equilibria in toroidal helical systems.” *Journal of Computational Physics* 81, 169–192 (1989). [Publisher abstract](https://www.sciencedirect.com/science/article/pii/0021999189900697), DOI 10.1016/0021-9991(89)90069-7. Full text not obtained.
2. Y. Suzuki, N. Nakajima, K. Y. Watanabe, Y. Nakamura and T. Hayashi. “Development and application of HINT2 to helical system plasmas.” *Nuclear Fusion* 46, L19–L24 (2006). [DOI](https://doi.org/10.1088/0029-5515/46/11/L01). Bibliographic identification; full text not obtained.
3. National Institute for Fusion Science, Numerical Simulation Reactor Research Project. [Code activities: HINT2](https://nsrp.nifs.ac.jp/activity/activity01.html). Institutional description in Japanese; historical page, no current-version claim.
4. Y. Suzuki. “HINT modeling of three-dimensional tokamaks with resonant magnetic perturbation.” *Plasma Physics and Controlled Fusion* 59, 054008 (2017). [DOI](https://doi.org/10.1088/1361-6587/aa5adc). Full text not obtained.
5. J. C. Schmitt et al. “Magnetohydrodynamic equilibrium and stability properties of the Infinity Two fusion pilot plant.” *Journal of Plasma Physics*, published online 24 March 2025. [Full publisher article](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/magnetohydrodynamic-equilibrium-and-stability-properties-of-the-infinity-two-fusion-pilot-plant/6348ED5B1CA97BFF845C75F6284D5415). Full HTML inspected, especially Appendices C–D. Device-specific results; published equations retained as printed.
6. M. Landreman, S. Buller and M. Drevlak. “Optimization of quasisymmetric stellarators with self-consistent bootstrap current and energetic particle confinement.” *Physics of Plasmas* 29, 082501 (2022). [arXiv:2205.02914v2](https://arxiv.org/html/2205.02914v2), [DOI](https://doi.org/10.1063/5.0098166). Previously retained full text reused, Sections IV–V.
7. Y. Suzuki, S. Sakakibara, K. Y. Watanabe, N. Nakajima and N. Ohyabu. “Theoretical studies of equilibrium beta limit in heliotron plasmas.” [arXiv:2004.07565v1, full PDF](https://arxiv.org/pdf/2004.07565v1), 16 April 2020. Eleven-page preprint inspected. Do not equate bibliographic version with the differently titled/attributed journal paper, DOI 10.1063/5.0015106, without verification.
8. A. Civit-Bertran, Y. Suzuki and S. Futatani. “Systematic comparison of VMEC and HINT equilibrium calculations for finite-beta LHD plasmas.” [arXiv:2606.10490v2](https://arxiv.org/html/2606.10490v2), 10 June 2026. Full short preprint inspected; no claim of peer-reviewed status.
9. J. E. Geiger et al. IAEA Fusion Energy Conference manuscript (2021), Section 3, Figs. 8–10. [Original manuscript URL](https://conferences.iaea.org/event/214/contributions/17520/attachments/10058/15492/IAEA2020_JGeiger_Manuscript_8p_finalversion.pdf). Full text reused from retained extraction (source/evidence reference; see handoff index); renewed web retrieval failed, existing local PDF remains available.
10. J. Geiger, Y. Suzuki and W7-X Team. “HINT equilibrium calculations for W7-X for cases with internal magnetic islands.” 23rd ISHW, Warsaw, 20–24 June 2022. Supplied single-page poster (source/evidence reference; see handoff index). Existing extraction reused; no public URL assumed.
11. Princeton University STELLOPT project. [EXTENDER documentation](https://princetonuniversity.github.io/STELLOPT/EXTENDER.html). Official implementation documentation inspected; historical EXTENDER, distinct from JAX VMEX.
12. M. Drevlak, D. Monticello and A. Reiman. “PIES free boundary stellarator equilibria with improved initial conditions.” *Nuclear Fusion* 45, 731–740 (2005). [Princeton institutional record](https://collaborate.princeton.edu/en/publications/pies-free-boundary-stellarator-equilibria-with-improved-initial-c/), DOI 10.1088/0029-5515/45/7/022. Abstract only.

Online evidence checked 12 September 2026. Existing manuals, source code and local numerical results are distinguished from published literature throughout. No paper or code is treated as a certificate of convergence for an untested configuration.
