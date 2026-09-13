> Archived context, 13 September 2026. The top-level handoff gives current status; older job statements below are historical. Machine locations were removed; unbundled raw data are not implied to be available.

# HINT and VMEX exterior-field comparison

## Scientific question

A useful HINT–VMEX comparison asks two related questions. First, how well does the field reconstructed from a specified nested-surface equilibrium and its associated coils approximate the field of a relaxed free-boundary plasma? Second, which differences survive when pressure, current, external fields and numerical uncertainty are controlled? The answers need not be identical. A reconstruction can be accurate for its specified source distribution while the source distribution itself differs from a topology-changing equilibrium.

The present application concerns two existing finite-beta quasi-axisymmetric configurations with nominal beta 0.5% and 2.5%. They have different equilibrium files and associated coil sets. They should therefore be treated as two case studies, not as a controlled beta scan of one fixed configuration. Their numerical provenance and actual WOUT beta/current values are recorded in the study protocol.

## Equilibrium models and historical context

The original HINT lineage addressed three-dimensional helical equilibria using an Eulerian relaxation approach. Harafuji, Hayashi and Sato's 1989 paper is the foundational reference. Its bibliographic record and abstract support the historical identification; detailed claims about its discretization should be checked against the full article before reproducing that particular version.[1]

The HINT2 paper by Suzuki and colleagues introduced the subsequent implementation and peripheral-plasma applications. The maintained HINT3D repository identifies both this 2006 work and Suzuki's 2017 treatment of perturbed tokamaks as principal references. Their relevance is the development of the relaxation model and topology-capable equilibrium calculation, rather than a guarantee that a present Git branch has identical numerical behavior.[2,3]

The maintained code and supplied guide provide a more specific operational description. Step A holds the field fixed and relaxes pressure along field lines using a line average weighted by 1/B. Step B holds pressure fixed while advancing the dissipative magnetic/velocity system. The finite tracing length, wall intersections, smoothing, resistivity, pressure-reset schedule and convergence history are part of the actual calculation. Consequently, “same beta input” is not a sufficient definition of the same equilibrium problem.[4]

VMEC represents nested toroidal magnetic surfaces and computes an equilibrium within that representation. The W7-X benchmark contrasts this with HINT's ability to develop islands and stochastic regions. EXTENDER then combines a plasma-field reconstruction from VMEC with the vacuum field. The poster explicitly cautions that the resulting island-containing reconstruction is not itself a self-consistent MHD equilibrium with pressure and currents readjusted to those islands.[5]

This distinction is central to interpretation. Finding an island in a reconstructed field does not demonstrate that the prescribed nested-surface pressure remains compatible with that field. Conversely, a different HINT island width does not by itself show that virtual casing is inaccurate. The difference may enter through the source equilibrium, the reconstruction numerics, the external coil field, or HINT's own relaxation and boundary treatment.

## What the previous benchmarks establish

The supplied Geiger/Suzuki poster reports W7-X calculations with internal 5/5 islands and comparisons with VMEC/EXTENDER. The general geometry agrees, while internal island sizes and boundary structure differ. Its strongest methodological lesson is the effort to match the final HINT pressure distribution in VMEC, including comparisons with and without flattening at the same stored kinetic energy. It also notes a vacuum-reconstruction discrepancy, making a vacuum control essential. The poster's acronym “VMEX” denotes historical VMEC + EXTENDER; it is not a version identifier for the modern JAX package.[5]

The accompanying 2021 IAEA paper provides a fuller description of this family of comparisons. It considers pressure-profile mapping between HINT and VMEC and differences around internal islands and the edge. The results support examining geometry, current and pressure together rather than comparing only a Poincaré image. Their W7-X configurations and control-coil settings are specific to that experiment, so numerical island widths cannot be transferred to an unrelated QA equilibrium.[6]

A 2026 rapid-communication preprint compares VMEC and HINT for three LHD vacuum-axis configurations over an axis-beta scan. It reports agreement at low beta and configuration-dependent divergence as edge stochasticity reduces the region enclosed by closed surfaces. The compared quantities include magnetic-axis position, axis rotational transform and enclosed volume. This supports including boundary volume and axis observables in the QA study, but does not predict a QA threshold at either 0.5% or 2.5% volume beta.[7]

An earlier HINT2 study of the LHD equilibrium beta limit emphasizes the interaction between stochastic edge fields, pressure redistribution and stored energy. It explains why holding an initial pressure prescription does not automatically hold the relaxed pressure or volume-averaged beta. It also discusses finite connection lengths in the peripheral plasma. Those observations make pressure relaxation length and final pressure diagnostics substantive model choices, not merely performance settings.[8]

These sources do not establish a universal ordering of island widths between methods. Device geometry, resonances, pressure profile, current drive and boundary conditions all matter. The defensible hypothesis is narrower: once numerical errors and input mismatches are controlled, residual differences may identify the limitations of representing a topology-changing plasma by a nested-surface source equilibrium. Both agreement and disagreement would be informative outcomes.

## Virtual casing and the exterior domain

Virtual casing replaces the field of enclosed plasma currents by an equivalent surface representation. Lazerson's derivation describes the toroidal-system formulation and field decomposition. For the intended exterior calculation, the integration surface must enclose the represented plasma current and the external coil contribution must be accounted for consistently. The original surface field, orientation and current/field signs are therefore essential inputs.[9]

For comparison purposes, write the two results schematically as

\[
\mathbf B_{\rm HINT}=\mathbf B_{\rm coils}+\Delta\mathbf B_{\rm HINT},\qquad
\mathbf B_{\rm ext}=\mathbf B_{\rm coils}+\mathbf B_{\rm VC}.
\]

With exactly the same coil evaluator, their difference isolates the plasma-response difference plus any field-transfer error. If one side uses a tabulated/interpolated coil field and the other direct Biot–Savart evaluation, that cancellation is only approximate. A coil-only comparison on the same points is therefore required before attributing the remainder to plasma physics.

A sample point can be outside the original WOUT boundary yet still lie within the relaxed HINT pressure/current region. Such a point is useful for studying boundary displacement, but should not automatically be included in a comparison advertised as exterior to both plasmas. A pressure threshold helps classify points but is not a proof that every relevant current is enclosed. The boundary definition and exclusion criteria must be reported explicitly.

Near the source surface, field evaluation becomes numerically demanding. Malhotra and colleagues develop high-order singular quadrature for magnetic-fusion applications. The relevance here is that ordinary fixed quadrature can lose accuracy close to the surface, so a nominal tolerance or a visually smooth field is insufficient evidence of convergence.[10]

The practical validation is to hold target points fixed while refining the source surface and quadrature. Compare the configured modern VMEX extender against a converged exterior reference at the distances where both are valid. If near-surface continuation is used, give its domain, order and comparison error separately. Do not treat an exterior reconstruction formula as an independently validated interior island model.

## The finite-beta QA configurations

Landreman and Paul's high-precision quasisymmetry work supplies the vacuum-configuration context. Landreman, Buller and Drevlak subsequently optimized finite-beta quasisymmetric configurations while accounting for bootstrap-current consistency. The latter is the appropriate scientific context for the existing finite-beta fixtures, even though their filenames retain `LandremanPaul`.[11,12]

In the finite-beta optimization, the pressure and parallel-current profile participate in the equilibrium and optimization problem. A QA comparison that drops the toroidal current changes a major part of that problem. The selected WOUT files contain substantial net toroidal currents, so pressure-only HINT calculations would not test the intended configurations.[12]

The maintained HINT implementation uses a flux-function profile multiplying the local magnetic field for its imposed net-current contribution and rescales that contribution by its toroidal integral on a cut. That is not the same object as a VMEC current-profile derivative. A candidate transfer through the flux-function part of parallel current must be checked against both its integrated current and its spatial distribution. This conclusion follows from the current source implementation, not from a claim that the literature prescribes a unique discrete transfer rule.

Bootstrap consistency is also conditional. A current profile optimized for one nested equilibrium does not remain automatically consistent after pressure flattening, islands or a changed boundary. Initially, holding the specified current profile is a controlled modeling choice. A later bootstrap update would be a separate physical experiment and should not silently replace that baseline.

A September 12 local coefficient audit finds the supplied WOUT outer boundaries identical, with differing interior equilibria and current/coil constraints. This is a property of the local fixtures, not a claim that the paper optimized one common boundary at all beta values.

The associated coils require their own check. Coil optimization generally leaves a finite normal-field residual on the target surface. Comparing a fixed-boundary WOUT plus approximate coils with a free-boundary HINT state includes that residual in the experiment. A second calculation with a self-consistent free-boundary VMEX state can help isolate this effect while preserving the original given-boundary application as the primary reference.

## Boundary, pressure and convergence definitions

Three surfaces must be distinguished: the reference VMEC plasma boundary; the numerical or physical wall that terminates HINT field lines; and the boundary of the relaxed plasma region inferred from pressure and field-line topology. Equating them in the input would suppress part of the free-boundary question. The LHD example supplies a real vessel geometry; the QA fixtures do not, so a numerical-wall construction and sensitivity study are required.[4]

Likewise, distinguish axis beta, volume beta and pressure normalization. HINT's input pressure scale uses a specified reference toroidal field. The final field and plasma volume can change during relaxation. The final report should therefore include axis pressure, the exact beta definition, its integration volume, total pressure integral or kinetic energy, and the corresponding source WOUT values. Agreement between two percentages with different definitions has no useful physical meaning.

Pressure reset is another controlled choice. A continuation that reconstructs pressure is not equivalent to restarting from an unchanged pressure field. This can be tested directly by comparing uninterrupted and restarted trajectories. The supplied LHD deck and the W7-X comparison use different pressure-management choices, so reproducing one should not silently inherit the other's schedule.

Convergence needs both iteration and resolution evidence. A decreasing time derivative can indicate a stationary discretized state without demonstrating a small physical force residual. A small residual on a coarse grid can coexist with unresolved islands. Pressure, net current, axis position, boundary extent and fields at fixed exterior points should be checked after additional relaxation and after refining the grid or field-line integration.

Local numerical review has also shown why diagnostic implementation matters. A ratio can be undefined in zero-gradient regions; masking it after division still permits invalid floating-point operations. A plotting routine may deliberately suppress small-gradient regions, creating a visually small residual that is not a global force-balance certificate. Independent residual evaluation must state its denominator and mask and should accompany the native histories.

## Quantitative comparison design

The most efficient design begins with a vacuum control and a single low-beta case. Establish units, handedness, coil direction, grid transfer and common tracing conventions before scanning pressure or resolution. Then analyze the higher-beta case using the same validated measurement definitions. Additional variants should resolve an observed ambiguity rather than create an indiscriminate parameter sweep.

Measure exterior field differences in tesla as well as relative to the total field. Since a strong coil field can dominate the total norm, also report differences relative to the plasma-response field where that normalization is well-conditioned. Include vector direction and normal-field components, not just field magnitude. State the handling of points where a reference denominator approaches zero.

Poincaré comparisons should use common toroidal sections, seeds, integration tolerances and trace lengths. Extract island widths with a declared geometrical definition and an uncertainty tied to grid spacing and tracing resolution. A resonance that is thinner than the effective numerical resolution is unresolved, even when a plotted point cloud appears suggestive. Connection length is meaningful only with a specified wall and maximum trace length.

The two principal interpretations should remain separate. The given-boundary comparison asks how the original WOUT-based extender performs against a relaxed HINT state using its associated coils. The controlled-equilibrium comparison additionally matches equilibrium constraints and, where useful, the final HINT pressure/stored energy in the spirit of the W7-X work. Retuning one result until its contours agree would obscure rather than answer the first question.

No desired agreement threshold follows automatically from the literature. Working tolerances should be chosen relative to the feature being measured and demonstrated by refinement. A reported physical discrepancy should be substantially larger than the controlled numerical changes. Otherwise, the conclusion is that the comparison remains numerically unresolved.

## Evidence limits and remaining questions

The source set supports the comparison framework and the reasons to expect topology-related differences, but it does not provide a quantitative prediction for these exact two QA cases. Raw configuration-specific inputs are needed to reproduce the W7-X poster numerically. The LHD example provides a practical validation workflow, not a complete independent certification of its displayed solution.

The original HINT and HINT2 journal articles and the 2017 article are identified as foundational reading, with full-text access limitations retained below. Their implementation details should not be inferred solely from abstracts. The 2026 LHD study is a short preprint; its reported device dependence is relevant context, while numerical-resolution information must come from the actual calculation and any additional author material.

The decisive remaining questions are empirical: whether final QA pressure/current constraints can be matched; how much the supplied coil fit moves the free boundary; whether the inferred islands and edge stochasticity survive independent refinement; and whether exterior differences exceed quadrature and field-transfer errors. Answering those questions produces a useful benchmark regardless of whether HINT and the extender ultimately agree closely.

## Sources

1. Harafuji, K., Hayashi, T., and Sato, T. “Computational study of three-dimensional magnetohydrodynamic equilibria in toroidal helical systems.” *Journal of Computational Physics* 81, 169–192 (1989). [Publisher record](https://www.sciencedirect.com/science/article/pii/0021999189900697). Bibliographic record/abstract; full text not available in the retained collection.
2. Suzuki, Y., Nakajima, N., Watanabe, K. Y., Nakamura, Y., and Hayashi, T. “Development and application of HINT2 to helical system plasmas.” *Nuclear Fusion* 46, L19–L24 (2006). [DOI](https://doi.org/10.1088/0029-5515/46/11/L01). Repository reference and bibliographic identification; full text not obtained.
3. Suzuki, Y. “HINT modeling of three-dimensional tokamaks with resonant magnetic perturbation.” *Plasma Physics and Controlled Fusion* 59, 054008 (2017). [DOI](https://doi.org/10.1088/1361-6587/aa5adc). Repository reference; full text not obtained.
4. *HINT Example LHD Case manual*, supplied 30-page PDF, especially pp. 1–10 and 18–29; Suzuki, Y., *Guide to using HINT*, 25 May 2026, supplied through the [example collection](https://u.pcloud.link/publink/show?code=kZwo5G5ZYPUNwfLwGiXRxQ4g7zFqTQG6Cl6y); HINT3D `current`, commit bf31fc39. Source files are preserved locally with hashes.
5. Geiger, J., Suzuki, Y., and the W7-X Team. “HINT equilibrium calculations for W7-X for cases with internal magnetic islands.” Poster, 23rd ISHW, Warsaw, June 20–24, 2022. Supplied file `Poster_hoch_ISHW22_JGeiger_v2.pdf`, single page. No public URL assumed.
6. Geiger and colleagues. W7-X HINT/VMEC–EXTENDER comparison, IAEA Fusion Energy Conference manuscript (2021). [Full manuscript](https://conferences.iaea.org/event/214/contributions/17520/attachments/10058/15492/IAEA2020_JGeiger_Manuscript_8p_finalversion.pdf). Full text retained.
7. Civit-Bertran, A., Suzuki, Y., and Futatani, S. “Systematic comparison of VMEC and HINT equilibrium calculations for finite-beta LHD plasmas.” [arXiv:2606.10490v2](https://arxiv.org/html/2606.10490v2), revised 10 June 2026. Preprint; full HTML available.
8. Suzuki, Y., Watanabe, K. Y., Sakakibara, S., Nakajima, N., and Ohyabu, N. “Theoretical studies of equilibrium beta limit in heliotron plasmas.” IAEA FEC 2008, TH/P9-19. [Full paper](https://www-pub.iaea.org/mtcd/meetings/fec2008/th_p9-19.pdf).
9. Lazerson, S. A. “The Virtual Casing Principle for 3D Toroidal Systems.” *Plasma Physics and Controlled Fusion* 54, 122002 (2012). [PPPL report copy](https://bp-pub.pppl.gov/pub_report/2014/PPPL-4993.pdf). The report path is dated 2014; the journal article is 2012.
10. Malhotra, D., Cerfon, A. J., O'Neil, M., and Toler, E. “Efficient high-order singular quadrature schemes in magnetic fusion.” *Plasma Physics and Controlled Fusion* 62, 024004 (2020). [arXiv:1909.07417](https://arxiv.org/abs/1909.07417).
11. Landreman, M., and Paul, E. “Magnetic Fields with Precise Quasisymmetry for Plasma Confinement.” *Physical Review Letters* 128, 035001 (2022). [arXiv:2108.03711v2](https://arxiv.org/html/2108.03711v2).
12. Landreman, M., Buller, S., and Drevlak, M. “Optimization of quasisymmetric stellarators with self-consistent bootstrap current and energetic particle confinement.” *Physics of Plasmas* 29, 082501 (2022). [arXiv:2205.02914v2](https://arxiv.org/html/2205.02914v2); [DOI](https://doi.org/10.1063/5.0098166).
