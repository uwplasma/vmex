# Three-terms free boundary: what it does and where it stops

This page states what the three-terms free boundary does today
(`boundary_condition="three_terms"`, {doc}`/howto/free-boundary`) and where it
stops, with the measurement behind each statement. "LP QA 2.5 %" is the
Landreman-Paul QA design at 2.5 % beta with its fitted coils. Distances are
between last closed flux surfaces: point to curve, on 6 toroidal planes per
half field period. Every number on this page, with its case, its source run and
the vmex revision it ran on, is in `benchmarks/free_boundary_limits_2026-10-16.json`.

## Labelling and start independence

The deck's poloidal (θ) labelling is the reference. The unknown is the
displacement of the deck's boundary along the deck equilibrium's radial lines.

- **Start independence.** The answer does not depend on where the solve
  starts. LP QA 2.5 % at 8x8, started from a boundary perturbed by 4 cm, ends
  0.00 mm from the solve started at the deck, with the same QS to 2e-8
  relative (3.5647e-5).
- **Labelling dependence.** The answer depends on the deck's labelling at the
  truncation level, and the dependence falls with resolution. With the same
  physics and coils, a deck relabelled in θ moves the answer by:

  | modes | RMS | max |
  |---|---|---|
  | 8x8 | 0.18 mm | 0.54 mm |
  | 12x12 | 0.10 mm | 0.24 mm |

  For comparison, the resolution change alone moves the answer by:

  | modes | RMS | max |
  |---|---|---|
  | 8 to 12 | 0.88 mm | 2.2 mm |
  | 12 to 14 | 0.34 mm | 0.89 mm |

## Vacuum

In vacuum the free boundary is judged by B.n alone: there is no pressure, and
the sheet current follows from the field. Distances between last closed flux
surfaces, max / RMS:

- **QA.** NESTOR (VMEC's free boundary) agrees with DESC to 0.75-1.7 mm RMS at
  M 5-8. The three-terms surface is 0.03-0.05 mm RMS from DESC.
- **QH.** NESTOR is 7-22 mm RMS away at M 8-12, while three-terms is 0.43 mm
  RMS from DESC at M 8. DESC, not NESTOR, is the QH vacuum reference.

| case | M | three-terms to DESC | NESTOR to DESC | three-terms to NESTOR |
|---|---|---|---|---|
| QA | 5 | 0.18 / 0.05 mm | 5.4 / 1.7 mm | |
| QA | 8 | 0.08 / 0.03 mm | 2.7 / 0.75 mm | |
| QA | 12 | | | 1.2 / 0.32 mm |
| QH | 8 | 1.6 / 0.43 mm | 53 / 22 mm | |
| QH | 12 | | | 20 / 7.0 mm |

## Resolution convergence of QS

- **Vacuum.** QS (the quasisymmetry residual) converges by M = 16. On a QA
  vacuum single-stage design with 30 coils:

  | M | 12 | 14 | 16 | 18 | 20 |
  |---|---|---|---|---|---|
  | QS | 4.41e-7 | 5.30e-7 | 5.88e-7 | 5.93e-7 | 5.90e-7 |

  From 16 to 18 it changes by 0.9 %.
- **Finite beta.** QS is not yet converged at M = 14. On LP QA 2.5 %, deck
  start, ns 51:

  | M | 8 | 10 | 12 | 14 |
  |---|---|---|---|---|
  | QS | 3.56e-5 | 1.11e-4 | 2.16e-4 | 2.35e-4 |

  From 12 to 14 it changes by 9.0 %. A re-evaluation at 16 modes and ns 101
  gives the same 9.1 %. Compare finite-beta designs at one stated resolution.

## Local force error

The interior is a VMEC fixed-boundary solve, and it carries VMEC's local force
error. This error comes from the method, not from the free boundary. The
measure used here is ⟨|J×B − ∇p|⟩ on a surface over ⟨|∇p|⟩ in the volume.

- **Same as VMEC2000.** VMEX matches VMEC2000 to under 1 % on every deck
  tested.
- **LP QA 2.5 %, mpol 12, ns 51:**

  | | s = 0.26 | s = 0.5 | s = 0.76 |
  |---|---|---|---|
  | VMEX | 0.031 | 0.031 | 0.046 |
  | VMEC2000 | 0.031 | 0.031 | 0.046 |
  | DESC (L 24, M 12) | 1.4e-4 | 1.2e-4 | 1.5e-4 |

- **More radial surfaces help.** At ns 201 VMEX gives 0.011 / 0.019 / 0.030 at
  the same three surfaces.
- **The first surfaces stay large.** On the first 2 or 3 surfaces off the axis
  the error is 2 to 5 at every resolution tested.

## QH fixed-boundary solves with mpol ≥ 13

The fixed-boundary descent on the QH vacuum deck stops converging once poloidal
modes up to m = 12 are present. The runs use ns 16 / 31 / 51, ftol 1e-17 on the
last stage, and a cap of 40000 iterations.

- **Converges:** mpol 12, in about 3860 iterations, at ntor 12 or 16.
- **Stalls at fsq 3e-12 to 2e-11:** mpol 13, 14 and 16, at any ntor.

What was checked:

- **More radial surfaces:** no fix. At mpol 13 the floor is 5e-12, 1.7e-11 and
  1.4e-12 at ns 51, 101 and 151.
- **The 16/13 rational surface:** not the cause, because mpol 13 has no m = 13
  mode and still stalls.
- **The deck's boundary:** not the cause, because it has only m ≤ 5.
- **The GMRES 2-D preconditioner:** lowers the floor to 2.3e-13 once, then the
  residual stops changing.
- **VMEC2000:** stalls the same way at mpol 16.

QH vacuum at 16 modes is therefore not available at ftol 1e-17. Such a state is
reported as not converged, with its fsq.

## Redl bootstrap current

- **Quasi-axisymmetric (QA) and quasi-helically symmetric (QH) fields only.**
  The Redl model needs the field's quasisymmetry helicity: `helicity_n` must be
  0 or ±1. Other fields, quasi-isodynamic designs included, are refused, because
  they used to take the QA mapping silently (#52).
- **Quasi-symmetric part of |B|.** In the solve, Redl uses the quasi-symmetric
  part of |B|: its Boozer harmonics in θ_B − N ζ_B (#53).
- **Why.** On the solver grid the full |B| of LP QA 2.5 % at 8x8 has 2 to 8
  near-equal local maxima per surface. The gap between them is 2.1e-2 to
  2.5e-4 of B_max − B_min, so B_max, and with it the trapped fraction, jumps as
  the boundary moves. The quasi-symmetric part has one maximum on every
  surface.
- **Exact derivatives.** With the quasi-symmetric part, the bootstrap rows'
  Jacobian columns along the boundary agree with central differences to
  1.8e-8 to 8.7e-3 relative (median 2.4e-5, 33 columns) at the converged LP
  QA 2.5 % 8x8 point. With the full |B| they were 17 % to 99 % off. The
  design gradient of QS agrees with finite differences to 6e-5 to 4e-4.
- **Effect on the results.** The trapped fraction changes by −0.4 % to −1.7 %,
  and the bootstrap current dI/ds by at most 1.0 % relative. The full-field
  Redl geometry stays available as the parity check against simsopt.

## Bootstrap current: two routes

- **Production route.** Finite-beta runs, like simsopt and DESC, make the
  current profile a design variable and add a bootstrap-mismatch constraint:
  Redl for QA and QH, DKX for quasi-isodynamic fields.
- **In-solve route.** Solving Redl inside the free-boundary Newton is still
  available, but it is not the default, because it is fragile on a seed that
  is far from quasisymmetric. The QA 2.5 % Redl 12x12 seed (on main after #53)
  shows why:
  - The Jacobian is correct. Its bootstrap-row columns agree with central
    differences to a median of 1.4e-7 over 17 columns, and its interface-row
    columns to a median of 3.9e-6.
  - The coupled system is near-singular: the scaled Jacobian's condition
    number is 1.6e7.
  - The first Newton step lands where the interface rows are nonlinear. At
    the full step, those rows are 3.2e-3 against a linear prediction of 0. At
    1/16 of the step they still follow the model to 2.7e-5.

## Unconverged states raise

`opt.solve_equilibrium` raises `VmecConvergenceError` when the final stage of
its ladder stops at the iteration cap short of its tolerance (#50). It used to
return the unconverged state silently. `raise_on_max_iterations=False` is the
explicit opt-out for a caller that checks `result` itself.

## The boundary's band, and continuation where Newton stalls

- **Band.** The boundary carries the interior's band, max(mpol − 1, ntor), in
  vacuum and at finite beta (#54). Before #54 it had two fewer modes at finite
  beta. At 12x12 the band-limited B.n / pressure jump / sheet current improve
  from band 10 to band 12:
  - QH 2.5 %: 7.02 / 5.48 / 7.02e-4 to 5.74 / 3.48 / 5.69e-4.
  - QI 2.5 %: 2.497e-3 / 4.29e-4 / 1.82e-3 to 2.336e-3 / 3.18e-4 / 1.70e-3.

  The larger band (288 boundary coordinates instead of 221) costs 13-32 % more
  wall time at finite beta.
- **Continuation.** Newton runs on the full band first, so a start in its basin
  costs what it did before. If Newton stalls, the band is built up from
  m ≤ 1, one mode at a time, each band warm-started from the previous one.
  - A stall on the way returns `converged=False`, and `band` gives the last
    band that converged. A stall is never returned as an answer.
  - The band used is reported as `fit.band`.
- **No trust region.** Continuation replaces the trust region of #51, chosen by
  measurement on QA vacuum 12x12 far starts. From 2 cm, continuation reaches
  the deck's answer to 3e-12 m:

  | start | continuation | trust region (#51) |
  |---|---|---|
  | 2 cm | 1145 s | 4134 s |
  | 4 cm | 1103 s | not converged after 5240 s |

- **Open case.** QH 1 % at 12x12 from the deck does not converge on any route.
  Continuation reaches band 8 and stalls at band 9, at cost 1.4e-8. The result
  is reported as `converged=False` with band 8.
