# Cross-code benchmarks

Two records sit behind this page. `benchmarks/regression_matrix.json` is a
VMEX-only release check; the DESC comparison below scores VMEX, VMEC2000 and
VMEC++ under one metric.

## Release regression matrix

`python benchmarks/regression_matrix.py` solves eight decks from
`examples/data` as written (tokamak, vacuum QA and QH, finite-beta QH,
non-symmetric, free boundary) and writes ns, mpol, ntor, iterations, wall time,
residuals, `wb`, beta, axis R0 and iota at ρ = 0.1, 0.5, 1. With
`--against OLD.json` it exits non-zero when a physics scalar moves by more
than 1e-6 relative. Timings are reported, not gated. Two decks stop at their
`NITER` cap as written: `LandremanPaul2021_QA_lowres` at a residual of 2.6e-13
and the small free-boundary deck `cth_like_free_bdy_lasym_small` at 7e-2;
both are kept because the check is for drift, not convergence.

## DESC comparison protocol

Panici et al. (2023), *J. Plasma Phys.* 89, 955890303, compare equilibrium
codes by loading each output into DESC and evaluating one force error. We use
their normalized error

$$ f = \langle |\mathbf{J}\times\mathbf{B} - \nabla p| \rangle_{\mathrm{vol}} \,/\, \langle |\nabla p| \rangle_{\mathrm{vol}}, $$

computed by DESC on every `wout` read through `desc.vmec.VMECIO` (radial
fit order L = 2(mpol − 1), lambda moved to the full mesh), over
0.2 ≤ ρ ≤ 0.9 to keep the axis and edge fits out.

**Force error against time (DSHAPE, ns = 51).** mpol 4 to 12, 8 cores each.

| mpol | VMEX f | VMEX warm [s] | VMEC2000 [s] | VMEC++ [s] | DESC f | DESC warm [s] |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.86 | 0.34 | 3.0 | 1.8 | 2.4e-2 | 2.5 |
| 6 | 0.13 | 0.43 | 2.6 | 10.6 | 3.3e-3 | 3.8 |
| 8 | 0.080 | 0.94 | 9.9 | 2.2 | 2.3e-4 | 12.9 |
| 10 | 0.049 | 0.65 | 6.7 | 1.7 | 1.4e-5 | 168 |
| 12 | 0.046 | 0.86 | 3.8 | 0.27 | — | > 1500 |

The three VMEC-type codes give the same f to four digits, as they solve the
same discretization. Their error is set by the finite-difference radial grid
and the spectral fit in DESC, so it falls slowly with mpol and does not fall
with ns; DESC's spectral solution reaches 1e-5. This is the result Panici et
al. report. The host was shared (load 15 to 24), so single timings scatter
by up to 10x.

On the 3-D QA deck below, f is 0.76, 0.99 and 1.37 at ns = 65, 129 and 257 for
all three VMEC-type codes: the VMECIO fit of a bootstrap-current equilibrium
dominates, so this metric does not rank them in 3-D. DESC 0.17.3 did not
converge on this deck with automatic continuation.

**Near-axis iota (`input.LandremanPaul2021_QA_beta2p5_bootstrap`).** mpol 7,
ntor 6, ftol 1e-12, one core each.

| ns | VMEX | VMEC2000 | VMEC++ | VMEX warm [s] |
|---:|---:|---:|---:|---:|
| 65 | 0.18806 | 0.18806 | 0.18804 | 33 |
| 129 | 0.17689 | 0.17689 | 0.17678 | 132 |
| 257 | 0.16688 | 0.16688 | 0.16682 | 380 |

The successive differences give an observed order p = 0.16 in h = 1/ns
(0.18 for VMEC++): ι(0.1) is not in its asymptotic range at these ns.
VMEX matches VMEC2000 to five digits. The DESC value of this deck, 0.166, is
reached only at ns = 257.

## Versions and limits

VMEX 0.11.10 on JAX 0.11.2; DESC 0.17.3 on JAX 0.9.2 (DESC requires
`jax<0.10`); VMEC++ 0.9.1; VMEC2000 at hiddenSymmetries `70c637a`. GVEC 1.5.0
writes no `wout`, so it is not scored. The cross-code driver is out of tree.
