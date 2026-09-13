> Archived context, 13 September 2026. The top-level handoff gives current status; older job statements below are historical. Machine locations were removed; unbundled raw data are not implied to be available.

# Bounded QA edge-grid refinement protocol

The next comparison should test whether the pressure-edge return-current structure survives spatial refinement. Native HINT does **not** support changing grid dimensions through its maintained restart reader: read_eq_field.f90:279 (source/evidence reference; see handoff index) aborts on mismatched R/Z/toroidal counts, and later checks reject different field periods and box extents. The HDF5 and binary paths have equivalent checks. No native restart interpolation was found in this reader. Do not relabel a coarse restart or introduce an ad hoc field remap.

## Prepared pair and scope

Use fresh initialization from the same original beta0p5 WOUT, coil data, pressure/current profiles and numerical 12 cm wall, at 64×64×32 and 128×128×64. This compares the complete spatial discretization, including toroidal resolution and native flux/limiter preparation; it does not isolate R/Z order. It is an early-transient edge test and does not extend or reproduce the existing t=1.08 trajectories.

Exact decks and hashes are in preparation-manifest.json (see bundled evidence index):

- 64-grid HINT input (see bundled evidence index).
- 128-grid HINT input (see bundled evidence index).
- 128-grid native limiter input (see bundled evidence index).

Both use four toroidal MPI ranks, OMP=1, four outer steps, 250 magnetic steps per outer step, dt=1e-5, nsave=1, npchg=1 and no pressure resets. Total magnetic time is 0.01. Resistivity remains eta0=0.001, tracing h=0.01 and lc=20. The smaller common timestep provides margin for the refined grid; its stability is not certified before execution. The same timestep on both grids avoids conflating a grid change with a timestep change. The one-step pressure ramp is common to both; this pair has different startup controls from the earlier eight-step pilot and must be labeled accordingly.

## Input readiness

Remote validation host already holds validated scalar box extents and field periods for both native flux maps and coil grids. Their SHA-256 hashes and exact paths are recorded in the manifest. The 128 vacuum file is 25,181,350 bytes and native flux file 8,399,749 bytes. Both have the same physical box as their coarse counterparts. These grids refine samples of the same source constraints, not an interpolation of a relaxed HINT field.

The 64 limiter exists. A 128 limiter was not present in the inspected input directories; its native MKLIM deck is prepared but **has not been executed**. Before HINT:

1. Create the proposed fresh run directories and link read-only input files from the manifest rather than duplicating the vacuum arrays.
2. Generate the 128 limiter with maintained native MKLIM using the same wall-12cm.dat, then hash it. Allow at most 120 seconds for this preprocessing.
3. Verify identical coordinate extents/field periods and the expected dimensions across vacuum, flux and limiter files; check all initial s<1 cells lie inside the corresponding wall mask. Preserve the earlier continuous wall/coil-clearance evidence; a finer binary mask is not a changed wall model.
4. Confirm the input profile and target-current hashes, native executable hash and linked-library environment. Preserve the source/binary used for both runs.
5. Recheck task-owned process locks, free memory and disk immediately before execution. No solver or preprocessing job was launched during this preparation.

## Cost and stop conditions

The fresh fine grid has eight times as many cells. The earlier coarse pilot spent about 164 seconds tracing eight outer steps; simple cell/trace scaling suggests roughly 650 seconds of fine tracing for four steps, plus magnetic evolution. This is a planning estimate, not a measured fine-grid runtime or promise. Different startup pressure support can change it substantially.

Use the existing bounded runner sequentially, one heavy remote validation host job at a time. Cap the coarse pair member at 180 seconds and the fine member at 1200 seconds. A timeout is a capped pilot, not convergence; preserve matched completed snapshot times and do not extend the budget automatically. If the fine member cannot complete the same interval, report the comparison as incomplete and choose a smaller common interval explicitly.

Remote validation host showed approximately 6.6 GiB free at inspection. Budget at most 0.5 GiB of additional disk for input metadata, up to five native field snapshots per grid and compact diagnostics; verify actual file sizes while running. Seven double-precision arrays at 128×128×64 require about 56 MiB per snapshot before format overhead. Reuse immutable vacuum/flux files, avoid duplicate raw final fields, and stop before consuming the reserved disk margin. Do not delete unrelated data.

Fine-grid peak RSS has not been measured. Watch the process group against a provisional 16 GiB stop threshold, not a predicted requirement. The runner timeout does not enforce an RSS cap; the execution owner must inspect memory or use an appropriate existing monitor. The machine has sufficient nominal memory, but contemporaneous jobs determine availability.

## Diagnostic and acceptance sequence

At matched saved times, compare pressure integral/peak, attained current on multiple supports, response divergence and physical force ratios with the same definitions. Apply the reviewed return-current shell and Stokes diagnostics. Use fixed physical-distance bins (1–2, 2–4, 4–8, 8–16 cm) and report the number of grid cells spanning each shell; do not compare a one-cell-wide coarse feature to a one-cell-wide fine feature as equal physical width.

Compare sections at the same phi and native field values at the fixed Cartesian target set. For pressure/field norms requiring common coordinates, use the already reviewed native samplers or an explicitly validated interpolation path; simple array subtraction is invalid across different grids. The existing cadence comparison helper expects identical coordinates and must not be used unchanged for this pair.

Separate these questions:

- Does the return-current band remain at the same physical location and width relative to pressure and wall, or follow grid spacing?
- Does the cumulative signed-current distribution agree at fixed physical contours, with Stokes quadrature error smaller than the coarse/fine difference?
- Does the positive-pressure current change because of actual current density or changing membership of a pressure threshold?
- Do native pressure labels and the approximate snapshot drive reconstruction tell the same qualitative story, with the latter's limitations retained?

This pair can identify strong grid sensitivity; two resolutions do not establish an asymptotic order. Absence of a visible difference at t=0.01 does not certify the mature t=1.08 return-current shell. If startup differences are controlled and the fine run is affordable, continue each run on its **own native grid** to a common later time under a separately bounded protocol. Only then compare with the original mature-transient question.

A future remap would require a separately reviewed vector-field transfer, preservation of prescribed vacuum decomposition, coordinate/component conventions, boundary conditions, and measured changes in divergence, current, pressure and magnetic energy before/after transfer. It is not part of this prepared pair. The original scientific equilibrium, wall-sensitivity and topology gates remain unchanged.

Prepared 13 September 2026. Inputs are ready for review and the listed preprocessing; no execution or convergence is claimed.

## Reviewed preparation and launch checkpoint

Native 128-grid MKLIM subsequently completed under the remote validation host CPU lock in 4.78 seconds. Its 11,867,940-byte output has SHA-256 `8ec31cdda817d21965db6f9cf65c76fe0e3838138dbf373b88bae467ffe5b13f`; all 89,754 initial plasma cells are inside its 288,880-cell wall mask. Grid dimensions, domain and field period match the native flux and vacuum files exactly. The limiter validation (see bundled evidence index) and preparation manifest record the completed gate.

The execution owner reviewed and authorized the four-step pair after this preparation. Four outer steps retain three intervals after startup, whereas two steps provide only one such interval; reducing to two would be a cost/stability probe, with even weaker evidence of developing edge structure. Neither choice answers the mature t=1.08 comparison without subsequent matched-time continuation. Four steps are retained within the reviewed 1200-second fine cap; no fixed 600-second policy ceiling applies here.

The reviewed launcher (see bundled evidence index) validates source hashes and containment again, runs the coarse member first, and does not start the fine member after a coarse failure. It samples task-descendant RSS and newly written pair data every five seconds, interrupting the bounded runner if the 16 GiB or 0.5 GiB limits are exceeded. Its progress/status files, rather than this checkpoint, establish the current execution state. The launch is in progress at this update; no successful run or scientific acceptance is assumed.
