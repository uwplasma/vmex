"""Accepted-step measurements for the fixed-boundary scalar reference."""
import csv
from pathlib import Path
import time

import numpy as np


class StepHistory:
    """Step 0 is the fitted-coil start; every later row is an accepted iterate.

    Step size means the norm of the actual accepted increment, not the 0.05
    coordinate scale. Coil displacement compares corresponding filament nodes
    in metres. These are sampled displacements, not Hausdorff distances.
    """

    def __init__(self, output):
        self.output = Path(output)
        self.rows = []
        self.previous = None
        self.initial_gamma = None
        self.started = time.perf_counter()

    def record(self, u, coils, surface, *, objective, qa, aspect, minimum_iota,
               aspect_target=5.0, iota_floor=0.19, coil_step=0.05, **extra):
        import jax
        from essos.fields import BiotSavart

        u = np.asarray(u, dtype=float)
        dofs = np.asarray(coils.curves.dofs).ravel()
        raw = np.asarray(coils.curves.dofs) / np.asarray(coils.curves.scaling)
        gamma = np.asarray(coils.gamma)
        currents = np.asarray(coils.dofs_currents_raw)
        current = dict(u=u.copy(), dofs=dofs.copy(), raw=raw.copy(),
                       gamma=gamma.copy(), currents=currents.copy())
        previous = current if self.previous is None else self.previous
        if self.initial_gamma is None:
            self.initial_gamma = gamma.copy()
        du = u - previous["u"]
        coil_du = (dofs - previous["dofs"]) / coil_step
        displacement = np.linalg.norm(gamma - previous["gamma"], axis=-1)
        cumulative = np.linalg.norm(gamma - self.initial_gamma, axis=-1)
        field = np.asarray(jax.vmap(BiotSavart(coils).B)(surface.gamma.reshape(-1, 3)))
        field = field.reshape(surface.gamma.shape)
        bn = np.sum(field * np.asarray(surface.unitnormal), axis=-1) / np.linalg.norm(field, axis=-1)
        area = np.asarray(surface.area_element)
        row = dict(step=len(self.rows), objective=float(objective), qa=float(qa),
            qa_residual_l2=float(np.sqrt(qa)), aspect=float(aspect),
            aspect_error=float(aspect-aspect_target), min_abs_iota=float(minimum_iota),
            iota_violation=float(max(iota_floor-minimum_iota, 0)),
            step_u_l2=float(np.linalg.norm(du)), step_u_linf=float(np.max(np.abs(du))),
            coil_step_u_l2=float(np.linalg.norm(coil_du)),
            coil_step_u_linf=float(np.max(np.abs(coil_du))),
            coil_coefficient_step_l2_m=float(np.linalg.norm(raw-previous["raw"])),
            coil_displacement_rms_m=float(np.sqrt(np.mean(displacement**2))),
            coil_displacement_max_m=float(displacement.max()),
            coil_displacement_from_start_max_m=float(cumulative.max()),
            current_step_max_A=float(np.max(np.abs(currents-previous["currents"]))),
            normal_field_rms=float(np.sqrt(np.sum(area*bn**2)/area.sum())),
            normal_field_max=float(np.max(np.abs(bn))),
            elapsed_seconds=time.perf_counter()-self.started, **extra)
        if not all(np.isfinite(value) for value in row.values()):
            raise ValueError("nonfinite accepted-step diagnostic")
        self.rows.append(row)
        with (self.output / "accepted_steps.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)
        np.savez_compressed(self.output / f"step_{row['step']:04d}.npz", **current)
        self.previous = current
        print(f"[accepted {row['step']}] QA={qa:.8e}, |du|={row['step_u_l2']:.4e}, "
              f"|du_coil|={row['coil_step_u_l2']:.4e}, "
              f"coil motion max={1e3*row['coil_displacement_max_m']:.4f} mm", flush=True)
        return row
