"""Solved two-coil free-boundary mirror beta scan and physics plots.

Run from the repository root with::

    python examples/mirror_free_boundary_beta_scan.py

The first four points are the 0--10% validation scan.  The last two expose
the higher-beta diamagnetic trend.  Every curve and surface comes from a
coupled plasma-boundary-vacuum equilibrium solve with residual tolerance
``FTOL``; no prescribed finite-beta boundary is plotted.
"""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax.mirror import (  # noqa: E402
    BiMaxwellianPressureClosure,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    TabulatedPressureClosure,
    build_vacuum_grid,
    mout_from_result,
    plot_mout,
    write_mout,
)
from vmec_jax.mirror.free_boundary import solve_axisymmetric_beta_scan_cli  # noqa: E402
from vmec_jax.mirror.diagnostics import (  # noqa: E402
    summarize_axisymmetric_beta_scan,
)
from vmec_jax.mirror.output import (  # noqa: E402
    FreeBoundaryRestart,
    load_free_boundary_restart,
    save_free_boundary_restart,
)

# Inputs: edit these values, then run the file directly.
BETAS = np.asarray([0.0, 0.01, 0.03, 0.10, 0.25, 0.50])
PRESSURE_MODEL = "isotropic"  # "bi_maxwellian" or "tabulated"
HOT_FRACTION = 0.2
TEMPERATURE_RATIO = 0.7
NS = 7
NXI = 13
NRHO = 7
VACUUM_BACKEND = "exterior"  # "annulus" retains a finite outer cylinder
EXTERIOR_NTHETA = 12
EXTERIOR_ORDER = 8
EXTERIOR_SPECTRAL_SIDE_DENSITY = False  # More accurate, but costlier.
EXTERIOR_JACOBIAN_CHUNK_SIZE = 6
FTOL = 1.0e-12
MAX_ITERATIONS = 2000
Z_MIN, Z_MAX = -0.8, 0.8
COIL_RADIUS = 0.9
COIL_SEPARATION = 2.0
COIL_CURRENT = 2.0e5
CENTER_RADIUS = 0.25
OUTER_RADIUS = 0.65
OUTPUT_DIR = Path(f"results/mirror_free_boundary_beta_scan_{VACUUM_BACKEND}")
SAVE_RESTARTS = True
RESTART_FROM = None  # e.g. OUTPUT_DIR / "beta_003p0pct.npz"; then trim BETAS
PLEIADES_REFERENCE = Path(__file__).resolve().parent / "data" / "pleiades_two_coil_beta_reference.csv"

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

coil_dofs = np.zeros((2, 3, 3))
coil_dofs[:, 0, 2] = COIL_RADIUS
coil_dofs[:, 1, 1] = COIL_RADIUS
coil_dofs[:, 2, 0] = np.asarray([-0.5, 0.5]) * COIL_SEPARATION
try:
    from essos.coils import Coils, Curves
    from essos.fields import BiotSavart
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "This example requires ESSOS: pip install -e /path/to/ESSOS"
    ) from error

coils = Coils(
    Curves(jnp.asarray(coil_dofs), n_segments=128, nfp=1, stellsym=False),
    jnp.full(2, COIL_CURRENT),
)
biot_savart = BiotSavart(coils)


def external_field(points):
    """Evaluate the ESSOS field on an arbitrary array of Cartesian points."""

    points = jnp.asarray(points)
    return jax.vmap(biot_savart.B)(points.reshape(-1, 3)).reshape(points.shape)

config = MirrorConfig(
    resolution=MirrorResolution(ns=NS, mpol=0, ntheta=1, nxi=NXI),
    z_min=Z_MIN,
    z_max=Z_MAX,
    ftol=FTOL,
    max_iterations=MAX_ITERATIONS,
)
grid = config.build_grid()
vacuum_grid = build_vacuum_grid(grid, nrho=NRHO)
initial_restart = None if RESTART_FROM is None else load_free_boundary_restart(RESTART_FROM, grid, vacuum_grid)
z = jnp.asarray(grid.z)
coil_z = 0.5 * COIL_SEPARATION
vacuum_axis_field = sum(
    4.0e-7
    * jnp.pi
    * COIL_CURRENT
    * COIL_RADIUS**2
    / (2.0 * (COIL_RADIUS**2 + (z - position) ** 2) ** 1.5)
    for position in (-coil_z, coil_z)
)
center = int(np.argmin(np.abs(grid.z)))
axial_flux_derivative = 0.5 * vacuum_axis_field[center] * CENTER_RADIUS**2
initial_boundary = MirrorBoundary.from_axis_field(
    axial_flux_derivative,
    vacuum_axis_field,
    grid,
)
pressure_closure = None
if PRESSURE_MODEL in {"bi_maxwellian", "tabulated"}:
    bi_maxwellian = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([1.0, -1.0]),
        hot_fraction_coefficients=jnp.asarray([HOT_FRACTION]),
        temperature_ratio=TEMPERATURE_RATIO,
        critical_field=float(vacuum_axis_field[center]),
        gamma=0.0,
    )
    if PRESSURE_MODEL == "bi_maxwellian":
        pressure_closure = bi_maxwellian
    else:
        s_nodes = jnp.linspace(0.0, 1.0, 5)
        b_nodes = jnp.linspace(0.4 * vacuum_axis_field[center], 2.0 * vacuum_axis_field[center], 9)
        pressure_closure = TabulatedPressureClosure(
            s_nodes,
            b_nodes,
            bi_maxwellian.parallel_pressure(s_nodes[:, None], b_nodes[None, :]),
            gamma=0.0,
        )
elif PRESSURE_MODEL != "isotropic":
    raise ValueError("PRESSURE_MODEL must be 'isotropic', 'bi_maxwellian', or 'tabulated'")

print(f"Solving {BETAS.size} beta points at ns={NS}, nxi={NXI}, vacuum={VACUUM_BACKEND}, ftol={FTOL:.0e}")
results = solve_axisymmetric_beta_scan_cli(
    initial_boundary,
    grid,
    vacuum_grid,
    config,
    external_field,
    jnp.asarray(BETAS),
    outer_radius=OUTER_RADIUS,
    axial_flux_derivative=axial_flux_derivative,
    reference_field=float(vacuum_axis_field[center]),
    initial_restart=initial_restart,
    pressure_closure=pressure_closure,
    vacuum_backend=VACUUM_BACKEND,
    exterior_ntheta=EXTERIOR_NTHETA,
    exterior_order=EXTERIOR_ORDER,
    exterior_spectral_side_density=EXTERIOR_SPECTRAL_SIDE_DENSITY,
    exterior_jacobian_chunk_size=EXTERIOR_JACOBIAN_CHUNK_SIZE,
)
gamma = np.asarray(coils.gamma)
if SAVE_RESTARTS:
    for beta, result in zip(BETAS, results, strict=True):
        label = f"beta_{100 * beta:05.1f}pct".replace(".", "p")
        save_free_boundary_restart(OUTPUT_DIR / label, FreeBoundaryRestart.from_result(result))
for beta, result in zip(BETAS, results, strict=True):
    label = f"beta_{100 * beta:05.1f}pct".replace(".", "p")
    if pressure_closure is None:
        parallel_pressure = result.perpendicular_pressure
    else:
        field_strength = jnp.sqrt(result.plasma_b_squared)
        parallel_pressure = result.mass_scale * pressure_closure.moments(
            jnp.asarray(grid.s)[:, None, None], field_strength
        ).parallel
    write_mout(
        OUTPUT_DIR / f"mout_mirror_{label}.nc",
        mout_from_result(
            result,
            grid,
            config,
            axial_flux_derivative=axial_flux_derivative,
            parallel_pressure=parallel_pressure,
            coil_xyz=gamma,
            closure=PRESSURE_MODEL,
        ),
    )
diagnostics = summarize_axisymmetric_beta_scan(
    results,
    jnp.asarray(BETAS),
    grid,
    reference_field=float(vacuum_axis_field[center]),
)

summary = np.asarray(
    [
        [
            float(item.requested_beta),
            float(item.achieved_reference_beta),
            float(item.volume_averaged_beta),
            float(item.center_radius),
            float(item.center_axis_field),
            float(item.diamagnetic_field_ratio),
            float(item.paraxial_field_ratio),
            float(item.paraxial_relative_error),
            float(result.variational_max),
            float(result.plasma_staggered_weak_force.maximum),
            float(result.plasma_force.normalized_rms),
            float(result.normalized_divergence_rms),
            float(result.interface.normal_stress_rms),
            float(result.interface.vacuum_b_normal_rms),
            float(result.mass_scale),
            float(result.iterations),
            float(result.vacuum_field.neumann_result.compatibility_error) if VACUUM_BACKEND == "exterior" else np.nan,
            float(result.vacuum_field.neumann_result.condition_number) if VACUUM_BACKEND == "exterior" else np.nan,
        ]
        for item, result in zip(diagnostics, results, strict=True)
    ]
)
header = (
    "requested_beta,achieved_reference_beta,volume_averaged_beta,center_radius_m,"
    "center_axis_field_T,diamagnetic_field_ratio,paraxial_field_ratio,"
    "paraxial_relative_error,variational_max,staggered_weak_max,pointwise_force_rms,"
    "normalized_divergence_rms,normal_stress_rms,bnormal_rms_normalized,"
    "mass_scale,iterations,exterior_compatibility,exterior_condition_number"
)
np.savetxt(OUTPUT_DIR / "beta_scan.csv", summary, delimiter=",", header=header, comments="")

colors_beta = plt.cm.viridis(np.linspace(0.08, 0.92, BETAS.size))
display_xi = np.linspace(-1.0, 1.0, 121)
display_matrix = np.asarray(grid.axial_basis.interpolation_matrix(display_xi))
z = 0.5 * (Z_MIN + Z_MAX) + 0.5 * (Z_MAX - Z_MIN) * display_xi
fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.4), constrained_layout=True)
for beta, result, color in zip(BETAS, results, colors_beta, strict=True):
    radius = display_matrix @ np.asarray(result.boundary.radius_scale[0])
    b_axis = display_matrix @ np.sqrt(np.asarray(result.plasma_b_squared[0, 0]))
    b_lcfs = display_matrix @ np.sqrt(np.asarray(result.plasma_b_squared[-1, 0]))
    axes[0, 0].plot(z, radius, color=color, lw=2, label=f"{100 * beta:g}%")
    baseline_radius = display_matrix @ np.asarray(results[0].boundary.radius_scale[0])
    axes[0, 1].plot(z, 1e3 * (radius - baseline_radius), color=color, lw=2)
    axes[0, 2].plot(z, b_axis, color=color, lw=2)
    axes[1, 0].plot(z, b_lcfs, color=color, lw=2)
axes[0, 2].plot(z, display_matrix @ np.asarray(vacuum_axis_field), "k--", lw=1.6, label="analytic coil vacuum")
axes[0, 0].set(title="Solved LCFS", xlabel="Axial position z [m]", ylabel="Radius [m]")
axes[0, 0].legend(title="Requested beta", ncol=2, fontsize=8)
axes[0, 1].set(title="LCFS displacement from beta=0", xlabel="Axial position z [m]", ylabel="Delta radius [mm]")
axes[0, 2].set(title="On-axis |B|", xlabel="Axial position z [m]", ylabel="Magnetic field [T]")
axes[0, 2].legend(fontsize=8)
axes[1, 0].set(title="LCFS |B|", xlabel="Axial position z [m]", ylabel="Magnetic field [T]")

achieved = summary[:, 1]
axes[1, 1].plot(100 * achieved, summary[:, 5], "o-", color="#0072B2", lw=2, label="coupled solve")
axes[1, 1].plot(100 * achieved, summary[:, 6], "--", color="#D55E00", lw=2, label=r"$\sqrt{1-\beta}$")
pleiades = np.genfromtxt(PLEIADES_REFERENCE, delimiter=",", names=True, comments="#", skip_header=4)
pleiades = pleiades[pleiades["nr"] == np.max(pleiades["nr"])]
axes[1, 1].plot(
    100 * np.concatenate([[0.0], pleiades["beta"]]),
    np.concatenate([[1.0], pleiades["field_ratio"]]),
    "s:",
    color="#009E73",
    lw=1.8,
    label="Pleiades 51x101",
)
axes[1, 1].set(title="Central diamagnetic response", xlabel="Achieved central beta [%]", ylabel=r"$B(\beta)/B(0)$")
axes[1, 1].legend(fontsize=8)

for beta, result, color in zip(BETAS, results, colors_beta, strict=True):
    history = np.asarray(result.history)
    axes[1, 2].semilogy(history[:, 0], np.maximum(history[:, -1], 1e-18), color=color, lw=1.5, label=f"{100 * beta:g}%")
axes[1, 2].axhline(FTOL, color="0.25", ls="--", lw=1, label="ftol")
axes[1, 2].set(title="Coupled residual history", xlabel="Residual evaluation", ylabel="Maximum normalized residual")
axes[1, 2].legend(ncol=2, fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=0.22)
fig.savefig(OUTPUT_DIR / "beta_scan_diagnostics.png", dpi=180)
plt.close(fig)

middle_beta = min(0.10, 0.5 * float(BETAS[-1]))
display_indices = sorted({0, int(np.argmin(np.abs(BETAS - middle_beta))), len(BETAS) - 1})
fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
radial_coordinate = np.sqrt(np.asarray(grid.s))
for index in display_indices:
    result = results[index]
    pressure = np.asarray(result.perpendicular_pressure[:, 0, center])
    magnetic_pressure = np.asarray(result.plasma_b_squared[:, 0, center]) / (2.0 * 4.0e-7 * np.pi)
    color = colors_beta[index]
    label = f"{100 * BETAS[index]:g}%"
    axes[0].plot(radial_coordinate, pressure / 1e3, "o-", color=color, lw=2, label=label)
    if pressure_closure is not None and BETAS[index] > 0.0:
        field_strength = np.sqrt(np.asarray(result.plasma_b_squared[:, 0, center]))
        parallel = (
            result.mass_scale * pressure_closure.moments(jnp.asarray(grid.s), jnp.asarray(field_strength)).parallel
        )
        axes[0].plot(radial_coordinate, np.asarray(parallel) / 1e3, "--", color=color, lw=1.5)
    axes[1].plot(radial_coordinate, magnetic_pressure / 1e3, "o-", color=color, lw=2, label=label)
    axes[1].plot(radial_coordinate, (pressure + magnetic_pressure) / 1e3, "--", color=color, lw=1.5)
axes[0].set(title=r"Midplane $p_\perp$", xlabel=r"Normalized radius $\sqrt{s}$", ylabel="Pressure [kPa]")
axes[1].set(title="Midplane pressure balance", xlabel=r"Normalized radius $\sqrt{s}$", ylabel="Pressure [kPa]")
beta_legend = axes[0].legend(title="Requested beta", fontsize=8)
if pressure_closure is not None:
    axes[0].add_artist(beta_legend)
    axes[0].legend(
        handles=[
            Line2D([], [], color="0.2", lw=1.8, label=r"$p_\perp$"),
            Line2D([], [], color="0.2", lw=1.5, ls="--", label=r"$p_\parallel$"),
        ],
        loc="lower left",
        fontsize=8,
    )
axes[1].legend(["magnetic", "total"], fontsize=8)
axes[2].plot(100 * BETAS, 100 * summary[:, 1], "o-", color="#0072B2", lw=2, label="achieved central")
axes[2].plot(100 * BETAS, 100 * summary[:, 2], "s-", color="#D55E00", lw=2, label="volume averaged")
axes[2].plot(100 * BETAS, 100 * BETAS, "k--", lw=1.2, label="requested")
axes[2].set(title="Beta definitions", xlabel="Requested beta [%]", ylabel="Solved beta [%]")
axes[2].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=0.22)
fig.savefig(OUTPUT_DIR / "beta_scan_pressure.png", dpi=180)
plt.close(fig)

theta = np.linspace(0.0, 2.0 * np.pi, 73)
surface_fields = [
    display_matrix @ np.sqrt(np.asarray(results[index].plasma_b_squared[-1, 0])) for index in display_indices
]
field_norm = colors.Normalize(
    min(np.min(value) for value in surface_fields), max(np.max(value) for value in surface_fields)
)
fig = plt.figure(figsize=(15, 4.8), constrained_layout=True)
for panel, index in enumerate(display_indices, start=1):
    result = results[index]
    radius = display_matrix @ np.asarray(result.boundary.radius_scale[0])
    zz, tt = np.meshgrid(z, theta)
    rr = np.broadcast_to(radius, zz.shape)
    ax = fig.add_subplot(1, 3, panel, projection="3d")
    ax.plot_surface(
        zz,
        rr * np.cos(tt),
        rr * np.sin(tt),
        facecolors=plt.cm.viridis(field_norm(np.broadcast_to(surface_fields[panel - 1], zz.shape))),
        rstride=2,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.78,
    )
    for curve in gamma:
        closed = np.vstack([curve, curve[0]])
        ax.plot(closed[:, 2], closed[:, 0], closed[:, 1], color="#C44E52", lw=2.0)
    for angle in np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False):
        line_x = 1.05 * radius * np.cos(angle)
        line_y = 1.05 * radius * np.sin(angle)
        ax.plot(z, line_x, line_y, color="black", lw=3.4, alpha=0.9)
        ax.plot(z, line_x, line_y, color="#00BFC4", lw=1.7, alpha=1.0)
    arrow_indices = np.arange(12, z.size - 12, 24)
    radial_slope = np.gradient(radius, z)
    arrow_norm = np.sqrt(1.0 + radial_slope[arrow_indices] ** 2)
    for angle in (0.0, np.pi):
        ax.quiver(
            z[arrow_indices],
            1.06 * radius[arrow_indices] * np.cos(angle),
            1.06 * radius[arrow_indices] * np.sin(angle),
            0.12 / arrow_norm,
            0.12 * radial_slope[arrow_indices] * np.cos(angle) / arrow_norm,
            0.12 * radial_slope[arrow_indices] * np.sin(angle) / arrow_norm,
            color="#111111",
            linewidth=1.0,
            arrow_length_ratio=0.32,
        )
    ax.set(title=f"beta={100 * BETAS[index]:g}%", xlabel="z [m]", ylabel="x [m]", zlabel="y [m]")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_box_aspect((2.3, 2.0, 2.0))
    ax.view_init(elev=23, azim=-56)
colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=field_norm, cmap="viridis"), ax=fig.axes, shrink=0.72, pad=0.03)
colorbar.set_label("LCFS |B| [T]")
fig.savefig(OUTPUT_DIR / "beta_scan_3d.png", dpi=180)
plt.close(fig)
endpoint_label = f"beta_{100 * BETAS[-1]:05.1f}pct".replace(".", "p")
plot_mout(
    OUTPUT_DIR / f"mout_mirror_{endpoint_label}.nc",
    OUTPUT_DIR,
    name="mirror_endpoint",
)

np.set_printoptions(precision=6, suppress=False)
print(header)
print(summary)
print(f"Wrote {OUTPUT_DIR / 'beta_scan_diagnostics.png'}")
print(f"Wrote {OUTPUT_DIR / 'beta_scan_pressure.png'}")
print(f"Wrote {OUTPUT_DIR / 'beta_scan_3d.png'}")
