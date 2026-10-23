"""Render the saved li383 README diagnostics. Author: rogeriojorge.

Requires NumPy, Matplotlib, and Pillow (Matplotlib's image backend).
From the repository root:
    python docs/_static/figures/sources/make_cyclometry_figures.py
    python docs/_static/figures/sources/make_cyclometry_figures.py --format png pdf

Only replots compact saved data from benchmarks/li383_readme_sections.npz.
It does not rerun ESSOS, regenerate equilibria, or test orbit convergence.
Embedded metadata_json records settings, numerical diagnostics, source hashes,
and repository revisions. No pickle loading is needed.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

PHASE_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
                '#6A51A3', '#A6761D', '#008F95', '#AF3150']

def render(sections, markers, radii, chi, hamiltonian, action_meta, energy_ev):
    """Create a figure using only supplied saved plotting arrays."""
    style = {
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelsize': 11, 'axes.linewidth': .7,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.width': .7, 'ytick.major.width': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharex=True, sharey=True)
        fig.subplots_adjust(left=.080, right=.985, bottom=.235, top=.975, wspace=.065)
        chi_periodic = np.r_[chi, chi[0] + 2 * np.pi]
        for index, sign in enumerate((1., -1.)):
            ax = axes[index]
            h = hamiltonian[index]
            h_periodic = np.concatenate([h, h[:, :1]], axis=1)
            levels = np.linspace(float(np.min(h)), min(float(np.max(h)),
                float(np.min(h)) + 3 * action_meta['root_potential_oscillation']), 14)
            ax.contour(chi_periodic, radii, h_periodic, levels=levels,
                       colors='#B4B4B4', linewidths=.65, linestyles='solid', zorder=1)
            ax.axhline(action_meta['rational_root'], color='#242424',
                       dashes=(5, 3), linewidth=1.05, zorder=2)
            ax.axhline(action_meta['centers'][index], color='#B74F12',
                       dashes=(1, 2), linewidth=1.25, zorder=2)
            for marker in markers:
                if marker['sign'] != sign:
                    continue
                sec = sections[sections[:, 0] == marker['id']]
                phase_index = int(np.rint(marker['birth_chi'] / (np.pi / 4))) % 8
                ax.scatter(sec[:, 1], sec[:, 2], s=6.5, alpha=.86,
                           color=PHASE_COLORS[phase_index], linewidths=0, zorder=3)
            ax.text(.035, .945, rf'$\sigma={sign:+.0f}$', transform=ax.transAxes,
                    ha='left', va='top', fontsize=13,
                    bbox=dict(facecolor='white', alpha=.93, edgecolor='none', pad=2.5),
                    zorder=5)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                          [r'$-\pi$', r'$-\pi/2$', '$0$', r'$\pi/2$', r'$\pi$'])
            ax.set_xlabel(r'$\chi=5\theta-3\zeta$')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(pad=4, length=3.5)
            for spine in ax.spines.values():
                spine.set_color('#444444')
        if energy_ev < 1e4:
            axes[0].set_ylim(.54, .74)
            axes[0].set_yticks([.55, .60, .65, .70])
        else:
            axes[0].set_ylim(.39, .89)
            axes[0].set_yticks([.40, .50, .60, .70, .80])
        axes[0].set_ylabel(r'Normalized flux $s$')
        handles = [
            Line2D([], [], color=PHASE_COLORS[0], marker='o', linestyle='none',
                   markersize=4, label='Orbit sections'),
            Line2D([], [], color='#B4B4B4', linewidth=1.1, label=r'Leading $H_0$'),
            Line2D([], [], color='#242424', dashes=(5, 3), linewidth=1.05,
                   label=r'$\iota=3/5$'),
            Line2D([], [], color='#B74F12', dashes=(1, 2), linewidth=1.25,
                   label=r'Mean $H_0$ center'),
        ]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .025),
                   ncol=4, frameon=False, fontsize=9.5, handlelength=2.5,
                   columnspacing=2.1, handletextpad=.65)
    return fig


def render_rational_surface(data):
    """Draw saved geometry and closed coordinate curves, not particle orbits."""
    xyz, field = data['rational_surface_xyz'], data['rational_surface_B']
    fig = plt.figure(figsize=(9.6, 6.4))
    ax = fig.add_axes([.015, .11, .81, .88], projection='3d')
    norm = Normalize(float(field.min()), float(field.max()))
    ax.plot_surface(*np.moveaxis(xyz, -1, 0), facecolors=plt.cm.viridis(norm(field)),
                    rstride=1, cstride=1, linewidth=0, alpha=.90, shade=False,
                    antialiased=True, rasterized=True)
    colors = ('#171717', '#B51955')
    for k, color in enumerate(colors):
        ax.plot(*data[f'rational_fieldline_{k}'].T, color=color, lw=2.2)
    ax.view_init(elev=28, azim=-52)
    ax.set_proj_type('ortho')
    extent = np.ptp(xyz, axis=(0, 1))
    center = (xyz.max(axis=(0, 1)) + xyz.min(axis=(0, 1))) / 2
    for setter, c, e in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, extent):
        setter(c-.54*e, c+.54*e)
    ax.set_box_aspect(extent)
    ax.set_xlabel('$X$ [m]', labelpad=9)
    ax.set_ylabel('$Y$ [m]', labelpad=9)
    ax.set_zlabel('$Z$ [m]', labelpad=7)
    ax.tick_params(labelsize=10, pad=0)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'),
                           cax=fig.add_axes([.865, .31, .022, .44]))
    colorbar.set_label('$|B|$ [T]', labelpad=11)
    fig.legend(handles=[Line2D([], [], color=c, lw=2, label=label)
                        for c, label in zip(colors, (r'$\chi=0$', r'$\chi=\pi$'))],
               loc='lower center', bbox_to_anchor=(.46, .14), frameon=False, ncol=2)
    return fig


def render_joint_refinement(data):
    """Separate equilibrium and continuous-field resolution diagnostics."""
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7), layout='constrained')
        for m in (10, 12, 16):
            rows = sorted((r for r in data['action_rows'] if r['mpol'] == m),
                          key=lambda r: r['ns'])
            axes[0].plot([r['ns'] for r in rows], [r['action_osc'] for r in rows],
                         'o-', label=f'M={m}, N={m-1}')
        for family, label in [('stellarator12', 'Stellarator M=12'),
                              ('stellarator16', 'Stellarator M=16'),
                              ('axisymmetric', 'Axisymmetric control')]:
            rows = sorted((r for r in data['force_rows'] if r['family'] == family),
                          key=lambda r: r['ns'])
            axes[1].semilogy([r['ns'] for r in rows],
                            [100*r['relative_pressure'] for r in rows], 'o-', label=label)
        axes[0].set_ylabel('Maximum sampled action variation')
        axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axes[1].set_ylabel('Mean force / mean pressure gradient (%)')
        for ax in axes:
            ax.set_xlabel('Radial grid points')
            ax.set_xticks([31, 61, 121, 241])
            ax.grid(alpha=.18)
            ax.legend(frameon=False, fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--data', type=Path,
                        default=here.parents[3] / 'benchmarks/li383_readme_sections.npz')
    parser.add_argument('--output-dir', type=Path, default=here.parent)
    parser.add_argument('--format', nargs='+', choices=['webp', 'png', 'pdf'], default=['webp'])
    parser.add_argument('--dpi', type=int, default=180)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def save(fig, name, **kwargs):
        for ext in args.format:
            options = {'pil_kwargs': {'quality': 90, 'method': 6}} if ext == 'webp' else {}
            if ext == 'pdf':
                options['metadata'] = {'Author': 'rogeriojorge'}
            path = args.output_dir / f'readme_cyclometry_{name}.{ext}'
            fig.savefig(path, dpi=args.dpi, **options, **kwargs)
            print(path)
        plt.close(fig)

    with np.load(args.data, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json']))
        for key in ('low_energy', 'alpha'):
            meta = metadata['cases'][key]
            save(render(data[key + '_sections'], meta['orbit']['markers'],
                        data[key + '_radii'], data[key + '_chi'], data[key + '_H'],
                        meta['action'], meta['orbit']['energy_ev']), key)
        with plt.rc_context({'font.size': 12, 'axes.labelsize': 13, 'pdf.fonttype': 42}):
            save(render_rational_surface(data), 'surface',
                 bbox_inches=Bbox.from_extents(.15, .70, 9.3, 5.6))
        save(render_joint_refinement(metadata['joint_refinement']), 'convergence')


if __name__ == '__main__':
    main()
