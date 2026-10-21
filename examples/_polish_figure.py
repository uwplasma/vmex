"""Two-panel README figure shared by the polishing examples: radial profile and regional bars."""
import matplotlib.pyplot as plt
import numpy as np

INK2, GRID, BASELINE = "#52514e", "#e1e0d9", "#c3c2b7"
COLORS = {"VMEC solve": "#eda100", "polished": "#2a78d6"}


def polish_figure(path, profiles, bars, names, profile_label, bar_label, bar_format="%.0f", headroom=1.0):
    """Save the profile (rho, value) and the three regional values, per stage, to ``path``."""
    plt.rcParams.update({"axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False})
    figure, (profile, regions) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                              gridspec_kw={"width_ratios": [1.6, 1.0]},
                                              layout="constrained")
    for label, (rho, value) in profiles.items():
        profile.semilogy(rho, value, color=COLORS[label], linewidth=2.0, label=label)
    profile.set(xlim=(0.0, 1.0), xlabel=r"$\rho=\sqrt{s}$", ylabel=profile_label)
    profile.grid(True, which="major", color=GRID)
    profile.legend(loc="upper center")
    x, width = np.arange(3), 0.38
    for i, (label, values) in enumerate(bars.items()):
        container = regions.bar(x + (i - 0.5) * (width + 0.02), values, width,
                                color=COLORS[label], label=label)
        regions.bar_label(container, fmt=bar_format, fontsize=8, color=INK2, padding=2)
    regions.set_yscale("log")
    regions.set_ylim(top=regions.get_ylim()[1] * headroom)
    regions.set_xticks(x, names)
    regions.set_ylabel(bar_label)
    regions.grid(True, axis="y", which="major", color=GRID)
    regions.legend(loc="upper right")
    figure.savefig(path, dpi=160, pil_kwargs={"lossless": True})
    plt.close(figure)
