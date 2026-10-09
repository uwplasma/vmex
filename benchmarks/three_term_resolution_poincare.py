#!/usr/bin/env python3
"""Vacuum ground truth for ``run_three_term_resolution.py --case vacuum``: field lines from each method's surfaces.

    python benchmarks/three_term_resolution_poincare.py OUT [--only truth 4 5 ...] [--s 0.1 0.3 ... 1]
    python benchmarks/three_term_resolution_poincare.py OUT --plot [--figure 4 8 12]

In vacuum the external field is the whole field, so its field lines define the flux surfaces without virtual
casing.  A VMEC surface is a flux surface only if field lines started on it stay on it.  From ``--lines`` points
on each of a method's surfaces ``--s`` at phi = 0 (and the 12 x 12 target's, the reference) the lines of
``OUT/field.npz`` are traced for ``--transits`` toroidal transits (RK4 in phi, ``--steps`` per field period,
dR/dphi = R B_R / B_phi, dZ/dphi = R B_Z / B_phi).  Their crossings of phi = 0 and of half a field period are
compared with the surface they started on.  Each ``OUT/mM/<method>.json`` gains ``poincare_surfaces``
({s: [max, RMS]} distance of the crossings from that surface's curve, in mm), ``poincare_mm_max`` and
``poincare_mm_rms`` (the LCFS's), ``poincare_lost`` (lines that left the grid or stalled), ``axis_mm`` (largest
distance over a field period from the field's closed field line, found by Newton on the field-period map),
``coil_bn_max`` and ``coil_bn_rms`` (B.n/|B| on the LCFS, the exact flux-surface condition in vacuum) and
``resonant_bn`` ([m, n, amplitude, m iota - n] of its harmonic in straight-field-line angles that field lines amplify
most, |b_mn| / |m iota - n|).  ``--only`` traces a
subset (``truth`` and/or resolutions), so several processes can share the work; each saves
``OUT/poincare/<key>.npz``.  ``--plot`` collects them into ``OUT/poincare.json``, draws
``OUT/three_term_resolution_poincare.png`` (the distances against s at the ``--figure`` resolutions, and their mean
against the resolution) and ``OUT/poincare/mM.png`` per resolution (the distances against s, and the signed
distance of the LCFS crossings against the poloidal angle).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

METHODS = ("three_term", "nestor", "nestor_mgrid", "vmec2000", "desc", "desc_lm")
WOUTS = {"desc": "desc/wout_desc.nc", "desc_lm": "desc_lm/wout_desc.nc"}


def boundary(path):
    with netCDF4.Dataset(path) as nc:
        return (np.asarray(nc["xm"][:], float), np.asarray(nc["xn"][:], float), np.asarray(nc["rmnc"][:])[-1],
                np.asarray(nc["zmns"][:])[-1], int(nc["nfp"][:]))


def lcfs(b, theta, phi):
    xm, xn, rmnc, zmns, _ = b
    a = np.outer(theta, xm) - xn * phi
    return np.cos(a) @ rmnc.T, np.sin(a) @ zmns.T


def surfaces(path, s):
    """The surfaces nearest to ``s`` of a wout, in the layout of :func:`boundary`, and their exact s."""
    with netCDF4.Dataset(path) as nc:
        rmnc, zmns = np.asarray(nc["rmnc"][:]), np.asarray(nc["zmns"][:])
        j = np.round(np.asarray(s) * (rmnc.shape[0] - 1)).astype(int)
        return (np.asarray(nc["xm"][:], float), np.asarray(nc["xn"][:], float), rmnc[j], zmns[j],
                int(nc["nfp"][:])), j / (rmnc.shape[0] - 1)


def curve_distance(R, Z, Rc, Zc):
    """Distance of each point (R, Z) to the closed polyline (Rc, Zc)."""
    P = np.stack([R, Z], -1)[:, None, :]
    A = np.stack([Rc, Zc], -1)[None]
    B = np.roll(A, -1, axis=1)
    AB = B - A
    t = np.clip(np.sum((P - A) * AB, -1) / np.maximum(np.sum(AB * AB, -1), 1e-300), 0, 1)
    return np.min(np.linalg.norm(P - (A + t[..., None] * AB), axis=-1), axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--lines", type=int, default=32)
    p.add_argument("--transits", type=int, default=30)
    p.add_argument("--steps", type=int, default=128, help="RK4 steps per field period")
    p.add_argument("--s", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                   help="normalized toroidal flux of the starting surfaces")
    p.add_argument("--figure", type=int, nargs="+", default=[4, 8, 12])
    p.add_argument("--only", nargs="+", help="trace only these: truth and/or resolutions M")
    p.add_argument("--methods", nargs="+", choices=METHODS, help="trace only these methods (default all)")
    p.add_argument("--plot", action="store_true", help="collect OUT/poincare/*.npz into poincare.json and the figure")
    a = p.parse_args()
    if a.plot:
        return plot(a)

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    data = np.load(a.out / "field.npz")
    gamma = jnp.asarray(data["gamma"]).reshape(-1, 3)
    npts = data["gamma"].shape[-2]
    dgamma = jnp.asarray(data["gamma_dash"]).reshape(-1, 3) * jnp.asarray(
        np.repeat(data["currents"], npts) / npts * 1e-7)[:, None]

    def b_cyl(R, phi, Z):  # the filament sum of the benchmark's field, at (lines,) points
        x = jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), Z], -1)
        d = x[:, None, :] - gamma[None]
        B = jnp.sum(jnp.cross(dgamma[None], d) / (jnp.sum(d * d, -1) ** 1.5)[..., None], 1)
        c, s = jnp.cos(phi), jnp.sin(phi)
        return c * B[:, 0] + s * B[:, 1], -s * B[:, 0] + c * B[:, 1], B[:, 2]

    nfp = boundary(a.out / "truth.nc")[4]
    h = 2 * np.pi / (nfp * a.steps)

    def rhs(phi, y):
        R, Z = y
        BR, Bp, BZ = b_cyl(R, jnp.full_like(R, phi), Z)
        return jnp.stack([R * BR / Bp, R * BZ / Bp])

    def rk4(y, phi):
        k1 = rhs(phi, y)
        k2 = rhs(phi + h / 2, y + h / 2 * k1)
        k3 = rhs(phi + h / 2, y + h / 2 * k2)
        k4 = rhs(phi + h, y + h * k3)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    half = a.steps // 2

    @jax.jit
    def period(y, phi0):
        """One field period from phi0: the state at its end and after every step."""
        def body(carry, k):
            y = rk4(carry, phi0 + k * h)
            return y, y

        return jax.lax.scan(body, y, jnp.arange(a.steps))

    def trace(R0, Z0):
        y = jnp.stack([jnp.asarray(R0), jnp.asarray(Z0)])
        at0, athalf = [], []
        for k in range(a.transits * nfp):
            y, ys = period(y, k * 2 * np.pi / nfp)
            at0.append(np.asarray(y))
            athalf.append(np.asarray(ys[half - 1]))
        return np.stack(at0), np.stack(athalf)  # (crossings, 2, lines)

    def magnetic_axis(R0, eps=1e-7):
        """The field's closed field line: Newton on the field-period map from (R0, 0), and its (R, Z) at phi = k h."""
        y = np.array([R0, 0.0])
        for _ in range(20):
            P = np.asarray(period(jnp.asarray(np.stack([y, y + [eps, 0], y + [0, eps]], 1)), 0.0)[0])
            step = np.linalg.solve((P[:, 1:] - P[:, :1]) / eps - np.eye(2), y - P[:, 0])
            y = y + step
            if np.linalg.norm(step) < 1e-13:
                break
        ys = np.asarray(period(jnp.asarray(y[:, None]), 0.0)[1])[:, :, 0]
        return np.concatenate([y[None], ys[:-1]])

    phis = h * np.arange(a.steps)
    true_axis = magnetic_axis(float(lcfs(surfaces(a.out / "truth.nc", [0.0])[0], [0.0], 0.0)[0][0, 0]))

    def axis_distance(w):
        """Largest distance (mm) over one field period between ``w``'s s = 0 curve and the field's axis."""
        xm, xn, rmnc, zmns, _ = surfaces(w, [0.0])[0]
        ang = -np.outer(phis, xn[xm == 0])
        R, Z = np.cos(ang) @ rmnc[0][xm == 0], np.sin(ang) @ zmns[0][xm == 0]
        return round(1e3 * float(np.max(np.hypot(R - true_axis[:, 0], Z - true_axis[:, 1]))), 3)

    def coil_bn(w, ntheta=128, nzeta=64):
        """The field's B.n/|B| on ``w``'s LCFS: max, RMS, and the harmonic (m, n) of largest |b_mn| / |m iota - n|
        in straight-field-line angles (amplitude and detuning m iota - n, iota at the edge)."""
        with netCDF4.Dataset(w) as nc:
            lam, iota = np.asarray(nc["lmns"][:]), float(nc["iotaf"][:][-1])
        xm, xn, rmnc, zmns, _ = surfaces(w, [1.0])[0]
        th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, ntheta, endpoint=False),
                             np.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False), indexing="ij")
        c, s_ = np.cos(th[..., None] * xm - ph[..., None] * xn), np.sin(th[..., None] * xm - ph[..., None] * xn)
        R, Z = c @ rmnc[0], s_ @ zmns[0]
        e_theta = np.stack([(-s_ * xm) @ rmnc[0], 0 * R, (c * xm) @ zmns[0]])  # (R, phi, Z) components
        e_phi = np.stack([(s_ * xn) @ rmnc[0], R, (-c * xn) @ zmns[0]])
        normal = np.cross(e_phi, e_theta, axis=0)
        points = np.stack([R.ravel(), ph.ravel(), Z.ravel()])
        B = np.concatenate([np.stack(b_cyl(*map(jnp.asarray, p))) for p in np.array_split(points, 16, axis=1)], 1)
        B = B.reshape(3, *R.shape)
        f = np.sum(B * normal, 0) / np.linalg.norm(B, axis=0) / np.linalg.norm(normal, axis=0)
        lmns = 1.5 * lam[-1] - 0.5 * lam[-2]  # lambda at the LCFS, from the half mesh
        ts, jac = th + s_ @ lmns, 1 + (c * xm) @ lmns
        harmonics = [(2 * abs(np.mean(f * np.exp(-1j * (m * ts - n * ph)) * jac)), m, n)
                     for m in range(1, 25) for n in range(-8 * nfp, 8 * nfp + 1, nfp)]
        amp, m, n = max(harmonics, key=lambda x: x[0] / max(abs(x[1] * iota - x[2]), 1e-3))
        return dict(coil_bn_max=float(np.abs(f).max()), coil_bn_rms=float(np.sqrt(np.mean(f ** 2))),
                    resonant_bn=[m, n, float(amp), round(m * iota - n, 4)])

    theta0 = np.linspace(0, 2 * np.pi, a.lines, endpoint=False)
    theta_dense = np.linspace(0, 2 * np.pi, 1441)
    planes = (0.0, np.pi / nfp)
    only = set(a.only or [])
    targets = [("truth", a.out / "truth.nc", None)] if not only or "truth" in only else []
    for mdir in sorted((q for q in a.out.glob("m*") if q.is_dir() and q.name[1:].isdigit()),
                       key=lambda q: int(q.name[1:])):
        if only and mdir.name[1:] not in only:
            continue
        for m in a.methods or METHODS:
            w = mdir / WOUTS.get(m, f"wout_{m}.nc")
            if w.exists() and (mdir / f"{m}.json").exists():
                targets.append((m, w, mdir))
    for name, w, mdir in targets:
        b, s = surfaces(w, a.s)
        R0, Z0 = lcfs(b, theta0, 0.0)  # (lines, surfaces)
        crossings = np.stack(trace(R0.T.ravel(), Z0.T.ravel()))  # (planes, crossings, 2, surfaces * lines)
        lost = int(np.sum(~np.all(np.isfinite(crossings[:, :, 0, :]), axis=(0, 1))))
        per_surface = {}
        for i, si in enumerate(s):
            distances = []
            for points, phi in zip(crossings[..., i * a.lines:(i + 1) * a.lines], planes):
                R, Z = points[:, 0, :].ravel(), points[:, 1, :].ravel()
                ok = np.isfinite(R) & np.isfinite(Z)
                Rc, Zc = lcfs(b, theta_dense, phi)
                distances.append(curve_distance(R[ok], Z[ok], Rc[:, i], Zc[:, i]))
            dist = np.concatenate(distances)
            rms = np.sqrt(np.mean(dist**2))
            per_surface[f"{si:g}"] = [round(1e3 * float(dist.max()), 3), round(1e3 * float(rms), 3)]
        edge = per_surface.get("1", [None, None])
        row = dict(poincare_surfaces=per_surface, poincare_mm_max=edge[0], poincare_mm_rms=edge[1],
                   poincare_lost=lost, axis_mm=axis_distance(w), **coil_bn(w))
        key = "truth" if mdir is None else f"{mdir.name}/{name}"
        print(key, row, flush=True)
        (a.out / "poincare").mkdir(exist_ok=True)
        np.savez(a.out / "poincare" / f"{key.replace('/', '_')}.npz", crossings=crossings, wout=str(w), s=s,
                 lines=a.lines, row=json.dumps(row))
        if mdir is not None:
            path = mdir / f"{name}.json"
            record = json.loads(path.read_text())
            record.update(row)
            path.write_text(json.dumps(record, indent=1) + "\n")


def plot(a):
    rows = {}
    for f in sorted((a.out / "poincare").glob("*.npz")):
        if "row" in (d := np.load(f)):
            rows[f.stem if f.stem == "truth" else f.stem.replace("_", "/", 1)] = json.loads(str(d["row"]))
    (a.out / "poincare.json").write_text(json.dumps(dict(lines=a.lines, transits=a.transits, steps=a.steps,
                                                          rows=rows), indent=1) + "\n")
    colours = {"three_term": "tab:blue", "nestor": "tab:red", "nestor_mgrid": "tab:orange", "vmec2000": "tab:purple",
               "desc": "tab:green", "desc_lm": "tab:olive"}
    curve = lambda row, k: ([float(s) for s in row["poincare_surfaces"]],  # noqa: E731
                            [v[k] for v in row["poincare_surfaces"].values()])
    modes = [m for m in a.figure if any(key.startswith(f"m{m}/") for key in rows)]
    fig, axes = plt.subplots(2, len(modes) + 1, figsize=(3.4 * (len(modes) + 1), 6.4), squeeze=False, sharey="row")
    for col, m in enumerate(modes):
        for r, k in enumerate((0, 1)):
            ax = axes[r, col]
            if "truth" in rows:
                ax.plot(*curve(rows["truth"], k), "k--", lw=1, label="12 x 12 target")
            for name, c in colours.items():
                if f"m{m}/{name}" in rows:
                    ax.plot(*curve(rows[f"m{m}/{name}"], k), "o-", color=c, ms=3, lw=1.2, label=name)
            ax.set_title(f"mpol = ntor = {m}", fontsize=9)
            ax.set_xlabel("s of the starting surface", fontsize=8)
            ax.tick_params(labelsize=7)
    for r, k in enumerate((0, 1)):  # the last column: all resolutions, averaged over the surfaces
        ax = axes[r, -1]
        for name, c in colours.items():
            ms = sorted(int(key.split("/")[0][1:]) for key in rows if key.endswith(f"/{name}"))
            if ms:
                ax.plot(ms, [np.mean(curve(rows[f"m{m}/{name}"], k)[1]) for m in ms], "o-", color=c, ms=3, label=name)
        if "truth" in rows:
            ax.axhline(np.mean(curve(rows["truth"], k)[1]), color="k", ls="--", lw=1)
        ax.set_title("mean over the surfaces", fontsize=9)
        ax.set_xlabel("mpol = ntor", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("largest distance (mm)", fontsize=8)
    axes[1, 0].set_ylabel("RMS distance (mm)", fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"Vacuum field lines from each method's own surfaces ({a.lines} per surface, {a.transits} transits): "
                 "distance of their crossings from the surface they started on", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out / "three_term_resolution_poincare.png", dpi=120)
    plt.close(fig)
    for m in sorted({int(key.split("/")[0][1:]) for key in rows if key != "truth"}):
        case_figure(a, rows, m, colours, curve)
    print("wrote", a.out / "poincare.json", a.out / "three_term_resolution_poincare.png", a.out / "poincare/m*.png")


def signed_distance(R, Z, Rc, Zc):
    """Index of the nearest vertex of the closed polyline (Rc, Zc) and the distance to it, positive outside."""
    A = np.stack([Rc, Zc], -1)
    d = np.stack([R, Z], -1)[:, None, :] - A[None]
    k = np.argmin(np.sum(d * d, -1), axis=1)
    tangent = np.roll(A, -1, 0) - np.roll(A, 1, 0)
    orientation = np.sign(np.sum(Rc * np.roll(Zc, -1) - np.roll(Rc, -1) * Zc))  # +1: counter-clockwise
    normal = orientation * np.stack([tangent[:, 1], -tangent[:, 0]], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return k, np.sum(d[np.arange(len(k)), k] * normal[k], -1)


def case_figure(a, rows, m, colours, curve):
    """``OUT/poincare/mM.png``: every method at one resolution, per surface and around the LCFS."""
    nfp = boundary(a.out / "truth.nc")[4]
    theta_dense = np.linspace(0, 2 * np.pi, 1441)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    names = ["truth"] + [n for n in colours if f"m{m}/{n}" in rows]
    for name in names:
        key = "truth" if name == "truth" else f"m{m}/{name}"
        style = dict(color=colours.get(name, "k"), label="12 x 12 target" if name == "truth" else name)
        for k, ax in zip((0, 1), axes[:2]):
            ax.plot(*curve(rows[key], k), "--" if name == "truth" else "o-", ms=3, lw=1.1, **style)
        d = np.load(a.out / "poincare" / f"{key.replace('/', '_')}.npz")
        s = list(np.asarray(d["s"]))
        if 1.0 not in s:
            continue
        lines = int(d["lines"])
        i = s.index(1.0)
        b, _ = surfaces(str(d["wout"]), [1.0])
        for j, (ax, phi) in enumerate(zip(axes[2:], (0.0, np.pi / nfp))):
            pts = d["crossings"][j][..., i * lines:(i + 1) * lines]
            R, Z = pts[:, 0, :].ravel(), pts[:, 1, :].ravel()
            ok = np.isfinite(R) & np.isfinite(Z)
            Rc, Zc = lcfs(b, theta_dense, phi)
            k, dist = signed_distance(R[ok], Z[ok], Rc[:, 0], Zc[:, 0])
            ax.plot(theta_dense[k] / np.pi, 1e3 * dist, ".", ms=1.5, alpha=0.5, color=style["color"])
    for ax, label in zip(axes[:2], ("largest distance (mm)", "RMS distance (mm)")):
        ax.set_xlabel("s of the starting surface", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
    for ax, title in zip(axes[2:], ("phi = 0", "phi = half a field period")):
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlabel("VMEC theta / pi of the nearest LCFS point", fontsize=8)
        ax.set_ylabel("LCFS crossings: distance outside the LCFS (mm)", fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.suptitle(f"mpol = ntor = {m}: vacuum field lines from each method's own surfaces ({a.lines} lines per surface, "
                 f"{a.transits} transits)", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out / "poincare" / f"m{m}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
