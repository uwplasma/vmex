"""DESC input bridge; WOUT is produced by the ordinary VMEX solve."""

from dataclasses import replace
from pathlib import Path
import re
import pickle

import numpy as np

from .errors import VmecInputError
from .input import VmecInput

_FORMATS = {".h5": "hdf5", ".hdf5": "hdf5", ".pkl": "pickle", ".pickle": "pickle"}


def is_desc_file(path: Path) -> bool:
    """Recognize DESC outputs or native text decks without importing DESC."""
    if path.suffix.lower() in _FORMATS:
        return True
    try:
        with path.open("rb") as stream:
            text = stream.read(65536).decode("utf-8", errors="replace")
    except OSError as err:
        raise VmecInputError(f"Cannot read input file: {path}", hint=str(err)) from err
    text = re.sub(r"[!#].*", "", text)
    return not re.search(r"&indata|^\s*\{", text, re.I) and bool(
        re.search(r"^\s*(?:sym|M_pol|L_rad|N_tor|Psi)\s*=", text, re.M | re.I)
    )


def _compact_boundary(inp, tolerance):
    # Triangle inequality bounds the displacement everywhere, relative to the
    # RMS nonconstant R/Z geometry (not the much larger major radius). Apply
    # the same bound to both angular derivatives to retain small sharp modes.
    coefficients = np.stack([getattr(inp, key) for key in ("rbc", "zbs", "rbs", "zbc")])
    m = np.arange(inp.mpol)[None, :]
    n = np.arange(-inp.ntor, inp.ntor + 1)[:, None]
    amplitude = np.sum(np.abs(coefficients), axis=0)
    weights = np.stack(np.broadcast_arrays(np.ones_like(amplitude), m, abs(n)))
    scales = np.sqrt(np.sum((coefficients[None] * weights[:, None]) ** 2, axis=(1, 2, 3)) / 2)
    scales[0] = np.sqrt(np.sum(coefficients[:, (n != 0) | (m != 0)] ** 2) / 2)
    candidates = []
    for mp in range(2, inp.mpol + 1):
        for nt in range(inp.ntor + 1):
            omitted = (m >= mp) | (abs(n) > nt)
            tail = np.sum(weights * amplitude * omitted, axis=(1, 2))
            if np.all(tail <= tolerance * scales):
                candidates.append((mp * (2 * nt + 1), mp, nt))
    _, mp, nt = min(candidates)
    cut = slice(inp.ntor - nt, inp.ntor + nt + 1)
    return replace(inp, mpol=mp, ntor=nt, **{
        key: getattr(inp, key)[cut, :mp] for key in ("rbc", "zbs", "rbs", "zbc")
    })


def _hdf5(node):
    import h5py

    if isinstance(node, h5py.Group):
        values = {key: _hdf5(value) for key, value in node.items()}
        if values.get("__class__") == "list":
            return [values[str(i)] for i in range(len(values) - 1)]
        return values
    value = node[()]
    if isinstance(value, bytes):
        return None if value == b"None" else value.decode()
    return value


class _Record:
    """Inert replacement for pickled DESC objects; never invokes their code."""


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("desc."):
            return type(name, (_Record,), {})
        if (module, name) == ("jax._src.array", "_reconstruct_array"):
            return _numpy_array
        if module in ("numpy", "numpy.core.multiarray", "numpy._core.multiarray",
                      "numpy.core.numeric", "numpy._core.numeric") and name in (
                          "ndarray", "dtype", "_reconstruct", "scalar", "_frombuffer"):
            return super().find_class(module, name)
        raise ValueError(f"unsupported pickle global: {module}.{name}")


def _numpy_array(function, args, state, aval):
    array = function(*args)
    array.__setstate__(state)
    return array


def _records(value):
    if isinstance(value, _Record):
        return {"__class__": type(value).__name__, **{k: _records(v) for k, v in vars(value).items()}}
    if isinstance(value, list):
        return [_records(v) for v in value]
    return value


def _kind(value):
    return value.get("__class__", "").rsplit(".", 1)[-1]


def _profile(profile, rho):
    from scipy.interpolate import CubicHermiteSpline, CubicSpline, interp1d

    kind = _kind(profile)
    p = np.asarray(profile.get("_params", []))
    if kind == "PowerSeriesProfile":
        return rho[:, None] ** profile["_basis"]["_modes"][:, 0] @ p
    if kind == "MTanhProfile":
        z = (rho - p[2]) / p[3]
        zz = z * (1 - np.tanh(z)) / 2
        return (p[0] - p[1]) * (1 - np.tanh(z)) / 2 + p[1] + np.polynomial.polynomial.polyval(zz, np.r_[0, p[4:]]) * (p[1] - p[0]) / 2
    if kind == "TwoPowerProfile":
        return p[0] * (1 - rho**p[1])**p[2]
    if kind in ("ScaledProfile", "PowerProfile"):
        value = _profile(profile["_profile"], rho)
        return value * profile["_scale"] if kind == "ScaledProfile" else value**profile["_power"]
    if kind in ("SumProfile", "ProductProfile"):
        values = [_profile(v, rho) for v in profile["_profiles"]]
        return np.sum(values, axis=0) if kind == "SumProfile" else np.prod(values, axis=0)
    if kind in ("SplineProfile", "HermiteSplineProfile"):
        x = profile["_knots"]
        method = profile.get("_method", "hermite")
        if method == "cubic2":
            return CubicSpline(x, p)(rho)
        if method in ("nearest", "linear"):
            return interp1d(x, p, kind=method, fill_value="extrapolate")(rho)
        if method == "hermite":
            p, slopes = np.split(p, 2)
        elif method in ("cubic", "catmull-rom"):
            slopes = np.diff(p) / np.diff(x)
            middle = ((slopes[:-1] + slopes[1:]) / 2 if method == "cubic"
                      else (p[2:] - p[:-2]) / (x[2:] - x[:-2]))
            slopes = np.r_[slopes[0], middle, slopes[-1]]
        else:
            raise ValueError(f"unsupported DESC spline method: {method}")
        return CubicHermiteSpline(x, p, slopes)(rho)
    raise ValueError(f"unsupported DESC profile: {kind}")


def _native(source):
    # Only the final continuation stage contributes to the equilibrium.
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?"
    options, boundary, axis = {}, {}, {}
    profiles = {key: {} for key in ("p", "i", "c")}
    for line in re.sub(r"[!#].*", "", source.read_text()).splitlines():
        indices = {k.lower(): int(v) for k, v in re.findall(r"([lmn])\s*:\s*(-?\d+)", line, re.I)}
        if indices:
            for key, value in re.findall(rf"(R1|Z1|R0|Z0|p|i|c)\s*=\s*({number})", line, re.I):
                key, value = key.lower(), float(value.replace("D", "e").replace("d", "e"))
                if key in profiles:
                    profiles[key][indices["l"]] = value
                else:
                    target = boundary if key.endswith("1") else axis
                    target.setdefault(key[0], {})[(indices.get("m", 0), indices["n"])] = value
        elif "=" in line:
            key, value = map(str.strip, line.split("=", 1))
            tokens = re.split(r"[\s,;]+", value)
            last = tokens[-1].split("x")[0]
            if ":" in last:
                start, step, stop = map(float, last.split(":"))
                last = str(start + np.ceil((stop - start) / step) * step)
            options[key.lower()] = float(last.replace("D", "e").replace("d", "e")) if re.fullmatch(number, last) else last
    def series(values):
        return {"__class__": "PowerSeriesProfile", "_params": np.array(list(values.values())),
                "_basis": {"_modes": np.array([[ell, 0, 0] for ell in values])}}
    eq = {"__class__": "Equilibrium", "_NFP": options.get("nfp", 1),
          "_Psi": options.get("psi", 1), "_sym": bool(options.get("sym", False)),
          "_M": int(options.get("m_pol", 0)), "_N": int(options.get("n_tor", 0)),
          "_bdry_mode": options.get("bdry_mode", "lcfs")}
    if profiles["i"] and profiles["c"]:
        raise ValueError("DESC input specifies both current and iota")
    for key, name, ratio in (("p", "pressure", "pres_ratio"), ("i", "iota", ""), ("c", "current", "curr_ratio")):
        values = profiles[key]
        if key == "p" or (key == "c" and not profiles["i"]):
            values = values or {0: 0.0}
        eq["_" + name] = series({ell: v * options.get(ratio, 1) for ell, v in values.items()}) if values else None
    if options.get("objective") == "vacuum":
        eq.update(_pressure=series({0: 0.0}), _iota=None, _current=series({0: 0.0}))
    if not boundary:
        raise ValueError("DESC input has no boundary")
    eq["_surface"] = {}
    for coordinate in ("R", "Z"):
        values = boundary[coordinate.lower()]
        eq["_surface"][f"_{coordinate}_basis"] = {"_modes": np.array([[0, m, n] for m, n in values])}
        eq["_surface"][f"_{coordinate}_lmn"] = np.array([v * options.get("bdry_ratio", 1) if n else v for (m, n), v in values.items()])
        values = axis.get(coordinate.lower(), {(0, n): v for (m, n), v in values.items() if m == 0})
        eq[f"_{coordinate}_basis"] = {"_modes": np.array([[0, m, n] for m, n in values])}
        eq[f"_{coordinate}_lmn"] = np.array(list(values.values()))
    # Match DESC's right-handed initial coordinates (evaluated at theta=zeta=0).
    derivatives = []
    for c in ("R", "Z"):
        modes, values = eq["_surface"][f"_{c}_basis"]["_modes"], eq["_surface"][f"_{c}_lmn"]
        m, n = modes[:, 1], modes[:, 2]
        origin = sum(v for (_, nn), v in axis.get(c.lower(), {}).items() if nn >= 0)
        if c.lower() not in axis:
            origin = np.sum(values[(m == 0) & (n >= 0)])
        radial = np.sum(values * np.where(m == 0, 2, abs(m)) * (m >= 0) * (n >= 0)) - 2 * origin
        angular = np.sum(values * abs(m) * (m < 0) * (n >= 0))
        derivatives.append((radial, angular))
    (rr, rt), (zr, zt) = derivatives
    if zr * rt - rr * zt < 0:
        for c in ("R", "Z"):
            eq["_surface"][f"_{c}_lmn"][eq["_surface"][f"_{c}_basis"]["_modes"][:, 1] < 0] *= -1
        profile = eq["_iota"] if eq["_iota"] is not None else eq["_current"]
        profile["_params"] *= -1
    return eq


def _input(eq):
    if _kind(eq) == "EquilibriaFamily" and eq["_equilibria"]:
        eq = eq["_equilibria"][-1]
    if _kind(eq) != "Equilibrium":
        raise ValueError("expected a DESC Equilibrium or nonempty EquilibriaFamily")
    if eq.get("_bdry_mode", "lcfs") != "lcfs":
        raise ValueError("DESC conversion requires a fixed outer boundary")
    if eq.get("_anisotropy") is not None:
        raise ValueError("DESC anisotropic pressure is unsupported")
    surface = eq["_surface"]
    if surface.get("_rho", 1) != 1:
        raise ValueError("DESC surface must be at rho=1")
    modes = np.concatenate([surface[f"_{c}_basis"]["_modes"] for c in ("R", "Z")])
    mp, nt = max(2, int(abs(modes[:, 1]).max()) + 1), int(abs(modes[:, 2]).max())
    kwargs = dict(mpol=mp, ntor=nt, nfp=eq["_NFP"], lasym=not eq["_sym"],
                  phiedge=eq["_Psi"], ns_array=[17, 33, 65], niter_array=[2000, 4000, 8000],
                  ftol_array=[1e-8, 1e-10, 1e-12], delt=0.9, nstep=250)
    for c, cosine, sine in (("R", "rbc", "rbs"), ("Z", "zbc", "zbs")):
        bc, bs = np.zeros((2*nt + 1, mp)), np.zeros((2*nt + 1, mp))
        # Product-to-sum identities map DESC's signed sine/cosine indices
        # directly to cos(m*theta-n*zeta), sin(m*theta-n*zeta).
        for (_, m, n), value in zip(surface[f"_{c}_basis"]["_modes"], surface[f"_{c}_lmn"]):
            m, n = int(m), int(n)
            for sign in (-1, 1):
                nn = sign * abs(n)
                is_sine = (m < 0) != (n < 0)
                factor = (sign if m >= 0 and n < 0 else -sign if m < 0 and n < 0 else 1) / 2
                nn = -nn
                if m == 0 and nn < 0:
                    nn, factor = -nn, -factor if is_sine else factor
                (bs if is_sine else bc)[nn + nt, abs(m)] += factor * value
        kwargs.update({cosine: bc, sine: bs})
        axis = np.zeros(2 * max(nt, int(eq["_N"])) + 1)
        for (ell, m, n), value in zip(eq[f"_{c}_basis"]["_modes"], eq[f"_{c}_lmn"]):
            if m == 0:
                axis[int(n) + len(axis)//2] += (-1.)**(int(ell)//2) * value
        center = len(axis)//2
        kwargs[c.lower() + "axis_c"] = axis[center:center+nt+1]
        kwargs[c.lower() + "axis_s"] = np.r_[0, -axis[center-nt:center][::-1]]
    rho = np.linspace(0, 1, 101)
    for name, prefix, kind in (("pressure", "am", "pmass"), ("iota", "ai", "piota"), ("current", "ac", "pcurr")):
        profile = eq["_" + name]
        if profile is None and name != "pressure":
            continue
        if profile is None:
            from scipy.constants import elementary_charge
            ne, te, ti, zeff = [_profile(eq["_" + key], rho) for key in
                                ("electron_density", "electron_temperature", "ion_temperature", "atomic_number")]
            values = elementary_charge * ne * (te + ti / zeff)
            profile = {}
        else:
            values = _profile(profile, rho)
        if not np.all(np.isfinite(values)):
            raise ValueError("nonfinite DESC profile")
        polynomial = _kind(profile) == "PowerSeriesProfile" and np.all(profile["_basis"]["_modes"][:, 0] % 2 == 0)
        if name == "current":
            if values[0] != 0:
                raise ValueError("DESC enclosed current must vanish on axis")
            kwargs.update(ncurr=1, curtor=values[-1])
        if polynomial:
            powers = profile["_basis"]["_modes"][:, 0].astype(int)//2
            coefficients = np.zeros(max(powers) + 1)
            coefficients[powers] = profile["_params"]
            if name == "current":
                coefficients = coefficients[1:]
            kwargs.update({prefix: coefficients, f"{kind}_type": "power_series_I" if name == "current" else "power_series"})
        else:
            kwargs.update({f"{kind}_type": "cubic_spline_I" if name == "current" else "cubic_spline",
                           f"{prefix}_aux_s": rho**2, f"{prefix}_aux_f": values})
    return VmecInput(**kwargs)


def write_desc_input(source: Path, outdir: Path | None = None, *, tolerance: float = 0.01) -> Path:
    """Read a DESC equilibrium without DESC and write a compact VMEC input.

    The tolerance bounds discarded boundary position and angular derivatives,
    not the solved magnetic field. Zero retains all nonzero boundary modes.
    """
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 0.01:
        raise VmecInputError("DESC boundary tolerance must lie between 0 and 0.01")
    try:
        file_format = _FORMATS.get(source.suffix.lower())
        if file_format == "hdf5":
            import h5py
            with h5py.File(source, "r") as stream:
                eq = _hdf5(stream)
        elif file_format == "pickle":
            with source.open("rb") as stream:
                eq = _records(_Unpickler(stream).load())
        else:
            eq = _native(source)
        inp = _compact_boundary(_input(eq), tolerance)
        # One extra solver harmonic resolves the interior reparameterization.
        extra_n = int(inp.ntor > 0)
        inp = replace(inp, mpol=inp.mpol + 1, ntor=inp.ntor + extra_n, **{
            key: np.pad(getattr(inp, key), ((extra_n, extra_n), (0, 1)))
            for key in ("rbc", "zbs", "rbs", "zbc")
        })
        name = source.stem if file_format else source.name
        return inp.to_indata((outdir or source.parent) / f"input.{name}")
    except Exception as err:
        raise VmecInputError(f"Cannot convert DESC file: {err}", hint=str(source)) from err
