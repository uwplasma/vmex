"""Free-boundary equilibria by least squares of the interface conditions (``method="least_squares"``).

The second method of :func:`~vmex.core.freeboundary_vc.solve_free_boundary_three_terms` (the first,
``method="galerkin"``, is the module :mod:`vmex.core.freeboundary_vc`'s square system on the deck's normal
displacement).  It minimizes, subject to equality constraints, the area-weighted sum of squares on a grid of: in
vacuum ``B_coil . n / B0``; at finite beta the jump ``(B_out - B_in) / B0`` (:func:`_jump_rows`: the normal
field and the sheet current in one norm), with the interior edge field kept on VMEC's Galerkin band
(:func:`_band_limited`).  ``G_plasma = G_coil`` (the net poloidal current) is a constraint.

The unknowns are those of the Galerkin method: the displacement of the deck's boundary along its radial
coordinate lines, one coefficient per boundary mode.  The boundary's Fourier coefficients themselves are not
used: they describe a surface and an angle on it, and the interface conditions see the angle only through the
truncation: the cost is nearly flat along tangential (relabelling) displacements, and steps along them change
the surface only at second order, so Newton steps there fail and the answer would depend on the start.  The deck's
boundary fixes the labelling, as in the Galerkin method; the answer does not depend on the start
(``initial_boundary``).  With the bootstrap current in the solve, its rows are equality constraints: the current
is Redl's exactly, whatever the rows' scale.

The solver (:meth:`LeastSquaresFreeBoundaryModel._newton`) is a trust-region (quasi-)Newton method on the constraint
manifold with exact second derivatives through the equilibrium (second-order adjoint,
:meth:`LeastSquaresFreeBoundaryModel.second_order`): Gauss-Newton and BFGS models first, the exact Hessian as
products in a preconditioned Lanczos subspace where they fail and to certify convergence.  Design gradients
(:class:`LeastSquaresFreeBoundaryProblem`) use the exact reduced Hessian: Gauss-Newton's would be wrong by
``sum_i r_i d^2 r_i`` (up to 21 % at finite beta).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from . import virtual_casing as vc
from .loops import map_columns
from .errors import TrialRejected
from .freeboundary_vc import (BoundaryResidual, ThreeTermsFreeBoundaryModel, ThreeTermsFreeBoundaryProblem,
                              _REPORT_FTOL, _axis_loop, _edge_pressure, _fingerprint, _interface_terms,
                              _net_current, displacement_rms, summarize_boundary_residual)
from .input import VmecInput


def _band_limited(data, mpol, ntor):
    """``data`` with its interior field ``B_total`` kept on the interior's own Galerkin band.

    VMEC's lambda equation zeroes the current through the boundary only on the modes it solves for (``m <= mpol -
    1``, ``|n| <= ntor``); outside that band the edge field is not determined, in VMEC2000 as here.  The covariant
    surface components ``B . e_theta``, ``B . e_phi`` (periodic over a field period) are filtered to that band and the
    tangential field rebuilt from them (``B_in . n = 0``).
    """
    gamma, B = jnp.asarray(data.gamma), jnp.asarray(data.B_total)
    nphi, ntheta = gamma.shape[1], gamma.shape[2]
    phi = jnp.asarray(data.phi)[:, None]
    R, Z = jnp.hypot(gamma[0], gamma[1]), gamma[2]
    k_t = jnp.fft.fftfreq(ntheta, 1.0 / ntheta)[None, :]
    k_p = jnp.fft.fftfreq(nphi, 1.0 / nphi)[:, None]  # field-period harmonics (n in units of nfp)
    nfp = int(data.nfp)

    def d_theta(f):
        return jnp.real(jnp.fft.ifft(1j * k_t * jnp.fft.fft(f, axis=-1), axis=-1))

    def d_phi(f):
        return jnp.real(jnp.fft.ifft(1j * nfp * k_p * jnp.fft.fft(f, axis=0), axis=0))

    c, s_ = jnp.cos(phi), jnp.sin(phi)
    Rt, Zt, Rp, Zp = d_theta(R), d_theta(Z), d_phi(R), d_phi(Z)
    e_t = jnp.stack([Rt * c, Rt * s_, Zt])
    e_p = jnp.stack([Rp * c - R * s_, Rp * s_ + R * c, Zp])
    keep = (jnp.abs(k_t) <= mpol - 1) & (jnp.abs(k_p) <= ntor)

    def band(f):
        return jnp.real(jnp.fft.ifft2(jnp.where(keep, jnp.fft.fft2(f), 0.0)))

    Bt, Bp = band(jnp.sum(B * e_t, axis=0)), band(jnp.sum(B * e_p, axis=0))
    gtt, gtp, gpp = jnp.sum(e_t * e_t, axis=0), jnp.sum(e_t * e_p, axis=0), jnp.sum(e_p * e_p, axis=0)
    det = gtt * gpp - gtp * gtp
    bt, bp = (gpp * Bt - gtp * Bp) / det, (gtt * Bp - gtp * Bt) / det
    filtered = bt[None] * e_t + bp[None] * e_p
    return replace(data, B_total=filtered)


@dataclass(frozen=True)
class _Bundle:
    """An external field with the run's constants (seed axis, field scale) as one pytree.

    The compiled rows take these as arguments, not as constants closed over: two runs at one resolution then
    lower to the same program and share the persistent compilation cache.
    """

    field: Any
    consts: Any


jax.tree_util.register_dataclass(_Bundle, data_fields=["field", "consts"], meta_fields=[])


def _jump_rows(inp, state, external_field, *, runtime, nphi, ntheta, digits, precision, B0):
    """``sqrt(area / A) (B_out - B_in) / B0`` (three Cartesian components a grid point), the interior edge field on
    VMEC's Galerkin band (:func:`_band_limited`).  ``B_in . n = 0``, so its squares sum to the area mean of
    ``(B_out . n)^2 + |n x (B_out - B_in)|^2`` over ``B0^2``: the normal field and the sheet current in one norm,
    without relative weights.  With zero edge pressure the pressure balance ``|B_out| = |B_in|`` follows from
    ``B_out = B_in``, so it is not a row (it is reported, :meth:`LeastSquaresFreeBoundaryModel.boundary_residual`).
    """
    data = _band_limited(vc.surface_field_data_from_state(inp, state, runtime=runtime, nphi=nphi, ntheta=ntheta),
                         int(inp.mpol), int(inp.ntor))
    interface = vc.PlasmaVacuumInterface.from_surface_data(data, p_edge=0.0, digits=digits, precision=precision)
    jump = interface.total_B_out(external_field) - jnp.asarray(data.B_total)
    return (jnp.sqrt(interface.weights)[None] * jump / B0).ravel()


def _trust_region_solution(w, gq, radius, newton):
    """The minimizer of ``gq . z + 1/2 sum w z^2`` over ``|z| <= radius`` (eigenbasis of the model).

    ``newton`` is ``|gq / w|`` (``inf`` for an indefinite model).  Inside the radius it is the Newton step; on
    it, ``-(w + mu)^-1 gq`` with ``mu > max(0, -min w)`` found by bisection in ``log mu``.
    """
    if newton <= radius:
        return -gq / w
    shift = max(0.0, -w[0])
    lo = np.log(1e-300 + 1e-14 * max(abs(w[-1]), 1e-300))
    hi = np.log(np.linalg.norm(gq) / radius + abs(w[-1]) + 1e-300)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.linalg.norm(gq / (w + shift + np.exp(mid))) > radius else (lo, mid)
    return -gq / (w + shift + np.exp(hi))


class LeastSquaresFreeBoundaryModel(ThreeTermsFreeBoundaryModel):
    """The least-squares free boundary of one deck for many external fields and plasma parameters.

    :class:`~vmex.core.freeboundary_vc.ThreeTermsFreeBoundaryModel` (its equilibria, tangents, pushes,
    pullbacks, warm-up, keywords and coordinates: the deck boundary's displacement along its radial lines), the
    least-squares rows of the module docstring and exact second derivatives (:meth:`second_order`,
    :meth:`gradient`).  :meth:`solve_boundary` is the trust-region (quasi-)Newton solve (:meth:`_newton`).
    ``hessian_chunk`` bounds the second-derivative columns per program call (default and least: 2).
    """

    def __init__(self, inp: VmecInput, *, hessian_chunk=None, **options):
        super().__init__(inp, **options)
        # Hessian columns a program call: by default two, the least vmex.core.loops allows; wider blocks take
        # several times the device memory for no speed.
        self.hessian_chunk = max(2, int(hessian_chunk or 2))
        # The net-current row and the bootstrap rows (the last rows) are equality constraints of the free-boundary
        # solve, not least-squares rows: G = G_coil and the current is Redl's, whatever the rows' scale.
        self.n_constraint_rows = 0 if self.fixed_boundary else 1 + (0 if self.bootstrap is None
                                                                    else int(self.bootstrap.x0.size))
        self.require_converged = False  # LeastSquaresFreeBoundaryProblem: an unconverged trial is a failed one
        # LeastSquaresFreeBoundaryProblem: params -> the plasma directions of its design gradient; the exact model
        # is then built with every column and these mixed columns, which the design gradient at the answer reuses
        self.certificate_extra = None
        self._linearization = None
        self._last_linearize = None
        self._second_order_setup(self.rows, self._im._active_state_fields(self.cfg), self._im._edge_mask(self.cfg))

    def _interface_setup(self, state, nphi, ntheta, digits):
        """The least-squares rows, with the base model's virtual-casing plan of the deck's surface
        (:attr:`precision`, planned once for every band; none in vacuum)."""
        from .freeboundary import _vacuum_scalars
        from types import SimpleNamespace

        im, cfg, fixed = self._im, self.cfg, self.fixed
        nfp = int(fixed.nfp)
        if self.vacuum:  # the coils' field alone: no virtual-casing quadrature
            precision, digits = None, None
        else:
            precision = self.precision
        _, _, rmnc, zmns = map(np.asarray, vc._state_geometry_spectra(state, cfg.resolution))
        n_axis = int(cfg.resolution.ntor) + 1
        axis_points, axis_tangent, axis_dphi = _axis_loop(
            SimpleNamespace(raxis_cc=rmnc[0, :n_axis], zaxis_cs=zmns[0, :n_axis], nfp=nfp))
        # The rows' field unit is self.B0 (the area-RMS |B| on the seed boundary).  The run's constants enter the
        # compiled rows as arguments (_Bundle), not as closed-over arrays.
        self._consts = dict(axis=(axis_points, axis_tangent, jnp.asarray(axis_dphi)), B0=jnp.asarray(self.B0))
        vacuum = self.vacuum

        def interface_rows(state, params, field, moments=False):
            """The jump rows (:func:`_jump_rows`; in vacuum ``sqrt(area / A) B_coil . n / B0``), then the
            net-current row ``(G - G_coil) / G_coil`` (an equality constraint of the solve)."""
            field, consts = (field.field, field.consts) if isinstance(field, _Bundle) else (field, self._consts)
            runtime = im.runtime_from_params(params, cfg)
            G_coil = jnp.abs(_net_current(field, *consts["axis"]))
            G = jnp.abs(_vacuum_scalars(state, runtime)[1])
            current = ((G - G_coil) / G_coil)[None]
            if vacuum:  # B_out = B_coil exactly: B_coil . n = 0 is the whole condition
                data = vc.surface_field_data_from_state(fixed, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
                area = jnp.linalg.norm(jnp.asarray(data.area_vector), axis=0)
                B = vc.external_B_cartesian(field, jnp.asarray(data.gamma), jnp.asarray(data.phi))
                bn = jnp.sum(B * jnp.asarray(data.normal), axis=0) / consts["B0"]
                return jnp.concatenate([(jnp.sqrt(area / jnp.sum(area)) * bn).ravel(), current])
            jump = _jump_rows(fixed, state, field, runtime=runtime, nphi=nphi, ntheta=ntheta, digits=digits,
                              precision=precision, B0=consts["B0"])
            return jnp.concatenate([jump, current])

        self.interface_rows = interface_rows
        self._normal_field = self._make_normal_field(fixed, nphi, ntheta, digits, precision)
        terms = jax.jit(lambda state, params, field: _interface_terms(
            fixed, state, field.field if isinstance(field, _Bundle) else field,
            runtime=im.runtime_from_params(params, cfg), nphi=nphi, ntheta=ntheta, digits=digits,
            precision=precision, p_edge=0.0)[:2])
        # every caller's state and parameters on the model's device: a caller's parameters may sit on the default
        # (GPU) device while the solve's state is on the model's (a CPU model on a GPU machine)
        self._terms = lambda state, params, field: terms(*jax.device_put((state, params), cfg.device), field)

    def evaluate(self, x, params, field, seed=None, tight=False):
        """:meth:`ThreeTermsFreeBoundaryModel.evaluate` with the run's constants bundled with ``field``."""
        return super().evaluate(x, params, self._bundle(field), seed, tight)

    def linearize(self, x, aux, field, extra=None):
        """:meth:`ThreeTermsFreeBoundaryModel.linearize` with the run's constants bundled with ``field``."""
        return super().linearize(x, aux, self._bundle(field), extra)

    def boundary_residual(self, state, params, field) -> BoundaryResidual:
        """:class:`~vmex.core.freeboundary_vc.BoundaryResidual` of a solved state (the common evaluator)."""
        return summarize_boundary_residual(*self._terms(state, params, field))

    def _newton_boundary(self, params, field, *, x0=None, jacobian=None, max_nfev=60, verbose=0, undamped=False):
        """The optimizer's warm trial (``ThreeTermsFreeBoundaryProblem._solve``): :meth:`solve_boundary` from ``x0``
        with ``jacobian`` as its first Gauss-Newton model (``undamped`` is the Galerkin Newton's option)."""
        del undamped
        return self.solve_boundary(params, field, x0=x0, jacobian=jacobian, max_nfev=max_nfev, verbose=verbose)

    def solve_boundary(self, params, field, *, x0=None, jacobian=None, max_nfev=None,
                       max_iter=60, callback=None, verbose=0, continuation=True):
        """The free boundary for ``params`` and ``field``, from ``x0`` (default: the deck's boundary).

        :meth:`_newton`; ``max_nfev`` (the Galerkin solve's name) caps its models like ``max_iter``; ``continuation``
        (the Galerkin solve's band continuation) has no counterpart here: this solve is always its own Newton;
        ``jacobian`` (rows by coordinates, a nearby boundary's: an optimizer's previous trial) is the first
        Gauss-Newton model instead of a fresh linearization.
        """
        out = self._newton(params, field, x0=x0, jacobian=jacobian,
                           max_iter=max_iter if max_nfev is None else int(max_nfev), callback=callback,
                           verbose=verbose)
        if self.require_converged and not out["converged"]:  # an optimizer's trial: no answer, not a value
            raise RuntimeError(f"the least-squares free boundary did not converge ({out['status']})")
        return out

    def _second_order_setup(self, rows, active, edge):
        """Exact second derivatives of the cost ``1/2 |rows|^2`` through the equilibrium.

        With ``F(z, p) = 0`` the raw force balance of the projected state ``z`` and ``Phi(z, p) = 1/2 |rows|^2``,
        the multiplier ``lam`` solves ``F_z^T lam = Phi_z`` (one transposed block solve) and the Lagrangian is
        ``L = Phi - lam . F``.  The gradient of the reduced cost ``f(p) = Phi(z(p), p)`` is ``L_p``; its Hessian
        along tangents ``(dz_i, dp_i)`` (``F_z dz_i = -F_p dp_i``) is ``H_ij = [dp_i; dz_i]^T grad^2 L [dp_j;
        dz_j]``, one forward-over-reverse pass of ``L`` per column.  This includes ``sum_k r_k grad^2 r_k``, the
        term Gauss-Newton drops, which is not small here: the residual stays finite at a truncated boundary.

        ``shift`` (``None``, or a constant added to the rows) makes ``Phi = 1/2 |rows + shift|^2``: with the
        constraint rows' entries ``nu - r`` (zeros elsewhere) its Hessian is the Lagrangian's of the interface
        cost with multipliers ``nu`` on the constraint rows, plus ``J_c^T J_c``, which vanishes on their null space.
        """
        im, cfg = self._im, self.cfg

        def tree_dot(a, b):
            return sum(jnp.sum(u * v) for u, v in zip(jax.tree.leaves(a), jax.tree.leaves(b)))

        def lagrange_parts(state, mask, field, shift):
            frozen = jax.lax.stop_gradient(state)
            P = im._dof_projector(cfg, mask)
            F = im.residual_fn(cfg, frozen, mask, formulation="raw")

            def phi(z, prm):
                r = rows(im._assemble(z, im.runtime_from_params(prm, cfg), frozen, P, edge), prm, field)
                r = r if shift is None else r + shift  # constant: second derivatives sum (r + shift) grad^2 r
                return 0.5 * jnp.sum(r * r)

            return frozen, P, F, P(frozen), phi

        def multiplier(state, mask, params, field, shift):
            """``(lam, L_p, adjoint residual / tolerance)``."""
            frozen, P, F, z, phi = lagrange_parts(state, mask, field, shift)
            system = im._raw_block_system(params, cfg, frozen, mask, active, self.chunk)
            phi_z, phi_p = jax.grad(phi, argnums=(0, 1))(z, params)
            _, vjp_z = jax.vjp(lambda zz: F(zz, params), z)
            _, vjp_p = jax.vjp(lambda prm: F(z, prm), params)
            lam, residual, tolerance = im._block_adjoint(system, lambda v: vjp_z(v)[0], P(phi_z), cfg)
            return lam, jax.tree.map(jnp.subtract, phi_p, vjp_p(lam)[0]), residual / tolerance

        def hessian_columns(state, mask, params, field, shift, lam, dz_cols, dp_cols, dz_basis, dp_basis):
            """``[dp_i; dz_i]^T grad^2 L [dp_j; dz_j]``: a row per column ``j``, a column per basis vector ``i``."""
            _, P, F, z, phi = lagrange_parts(state, mask, field, shift)

            def lagrangian(zz, prm):
                return phi(zz, prm) - tree_dot(lam, F(zz, prm))

            grad = jax.grad(lagrangian, argnums=(0, 1))
            basis_z = jax.vmap(P)(dz_basis)

            def column(a):
                wz, wp = jax.jvp(grad, (z, params), (P(a[0]), a[1]))[1]
                return (jax.vmap(lambda d: tree_dot(d, wz))(basis_z)
                        + jax.vmap(lambda d: tree_dot(d, wp))(dp_basis))

            return map_columns(column, (dz_cols, dp_cols), batch_size=self.hessian_chunk)

        def project(dp_basis, tree):
            return jax.vmap(lambda d: tree_dot(d, tree))(dp_basis)

        self._tree_dot = tree_dot
        self._lagrange_parts = lagrange_parts
        self._multiplier = jax.jit(multiplier)
        self._hessian_columns = jax.jit(hessian_columns)
        self._project = jax.jit(project)
        self._last_second_order = None

    @staticmethod
    def _reduced(gs_f, Jc):
        """The constraint rows' null space and multipliers (ESS-scaled coordinates).

        ``gs_f`` is the scaled gradient of the interface cost and ``Jc`` the scaled Jacobian of the constraint
        rows (``None`` without them); their multipliers ``nu`` solve ``Jc^T nu = -gs_f`` in least squares.
        Returns ``(N, gs, nu_c, Jc)``: an orthonormal null-space basis of ``Jc``, the Lagrangian's scaled
        gradient ``gs_f + Jc^T nu_c``, ``nu_c`` and ``Jc``.
        """
        if Jc is None:
            return np.eye(gs_f.size), gs_f, np.zeros(0), None
        nu = -np.linalg.lstsq(Jc.T, gs_f, rcond=None)[0]
        _, sv, Vt = np.linalg.svd(Jc)
        rank = int(np.sum(sv > np.finfo(float).eps * max(Jc.shape) * sv[0]))  # numpy's matrix_rank tolerance
        if rank < Jc.shape[0]:
            raise RuntimeError(f"the constraint rows are dependent (rank {rank} of {Jc.shape[0]})")
        return Vt[rank:].T, gs_f + Jc.T @ nu, nu, Jc

    def second_order(self, x, aux, field, extra=None, columns=True):
        """Exact gradient and Hessian of ``1/2 |rows|^2`` in the model coordinates at ``x``.

        With constraint rows (``n_constraint_rows``) the cost is the interface rows' and the Hessian the
        Lagrangian's, ``H + sum nu_k grad^2 r_k`` over the constraint rows (``nu`` of :meth:`_reduced`), and
        ``g`` its gradient; the dict adds ``nu_c`` and ``shift`` (see :meth:`_second_order_setup`).

        ``columns=False`` skips the Hessian's columns: the dict then has ``hvp`` (``v -> H v`` in the model
        coordinates, one forward-over-reverse pass each) instead of ``H`` and ``C``.

        Returns a dict with ``H`` (``n x n``, symmetrized; ``asymmetry`` its relative antisymmetric part),
        ``g`` (``J^T r``), ``g_adjoint`` (the same gradient from the multiplier: a consistency check), ``J``,
        ``r``, ``aux`` (the tight state), ``lam``, ``dz``/``batch`` (tangents and parameter directions of the
        columns, then of ``extra``), ``C`` (``n x n_extra``: the mixed second derivatives along ``extra``) and
        ``adjoint`` (multiplier residual over its tolerance).
        """
        field = self._bundle(field)
        key = _fingerprint(x, aux[2], field) + ("" if extra is None else _fingerprint(extra)) + str(columns)
        last = self._last_second_order
        if last is not None and last[0] == key:
            return last[1]
        t0 = time.perf_counter()
        J, dz, batch, aux = self.linearize(x, aux, field, extra=extra)
        t1 = time.perf_counter()
        state, mask, params_x, _ = aux
        n = self.x0.size
        r = np.asarray(self._rows(state, params_x, field))
        m, shift, nu_c = self.n_constraint_rows, None, np.zeros(0)
        if m:
            S = np.asarray(self.x_scale, dtype=float)
            k = r.size - m
            nu_c = self._reduced(S * (J[:k, :n].T @ r[:k]), J[k:, :n] * S[None])[2]
            shift = np.zeros(r.size)
            shift[k:] = nu_c - r[k:]
        shift_j = None if shift is None else jnp.asarray(shift)
        with vc.remat_scans():
            lam, grad_p, adjoint = self._multiplier(state, mask, params_x, field, shift_j)
        dz_b, dp_b = jax.tree.map(lambda a: a[:n], dz), jax.tree.map(lambda a: a[:n], batch)
        rs = r if shift is None else r + shift

        def blocked(dz_cols, dp_cols):
            """Hessian columns along the tangents ``(dz_cols, dp_cols)`` (leading axis), ``hessian_chunk`` a program
            call (the last block zero-padded: one executable).  A call that runs out of device memory halves
            ``hessian_chunk`` (for the rest of the run) and is repeated: the memory of a block depends on the
            quadrature the surface needs, unknown in advance."""
            count = jax.tree.leaves(dz_cols)[0].shape[0]
            out, start = [], 0
            while start < count:
                width = max(2, min(self.hessian_chunk, count))  # >= 2: no one-tangent program (vmex.core.loops)

                def take(a):  # this block's columns; the last block zero-padded to the width (one executable)
                    block = a[start:start + width]
                    if block.shape[0] < width:
                        block = jnp.concatenate([block, jnp.zeros((width - block.shape[0],) + a.shape[1:], a.dtype)])
                    return block

                try:
                    with vc.remat_scans():
                        block = np.asarray(self._hessian_columns(state, mask, params_x, field, shift_j, lam,
                                                                 jax.tree.map(take, dz_cols),
                                                                 jax.tree.map(take, dp_cols), dz_b, dp_b))
                except jax.errors.JaxRuntimeError as error:
                    if "RESOURCE_EXHAUSTED" not in str(error) or width <= 2:
                        raise
                    self.hessian_chunk = max(2, width // 2)
                    continue
                out.append(block[:count - start])
                start += width
            return np.vstack(out)

        out = dict(g=J[:, :n].T @ rs, nu_c=nu_c, shift=shift, g_adjoint=np.asarray(self._project(dp_b, grad_p)), J=J,
                   r=r, aux=aux, lam=lam, dz=dz, batch=batch, asymmetry=np.nan, adjoint=float(adjoint))
        t2 = time.perf_counter()
        out["seconds"] = dict(jacobian=t1 - t0, multiplier=t2 - t1)
        if columns:
            cols = blocked(dz, batch)
            H, C = cols[:n].T, cols[n:].T
            asymmetry = float(np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-300))
            out.update(H=0.5 * (H + H.T), C=C, asymmetry=asymmetry)
            out["seconds"]["hessian"] = time.perf_counter() - t2
        else:
            def hvp(v):
                """``H v`` for every row ``v`` of ``(b, n)`` (:func:`blocked`)."""
                vj = jnp.asarray(np.atleast_2d(v))

                def along(a):
                    return jnp.tensordot(vj, a, axes=1)

                return blocked(jax.tree.map(along, dz_b), jax.tree.map(along, dp_b))

            out["hvp"] = hvp
        self._last_second_order = (key, out)
        return out

    def gradient(self, x, aux, field, shift=None):
        """Exact gradient ``J^T (r + shift)`` of ``1/2 |rows + shift|^2`` at a tight state by one adjoint solve
        (no tangents); ``shift`` a constant (default none; minus the constraint rows: the interface cost's)."""
        field = self._bundle(field)
        state, mask, params_x, _ = aux
        with vc.remat_scans():
            _, grad_p, _ = self._multiplier(state, mask, params_x, field, None if shift is None else jnp.asarray(shift))
        return np.asarray(self._project(self.boundary_directions(params_x, x), grad_p))

    def _krylov_model(self, so, N, GN, S, gs, gq_norm_first):
        """Preconditioned block-Krylov subspace of the exact reduced Hessian; sets ``self._krylov_T``.

        Block Lanczos with ``hessian_chunk`` vectors a block (one program call computes a block's Hessian
        products), starting from the preconditioned gradient and fixed pseudo-random directions; the subspace
        grows until the Newton step on it has a residual below ``eta |g|`` (``eta = min(1/2, sqrt(|g| /
        |g_first|))``, the Eisenstat-Walker choice for superlinear convergence).  Returns the subspace basis in the
        scaled model coordinates (``n x k``, in place of ``N``; orthonormal, so radii mean the same for every
        model) and the forcing term's scale.
        """
        wg, Qg = np.linalg.eigh(N.T @ GN @ N)
        wg = np.maximum(wg, np.finfo(float).eps * wg[-1])  # J^T J is positive semidefinite: round-off floor
        Mih = (Qg / np.sqrt(wg)) @ Qg.T

        def op(V):  # V: (n_r, b) in the preconditioned reduced coordinates
            steps = N @ (Mih @ V)  # (n, b) scaled
            HV = S[:, None] * so["hvp"]((S[:, None] * steps).T).T
            return Mih @ (N.T @ HV)

        g = Mih @ (N.T @ gs)
        g_norm = float(np.linalg.norm(g))
        gq_norm_first = g_norm if gq_norm_first is None else gq_norm_first
        eta = min(0.5, np.sqrt(g_norm / max(gq_norm_first, 1e-300)))
        width = max(1, min(self.hessian_chunk, g.size))
        start = np.column_stack([g, np.random.default_rng(0).standard_normal((g.size, width - 1))])
        V = np.linalg.qr(start)[0]
        K, HK = np.zeros((g.size, 0)), np.zeros((g.size, 0))
        while True:
            W = op(V)
            K, HK = np.hstack([K, V]), np.hstack([HK, W])
            T = 0.5 * (K.T @ HK + HK.T @ K)
            ev = np.linalg.eigvalsh(T)
            if ev[0] > 0:  # the Newton step on the subspace and its true residual H K y - g
                y = np.linalg.solve(T, K.T @ g)
                if np.linalg.norm(HK @ y - g) <= eta * g_norm:
                    break
            if K.shape[1] >= g.size:
                break
            W = W - K @ (K.T @ W)
            W = W - K @ (K.T @ W)  # full reorthogonalization (the subspace stays small)
            Q, R = np.linalg.qr(W)
            keep = np.abs(np.diag(R)) > np.finfo(float).eps * max(np.abs(ev).max(), 1e-300) * 1e3
            if not np.any(keep):
                break
            V = Q[:, keep][:, :g.size - K.shape[1]]
        Qb, Rb = np.linalg.qr(Mih @ K)
        Rinv = np.linalg.inv(Rb)
        self._krylov_T = Rinv.T @ T @ Rinv
        return N @ Qb, gq_norm_first

    def _derivative_model(self, exact, x, aux, r, params, field, jacobian, shift):
        """The derivatives of the rows at ``x`` that a model of :meth:`_newton` is built from.

        ``exact``: :meth:`second_order` (every column, with the mixed columns of ``certificate_extra(params)``,
        when an optimizer's design gradient reuses them at the answer; else Hessian products).  Otherwise the
        Gauss-Newton derivatives: ``jacobian`` (a warm start's) with the exact gradient of the interface cost
        (``shift`` of :meth:`gradient`), or a fresh linearization.
        """
        if exact and self.certificate_extra is not None:
            return self.second_order(x, aux, field, extra=self.certificate_extra(params), columns=True)
        if exact:
            return self.second_order(x, aux, field, columns=False)
        if jacobian is not None:
            g_f = self.gradient(x, aux, field, shift=shift)
            return dict(J=np.asarray(jacobian, dtype=float), r=r, g=g_f, g_f=g_f, g_adjoint=g_f, aux=aux,
                        asymmetry=np.nan, adjoint=np.nan, seconds={})
        J, _, _, aux = self.linearize(x, aux, field)
        r_lin = np.asarray(self._rows(aux[0], aux[2], self._bundle(field)))
        g = J[:, :x.size].T @ r_lin
        return dict(J=J, r=r_lin, g=g, aux=aux, asymmetry=np.nan, adjoint=np.nan, g_adjoint=g, seconds={})

    def _newton(self, params, field, *, x0=None, max_iter=60, radius=None, callback=None, verbose=0, jacobian=None):
        """Minimize ``1/2 |rows|^2`` over the boundary by a trust-region (quasi-)Newton method.

        With constraint rows (the net current's and the bootstrap current's, :attr:`n_constraint_rows`) the cost
        is the interface rows' and the method is SQP: every step is the constraint rows' least-norm Newton step plus
        a trust-region step in their Jacobian's null space with the Lagrangian Hessian, and trials are judged by the
        l1 merit function; converged only where the constraint rows hold to ``sqrt(FTOL)``.  Every iterate is a tight
        equilibrium.  The step solves the trust-region subproblem exactly, in the eigenbasis of the scaled reduced
        model (indefinite ones included).  Models, cheapest first: Gauss-Newton ``J^T J`` from one Jacobian, then
        BFGS updates with the exact adjoint gradient (one block factorization a step); where a quasi-Newton model
        fails (a step gaining under a quarter of its prediction, or none) a fresh Gauss-Newton model, and only where
        that fails too the exact Hessian (second-order adjoint, :meth:`second_order`), which also certifies
        convergence.  A step that is not very successful is followed by a second-order correction ``-(J N)^+
        r(x_trial)``; a very successful step at the radius is extended along the same model while it gains.

        Converged: at an exact, positive-definite reduced Hessian, the Newton step passes the shared test
        :func:`~vmex.core.freeboundary_vc.newton_step_converged` (its RMS normal displacement against round-off of
        ``R0`` and against the step that the rows' measured noise produces -- the same boundary solved again from
        another state at every exact model), or no step gains although the exact model predicts one (the cost's
        noise floor).  Anything else is reported as a stall
        (``converged=False``).  Returns a dict with ``x``, ``rows``, ``aux``, ``jacobian``, ``hessian``,
        ``nfev``, ``njev``, ``converged``, ``status`` and ``history`` (a row per model: cost, reduced gradient
        norm, smallest and largest eigenvalue of the reduced model and of ``J^T J``, Newton decrement, gap
        between ``J^T r`` and the adjoint gradient and Hessian asymmetry (exact models), seconds).
        ``callback(x)``, if given, is called with every accepted boundary (e.g. to checkpoint a long solve).
        """
        x = np.array(self.x0 if x0 is None else x0, dtype=float)
        t_eval = time.perf_counter()
        got = self.evaluate(x, params, field, tight=True)
        if verbose:
            print(f"first evaluation {time.perf_counter() - t_eval:.1f} s", flush=True)
        if got is None:
            raise RuntimeError("the starting boundary has no certified equilibrium")
        r, aux = got
        S = np.asarray(self.x_scale, dtype=float)
        nfev, history, converged, status, started = 1, [], False, "max_iter", time.perf_counter()
        so = None
        m = self.n_constraint_rows
        k = r.size - m  # the interface rows; the constraint rows follow

        def cost_of(rows):
            return 0.5 * rows[:k] @ rows[:k]

        # The constraint rows are relative field (or current) mismatches; an equilibrium solved to FTOL determines
        # its fields to about sqrt(FTOL) relative, so the rows are restored to that level, not further.
        tol_c = float(np.sqrt(self.cfg.ftol))

        def violation(rows):
            """The merit function's penalty term: the constraint rows' l1 norm beyond their resolution (below it they
            are the equilibria's noise, which the penalty would otherwise weigh against real gains)."""
            return float(np.sum(np.maximum(np.abs(rows[k:]) - resolved_c(), 0.0)))

        def resolved_c():
            """The constraint rows' resolution: sqrt(FTOL), or their measured repeatability where larger."""
            return max(tol_c, 0.0 if noise_rows is None or not m else float(np.max(np.abs(noise_rows[k:]))))

        penalty = 0.0  # the l1 penalty weight: above max |nu| (exact penalty), never decreased
        H_full = gs_prev = x_prev = None
        need = "gauss_newton"  # the next model: "gauss_newton" (J^T J), "exact" (second order) or None (BFGS)
        noise = 0.0  # the cost's measured repeatability
        noise_rows = None  # the rows' measured repeatability (rows of one boundary solved from two states)
        gq_norm_first = None  # the first exact model's preconditioned gradient norm (the forcing term's scale)
        for it in range(int(max_iter)):
            if need is not None or H_full is None:
                warm = jacobian if need != "exact" and so is None else None
                so = self._derivative_model(need == "exact", x, aux, r, params, field, warm,
                                            None if not m else np.r_[np.zeros(k), -r[k:]])
                # the derivatives come from the linearization's refined state; the cost compared with trials stays
                # the evaluated one (same solver path as the trials: the refined state's rows differ from it by the
                # equilibria's noise, which no trial could beat)
                aux = so["aux"]
                Js = so["J"][:k, :x.size] * S[None]
                GN = Js.T @ Js
                H_full = GN
                gs_f = S * (so["g_f"] if "g_f" in so else so["J"][:k, :x.size].T @ so["r"][:k])
                model, need = ("exact" if need == "exact" else "gauss_newton"), None
            else:  # quasi-Newton: BFGS updates of the last model with the exact (adjoint) gradient
                shift = None if not m else np.r_[np.zeros(k), -r[k:]]  # the interface cost's gradient
                gs_f = S * self.gradient(x, aux, field, shift=shift)
                step_s, y = (x - x_prev) / S, gs_f - gs_prev
                sy = float(step_s @ y)
                if sy > 1e-12 * np.linalg.norm(step_s) * np.linalg.norm(y):  # curvature condition (round-off)
                    Bs = H_full @ step_s
                    H_full = H_full - np.outer(Bs, Bs) / float(step_s @ Bs) + np.outer(y, y) / sy
                model = "quasi-Newton"
            fresh = model == "exact"
            if fresh:  # the cost's repeatability here: the same boundary solved again from a second seed, the
                # solver's memos cleared (ThreeTermsFreeBoundaryModel.repeat; a hot restart would return the same
                # state and understate the noise)
                again = self.repeat(x, params, self._bundle(field), aux)
                nfev += 1
                if again is not None:
                    noise = max(noise, abs(cost_of(again) - cost_of(r)))
                    if noise_rows is None or np.linalg.norm(again - r) > np.linalg.norm(noise_rows):
                        noise_rows = again - r
            gs_prev, x_prev = gs_f, x.copy()
            cost = cost_of(r)
            # the null space of the constraint rows (their Jacobian from the last model) and the Lagrangian's gradient
            N, gs, nu, Jc = self._reduced(gs_f, so["J"][k:, :x.size] * S[None] if m else None)
            Hs = H_full
            N_full = N
            # SQP: every step is the constraint rows' Newton step p (least norm, scaled) plus a null-space step;
            # trials are judged by the l1 merit cost + penalty |r_c|_1, exact for any penalty above max |nu|
            # (Nocedal & Wright, Theorem 17.3; the factor 2 is their margin), so no restoration is needed
            p = np.zeros(x.size) if not m else -np.linalg.lstsq(Jc, r[k:], rcond=None)[0]
            penalty = max(penalty, 2.0 * float(np.max(np.abs(nu)))) if m else 0.0
            merit = cost_of(r) + penalty * violation(r)
            if fresh and "H" in so:  # every column of the exact Hessian (an optimizer's certificate)
                Hs = S[:, None] * so["H"] * S[None]
            elif fresh:
                # The exact model on a Krylov subspace: preconditioned Lanczos on the reduced Hessian (one product
                # H v a vector), preconditioner the reduced Gauss-Newton matrix.  The subspace grows until the
                # Newton step's residual is below eta |g| (forcing eta = min(1/2, sqrt(|g| / |g_first|)), the
                # Eisenstat-Walker choice for superlinear convergence); the trust-region steps below are then
                # exact in it.  The eigenpairs of the tridiagonal certify convergence on the subspace.
                N, gq_norm_first = self._krylov_model(so, N, GN, S, gs, gq_norm_first)
            if fresh and "H" not in so:
                Hp = S * so["hvp"]((S * p)[None])[0] if np.any(p) else 0.0 * p
            else:
                Hp = Hs @ p
            Hr, gr = (self._krylov_T if fresh and "H" not in so else N.T @ Hs @ N), N.T @ (gs + Hp)
            gain_p = -(float(gs_f @ p) + 0.5 * float(p @ Hp)) + penalty * violation(r)  # p's predicted merit gain
            w, Q = np.linalg.eigh(Hr)
            gq = Q.T @ gr
            w_gn = np.linalg.eigvalsh(N.T @ GN @ N)
            decrement = float(np.sum(gq**2 / w)) if w[0] > 0 else np.inf
            # the Newton step, and the step the rows' measured noise alone produces (same model): the shared
            # convergence test (newton_step_converged) compares their sizes in metres
            step_n = S * (p + N @ (Q @ (-gq / w))) if w[0] > 0 else None
            feasible = not m or float(np.max(np.abs(r[k:]))) <= resolved_c()
            step_noise = None
            if step_n is not None and noise_rows is not None:
                g_noise = N.T @ (S * (so["J"][:k, :x.size].T @ noise_rows[:k]))
                step_noise = S * (N @ (Q @ ((Q.T @ g_noise) / w)))
            moved = (displacement_rms(step_n[:self.n_boundary], self.boundary_modes)
                     if step_n is not None and self.n_boundary else np.inf)
            history.append([cost, float(np.linalg.norm(gr)), w[0], w[-1], w_gn[0], w_gn[-1], decrement,
                            float(np.linalg.norm(so["g"] - so["g_adjoint"]) / max(np.linalg.norm(so["g"]), 1e-300))
                            if fresh else np.nan, so["asymmetry"] if fresh else np.nan, time.perf_counter() - started])
            if verbose:
                print(f"{model} {it}: cost {cost:.10e} " + (f"|r_c| {np.max(np.abs(r[k:])):.1e} " if m else "")
                      + f"|g_r| {history[-1][1]:.2e} "
                      f"eig(H_r) [{w[0]:.2e}, {w[-1]:.2e}] "
                      f"eig(JtJ_r) [{w_gn[0]:.2e}, {w_gn[-1]:.2e}] decrement/cost {decrement / (2 * cost):.2e} "
                      + (f"g-g_adj {history[-1][7]:.1e} asym {so['asymmetry']:.1e} adj {so['adjoint']:.1e} "
                         f"noise {noise:.1e} r_c resolved {resolved_c():.1e} " if fresh else "")
                      + f"Newton step moves <= {moved:.1e} m "
                      + ("times " + " ".join(f"{k} {v:.1f}" for k, v in so["seconds"].items()) + " s " if fresh else "")
                      + f"({history[-1][-1]:.0f} s)", flush=True)
            if verbose > 1:
                self._print_weakest(S * (N @ Q[:, 0]))
            floor = noise  # gains below the cost's repeatability are not resolved by the equilibria
            if step_n is not None and feasible and self._step_converged(step_n, step_noise, x):
                if not fresh:  # certify with the exact Hessian
                    need = "exact"
                    continue
                converged, status = True, "converged"
                break
            if fresh and w[0] > 0 and feasible and 0.5 * decrement <= floor:
                # the exact model's whole remaining gain is below the cost's measured repeatability: the minimum is
                # located to the equilibria's noise, and further steps only move along it
                converged, status = True, "converged: gain below the cost's repeatability"
                break
            newton = np.linalg.norm(gq / w) if w[0] > 0 else np.inf
            if radius is None:
                # the first radius: the Newton step, or (indefinite model) the gradient step of the stiffest mode
                radius = newton if np.isfinite(newton) else float(np.linalg.norm(gq) / abs(w[-1]))
            if np.isfinite(newton):
                radius = max(radius, 1e-3 * newton)

            def model_step(radius):
                """The trust-region step of the reduced model for ``radius``, and its predicted gain."""
                z = _trust_region_solution(w, gq, radius, newton)
                return p + N @ (Q @ z), gain_p - (gq @ z + 0.5 * np.sum(w * z * z))

            def trial(x_try):
                """The trial and its rows, or ``(x_try, None)`` without equilibrium."""
                return x_try, self.evaluate(x_try, params, field, tight=True)

            def merit_of(rows):
                return cost_of(rows) + penalty * violation(rows)

            accepted, evaluated = False, False  # evaluated: some trial had an equilibrium (a cost to compare)
            for attempt in range(12):
                step, predicted = model_step(radius)
                if feasible and self._step_converged(S * step, step_noise, x):
                    # a trial step the rows' noise (or round-off) already accounts for carries no information:
                    # an evaluation there only returns the same state, which a ratio test would accept without end
                    if fresh:
                        converged, status = True, "converged: trial steps below the rows' noise"
                    else:
                        need = "exact"
                    break
                x_new, got = trial(x + S * step)
                nfev += 1
                cost_new = np.inf if got is None else merit_of(got[0])
                evaluated = evaluated or got is not None
                rho = (merit - cost_new) / predicted if predicted > 0 else -np.inf
                length = np.linalg.norm(step)
                if verbose:
                    print(f"  step {attempt}: |s| {length:.2e} (radius {radius:.2e}, Newton {newton:.2e}) cost "
                          f"{cost_new:.10e} predicted {predicted:.2e} actual {merit - cost_new:.2e} rho {rho:.3f} "
                          + (f"[cost {cost_of(got[0]):.10e} penalty {penalty:.1e} violation {violation(got[0]):.1e}] "
                             if m and got is not None else "")
                          + f"({time.perf_counter() - started:.0f} s)", flush=True)
                if got is not None and rho <= 0.75:
                    # a step that is not "very successful" by the radius update's own test: a second-order
                    # correction from the residual at the trial point with the Jacobian at hand, x_t - (J N)^+ r_t
                    JN = so["J"][:k, :x.size] @ (S[:, None] * N_full)
                    soc = N_full @ -np.linalg.lstsq(JN, got[0][:k], rcond=None)[0]
                    if m:  # and the constraint rows' Newton step from the trial
                        soc = soc - np.linalg.lstsq(Jc, got[0][k:], rcond=None)[0]
                    corrected, got2 = trial(x_new + S * soc)
                    nfev += 1
                    if got2 is not None and merit_of(got2[0]) < cost_new:
                        x_new, got, cost_new = corrected, got2, merit_of(got2[0])
                        rho = (merit - cost_new) / predicted if predicted > 0 else -np.inf
                        if verbose:
                            print(f"  second-order correction: cost {cost_new:.10e} rho {rho:.3f}", flush=True)
                tiny = predicted <= floor  # below the cost's repeatability: accept a non-increase
                if rho < 0.25 and not tiny:
                    radius = 0.25 * length
                elif rho > 0.75 and length > 0.99 * radius:
                    radius = 2.0 * radius
                    # a very successful step at the radius: go further along the same model while it keeps
                    # gaining (each extension one equilibrium, no new derivatives) -- long valleys of the
                    # low-order shape modes at finite beta take one model instead of one per doubling
                    while newton > length:
                        step2, predicted2 = model_step(radius)
                        x_try, got_try = trial(x + S * step2)
                        nfev += 1
                        if got_try is None or merit_of(got_try[0]) >= cost_new:
                            break
                        x_new, got, cost_new, length = x_try, got_try, merit_of(got_try[0]), np.linalg.norm(step2)
                        if verbose:
                            print(f"  extended: |s| {length:.2e} cost {cost_new:.10e} predicted {predicted2:.2e}",
                                  flush=True)
                        radius = 2.0 * radius
                if rho > 0.1 or (tiny and cost_new <= merit):
                    gained = merit - cost_new
                    x, (r, aux), accepted = x_new, got, True
                    if callback is not None:
                        callback(np.asarray(x).copy())
                    if fresh and noise > 0 and gained <= noise and feasible:
                        # the exact model's step gained no more than the cost's measured repeatability: what is
                        # left is below what the equilibria resolve
                        converged, status = True, "converged: gain below the cost's repeatability"
                    if rho < 0.25 and not tiny:  # a poor model: the next better one at the new point
                        need = "exact" if model == "gauss_newton" else "gauss_newton"
                    break
                if not fresh:  # a model that fails: a fresh Jacobian (quasi-Newton), else the exact Hessian
                    need = "exact" if model == "gauss_newton" else "gauss_newton"
                    break
                if tiny:  # the exact model's gain is below the cost's repeatability: nothing left to resolve
                    break
            if converged:
                break
            if not accepted and fresh:
                # no step gains although the exact model predicts one: the gain is below the cost's noise
                converged = bool(w[0] > 0) and evaluated and feasible  # trials without equilibria are no noise floor
                status = ("converged: noise floor" if converged else "stalled: no acceptable step") + \
                    f" (predicted gain {0.5 * decrement / cost:.1e} of the cost)"
                break
        if verbose:
            print(f"Newton finished: {status} after {len(history)} Hessians, {nfev} evaluations "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
        return dict(x=x, rows=r, aux=aux, jacobian=None if so is None else so["J"],
                    hessian=None if so is None else so.get("H"), nfev=nfev, njev=len(history), converged=converged,
                    status=status, accepted=len(history), history=np.asarray(history), radius=radius)

    def _print_weakest(self, direction):
        """Print the largest boundary-mode components of ``direction`` (the reduced Hessian's weakest)."""
        labels = [f"h({m},{n})" for m, n in self.boundary_modes]
        v = direction[:len(labels)] / np.max(np.abs(direction[:len(labels)]))
        top = np.argsort(-np.abs(v))[:6]
        print("  weakest:", " ".join(f"{labels[k]}:{v[k]:+.2f}" for k in top), flush=True)

    def _bundle(self, field):
        """``field`` with the run's constants, as the compiled rows take it (see :class:`_Bundle`)."""
        if isinstance(field, _Bundle) or not hasattr(self, "_consts"):
            return field
        return _Bundle(field, self._consts)


class LeastSquaresFreeBoundaryProblem(ThreeTermsFreeBoundaryProblem):
    """:class:`~vmex.core.freeboundary_vc.ThreeTermsFreeBoundaryProblem` on the least-squares free boundary.

    Every trial is :meth:`LeastSquaresFreeBoundaryModel.solve_boundary`; ``b*(x)`` minimizes
    ``f(b, x) = 1/2 |r|^2`` (on the constraint rows' manifold), so a row ``h`` (loss or quantity) has the design
    gradient

        dh/dx = dh/dx|_b - mu^T f_bx,   (N^T W N) mu_N = N^T dh/db,   mu = N mu_N,

    with ``W`` the exact (Lagrangian) Hessian and ``N`` a null-space basis of the constraint rows' Jacobian
    (:meth:`_exact_gradients`).  Build with ``FreeBoundaryProblem.from_loss(..., boundary_condition=
    "three_terms", three_terms_options=dict(method="least_squares", ...))``.
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None, scales=None, names=None,
                  quantities=(), parameter_quantities=(), boundary_max_nfev=100, **model_options):
        """Solve the seed design's free boundary (keywords as the Galerkin problem's)."""
        functions = (loss, field_from_parameters, *quantities, *parameter_quantities)
        if plasma_from_parameters is not None:
            functions += (plasma_from_parameters,)
        if not all(map(callable, functions)):
            raise TypeError("loss, parameter maps and quantities must be callable")
        if not jax.config.x64_enabled:
            raise ValueError("free-boundary optimization requires JAX_ENABLE_X64=1")
        model = LeastSquaresFreeBoundaryModel(inp, **model_options)
        return cls(model, loss, x0, field_from_parameters=field_from_parameters,
                   plasma_from_parameters=plasma_from_parameters, scales=scales, names=names,
                   quantities=tuple(quantities), parameter_quantities=tuple(parameter_quantities),
                   boundary_max_nfev=boundary_max_nfev)

    def __init__(self, model, loss, x0, **options):
        from .implicit import _assemble, _dof_projector, _edge_mask, runtime_from_params

        field_from_parameters = options["field_from_parameters"]
        model.require_converged = True
        edge, cfg = _edge_mask(model.cfg), model.cfg

        def coil_mixed(state, mask, params, x, shift, dz_mu, dp_mu):
            """``d/d eps grad_x 1/2 |rows(field(x)) + shift|^2`` along the boundary response ``(dz_mu, dp_mu)``
            (``shift`` of :meth:`LeastSquaresFreeBoundaryModel.second_order`: the Lagrangian's mixed derivative)."""
            frozen = jax.lax.stop_gradient(state)
            P = _dof_projector(cfg, mask)

            def grad_x(z, prm):
                def cost(xx):
                    r = model.rows(_assemble(z, runtime_from_params(prm, cfg), frozen, P, edge), prm,
                                   field_from_parameters(xx))
                    r = r if shift is None else r + shift
                    return 0.5 * jnp.sum(r * r)
                return jax.grad(cost)(x)

            return jax.jvp(grad_x, (P(frozen), params), (P(dz_mu), dp_mu))[1]

        def constraint_field_jacobian(state, params, x):
            """``d r_c / dx`` through the field at a fixed equilibrium (the net-current row's G_coil).

            One linearization, its tangent pushed for two design directions at a time (``map_columns``): ``jacfwd``
            carried every design direction through the whole rows (the interface field on the grid included) at
            once -- 16 GiB for a QI A8 2.5 % design (357 directions) -- to keep only the constraint rows."""
            _, push = jax.linearize(lambda xx: model.rows(state, params, field_from_parameters(xx))[
                -model.n_constraint_rows:], x)
            return map_columns(push, jnp.eye(x.size, dtype=x.dtype)).T

        self._coil_mixed = jax.jit(coil_mixed)
        self._constraint_field_jacobian = jax.jit(constraint_field_jacobian)
        super().__init__(model, loss, x0, **options)

    def _solve(self, x):
        """:meth:`ThreeTermsFreeBoundaryProblem._solve`, the boundary's certificate built with what
        :meth:`_exact_gradients` needs at the answer (every Hessian column and the plasma directions' mixed
        columns): the design gradient then reuses it instead of linearizing again."""
        xj = jnp.asarray(np.asarray(x, dtype=float))
        self.model.certificate_extra = ((lambda params: self._plasma_directions(xj)) if self._plasma.size
                                        else (lambda params: None))
        try:
            return super()._solve(x)
        finally:
            self.model.certificate_extra = None

    def _linearize(self, x, sol):
        """Every row's design gradient at ``sol`` (cached): :meth:`_exact_gradients`."""
        if "gradients" in sol:
            return sol["gradients"]
        return self._exact_gradients(x, sol)

    def _exact_gradients(self, x, sol):
        """Design gradients of every row with the exact Hessian of the boundary least squares.

        ``b*(x)`` minimizes ``f(b, x) = 1/2 |r|^2``; with ``H = f_bb`` (exact, :meth:`ThreeTermsFreeBoundaryModel.
        second_order`) and ``H mu = dh/db`` the row ``h`` has ``dh/dx = dh/dx|_b - mu^T f_bx``.  ``f_bx`` along
        ``mu`` is a forward-over-reverse pass through the field and, for plasma parameters, the Hessian columns
        of their directions.  (Gauss-Newton would replace ``H`` by ``J^T J`` and ``f_bx`` by ``J_b^T J_x``, wrong
        by ``sum_i r_i d^2 r_i`` when the rows do not vanish: by up to 21 % at finite beta.)  With the
        constraint rows ``r_c(b, x) = 0``, ``H`` is the Lagrangian Hessian on their null space and the
        constraint rows' dependence on the plasma parameters adds the response off the null space.
        """
        model, xj = self.model, jnp.asarray(x)
        field = self.field(xj)
        extra = self._plasma_directions(xj) if self._plasma.size else None
        so = model.second_order(sol["x"], sol["aux"], field, extra=extra)
        state, mask, params, _ = so["aux"]
        nb = model.x0.size
        sol.update(aux=so["aux"], state=state, mask=mask, params=params, rows=so["r"],
                   values=np.asarray(self._rows(state, params, xj)))
        self.stats["linearizations"] += 1
        if self._warm_threads[1] is not None:  # the pullback and direct Jacobian
            self._warm_threads[1].join()
        state, mask, params = model._im._canonical_arrays(model.cfg, (state, mask, params))
        factors = model._im.block_factors_at(model.cfg, params, state, mask)
        params_bar = self._pullback(state, mask, params, xj, factors)
        G_b, G_x = map(np.asarray, self._coordinates(params, jnp.asarray(sol["x"]), xj, params_bar))
        # The boundary solves min f s.t. r_c(b, x) = 0 (the constraint rows): the sensitivity
        # lives in the null space N of their Jacobian B, with the Lagrangian Hessian W (ESS-scaled).
        S = np.asarray(model.x_scale, dtype=float)
        J, r, m = so["J"], so["r"], model.n_constraint_rows
        k = r.size - m
        N, _, _, B = model._reduced(S * (J[:k, :nb].T @ r[:k]), J[k:, :nb] * S[None] if m else None)
        W = S[:, None] * so["H"] * S[None]
        w, Q = np.linalg.eigh(N.T @ W @ N)
        if w[0] <= 0:
            raise TrialRejected(f"the reduced boundary Hessian is not positive definite (smallest eigenvalue "
                                f"{w[0]:.3e})")
        NQ = N @ Q
        mu = S[:, None] * (NQ @ ((NQ.T @ (S[:, None] * G_b.T)) / w[:, None]))  # a column per row
        gradients = np.asarray(self._direct(state, params, xj)) + G_x
        dz_b = jax.tree.map(lambda a: a[:nb], so["dz"])
        dp_b = jax.tree.map(lambda a: a[:nb], so["batch"])
        shift = None if so["shift"] is None else jnp.asarray(so["shift"])
        for row, column in enumerate(mu.T):
            mj = jnp.asarray(column)
            dz_mu = jax.tree.map(lambda a: jnp.tensordot(mj, a, axes=1), dz_b)
            dp_mu = jax.tree.map(lambda a: jnp.tensordot(mj, a, axes=1), dp_b)
            with vc.remat_scans():
                gradients[row] -= np.asarray(self._coil_mixed(state, mask, params, xj, shift, dz_mu, dp_mu))
        if self._plasma.size:
            gradients[:, self._plasma] -= mu.T @ so["C"]
        if m:
            # the constraint rows move with the design (G_coil with the coils, the currents' rows with the plasma
            # parameters): the boundary's response has the part -B^+ dr_c/dx off the null space, which enters
            # through eta = (B^+)^T (S dh/db - W mu / S)
            eta = np.linalg.lstsq(B.T, S[:, None] * G_b.T - W @ (mu / S[:, None]), rcond=None)[0]
            J_cx = np.array(self._constraint_field_jacobian(state, params, xj))
            if self._plasma.size:
                J_cx[:, self._plasma] += J[k:, nb:]
            gradients -= eta.T @ J_cx
        self._anchor = dict(x=self._x(x).copy(), b=sol["x"], J_b=J[:, :nb], J_p=J[:, nb:], state=state,
                            params=params, cost=0.5 * so["r"][:k] @ so["r"][:k])
        sol["gradients"] = gradients
        sol["hessian_eigenvalues"] = w
        return gradients


def solve_free_boundary_least_squares(inp: VmecInput, *, external_field: Any = None, mgrid_path=None,
                                      initial_boundary: VmecInput | None = None, max_mode: int | None = None,
                                      nphi: int | None = None, ntheta: int | None = None, digits: int = 4,
                                      max_nfev: int = 60, quadrature: tuple[int, int] | None = None,
                                      trial_ftol: float | None = None, chunk: int = 8, bootstrap=None,
                                      bootstrap_helicity: int | None = None, previous=None,
                                      verbose: int = 0):
    """Solve the least-squares free boundary of a deck.

    This is :func:`~vmex.core.freeboundary_vc.solve_free_boundary_three_terms` with ``method="least_squares"``,
    and takes the same keywords.

    Parameters
    ----------
    inp
        The deck; its boundary fixes the coordinates (the displacement along its radial lines).
    external_field, mgrid_path
        The coils' field, or the mgrid file it is read from (default: the deck's).
    initial_boundary
        The start, as a deck whose surface is a displacement of ``inp``'s
        (:meth:`~vmex.core.freeboundary_vc.ThreeTermsFreeBoundaryModel.displacement_of`); default ``inp``.
    max_nfev
        Most models (derivative evaluations) of the trust-region Newton solve.
    previous
        A former result: its model and boundary are reused (a warm start of the same deck).
    max_mode, nphi, ntheta, digits, quadrature, trial_ftol, chunk, bootstrap, bootstrap_helicity
        :class:`LeastSquaresFreeBoundaryModel` options.
    verbose
        Print every model (``2``: also the weakest direction).

    Returns
    -------
    scipy.optimize.OptimizeResult
        ``x`` (the boundary coordinates), ``fun`` (rows), ``jac``, ``cost``, ``success``, ``status``, ``history``,
        ``input`` (the solved deck), ``equilibrium`` (re-solved to the report tolerance), ``model``,
        ``boundary_residual`` and ``initial_boundary_residual``.

    Raises
    ------
    NotImplementedError
        For ``lasym = True``.
    ValueError
        For a nonzero edge pressure.
    """
    from scipy.optimize import OptimizeResult

    from .freeboundary import _external_field_from_input
    from .implicit import input_with_params
    from .optimize import solve_equilibrium

    if bool(inp.lasym):
        raise NotImplementedError("the virtual-casing free boundary supports lasym = False only")
    p_edge, p_max = _edge_pressure(inp)
    if abs(p_edge) > 1e-8 * p_max:
        raise ValueError(f"edge pressure is {p_edge:g} Pa: without a sheet current the "
                         "boundary pressure must vanish")
    if external_field is None:
        external_field = _external_field_from_input(inp, mgrid_path)
    if previous is not None:
        model, x0 = previous.model, previous.x
    else:
        model = LeastSquaresFreeBoundaryModel(
            inp, max_mode=max_mode, nphi=int(nphi or 48), ntheta=int(ntheta or 48), digits=digits,
            quadrature=quadrature, trial_ftol=trial_ftol, chunk=chunk, bootstrap=bootstrap,
            bootstrap_helicity=bootstrap_helicity)
        x0 = None
        if initial_boundary is not None:
            x0 = np.r_[model.displacement_of(initial_boundary), model.x0[model.n_boundary:]]
    initial = model.boundary_residual(model.seed[0], model.params0, external_field)
    out = model.solve_boundary(model.params0, external_field, x0=x0, max_iter=max_nfev, verbose=verbose)
    state, _, params, _ = out["aux"]
    boundary = input_with_params(model.fixed, params)
    # from the rows' own state on its own rung, and an equilibrium short of the tolerance is an error (as Galerkin's)
    equilibrium = solve_equilibrium(
        boundary, restart_from=state, forward_ftol=min(_REPORT_FTOL, float(np.asarray(boundary.ftol_array)[-1])),
        forward_max_iterations=max(int(np.asarray(boundary.niter_array)[-1]), 200000), raise_on_max_iterations=True)
    final = model.boundary_residual(equilibrium.solution, params, external_field)
    return OptimizeResult(
        x=out["x"], fun=out["rows"], jac=out["jacobian"], cost=0.5 * out["rows"] @ out["rows"], nfev=out["nfev"],
        njev=out["njev"], success=bool(out["converged"]), status=out["status"], message=out["status"],
        history=out["history"], input=boundary, equilibrium=equilibrium, model=model, aux=out["aux"],
        boundary_residual=final, normal_outside_modes=model.normal_outside_modes(equilibrium.solution, params,
                                                                                 external_field),
        initial_boundary_residual=initial)
