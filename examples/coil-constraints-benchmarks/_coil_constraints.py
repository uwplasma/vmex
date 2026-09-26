"""Shared differentiable coil inequalities and independent endpoint checks.

Geometry uses raw metre-valued coefficients, independently of field quadrature.
All returned inequality rows are dimensionless and favourable-positive.
Distance constraints are on polygonal curves / a sampled moving surface;
verification refines both and reports that scope, never a winding-pack claim.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar

import parameters as P

CURVATURE_POINTS = max(512, 64*P.COIL_ORDER)
DISTANCE_POINTS = max(128, 16*P.COIL_ORDER)
SURFACE_GRID = (61, 64)
VERIFY_SURFACE_GRID = (121, 128)
VERIFY_POINTS = max(2048, 256*P.COIL_ORDER)
SELF_CLEARANCE = 1e-6  # numerical nonintersection guard, metres


def geometry(raw, points):
    """Analytic Fourier coordinates, speed and curvature; raw shape (coil,xyz,mode)."""
    t = jnp.arange(points) / points
    w = 2*jnp.pi*jnp.arange(1, (raw.shape[-1]+1)//2)
    sn, cs = jnp.sin(t[:, None]*w), jnp.cos(t[:, None]*w)
    a, b = raw[:, :, 1::2], raw[:, :, 2::2]
    xyz = raw[:, None, :, 0] + jnp.einsum('tk,ijk->itj', sn, a) + jnp.einsum('tk,ijk->itj', cs, b)
    v = jnp.einsum('tk,ijk->itj', cs*w, a) - jnp.einsum('tk,ijk->itj', sn*w, b)
    acc = -jnp.einsum('tk,ijk->itj', sn*w*w, a) - jnp.einsum('tk,ijk->itj', cs*w*w, b)
    speed = jnp.sqrt(jnp.sum(v*v, axis=-1)+1e-30)
    cross = jnp.cross(v, acc)
    curvature = jnp.sqrt(jnp.sum(cross*cross, axis=-1)+1e-30)/speed**3
    return xyz, speed, curvature


def segment_distances(a, b):
    """All distances between two closed polygons' segments, including interiors."""
    u, v = jnp.roll(a, -1, axis=0)-a, jnp.roll(b, -1, axis=0)-b
    w = a[:, None]-b[None, :]
    aa, bb = jnp.sum(u*u, axis=-1)[:, None], jnp.sum(v*v, axis=-1)[None, :]
    uv = jnp.einsum('ik,jk->ij', u, v)
    uw, vw = jnp.sum(u[:, None]*w, axis=-1), jnp.sum(v[None, :]*w, axis=-1)
    aa, bb = jnp.maximum(aa, 1e-30), jnp.maximum(bb, 1e-30)
    den = aa*bb-uv*uv
    safe = jnp.where(den > 1e-24, den, 1.0)
    s, t = (uv*vw-bb*uw)/safe, (aa*vw-uv*uw)/safe
    def distance(s, t):
        q = w+s[..., None]*u[:, None]-t[..., None]*v[None, :]
        return jnp.sum(q*q, axis=-1)
    candidates = [distance(jnp.zeros_like(uw), jnp.clip(vw/bb, 0, 1)),
                  distance(jnp.ones_like(uw), jnp.clip((vw+uv)/bb, 0, 1)),
                  distance(jnp.clip(-uw/aa, 0, 1), jnp.zeros_like(uw)),
                  distance(jnp.clip((uv-uw)/aa, 0, 1), jnp.ones_like(uw)),
                  jnp.where((den > 1e-24)&(s>=0)&(s<=1)&(t>=0)&(t<=1), distance(s,t), jnp.inf)]
    return jnp.sqrt(jnp.min(jnp.stack(candidates), axis=0)+1e-30)


def separations(points):
    """Minimum intercoil and nonadjacent self-segment distance, all symmetry copies."""
    pairs = jnp.asarray([(i,j) for i in range(len(points)) for j in range(i+1,len(points))])
    inter = jax.lax.map(lambda ij: jnp.min(segment_distances(points[ij[0]], points[ij[1]])), pairs)
    n = points.shape[1]
    diff = jnp.abs(jnp.arange(n)[:,None]-jnp.arange(n)[None,:])
    nonadjacent = jnp.minimum(diff, n-diff)>1
    own = jax.lax.map(lambda p: jnp.min(jnp.where(nonadjacent, segment_distances(p,p), jnp.inf)), points)
    return jnp.min(inter), jnp.min(own)


def coil_metrics(coils, *, curvature_points=CURVATURE_POINTS, distance_points=DISTANCE_POINTS):
    raw = coils.curves.curves
    _, speed, curvature = geometry(raw[:P.N_COILS], curvature_points)
    points, _, _ = geometry(raw, distance_points)
    cc, own = separations(points)
    return dict(length=jnp.mean(speed, axis=1), peak=jnp.max(curvature, axis=1),
                msc=jnp.sum(curvature**2*speed, axis=1)/jnp.sum(speed, axis=1),
                coil_distance=cc, self_distance=own, min_speed=jnp.min(speed,axis=1))


def coil_inequalities(coils):
    m = coil_metrics(coils)
    return jnp.concatenate(((P.LENGTH_LIMIT-P.LENGTH_MARGIN-m['length'])/P.LENGTH_LIMIT,
        (P.CURVATURE_LIMIT-P.CURVATURE_MARGIN-m['peak'])/P.CURVATURE_LIMIT,
        (P.MSC_LIMIT-P.MSC_MARGIN-m['msc'])/P.MSC_LIMIT,
        jnp.atleast_1d((m['coil_distance']-P.COIL_DISTANCE_LIMIT-P.DISTANCE_MARGIN)/P.COIL_DISTANCE_LIMIT),
        jnp.atleast_1d((m['self_distance']-SELF_CLEARANCE)/P.COIL_DISTANCE_LIMIT),
        (m['min_speed']-1e-4)/P.LENGTH_LIMIT))


def surface_distance(coils, surface):
    """Sampled coil-to-moving-surface clearance; JAX differentiates both sides."""
    points, _, _ = geometry(coils.curves.curves, DISTANCE_POINTS)
    targets = surface.gamma.reshape(-1,3)
    # Mapping bounds memory and preserves exact differentiation of the active min.
    return jnp.min(jax.lax.map(lambda p: jnp.sqrt(jnp.min(jnp.sum((p-targets)**2,axis=1))+1e-30),
                               points.reshape(-1,3)))


def constraint(coils_from_x):
    """Pure-coil rows require no equilibrium solves or adjoint right-hand sides."""
    from scipy.optimize import NonlinearConstraint
    fun = jax.jit(lambda x: coil_inequalities(coils_from_x(x)))
    jac = jax.jit(jax.jacrev(fun))
    return NonlinearConstraint(lambda x: np.asarray(fun(jnp.asarray(x))), 0, np.inf,
                               jac=lambda x: np.asarray(jac(jnp.asarray(x))))


def verify(coils, surface, path=None):
    """Independent dense CPU quadrature, refined peaks and polygon distance checks.

    Positive gates qualify only the declared sampling, not finite-build coils.
    Curvature peaks are refined in every dense-grid local maximum's bracket.
    """
    from scipy.spatial import cKDTree
    raw = np.asarray(coils.curves.curves)
    def numpy_geometry(a, t):
        t = np.atleast_1d(t)
        w = 2*np.pi*np.arange(1,(a.shape[-1]+1)//2)
        sn,cs=np.sin(t[:,None]*w),np.cos(t[:,None]*w)
        xyz=a[:,0]+sn@a[:,1::2].T+cs@a[:,2::2].T
        v=(cs*w)@a[:,1::2].T-(sn*w)@a[:,2::2].T
        acc=-(sn*w*w)@a[:,1::2].T-(cs*w*w)@a[:,2::2].T
        speed=np.linalg.norm(v,axis=1)
        k=np.linalg.norm(np.cross(v,acc),axis=1)/np.maximum(speed,1e-30)**3
        return xyz,speed,k
    rows=[]
    for a in raw[:P.N_COILS]:
        t=np.arange(VERIFY_POINTS)/VERIFY_POINTS
        _,speed,k=numpy_geometry(a,t)
        peaks=np.where((k>=np.roll(k,1))&(k>=np.roll(k,-1)))[0]
        refined=[-minimize_scalar(lambda x:-numpy_geometry(a,x)[2][0],
                    bounds=((j-1)/VERIFY_POINTS,(j+1)/VERIFY_POINTS),method='bounded',
                    options={'xatol':1e-13}).fun for j in peaks]
        _,v2,k2=numpy_geometry(a,np.arange(2*VERIFY_POINTS)/(2*VERIFY_POINTS))
        rows.append(dict(length_m=float(speed.mean()),msc_per_m2=float(np.average(k*k,weights=speed)),
            rms_per_m=float(np.sqrt(np.average(k*k,weights=speed))),peak_per_m=float(max(refined)),
            minimum_speed=float(speed.min()),msc_refinement_change=float(abs(np.average(k*k,weights=speed)-np.average(k2*k2,weights=v2)))))
    # Refine nearest segment pairs from dense points; account for unsampled
    # segments with a conservative Fourier chord error bound.
    n=VERIFY_POINTS
    points=[numpy_geometry(a,np.arange(n)/n)[0] for a in raw]
    w=2*np.pi*np.arange(1,(raw.shape[-1]+1)//2)
    chord=np.sum(w*w*(np.linalg.norm(raw[:,:,1::2],axis=1)+np.linalg.norm(raw[:,:,2::2],axis=1)),axis=1)/(8*n*n)
    cc=np.inf;own=np.inf
    # Dense node distances minus a rigorous speed cover bound give conservative
    # lower bounds on continuous-curve separation. No false passing from nodes.
    speed_bound=np.sum(w*(np.linalg.norm(raw[:,:,1::2],axis=1)+np.linalg.norm(raw[:,:,2::2],axis=1)),axis=1)
    for i,a in enumerate(points):
        tree=cKDTree(a)
        for j in range(i):
            dist,_=tree.query(points[j]);cc=min(cc,float(dist.min()-(speed_bound[i]+speed_bound[j])/(2*n)))
        # Nonlocal self-intersections: refine a neighbourhood of every segment
        # whose bounding-sphere centres could intersect. Adjacent segments share
        # endpoints and are excluded by definition.
        mid=(a+np.roll(a,-1,axis=0))/2
        radius=np.linalg.norm(np.roll(a,-1,axis=0)-a,axis=1)/2
        pairs=cKDTree(mid).query_pairs(2*radius.max()+2*chord[i]+SELF_CLEARANCE)
        for j,k in pairs:
            if min(abs(j-k),n-abs(j-k))<=1:
                continue
            from scipy.optimize import lsq_linear
            u=a[(j+1)%n]-a[j];v=a[(k+1)%n]-a[k]
            fit=lsq_linear(np.column_stack((u,-v)),a[k]-a[j],bounds=(0.,1.),tol=1e-13)
            own=min(own,float(np.linalg.norm(a[j]+fit.x[0]*u-a[k]-fit.x[1]*v)-2*chord[i]))
    # Independently minimize distance in continuous coil/surface coordinates.
    # Dense local minima supply seeds; this is a numerical check, not a proof
    # of global separation or a finite-build winding-pack certificate.
    from scipy.optimize import minimize
    target = np.asarray(surface.gamma).reshape(-1,3)
    tree=cKDTree(target)
    theta,phi=np.asarray(surface.theta2d).ravel(),np.asarray(surface.phi2d).ravel()
    rc,zs=np.asarray(surface.rc),np.asarray(surface.zs)
    m,n=np.asarray(surface.xm),np.asarray(surface.xn)
    def surface_point(theta,phi):
        phase=m*theta-n*phi
        r=rc@np.cos(phase);z=zs@np.sin(phase)
        return np.array([r*np.cos(phi),r*np.sin(phi),z])
    cs=np.inf;refined_cs=np.inf;refinement_success=True
    for a,xyz in zip(raw,points):
        dist,indices=tree.query(xyz)
        cs=min(cs,float(dist.min()))
        minima=np.flatnonzero((dist<=np.roll(dist,1))&(dist<=np.roll(dist,-1)))
        # Refine every sampled local minimum (capped only for a flat curve).
        if len(minima)>64:
            minima=np.unique(np.r_[minima[np.argsort(dist[minima])[:32]],minima[::max(1,len(minima)//32)]])
        for index in minima:
            guess=[index/VERIFY_POINTS,theta[indices[index]],phi[indices[index]]]
            def squared_distance(v):
                difference=numpy_geometry(a,v[0])[0][0]-surface_point(v[1],v[2])
                return difference@difference
            fit=minimize(squared_distance,guess,method='BFGS',
                         options={'gtol':1e-9,'maxiter':100})
            # Numerical finite differences can end with precision-loss after
            # convergence: check gradient and finite distance explicitly.
            refinement_success &= bool(np.isfinite(fit.fun) and np.linalg.norm(fit.jac)<1e-5)
            refined_cs=min(refined_cs,float(np.sqrt(max(0,fit.fun))))
    refined_cs=min(refined_cs,cs)
    checks=dict(length=all(r['length_m']<=P.LENGTH_LIMIT for r in rows),
        peak_curvature=all(r['peak_per_m']<=P.CURVATURE_LIMIT for r in rows),
        msc=all(r['msc_per_m2']<=P.MSC_LIMIT for r in rows),
        regular_curves=all(r['minimum_speed']>1e-4 for r in rows),
        coil_coil=cc>=P.COIL_DISTANCE_LIMIT,
        self_intersection=own>=SELF_CLEARANCE,
        refined_coil_surface=refined_cs>=P.COIL_SURFACE_DISTANCE_LIMIT,
        clearance_refinement=refinement_success,
        msc_resolution=all(r['msc_refinement_change']<1e-6 for r in rows))
    result=dict(coils=rows,coil_coil_lower_bound_m=cc,
        nonlocal_self_distance_lower_bound_m=None if np.isinf(own) else own,
        sampled_coil_surface_distance_m=cs,refined_coil_surface_distance_m=refined_cs,checks=checks,
        all_reported_checks_pass=all(checks.values()),continuous_surface_clearance_certified=False,
        scope='Filament geometry; independently refined curvature and MSC; conservative intercoil bound; dense nonlocal self-segment intersection check; multistart continuous moving-surface clearance (numerical, not certified globally).',
        curvature_points=VERIFY_POINTS,surface_grid=list(surface.gamma.shape[:2]))
    if path is not None:
        Path(path).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
