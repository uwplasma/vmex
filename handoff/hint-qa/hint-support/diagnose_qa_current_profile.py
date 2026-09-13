"""Snapshot current/drive diagnostics; this program never runs an equilibrium.

Example: .venv/bin/python local-support/diagnose_qa_current_profile.py RUN \
  --profile-csv ORIGINAL_CASE/current_mapping.csv --target-current -95778.35 \
  --output RUN/current-profile-diagnostic-001.json

The reconstructed drive is a snapshot candidate, not a saved native Jnet.
Full-vector J-minus-drive RMS includes pressure-driven perpendicular current
and other equilibrium current structure absent from the parallel drive. It is
not expected to vanish and is not a standalone resistivity-selection gate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

MU0 = 4e-7*np.pi


def derivative(v, h, axis):
    return (-np.roll(v, -2, axis)+8*np.roll(v, -1, axis)
            -8*np.roll(v, 1, axis)+np.roll(v, 2, axis))/(12*h)


def diagnose(R, Z, phi, B, vacuum, P, wall, profile_s, profile, target,
             paxis=None, pedge=.01, bins=10):
    """P is mu0*p, B cylindrical tesla; returns JSON-compatible diagnostics."""
    shape = (len(phi), len(Z), len(R))
    if min(shape)<5 or B.shape!=shape+(3,) or vacuum.shape!=B.shape or P.shape!=shape or wall.shape!=shape:
        raise ValueError("Inconsistent grid/field shapes or insufficient stencil points")
    if not all(np.isfinite(x).all() for x in (R,Z,phi,B,vacuum,P,profile_s,profile)):
        raise ValueError("Nonfinite input")
    if not np.isfinite(target) or target==0 or not 0<=pedge<1 or bins<1:
        raise ValueError("Invalid target current, pedge or bins")
    steps = [x[1]-x[0] for x in (R,Z,phi)]
    if R[0]<=0 or any(h<=0 or not np.allclose(np.diff(x),h,rtol=1e-11,atol=1e-13) for x,h in zip((R,Z,phi),steps)):
        raise ValueError("Expected increasing uniform cylindrical grid")
    dr,dz,dp = steps
    inferred_nfp = 2*np.pi/(dp*len(phi))
    if abs(phi[0])>1e-12 or not np.isclose(inferred_nfp,round(inferred_nfp)) or inferred_nfp<1:
        raise ValueError("Expected one endpoint-excluded toroidal period starting at zero")
    if len(profile_s)<2 or len(profile)!=len(profile_s) or not np.allclose(profile_s,np.linspace(0,1,len(profile_s)),atol=1e-12,rtol=1e-12):
        raise ValueError("Native linear current profile requires a uniform table on [0,1]")
    paxis_used = float(P.max()) if paxis is None else float(paxis)
    if not np.isfinite(paxis_used) or paxis_used<=0:
        raise ValueError("Positive finite axis pressure required")
    rr=R[None,None,:]
    interior=np.zeros(shape,dtype=bool);interior[:,2:-2,2:-2]=True
    br,bp,bz=np.moveaxis(B-vacuum,-1,0)
    J=np.stack((derivative(bz,dp,0)/rr-derivative(bp,dz,1),
                derivative(br,dz,1)-derivative(bz,dr,2),
                (derivative(rr*bp,dr,2)-derivative(br,dp,0))/rr),axis=-1)/MU0

    # Match get_psi's rectangular phi=0 pressure-enclosed toroidal-flux sum.
    # Sort once instead of performing 501 whole-plane scans.
    levels=np.linspace(paxis_used*pedge,paxis_used,501)
    cutP=P[0,2:-2,2:-2].ravel();cutB=B[0,2:-2,2:-2,1].ravel()*dr*dz
    order=np.argsort(cutP);sortedP=cutP[order]
    tail=np.r_[np.cumsum(cutB[order][::-1])[::-1],0.]
    psi=tail[np.searchsorted(sortedP,levels,side="left")]
    if abs(psi[0])<=np.finfo(float).eps*np.sum(abs(cutB)):
        raise ValueError("Pressure-enclosed toroidal flux is zero or cancellation-dominated")
    normalized_psi=psi/psi[0]
    if np.any(np.diff(normalized_psi)>1e-10) or normalized_psi.min()<-1e-10:
        raise ValueError("Pressure-enclosed flux is not monotone: cannot define ordered profile bins")
    s=np.interp(P,levels,normalized_psi)
    active=s<1.
    raw=np.where(active,np.interp(s,profile_s,profile),0.)
    cut_integral=float(np.sum(raw[0,2:-2,2:-2]*B[0,2:-2,2:-2,1])*dr*dz)
    if not np.isfinite(cut_integral) or abs(cut_integral)<=np.finfo(float).eps*np.sum(abs(raw[0]*B[0,...,1]))*dr*dz:
        raise ValueError("Zero/cancellation-dominated drive normalization")
    scale=target/cut_integral
    drive=scale*raw[...,None]*B
    dv=np.broadcast_to(rr*dr*dz*2*np.pi/len(phi),shape)
    B2=np.sum(B*B,axis=-1);JB=np.sum(J*B,axis=-1);driveJB=np.sum(drive*B,axis=-1)

    def metrics(mask):
        mask=mask & interior
        volume=float(dv[mask].sum())
        result={"cells":int(mask.sum()),"volume_m3":volume}
        for name,field in (("attained",J),("candidate_drive",drive)):
            cuts=np.sum(np.where(mask,field[...,1],0.),axis=(1,2))*dr*dz
            result[name+"_current_A_by_cut"]=cuts.tolist()
            result[name+"_mean_current_A"]=float(cuts.mean())
            result[name+"_abs_toroidal_current_A_mean"]=float(np.mean(np.sum(np.where(mask,abs(field[...,1]),0.),axis=(1,2))*dr*dz))
        if volume:
            b2integral=float(np.sum(B2[mask]*dv[mask]))
            result.update(attained_JdotB_volume_mean_A_T_per_m2=float(np.sum(JB[mask]*dv[mask])/volume),
                          candidate_drive_JdotB_volume_mean_A_T_per_m2=float(np.sum(driveJB[mask]*dv[mask])/volume),
                          attained_JdotB_over_B2_A_per_T_m2=float(np.sum(JB[mask]*dv[mask])/b2integral) if b2integral>0 else None,
                          candidate_drive_JdotB_over_B2_A_per_T_m2=float(np.sum(driveJB[mask]*dv[mask])/b2integral) if b2integral>0 else None,
                          current_difference_rms_A_per_m2=float(np.sqrt(np.sum(np.sum((J-drive)[mask]**2,axis=-1)*dv[mask])/volume)))
        return result

    rows=[]
    for lo,hi in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
        rows.append(dict(s_lower=float(lo),s_upper_excluded=float(hi),**metrics((s>=lo)&(s<hi))))
    return dict(convergence_certified=False,drive_is_saved_native_field=False,
                method="curl(B-Bvac)/mu0; native-style 501 pressure-enclosed toroidal-flux levels at phi=0; snapshot candidate C*profile(s)*B normalized on the same cut",
                paxis_mu0_Pa=paxis_used,paxis_source="snapshot maximum proxy" if paxis is None else "explicit supplied axis pressure",
                pedge=pedge,nfp=int(round(inferred_nfp)),target_current_A=target,
                candidate_drive_scale_A_per_T_m2=scale,pressure_enclosed_flux_edge_Wb=float(psi[0]),
                pressure_levels_mu0_Pa=levels.tolist(),normalized_flux_levels=normalized_psi.tolist(),
                profile_bins=rows,
                supports={name:metrics(mask) for name,mask in {
                    "all_interior":interior,"inside_wall":wall,"outside_wall":~wall,
                    "zero_pressure":P<=0,"positive_pressure":P>0,"drive_excluded_s_ge_1":~active}.items()},
                current_difference_interpretation="current_difference_rms_A_per_m2 compares full attained J with the parallel candidate drive. It includes required pressure-driven perpendicular current and other equilibrium current structure; zero is not an equilibrium target, and this RMS alone must not select resistivity or certify convergence.",
                limitations=["Bins are finite-volume pressure-derived label shells, not flux-surface averages or bootstrap closure.",
                             "Labels use the saved snapshot. Native labels are formed at the start of step B and held during magnetic substeps; exact historical Jnet needs saved labels/axis pressure.",
                             "Without --paxis-pa, grid maximum substitutes for native traced-axis pressure; this is an explicitly approximate reconstruction.",
                             "Assumes linear profj, jcuts=1, lpedge=false and no additional current component. CLI profile and target must match the actual run.",
                             "All curl/current reductions exclude two R/Z boundary layers; volume is expanded to full torus, cut currents are not multiplied by nfp.",
                             "Zero pressure or zero imposed drive does not imply zero attained response current; exterior support is reported independently."])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--profile-csv",type=Path,required=True)
    parser.add_argument("--target-current",type=float,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--paxis-pa",type=float)
    parser.add_argument("--pedge",type=float,default=.01)
    parser.add_argument("--bins",type=int,default=10)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output path; existing evidence is never overwritten")
    snapshot=args.run/"final-snapshot.npz"
    with np.load(snapshot) as f:
        R,Z,phi,B,P=[f[k] for k in ("R","Z","phi","B_cyl","P_mu0_Pa")]
    for name in ("vacuum.nc","limiter-12.nc"):
        with Dataset(args.run/name) as f:
            for key,expected in (("rminb",R[0]),("rmaxb",R[-1]),("zminb",Z[0]),("zmaxb",Z[-1]),("mtor",2*np.pi/(len(phi)*(phi[1]-phi[0])))):
                if not np.isclose(float(np.asarray(f[key][:]).item()),expected,rtol=1e-12,atol=1e-12):
                    raise ValueError(f"{name}: mismatched {key}")
            fields=("Bvac_R","Bvac_phi","Bvac_Z") if name=="vacuum.nc" else ("limiter",)
            for key in fields:
                if f[key].dimensions!=("phi","Z","R"):
                    raise ValueError(f"{name}: wrong {key} dimensions")
            if name=="vacuum.nc":
                for key,expected in (("R",R),("Z",Z),("phi",phi)):
                    if not np.allclose(f[key][:],expected,rtol=1e-12,atol=1e-12):
                        raise ValueError(f"{name}: mismatched coordinates")
                vacuum=np.stack([f[k][:] for k in fields],axis=-1)
            else:
                raw=np.asarray(f["limiter"][:])
                if not np.isin(raw,(0,1)).all():raise ValueError("Nonbinary limiter")
                wall=raw>.5
    table=np.genfromtxt(args.profile_csv,delimiter=",",names=True)
    report=diagnose(R,Z,phi,B,vacuum,P,wall,table["s"],table["normalized_lambda"],args.target_current,
                    None if args.paxis_pa is None else args.paxis_pa*MU0,args.pedge,args.bins)
    report["source_sha256"]={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                             (snapshot,args.run/"vacuum.nc",args.run/"limiter-12.nc",args.profile_csv,Path(__file__))}
    with args.output.open("x") as f:json.dump(report,f,indent=2,allow_nan=False);f.write("\n")
    print(args.output)


if __name__=="__main__":main()
