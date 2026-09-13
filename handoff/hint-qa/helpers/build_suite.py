#!/usr/bin/env python3
"""Build the native suite with upstream object ordering in isolated directories."""
import argparse, os, pathlib, subprocess, json, platform
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=['Debug','Release'],default='Debug')
p.add_argument('--only',nargs='+',choices=['HINT','MKVAC','MKFLX','MKLIM','HMAG','GPRTS','MAGVAL'])
p.add_argument('--hint-root',type=pathlib.Path,default=pathlib.Path.cwd())
p.add_argument('--hdf5-prefix',default=os.environ.get('HDF5_PREFIX'))
a=p.parse_args()
if not a.hdf5_prefix:
 p.error('Supply --hdf5-prefix or HDF5_PREFIX for the HDF5 Fortran installation')
r=a.hint_root.resolve()
if not (r/'src/HINT/Makefile.template').exists():
 p.error('--hint-root must contain the maintained HINT src/HINT tree')
fc=os.environ.get('FC','mpif90')
nf=os.environ.get('NF_CONFIG','nf-config')
prefix=subprocess.check_output([nf,'--prefix'],text=True).strip()
hdf=str(pathlib.Path(a.hdf5_prefix).resolve())
flags='-cpp -DNETCDF -ffree-line-length-none -fopenmp '+('-O0 -g -Wall -Wextra -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow' if a.mode=='Debug' else '-O2')
inc=f'-I{hdf}/include -I{prefix}/include'
libs=f'-L{hdf}/lib -L{prefix}/lib -lhdf5hl_fortran -lhdf5_fortran -lhdf5_hl -lhdf5 -lnetcdff -lnetcdf -fopenmp'
makefiles={'HINT':'Makefile.template','MKVAC':'Makefile.gfortran','MKFLX':'Makefile.gfortran','MKLIM':'Makefile.gfortran','HMAG':'Makefile.noplplot','GPRTS':'Makefile.noplplot','MAGVAL':'Makefile.gfortran'}
b=r/('build-'+a.mode.lower());b.mkdir(exist_ok=True)
for name in a.only or makefiles:
 src=r/'src'/name; dest=b/name;dest.mkdir(exist_ok=True)
 for f in src.glob('*.f90'):
  link=dest/f.name
  if not link.exists():link.symlink_to(f)
 if name=='HINT':
  link=dest/'stepa_mod.f90'
  if not link.exists():link.symlink_to(src/'stepa_modules/fline_method_mod.f90')
 (dest/'Makefile').write_bytes((src/makefiles[name]).read_bytes())
 cmd=['make','-j1',f'FC={fc}',f'LINK={fc}',f'FFLAGS={flags}',f'MOD={inc}',f'LIBS={libs}']
 print('Building',name,flush=True)
 with (dest/'build.log').open('w') as log:
  result=subprocess.run(cmd,cwd=dest,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  print((dest/'build.log').read_text()[-7000:]);raise SystemExit(result.returncode)
commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip() if (r/'.git').exists() else ((r/'SOURCE_COMMIT').read_text().strip() if (r/'SOURCE_COMMIT').exists() else None)
(b/'build_config.json').write_text(json.dumps(dict(mode=a.mode,compiler=fc,flags=flags,includes=inc,libraries=libs,platform=platform.platform(),commit=commit),indent=2)+'\n')
