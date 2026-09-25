#!/usr/bin/env python3
# ruff: noqa: D103,E401,E701
"""Build the native suite with upstream object ordering in isolated directories."""
import argparse, hashlib, os, pathlib, subprocess, json, platform, shlex, shutil
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=['Debug','Release'],default='Debug')
p.add_argument('--only',nargs='+',choices=['HINT','MKVAC','MKFLX','MKLIM','HMAG','GPRTS','MAGVAL'])
p.add_argument('--hint-root',type=pathlib.Path,default=pathlib.Path.cwd())
p.add_argument('--build-root',type=pathlib.Path,help='isolated output directory; defaults below --hint-root')
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
programs={'HINT':'hint.exe','MKVAC':'mkvac.exe','MKFLX':'mkflx.exe','MKLIM':'mklim.exe','HMAG':'hmag.exe','GPRTS':'gprts.exe','MAGVAL':'magval.exe'}
b=(a.build_root.resolve() if a.build_root else r/('build-'+a.mode.lower()))
if b==r or b==r/'src' or b.is_relative_to(r/'src'):
 p.error('--build-root must not be the source root, src directory, or an src descendant')
b.mkdir(parents=True,exist_ok=True)
config_path=b/'build_config.json'
config_path.unlink(missing_ok=True)  # A failed rebuild must not leave a valid-looking old manifest.
helper=pathlib.Path(__file__).resolve();driver=helper.with_name('sample-points.f90')

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):
   h.update(chunk)
 return h.hexdigest()

def sources(name,src):
 rows=[(str(path.relative_to(r)),path) for path in sorted(src.glob('*.f90'))]
 rows.append((str((src/makefiles[name]).relative_to(r)),src/makefiles[name]))
 if name=='HINT':rows.append(('src/HINT/stepa_modules/fline_method_mod.f90',src/'stepa_modules/fline_method_mod.f90'))
 if name=='MAGVAL':rows.append(('handoff/hint-qa/sample-points.f90',driver))
 return rows

def snapshot(rows):
 return {name:{'bytes':path.stat().st_size,'sha256':digest(path)} for name,path in rows}

def set_digest(rows):
 h=hashlib.sha256()
 for name,record in sorted(rows.items()):h.update(name.encode()+b'\0'+bytes.fromhex(record['sha256']))
 return h.hexdigest()

def git_value(*args):
 return subprocess.check_output(['git','-C',str(r),*args],text=True).strip()

def source_commit():
 return git_value('rev-parse','HEAD') if (r/'.git').exists() else ((r/'SOURCE_COMMIT').read_text().strip() if (r/'SOURCE_COMMIT').exists() else None)

fc_argv=shlex.split(fc)
compiler=shutil.which(fc_argv[0])
if not compiler:p.error(f'compiler is not executable: {fc_argv[0]}')
compiler=pathlib.Path(compiler).resolve()
compiler_version=subprocess.check_output(fc_argv+['--version'])
compiler_record={'kind':'compiler_driver_or_wrapper','basename':pathlib.Path(fc_argv[0]).name,'resolved_executable_sha256':digest(compiler),'version_first_line':compiler_version.decode(errors='replace').splitlines()[0],'version_sha256':hashlib.sha256(compiler_version).hexdigest(),'limitation':'This identifies the invoked compiler driver or wrapper, not every underlying compiler, linker, or library.'}
helper_hash=digest(helper);commit=source_commit();tree=git_value('rev-parse','HEAD^{tree}') if (r/'.git').exists() else None
selected=list(a.only or makefiles)
source_rows={name:sources(name,r/'src'/name) for name in selected}
source_baselines={name:snapshot(rows) for name,rows in source_rows.items()}
built={}
for name in selected:
 src=r/'src'/name;dest=b/name;dest.mkdir(exist_ok=True)
 for f in src.glob('*.f90'):
  link=dest/f.name
  if link.exists() or link.is_symlink():link.unlink()
  link.symlink_to(f)
 if name=='HINT':
  link=dest/'stepa_mod.f90'
  if link.exists() or link.is_symlink():link.unlink()
  link.symlink_to(src/'stepa_modules/fline_method_mod.f90')
 (dest/'Makefile').write_bytes((src/makefiles[name]).read_bytes())
 cmd=['make','-B','-j1',f'FC={fc}',f'LINK={fc}',f'FFLAGS={flags}',f'MOD={inc}',f'LIBS={libs}']
 print('Building',name,flush=True)
 with (dest/'build.log').open('w') as log:
  result=subprocess.run(cmd,cwd=dest,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:
  print((dest/'build.log').read_text()[-7000:]);raise SystemExit(result.returncode)
 if name=='MAGVAL':
  objects=[str(dest/name) for name in ('module.o','spline_mod.o','free_mem.o','magout.o','magset.o','make_mem.o','mgcpu.o','mgval1.o','mgval2.o','mgval3.o','polint.o','read_eq_field.o','vsetup.o')]
  cmd=fc_argv+shlex.split(flags)+shlex.split(inc)+[f'-I{dest}',str(driver)]+objects+shlex.split(libs)+['-o',str(dest/'sample-points.exe')]
  with (dest/'build.log').open('a') as log:
   result=subprocess.run(cmd,cwd=dest,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:
   print((dest/'build.log').read_text()[-7000:]);raise SystemExit(result.returncode)
 outputs=[programs[name]]+(['sample-points.exe'] if name=='MAGVAL' else [])
 for output in outputs:
  if not (dest/output).is_file():raise RuntimeError(f'{name} did not produce {output}')
 before=source_baselines[name]
 built[name]={'source_files':before,'source_set_sha256':set_digest(before),'build_log_sha256':digest(dest/'build.log'),'binaries':{output:{'bytes':(dest/output).stat().st_size,'sha256':digest(dest/output)} for output in outputs}}
if digest(helper)!=helper_hash:raise RuntimeError('build helper changed during build')
if digest(compiler)!=compiler_record['resolved_executable_sha256']:raise RuntimeError('compiler driver changed during build')
if source_commit()!=commit:raise RuntimeError('source revision changed during build')
for name,rows in source_rows.items():
 if snapshot(rows)!=source_baselines[name]:raise RuntimeError(f'{name} source changed during build')
config=dict(schema=2,mode=a.mode,compiler=fc,flags=flags,includes=inc,libraries=libs,platform=platform.platform(),commit=commit,tree=tree,checkout_clean=not bool(git_value('status','--porcelain=v1')) if (r/'.git').exists() else None,selected_tools=selected,forced_rebuild=True,build_helper={'path':'handoff/hint-qa/build.py','sha256':helper_hash},compiler_identity=compiler_record,tools=built)
config_path.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
