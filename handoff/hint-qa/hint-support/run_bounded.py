"""Run one task-owned validation per compute resource, with timeout and provenance."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,time,socket
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--resource',choices=['cpu','gpu0','gpu1'],required=True);p.add_argument('--label',required=True);p.add_argument('--timeout',type=int,default=600);p.add_argument('--cwd',type=Path,default=Path.cwd());p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
if not a.command:p.error('Supply executable and arguments after --')
cmd=a.command[1:] if a.command[0]=='--' else a.command
if not cmd or a.timeout<=0:p.error('Need command and positive timeout')
root=Path(__file__).resolve().parents[1];locks=root/'runs/resource-locks';locks.mkdir(parents=True,exist_ok=True)
lock=(locks/(a.resource+'.lock')).open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit(f'Another task validation holds {a.resource}; wait for its status instead of launching a duplicate.')
if Path(a.label).name!=a.label:p.error('Label must be a single directory name')
d=root/'runs/job-records'/a.label;d.mkdir(parents=True,exist_ok=False)
env={**os.environ,'OMP_NUM_THREADS':'2','XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
if a.resource.startswith('gpu'):env.update(CUDA_VISIBLE_DEVICES=a.resource[-1],JAX_PLATFORMS='cuda,cpu')
else:env['JAX_PLATFORMS']='cpu'
start=time.time();status={'host':socket.gethostname(),'runner_pid':os.getpid(),'command':cmd,'cwd':str(a.cwd.resolve()),'resource':a.resource,'timeout_seconds':a.timeout,'start_unix':start,'state':'starting','pid':None,'execution_environment':{key:env.get(key) for key in ['OMP_NUM_THREADS','JAX_PLATFORMS','CUDA_VISIBLE_DEVICES','XLA_PYTHON_CLIENT_PREALLOCATE']}};sf=d/'status.json'
def save():sf.write_text(json.dumps(status,indent=2)+'\n')
save()
with (d/'output.log').open('w') as log:
 proc=subprocess.Popen(cmd,cwd=a.cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);status.update(state='running',pid=proc.pid);save()
 def interrupted(signum,frame):
  status['signal']=signum
  raise KeyboardInterrupt
 for sig in [signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,interrupted)
 try:code=proc.wait(timeout=a.timeout)
 except (subprocess.TimeoutExpired,KeyboardInterrupt) as err:
  status['interruption']=type(err).__name__;os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  code=124 if isinstance(err,subprocess.TimeoutExpired) else 130
status.update(state='finished',returncode=code,elapsed_seconds=time.time()-start);save();print(sf);raise SystemExit(code)
