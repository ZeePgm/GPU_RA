from pathlib import Path
import shutil
from common import read,write,sha
from study import ROOT
out=ROOT/'output/gpu-a1/20261102-formal-001';a=read(out/'analysis.json')
a['limitations']=['Fixed-budget wall equivalence inconclusive in all 48 endpoints','No standalone high-resolution injector accuracy calibration; measured overshoot and steady_clock resolution are not hardware timing accuracy','Paired Student-t intervals use independence/distribution approximations; drift/heavy tails remain possible','Late-gate pilot not powered to demonstrate zero effect before gate']
a['full_trace_audit']=read(ROOT/'output/gpu-a1/20261102-full-audit-001/result.json')
a['negative_controls']=read(ROOT/'output/gpu-a1/20261102-negative-001/result.json')
a['overall_status']='inconclusive';write(out/'analysis.json',a);write(ROOT/'docs/validation/gpu-a1.json',a)
snap=out/'sources';snap.mkdir(exist_ok=True)
for p in (ROOT/'experiments/gpu_s').glob('a1*'):shutil.copyfile(p,snap/p.name)
for p in [ROOT/'experiments/gpu_s/native_a1.cpp',ROOT/'experiments/gpu_s/CMakeLists.txt',ROOT/'build-s3/bin/gpu_a1.exe']:shutil.copyfile(p,snap/p.name)
write(out/'artifact-index.json',{'files':[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.rglob('*') if p.is_file() and p.name!='artifact-index.json']})
