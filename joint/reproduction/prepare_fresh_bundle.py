"""Archive the complete V3 migration validation, including failed resource-gated attempt."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'reproduction'))
from data_paths import require_external_directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--bundle',type=Path,required=True,help='Fresh external bundle output directory')
    args=parser.parse_args();root=args.repository.resolve()
    dest=args.bundle.resolve()
    require_external_directory(dest)
    paths=['joint/output/correctness-new-001','joint/output/correctness-new-002',
           'joint/output/statistics-prepare-001','joint/reproduction-validation',
           'joint/build-reproduction-host/build-acceptance.json',
           'joint/source-bundle/manifest.json','joint/configs',
           'joint/scripts/run_correctness.py','joint/statistical-batch/protocol.json',
           'joint/statistical-batch/scripts','joint/reproduction/evidence_validators']
    paths += [p.relative_to(root).as_posix()for p in sorted(root.glob('build-v3*.log'))]
    paths += ['joint/reproduction/'+name for name in ['run_correctness.py','run_statistics.py',
              'runtime_common.py','process_probe.py','verify_source.py','check_build.py',
              'requirements-validated.txt','python-environment.json']]
    bins=sorted((root/'joint/build-reproduction-host/model/bin').glob('*'))
    bins=[p for p in bins if p.suffix.lower()in('.exe','.dll')]
    assert len(bins)==8 and sum(p.suffix.lower()=='.exe'for p in bins)==3
    paths += [p.relative_to(root).as_posix()for p in bins]
    status=json.loads((root/'joint/output/correctness-new-002/status.json').read_text())
    assert status['status']=='pass' and status['completed']==16,status
    # Validate dependencies before freezing this archive, without changing any input.
    dependencies=[]
    for name in ['correctness-new-001','correctness-new-002']:
        for row in json.loads((root/'joint/output'/name/'dependencies.json').read_text()):
            p=Path(row['path'])
            with p.open('rb')as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256'],str(p)
            assert p.stat().st_size==row['bytes']
            dependencies.append({'case':name,**row})
            relative=p.relative_to(root).as_posix()
            if relative not in paths:paths.append(relative)
    statistics_dependencies=[]
    statistics=json.loads((root/'joint/output/statistics-prepare-001/protocol.json').read_text())
    for row in statistics['dependencies']:
        p=Path(row['path']);relative=p.relative_to(root).as_posix()
        with p.open('rb')as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256'],str(p)
        assert p.stat().st_size==row['bytes']
        statistics_dependencies.append(row)
        if relative not in paths:paths.append(relative)
    subprocess.run([sys.executable,str(Path(__file__).with_name('evidence_bundle.py')),
                    'pack','--source-root',str(root),'--bundle',str(dest),
                    '--name','V3-fresh-build-correctness-and-statistics-prepare-20261115',
                    '--shard-mib','32',*paths],check=True)
    scope={'status':'complete','fresh_correctness_processes_accepted':16,
           'failed_attempt':'correctness-new-001: two completed processes and the interrupted third preserved in full',
           'new_statistics_model_processes':0,'statistics_prepare_reference_groups':14,
           'includes':paths,'frozen_correctness_dependencies_verified':dependencies,
           'frozen_statistics_prepare_dependencies_verified':statistics_dependencies,
           'runtime_binary_policy':'Eight new source-built exe/DLL files are retained exclusively for offline identity hashing. Restore does not execute them or link them as build inputs.',
           'excluded':'GGUF model weights, object/static-library files and CMake/compiler caches. Python bytecode omissions are listed in manifest.json.'}
    (dest/'SCOPE.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
