"""Pack full historical V3 records; excludes source/build caches but retains frozen runtime identities."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    args = p.parse_args()
    root = args.workspace.resolve()
    first = 'integrated/20261112-dual-line-r1'
    second = 'integrated/20261113-dual-line-stat-r1'
    proto = json.loads((root/second/'protocol.json').read_text(encoding='utf-8'))
    paths = [f'{first}/{part}' for part in ['README.md','configs','provenance','reports','results','scripts']]
    paths += [second, 'experiments/readonly_f_compare/process_probe.py']
    identities = []
    for row in proto['dependencies']:
        original = Path(row['path'])
        rel = original.relative_to(root).as_posix()
        with (root/rel).open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest() == row['sha256'], rel
        assert (root/rel).stat().st_size == row['bytes'],rel
        identities.append({'path':rel,'bytes':row['bytes'],'sha256':row['sha256']})
        if rel.lower().endswith(('.exe','.dll')):
            paths.append(rel)
    subprocess.run([sys.executable,str(Path(__file__).with_name('evidence_bundle.py')),
                    'pack','--source-root',str(root),'--bundle',str(args.bundle),
                    '--name','V3-joint-correctness-and-142-process-statistics','--shard-mib','32',*paths],check=True)
    scope = {'status':'complete','lossless_records':True,'historical_model_processes':158,
             'correctness_processes':16,'statistical_processes':142,
             'included':paths,'frozen_dependencies_verified':identities,
             'omitted_top_level_categories':[
                 {'path':first+'/source','reason':'Exact 4276-file source snapshot is separately delivered in joint/source-bundle, not duplicated in experimental data.'},
                 {'path':first+'/build','reason':'Objects, static libraries, compiler caches and other generated builds are reproducible. Only the protocol runtime exe/DLL identity dependencies listed above are retained, exclusively for offline SHA-256 audit.'}],
             'cache_policy':'Python bytecode excluded individually in the bundle manifest; no successful, failed or interrupted experimental record is omitted.'}
    (args.bundle/'SCOPE.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
