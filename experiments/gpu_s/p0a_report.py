"""Order-effect diagnostics, separate from frozen primary decisions."""
import argparse
from statistics import mean
from pathlib import Path
from common import read,write


def summarize(study):
    results=read(study/'result.json'); report={}
    for protocol,result in results.items():
        if result['formal_status']!='completed':
            report[protocol]={'status':'inconclusive','reason':'dose qualification did not pass; no formal samples'}
            continue
        orders=read(study/f'{protocol}-orders.json')
        errors=result['errors_ns']
        per_order={}
        for first in ['zero','inject']:
            subset=[row for row,order in zip(errors,orders) if order[0]==first]
            per_order[first+'_first']={'pairs':len(subset),'mean_error_ns':[mean(r[k] for r in subset) for k in range(8)]}
        half=len(errors)//2
        report[protocol]=dict(status=[x['status'] for x in result['results']],order=per_order,
            first_half_mean_ns=[mean(r[k] for r in errors[:half]) for k in range(8)],
            second_half_mean_ns=[mean(r[k] for r in errors[half:]) for k in range(8)],
            uncertainty_model='Bonferroni Student-t paired means; independent approximately stable pair errors assumed. Order/drift diagnostics do not override frozen primary intervals.',
            scope='100 ms first-gate injected delay; not an HBF hardware speedup estimate, not qualification of shorter doses')
    write(study/'order-diagnostics.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);a=p.parse_args();summarize(a.study)
