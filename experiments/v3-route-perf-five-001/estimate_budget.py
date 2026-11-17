"""Recompute this experiment's prospective budget from sealed scalar records.

No GPU launch, no new performance samples, no statistical approval or threshold
changes. The 25% factor is a planning allowance, not a confidence interval.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

ID='v3-route-perf-five-001'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ROUTES=('symbolic','ptx','cuda')


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dt(s):return datetime.fromisoformat(s.replace('Z','+00:00'))
def identity(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def estimate(runs):
    inputs=[]
    def acceptance(run):
        p=runs/run/'data/acceptance.json';a=read(p)
        assert a['status']=='pass' and a['formal_sample'] is False
        report=read(ROOT/f'reports/{ID}/{run}.json');assert report['verdict']=='passed'
        inputs.append({'run_id':run,**identity(p)})
        return a
    matrix=read(HERE/'matrix.json')
    pilots={}
    for case in ('w1-p','w4-p','w5-d'):
        for route in ROUTES:pilots[case,route]=acceptance(f'pilot-{case}-{route}-20261117-001')
    ptx_model=acceptance('pilot-model-ptx-20261117-001')
    cells=[]
    for c in matrix['weight_cells']:
        case,route=c['scenario_id'],c['route'];k=c['repeated_calls']
        suffix='002' if (case,route)==('w1-p','symbolic') else '001'
        q=acceptance(f'qual-{case}-{route}-20261117-{suffix}')
        if (case,route) in pilots:
            p=pilots[case,route]
            predicted=p['process_wall_seconds']+(k-1)*p['repeated_mean_ms']/1000
            basis='observed first+one-repeat whole process + (K-1)*observed repeated latency'
        elif case=='w5-p':
            p=pilots['w5-d',route]
            repeated=max(q['first_call_ms'],p['first_call_ms'],p['repeated_mean_ms'])/1000
            predicted=q['process_wall_seconds']+k*repeated
            basis='qualified P whole process + K*max(P first, D pilot first/repeated); same output-projection shape'
        else:
            predicted=q['process_wall_seconds']+k*q['first_call_ms']/1000
            basis='qualified whole first-call process + K*qualified first latency; conservative cold-call proxy where no repeated pilot exists'
        cells.append({'cell_id':c['cell_id'],'process_repeats':3,'repeated_calls_per_process':k,
            'predicted_cpp_seconds_per_process':predicted,'predicted_cpp_seconds_total':3*predicted,'basis':basis})
    models=[]
    for route in ROUTES:
        smoke=acceptance(f'smoke-{route}-20261117-001')
        value=max(smoke['process_wall_seconds'],ptx_model['process_wall_seconds']) if route=='ptx' else smoke['process_wall_seconds']
        models.append({'route':route,'process_repeats':3,'predicted_cpp_seconds_per_process':value,'predicted_cpp_seconds_total':3*value,
            'basis':'max(qualified model smoke, fixed pilot)' if route=='ptx' else 'qualified same-condition model smoke; no additional route pilot authorized'})
    overhead=[]
    pilot_ids=[f'pilot-{case}-{route}-20261117-001' for case in ('w1-p','w4-p','w5-d') for route in ROUTES]+['pilot-model-ptx-20261117-001']
    for run in pilot_ids:
        r=read(runs/run/'run.json');a=read(runs/run/'data/acceptance.json')
        overhead.append((dt(r['finished_utc'])-dt(r['registered_utc'])).total_seconds()-a['process_wall_seconds'])
    # finish + committed-report verify each took about 1.3-1.5s in this session;
    # use 4s combined in addition to observed registration/launch/identity overhead.
    overhead_per_process=math.ceil(statistics.median(overhead)+4)
    weight_seconds=sum(c['predicted_cpp_seconds_total'] for c in cells)
    model_seconds=sum(c['predicted_cpp_seconds_total'] for c in models)
    total_cpp=weight_seconds+model_seconds;overhead_seconds=99*overhead_per_process
    nominal=total_cpp+overhead_seconds
    allowance=1.25*total_cpp+overhead_seconds
    recommended=math.ceil(allowance/900)*900
    first=min(dt(read(runs/r/'run.json')['registered_utc']) for r in pilot_ids)
    last=max(dt(read(runs/r/'run.json')['finished_utc']) for r in pilot_ids)
    return {'schema':1,'experiment_id':ID,'scope':'Prospective estimate only. Formal matrix never started.',
        'formal_matrix_processes':99,'weight_processes':90,'model_processes':9,'weight_cells':cells,'model_routes':models,
        'projected_weight_cpp_seconds':weight_seconds,'projected_model_cpp_seconds':model_seconds,
        'observed_pilot_registration_launch_identity_overhead_seconds':overhead,
        'observed_overhead_median_seconds':statistics.median(overhead),'observed_overhead_max_seconds':max(overhead),
        'finish_and_verify_allowance_per_run_seconds':4,'total_workflow_overhead_per_run_seconds':overhead_per_process,
        'projected_workflow_overhead_seconds':overhead_seconds,'projected_nominal_formal_seconds':nominal,
        'cpp_planning_factor':1.25,'projected_allowance_seconds':allowance,'recommended_formal_cap_seconds':recommended,
        'existing_approved_formal_cap_seconds':14400,'existing_total_experiment_cap_seconds':27000,
        'budget_change_is_proposal_only':True,'pilot_stage_observed_seconds':(last-first).total_seconds(),
        'uncertainty':'One pilot process per selected case/route, and one model smoke per other route; no steady-state or confidence claim. No clock locking. 25% is a planning scenario. All original numerical/resource/statistical thresholds remain unchanged.',
        'inputs':inputs}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=estimate(args.runs_root)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('projected_weight_cpp_seconds','projected_model_cpp_seconds','total_workflow_overhead_per_run_seconds','projected_nominal_formal_seconds','projected_allowance_seconds','recommended_formal_cap_seconds','pilot_stage_observed_seconds')},indent=2))
