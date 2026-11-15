"""Domain-separated interval unions and overlap-preserving cost timelines."""
import argparse
from collections import defaultdict
from common import read, write, rows
from pathlib import Path


def union_ns(intervals):
    end = None
    total = 0
    for a,b in sorted(intervals):
        if b < a: raise ValueError('Reversed interval')
        total += max(0,b-max(a,end if end is not None else a))
        end=max(b,end if end is not None else b)
    return total


def report(folder):
    grouped=defaultdict(list)
    spans=read(folder/'p0a-costs.json')['spans']
    for row in spans:
        if row['clock']!='steady_clock': raise ValueError('Unexpected host clock')
        grouped[('steady_clock',row['category'])].append((row['start_ns'],row['end_ns']))
    for filename, category in [('kernels.jsonl','gpu_kernel'),('transfers.jsonl','gpu_transfer')]:
        for row in rows(folder/filename):
            grouped[('CUPTI',category)].append((row['start_ns'],row['end_ns']))
    clocks=read(folder/'a1-clocks.json')
    host_start=clocks['process_entry_ns']
    host_end=max(r['end_ns'] for r in spans)
    coverage=union_ns([(r['start_ns'],r['end_ns']) for r in spans])
    return dict(host_coverage=dict(start_ns=host_start,end_ns=host_end,covered_union_ns=coverage,unattributed_ns=host_end-host_start-coverage,excluded='Final cost-file serialization and process exit are outside this envelope'),categories=[dict(clock=d,category=c,union_ns=union_ns(v),sum_ns=sum(b-a for a,b in v),spans=len(v)) for (d,c),v in grouped.items()],
                host_timeline=spans,
                interpretation='Category unions overlap. No cross-clock subtraction or sum. GPU and host tracks have independent origins; no claimed cross-domain critical path.',
                critical_path_scope='Host request boundary chain in a1-clocks.json; synchronized device work is nested inside host inference envelopes. GPU correlation IDs retained in original files.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args()
    write(args.folder/'p0a-timeline.json',report(args.folder))
