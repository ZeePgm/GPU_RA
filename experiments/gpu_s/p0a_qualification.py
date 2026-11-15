"""Fail-closed P0-A statistics, independent of simulator wait identities.

Freeze the contract before collecting independent calibration samples. Arrays
contain paired host endpoint differences, never differences across clocks.
"""
import hashlib
import json
import math
from statistics import mean, stdev
from scipy.stats import t


def freeze(contract):
    required = ('target_ns', 'relative_margin', 'alpha', 'endpoints',
                'warmup_runs', 'pairs', 'order_seed', 'protocol', 'binary_sha256')
    if any(k not in contract for k in required):
        raise ValueError('Incomplete frozen contract')
    if contract['relative_margin'] != 0.1:
        raise ValueError('Original 10% criterion must not be relaxed')
    if not math.isfinite(contract['target_ns']) or contract['target_ns'] <= 0:
        raise ValueError('Target must be positive and finite')
    if not 0 < contract['alpha'] < 1:
        raise ValueError('Invalid alpha')
    for k in ('endpoints', 'pairs', 'warmup_runs'):
        if type(contract[k]) is not int or contract[k] < (2 if k == 'pairs' else 1):
            raise ValueError('Invalid count: ' + k)
    if contract['protocol'] not in ('fresh_process', 'reuse_context'):
        raise ValueError('Unknown protocol')
    return hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()


def interval(samples, contract):
    freeze(contract)
    if len(samples) < 2 or not all(math.isfinite(x) for x in samples):
        raise ValueError('Need at least two finite independent samples')
    half = float(t.ppf(1-contract['alpha']/(2*contract['endpoints']), len(samples)-1)) * stdev(samples)/math.sqrt(len(samples))
    return mean(samples)-half, mean(samples)+half


def qualify(contract, noise_ns, overshoot_ns, evidence):
    """Separate zero/zero noise and injector overshoot calibration cohorts.

    Conservatively require both intervals within the fixed uncertainty budget.
    A caller must preserve source records and collect formal samples separately.
    """
    digest = freeze(contract)
    checks = ['frozen_before_calibration', 'independent_calibration',
              'math_equal', 'access_identity_equal', 'frequency_recorded',
              'warmup_fixed', 'crossover_order_fixed']
    if contract['protocol'] == 'reuse_context':
        checks += ['kv_reset_verified', 'cache_reset_verified', 'backend_reset_verified']
    missing = [k for k in checks if evidence.get(k) is not True]
    noise = interval(noise_ns, contract)
    overshoot = interval(overshoot_ns, contract)
    uncertainty = max(map(abs, noise)) + max(map(abs, overshoot))
    return {'contract_sha256': digest, 'status': 'qualified' if not missing and uncertainty <= contract['target_ns']*.1 else 'inconclusive',
            'missing_evidence': missing, 'noise_ci_ns': noise,
            'overshoot_ci_ns': overshoot, 'uncertainty_ns': uncertainty}


def adjudicate(contract, frozen_sha256, errors_ns):
    if freeze(contract) != frozen_sha256:
        raise ValueError('Frozen contract changed')
    if len(errors_ns) != contract['pairs']:
        raise ValueError('Fixed sample budget not met')
    lo, hi = interval(errors_ns, contract)
    margin = contract['target_ns']*.1
    status = 'pass' if -margin <= lo <= hi <= margin else 'fail' if lo > margin or hi < -margin else 'inconclusive'
    return {'status': status, 'ci_ns': [lo, hi], 'margin_ns': margin}
