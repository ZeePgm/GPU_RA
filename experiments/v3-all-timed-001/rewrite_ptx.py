"""Use an address-sorted directory for the original load probes at model scale."""

import importlib.util
from pathlib import Path


_probe_path = Path(__file__).resolve().parents[1] / "v3-five-timed-001" / "rewrite_ptx.py"
_spec = importlib.util.spec_from_file_location("five_weight_timing_probe", _probe_path)
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)


def find_range(label: str, width: int) -> str:
    # Upper bound on sorted bases. Check the immediate successor for a load
    # starting outside the ranges, and recover the logical ordinal from the
    # matched record. Page/bitmap offsets and native load readiness stay intact.
    return f"""    mov.u64 %f_offset, 0;
    mov.u64 %f_end, %f_count;
{label}_lookup:
    setp.ge.u64 %f_test, %f_offset, %f_end;
    @%f_test bra {label}_bound;
    add.u64 %f_index, %f_offset, %f_end;
    shr.u64 %f_index, %f_index, 1;
    shl.b64 %f_pos, %f_index, 6;
    add.u64 %f_entry, %f_ranges, %f_pos;
    ld.global.u64 %f_base, [%f_entry];
    setp.lt.u64 %f_test, %f_actual, %f_base;
    @%f_test mov.u64 %f_end, %f_index;
    @!%f_test add.u64 %f_offset, %f_index, 1;
    bra {label}_lookup;
{label}_bound:
    setp.eq.u64 %f_test, %f_offset, %f_count;
    @%f_test bra {label}_candidate;
    shl.b64 %f_pos, %f_offset, 6;
    add.u64 %f_entry, %f_ranges, %f_pos;
    ld.global.u64 %f_base, [%f_entry];
    add.u64 %f_work, %f_actual, {width};
    setp.gt.u64 %f_test, %f_work, %f_base;
    @%f_test bra {label}_fault;
{label}_candidate:
    setp.eq.u64 %f_test, %f_offset, 0;
    @%f_test bra {label}_load;
    sub.u64 %f_index, %f_offset, 1;
    shl.b64 %f_pos, %f_index, 6;
    add.u64 %f_entry, %f_ranges, %f_pos;
    ld.global.u64 %f_base, [%f_entry];
    ld.global.u64 %f_size, [%f_entry+8];
    sub.u64 %f_offset, %f_actual, %f_base;
    setp.ge.u64 %f_test, %f_offset, %f_size;
    @%f_test bra {label}_load;
    ld.global.u64 %f_index, [%f_entry+32];
"""


def rewrite(source: str, unit: str) -> tuple[str, list[dict]]:
    intermediate, sites = _probe.rewrite(source, unit)
    pieces = []
    cursor = 0
    for site in sites:
        label = f"HBFLAB_FIVE_{unit}_{site['site']}"
        start = intermediate.index(f"    mov.u64 %f_index, 0;\n{label}_lookup:", cursor)
        end = intermediate.index(f"{label}_match:", start)
        if not intermediate[start:end].endswith(f"    bra {label}_lookup;\n"):
            raise ValueError(f"Unexpected lookup probe layout at {unit}/{site['site']}")
        pieces.append(intermediate[cursor:start])
        pieces.append(find_range(label, site["width"]))
        cursor = end
    pieces.append(intermediate[cursor:])
    return "".join(pieces), sites
