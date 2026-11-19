"""Add a demand-page wait before the original native loads matched by the Q probe."""

import importlib.util
from pathlib import Path


_trace_path = Path(__file__).resolve().parents[1] / "v3-q-auto-001" / "rewrite_ptx.py"
_spec = importlib.util.spec_from_file_location("q_read_trace_rewrite", _trace_path)
_trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trace)

STATE_REGISTERS = ".reg .b64 %q_state, "
COUNT_LOAD = "ld.global.u64 %q_ptr, [%q_state+16];"


def wait_for_page(unit: str, site: dict) -> str:
    label = f'HBFLAB_Q_TIMED_{unit}_{site["site"]}'
    width = site["width"]
    # ProbeState and QMediaPage wire offsets are asserted by q_auto_device.cu.
    # Both completion flags are acquire/release. The device-only flag is published
    # after the first waiter has acquired the corresponding host completion.
    return f"""
{{
    .reg .b64 %qt_claim, %qt_index, %qt_disp, %qt_page, %qt_started;
    .reg .b64 %qt_now, %qt_first, %qt_delay, %qt_elapsed, %qt_observed, %qt_store;
    .reg .b64 %qt_cache, %qt_base;
    .reg .b32 %qt_status, %qt_old;
    .reg .pred %qt_p, %qt_won;
    ld.global.u64 %qt_claim, [%q_state+40];
    setp.eq.u64 %qt_p, %qt_claim, 0;
    @%qt_p bra {label}_done;
    and.b64 %qt_disp, %q_offset, 4095;
    setp.gt.u64 %qt_p, %qt_disp, {4096-width};
    @%qt_p bra {label}_fault;
    shr.u64 %qt_index, %q_offset, 12;
    ld.global.u64 %qt_cache, [%q_state+72];
    shl.b64 %qt_disp, %qt_index, 3;
    add.u64 %qt_cache, %qt_cache, %qt_disp;
    ld.acquire.gpu.global.u64 %qt_base, [%qt_cache];
    setp.ne.u64 %qt_p, %qt_base, 0;
    @%qt_p bra {label}_observed;

    shl.b64 %qt_disp, %qt_index, 2;
    add.u64 %qt_claim, %qt_claim, %qt_disp;
    ld.acquire.gpu.global.u32 %qt_old, [%qt_claim];

    ld.global.u64 %qt_page, [%q_state+48];
    shl.b64 %qt_disp, %qt_index, 6;
    add.u64 %qt_page, %qt_page, %qt_disp;
    mov.u64 %qt_started, %globaltimer;
    setp.eq.u32 %qt_p, %qt_old, 0;
    @%qt_p atom.global.cas.b32 %qt_old, [%qt_claim], 0, 1;
    setp.eq.u32 %qt_won, %qt_old, 0;
    @%qt_won st.global.u64 [%qt_page], %qt_started;
    @%qt_won st.release.sys.global.u32 [%qt_page+16], 1;

{label}_poll:
    ld.acquire.sys.global.u32 %qt_status, [%qt_page+16];
    setp.eq.u32 %qt_p, %qt_status, 2;
    @%qt_p bra {label}_host_ready;
    setp.eq.u32 %qt_p, %qt_status, 3;
    @%qt_p bra {label}_fault;
    mov.u64 %qt_now, %globaltimer;
    sub.u64 %qt_elapsed, %qt_now, %qt_started;
    setp.ge.u64 %qt_p, %qt_elapsed, 1000000000;
    @%qt_p bra {label}_fault;
    nanosleep.u32 256;
    bra {label}_poll;

{label}_host_ready:
    ld.global.u64 %qt_first, [%qt_page];
    ld.global.u64 %qt_delay, [%qt_page+8];
    setp.gt.u64 %qt_p, %qt_delay, 1000000000;
    @%qt_p bra {label}_fault;
{label}_deadline:
    mov.u64 %qt_now, %globaltimer;
    sub.u64 %qt_elapsed, %qt_now, %qt_first;
    setp.ge.u64 %qt_p, %qt_elapsed, %qt_delay;
    @!%qt_p bra {label}_deadline;
    ld.global.u64 %qt_base, [%qt_page+24];
    setp.eq.u64 %qt_p, %qt_base, 0;
    @%qt_p bra {label}_fault;
    st.release.gpu.global.u64 [%qt_cache], %qt_base;

{label}_observed:
    and.b64 %qt_disp, %q_offset, 4095;
    add.u64 %q_actual, %qt_base, %qt_disp;
    mov.u64 %qt_observed, %globaltimer;
    shl.b64 %qt_disp, %q_offset, 2;
    ld.global.u64 %qt_store, [%q_state+56];
    add.u64 %qt_store, %qt_store, %qt_disp;
    red.global.max.u64 [%qt_store], %qt_observed;
    mov.u64 %qt_now, %globaltimer;
    ld.global.u64 %qt_store, [%q_state+64];
    add.u64 %qt_store, %qt_store, %qt_disp;
    red.global.max.u64 [%qt_store], %qt_now;
    bra {label}_done;

{label}_fault:
    trap;
{label}_done:
}}
"""


def rewrite(source: str, unit: str) -> tuple[str, list[dict]]:
    intermediate, sites = _trace.rewrite(source, unit)
    output = []
    index = 0
    site = None
    inserted = False
    for line in intermediate.splitlines(keepends=True):
        if line.startswith(STATE_REGISTERS):
            if site is not None or index >= len(sites):
                raise ValueError("Q timing probe is missing a native load")
            site = sites[index]
            inserted = False
            load = _trace.LOAD.match(site["instruction"])
            base = load[3]
            offset = load[5] or "0"
            delta = int(offset, 16 if offset.lstrip("-").lower().startswith("0x") else 10)
            if load[4] == "-":
                delta = -delta
            guard = (load[1] + " ") if load[1] else ""
            output.extend([line, ".reg .b64 %q_actual;\n", f"{guard}add.u64 %q_actual, {base}, {delta};\n"])
            continue
        if site and line.strip() == COUNT_LOAD:
            if inserted:
                raise ValueError("Multiple timing insertion points for one native load")
            output.append(wait_for_page(unit, site))
            inserted = True
        if site and line.strip() == site["instruction"]:
            if not inserted or output.pop().strip() != "}":
                raise ValueError("Q probe cannot carry the original load to its released address")
            load = _trace.LOAD.match(line)
            start = line.rfind("[", 0, load.start(3))
            end = line.index("]", load.end(3))
            # The native load consumes the page address published by the host
            # only after completion. This keeps a data dependency on the wait
            # through assembly, even for a noncoherent native opcode.
            output.extend([line[:start] + "[%q_actual]" + line[end + 1:], "}\n"])
            index += 1
            site = None
            continue
        output.append(line)
    if site is not None or index != len(sites):
        raise ValueError("Q timing gates and native load sites disagree")
    return "".join(output), sites
