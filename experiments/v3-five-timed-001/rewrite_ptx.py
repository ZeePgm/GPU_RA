"""Address-based PTX demand probes for registered read-only GPU weight ranges."""

import importlib.util
from pathlib import Path


_trace_path = Path(__file__).resolve().parents[1] / "v3-q-auto-001" / "rewrite_ptx.py"
_spec = importlib.util.spec_from_file_location("q_trace_syntax", _trace_path)
_syntax = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_syntax)
UNITS = ("convert", "mmvf", "binbcast", "getrows")
SITES_PER_UNIT = 32768


def instrument_load(line: str, unit: str, site: int, match, width: int) -> str:
    label = f"HBFLAB_FIVE_{unit}_{site}"
    offset = match[5] or "0"
    delta = int(offset, 16 if offset.lstrip("-").lower().startswith("0x") else 10)
    if match[4] == "-":
        delta = -delta
    native_start = line.rfind("[", 0, match.start(3))
    native_end = line.index("]", match.end(3))
    native = line[:native_start] + "[%f_actual]" + line[native_end + 1:].rstrip()
    predicate = match[1]
    inverse = ("@" + predicate[2:] if predicate.startswith("@!") else "@!" + predicate[1:]) if predicate else ""
    check_predicate = f"{inverse} bra {label}_load;" if inverse else ""
    site_offset = (UNITS.index(unit) * SITES_PER_UNIT + site) * 8
    count_words = width // 2
    word_updates = "\n".join(f"    red.global.add.u32 [%f_ptr+{i * 4}], 1;" for i in range(count_words))
    return f"""
{{
    .reg .b64 %f_root, %f_ranges, %f_count, %f_index, %f_entry, %f_base, %f_size, %f_end;
    .reg .b64 %f_actual, %f_offset, %f_work, %f_page_index, %f_in_page, %f_pos, %f_ptr;
    .reg .b64 %f_header, %f_timer, %f_started, %f_request, %f_delay, %f_elapsed, %f_released;
    .reg .b64 %f_mask, %f_audit, %f_words;
    .reg .b32 %f_old, %f_status, %f_ticket, %f_queue, %f_bit, %f_lo, %f_hi;
    .reg .pred %f_test, %f_owns;
    {check_predicate}
    add.u64 %f_actual, {match[3]}, {delta};
    ld.volatile.global.u64 %f_root, [hbflab_q_auto_state_{unit}];
    setp.eq.u64 %f_test, %f_root, 0;
    @%f_test bra {label}_load;
    ld.global.u64 %f_ranges, [%f_root];
    ld.global.u64 %f_count, [%f_root+8];
    mov.u64 %f_index, 0;
{label}_lookup:
    setp.ge.u64 %f_test, %f_index, %f_count;
    @%f_test bra {label}_load;
    shl.b64 %f_pos, %f_index, 5;
    add.u64 %f_entry, %f_ranges, %f_pos;
    ld.global.u64 %f_base, [%f_entry];
    ld.global.u64 %f_size, [%f_entry+8];
    setp.lt.u64 %f_test, %f_actual, %f_base;
    @%f_test bra {label}_check_below;
    sub.u64 %f_offset, %f_actual, %f_base;
    setp.lt.u64 %f_test, %f_offset, %f_size;
    @%f_test bra {label}_match;
    bra {label}_next;
{label}_check_below:
    add.u64 %f_end, %f_actual, {width};
    setp.gt.u64 %f_test, %f_end, %f_base;
    @%f_test bra {label}_fault;
{label}_next:
    add.u64 %f_index, %f_index, 1;
    bra {label}_lookup;
{label}_match:
    {f"bra {label}_fault;" if width % 2 else ""}
    sub.u64 %f_end, %f_size, {width};
    setp.gt.u64 %f_test, %f_offset, %f_end;
    @%f_test bra {label}_fault;
    and.b64 %f_in_page, %f_offset, 4095;
    setp.gt.u64 %f_test, %f_in_page, {4096-width};
    @%f_test bra {label}_fault;
    and.b64 %f_work, %f_offset, 1;
    setp.ne.u64 %f_test, %f_work, 0;
    @%f_test bra {label}_fault;
    ld.global.u64 %f_page_index, [%f_entry+16];
    shr.u64 %f_pos, %f_offset, 12;
    add.u64 %f_page_index, %f_page_index, %f_pos;
    shl.b64 %f_pos, %f_page_index, 6;
    ld.global.u64 %f_audit, [%f_root+24];
    add.u64 %f_audit, %f_audit, %f_pos;
    ld.acquire.gpu.global.u64 %f_released, [%f_audit];
    setp.ne.u64 %f_test, %f_released, 0;
    @%f_test bra {label}_consume;

    ld.global.u64 %f_header, [%f_root+16];
    add.u64 %f_header, %f_header, %f_pos;
    mov.u64 %f_started, %globaltimer;
    ld.acquire.gpu.global.u32 %f_old, [%f_audit+40];
    setp.eq.u32 %f_test, %f_old, 0;
    @%f_test atom.global.cas.b32 %f_old, [%f_audit+40], 0, 1;
    setp.eq.u32 %f_owns, %f_old, 0;
    @%f_owns st.global.u64 [%f_header], %f_started;
    @%f_owns st.release.sys.global.u32 [%f_header+16], 1;
    @%f_owns ld.global.u64 %f_ptr, [%f_root+56];
    @%f_owns atom.global.add.u32 %f_ticket, [%f_ptr], 1;
    @%f_owns mul.wide.u32 %f_work, %f_ticket, 4;
    @%f_owns ld.global.u64 %f_ptr, [%f_root+48];
    @%f_owns add.u64 %f_ptr, %f_ptr, %f_work;
    @%f_owns cvt.u32.u64 %f_queue, %f_page_index;
    @%f_owns add.u32 %f_queue, %f_queue, 1;
    @%f_owns st.release.sys.global.u32 [%f_ptr], %f_queue;
{label}_poll:
    ld.acquire.sys.global.u32 %f_status, [%f_header+16];
    setp.eq.u32 %f_test, %f_status, 2;
    @%f_test bra {label}_host_done;
    setp.eq.u32 %f_test, %f_status, 3;
    @%f_test bra {label}_fault;
    mov.u64 %f_timer, %globaltimer;
    sub.u64 %f_elapsed, %f_timer, %f_started;
    setp.ge.u64 %f_test, %f_elapsed, 1000000000;
    @%f_test bra {label}_fault;
    nanosleep.u32 256;
    bra {label}_poll;
{label}_host_done:
    ld.global.u64 %f_request, [%f_header];
    ld.global.u64 %f_delay, [%f_header+8];
    setp.gt.u64 %f_test, %f_delay, 1000000000;
    @%f_test bra {label}_fault;
{label}_deadline:
    mov.u64 %f_timer, %globaltimer;
    sub.u64 %f_elapsed, %f_timer, %f_request;
    setp.ge.u64 %f_test, %f_elapsed, %f_delay;
    @!%f_test bra {label}_deadline;
    ld.global.u64 %f_released, [%f_header+24];
    setp.eq.u64 %f_test, %f_released, 0;
    @%f_test bra {label}_fault;
    st.release.gpu.global.u64 [%f_audit], %f_released;
{label}_consume:
    add.u64 %f_actual, %f_released, %f_in_page;
    mov.u64 %f_timer, %globaltimer;
    red.global.min.u64 [%f_audit+8], %f_timer;
    red.global.max.u64 [%f_audit+16], %f_timer;
    red.global.add.u64 [%f_audit+24], 1;
    red.global.add.u64 [%f_audit+32], {width};
    ld.global.u64 %f_ptr, [%f_root+32];
    shl.b64 %f_pos, %f_page_index, 8;
    add.u64 %f_ptr, %f_ptr, %f_pos;
    shr.u64 %f_pos, %f_in_page, 6;
    shl.b64 %f_pos, %f_pos, 2;
    add.u64 %f_ptr, %f_ptr, %f_pos;
    cvt.u32.u64 %f_bit, %f_in_page;
    shr.u32 %f_bit, %f_bit, 1;
    and.b32 %f_bit, %f_bit, 31;
    shl.b64 %f_mask, {(1 << count_words)-1}, %f_bit;
    mov.b64 {{%f_lo, %f_hi}}, %f_mask;
    red.global.or.b32 [%f_ptr], %f_lo;
    setp.ne.u32 %f_test, %f_hi, 0;
    @%f_test red.global.or.b32 [%f_ptr+4], %f_hi;
    ld.global.u64 %f_words, [%f_entry+24];
    setp.eq.u64 %f_test, %f_words, 0;
    @%f_test bra {label}_site;
    shl.b64 %f_pos, %f_offset, 1;
    add.u64 %f_ptr, %f_words, %f_pos;
{word_updates}
{label}_site:
    ld.global.u64 %f_ptr, [%f_root+40];
    shl.b64 %f_pos, %f_index, 20;
    add.u64 %f_ptr, %f_ptr, %f_pos;
    red.global.add.u64 [%f_ptr+{site_offset}], 1;
    bra {label}_load;
{label}_fault:
    ld.global.u64 %f_ptr, [%f_root+64];
    red.global.add.u64 [%f_ptr], 1;
    trap;
{label}_load:
{native}
}}
"""


def rewrite(source: str, unit: str) -> tuple[str, list[dict]]:
    if unit not in UNITS or not f"hbflab_q_auto_state_{unit}" in source:
        raise ValueError(f"Missing device symbol in {unit}")
    output, sites = [], []
    entry = None
    for number, line in enumerate(source.splitlines(keepends=True), 1):
        function = _syntax.ENTRY.search(line) or _syntax.FUNCTION.search(line)
        if function:
            entry = function[1]
        if not _syntax.GLOBAL_LOAD.match(line):
            output.append(line)
            continue
        match = _syntax.LOAD.match(line)
        type_width = _syntax.TYPE.search(match[2]) if match else None
        if not type_width or not entry:
            raise ValueError(f"Cannot instrument {unit}:{number}: {line.strip()}")
        vector = _syntax.VECTOR.search(match[2])
        width = (int(vector[1]) if vector else 1) * (int(type_width[1]) // 8 if type_width[1] else 4)
        site = len(sites)
        if site >= SITES_PER_UNIT:
            raise ValueError(f"Too many global load sites in {unit}")
        sites.append(dict(site=site, entry=entry, line=number, opcode=match[2], width=width,
                          instruction=line.strip()))
        output.append(instrument_load(line, unit, site, match, width))
    if not sites:
        raise ValueError(f"No instrumentation sites in {unit}")
    return "".join(output), sites
