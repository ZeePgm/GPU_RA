"""Insert an address-range probe before compiler-emitted global load instructions."""

import re


SITE_COUNT = 16384
LOAD = re.compile(
    r"^\s*(?:(@!?%\w+)\s+)?(ld(?:\.volatile)?\.global(?:\.[\w:]+)+)\s+"
    r"[^;]+?,\s*\[\s*(%[\w$]+)(?:\s*([+-])\s*(-?(?:0[xX][\da-fA-F]+|\d+)))?\s*\]\s*;\s*(?://.*)?$"
)
GLOBAL_LOAD = re.compile(r"^\s*(?:@!?%\w+\s+)?ld(?:\.volatile)?\.global\b")
TYPE = re.compile(r"\.(?:[sufb](8|16|32|64|128)|f16x2)$")
VECTOR = re.compile(r"\.v(2|4|8)(?:\.|$)")
ENTRY = re.compile(r"(?<!\w)\.entry\s+([\w$]+)\s*\(")
FUNCTION = re.compile(r"(?<!\w)\.func(?:\s+\([^)]*\))?\s+([\w$]+)\s*\(")


def probe(unit, site, predicate, address, width):
    section = 0 if unit == "convert" else 1
    site_offset = (section * SITE_COUNT + site) * 8
    label = f"HBFLAB_Q_AUTO_{unit}_{site}"
    lines = [
        "{",
        ".reg .pred %q_skip;",
        ".reg .b64 %q_state, %q_addr, %q_base, %q_size, %q_offset, %q_limit, %q_ptr, %q_end;",
    ]
    if predicate:
        inverse = "@" + predicate[2:] if predicate.startswith("@!") else "@!" + predicate[1:]
        lines.append(f"{inverse} bra {label}_done;")
    lines.extend(line for line in [
        f"ld.volatile.global.u64 %q_state, [hbflab_q_auto_state_{unit}];",
        "setp.eq.u64 %q_skip, %q_state, 0;",
        f"@%q_skip bra {label}_done;",
        address,
        "ld.global.u64 %q_base, [%q_state];",
        "ld.global.u64 %q_size, [%q_state+8];",
        "setp.lt.u64 %q_skip, %q_addr, %q_base;",
        f"@%q_skip bra {label}_below;",
        "sub.u64 %q_offset, %q_addr, %q_base;",
        "setp.ge.u64 %q_skip, %q_offset, %q_size;",
        f"@%q_skip bra {label}_done;",
        f"bra {label}_bad;" if width == 1 else "",
        f"sub.u64 %q_limit, %q_size, {width};",
        "setp.gt.u64 %q_skip, %q_offset, %q_limit;",
        f"@%q_skip bra {label}_bad;",
        "and.b64 %q_limit, %q_offset, 1;",
        "setp.ne.u64 %q_skip, %q_limit, 0;",
        f"@%q_skip bra {label}_bad;",
        "ld.global.u64 %q_ptr, [%q_state+16];",
        "shl.b64 %q_offset, %q_offset, 1;",
        "add.u64 %q_ptr, %q_ptr, %q_offset;",
    ] if line)
    lines.extend(f"red.global.add.u32 [%q_ptr+{i*4}], 1;" for i in range(width // 2))
    lines.extend([
        "ld.global.u64 %q_ptr, [%q_state+24];",
        f"red.global.add.u64 [%q_ptr+{section*16}], 1;",
        f"red.global.add.u64 [%q_ptr+{section*16+8}], {width};",
        "ld.global.u64 %q_ptr, [%q_state+32];",
        f"red.global.add.u64 [%q_ptr+{site_offset}], 1;",
        f"bra {label}_done;",
        f"{label}_below:",
        f"add.u64 %q_end, %q_addr, {width};",
        "setp.le.u64 %q_skip, %q_end, %q_base;",
        f"@%q_skip bra {label}_done;",
        f"{label}_bad:",
        "ld.global.u64 %q_ptr, [%q_state+24];",
        "red.global.add.u64 [%q_ptr+32], 1;",
        f"{label}_done:",
        "}",
    ])
    return "\n".join(lines) + "\n"


def rewrite(source: str, unit: str) -> tuple[str, list[dict]]:
    if unit not in ("convert", "mmvf"):
        raise ValueError(f"Unexpected translation unit {unit}")
    symbol = f"hbflab_q_auto_state_{unit}"
    if not re.search(rf"\.global[^;\n]*\b{symbol}\b", source):
        raise ValueError(f"Compiler output omitted device symbol {symbol}")
    output = []
    manifest = []
    entry = None
    for number, line in enumerate(source.splitlines(keepends=True), 1):
        function = ENTRY.search(line) or FUNCTION.search(line)
        if function:
            entry = function[1]
        if GLOBAL_LOAD.match(line):
            match = LOAD.match(line)
            data_type = TYPE.search(match[2]) if match else None
            if not data_type or entry is None:
                raise ValueError(f"Cannot probe global load at {unit}:{number}: {line.rstrip()}")
            opcode = match[2]
            vector = VECTOR.search(opcode)
            width = (int(vector[1]) if vector else 1) * (int(data_type[1]) // 8 if data_type[1] else 4)
            site = len(manifest)
            site_index = site + (SITE_COUNT if unit == "mmvf" else 0)
            if site >= SITE_COUNT:
                raise ValueError(f"Too many sites in {unit}")
            immediate = match[5]
            offset = int(immediate, 16 if immediate.lstrip("-").lower().startswith("0x") else 10) if immediate else 0
            if match[4] == "-":
                offset = -offset
            address = (f"add.u64 %q_addr, {match[3]}, {offset};" if offset
                       else f"mov.u64 %q_addr, {match[3]};")
            output.append(probe(unit, site, match[1], address, width))
            manifest.append(dict(site=site_index, entry=entry, line=number, opcode=opcode,
                                 width=width, instruction=line.strip()))
        output.append(line)
    if not manifest:
        raise ValueError(f"No global loads found in {unit}")
    return "".join(output), manifest
