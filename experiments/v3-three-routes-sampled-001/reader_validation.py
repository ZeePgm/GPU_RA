"""Independent binary request, every-lane dependency and chunk-publication audit."""
import json
from pathlib import Path
import numpy as np

REQUEST = np.dtype([(name, '<u8') for name in (
    'launch', 'ticket', 'offset', 'logical', 'arrival_ns', 'arrival_ps', 'end_ps',
    'first_ns', 'last_ns', 'time_file_offset', 'grant_floor_ns')]
    + [(name, '<u4') for name in ('range', 'bytes', 'producer', 'unit')])


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_reader(folder, spans, inventory, route, addresses):
    folder = Path(folder)
    require(REQUEST.itemsize == 104, 'Request ABI')
    require((folder / 'requests.bin').stat().st_size % 104 == 0, 'Truncated request record')
    requests = np.memmap(folder / 'requests.bin', dtype=REQUEST, mode='r')
    timing = np.memmap(folder / 'lane-times.bin', dtype='<u8', mode='r')
    ranges = [t for t in inventory['all_tensors'] if t['selected']]
    indices = {t['name']: i for i, t in enumerate(ranges)}
    tickets = np.zeros(2048, dtype='<u8')
    request_position = time_position = total_bytes = total_lanes = total_chunks = 0
    with (folder / 'chunks.jsonl').open(encoding='utf-8') as stream:
        chunks = iter(stream)
        for span in spans:
            tensor = next(t for t in ranges if t['name'] == span['tensor'])
            index = indices[tensor['name']]
            element = 4 if tensor['dtype'] == 'F32' else 2
            offset, size = span['offset'], span['bytes']
            require(offset % element == size % element == 0 and size > 0 and offset + size <= tensor['bytes'], 'Expected typed span invalid')
            unit = 16 if (addresses[tensor['name']] + offset) % 16 == size % 16 == 0 else 2
            limit = (32768 if unit == 16 else 4096) if route == 'ptx' else (262144 if unit == 16 else 32768)
            done = 0
            while done < size:
                line = next(chunks, None)
                require(line is not None, 'Missing read chunk')
                chunk = json.loads(line)
                count = min(limit, size - done)
                require(chunk['range'] == index and chunk['tensor'] == tensor['name'] and chunk['offset'] == offset + done
                        and chunk['bytes'] == count and chunk['unit'] == unit, 'Chunk does not match independent expected span')
                groups = (count + 32 * unit - 1) // (32 * unit)
                require(chunk['requests'] == groups and chunk['time_file_offset'] == time_position
                        and chunk['time_bytes'] == count // unit * 16, 'Chunk request/timing extent mismatch')
                records = requests[request_position:request_position + groups]
                require(len(records) == groups, 'Missing request records')
                records = records[np.argsort(records['offset'])]
                q = np.arange(groups, dtype=np.uint64)
                relative = q * (32 * unit)
                expected_offsets = np.uint64(offset + done) + relative
                expected_bytes = np.minimum(32 * unit, count - relative)
                expected_producers = (q % 64) * 32
                require(np.all(records['range'] == index) and np.all(records['launch'] == chunk['launch'])
                        and np.all(records['offset'] == expected_offsets)
                        and np.all(records['logical'] == tensor['logical_base'] + expected_offsets)
                        and np.all(records['bytes'] == expected_bytes) and np.all(records['producer'] == expected_producers)
                        and np.all(records['unit'] == unit), 'Request coverage or producer layout mismatch')
                expected_tickets = tickets[expected_producers] + q // 64 + 1
                require(np.all(records['ticket'] == expected_tickets), 'Ticket reuse or missing producer request')
                np.maximum.at(tickets, expected_producers, records['ticket'])
                require(np.all(records['arrival_ns'] >= records['grant_floor_ns']) and
                        np.all(records['end_ps'] >= records['arrival_ps']), 'Request causality mismatch')
                deadline = records['arrival_ns'] + (records['end_ps'] - records['arrival_ps'] + 999) // 1000
                lane_count = count // unit
                lane = timing[time_position // 8:time_position // 8 + 2 * lane_count].reshape(lane_count, 2)
                require(np.all(records['time_file_offset'] == time_position), 'Request timing identity mismatch')
                full = lane_count // 32
                if full:
                    blocks = lane[:full * 32].reshape(full, 32, 2)
                    require(np.all(blocks[:, :, 0] <= records['arrival_ns'][:full, None]) and
                            np.all(blocks[:, :, 1] >= deadline[:full, None]), 'Actual lane consumed before completion')
                    require(np.all(blocks[:, :, 0].min(axis=1) == records['first_ns'][:full]) and
                            np.all(blocks[:, :, 1].max(axis=1) == records['last_ns'][:full]), 'Lane summary differs from raw times')
                if lane_count % 32:
                    tail = lane[full * 32:]
                    require(np.all(tail[:, 0] <= records['arrival_ns'][-1]) and np.all(tail[:, 1] >= deadline[-1]), 'Tail lane dependency')
                    require(tail[:, 0].min() == records['first_ns'][-1] and tail[:, 1].max() == records['last_ns'][-1], 'Tail lane summary')
                require(chunk['release_stamp_ns'] >= int(records['last_ns'].max()), 'Publication released before reader completion')
                if route == 'symbolic':
                    require(chunk['hashes'] is None and not chunk['payload_published'], 'Symbolic route claimed payload publication')
                else:
                    hashes = chunk['hashes']
                    require(chunk['payload_published'] and len(hashes['reference']) == 64 and
                            hashes['reference'] == hashes['returned'] == hashes['published'], 'Returned/publication chunk hash differs')
                request_position += groups
                time_position += count // unit * 16
                total_lanes += lane_count
                total_bytes += count
                total_chunks += 1
                done += count
        require(next(chunks, None) is None, 'Unplanned extra read chunk')
    require(request_position == len(requests) and time_position == timing.nbytes, 'Unplanned request or raw lane timing records')
    summary = read(folder / 'summary.json')
    payload = 0 if route == 'symbolic' else total_bytes
    require(summary['status'] == 'pass' and summary['route'] == route and summary['requests'] == request_position
            and summary['requested_bytes'] == total_bytes and summary['payload_bytes'] == payload
            and summary['publication_bytes'] == payload and summary['byte_checks'] == 2 * payload
            and summary['lane_checks'] == total_lanes and summary['launches'] == total_chunks, 'Reader summary accounting')
    activities = [json.loads(line) for line in (folder.parent / 'gpu-activity.jsonl').read_text(encoding='utf-8').splitlines()]
    ptx_kernels = [r['name'] for r in activities if r['kind'] == 'kernel' and 'sampled_copy' in r['name']]
    cuda_kernels = [r['name'] for r in activities if r['kind'] == 'kernel' and 'sampled_window' in r['name']]
    require((len(ptx_kernels) == total_chunks and not cuda_kernels) if route == 'ptx'
            else (len(cuda_kernels) == total_chunks and not ptx_kernels), 'Actual GPU reader route differs from declared entrance')
    return {'status': 'pass', 'requests': request_position, 'requested_bytes': total_bytes, 'payload_bytes': payload,
            'publication_bytes': payload, 'lane_checks': total_lanes, 'chunks': total_chunks,
            'all_lane_times_recomputed': True, 'chunk_publication_hashes_equal': route != 'symbolic',
            'actual_reader_kernel_names': sorted(set(ptx_kernels + cuda_kernels))}
