#!/usr/bin/env python3
"""Simple encoder for OpenBCI-style EEG sample text files.

Produces CSV, JSONL, or per-sample base64-packed binary of channel values.

Usage:
  python encode_neural_to_text.py --input input.txt --output out.jsonl --format jsonl
"""
import argparse
import csv
import json
import os
import struct
import base64


def parse_header(lines):
    meta = {}
    for l in lines:
        if not l.startswith('%'):
            break
        if '=' in l:
            k, v = l[1:].strip().split('=', 1)
            meta[k.strip()] = v.strip()
    if 'Number of channels' in meta:
        try:
            meta['num_channels'] = int(meta['Number of channels'])
        except Exception:
            meta['num_channels'] = None
    else:
        meta['num_channels'] = None
    return meta


def parse_line(line, num_channels):
    # split on commas, keep tokens
    toks = [t.strip() for t in line.strip().split(',') if t.strip() != '']
    if not toks:
        return None

    # detect trailing timestamp/time string
    timestamp_ms = None
    timestr = None
    if len(toks) >= 2 and (':' in toks[-1]) and toks[-2].isdigit():
        try:
            timestamp_ms = int(float(toks[-2]))
            timestr = toks[-1]
            toks = toks[:-2]
        except Exception:
            pass

    # first token may be sample index
    idx = None
    try:
        maybe_idx = int(toks[0])
        idx = maybe_idx
        data_start = 1
    except Exception:
        data_start = 0

    # collect channel values
    channels = []
    if num_channels is None:
        # fallback: take first 8 values or as many as available
        count = min(8, max(0, len(toks) - data_start))
    else:
        count = min(num_channels, max(0, len(toks) - data_start))

    for i in range(count):
        try:
            channels.append(float(toks[data_start + i]))
        except Exception:
            channels.append(None)

    # remaining tokens after channels considered aux
    aux = []
    for j in range(data_start + count, len(toks)):
        try:
            aux.append(float(toks[j]))
        except Exception:
            aux.append(toks[j])

    return {
        'index': idx,
        'channels': channels,
        'aux': aux,
        'timestamp_ms': timestamp_ms,
        'time_str': timestr,
    }


def encode_base64_int32(samples):
    # pack channels as little-endian int32 per sample, return base64 bytes (single blob)
    buf = bytearray()
    for s in samples:
        for v in s['channels']:
            if v is None:
                buf += struct.pack('<i', 0)
            else:
                # cast float to int
                buf += struct.pack('<i', int(v))
    return base64.b64encode(bytes(buf)).decode('ascii')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--output', '-o', required=True)
    p.add_argument('--format', '-f', choices=['jsonl', 'csv', 'base64'], default='jsonl')
    p.add_argument('--num-channels', type=int, default=None)
    args = p.parse_args()

    with open(args.input, 'r', encoding='utf-8', errors='ignore') as fh:
        lines = [l.rstrip('\n') for l in fh]

    meta = parse_header(lines)
    num_channels = args.num_channels if args.num_channels is not None else meta.get('num_channels')

    # find first non-header line
    data_lines = [l for l in lines if not l.startswith('%') and l.strip() != '']
    samples = []
    for line in data_lines:
        parsed = parse_line(line, num_channels)
        if parsed:
            samples.append(parsed)

    outdir = os.path.dirname(args.output)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    if args.format == 'jsonl':
        with open(args.output, 'w', encoding='utf-8') as out:
            for s in samples:
                out.write(json.dumps(s) + '\n')

    elif args.format == 'csv':
        # columns: index, ch0..chN-1, aux0..M-1, timestamp_ms, time_str
        max_aux = max((len(s['aux']) for s in samples), default=0)
        chan_count = max((len(s['channels']) for s in samples), default=0)
        headers = ['index'] + [f'ch{i}' for i in range(chan_count)] + [f'aux{i}' for i in range(max_aux)] + ['timestamp_ms', 'time_str']
        with open(args.output, 'w', newline='', encoding='utf-8') as out:
            w = csv.writer(out)
            w.writerow(headers)
            for s in samples:
                row = [s['index']] + s['channels'] + s['aux'] + [s['timestamp_ms'], s['time_str']]
                # pad
                if len(row) < len(headers):
                    row += [None] * (len(headers) - len(row))
                w.writerow(row)

    elif args.format == 'base64':
        b64 = encode_base64_int32(samples)
        with open(args.output, 'w', encoding='utf-8') as out:
            out.write(b64 + '\n')

    print(f'Wrote {len(samples)} samples to {args.output}')


if __name__ == '__main__':
    main()
