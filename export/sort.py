#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-efficient external sort & filter for LANL EDR/XDR master CSVs.

1. Streams input CSV in chunks, filters unwanted days.
2. Sorts each filtered chunk by (day, win) and writes to a temp file.
3. Merges all sorted chunks into a final chronologically ordered CSV.

Edit DAYS_TO_KEEP to specify which days to retain.
"""

import csv
import heapq
import os
import tempfile

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit as needed:
# ─────────────────────────────────────────────────────────────────────────────

# Days to keep (e.g., 0 through 57):
DAYS_TO_KEEP = {1, 2, 5, 6, 7, 8, 9,
                12, 13, 14, 15,
                20, 21, 22,
                26, 27, 28, 29}

# Input/output paths:
INPUT_OUTPUT = {
    'edr': {
        'in':  'lanl_output/edr_master.csv',
        'out': 'lanl_output/edr_master_sorted_filtered.csv'
    },
    'xdr': {
        'in':  'lanl_output/xdr_master.csv',
        'out': 'lanl_output/xdr_master_sorted_filtered.csv'
    }
}

# Number of rows to buffer per chunk (tune to available RAM):
CHUNK_SIZE = 200_000

# ─────────────────────────────────────────────────────────────────────────────
# IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def write_sorted_chunk(header, rows, day_idx, win_idx):
    """
    Sort rows by (day, win) and write to a temporary CSV file.
    Returns the temp file path.
    """
    rows.sort(key=lambda r: (int(r[day_idx]), int(r[win_idx])))
    tf = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', newline='')
    writer = csv.writer(tf)
    writer.writerow(header)
    writer.writerows(rows)
    path = tf.name
    tf.close()
    return path

def process_master(input_path, output_path):
    """
    External sort & filter:
      1) Read in streaming mode, buffer CHUNK_SIZE rows.
      2) Keep only rows with day in DAYS_TO_KEEP.
      3) Sort each chunk by (day, win) and write to temp files.
      4) Merge all sorted chunks into `output_path`.
    """
    print(f"→ Filtering & chunk-sorting {input_path}")
    temp_files = []

    # Phase 1: chunk, filter, sort, write
    with open(input_path, newline='') as fin:
        reader = csv.reader(fin)
        header = next(reader)
        day_idx = header.index('day')
        win_idx = header.index('win')
        buffer = []

        for row in reader:
            try:
                day = int(row[day_idx])
            except Exception:
                continue
            if day in DAYS_TO_KEEP:
                buffer.append(row)
            if len(buffer) >= CHUNK_SIZE:
                temp_files.append(write_sorted_chunk(header, buffer, day_idx, win_idx))
                buffer.clear()

        # final buffer
        if buffer:
            temp_files.append(write_sorted_chunk(header, buffer, day_idx, win_idx))
            buffer.clear()

    # Phase 2: k-way merge
    print(f"→ Merging {len(temp_files)} sorted chunks into {output_path}")
    with open(output_path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(header)

        # prepare iterators
        files = []
        iters = []
        for tf in temp_files:
            f = open(tf, newline='')
            files.append(f)
            r = csv.reader(f)
            next(r)  # skip header
            iters.append(r)

        # merge by key
        merged = heapq.merge(*iters, key=lambda r: (int(r[day_idx]), int(r[win_idx])))
        for row in merged:
            writer.writerow(row)

    # clean up
    for f, tf in zip(files, temp_files):
        f.close()
        os.remove(tf)

    total = sum(1 for _ in open(output_path)) - 1
    print(f"→ Done: {output_path} ({total} rows)")

def main():
    for key, paths in INPUT_OUTPUT.items():
        process_master(paths['in'], paths['out'])

if __name__ == '__main__':
    main()
