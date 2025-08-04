#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split sorted & filtered EDR/XDR master CSVs into train/test sets based on specified day lists.

This script streams through each master file once and writes out two new CSVs per input:
  - *_train.csv  (rows whose `day` is in TRAIN_DAYS)
  - *_test.csv   (rows whose `day` is in TEST_DAYS)

Adjust TRAIN_DAYS and TEST_DAYS as needed.
"""
import csv

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these sets of days
# ─────────────────────────────────────────────────────────────────────────────
# 80% train (first 14 of your malicious days):
TRAIN_DAYS = {1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 20, 21, 22}
# 20% test (last 4 of your malicious days):
TEST_DAYS  = {26, 27, 28, 29}

# Input → (train_output, test_output)
FILES = {
    'edr': (
        'lanl_output/edr_master_sorted_filtered.csv',
        'lanl_output/edr_train_sorted_adjusted.csv',
        'lanl_output/edr_test_sorted_adjusted.csv'
    ),
    'xdr': (
        'lanl_output/xdr_master_sorted_filtered.csv',
        'lanl_output/xdr_train_sorted_adjusted.csv',
        'lanl_output/xdr_test_sorted_adjusted.csv'
    )
}


def split_by_day(input_path: str, train_path: str, test_path: str):
    """
    Stream `input_path` and write rows into `train_path` or `test_path`
    based on `day` column membership in TRAIN_DAYS/TEST_DAYS.
    """
    with open(input_path, newline='') as fin, \
         open(train_path, 'w', newline='') as ftrain, \
         open(test_path, 'w', newline='') as ftest:

        reader = csv.reader(fin)
        header = next(reader)
        day_idx = header.index('day')

        writer_train = csv.writer(ftrain)
        writer_test  = csv.writer(ftest)
        # write headers to both files
        writer_train.writerow(header)
        writer_test.writerow(header)

        for row in reader:
            try:
                day = int(row[day_idx])
            except ValueError:
                # skip rows with invalid day
                continue
            if day in TRAIN_DAYS:
                writer_train.writerow(row)
            elif day in TEST_DAYS:
                writer_test.writerow(row)
            # otherwise skip


def main():
    for name, (inp, train_out, test_out) in FILES.items():
        print(f"Processing {name.upper()} master → split into train/test...")
        split_by_day(inp, train_out, test_out)
        print(f"  ✓ Wrote {train_out} and {test_out}\n")


if __name__ == '__main__':
    main()
