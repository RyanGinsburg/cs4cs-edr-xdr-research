#!/usr/bin/env python3
import pandas as pd #type: ignore
import numpy as np #type: ignore

def build_shared_keep_set(
    df: pd.DataFrame,
    label_col: str,
    pos_idx: list[int],
    k_before_list: np.ndarray,
    k_after_list: np.ndarray,
    window: int,
    rng: np.random.Generator
) -> list[int]:
    """
    Given the master df and pre-generated per-event k_before/k_after,
    collect a single 'keep' index set for all events:
      - include each i in pos_idx
      - for each i, sample k_before from [i-window..i-1] where label==0
                    and k_after  from [i+1..i+window] where label==0
    Returns a sorted list of unique row-indices.
    """
    n = len(df)
    keep = set()

    for i, k_before, k_after in zip(pos_idx, k_before_list, k_after_list):
        start, mid, end = max(0, i - window), i, min(n, i + window + 1)

        # negative candidates from master df
        before = [j for j in range(start, mid) if df.at[j, label_col] == 0]
        after  = [j for j in range(mid + 1, end) if df.at[j, label_col] == 0]

        # sample exactly k_before/k_after (clamped)
        sel_before = rng.choice(before, size=min(k_before, len(before)), replace=False) if before else []
        sel_after  = rng.choice(after,  size=min(k_after,  len(after)),  replace=False) if after else []

        keep.add(i)
        keep.update(sel_before)
        keep.update(sel_after)

    return sorted(keep)

def main():
    # Configuration
    EDR_MASTER_PATH = 'lanl_output/edr_master_sorted_filtered.csv'
    XDR_MASTER_PATH = 'lanl_output/xdr_master_sorted_filtered.csv'
    MIN_CTX   = 75      # min negatives to draw on each side
    MAX_CTX   = 150      # max negatives to draw on each side
    WINDOW    = MAX_CTX      # how far to look around each malicious event
    SEED      = None    # None => non-deterministic each run

    # 1) Load masters
    edr_master = pd.read_csv(EDR_MASTER_PATH).reset_index(drop=True)
    xdr_master = pd.read_csv(XDR_MASTER_PATH).reset_index(drop=True)

    # 2) Find all malicious indices once (EDR)
    pos_idx = edr_master.index[edr_master['label'] == 1].tolist()

    # 3) Pre-generate random counts
    rng = np.random.default_rng(SEED)
    k_before_list = rng.integers(MIN_CTX, MAX_CTX + 1, size=len(pos_idx))
    k_after_list  = rng.integers(MIN_CTX, MAX_CTX + 1, size=len(pos_idx))

    # 4) Build a single keep_set from the EDR master
    keep_indices = build_shared_keep_set(
        edr_master,
        label_col='label',
        pos_idx=pos_idx,
        k_before_list=k_before_list,
        k_after_list=k_after_list,
        window=WINDOW,
        rng=rng
    )

    # 5) Slice both datasets by the same indices
    edr_adjusted = edr_master.iloc[keep_indices].reset_index(drop=True)
    xdr_adjusted = xdr_master.iloc[keep_indices].reset_index(drop=True)

    # 6) Save out
    edr_adjusted.to_csv("lanl_output/adjusted_master_edr.csv", index=False)
    xdr_adjusted.to_csv("lanl_output/adjusted_master_xdr.csv", index=False)

    # 7) Report
    print("Wrote adjusted_master_edr.csv:", edr_adjusted.shape)
    print("Wrote adjusted_master_xdr.csv:", xdr_adjusted.shape)

if __name__ == "__main__":
    main()
