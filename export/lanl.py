#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build richer EDR (endpoint-only) and XDR (endpoint + network) datasets from the LANL
Comprehensive, Multi-Source Cyber-Security Events corpus.

Inputs (in ./lanl/):
  - auth.txt.gz, proc.txt.gz, flows.txt.gz, dns.txt.gz, redteam.txt.gz

Outputs (in ./lanl_output/):
  - edr_master.csv            (per host × time-window features + label)
  - xdr_master.csv            (EDR features + network features + label)
  - edr_train_all.csv, edr_test_all.csv
  - xdr_train_all.csv, xdr_test_all.csv

Design:
- Uses fixed time windows (default 5 minutes) and aggregates by source computer ("host").
- EDR features come from host logs (auth, proc).
- XDR = EDR + network (flows, dns) context.
- Labels are assigned per (host, window) if redteam touches that host in that window.
- Avoids leakage: all features are computed from the same window only (no future info).
"""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd #type:ignore
import numpy as np #type: ignore

# ---------------------------- CONFIG ---------------------------------

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "lanl"              # data folder (sibling to this script)
OUT_DIR  = BASE_DIR / "lanl_output"       # outputs written here
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_WINDOW_SECS = 300      # 5 minutes
CHUNKSIZE         = 2_000_000   # tune to your RAM/CPU; 2M rows/chunk is a good start

# Input files
AUTH_F     = DATA_DIR / "auth.txt"
PROC_F     = DATA_DIR / "proc.txt"
FLOWS_F    = DATA_DIR / "flows.txt"
DNS_F      = DATA_DIR / "dns.txt"
REDTEAM_F  = DATA_DIR / "redteam.txt"

# Output files
EDR_MASTER_CSV = OUT_DIR / "edr_master.csv"
XDR_MASTER_CSV = OUT_DIR / "xdr_master.csv"
EDR_TRAIN_CSV  = OUT_DIR / "edr_train_all.csv"
EDR_TEST_CSV   = OUT_DIR / "edr_test_all.csv"
XDR_TRAIN_CSV  = OUT_DIR / "xdr_train_all.csv"
XDR_TEST_CSV   = OUT_DIR / "xdr_test_all.csv"

# ------------------------- HELPERS ------------------------------------

def to_int_series(s: pd.Series) -> pd.Series:
    """Coerce to integer, treating '?' and invalid as NaN (nullable Int64)."""
    s = pd.to_numeric(s, errors="coerce")
    return s.astype("Int64")

def window_floor(ts: pd.Series, width: int) -> pd.Series:
    """Floor integer seconds to the start of the time window."""
    v = ts.to_numpy(dtype="float64")
    v = np.floor(v / width) * width
    return pd.Series(v.astype("int64"), index=ts.index)

def day_id_from_time(ts: pd.Series) -> pd.Series:
    """Map integer seconds to day index starting at 0 (86400 seconds per day)."""
    v = ts.to_numpy(dtype="float64")
    v = np.floor(v / 86400.0)
    return pd.Series(v.astype("int64"), index=ts.index)

# ---------------------- LABELS FROM REDTEAM ---------------------------

def build_redteam_host_window_set() -> set[tuple[str, int]]:
    """
    Build a set of keys (host, window_start) that should be labeled malicious.
    We consider both src_computer and dst_computer from redteam.txt.gz.
    """
    print("  → Reading redteam.txt in chunks...")
    cols = ["time", "user", "src_comp", "dst_comp"]
    rt_iter = pd.read_csv(
        REDTEAM_F, header=None, names=cols,
        na_values=["?"], dtype=str, chunksize=200_000
    )
    bad_keys: set[tuple[str, int]] = set()

    chunk_count = 0
    for chunk in rt_iter:
        chunk_count += 1
        if chunk_count % 5 == 0:
            print(f"    Processing redteam chunk {chunk_count}...")
        chunk["time"] = to_int_series(chunk["time"])
        chunk = chunk.dropna(subset=["time", "src_comp", "dst_comp"])
        chunk["win"] = window_floor(chunk["time"].astype("int64"), TIME_WINDOW_SECS)

        for col in ("src_comp", "dst_comp"):
            tmp = chunk[[col, "win"]].dropna()
            bad_keys.update((h, int(w)) for h, w in zip(tmp[col], tmp["win"]))
    
    print(f"  → Found {len(bad_keys)} malicious (host, window) pairs")
    return bad_keys

# ---------------------- AGGREGATORS (CHUNKED) -------------------------

def aggregate_auth() -> pd.DataFrame:
    """
    auth.txt per host×window aggregates (richer but additive features):
      - auth_total, auth_success, auth_failure
      - logon type counts: auth_interactive, auth_remote, auth_service, auth_batch, auth_network, auth_unlock, auth_other
      - Derived: auth_fail_rate = failure / total
    """
    print("  → Reading auth.txt in chunks...")
    cols = ["time","src_user","dst_user","src_comp","dst_comp","auth_type","logon_type","orientation","result"]
    it = pd.read_csv(
        AUTH_F, header=None, names=cols,
        na_values=["?"], dtype=str, chunksize=CHUNKSIZE
    )

    out_parts = []
    chunk_count = 0
    for df in it:
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"    Processing auth chunk {chunk_count}...")

        df["time"] = to_int_series(df["time"])
        df = df.dropna(subset=["time","src_comp"])
        df["win"] = window_floor(df["time"].astype("int64"), TIME_WINDOW_SECS)
        df["day"] = day_id_from_time(df["time"].astype("int64"))

        # Booleans
        df["is_success"] = (df["result"] == "Success").astype("int8")
        df["is_failure"] = (df["result"] == "Failure").astype("int8")

        df["is_interactive"] = (df["logon_type"] == "Interactive").astype("int8")
        df["is_remote"]      = (df["logon_type"] == "RemoteInteractive").astype("int8")
        df["is_service"]     = (df["logon_type"] == "Service").astype("int8")
        df["is_batch"]       = (df["logon_type"] == "Batch").astype("int8")
        df["is_network"]     = (df["logon_type"] == "Network").astype("int8")
        df["is_unlock"]      = (df["logon_type"] == "Unlock").astype("int8")

        # "other" = known types NOT matched, and logon_type present
        known = {"Interactive","RemoteInteractive","Service","Batch","Network","Unlock"}
        df["is_other_logon"] = (
            (~df["logon_type"].isin(known)) & (df["logon_type"].notna())
        ).astype("int8")

        g = df.groupby(["src_comp","day","win"], sort=False).agg(
            auth_total        = ("result","size"),
            auth_success      = ("is_success","sum"),
            auth_failure      = ("is_failure","sum"),
            auth_interactive  = ("is_interactive","sum"),
            auth_remote       = ("is_remote","sum"),
            auth_service      = ("is_service","sum"),
            auth_batch        = ("is_batch","sum"),
            auth_network      = ("is_network","sum"),
            auth_unlock       = ("is_unlock","sum"),
            auth_other        = ("is_other_logon","sum"),
        ).reset_index()

        out_parts.append(g)

    if not out_parts:
        cols_out = [
            "src_comp","day","win","auth_total","auth_success","auth_failure",
            "auth_interactive","auth_remote","auth_service","auth_batch",
            "auth_network","auth_unlock","auth_other","auth_fail_rate"
        ]
        return pd.DataFrame(columns=cols_out)

    auth_agg = (
        pd.concat(out_parts, ignore_index=True)
          .groupby(["src_comp","day","win"], as_index=False).sum()
    )

    # Vectorized failure rate (avoid apply)
    auth_agg["auth_fail_rate"] = np.where(
        auth_agg["auth_total"] > 0,
        auth_agg["auth_failure"] / auth_agg["auth_total"],
        0.0
    )

    print(f"  → Auth aggregation complete: {len(auth_agg)} records")
    return auth_agg


def aggregate_proc() -> pd.DataFrame:
    """
    proc.txt.gz per host×window aggregates:
      - proc_events, proc_start, proc_end
      - proc_service_events (user endswith '$@DOM*'), proc_user_events (= events - service)
    """
    print("  → Reading proc.txt in chunks...")
    cols = ["time","user","computer","process","event"]
    it = pd.read_csv(PROC_F, header=None, names=cols,
                     na_values=["?"], dtype=str, chunksize=CHUNKSIZE)

    out_parts = []
    chunk_count = 0
    for df in it:
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"    Processing proc chunk {chunk_count}...")
        df["time"] = to_int_series(df["time"])
        df = df.dropna(subset=["time","computer"])
        df["win"] = window_floor(df["time"].astype("int64"), TIME_WINDOW_SECS)
        df["day"] = day_id_from_time(df["time"].astype("int64"))

        df["is_start"] = (df["event"] == "Start").astype("int8")
        df["is_end"]   = (df["event"] == "End").astype("int8")
        # service accounts typically end with '$' (e.g., C625$@DOM1)
        df["is_service_acct"] = df["user"].fillna("").str.contains(r"\$\@").astype("int8")

        g = df.groupby(["computer","day","win"], sort=False).agg(
            proc_events          = ("event","size"),
            proc_start           = ("is_start","sum"),
            proc_end             = ("is_end","sum"),
            proc_service_events  = ("is_service_acct","sum"),
        ).reset_index().rename(columns={"computer":"src_comp"})

        g["proc_user_events"] = g["proc_events"] - g["proc_service_events"]
        out_parts.append(g)

    if not out_parts:
        cols_out = ["src_comp","day","win","proc_events","proc_start","proc_end","proc_service_events","proc_user_events"]
        return pd.DataFrame(columns=cols_out)

    result = pd.concat(out_parts, ignore_index=True).groupby(["src_comp","day","win"], as_index=False).sum()
    print(f"  → Proc aggregation complete: {len(result)} records")
    return result


def _to_int_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def aggregate_flows() -> pd.DataFrame:
    """
    flows.txt.gz per host×window aggregates (add richer but additive features):
      - flows_count, flows_bytes, flows_packets, flows_duration_sum
      - tcp_flows, udp_flows
      - wellknown_dst_flows (dst_port <= 1024 when numeric)
      - http80_flows, https443_flows
      - Derived (computed later): bytes_per_flow, packets_per_flow, tcp_fraction, wellknown_fraction
    """
    print("  → Reading flows.txt in chunks...")
    cols = ["time","duration","src_comp","src_port","dst_comp","dst_port","protocol","packets","bytes"]
    it = pd.read_csv(FLOWS_F, header=None, names=cols,
                     na_values=["?"], dtype=str, chunksize=CHUNKSIZE)

    out_parts = []
    chunk_count = 0
    for df in it:
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"    Processing flows chunk {chunk_count}...")
        # numeric conversions
        for c in ("time","duration","packets","bytes","protocol"):
            df[c] = _to_int_safe(df[c])

        df = df.dropna(subset=["time","src_comp"])
        df["win"] = window_floor(df["time"].astype("int64"), TIME_WINDOW_SECS)
        df["day"] = day_id_from_time(df["time"].astype("int64"))

        # protocol flags
        df["is_tcp"] = (df["protocol"] == 6).astype("int8")
        df["is_udp"] = (df["protocol"] == 17).astype("int8")

        # destination port parsing (numeric when possible)
        dstp = pd.to_numeric(df["dst_port"], errors="coerce")
        df["dst_port_num"] = dstp
        df["is_wellknown"] = ((dstp.notna()) & (dstp <= 1024)).astype("int8")
        df["is_http80"]    = (dstp == 80).astype("int8")
        df["is_https443"]  = (dstp == 443).astype("int8")

        g = df.groupby(["src_comp","day","win"], sort=False).agg(
            flows_count        = ("time","size"),
            flows_bytes        = ("bytes","sum"),
            flows_packets      = ("packets","sum"),
            flows_duration_sum = ("duration","sum"),
            tcp_flows          = ("is_tcp","sum"),
            udp_flows          = ("is_udp","sum"),
            wellknown_dst_flows= ("is_wellknown","sum"),
            http80_flows       = ("is_http80","sum"),
            https443_flows     = ("is_https443","sum"),
        ).reset_index()

        out_parts.append(g)

    if not out_parts:
        cols_out = ["src_comp","day","win","flows_count","flows_bytes","flows_packets","flows_duration_sum",
                    "tcp_flows","udp_flows","wellknown_dst_flows","http80_flows","https443_flows"]
        return pd.DataFrame(columns=cols_out)

    flows_agg = pd.concat(out_parts, ignore_index=True).groupby(["src_comp","day","win"], as_index=False).sum()

    # Derived ratios (safe division)
    print("  → Computing flow ratios...")
    flows_agg["bytes_per_flow"]     = flows_agg.apply(lambda r: float(r["flows_bytes"]) / float(r["flows_count"]) if r["flows_count"] else 0.0, axis=1)
    flows_agg["packets_per_flow"]   = flows_agg.apply(lambda r: float(r["flows_packets"]) / float(r["flows_count"]) if r["flows_count"] else 0.0, axis=1)
    flows_agg["tcp_fraction"]       = flows_agg.apply(lambda r: float(r["tcp_flows"]) / float(r["flows_count"]) if r["flows_count"] else 0.0, axis=1)
    flows_agg["wellknown_fraction"] = flows_agg.apply(lambda r: float(r["wellknown_dst_flows"]) / float(r["flows_count"]) if r["flows_count"] else 0.0, axis=1)

    print(f"  → Flows aggregation complete: {len(flows_agg)} records")
    return flows_agg


def aggregate_dns() -> pd.DataFrame:
    """
    dns.txt.gz per host×window aggregates:
      - dns_queries (total count in window)
    (Richer DNS features like rarity or unique domains would require uniqueness
     across chunks; omitted here to keep it scalable and additive.)
    """
    print("  → Reading dns.txt in chunks...")
    cols = ["time","src_comp","resolved"]
    it = pd.read_csv(DNS_F, header=None, names=cols,
                     na_values=["?"], dtype=str, chunksize=CHUNKSIZE)

    out_parts = []
    chunk_count = 0
    for df in it:
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"    Processing dns chunk {chunk_count}...")
        df["time"] = to_int_series(df["time"])
        df = df.dropna(subset=["time","src_comp"])
        df["win"] = window_floor(df["time"].astype("int64"), TIME_WINDOW_SECS)
        df["day"] = day_id_from_time(df["time"].astype("int64"))

        g = df.groupby(["src_comp","day","win"], sort=False).agg(
            dns_queries=("time","size"),
        ).reset_index()

        out_parts.append(g)

    if not out_parts:
        cols_out = ["src_comp","day","win","dns_queries"]
        return pd.DataFrame(columns=cols_out)

    result = pd.concat(out_parts, ignore_index=True).groupby(["src_comp","day","win"], as_index=False).sum()
    print(f"  → DNS aggregation complete: {len(result)} records")
    return result

# ---------------------- BUILD DATASETS --------------------------------

def build_edr_xdr():
    print("Building redteam label set (host × window)…")
    bad_keys = build_redteam_host_window_set()
    bad_df = pd.DataFrame(list(bad_keys), columns=["src_comp", "win"]).drop_duplicates()
    bad_df["label"] = 1

    print("Aggregating AUTH (richer)…")
    auth_agg = aggregate_auth()

    print("Aggregating PROC (richer)…")
    proc_agg = aggregate_proc()

    print("Merging AUTH + PROC → EDR features…")
    edr = pd.merge(auth_agg, proc_agg, on=["src_comp","day","win"], how="outer").fillna(0)

    # Label EDR
    edr = edr.merge(bad_df, on=["src_comp","win"], how="left")
    edr["label"] = edr["label"].fillna(0).astype("uint8")

    print("Saving master EDR →", EDR_MASTER_CSV)
    edr.to_csv(EDR_MASTER_CSV, index=False)

    print("Aggregating FLOWS (richer)…")
    flows_agg = aggregate_flows()

    print("Aggregating DNS…")
    dns_agg = aggregate_dns()

    print("Merging EDR + FLOWS + DNS → XDR features (chunked)…")
    if XDR_MASTER_CSV.exists():
        XDR_MASTER_CSV.unlink()  # remove if exists

    chunk_size = 1_000_000
    total_rows = len(edr)
    for i in range(0, total_rows, chunk_size):
        chunk = edr.iloc[i:i+chunk_size].drop(columns=["label"])
        # Merge flows and dns for the chunk
        chunk = chunk.merge(flows_agg, on=["src_comp","day","win"], how="left") \
                     .merge(dns_agg, on=["src_comp","day","win"], how="left") \
                     .fillna(0)
        # Add label
        chunk = chunk.merge(bad_df, on=["src_comp","win"], how="left")
        chunk["label"] = chunk["label"].fillna(0).astype("uint8")

        # Append to master CSV
        chunk.to_csv(XDR_MASTER_CSV, mode="a", index=False, header=not XDR_MASTER_CSV.exists())

        print(f"  → Processed {i+len(chunk)}/{total_rows} rows for XDR")

    print("XDR master CSV saved.")
    
    # Return both edr and a placeholder/summary for xdr
    print("Loading XDR summary for split...")
    xdr_summary = pd.read_csv(XDR_MASTER_CSV, usecols=["src_comp", "day", "win", "label"])
    return edr, xdr_summary


# ---------------------- TRAIN/TEST SPLIT ------------------------------

def split_by_last_days(df: pd.DataFrame, last_n_days: int = 7):
    # Find max day in the dataset
    max_day = int(df["day"].max())
    test_days = set(range(max_day - last_n_days + 1, max_day + 1))
    # Chronological order within each split
    train = df[~df["day"].isin(test_days)].sort_values(["day", "win"]).reset_index(drop=True)
    test  = df[df["day"].isin(test_days)].sort_values(["day", "win"]).reset_index(drop=True)
    return train, test

def split_and_save(edr: pd.DataFrame, xdr_summary: pd.DataFrame, last_n_days: int = 7):
    print(f"Splitting by last {last_n_days} full days → test…")
    edr_train, edr_test = split_by_last_days(edr, last_n_days=last_n_days)
    
    # For XDR, we'll read and split the master CSV in chunks
    print("Splitting XDR master CSV...")
    xdr_train_rows = []
    xdr_test_rows = []
    
    # Use the summary to determine split
    xdr_train_summary, xdr_test_summary = split_by_last_days(xdr_summary, last_n_days=last_n_days)
    
    # Create sets for faster lookup
    train_keys = set(zip(xdr_train_summary['src_comp'], xdr_train_summary['day'], xdr_train_summary['win']))
    test_keys = set(zip(xdr_test_summary['src_comp'], xdr_test_summary['day'], xdr_test_summary['win']))
    
    # Read XDR master in chunks and split
    chunk_iter = pd.read_csv(XDR_MASTER_CSV, chunksize=100_000)
    for chunk in chunk_iter:
        chunk_keys = list(zip(chunk['src_comp'], chunk['day'], chunk['win']))
        
        train_mask = [key in train_keys for key in chunk_keys]
        test_mask = [key in test_keys for key in chunk_keys]
        
        train_chunk = chunk[train_mask]
        test_chunk = chunk[test_mask]
        
        if len(train_chunk) > 0:
            train_chunk.to_csv(XDR_TRAIN_CSV, mode='a', index=False, 
                             header=not XDR_TRAIN_CSV.exists())
        if len(test_chunk) > 0:
            test_chunk.to_csv(XDR_TEST_CSV, mode='a', index=False, 
                             header=not XDR_TEST_CSV.exists())

    print(f"  → EDR: {len(edr_train)} train, {len(edr_test)} test")
    print(f"  → XDR: {len(xdr_train_summary)} train, {len(xdr_test_summary)} test")

    edr_train.to_csv(EDR_TRAIN_CSV, index=False)
    edr_test.to_csv(EDR_TEST_CSV, index=False)

# ---------------------- MAIN -----------------------------------------

def main():
    print("=== LANL EDR/XDR Dataset Builder ===")
    print("Checking input files exist...")
    # Check inputs exist
    for p in (AUTH_F, PROC_F, FLOWS_F, DNS_F, REDTEAM_F):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
    print("✓ All input files found")

    print("\nBuilding datasets...")
    edr, xdr_summary = build_edr_xdr()
    
    print(f"\nDataset summary:")
    print(f"  EDR records: {len(edr)}")
    print(f"  XDR records: {len(xdr_summary)}")
    print(f"  EDR malicious: {edr['label'].sum()}")
    print(f"  XDR malicious: {xdr_summary['label'].sum()}")
    
    split_and_save(edr, xdr_summary)
    print("\n=== Processing complete! ===")

if __name__ == "__main__":
    pd.set_option("display.width", 150)
    main()
