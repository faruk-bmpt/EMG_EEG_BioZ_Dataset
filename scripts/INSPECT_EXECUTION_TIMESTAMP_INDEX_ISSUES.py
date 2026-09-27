#!/usr/bin/env python3
"""
INSPECT_EXECUTION_TIMESTAMP_INDEX_ISSUES.py

Purpose
-------
Systematic inspection of timestamp and Sample Index anomalies across all
Motor Execution CSV recordings.

This script is READ-ONLY:
    - No raw CSV is modified
    - No timestamps are repaired
    - No Sample Index is repaired
    - No rows are deleted
    - No interpolation is performed

Primary question
----------------
Determine whether the timestamp/index anomalies seen in files such as:

    252 -> 254 -> 0 -> 2 -> 4

represent acquisition-counter wrapping / export behavior while the EEG
signal itself remains intact, or whether they indicate actual signal
corruption.

Scope
-----
All Motor Execution recordings.

Expected:
    40 subjects
    7 gestures
    3 sets
    840 CSV files
    5,625 rows per recording
    13 EEG channels
    3 EMG channels
"""

from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    "/mnt/f/Faruk/OFS_Paper_Work/"
    "Data_Set_Paper_Work/EEG_EMG_BIOZ_DATASET"
)

RAW_ROOT = ROOT / "01_RAW_DATA" / "MOTOR_EXECUTION"

QC_ROOT = ROOT / "05_QC" / "TIMESTAMP_INDEX_SIGNAL_INSPECTION"

REPORT_CSV = QC_ROOT / "execution_timestamp_index_signal_inspection.csv"
SUMMARY_JSON = QC_ROOT / "execution_timestamp_index_signal_summary.json"

# Timestamp thresholds
TIMESTAMP_WARN_MS = 12.0
TIMESTAMP_MODERATE_MS = 20.0
TIMESTAMP_MAJOR_MS = 50.0
TIMESTAMP_SEVERE_MS = 100.0

# Expected recording structure
EXPECTED_ROWS = 5625
EXPECTED_SIGNAL_CHANNELS = 16
EXPECTED_EEG_CHANNELS = 13
EXPECTED_EMG_CHANNELS = 3

# Signal continuity inspection
LOCAL_CONTEXT = 20

# A signal is considered "non-flat" if its standard deviation is above
# this very small numerical threshold.
FLAT_STD_THRESHOLD = 1e-12

# Correlation requires enough non-constant samples.
MIN_CORR_SAMPLES = 20


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def identify_signal_columns(df: pd.DataFrame):
    """
    Identify EMG and EEG columns without assuming exact total ordering.
    """

    emg = [
        c for c in df.columns
        if str(c).strip().upper().startswith("EMG_")
    ]

    eeg = [
        c for c in df.columns
        if str(c).strip().upper().startswith("EEG_")
    ]

    return emg, eeg


def parse_timestamp_series(df: pd.DataFrame):
    """
    Parse the formatted timestamp column.

    Returns
    -------
    timestamp_series, valid_mask
    """

    candidates = [
        "Timestamp (Formatted)",
        "Timestamp",
        "timestamp",
    ]

    timestamp_col = None

    for c in candidates:
        if c in df.columns:
            timestamp_col = c
            break

    if timestamp_col is None:
        return None, None

    ts = pd.to_datetime(
        df[timestamp_col],
        errors="coerce"
    )

    return ts, ts.notna()


def safe_corr(a, b):
    """
    Pearson correlation, returning NaN when mathematically undefined.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    finite = np.isfinite(a) & np.isfinite(b)

    if finite.sum() < MIN_CORR_SAMPLES:
        return np.nan

    a = a[finite]
    b = b[finite]

    if np.std(a) <= FLAT_STD_THRESHOLD:
        return np.nan

    if np.std(b) <= FLAT_STD_THRESHOLD:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def classify_index_transition(delta):
    """
    Classify Sample Index transitions.

    The common observed pattern:

        252 -> 254 -> 0 -> 2 -> 4

    produces:

        +2, -254, +2, +2

    The -254 transition is therefore treated separately from arbitrary
    negative transitions.
    """

    if np.isnan(delta):
        return "NA"

    if delta == 0:
        return "ZERO"

    if delta == -254:
        return "COUNTER_WRAP_-254"

    if delta > 0:
        return "POSITIVE"

    if delta < 0:
        return "NEGATIVE"

    return "OTHER"


def contiguous_local_window(start, end, n):
    """
    Return safe inclusive/exclusive bounds.
    """

    lo = max(0, start)
    hi = min(n, end)

    return lo, hi


# ============================================================
# SINGLE FILE INSPECTION
# ============================================================

def inspect_file(path: Path):
    result = {
        "file": str(path.relative_to(RAW_ROOT)),
        "absolute_path": str(path),
        "sha256": "",
        "readable": False,

        "rows": np.nan,
        "columns": np.nan,

        "emg_channels": np.nan,
        "eeg_channels": np.nan,
        "signal_channels": np.nan,

        "nan_total": np.nan,
        "inf_total": np.nan,

        # Timestamp
        "timestamp_valid": False,
        "timestamp_unique": np.nan,
        "timestamp_duplicate_rows": np.nan,
        "timestamp_zero_steps": np.nan,
        "timestamp_negative_steps": np.nan,
        "timestamp_median_step_ms": np.nan,
        "timestamp_max_gap_ms": np.nan,
        "timestamp_gaps_gt_12ms": np.nan,
        "timestamp_gaps_gt_20ms": np.nan,
        "timestamp_gaps_gt_50ms": np.nan,
        "timestamp_gaps_gt_100ms": np.nan,

        # Sample Index
        "sample_index_present": False,
        "sample_index_unique": np.nan,
        "sample_index_duplicate_rows": np.nan,
        "sample_index_zero_steps": np.nan,
        "sample_index_negative_steps": np.nan,
        "sample_index_positive_steps": np.nan,
        "sample_index_minus254_wraps": np.nan,
        "sample_index_other_negative": np.nan,
        "sample_index_max_positive_step": np.nan,
        "sample_index_min_step": np.nan,

        # Signal integrity
        "signal_nan": np.nan,
        "signal_inf": np.nan,
        "constant_signal_channels": np.nan,
        "all_signal_nonconstant": False,

        # Local anomaly information
        "major_timestamp_anomaly": False,
        "index_anomaly": False,

        "max_timestamp_gap_row": np.nan,
        "max_timestamp_gap_start_row": np.nan,
        "max_timestamp_gap_end_row": np.nan,

        "classification": "UNCLASSIFIED",
        "notes": "",
    }

    try:
        result["sha256"] = sha256_file(path)

        df = pd.read_csv(path)

        result["readable"] = True
        result["rows"] = len(df)
        result["columns"] = len(df.columns)

    except Exception as e:
        result["classification"] = "UNREADABLE"
        result["notes"] = f"CSV_READ_ERROR: {type(e).__name__}: {e}"
        return result

    # --------------------------------------------------------
    # CHANNEL IDENTIFICATION
    # --------------------------------------------------------

    emg_cols, eeg_cols = identify_signal_columns(df)

    result["emg_channels"] = len(emg_cols)
    result["eeg_channels"] = len(eeg_cols)
    result["signal_channels"] = len(emg_cols) + len(eeg_cols)

    signal_cols = emg_cols + eeg_cols

    if signal_cols:
        signal = df[signal_cols].apply(
            pd.to_numeric,
            errors="coerce"
        )

        result["signal_nan"] = int(signal.isna().sum().sum())

        arr = signal.to_numpy(dtype=float)

        result["signal_inf"] = int(
            np.isinf(arr).sum()
        )

        stds = np.nanstd(arr, axis=0)

        result["constant_signal_channels"] = int(
            np.sum(stds <= FLAT_STD_THRESHOLD)
        )

        result["all_signal_nonconstant"] = bool(
            np.all(stds > FLAT_STD_THRESHOLD)
        )

    # --------------------------------------------------------
    # SAMPLE INDEX
    # --------------------------------------------------------

    if "Sample Index" in df.columns:

        result["sample_index_present"] = True

        idx = pd.to_numeric(
            df["Sample Index"],
            errors="coerce"
        )

        valid_idx = idx.notna()

        if valid_idx.any():

            idx_valid = idx[valid_idx]

            result["sample_index_unique"] = int(
                idx_valid.nunique()
            )

            result["sample_index_duplicate_rows"] = int(
                len(idx_valid) - idx_valid.nunique()
            )

            d_idx = idx_valid.diff().dropna()

            result["sample_index_zero_steps"] = int(
                (d_idx == 0).sum()
            )

            result["sample_index_negative_steps"] = int(
                (d_idx < 0).sum()
            )

            result["sample_index_positive_steps"] = int(
                (d_idx > 0).sum()
            )

            result["sample_index_minus254_wraps"] = int(
                (d_idx == -254).sum()
            )

            result["sample_index_other_negative"] = int(
                ((d_idx < 0) & (d_idx != -254)).sum()
            )

            result["sample_index_max_positive_step"] = float(
                d_idx.max()
            )

            result["sample_index_min_step"] = float(
                d_idx.min()
            )

            result["index_anomaly"] = bool(
                result["sample_index_zero_steps"] > 0
                or result["sample_index_negative_steps"] > 0
            )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    ts, valid_mask = parse_timestamp_series(df)

    if ts is not None:

        result["timestamp_valid"] = bool(
            valid_mask.all()
        )

        if valid_mask.any():

            ts_valid = ts[valid_mask]

            result["timestamp_unique"] = int(
                ts_valid.nunique()
            )

            result["timestamp_duplicate_rows"] = int(
                len(ts_valid) - ts_valid.nunique()
            )

            d_ts = (
                ts_valid
                .diff()
                .dt.total_seconds()
                .mul(1000.0)
                .dropna()
            )

            if len(d_ts):

                result["timestamp_zero_steps"] = int(
                    (d_ts == 0).sum()
                )

                result["timestamp_negative_steps"] = int(
                    (d_ts < 0).sum()
                )

                positive = d_ts[d_ts > 0]

                if len(positive):
                    result["timestamp_median_step_ms"] = float(
                        positive.median()
                    )

                    result["timestamp_max_gap_ms"] = float(
                        positive.max()
                    )

                    result["timestamp_gaps_gt_12ms"] = int(
                        (positive > TIMESTAMP_WARN_MS).sum()
                    )

                    result["timestamp_gaps_gt_20ms"] = int(
                        (positive > TIMESTAMP_MODERATE_MS).sum()
                    )

                    result["timestamp_gaps_gt_50ms"] = int(
                        (positive > TIMESTAMP_MAJOR_MS).sum()
                    )

                    result["timestamp_gaps_gt_100ms"] = int(
                        (positive > TIMESTAMP_SEVERE_MS).sum()
                    )

                    max_gap_pos = int(
                        np.argmax(
                            positive.to_numpy()
                        )
                    )

                    # Map back to timestamp-valid row positions.
                    positive_positions = np.flatnonzero(
                        d_ts.to_numpy() > 0
                    )

                    if (
                        max_gap_pos <
                        len(positive_positions)
                    ):
                        gap_end_pos = (
                            positive_positions[max_gap_pos] + 1
                        )

                        gap_start_pos = (
                            positive_positions[max_gap_pos]
                        )

                        valid_indices = np.flatnonzero(
                            valid_mask.to_numpy()
                        )

                        if (
                            gap_start_pos <
                            len(valid_indices)
                            and
                            gap_end_pos <
                            len(valid_indices)
                        ):
                            start_row = int(
                                valid_indices[gap_start_pos]
                            )

                            end_row = int(
                                valid_indices[gap_end_pos]
                            )

                            result[
                                "max_timestamp_gap_start_row"
                            ] = start_row

                            result[
                                "max_timestamp_gap_end_row"
                            ] = end_row

                            result[
                                "max_timestamp_gap_row"
                            ] = end_row

            result["major_timestamp_anomaly"] = bool(
                (
                    np.isfinite(
                        result["timestamp_max_gap_ms"]
                    )
                )
                and
                result["timestamp_max_gap_ms"]
                > TIMESTAMP_MAJOR_MS
            )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    timestamp_bad = (
        result["major_timestamp_anomaly"]
        or result["timestamp_negative_steps"] > 0
    )

    signal_bad = (
        result["signal_nan"] > 0
        or result["signal_inf"] > 0
        or result["constant_signal_channels"] > 0
    )

    index_bad = result["index_anomaly"]

    minus254_only = (
        result["sample_index_minus254_wraps"] > 0
        and
        result["sample_index_other_negative"] == 0
    )

    # The classification is intentionally conservative.
    if signal_bad:
        classification = "SIGNAL_INTEGRITY_ISSUE"

    elif timestamp_bad and index_bad and minus254_only:
        classification = (
            "TIMESTAMP_INDEX_QC_ISSUE_COUNTER_WRAP"
        )

    elif timestamp_bad and index_bad:
        classification = (
            "TIMESTAMP_AND_INDEX_IRREGULARITY_SIGNAL_INTACT"
        )

    elif timestamp_bad:
        classification = "TIMESTAMP_QC_ISSUE_SIGNAL_INTACT"

    elif index_bad:
        classification = "INDEX_QC_ISSUE_SIGNAL_INTACT"

    else:
        classification = "NO_MAJOR_ANOMALY"

    result["classification"] = classification

    # --------------------------------------------------------
    # NOTES
    # --------------------------------------------------------

    notes = []

    if result["rows"] != EXPECTED_ROWS:
        notes.append(
            f"ROW_COUNT={result['rows']}"
        )

    if result["signal_channels"] != EXPECTED_SIGNAL_CHANNELS:
        notes.append(
            f"SIGNAL_CHANNEL_COUNT={result['signal_channels']}"
        )

    if result["sample_index_minus254_wraps"] > 0:
        notes.append(
            f"MINUS254_WRAPS="
            f"{result['sample_index_minus254_wraps']}"
        )

    if result["sample_index_other_negative"] > 0:
        notes.append(
            f"OTHER_NEGATIVE_INDEX="
            f"{result['sample_index_other_negative']}"
        )

    if result["timestamp_gaps_gt_50ms"] > 0:
        notes.append(
            f"TIMESTAMP_GT50MS="
            f"{result['timestamp_gaps_gt_50ms']}"
        )

    if result["timestamp_gaps_gt_100ms"] > 0:
        notes.append(
            f"TIMESTAMP_GT100MS="
            f"{result['timestamp_gaps_gt_100ms']}"
        )

    if result["constant_signal_channels"] > 0:
        notes.append(
            f"CONSTANT_CHANNELS="
            f"{result['constant_signal_channels']}"
        )

    result["notes"] = "; ".join(notes)

    return result


# ============================================================
# DETAILED LOCAL INSPECTION
# ============================================================

def inspect_anomaly_context(path: Path, row: int):
    """
    Print local context around a timestamp anomaly.

    This is for human inspection only.
    """

    try:
        df = pd.read_csv(path)

    except Exception as e:
        print(
            f"Could not reopen {path}: {e}"
        )
        return

    start = max(0, row - LOCAL_CONTEXT)
    end = min(len(df), row + LOCAL_CONTEXT + 1)

    print("\n" + "=" * 90)
    print(
        f"LOCAL CONTEXT: {path.name} | "
        f"target row={row}"
    )
    print("=" * 90)

    cols = []

    if "Sample Index" in df.columns:
        cols.append("Sample Index")

    if "Timestamp (Formatted)" in df.columns:
        cols.append("Timestamp (Formatted)")
    elif "Timestamp" in df.columns:
        cols.append("Timestamp")

    print(
        df.loc[start:end - 1, cols].to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("EXECUTION TIMESTAMP / SAMPLE INDEX / SIGNAL INSPECTION")
    print("=" * 90)

    print(f"RAW ROOT : {RAW_ROOT}")
    print(f"QC ROOT  : {QC_ROOT}")
    print()

    if not RAW_ROOT.exists():
        print(
            f"ERROR: RAW ROOT DOES NOT EXIST:\n{RAW_ROOT}"
        )
        sys.exit(1)

    QC_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_files = sorted(
        RAW_ROOT.rglob("*.csv")
    )

    print(
        f"CSV files discovered: {len(csv_files)}"
    )

    if not csv_files:
        print("ERROR: No CSV files found.")
        sys.exit(1)

    results = []

    for i, path in enumerate(csv_files, start=1):

        print(
            f"[{i:03d}/{len(csv_files):03d}] "
            f"{path.relative_to(RAW_ROOT)}",
            end="",
            flush=True
        )

        r = inspect_file(path)

        results.append(r)

        print(
            f" -> {r['classification']}"
        )

    report = pd.DataFrame(results)

    report.to_csv(
        REPORT_CSV,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        f"Total CSV files: {len(report)}"
    )

    print(
        f"Readable: "
        f"{int(report['readable'].sum())}"
    )

    print("\nClassification:")
    print(
        report["classification"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nTimestamp gap categories:")

    print(
        f"> {TIMESTAMP_WARN_MS:.0f} ms : "
        f"{int((report['timestamp_max_gap_ms'] > TIMESTAMP_WARN_MS).sum())}"
    )

    print(
        f"> {TIMESTAMP_MODERATE_MS:.0f} ms : "
        f"{int((report['timestamp_max_gap_ms'] > TIMESTAMP_MODERATE_MS).sum())}"
    )

    print(
        f"> {TIMESTAMP_MAJOR_MS:.0f} ms : "
        f"{int((report['timestamp_max_gap_ms'] > TIMESTAMP_MAJOR_MS).sum())}"
    )

    print(
        f"> {TIMESTAMP_SEVERE_MS:.0f} ms : "
        f"{int((report['timestamp_max_gap_ms'] > TIMESTAMP_SEVERE_MS).sum())}"
    )

    print("\nSample Index:")
    print(
        "Files with -254 counter wraps: "
        f"{int((report['sample_index_minus254_wraps'] > 0).sum())}"
    )

    print(
        "Files with other negative transitions: "
        f"{int((report['sample_index_other_negative'] > 0).sum())}"
    )

    print(
        "Files with zero index steps: "
        f"{int((report['sample_index_zero_steps'] > 0).sum())}"
    )

    print("\nSignal:")
    print(
        "Files with signal NaN: "
        f"{int((report['signal_nan'] > 0).sum())}"
    )

    print(
        "Files with signal Inf: "
        f"{int((report['signal_inf'] > 0).sum())}"
    )

    print(
        "Files with constant signal channel(s): "
        f"{int((report['constant_signal_channels'] > 0).sum())}"
    )

    # ========================================================
    # THE 136+ FILES
    # ========================================================

    major = report[
        report["timestamp_max_gap_ms"]
        > TIMESTAMP_MAJOR_MS
    ].copy()

    major = major.sort_values(
        "timestamp_max_gap_ms",
        ascending=False
    )

    print("\n" + "=" * 90)
    print(
        f"FILES WITH MAX TIMESTAMP GAP > "
        f"{TIMESTAMP_MAJOR_MS:.0f} ms"
    )
    print("=" * 90)

    print(
        f"Count: {len(major)}"
    )

    display_cols = [
        "file",
        "timestamp_max_gap_ms",
        "timestamp_gaps_gt_50ms",
        "timestamp_gaps_gt_100ms",
        "sample_index_minus254_wraps",
        "sample_index_other_negative",
        "sample_index_zero_steps",
        "constant_signal_channels",
        "classification",
    ]

    print(
        major[display_cols]
        .to_string(index=False)
    )

    # ========================================================
    # SEVERE FILES
    # ========================================================

    severe = report[
        report["timestamp_max_gap_ms"]
        > TIMESTAMP_SEVERE_MS
    ].copy()

    severe = severe.sort_values(
        "timestamp_max_gap_ms",
        ascending=False
    )

    severe_path = (
        QC_ROOT /
        "execution_severe_timestamp_files.csv"
    )

    severe.to_csv(
        severe_path,
        index=False
    )

    major_path = (
        QC_ROOT /
        "execution_major_timestamp_files.csv"
    )

    major.to_csv(
        major_path,
        index=False
    )

    # ========================================================
    # CLASSIFICATION COUNTS
    # ========================================================

    class_counts = (
        report["classification"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "total_csv_files": int(len(report)),
        "readable_files": int(report["readable"].sum()),

        "timestamp_gap_gt_12ms_files": int(
            (report["timestamp_max_gap_ms"] > 12).sum()
        ),

        "timestamp_gap_gt_20ms_files": int(
            (report["timestamp_max_gap_ms"] > 20).sum()
        ),

        "timestamp_gap_gt_50ms_files": int(
            (report["timestamp_max_gap_ms"] > 50).sum()
        ),

        "timestamp_gap_gt_100ms_files": int(
            (report["timestamp_max_gap_ms"] > 100).sum()
        ),

        "files_with_minus254_wrap": int(
            (report["sample_index_minus254_wraps"] > 0).sum()
        ),

        "files_with_other_negative_index": int(
            (report["sample_index_other_negative"] > 0).sum()
        ),

        "files_with_zero_index_steps": int(
            (report["sample_index_zero_steps"] > 0).sum()
        ),

        "files_with_signal_nan": int(
            (report["signal_nan"] > 0).sum()
        ),

        "files_with_signal_inf": int(
            (report["signal_inf"] > 0).sum()
        ),

        "files_with_constant_signal_channel": int(
            (report["constant_signal_channels"] > 0).sum()
        ),

        "classification_counts": class_counts,

        "raw_data_modified": False,

        "inspection_status": "COMPLETE",
    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    # ========================================================
    # TOP 20
    # ========================================================

    print("\n" + "=" * 90)
    print("TOP 20 LARGEST TIMESTAMP GAPS")
    print("=" * 90)

    top20 = report.nlargest(
        20,
        "timestamp_max_gap_ms"
    )

    print(
        top20[
            [
                "file",
                "timestamp_max_gap_ms",
                "timestamp_gaps_gt_50ms",
                "timestamp_gaps_gt_100ms",
                "sample_index_minus254_wraps",
                "sample_index_other_negative",
                "classification",
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    print(
        f"Full report:\n{REPORT_CSV}"
    )

    print(
        f"\nMajor (>50 ms):\n{major_path}"
    )

    print(
        f"\nSevere (>100 ms):\n{severe_path}"
    )

    print(
        f"\nSummary:\n{SUMMARY_JSON}"
    )

    print("\nRAW DATA MODIFICATION: NONE")
    print("INSPECTION STATUS: COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()