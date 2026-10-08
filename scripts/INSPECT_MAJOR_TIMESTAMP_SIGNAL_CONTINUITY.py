#!/usr/bin/env python3
"""
INSPECT_MAJOR_TIMESTAMP_SIGNAL_CONTINUITY.py

Purpose
-------
Targeted inspection of actual EEG/EMG signal continuity around the largest
timestamp gaps in Motor Execution recordings.

This script is READ-ONLY.

It does NOT:
    - modify raw CSV files
    - repair timestamps
    - repair Sample Index
    - delete rows
    - interpolate signals
    - rewrite recordings

Scientific question
-------------------
When a recording contains a large timestamp gap, are the retained physiological
signal samples numerically valid and changing normally around that gap, while the
raw timestamps are preserved and the analytical time base remains sample order at
nominal 125 Hz?

The script:
    1. Reads the previously generated timestamp inspection report from
       INSPECT_EXECUTION_TIMESTAMP_INDEX_ISSUES.py.
    2. Selects the worst affected recordings.
    3. Finds the largest positive timestamp gap in each file.
    4. Prints ±20 rows around that gap.
    5. Shows Sample Index and Timestamp behavior.
    6. Reports EEG/EMG signal statistics before and after the gap.
    7. Checks NaN/Inf/constant behavior.
    8. Computes signal difference statistics.
    9. Writes a compact CSV summary.

Expected raw structure:
    01_RAW_DATA/EMG_EEG_SYNCHRONIZED/

Expected recording:
    5625 rows
    3 EMG channels
    13 EEG channels
"""


from pathlib import Path
import sys

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    r"F:\Faruk\OFS_Paper_Work\Data_Set_Paper_Work\EEG_EMG_BIOZ_DATASET"
)

RAW_ROOT = ROOT / "01_RAW_DATA" / "EMG_EEG_SYNCHRONIZED"

INSPECTION_ROOT = (
    ROOT
    / "05_QC"
    / "TIMESTAMP_INDEX_SIGNAL_INSPECTION"
)

INPUT_REPORT = (
    INSPECTION_ROOT
    / "execution_timestamp_index_signal_inspection.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "05_QC"
    / "TIMESTAMP_INDEX_SIGNAL_CONTINUITY"
)

OUTPUT_SUMMARY = (
    OUTPUT_ROOT
    / "major_timestamp_signal_continuity_summary.csv"
)

OUTPUT_DETAILS = (
    OUTPUT_ROOT
    / "major_timestamp_gap_details.txt"
)

# Number of worst files to inspect automatically
TOP_N_FILES = 10

# Context around largest gap
CONTEXT_ROWS = 20

# Threshold used to select major timestamp-gap recordings
MAJOR_GAP_MS = 50.0

# Number of samples used for local signal comparison
SIGNAL_CONTEXT = 20

# Numerical threshold for constant signal detection
FLAT_STD_THRESHOLD = 1e-12

# Frozen acquisition / retention specification
NOMINAL_SAMPLING_RATE_HZ = 125.0
EXPECTED_ROWS = 5625
NOMINAL_DURATION_SECONDS = EXPECTED_ROWS / NOMINAL_SAMPLING_RATE_HZ

# Raw timestamps are preserved acquisition/export metadata.
# Timestamp irregularities are diagnosed but are NOT used to redefine
# the sampling frequency or analytical duration.
# Downstream analysis uses sample order + nominal 125 Hz.


# ============================================================
# LOGGING
# ============================================================

def log(message=""):
    print(message, flush=True)


# ============================================================
# CHANNEL IDENTIFICATION
# ============================================================

def identify_signal_columns(df):
    emg_cols = [
        c for c in df.columns
        if str(c).strip().upper().startswith("EMG_")
    ]

    eeg_cols = [
        c for c in df.columns
        if str(c).strip().upper().startswith("EEG_")
    ]

    return emg_cols, eeg_cols


# ============================================================
# TIMESTAMP COLUMN
# ============================================================

def find_timestamp_column(df):

    candidates = [
        "Timestamp (Formatted)",
        "Timestamp",
        "timestamp",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


# ============================================================
# SAMPLE INDEX DIAGNOSTICS
# ============================================================

def sample_index_local_stats(df, gap_row):

    if "Sample Index" not in df.columns:
        return {
            "index_before": np.nan,
            "index_after": np.nan,
            "index_step_at_gap": np.nan,
        }

    idx = pd.to_numeric(
        df["Sample Index"],
        errors="coerce"
    )

    if gap_row <= 0 or gap_row >= len(df):
        return {
            "index_before": np.nan,
            "index_after": np.nan,
            "index_step_at_gap": np.nan,
        }

    before = idx.iloc[gap_row - 1]
    after = idx.iloc[gap_row]

    return {
        "index_before": before,
        "index_after": after,
        "index_step_at_gap": after - before,
    }


# ============================================================
# SIGNAL STATISTICS
# ============================================================

def signal_statistics(df, signal_cols, start, end, prefix):

    section = df.iloc[start:end][signal_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    arr = section.to_numpy(dtype=float)

    if arr.size == 0:
        return {}

    finite = np.isfinite(arr)

    result = {
        f"{prefix}_rows": int(len(section)),
        f"{prefix}_nan": int(np.isnan(arr).sum()),
        f"{prefix}_inf": int(np.isinf(arr).sum()),
    }

    stds = np.nanstd(arr, axis=0)

    result[
        f"{prefix}_constant_channels"
    ] = int(
        np.sum(stds <= FLAT_STD_THRESHOLD)
    )

    result[
        f"{prefix}_mean_abs"
    ] = float(
        np.nanmean(np.abs(arr))
    )

    result[
        f"{prefix}_median_abs"
    ] = float(
        np.nanmedian(np.abs(arr))
    )

    result[
        f"{prefix}_mean_std"
    ] = float(
        np.nanmean(stds)
    )

    result[
        f"{prefix}_min"
    ] = float(
        np.nanmin(arr)
    )

    result[
        f"{prefix}_max"
    ] = float(
        np.nanmax(arr)
    )

    result[
        f"{prefix}_finite_fraction"
    ] = float(
        finite.mean()
    )

    return result


# ============================================================
# SIGNAL CHANGE BETWEEN CONSECUTIVE ROWS
# ============================================================

def consecutive_signal_difference(
    df,
    signal_cols,
    start,
    end,
):

    section = df.iloc[start:end][signal_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    arr = section.to_numpy(dtype=float)

    if len(arr) < 2:
        return {
            "mean_abs_delta": np.nan,
            "median_abs_delta": np.nan,
            "max_abs_delta": np.nan,
            "zero_delta_fraction": np.nan,
        }

    delta = np.diff(arr, axis=0)

    abs_delta = np.abs(delta)

    finite = np.isfinite(abs_delta)

    if not finite.any():
        return {
            "mean_abs_delta": np.nan,
            "median_abs_delta": np.nan,
            "max_abs_delta": np.nan,
            "zero_delta_fraction": np.nan,
        }

    values = abs_delta[finite]

    return {
        "mean_abs_delta": float(np.mean(values)),
        "median_abs_delta": float(np.median(values)),
        "max_abs_delta": float(np.max(values)),
        "zero_delta_fraction": float(
            np.mean(values == 0)
        ),
    }


# ============================================================
# FIND LARGEST TIMESTAMP GAP
# ============================================================

def find_largest_timestamp_gap(df):

    timestamp_col = find_timestamp_column(df)

    if timestamp_col is None:
        raise ValueError(
            "Timestamp column not found."
        )

    ts = pd.to_datetime(
        df[timestamp_col],
        errors="coerce"
    )

    if ts.isna().all():
        raise ValueError(
            "Timestamp column contains no valid timestamps."
        )

    delta_ms = (
        ts.diff()
        .dt.total_seconds()
        .mul(1000.0)
    )

    positive = delta_ms[
        delta_ms > 0
    ]

    if positive.empty:
        return None

    gap_value = float(
        positive.max()
    )

    gap_row = int(
        positive.idxmax()
    )

    return {
        "timestamp_col": timestamp_col,
        "gap_row": gap_row,
        "gap_ms": gap_value,
        "timestamp_before": ts.iloc[gap_row - 1],
        "timestamp_after": ts.iloc[gap_row],
    }


# ============================================================
# PRINT LOCAL CONTEXT
# ============================================================

def print_local_context(
    df,
    signal_cols,
    gap,
    file_label,
):

    gap_row = gap["gap_row"]

    start = max(
        0,
        gap_row - CONTEXT_ROWS
    )

    end = min(
        len(df),
        gap_row + CONTEXT_ROWS
    )

    display_cols = []

    if "Sample Index" in df.columns:
        display_cols.append(
            "Sample Index"
        )

    display_cols.append(
        gap["timestamp_col"]
    )

    display_cols.extend(
        signal_cols
    )

    log("")
    log("=" * 110)
    log(
        f"LOCAL SIGNAL CONTEXT: {file_label}"
    )
    log("=" * 110)

    log(
        f"Largest timestamp gap : "
        f"{gap['gap_ms']:.3f} ms"
    )

    log(
        f"Gap between rows       : "
        f"{gap_row - 1} -> {gap_row}"
    )

    log(
        f"Timestamp before       : "
        f"{gap['timestamp_before']}"
    )

    log(
        f"Timestamp after        : "
        f"{gap['timestamp_after']}"
    )

    idx_stats = sample_index_local_stats(
        df,
        gap_row
    )

    log(
        f"Sample Index before    : "
        f"{idx_stats['index_before']}"
    )

    log(
        f"Sample Index after     : "
        f"{idx_stats['index_after']}"
    )

    log(
        f"Sample Index step      : "
        f"{idx_stats['index_step_at_gap']}"
    )

    log("")
    log(
        f"Showing rows "
        f"{start} through {end - 1}"
    )

    # Print metadata first
    metadata_cols = [
        c for c in display_cols
        if c not in signal_cols
    ]

    log("")
    log(
        df.iloc[start:end][metadata_cols]
        .to_string()
    )

    # Print actual signal values separately
    log("")
    log(
        "ACTUAL EEG/EMG SIGNAL VALUES"
    )
    log("-" * 110)

    # To prevent an extremely wide terminal dump,
    # print one channel at a time.
    for channel in signal_cols:

        log("")
        log(f"CHANNEL: {channel}")

        temp = pd.DataFrame({
            "row": np.arange(
                start,
                end
            ),
            "value": pd.to_numeric(
                df.iloc[start:end][channel],
                errors="coerce"
            ).to_numpy()
        })

        log(
            temp.to_string(
                index=False
            )
        )


# ============================================================
# INSPECT ONE FILE
# ============================================================

def inspect_one_file(
    relative_file,
    file_number,
):

    path = RAW_ROOT / relative_file

    log("")
    log("")
    log("#" * 110)
    log(
        f"[{file_number}] {relative_file}"
    )
    log("#" * 110)

    if not path.exists():

        log(
            f"ERROR: File does not exist:\n{path}"
        )

        return {
            "file": relative_file,
            "status": "MISSING_FILE",
        }

    try:
        df = pd.read_csv(path)

    except Exception as e:

        log(
            f"ERROR reading file: {e}"
        )

        return {
            "file": relative_file,
            "status": "READ_ERROR",
        }

    emg_cols, eeg_cols = identify_signal_columns(df)

    signal_cols = emg_cols + eeg_cols

    log(
        f"Rows: {len(df)}"
    )

    log(
        f"Columns: {len(df.columns)}"
    )

    log(
        f"EMG channels: {len(emg_cols)}"
    )

    log(
        f"EEG channels: {len(eeg_cols)}"
    )

    # --------------------------------------------------------
    # FROZEN RETENTION / ANALYTICAL TIME BASE
    # --------------------------------------------------------

    nominal_duration = len(df) / NOMINAL_SAMPLING_RATE_HZ

    log(f"Nominal sampling rate : {NOMINAL_SAMPLING_RATE_HZ:.1f} Hz")
    log(f"Retained rows          : {len(df)}")
    log(f"Nominal duration       : {nominal_duration:.3f} s")
    log("Analytical time base   : sample order + nominal 125 Hz")
    log("Raw timestamps         : preserved acquisition/export metadata")

    if len(df) != EXPECTED_ROWS:
        log(f"WARNING: Expected {EXPECTED_ROWS} rows, found {len(df)}.")

    # --------------------------------------------------------
    # BASIC SIGNAL INTEGRITY
    # --------------------------------------------------------

    if signal_cols:

        signal = df[signal_cols].apply(
            pd.to_numeric,
            errors="coerce"
        )

        arr = signal.to_numpy(
            dtype=float
        )

        nan_count = int(
            np.isnan(arr).sum()
        )

        inf_count = int(
            np.isinf(arr).sum()
        )

        stds = np.nanstd(
            arr,
            axis=0
        )

        constant_channels = int(
            np.sum(
                stds <= FLAT_STD_THRESHOLD
            )
        )

    else:

        nan_count = np.nan
        inf_count = np.nan
        constant_channels = np.nan

    # --------------------------------------------------------
    # FIND LARGEST TIMESTAMP GAP
    # --------------------------------------------------------

    try:

        gap = find_largest_timestamp_gap(
            df
        )

    except Exception as e:

        log(
            f"Timestamp analysis error: {e}"
        )

        return {
            "file": relative_file,
            "status": "TIMESTAMP_ERROR",
        }

    if gap is None:

        log(
            "No positive timestamp gap found."
        )

        return {
            "file": relative_file,
            "status": "NO_POSITIVE_TIMESTAMP_GAP",
        }

    gap_row = gap["gap_row"]

    # --------------------------------------------------------
    # LOCAL WINDOW
    # --------------------------------------------------------

    before_start = max(
        0,
        gap_row - SIGNAL_CONTEXT
    )

    before_end = gap_row

    after_start = gap_row

    after_end = min(
        len(df),
        gap_row + SIGNAL_CONTEXT
    )

    before_stats = signal_statistics(
        df,
        signal_cols,
        before_start,
        before_end,
        "before"
    )

    after_stats = signal_statistics(
        df,
        signal_cols,
        after_start,
        after_end,
        "after"
    )

    before_delta = consecutive_signal_difference(
        df,
        signal_cols,
        before_start,
        before_end
    )

    after_delta = consecutive_signal_difference(
        df,
        signal_cols,
        after_start,
        after_end
    )

    # --------------------------------------------------------
    # PRINT CONTEXT
    # --------------------------------------------------------

    print_local_context(
        df,
        signal_cols,
        gap,
        relative_file
    )

    # --------------------------------------------------------
    # PRINT LOCAL SIGNAL STATISTICS
    # --------------------------------------------------------

    log("")
    log(
        "LOCAL SIGNAL STATISTICS"
    )
    log("-" * 110)

    log(
        "Before-gap window:"
    )

    for k, v in before_stats.items():
        log(
            f"  {k}: {v}"
        )

    log("")
    log(
        "After-gap window:"
    )

    for k, v in after_stats.items():
        log(
            f"  {k}: {v}"
        )

    log("")
    log(
        "Consecutive signal differences BEFORE gap:"
    )

    for k, v in before_delta.items():
        log(
            f"  {k}: {v}"
        )

    log("")
    log(
        "Consecutive signal differences AFTER gap:"
    )

    for k, v in after_delta.items():
        log(
            f"  {k}: {v}"
        )

    # --------------------------------------------------------
    # SAMPLE INDEX LOCAL CONTEXT
    # --------------------------------------------------------

    idx_stats = sample_index_local_stats(
        df,
        gap_row
    )

    # --------------------------------------------------------
    # LOCAL CLASSIFICATION
    # --------------------------------------------------------

    local_signal_nan = (
        before_stats.get("before_nan", 0)
        +
        after_stats.get("after_nan", 0)
    )

    local_signal_inf = (
        before_stats.get("before_inf", 0)
        +
        after_stats.get("after_inf", 0)
    )

    local_constant_channels = (
        max(
            before_stats.get(
                "before_constant_channels",
                0
            ),
            after_stats.get(
                "after_constant_channels",
                0
            ),
        )
    )

    if (
        local_signal_nan > 0
        or local_signal_inf > 0
        or local_constant_channels > 0
    ):

        local_classification = (
            "LOCAL_SIGNAL_INTEGRITY_WARNING"
        )

    else:

        local_classification = (
            "LOCAL_SIGNAL_NUMERICALLY_INTACT"
        )

    log("")
    log(
        f"LOCAL CLASSIFICATION: "
        f"{local_classification}"
    )

    return {
        "file": relative_file,
        "status": "INSPECTED",

        "rows": len(df),
        "columns": len(df.columns),
        "nominal_sampling_rate_hz": NOMINAL_SAMPLING_RATE_HZ,
        "nominal_duration_seconds": nominal_duration,
        "expected_rows": EXPECTED_ROWS,
        "row_count_matches_expected": bool(len(df) == EXPECTED_ROWS),
        "analytical_time_base": "sample_order_nominal_125_Hz",

        "emg_channels": len(emg_cols),
        "eeg_channels": len(eeg_cols),

        "global_signal_nan": nan_count,
        "global_signal_inf": inf_count,
        "global_constant_channels": constant_channels,

        "largest_timestamp_gap_ms":
            gap["gap_ms"],

        "largest_gap_row":
            gap_row,

        "timestamp_before":
            str(gap["timestamp_before"]),

        "timestamp_after":
            str(gap["timestamp_after"]),

        "sample_index_before":
            idx_stats["index_before"],

        "sample_index_after":
            idx_stats["index_after"],

        "sample_index_step_at_gap":
            idx_stats["index_step_at_gap"],

        "local_before_nan":
            before_stats.get(
                "before_nan",
                np.nan
            ),

        "local_after_nan":
            after_stats.get(
                "after_nan",
                np.nan
            ),

        "local_before_inf":
            before_stats.get(
                "before_inf",
                np.nan
            ),

        "local_after_inf":
            after_stats.get(
                "after_inf",
                np.nan
            ),

        "local_before_constant_channels":
            before_stats.get(
                "before_constant_channels",
                np.nan
            ),

        "local_after_constant_channels":
            after_stats.get(
                "after_constant_channels",
                np.nan
            ),

        "before_mean_abs_delta":
            before_delta["mean_abs_delta"],

        "after_mean_abs_delta":
            after_delta["mean_abs_delta"],

        "before_median_abs_delta":
            before_delta["median_abs_delta"],

        "after_median_abs_delta":
            after_delta["median_abs_delta"],

        "before_zero_delta_fraction":
            before_delta["zero_delta_fraction"],

        "after_zero_delta_fraction":
            after_delta["zero_delta_fraction"],

        "local_classification":
            local_classification,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 110)
    log(
        "MAJOR TIMESTAMP GAP — SIGNAL CONTINUITY INSPECTION"
    )
    log("=" * 110)

    log(
        f"RAW ROOT:\n{RAW_ROOT}"
    )

    log(
        f"INPUT REPORT:\n{INPUT_REPORT}"
    )

    log(
        f"OUTPUT ROOT:\n{OUTPUT_ROOT}"
    )

    log("")

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not RAW_ROOT.exists():

        log(
            f"ERROR: Raw root does not exist:\n{RAW_ROOT}"
        )

        sys.exit(1)

    if not INPUT_REPORT.exists():

        log(
            f"ERROR: Inspection report does not exist:\n"
            f"{INPUT_REPORT}"
        )

        log("")
        log(
            "Run INSPECT_EXECUTION_TIMESTAMP_INDEX_ISSUES.py first."
        )

        sys.exit(1)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load previous inspection report
    # --------------------------------------------------------

    report = pd.read_csv(
        INPUT_REPORT
    )

    log(
        f"Inspection report rows: {len(report)}"
    )

    required_report_columns = {
        "file",
        "timestamp_max_gap_ms",
        "timestamp_gaps_gt_50ms",
        "timestamp_gaps_gt_100ms",
        "sample_index_minus254_wraps",
        "sample_index_other_negative",
        "classification",
    }

    missing_report_columns = sorted(
        required_report_columns.difference(report.columns)
    )

    if missing_report_columns:
        log(
            "ERROR: Required columns missing from inspection report: "
            + ", ".join(missing_report_columns)
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Select major timestamp files
    # --------------------------------------------------------

    major = report[
        pd.to_numeric(
            report["timestamp_max_gap_ms"],
            errors="coerce"
        )
        > MAJOR_GAP_MS
    ].copy()

    major = major.sort_values(
        "timestamp_max_gap_ms",
        ascending=False
    )

    selected = major.head(
        TOP_N_FILES
    )

    log("")
    log(
        f"Files with >{MAJOR_GAP_MS:.0f} ms maximum gap: "
        f"{len(major)}"
    )

    log(
        f"Files selected for detailed inspection: "
        f"{len(selected)}"
    )

    log("")
    log(
        "SELECTED FILES"
    )
    log("-" * 110)

    selected_display_cols = [
        "file",
        "timestamp_max_gap_ms",
        "timestamp_gaps_gt_50ms",
        "timestamp_gaps_gt_100ms",
        "sample_index_minus254_wraps",
        "sample_index_other_negative",
        "classification",
    ]
    selected_display_cols = [
        c for c in selected_display_cols if c in selected.columns
    ]

    log(
        selected[selected_display_cols].to_string(index=False)
    )

    # --------------------------------------------------------
    # Inspect selected files
    # --------------------------------------------------------

    results = []

    with OUTPUT_DETAILS.open(
        "w",
        encoding="utf-8"
    ) as detail_file:

        detail_file.write(
            "MAJOR TIMESTAMP GAP — SIGNAL CONTINUITY INSPECTION\n"
        )

        detail_file.write(
            "=" * 110 + "\n"
        )

        for n, (_, report_row) in enumerate(
            selected.iterrows(),
            start=1
        ):

            relative_file = str(
                report_row["file"]
            )

            result = inspect_one_file(
                relative_file,
                n
            )

            results.append(
                result
            )

            # ------------------------------------------------
            # Compact result saved to detail text file
            # ------------------------------------------------

            detail_file.write(
                "\n"
            )

            detail_file.write(
                "=" * 110 + "\n"
            )

            detail_file.write(
                f"FILE: {relative_file}\n"
            )

            for key, value in result.items():

                detail_file.write(
                    f"{key}: {value}\n"
                )

            detail_file.write(
                "=" * 110 + "\n"
            )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        results
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    log("")
    log("")
    log("=" * 110)
    log(
        "FINAL TARGETED INSPECTION SUMMARY"
    )
    log("=" * 110)

    if len(summary):

        log("")
        summary_display_cols = [
            "file",
            "largest_timestamp_gap_ms",
            "sample_index_step_at_gap",
            "global_signal_nan",
            "global_signal_inf",
            "global_constant_channels",
            "local_classification",
        ]
        summary_display_cols = [
            c for c in summary_display_cols if c in summary.columns
        ]

        log(
            summary[summary_display_cols].to_string(index=False)
        )

        log("")
        log(
            "LOCAL CLASSIFICATION COUNTS"
        )

        log(
            summary[
                "local_classification"
            ]
            .value_counts()
            .to_string()
        )

    log("")
    log(
        f"Summary CSV:\n{OUTPUT_SUMMARY}"
    )

    log(
        f"\nDetailed text report:\n{OUTPUT_DETAILS}"
    )

    log("")
    log("RAW DATA MODIFICATION: NONE")
    log("TIMESTAMP HANDLING: PRESERVED; NOT USED TO REDEFINE SAMPLING RATE")
    log("ANALYTICAL TIME BASE: SAMPLE ORDER + NOMINAL 125 HZ")
    log("RETENTION SPECIFICATION: 5625 SAMPLES = 45 NOMINAL SECONDS")
    log("INSPECTION STATUS: COMPLETE")

    log("=" * 110)


if __name__ == "__main__":
    main()
