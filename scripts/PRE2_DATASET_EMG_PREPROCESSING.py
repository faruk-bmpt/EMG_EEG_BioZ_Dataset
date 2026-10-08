#!/usr/bin/env python3
"""
PRE2 — EMG SIGNAL PREPROCESSING & ARTIFACT QC
DATASET PAPER / FINAL DATASET V1.0

Purpose
-------
Create a publication-ready processed EMG representation from the raw
Motor Execution recordings while preserving the raw dataset unchanged.

Input
-----
01_RAW_DATA/EMG_EEG_SYNCHRONIZED/
    Subject_01 ... Subject_40/
    7 gestures × 3 sets
    3 EMG channels
    nominal sampling rate = 125 Hz
    5625 samples per recording

Output
------
06_PROCESSED_DATA/EMG/
    one compressed NPZ + one JSON provenance file per recording

06_PROCESSED_DATA/EMG/PRE2_QC/EMG_MOTOR_EXECUTION/
    recording-level QC/provenance tables and final audit

Frozen processing contract
--------------------------
1. Read raw CSV only; never modify raw data.
2. Use row order as the processing time axis at nominal 125 Hz.
3. Preserve original Sample Index and formatted Timestamp in provenance.
4. EMG band-pass: 10–45 Hz, 4th-order Butterworth, zero-phase offline.
5. Power-line notch: 50 Hz, Q=30, zero-phase offline.
6. No rectification, envelope extraction, normalization, feature fitting,
   class balancing, calibration, or ML-dependent processing.
7. No automatic channel deletion.
8. QC flags are descriptive and do not silently alter the signal.
9. Per-recording resume is enabled and validates output/provenance/hash
   before skipping an existing result.

Scientific note
---------------
At 125 Hz, Nyquist frequency is 62.5 Hz. A 60-Hz low-pass leaves only a
2.5-Hz transition margin and is therefore not used as the canonical EMG
low-pass. The frozen 10–45-Hz band is used for this dataset-paper
preprocessing representation. The raw 3-channel EMG remains available for
users who require a different EMG bandwidth.

This block does not claim that 10–45 Hz represents the complete physiological
EMG spectrum. It is a conservative, reproducible gesture-analysis processing
band compatible with the recorded 125-Hz sampling rate.
"""

from pathlib import Path
from datetime import datetime
import hashlib
import json
import logging
import re
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt


# ============================================================================
# 1. DATASET PATHS / FROZEN CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "01_RAW_DATA").exists():
    ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "01_RAW_DATA").exists():
    ROOT = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        "Cannot locate dataset root. Place this script at dataset root "
        "or dataset_root/scripts/."
    )

RAW_ROOT = ROOT / "01_RAW_DATA"

EXECUTION_ROOT = RAW_ROOT / "EMG_EEG_SYNCHRONIZED"


PROCESSED_ROOT = ROOT / "06_PROCESSED_DATA" / "EMG"

OUTPUT_ROOT = PROCESSED_ROOT

QC_ROOT = PROCESSED_ROOT / "PRE2_QC"
EMG_METADATA = QC_ROOT / "EMG_MOTOR_EXECUTION"

SENTINEL = EMG_METADATA / "_PRE2_COMPLETE.json"

PROTOCOL_VERSION = "PRE2-DATASET-V1.0-FINAL-FREEZE"

EXPECTED_SUBJECTS = tuple(range(1, 41))
EXPECTED_GESTURES = (
    "Hand_Open_Close",
    "Index_Finger_Movement",
    "Little_Finger_Movement",
    "Middle_Finger_Movement",
    "Pen_Holding",
    "Ring_Finger_Movement",
    "Thumb_Finger_Movement",
)
EXPECTED_SETS = ("A", "B", "C")

EXPECTED_RECORDINGS = 40 * 7 * 3
EXPECTED_SAMPLES = 5625
EXPECTED_EMG_CHANNELS = (
    "EMG_ch-01",
    "EMG_ch-02",
    "EMG_ch-03",
)
SAMPLING_RATE_HZ = 125.0

# Frozen EMG preprocessing
BANDPASS_LOW_HZ = 10.0
BANDPASS_HIGH_HZ = 45.0
FILTER_ORDER = 4

NOTCH_FREQUENCY_HZ = 50.0
NOTCH_Q = 30.0

# Descriptive QC thresholds only; they do not trigger deletion.
CONST_STD_EPS = 1e-12
ROBUST_Z_THRESHOLD = 5.0
SATURATION_REPEAT_FRACTION = 0.95


# ============================================================================
# 2. LOGGING / ATOMIC WRITING
# ============================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log(message):
    logging.info(message)


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def atomic_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    log(
        f"WROTE | {path.relative_to(ROOT)} | "
        f"rows={len(df)} cols={len(df.columns)}"
    )


# ============================================================================
# 3. IDENTITY / INVENTORY
# ============================================================================

GESTURE_CODE = {
    "Hand_Open_Close": "HOC",
    "Index_Finger_Movement": "IFM",
    "Little_Finger_Movement": "LFM",
    "Middle_Finger_Movement": "MFM",
    "Pen_Holding": "PH",
    "Ring_Finger_Movement": "RFM",
    "Thumb_Finger_Movement": "TFM",
}


def parse_filename(path):
    """
    Parse an execution filename from either a pathlib.Path or a string.
    This accepts both forms so the function is safe for all callers.
    """
    filename = Path(path).name

    m = re.match(
        r"^S(\d{2})_([A-Z]{2,3})_Set([ABC])\.csv$",
        filename,
        re.I,
    )
    if not m:
        return None
    return int(m.group(1)), m.group(2).upper(), m.group(3).upper()


def discover_inventory():
    files = sorted(EXECUTION_ROOT.rglob("*.csv"))
    expected = set()

    for subject in EXPECTED_SUBJECTS:
        sid = f"Subject_{subject:02d}"
        for gesture in EXPECTED_GESTURES:
            code = GESTURE_CODE[gesture]
            for set_id in EXPECTED_SETS:
                expected.add((sid, gesture, code, set_id))

    actual = set()
    errors = []

    for path in files:
        try:
            rel = path.relative_to(EXECUTION_ROOT)
        except ValueError:
            errors.append(str(path))
            continue

        if len(rel.parts) != 3:
            errors.append(f"{path.relative_to(ROOT)}: invalid hierarchy")
            continue

        subject_folder, gesture_folder, filename = rel.parts
        parsed = parse_filename(path)

        if parsed is None:
            errors.append(f"{path.relative_to(ROOT)}: filename pattern mismatch")
            continue

        subject_num, file_code, set_id = parsed
        expected_subject = f"Subject_{subject_num:02d}"

        if (
            subject_folder != expected_subject
            or gesture_folder not in EXPECTED_GESTURES
            or GESTURE_CODE.get(gesture_folder) != file_code
            or set_id not in EXPECTED_SETS
            or subject_num not in EXPECTED_SUBJECTS
        ):
            errors.append(f"{path.relative_to(ROOT)}: identity mismatch")
            continue

        actual.add(
            (subject_folder, gesture_folder, file_code, set_id)
        )

    expected_count = len(expected)
    actual_count = len(actual)

    return files, {
        "expected_recordings": expected_count,
        "actual_recordings": actual_count,
        "missing_recordings": sorted(expected - actual),
        "extra_recordings": sorted(actual - expected),
        "identity_errors": errors,
        "pass": (
            len(files) == EXPECTED_RECORDINGS
            and actual_count == EXPECTED_RECORDINGS
            and not (expected - actual)
            and not (actual - expected)
            and not errors
        ),
    }


# ============================================================================
# 4. RAW LOADING / SIGNAL VALIDATION
# ============================================================================

def load_raw_emg(path):
    df = pd.read_csv(path)

    required = [
        "Sample Index",
        "Timestamp (Formatted)",
        *EXPECTED_EMG_CHANNELS,
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} rows, found {len(df)}"
        )

    emg = df[list(EXPECTED_EMG_CHANNELS)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(emg).all():
        raise ValueError("EMG contains NaN or Inf")

    return df, emg


# ============================================================================
# 5. FILTERING
# ============================================================================

def design_filters():
    nyquist = SAMPLING_RATE_HZ / 2.0

    if not (0 < BANDPASS_LOW_HZ < BANDPASS_HIGH_HZ < nyquist):
        raise ValueError("Invalid EMG band-pass frequencies.")

    if NOTCH_FREQUENCY_HZ >= nyquist:
        raise ValueError("Notch frequency must be below Nyquist.")

    band_sos = butter(
        FILTER_ORDER,
        [BANDPASS_LOW_HZ / nyquist, BANDPASS_HIGH_HZ / nyquist],
        btype="bandpass",
        output="sos",
    )

    notch_b, notch_a = iirnotch(
        w0=NOTCH_FREQUENCY_HZ / nyquist,
        Q=NOTCH_Q,
    )

    return band_sos, notch_b, notch_a


def preprocess_emg(emg, band_sos, notch_b, notch_a):
    """
    Offline zero-phase preprocessing.

    Notch is applied first, followed by band-pass. Because both are
    zero-phase offline filters, no causal phase delay is introduced.
    """
    x = filtfilt(
        notch_b,
        notch_a,
        emg,
        axis=0,
    )

    y = sosfiltfilt(
        band_sos,
        x,
        axis=0,
    )

    if not np.isfinite(y).all():
        raise FloatingPointError("Filtered EMG contains NaN/Inf")

    return y


# ============================================================================
# 6. DESCRIPTIVE QC
# ============================================================================

def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))

    if mad <= np.finfo(float).eps:
        return np.zeros_like(x)

    return 0.67448975 * (x - med) / mad


def channel_qc(raw, filtered):
    rows = []

    for i, channel in enumerate(EXPECTED_EMG_CHANNELS):
        x = raw[:, i]
        y = filtered[:, i]

        raw_std = float(np.std(x))
        filtered_std = float(np.std(y))

        z = robust_z(x)
        extreme_fraction = float(
            np.mean(np.abs(z) > ROBUST_Z_THRESHOLD)
        )

        vals, counts = np.unique(x, return_counts=True)
        max_repeat_fraction = (
            float(counts.max() / len(x)) if len(x) else np.nan
        )

        rows.append(
            {
                "channel": channel,
                "raw_mean": float(np.mean(x)),
                "raw_std": raw_std,
                "raw_min": float(np.min(x)),
                "raw_max": float(np.max(x)),
                "filtered_mean": float(np.mean(y)),
                "filtered_std": filtered_std,
                "filtered_min": float(np.min(y)),
                "filtered_max": float(np.max(y)),
                "robust_extreme_fraction": extreme_fraction,
                "max_exact_value_repeat_fraction": max_repeat_fraction,
                "constant_channel": bool(raw_std <= CONST_STD_EPS),
                "high_repeat_warning": bool(
                    max_repeat_fraction >= SATURATION_REPEAT_FRACTION
                ),
            }
        )

    return rows


def temporal_qc(df):
    ts = pd.to_datetime(
        df["Timestamp (Formatted)"],
        errors="coerce",
    )

    idx = pd.to_numeric(
        df["Sample Index"],
        errors="coerce",
    )

    result = {
        "timestamp_parse_pass": bool(ts.notna().all()),
        "timestamp_negative_steps": np.nan,
        "timestamp_duplicate_rows": np.nan,
        "timestamp_large_gap_over_50ms": np.nan,
        "timestamp_median_step_sec": np.nan,
        "timestamp_duration_sec": np.nan,
        "sample_index_numeric": bool(idx.notna().all()),
        "sample_index_negative_steps": np.nan,
        "sample_index_duplicate_rows": np.nan,
        "sample_index_median_step": np.nan,
    }

    if result["timestamp_parse_pass"] and len(ts) > 1:
        dts = ts.diff().dt.total_seconds().dropna().to_numpy()
        result["timestamp_negative_steps"] = int(np.sum(dts < 0))
        result["timestamp_duplicate_rows"] = int(ts.duplicated().sum())
        result["timestamp_large_gap_over_50ms"] = int(
            np.sum(dts > 0.050)
        )
        result["timestamp_median_step_sec"] = float(np.median(dts))
        result["timestamp_duration_sec"] = float(
            (ts.iloc[-1] - ts.iloc[0]).total_seconds()
        )

    if result["sample_index_numeric"] and len(idx) > 1:
        di = idx.diff().dropna().to_numpy()
        result["sample_index_negative_steps"] = int(np.sum(di < 0))
        result["sample_index_duplicate_rows"] = int(idx.duplicated().sum())
        result["sample_index_median_step"] = float(np.median(di))

    return result


# ============================================================================
# 7. OUTPUT / RESUME VALIDATION
# ============================================================================

def output_paths(path):
    rel = path.relative_to(EXECUTION_ROOT)
    subject, gesture, filename = rel.parts

    stem = Path(filename).stem
    out_dir = OUTPUT_ROOT / subject / gesture

    return (
        out_dir / f"{stem}_PRE2.npz",
        out_dir / f"{stem}_PRE2.json",
    )


def validate_existing_output(csv_path, npz_path, meta_path, input_hash):
    if not npz_path.exists() or not meta_path.exists():
        return False

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        if metadata.get("protocol_version") != PROTOCOL_VERSION:
            return False

        if metadata.get("input_sha256") != input_hash:
            return False

        if metadata.get("input_file") != str(csv_path.relative_to(ROOT)):
            return False

        with np.load(npz_path, allow_pickle=False) as z:
            if "emg_raw" not in z or "emg_preprocessed" not in z:
                return False

            raw = z["emg_raw"]
            processed = z["emg_preprocessed"]

            if raw.shape != (EXPECTED_SAMPLES, 3):
                return False

            if processed.shape != (EXPECTED_SAMPLES, 3):
                return False

            if not np.isfinite(raw).all():
                return False

            if not np.isfinite(processed).all():
                return False

        return True

    except Exception:
        return False


def process_recording(csv_path, band_sos, notch_b, notch_a):
    npz_path, meta_path = output_paths(csv_path)

    input_hash = sha256_file(csv_path)

    if validate_existing_output(
        csv_path,
        npz_path,
        meta_path,
        input_hash,
    ):
        return {
            "status": "SKIPPED_EXISTING",
            "input_file": str(csv_path.relative_to(ROOT)),
            "npz_file": str(npz_path.relative_to(ROOT)),
            "meta_file": str(meta_path.relative_to(ROOT)),
            "error": "",
        }

    df, raw = load_raw_emg(csv_path)

    filtered = preprocess_emg(
        raw,
        band_sos,
        notch_b,
        notch_a,
    )

    channel_rows = channel_qc(raw, filtered)
    temporal = temporal_qc(df)

    parsed = parse_filename(csv_path)
    subject_num, code, set_id = parsed

    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(csv_path.relative_to(ROOT)),
        "input_sha256": input_hash,
        "subject": f"Subject_{subject_num:02d}",
        "gesture_folder": csv_path.parent.name,
        "gesture_code": code,
        "set": set_id,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "samples": EXPECTED_SAMPLES,
        "channels": list(EXPECTED_EMG_CHANNELS),
        "time_axis_policy": "ROW_ORDER_PLUS_NOMINAL_125_HZ",
        "raw_timestamp_preserved": True,
        "raw_sample_index_preserved": True,
        "filter": {
            "notch_hz": NOTCH_FREQUENCY_HZ,
            "notch_q": NOTCH_Q,
            "bandpass_low_hz": BANDPASS_LOW_HZ,
            "bandpass_high_hz": BANDPASS_HIGH_HZ,
            "order": FILTER_ORDER,
            "phase": "ZERO_PHASE_OFFLINE",
        },
        "operations": [
            "50_HZ_NOTCH",
            "10_45_HZ_BANDPASS",
        ],
        "operations_not_performed": [
            "RECTIFICATION",
            "ENVELOPE_EXTRACTION",
            "NORMALIZATION",
            "BASELINE_CORRECTION",
            "AUTOMATIC_CHANNEL_REMOVAL",
            "FEATURE_EXTRACTION",
            "FEATURE_SELECTION",
            "CLASS_BALANCING",
            "CALIBRATION",
            "MODEL_FITTING",
        ],
        "timestamp_qc": temporal,
        "channel_qc": channel_rows,
        "raw_data_modified": False,
    }

    atomic_npz(
        npz_path,
        emg_raw=raw.astype(np.float64),
        emg_preprocessed=filtered.astype(np.float64),
        sample_index=pd.to_numeric(
            df["Sample Index"],
            errors="coerce",
        ).to_numpy(dtype=np.float64),
        timestamp_formatted=df["Timestamp (Formatted)"].astype(str).to_numpy(),
    )

    atomic_json(metadata, meta_path)

    # Mandatory post-write validation.
    if not validate_existing_output(
        csv_path,
        npz_path,
        meta_path,
        input_hash,
    ):
        raise RuntimeError(
            f"Post-write validation failed: {csv_path}"
        )

    return {
        "status": "PROCESSED",
        "input_file": str(csv_path.relative_to(ROOT)),
        "npz_file": str(npz_path.relative_to(ROOT)),
        "meta_file": str(meta_path.relative_to(ROOT)),
        "error": "",
    }


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    setup_logging()
    t0 = time.time()

    print("=" * 100)
    print("PRE2 — EMG MOTOR EXECUTION SIGNAL PREPROCESSING & ARTIFACT QC")
    print("DATASET PAPER / FINAL DATASET V1.0")
    print("=" * 100)

    log(f"Protocol version: {PROTOCOL_VERSION}")
    log("RAW DATA MODIFICATION: NONE")
    log("Resume mode: ENABLED")
    log("Processing time axis: row order + nominal 125 Hz.")
    log("Raw formatted timestamps and Sample Index are preserved.")
    log(
        "EMG filter: 4th-order Butterworth 10–45 Hz, "
        "zero-phase offline."
    )
    log(
        "Power-line notch: 50 Hz, Q=30, zero-phase offline."
    )
    log(
        "No rectification, envelope extraction, normalization, "
        "feature fitting, or model-dependent processing."
    )
    log(
        "Channel QC is descriptive only; no channel is automatically removed."
    )

    for label, path in [
        ("ROOT", ROOT),
        ("RAW", RAW_ROOT),
        ("EXECUTION", EXECUTION_ROOT),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} path not found: {path}"
            )
        log(f"FOUND | {label:<10} | {path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    QC_ROOT.mkdir(parents=True, exist_ok=True)
    EMG_METADATA.mkdir(parents=True, exist_ok=True)

    files, inventory = discover_inventory()

    log(
        f"INPUT INVENTORY "
        f"{'PASS' if inventory['pass'] else 'FAIL'} | "
        f"EMG Motor Execution: "
        f"{inventory['actual_recordings']}/{inventory['expected_recordings']}"
    )

    if not inventory["pass"]:
        print("\nINPUT INVENTORY FAILURE")
        print(json.dumps(inventory, indent=2))
        return 1

    band_sos, notch_b, notch_a = design_filters()

    results = []
    channel_qc_rows = []
    temporal_rows = []

    processed_now = 0
    skipped_existing = 0
    failed = 0

    for i, path in enumerate(files, 1):
        if i == 1 or i % 25 == 0 or i == len(files):
            log(
                f"EMG_MOTOR_EXECUTION PRE2: "
                f"{i}/{len(files)}"
            )

        try:
            result = process_recording(
                path,
                band_sos,
                notch_b,
                notch_a,
            )

            results.append(result)

            if result["status"] == "PROCESSED":
                processed_now += 1
            else:
                skipped_existing += 1

            # Load provenance for the QC tables.
            meta_path = Path(result["meta_file"])
            if not meta_path.is_absolute():
                meta_path = ROOT / meta_path

            metadata = json.loads(
                meta_path.read_text(encoding="utf-8")
            )

            subject = metadata["subject"]
            gesture = metadata["gesture_folder"]
            set_id = metadata["set"]

            for ch in metadata["channel_qc"]:
                row = {
                    "subject": subject,
                    "gesture": gesture,
                    "set": set_id,
                    **ch,
                }
                channel_qc_rows.append(row)

            temporal_rows.append(
                {
                    "subject": subject,
                    "gesture": gesture,
                    "set": set_id,
                    **metadata["timestamp_qc"],
                }
            )

        except Exception as exc:
            failed += 1
            results.append(
                {
                    "status": "FAILED",
                    "input_file": str(path.relative_to(ROOT)),
                    "npz_file": "",
                    "meta_file": "",
                    "error": repr(exc),
                }
            )
            logging.exception(
                f"PRE2 FAILED | {path.relative_to(ROOT)}"
            )

    results_df = pd.DataFrame(results)
    channel_df = pd.DataFrame(channel_qc_rows)
    temporal_df = pd.DataFrame(temporal_rows)

    write_csv(
        results_df,
        EMG_METADATA / "PRE2_recording_processing_status.csv",
    )

    write_csv(
        channel_df,
        EMG_METADATA / "EMG_signal_quality.csv",
    )

    write_csv(
        temporal_df,
        EMG_METADATA / "timestamp_QC.csv",
    )

    # Recording-level summary.
    summary = pd.DataFrame(
        [
            {
                "modality": "EMG_MOTOR_EXECUTION",
                "condition": "MOTOR_EXECUTION",
                "expected_recordings": EXPECTED_RECORDINGS,
                "input_recordings": len(files),
                "processed_now": processed_now,
                "skipped_existing": skipped_existing,
                "failed": failed,
                "complete": failed == 0
                and (processed_now + skipped_existing)
                == EXPECTED_RECORDINGS,
                "sampling_rate_hz": SAMPLING_RATE_HZ,
                "channels": 3,
                "samples_per_recording": EXPECTED_SAMPLES,
                "bandpass_hz": "10–45",
                "notch_hz": NOTCH_FREQUENCY_HZ,
                "raw_modified": False,
            }
        ]
    )

    write_csv(
        summary,
        EMG_METADATA / "PRE2_processing_summary.csv",
    )

    # Channel-level summary.
    if len(channel_df):
        channel_summary = (
            channel_df.groupby("channel", as_index=False)
            .agg(
                recordings=("channel", "size"),
                constant_files=("constant_channel", "sum"),
                high_repeat_warning_files=(
                    "high_repeat_warning",
                    "sum",
                ),
                mean_raw_std=("raw_std", "mean"),
                mean_filtered_std=("filtered_std", "mean"),
                max_robust_extreme_fraction=(
                    "robust_extreme_fraction",
                    "max",
                ),
            )
        )
    else:
        channel_summary = pd.DataFrame()

    write_csv(
        channel_summary,
        EMG_METADATA / "EMG_channel_quality_summary.csv",
    )

    # Final audit: every expected recording must have a valid output.
    valid_count = 0
    invalid_outputs = []

    for path in files:
        npz_path, meta_path = output_paths(path)
        try:
            input_hash = sha256_file(path)
            if validate_existing_output(
                path,
                npz_path,
                meta_path,
                input_hash,
            ):
                valid_count += 1
            else:
                invalid_outputs.append(
                    str(path.relative_to(ROOT))
                )
        except Exception:
            invalid_outputs.append(
                str(path.relative_to(ROOT))
            )

    complete = (
        valid_count == EXPECTED_RECORDINGS
        and failed == 0
        and not invalid_outputs
    )

    final_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol_version": PROTOCOL_VERSION,
        "dataset_root": str(ROOT),
        "modality": "EMG_MOTOR_EXECUTION",
        "condition": "MOTOR_EXECUTION",
        "raw_data_modified": False,
        "expected_recordings": EXPECTED_RECORDINGS,
        "input_recordings": len(files),
        "processed_now": processed_now,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "validated_output_recordings": valid_count,
        "invalid_outputs": invalid_outputs[:20],
        "processing": {
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "channels": list(EXPECTED_EMG_CHANNELS),
            "samples_per_recording": EXPECTED_SAMPLES,
            "notch_hz": NOTCH_FREQUENCY_HZ,
            "notch_q": NOTCH_Q,
            "bandpass_low_hz": BANDPASS_LOW_HZ,
            "bandpass_high_hz": BANDPASS_HIGH_HZ,
            "filter_order": FILTER_ORDER,
            "phase": "ZERO_PHASE_OFFLINE",
        },
        "timestamp_policy": (
            "RAW_PRESERVED; QC_ONLY; "
            "PROCESSING_ROW_ORDER_PLUS_NOMINAL_125HZ"
        ),
        "automatic_channel_removal": False,
        "status": "COMPLETE" if complete else "FAIL",
    }

    atomic_json(
        final_report,
        EMG_METADATA / "PRE2_FINAL_REPORT.json",
    )

    if complete:
        sentinel = {
            "block": "PRE2",
            "name": "EMG Signal Preprocessing & Artifact QC",
            "status": "COMPLETE",
            "protocol_version": PROTOCOL_VERSION,
            "expected_recordings": EXPECTED_RECORDINGS,
            "validated_output_recordings": valid_count,
            "failed_recordings": 0,
            "raw_data_modified": False,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        atomic_json(sentinel, SENTINEL)

    print("\n" + "=" * 100)
    print("PRE2 FINAL SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\nValidation:")
    print(
        f"  Validated processed recordings : "
        f"{valid_count}/{EXPECTED_RECORDINGS}"
    )
    print(f"  Failed recordings              : {failed}")
    print(f"  Raw data modified              : NO")

    print("\nProcessing contract:")
    print(f"  Sampling rate                  : {SAMPLING_RATE_HZ:g} Hz")
    print("  EMG channels                   : 3")
    print(f"  Samples/recording              : {EXPECTED_SAMPLES}")
    print(
        f"  Band-pass                      : "
        f"{BANDPASS_LOW_HZ:g}–{BANDPASS_HIGH_HZ:g} Hz"
    )
    print(
        f"  Notch                          : "
        f"{NOTCH_FREQUENCY_HZ:g} Hz (Q={NOTCH_Q:g})"
    )
    print("  Phase                          : zero-phase offline")

    print("\n" + "=" * 100)
    print(
        f"PRE2 STATUS: "
        f"{'COMPLETE' if complete else 'FAIL'}"
    )
    print("RAW DATA MODIFICATION: NONE")
    print(f"OUTPUT: {OUTPUT_ROOT}")
    print(f"QC/PROVENANCE: {QC_ROOT}")
    print(f"Runtime: {time.time() - t0:.1f} seconds")
    print("=" * 100)

    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
