#!/usr/bin/env python3
"""
PRE1 — EEG Signal Preprocessing & Artifact QC
DATASET PAPER / FINAL DATASET V1.0

Purpose
-------
Create a deterministic, publication-ready EEG processed view from the frozen
01_RAW_DATA files without modifying the raw dataset.

This block processes BOTH:
    1) Motor Execution EEG: 40 subjects × 7 gestures × 3 sets = 840 files
    2) Motor Imagery EEG:   25 MI IDs × 7 gestures × 3 sets = 525 files

Scientific policy
-----------------
* Raw CSV files are READ-ONLY.
* Motor Execution and Motor Imagery are processed independently.
* No cross-session/cross-modal timestamp synchronization is attempted.
* The formatted acquisition timestamp and Sample Index are retained as raw
  metadata only. Signal processing uses row order and nominal Fs = 125 Hz.
* EEG is filtered with a 4th-order zero-phase Butterworth band-pass, 1–40 Hz.
  This preserves the broad EEG range supported by the 125-Hz acquisition rate.
* No narrow alpha/beta/gamma-only filtering is performed at this stage.
  Band-specific features can be derived later without discarding information.
* No ICA, CAR, interpolation, baseline correction, normalization, CSP,
  feature fitting, feature selection, balancing, or model-dependent operation
  is performed here.
* Automatic channel detection is QC-only. Channels are NOT silently deleted or
  reconstructed.
* Processing is deterministic and resumable.
* A processed recording is skipped only after its NPZ + metadata pass integrity
  checks and the input SHA-256 matches the current raw file.
* Processing output is written only under 06_PROCESSED_DATA/EEG_MOTOR_EXECUTION and 06_PROCESSED_DATA/EEG_MOTOR_IMAGERY.
* PRE1 does not overwrite the frozen DATA2 files in 05_QC.

Output
------
06_PROCESSED_DATA/
├── EEG_MOTOR_EXECUTION/
│   └── <Subject>/<Gesture>/<recording>_PRE1.npz
├── EEG_MOTOR_IMAGERY/
│   └── <MI_ID>/<Gesture>/<recording>_PRE1.npz
└── PRE1_QC/
    ├── separate modality metadata CSVs under PRE1_QC/EEG_MOTOR_EXECUTION and PRE1_QC/EEG_MOTOR_IMAGERY
    ├── EEG_MOTOR_EXECUTION/
    │   ├── PRE1_EEG_MOTOR_EXECUTION_metadata.csv
    │   └── PRE1_EEG_MOTOR_EXECUTION_protocol.json
    ├── EEG_MOTOR_IMAGERY/
    │   ├── PRE1_EEG_MOTOR_IMAGERY_metadata.csv
    │   └── PRE1_EEG_MOTOR_IMAGERY_protocol.json
    ├── PRE1_processing_summary.csv
    ├── PRE1_FINAL_REPORT.json
    └── PRE1_RUN_LOG.txt

Each NPZ contains:
    eeg                filtered EEG, shape (13, 5625)
    raw_eeg            original EEG values used as input, shape (13, 5625)
    sample_index       original Sample Index, shape (5625,)
    timestamp          original formatted timestamp strings, shape (5625,)
    time_seconds       deterministic processing axis, shape (5625,)
    windows            diagnostic 2-s windows, shape (44, 13, 250)

The raw signal is retained inside each processed artifact so the processed
record remains auditable and reversible. The original source CSV itself is
never changed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import json
import logging
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


# ============================================================================
# 1. DATASET ROOT AND FROZEN CONTRACT
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "01_RAW_DATA").exists():
    ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "01_RAW_DATA").exists():
    ROOT = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        "Cannot locate dataset root. Place this script in the dataset root "
        "or in dataset_root/scripts/."
    )

RAW = ROOT / "01_RAW_DATA"

EXECUTION = RAW / "EMG_EEG_SYNCHRONIZED"

IMAGERY = RAW / "EEG_MOTOR_IMAGERY"

PROCESSED_DATA = ROOT / "06_PROCESSED_DATA"

EXEC_OUT = PROCESSED_DATA / "EEG_MOTOR_EXECUTION"

MI_OUT = PROCESSED_DATA / "EEG_MOTOR_IMAGERY"

PRE1_QC = PROCESSED_DATA / "PRE1_QC"
EXEC_METADATA = PRE1_QC / "EEG_MOTOR_EXECUTION"
MI_METADATA = PRE1_QC / "EEG_MOTOR_IMAGERY"
PROTOCOL_VERSION = "PRE1-DATASET-V1.0-FINAL-FREEZE"

GLOBAL_SEED = 42
SAMPLING_RATE_HZ = 125.0
EXPECTED_SAMPLES = 5625
EXPECTED_DURATION_SEC = 45.0
EXPECTED_CHANNELS = 13

BANDPASS_LOW_HZ = 1.0
BANDPASS_HIGH_HZ = 40.0
FILTER_ORDER = 4

WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(round(WINDOW_SECONDS * SAMPLING_RATE_HZ))
STEP_SAMPLES = WINDOW_SAMPLES // 2
EXPECTED_WINDOWS = (EXPECTED_SAMPLES - WINDOW_SAMPLES) // STEP_SAMPLES + 1

FLATLINE_STD_THRESHOLD = 1e-12
ROBUST_Z_THRESHOLD = 5.0

GESTURES = [
    "Hand_Open_Close",
    "Index_Finger_Movement",
    "Little_Finger_Movement",
    "Middle_Finger_Movement",
    "Pen_Holding",
    "Ring_Finger_Movement",
    "Thumb_Finger_Movement",
]

GESTURE_CODE = {
    "Hand_Open_Close": "HOC",
    "Index_Finger_Movement": "IFM",
    "Little_Finger_Movement": "LFM",
    "Middle_Finger_Movement": "MFM",
    "Pen_Holding": "PH",
    "Ring_Finger_Movement": "RFM",
    "Thumb_Finger_Movement": "TFM",
}

GESTURE_DISPLAY = {
    "Hand_Open_Close": "Hand Open–Close",
    "Index_Finger_Movement": "Index Finger Movement",
    "Little_Finger_Movement": "Little Finger Movement",
    "Middle_Finger_Movement": "Middle Finger Movement",
    "Pen_Holding": "Pen Holding",
    "Ring_Finger_Movement": "Ring Finger Movement",
    "Thumb_Finger_Movement": "Thumb Finger Movement",
}

SETS = ("A", "B", "C")

EXPECTED_EXEC_COLS = [
    "Sample Index",
    "Timestamp (Formatted)",
    "EMG_ch-01", "EMG_ch-02", "EMG_ch-03",
    "EEG_ch-01", "EEG_ch-02", "EEG_ch-03", "EEG_ch-04",
    "EEG_ch-05", "EEG_ch-06", "EEG_ch-07", "EEG_ch-08",
    "EEG_ch-09", "EEG_ch-10", "EEG_ch-11", "EEG_ch-12", "EEG_ch-13",
]

EXPECTED_MI_COLS = [
    "Sample Index",
    "Timestamp (Formatted)",
    "EEG_ch-01", "EEG_ch-02", "EEG_ch-03", "EEG_ch-04",
    "EEG_ch-05", "EEG_ch-06", "EEG_ch-07", "EEG_ch-08",
    "EEG_ch-09", "EEG_ch-10", "EEG_ch-11", "EEG_ch-12", "EEG_ch-13",
]


# ============================================================================
# 2. LOGGING / ATOMIC WRITERS
# ============================================================================

LOG_FILE = PRE1_QC / "PRE1_RUN_LOG.txt"


def setup_logging() -> None:
    PRE1_QC.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def log(message: str) -> None:
    logging.info(message)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ============================================================================
# 3. ID / FILENAME PARSING
# ============================================================================

def parse_exec_filename(name: str):
    m = re.fullmatch(r"S(\d{2})_([A-Z]{2,3})_Set([ABC])\.csv", name, re.I)
    if not m:
        return None
    return int(m.group(1)), m.group(2).upper(), m.group(3).upper()


def parse_mi_filename(name: str):
    m = re.fullmatch(r"MI_(\d{2})_([A-Z]{2,3})_Set([ABC])\.csv", name, re.I)
    if not m:
        return None
    return int(m.group(1)), m.group(2).upper(), m.group(3).upper()


def validate_path_identity(path: Path, modality: str) -> dict:
    if modality == "MOTOR_EXECUTION":
        rel = path.relative_to(EXECUTION)
        parsed = parse_exec_filename(path.name)
        if parsed is None or len(rel.parts) != 3:
            raise ValueError(f"Invalid execution path/name: {path}")

        subject_num, code, set_name = parsed
        subject = rel.parts[0]
        gesture = rel.parts[1]

        expected_subject = f"Subject_{subject_num:02d}"
        expected_code = GESTURE_CODE.get(gesture)

        if subject != expected_subject:
            raise ValueError(f"Subject mismatch: {path}")
        if expected_code != code:
            raise ValueError(f"Gesture code mismatch: {path}")
        if set_name not in SETS:
            raise ValueError(f"Set mismatch: {path}")

        return {
            "modality": modality,
            "local_id": subject,
            "canonical_id": subject,
            "subject_number": subject_num,
            "gesture": gesture,
            "gesture_code": code,
            "set": set_name,
        }

    if modality == "MOTOR_IMAGERY":
        rel = path.relative_to(IMAGERY)
        parsed = parse_mi_filename(path.name)
        if parsed is None or len(rel.parts) != 3:
            raise ValueError(f"Invalid MI path/name: {path}")

        mi_num, code, set_name = parsed
        local_id = rel.parts[0]
        gesture = rel.parts[1]

        expected_local = f"MI_{mi_num:02d}"
        expected_code = GESTURE_CODE.get(gesture)

        if local_id != expected_local:
            raise ValueError(f"MI ID mismatch: {path}")
        if expected_code != code:
            raise ValueError(f"Gesture code mismatch: {path}")
        if set_name not in SETS:
            raise ValueError(f"Set mismatch: {path}")

        return {
            "modality": modality,
            "local_id": local_id,
            "canonical_id": "",
            "subject_number": mi_num,
            "gesture": gesture,
            "gesture_code": code,
            "set": set_name,
        }

    raise ValueError(f"Unsupported modality: {modality}")


# ============================================================================
# 4. INPUT SCHEMA / SIGNAL LOADING
# ============================================================================

def execution_expected_columns(subject_number: int):
    if not 1 <= subject_number <= 40:
        raise ValueError(f"Unsupported execution subject: {subject_number}")
    return EXPECTED_EXEC_COLS, "EXECUTION_CANONICAL_EEG_01_13"


def identify_eeg_columns(columns, modality: str, subject_number: int):
    if modality == "MOTOR_EXECUTION":
        expected, _ = execution_expected_columns(subject_number)
    else:
        expected = EXPECTED_MI_COLS

    if list(columns) != expected:
        raise ValueError("Input schema does not match the frozen DATA2 contract.")

    eeg_cols = [c for c in expected if c.startswith("EEG_ch-")]
    canonical_eeg = [
        f"EEG_ch-{i:02d}" for i in range(1, EXPECTED_CHANNELS + 1)
    ]

    if eeg_cols != canonical_eeg:
        raise ValueError(
            "EEG channel names are not canonical. "
            f"Expected {canonical_eeg}, found {eeg_cols}"
        )

    if len(eeg_cols) != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected {EXPECTED_CHANNELS} EEG channels, found {len(eeg_cols)}"
        )

    return eeg_cols


def load_recording(path: Path, modality: str, identity: dict):
    df = pd.read_csv(path)

    if len(df) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} rows, found {len(df)}: {path}"
        )

    eeg_cols = identify_eeg_columns(
        df.columns,
        modality,
        identity["subject_number"],
    )

    eeg = df[eeg_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    if eeg.shape != (EXPECTED_SAMPLES, EXPECTED_CHANNELS):
        raise ValueError(f"Unexpected EEG shape: {eeg.shape}")

    if not np.isfinite(eeg).all():
        raise ValueError(f"Non-finite EEG values found: {path}")

    sample_index = pd.to_numeric(
        df["Sample Index"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(sample_index).all():
        raise ValueError(f"Non-finite Sample Index found: {path}")

    timestamps = df["Timestamp (Formatted)"].astype(str).to_numpy()

    parsed_ts = pd.to_datetime(df["Timestamp (Formatted)"], errors="coerce")
    if parsed_ts.isna().any():
        raise ValueError(f"Unparseable timestamp found: {path}")

    return eeg, sample_index, timestamps, parsed_ts, eeg_cols


# ============================================================================
# 5. FILTERING
# ============================================================================

def design_filter():
    nyquist = SAMPLING_RATE_HZ / 2.0
    if BANDPASS_HIGH_HZ >= nyquist:
        raise ValueError("High cutoff must be below Nyquist frequency.")
    sos = butter(
        FILTER_ORDER,
        [BANDPASS_LOW_HZ / nyquist, BANDPASS_HIGH_HZ / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sos


SOS = design_filter()


def filter_eeg(eeg_time_by_channel: np.ndarray) -> np.ndarray:
    """
    Zero-phase offline filtering.

    Input/output shape: (samples, channels).
    """
    filtered = sosfiltfilt(
        SOS,
        eeg_time_by_channel,
        axis=0,
    )
    return np.asarray(filtered, dtype=np.float64)


# ============================================================================
# 6. QC METRICS
# ============================================================================

def robust_z(x: np.ndarray) -> np.ndarray:
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad <= np.finfo(float).eps:
        return np.zeros_like(x, dtype=float)
    return 0.6744897501960817 * (x - median) / mad


def channel_qc(raw: np.ndarray, filtered: np.ndarray) -> dict:
    """
    QC is descriptive only.

    The algorithm flags:
      - flat channels
      - channels with unusually large robust amplitude
      - channels with unusually large residual energy

    It does NOT remove or reconstruct channels.
    """
    raw_std = np.std(raw, axis=0)
    filt_std = np.std(filtered, axis=0)
    raw_rms = np.sqrt(np.mean(raw ** 2, axis=0))
    filt_rms = np.sqrt(np.mean(filtered ** 2, axis=0))
    residual = raw - filtered
    residual_rms = np.sqrt(np.mean(residual ** 2, axis=0))

    amp_scores = robust_z(raw_rms)
    residual_scores = robust_z(residual_rms)

    flat_mask = raw_std <= FLATLINE_STD_THRESHOLD
    amplitude_flag = np.abs(amp_scores) > ROBUST_Z_THRESHOLD
    residual_flag = np.abs(residual_scores) > ROBUST_Z_THRESHOLD

    # A robust QC flag is not automatically considered a failed recording.
    # It is retained for later inspection.
    qc_flag = flat_mask | amplitude_flag | residual_flag

    return {
        "raw_std": raw_std,
        "filtered_std": filt_std,
        "raw_rms": raw_rms,
        "filtered_rms": filt_rms,
        "residual_rms": residual_rms,
        "flat_mask": flat_mask,
        "amplitude_flag": amplitude_flag,
        "residual_flag": residual_flag,
        "qc_flag": qc_flag,
    }


def timestamp_metrics(parsed_ts: pd.Series) -> dict:
    dts = parsed_ts.diff().dt.total_seconds().dropna().to_numpy(dtype=float)

    if len(dts) == 0:
        return {
            "timestamp_negative_steps": 0,
            "timestamp_duplicate_rows": 0,
            "timestamp_gap_over_50ms": 0,
            "timestamp_gap_max_sec": np.nan,
            "timestamp_median_step_sec": np.nan,
            "timestamp_duration_sec": 0.0,
        }

    return {
        "timestamp_negative_steps": int((dts < 0).sum()),
        "timestamp_duplicate_rows": int(parsed_ts.duplicated().sum()),
        "timestamp_gap_over_50ms": int((dts > 0.050).sum()),
        "timestamp_gap_max_sec": float(np.max(dts)),
        "timestamp_median_step_sec": float(np.median(dts)),
        "timestamp_duration_sec": float(
            (parsed_ts.iloc[-1] - parsed_ts.iloc[0]).total_seconds()
        ),
    }


def sample_index_metrics(index: np.ndarray) -> dict:
    d = np.diff(index)
    if len(d) == 0:
        return {
            "sample_index_negative_steps": 0,
            "sample_index_duplicate_rows": 0,
            "sample_index_median_step": np.nan,
        }

    return {
        "sample_index_negative_steps": int((d < 0).sum()),
        "sample_index_duplicate_rows": int(pd.Series(index).duplicated().sum()),
        "sample_index_median_step": float(np.median(d)),
    }


def make_windows(filtered: np.ndarray) -> np.ndarray:
    """
    Deterministic diagnostic windows.

    Output shape: (44, 13, 250)
    """
    starts = range(
        0,
        EXPECTED_SAMPLES - WINDOW_SAMPLES + 1,
        STEP_SAMPLES,
    )
    windows = np.stack(
        [filtered[s:s + WINDOW_SAMPLES].T for s in starts],
        axis=0,
    )

    if windows.shape != (EXPECTED_WINDOWS, EXPECTED_CHANNELS, WINDOW_SAMPLES):
        raise RuntimeError(f"Unexpected window shape: {windows.shape}")

    return windows.astype(np.float32)


# ============================================================================
# 7. RESUME / OUTPUT INTEGRITY
# ============================================================================

def output_paths(input_path: Path, modality: str):
    if modality == "MOTOR_EXECUTION":
        rel = input_path.relative_to(EXECUTION)
        out_dir = EXEC_OUT / rel.parent
    else:
        rel = input_path.relative_to(IMAGERY)
        out_dir = MI_OUT / rel.parent

    out_npz = out_dir / f"{input_path.stem}_PRE1.npz"
    out_json = out_dir / f"{input_path.stem}_PRE1.json"
    return out_npz, out_json


def load_existing_result(
    npz_path: Path,
    json_path: Path,
    input_path: Path,
    identity: dict,
):
    if not npz_path.exists() or not json_path.exists():
        return None

    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))

        if meta.get("protocol_version") != PROTOCOL_VERSION:
            return None
        if meta.get("input_sha256") != sha256_file(input_path):
            return None
        if meta.get("recording_key") != (
            f"{identity['local_id']}__{identity['gesture']}__Set{identity['set']}"
        ):
            return None

        with np.load(npz_path, allow_pickle=False) as z:
            required = {
                "eeg",
                "raw_eeg",
                "sample_index",
                "timestamp",
                "time_seconds",
                "windows",
            }
            if not required.issubset(set(z.files)):
                return None

            eeg = z["eeg"]
            raw_eeg = z["raw_eeg"]
            windows = z["windows"]

            if eeg.shape != (EXPECTED_CHANNELS, EXPECTED_SAMPLES):
                return None
            if raw_eeg.shape != (EXPECTED_CHANNELS, EXPECTED_SAMPLES):
                return None
            if windows.shape != (
                EXPECTED_WINDOWS,
                EXPECTED_CHANNELS,
                WINDOW_SAMPLES,
            ):
                return None

            if not np.isfinite(eeg).all():
                return None
            if not np.isfinite(raw_eeg).all():
                return None
            if not np.isfinite(windows).all():
                return None

        return meta

    except Exception:
        return None


# ============================================================================
# 8. PROCESS ONE RECORDING
# ============================================================================

def process_recording(path: Path, modality: str):
    identity = validate_path_identity(path, modality)
    out_npz, out_json = output_paths(path, modality)

    existing = load_existing_result(
        out_npz,
        out_json,
        path,
        identity,
    )

    if existing is not None:
        return {
            "status": "SKIPPED_EXISTING",
            "identity": identity,
            "input": path,
            "npz": out_npz,
            "meta": out_json,
            "qc": existing.get("qc", {}),
        }

    input_hash = sha256_file(path)

    raw, sample_index, timestamps, parsed_ts, eeg_cols = load_recording(
        path,
        modality,
        identity,
    )

    # Time-by-channel for signal processing.
    raw_time_by_channel = raw

    filtered = filter_eeg(raw_time_by_channel)

    if filtered.shape != raw.shape:
        raise RuntimeError(f"Filtered shape mismatch: {filtered.shape}")

    if not np.isfinite(filtered).all():
        raise RuntimeError("Non-finite filtered EEG generated.")

    qc = channel_qc(raw, filtered)
    ts_qc = timestamp_metrics(parsed_ts)
    idx_qc = sample_index_metrics(sample_index)

    # Deterministic processing time axis. This is NOT written back to raw CSV.
    time_seconds = np.arange(EXPECTED_SAMPLES, dtype=np.float64) / SAMPLING_RATE_HZ

    windows = make_windows(filtered)

    recording_key = (
        f"{identity['local_id']}__{identity['gesture']}__Set{identity['set']}"
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = out_npz.with_name(out_npz.name + ".tmp")

    with tmp_npz.open("wb") as f:
        np.savez_compressed(
            f,
            eeg=filtered.T.astype(np.float32),
            raw_eeg=raw.T.astype(np.float32),
            sample_index=sample_index.astype(np.float64),
            timestamp=timestamps.astype(str),
            time_seconds=time_seconds,
            windows=windows,
        )

    os.replace(tmp_npz, out_npz)

    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "modality": modality,
        "local_id": identity["local_id"],
        "canonical_id": identity["canonical_id"],
        "subject_number": identity["subject_number"],
        "gesture": identity["gesture"],
        "gesture_display": GESTURE_DISPLAY[identity["gesture"]],
        "gesture_code": identity["gesture_code"],
        "set": identity["set"],
        "recording_key": recording_key,
        "input_file": str(path.relative_to(ROOT)),
        "input_sha256": input_hash,
        "raw_data_modified": False,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "samples": EXPECTED_SAMPLES,
        "nominal_duration_sec": EXPECTED_DURATION_SEC,
        "channels": EXPECTED_CHANNELS,
        "channel_names": [
            f"EEG_ch-{i:02d}" for i in range(1, EXPECTED_CHANNELS + 1)
        ],
        "filter": {
            "type": "Butterworth band-pass",
            "order": FILTER_ORDER,
            "low_hz": BANDPASS_LOW_HZ,
            "high_hz": BANDPASS_HIGH_HZ,
            "phase": "zero-phase offline",
        },
        "notch_filter_hz": None,
        "reference": "None in PRE1",
        "channel_order": "EEG_ch-01 through EEG_ch-13 in ascending canonical order",
        "bad_channel_policy": "QC-only; no removal or interpolation",
        "ica": "Not performed",
        "baseline_correction": "Not performed",
        "normalization": "Not performed",
        "processing_time_axis": "row_order_plus_nominal_125Hz",
        "raw_timestamp_policy": "preserved_as_metadata_only",
        "sample_index_policy": "preserved_as_metadata_only",
        "diagnostic_windows": {
            "window_seconds": WINDOW_SECONDS,
            "window_samples": WINDOW_SAMPLES,
            "step_samples": STEP_SAMPLES,
            "overlap": 0.50,
            "number_of_windows": EXPECTED_WINDOWS,
        },
        "timestamp_qc": ts_qc,
        "sample_index_qc": idx_qc,
        "qc": {
            "flat_channels": [
                eeg_cols[i]
                for i, v in enumerate(qc["flat_mask"]) if bool(v)
            ],
            "amplitude_outlier_channels": [
                eeg_cols[i]
                for i, v in enumerate(qc["amplitude_flag"]) if bool(v)
            ],
            "residual_outlier_channels": [
                eeg_cols[i]
                for i, v in enumerate(qc["residual_flag"]) if bool(v)
            ],
            "flagged_channels": [
                eeg_cols[i]
                for i, v in enumerate(qc["qc_flag"]) if bool(v)
            ],
        },
        "output": {
            "npz": str(out_npz.relative_to(ROOT)),
            "metadata": str(out_json.relative_to(ROOT)),
        },
    }

    atomic_json(meta, out_json)

    # Serialization integrity check immediately after writing.
    checked = load_existing_result(out_npz, out_json, path, identity)
    if checked is None:
        raise RuntimeError(f"Serialization QA failed: {out_npz}")

    return {
        "status": "PROCESSED",
        "identity": identity,
        "input": path,
        "npz": out_npz,
        "meta": out_json,
        "qc": meta["qc"],
    }


# ============================================================================
# 9. INVENTORY
# ============================================================================

def discover_files(root: Path):
    return sorted(root.rglob("*.csv"))


def expected_execution_slots():
    return {
        (
            f"Subject_{s:02d}",
            g,
            set_name,
        )
        for s in range(1, 41)
        for g in GESTURES
        for set_name in SETS
    }


def expected_mi_slots():
    return {
        (
            f"MI_{s:02d}",
            g,
            set_name,
        )
        for s in range(1, 26)
        for g in GESTURES
        for set_name in SETS
    }


def inventory_check(files, modality):
    expected = (
        expected_execution_slots()
        if modality == "MOTOR_EXECUTION"
        else expected_mi_slots()
    )

    actual = set()
    errors = []

    for path in files:
        try:
            identity = validate_path_identity(path, modality)
            key = (
                identity["local_id"],
                identity["gesture"],
                identity["set"],
            )
            if key in actual:
                errors.append(f"DUPLICATE_SLOT: {path.relative_to(ROOT)}")
            actual.add(key)
        except Exception as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing:
        errors.extend([f"MISSING_SLOT: {x}" for x in missing[:20]])
    if extra:
        errors.extend([f"EXTRA_SLOT: {x}" for x in extra[:20]])

    expected_count = 840 if modality == "MOTOR_EXECUTION" else 525

    return {
        "expected": expected_count,
        "actual_files": len(files),
        "actual_slots": len(actual),
        "missing_slots": len(missing),
        "extra_slots": len(extra),
        "errors": errors,
        "pass": (
            len(files) == expected_count
            and len(actual) == expected_count
            and not missing
            and not extra
            and not errors
        ),
    }


# ============================================================================
# 10. RUN ONE MODALITY
# ============================================================================

def run_modality(files, modality):
    results = []
    total = len(files)

    for i, path in enumerate(files, 1):
        if i == 1 or i % 25 == 0 or i == total:
            log(f"{modality} PRE1: {i}/{total}")

        try:
            result = process_recording(path, modality)
            identity = result["identity"]
            results.append({
                "modality": modality,
                "local_id": identity["local_id"],
                "canonical_id": identity["canonical_id"],
                "gesture": identity["gesture"],
                "gesture_code": identity["gesture_code"],
                "set": identity["set"],
                "file": str(path.relative_to(ROOT)),
                "status": result["status"],
                "output_file": str(result["npz"].relative_to(ROOT)),
                "metadata_file": str(result["meta"].relative_to(ROOT)),
                "error": "",
            })
        except Exception as exc:
            log(f"ERROR | {path.relative_to(ROOT)} | {exc}")
            results.append({
                "modality": modality,
                "local_id": "",
                "canonical_id": "",
                "gesture": "",
                "gesture_code": "",
                "set": "",
                "file": str(path.relative_to(ROOT)),
                "status": "FAILED",
                "output_file": "",
                "metadata_file": "",
                "error": repr(exc),
            })

    return pd.DataFrame(results)


# ============================================================================
# 11. FINAL AUDIT / SENTINEL
# ============================================================================

def write_modality_metadata(result_df: pd.DataFrame, modality: str) -> None:
    """
    Write modality-specific PRE1 metadata/QC.

    Motor Execution and Motor Imagery are intentionally kept in separate
    metadata files because their identifiers and acquisition contexts differ.
    """
    if modality == "MOTOR_EXECUTION":
        out = EXEC_METADATA / "PRE1_EEG_MOTOR_EXECUTION_metadata.csv"
    elif modality == "MOTOR_IMAGERY":
        out = MI_METADATA / "PRE1_EEG_MOTOR_IMAGERY_metadata.csv"
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    atomic_csv(result_df, out)


def make_summary(result_df, inventory):
    processed = int((result_df["status"] == "PROCESSED").sum())
    skipped = int((result_df["status"] == "SKIPPED_EXISTING").sum())
    failed = int((result_df["status"] == "FAILED").sum())

    return {
        "expected_recordings": inventory["expected"],
        "input_files": inventory["actual_files"],
        "inventory_pass": bool(inventory["pass"]),
        "processed_now": processed,
        "skipped_existing": skipped,
        "failed": failed,
        "complete": bool(
            inventory["pass"]
            and failed == 0
            and (processed + skipped) == inventory["expected"]
        ),
    }


def write_protocol():
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "Dataset-paper EEG preprocessing",
        "raw_data_modified": False,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "expected_samples": EXPECTED_SAMPLES,
        "expected_channels": EXPECTED_CHANNELS,
        "filter": {
            "type": "Butterworth band-pass",
            "order": FILTER_ORDER,
            "low_hz": BANDPASS_LOW_HZ,
            "high_hz": BANDPASS_HIGH_HZ,
            "phase": "zero-phase offline",
        },
        "notch_filter_hz": None,
        "reference": "None in PRE1",
        "channel_order": "EEG_ch-01 through EEG_ch-13 in ascending canonical order",
        "artifact_policy": {
            "channel_detection": "QC-only",
            "channel_removal": False,
            "interpolation": False,
            "ICA": False,
            "baseline_correction": False,
            "normalization": False,
        },
        "timestamp_policy": (
            "Original formatted timestamps preserved; processing time axis "
            "derived from row order and nominal 125 Hz."
        ),
        "modalities": {
            "MOTOR_EXECUTION": "40 subjects × 7 gestures × 3 sets = 840",
            "MOTOR_IMAGERY": "25 MI IDs × 7 gestures × 3 sets = 525",
        },
    }
    execution_payload = dict(payload)
    execution_payload["modality"] = "MOTOR_EXECUTION"
    execution_payload["identifier_policy"] = (
        "Subject_01..Subject_40 are canonical execution participant IDs."
    )
    execution_payload["input_contract"] = {
        "files": 840,
        "structure": "Subject_<NN>/<Gesture>/S<NN>_<CODE>_Set[A-C].csv",
        "channels": ["EEG_ch-01", "EEG_ch-02", "EEG_ch-03", "EEG_ch-04",
                     "EEG_ch-05", "EEG_ch-06", "EEG_ch-07", "EEG_ch-08",
                     "EEG_ch-09", "EEG_ch-10", "EEG_ch-11", "EEG_ch-12",
                     "EEG_ch-13"],
        "emg_channels": ["EMG_ch-01", "EMG_ch-02", "EMG_ch-03"],
    }
    execution_payload["metadata_output"] = (
        "PRE1_QC/EEG_MOTOR_EXECUTION/PRE1_EEG_MOTOR_EXECUTION_metadata.csv"
    )
    atomic_json(
        execution_payload,
        EXEC_METADATA / "PRE1_EEG_MOTOR_EXECUTION_protocol.json",
    )

    imagery_payload = dict(payload)
    imagery_payload["modality"] = "MOTOR_IMAGERY"
    imagery_payload["identifier_policy"] = (
        "MI_01..MI_25 are modality-local motor-imagery recording IDs; "
        "they are not assumed to equal Subject_01..Subject_25."
    )
    imagery_payload["input_contract"] = {
        "files": 525,
        "structure": "MI_<NN>/<Gesture>/MI_<NN>_<CODE>_Set[A-C].csv",
        "channels": ["EEG_ch-01", "EEG_ch-02", "EEG_ch-03", "EEG_ch-04",
                     "EEG_ch-05", "EEG_ch-06", "EEG_ch-07", "EEG_ch-08",
                     "EEG_ch-09", "EEG_ch-10", "EEG_ch-11", "EEG_ch-12",
                     "EEG_ch-13"],
        "emg_channels": [],
    }
    imagery_payload["metadata_output"] = (
        "PRE1_QC/EEG_MOTOR_IMAGERY/PRE1_EEG_MOTOR_IMAGERY_metadata.csv"
    )
    atomic_json(
        imagery_payload,
        MI_METADATA / "PRE1_EEG_MOTOR_IMAGERY_protocol.json",
    )


def main():
    np.random.seed(GLOBAL_SEED)
    setup_logging()

    t0 = time.time()

    print("=" * 100)
    print("PRE1 — EEG SIGNAL PREPROCESSING & ARTIFACT QC")
    print("DATASET PAPER / FINAL DATASET V1.0")
    print("=" * 100)

    log(f"Protocol version: {PROTOCOL_VERSION}")
    log("RAW DATA MODIFICATION: NONE")
    log("Resume mode: ENABLED")
    log("Motor Execution and Motor Imagery are processed independently with separate PRE1 metadata.")
    log("Processing time axis: row order + nominal 125 Hz.")
    log("Raw formatted timestamps and Sample Index are preserved as metadata.")
    log(
        f"EEG filter: {FILTER_ORDER}th-order Butterworth "
        f"{BANDPASS_LOW_HZ:g}–{BANDPASS_HIGH_HZ:g} Hz, zero-phase offline."
    )
    log("No ICA, CAR, interpolation, baseline correction, normalization, or feature fitting.")
    log("Channel QC is descriptive only; no channel is automatically removed.")

    for label, path in [
        ("ROOT", ROOT),
        ("RAW", RAW),
        ("EXECUTION", EXECUTION),
        ("IMAGERY", IMAGERY),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} path not found: {path}")
        log(f"FOUND | {label:<10} | {path}")

    EXEC_OUT.mkdir(parents=True, exist_ok=True)
    MI_OUT.mkdir(parents=True, exist_ok=True)
    PRE1_QC.mkdir(parents=True, exist_ok=True)
    EXEC_METADATA.mkdir(parents=True, exist_ok=True)
    MI_METADATA.mkdir(parents=True, exist_ok=True)

    write_protocol()

    exec_files = discover_files(EXECUTION)
    mi_files = discover_files(IMAGERY)

    exec_inventory = inventory_check(exec_files, "MOTOR_EXECUTION")
    mi_inventory = inventory_check(mi_files, "MOTOR_IMAGERY")

    if not exec_inventory["pass"]:
        raise RuntimeError(
            "Motor Execution inventory does not satisfy the frozen DATA2 "
            f"coverage contract: {exec_inventory['errors'][:10]}"
        )

    if not mi_inventory["pass"]:
        raise RuntimeError(
            "Motor Imagery inventory does not satisfy the frozen DATA2 "
            f"coverage contract: {mi_inventory['errors'][:10]}"
        )

    log("INPUT INVENTORY PASS | Motor Execution: 840/840")
    log("INPUT INVENTORY PASS | Motor Imagery: 525/525")

    exec_df = run_modality(exec_files, "MOTOR_EXECUTION")
    mi_df = run_modality(mi_files, "MOTOR_IMAGERY")

    result_df = pd.concat([exec_df, mi_df], ignore_index=True)

    # Keep PRE1 metadata/QC separate by acquisition condition.
    write_modality_metadata(exec_df, "MOTOR_EXECUTION")
    write_modality_metadata(mi_df, "MOTOR_IMAGERY")

    exec_summary = make_summary(exec_df, exec_inventory)
    mi_summary = make_summary(mi_df, mi_inventory)

    summary_df = pd.DataFrame([
        {
            "modality": "MOTOR_EXECUTION",
            **exec_summary,
        },
        {
            "modality": "MOTOR_IMAGERY",
            **mi_summary,
        },
    ])

    atomic_csv(
        summary_df,
        PRE1_QC / "PRE1_processing_summary.csv",
    )

    total_expected = 840 + 525
    total_complete = int(
        exec_summary["processed_now"]
        + exec_summary["skipped_existing"]
        + mi_summary["processed_now"]
        + mi_summary["skipped_existing"]
    )
    total_failed = int(exec_summary["failed"] + mi_summary["failed"])

    complete = (
        exec_summary["complete"]
        and mi_summary["complete"]
        and total_complete == total_expected
        and total_failed == 0
    )

    run_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol_version": PROTOCOL_VERSION,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "raw_data_modified": False,
        "execution": exec_summary,
        "motor_imagery": mi_summary,
        "total_expected_recordings": total_expected,
        "total_complete_recordings": total_complete,
        "total_failed_recordings": total_failed,
        "output_root": str(PROCESSED_DATA.relative_to(ROOT)),
        "runtime_seconds": round(time.time() - t0, 2),
    }

    atomic_json(
        run_report,
        PRE1_QC / "PRE1_FINAL_REPORT.json",
    )

    print("\n" + "=" * 100)
    print("PRE1 FINAL SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print("\nTotal:")
    print(f"  Expected recordings : {total_expected}")
    print(f"  Complete recordings : {total_complete}")
    print(f"  Failed recordings   : {total_failed}")

    print("\n" + "=" * 100)
    print(f"PRE1 STATUS: {'COMPLETE' if complete else 'INCOMPLETE'}")
    print("RAW DATA MODIFICATION: NONE")
    print(f"OUTPUT: {PROCESSED_DATA}")
    print(f"QC/PROVENANCE: {PRE1_QC}")
    print(f"Runtime: {time.time() - t0:.1f} seconds")
    print("=" * 100)

    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
