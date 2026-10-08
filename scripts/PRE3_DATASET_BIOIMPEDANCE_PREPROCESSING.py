"""
PRE3_DATASET_BIOIMPEDANCE_PREPROCESSING.py

Dataset Paper / FINAL DATASET V1.0
PRE3 — Bioimpedance Measurement Standardization & Quality Control

Processed output mirrors the native BioZ hierarchy. No MOTOR_EXECUTION
directory is used for Bioimpedance.

Bioimpedance is a frequency-domain measurement, not a continuous
uniformly sampled time-series signal.

Native .spec measurement:
    frequency[Hz], Re, Im

Raw .spec files are NEVER modified.

PRE3 performs:
    1. Exact BioZ inventory discovery
    2. Exact recording-unit pairing
    3. Native .spec parsing
    4. Header/channel validation
    5. 100-point frequency-sweep validation
    6. Finite-value validation
    7. Frequency ordering validation
    8. Derived impedance magnitude and phase calculation
    9. Processed-data generation
   10. Resume validation using SHA-256
   11. Detailed QC/provenance

PRE3 does NOT:
    - reconstruct time
    - infer recording duration
    - filter
    - resample
    - interpolate
    - smooth
    - normalize
    - fit features
    - perform ML processing
    - synchronize with EEG/EMG

Expected canonical inventory:
    18 BioZ participants
    7 gestures/participant
    15 sets/gesture
    1,890 recording units
    2 channels/unit
    3,780 native .spec files
    100 frequency points/channel
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROTOCOL_VERSION = "PRE3-DATASET-V1.0-FINAL-FREEZE"

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

# Resolve the dataset root robustly whether this script is placed in:
#   <dataset_root>/scripts/
# or directly in:
#   <dataset_root>/
if (SCRIPT_DIR / "01_RAW_DATA").exists():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "01_RAW_DATA").exists():
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        "Cannot locate dataset root. Expected 01_RAW_DATA under the "
        "script directory or its parent."
    )

RAW_BIOZ = PROJECT_ROOT / "01_RAW_DATA" / "BIOIMPEDANCE"
PROCESSED_BIOZ = (
    PROJECT_ROOT
    / "06_PROCESSED_DATA"
    / "BIOIMPEDANCE"
)
QC_ROOT = (
    PROJECT_ROOT
    / "06_PROCESSED_DATA"
    / "BIOIMPEDANCE"
    / "PRE3_QC"
)

EXPECTED_PARTICIPANTS = 18
EXPECTED_GESTURES = 7
EXPECTED_SETS_PER_GESTURE = 15
EXPECTED_UNITS = 1890
EXPECTED_SPEC_FILES = 3780
EXPECTED_POINTS = 100
EXPECTED_CHANNELS = {1, 2}

BIOZ_CROSSWALK = {
    "BZ_01": "Subject_01",
    "BZ_02": "Subject_02",
    "BZ_03": "Subject_03",
    "BZ_04": "Subject_04",
    "BZ_05": "Subject_05",
    "BZ_06": "Subject_06",
    "BZ_07": "Subject_07",
    "BZ_08": "Subject_08",
    "BZ_09": "Subject_09",
    "BZ_10": "Subject_10",
    "BZ_11": "Subject_11",
    "BZ_12": "Subject_12",
    "BZ_13": "Subject_13",
    "BZ_14": "Subject_14",
    "BZ_15": "Subject_15",
    "BZ_16": "Subject_17",
    "BZ_17": "Subject_29",
    "BZ_18": "Subject_39",
}

GESTURE_CODES = {
    "HOC": "Hand Open–Close",
    "TFM": "Thumb Finger Movement",
    "IFM": "Index Finger Movement",
    "MFM": "Middle Finger Movement",
    "RFM": "Ring Finger Movement",
    "LFM": "Little Finger Movement",
    "PH": "Pen Holding",
}


# ============================================================
# OUTPUT STRUCTURE
# ============================================================
#
# Processed BioZ mirrors the raw hierarchy:
#
# 06_PROCESSED_DATA/
# └── BIOIMPEDANCE/
#     ├── BZ_01/
#     │   ├── BZ_01_HOC/
#     │   │   ├── Channel_1/
#     │   │   │   ├── set_01.npz
#     │   │   │   └── set_01.json
#     │   │   └── Channel_2/
#     │   │       ├── set_01.npz
#     │   │       └── set_01.json
#     │   ├── BZ_01_TFM/
#     │   ├── BZ_01_IFM/
#     │   ├── BZ_01_MFM/
#     │   ├── BZ_01_RFM/
#     │   ├── BZ_01_LFM/
#     │   └── BZ_01_PH/
#     └── ...
#
# There is intentionally NO MOTOR_EXECUTION directory.
# ============================================================

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    try:
        with open(tmp_name, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_name, path)

    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def write_json(path: Path, payload: Dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    atomic_write_bytes(path, text.encode("utf-8"))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".csv.tmp",
        dir=str(path.parent),
        text=True,
    )
    os.close(fd)
    try:
        with open(tmp_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


# ============================================================
# EXACT PATH PARSING
# ============================================================

def parse_bioz_path(path: Path) -> Dict:
    """
    Expected:
    BZ_01/BZ_01_HOC/Channel_1/set_01.spec
    """
    path = Path(path)

    if path.suffix.lower() != ".spec":
        raise ValueError(f"Not a .spec file: {path}")

    channel_match = re.fullmatch(
        r"Channel_([12])",
        path.parent.name,
        flags=re.IGNORECASE,
    )
    if not channel_match:
        raise ValueError(f"Invalid channel directory: {path.parent.name}")

    set_match = re.fullmatch(
        r"set_(\d{2})",
        path.stem,
        flags=re.IGNORECASE,
    )
    if not set_match:
        raise ValueError(f"Invalid set filename: {path.name}")

    gesture_dir = path.parent.parent.name
    gesture_match = re.fullmatch(
        r"(BZ_\d{2})_([A-Z]{2,3})",
        gesture_dir,
        flags=re.IGNORECASE,
    )
    if not gesture_match:
        raise ValueError(f"Invalid gesture directory: {gesture_dir}")

    bz_id = gesture_match.group(1).upper()
    gesture_code = gesture_match.group(2).upper()
    set_number = int(set_match.group(1))
    channel = int(channel_match.group(1))

    expected_bz_folder = path.parent.parent.parent.name.upper()
    if expected_bz_folder != bz_id:
        raise ValueError(
            f"Participant hierarchy mismatch: "
            f"{expected_bz_folder} vs {bz_id}"
        )

    if bz_id not in BIOZ_CROSSWALK:
        raise ValueError(f"Unexpected BioZ participant: {bz_id}")

    if gesture_code not in GESTURE_CODES:
        raise ValueError(f"Unexpected gesture code: {gesture_code}")

    if not 1 <= set_number <= EXPECTED_SETS_PER_GESTURE:
        raise ValueError(f"Invalid set number: {set_number}")

    return {
        "bz_id": bz_id,
        "subject_id": BIOZ_CROSSWALK[bz_id],
        "gesture_code": gesture_code,
        "gesture_name": GESTURE_CODES[gesture_code],
        "set_number": set_number,
        "channel": channel,
        "path": path,
    }


# ============================================================
# DISCOVERY AND INVENTORY
# ============================================================

def discover_all_spec_files() -> List[Path]:
    return sorted(
        RAW_BIOZ.rglob("*.spec"),
        key=lambda p: str(p).lower(),
    )


def build_exact_inventory(spec_files: List[Path]) -> List[Dict]:
    rows = []
    errors = []

    for path in spec_files:
        try:
            info = parse_bioz_path(path)
            info["relative_path"] = str(path.relative_to(PROJECT_ROOT))
            rows.append(info)
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})

    if errors:
        raise RuntimeError(
            "Invalid BioZ path structure detected:\n"
            + "\n".join(
                f"{x['path']} -> {x['error']}" for x in errors[:20]
            )
        )

    return rows


def pair_recording_units(inventory_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, int], Dict[int, Dict]] = {}

    for row in inventory_rows:
        key = (
            row["bz_id"],
            row["gesture_code"],
            row["set_number"],
        )
        grouped.setdefault(key, {})
        channel = row["channel"]

        if channel in grouped[key]:
            raise RuntimeError(
                f"Duplicate channel file for recording unit: {key}, "
                f"channel={channel}"
            )

        grouped[key][channel] = row

    expected_keys = {
        (bz_id, gesture_code, set_number)
        for bz_id in BIOZ_CROSSWALK
        for gesture_code in GESTURE_CODES
        for set_number in range(1, EXPECTED_SETS_PER_GESTURE + 1)
    }

    actual_keys = set(grouped.keys())

    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)

    if missing or extra:
        raise RuntimeError(
            f"BioZ exact recording-unit coverage mismatch. "
            f"Missing={len(missing)}, Extra={len(extra)}"
        )

    units = []

    for key in sorted(grouped.keys()):
        channels = grouped[key]

        if set(channels.keys()) != EXPECTED_CHANNELS:
            raise RuntimeError(
                f"Channel pairing failure for {key}: "
                f"found {sorted(channels.keys())}"
            )

        bz_id, gesture_code, set_number = key

        units.append({
            "unit_id": f"{bz_id}_{gesture_code}_Set_{set_number:02d}",
            "bz_id": bz_id,
            "subject_id": BIOZ_CROSSWALK[bz_id],
            "gesture_code": gesture_code,
            "gesture_name": GESTURE_CODES[gesture_code],
            "set_number": set_number,
            "channel_1_path": channels[1]["path"],
            "channel_2_path": channels[2]["path"],
        })

    return units


# ============================================================
# NATIVE .SPEC PARSING
# ============================================================

def parse_spec(path: Path) -> Dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Find the native measurement header.
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "frequency[hz],re,im":
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            f"Measurement header not found in {path}"
        )

    channel_name = None
    for line in lines[:header_idx]:
        stripped = line.strip()
        if stripped.lower().startswith("channel:"):
            channel_name = stripped.split(":", 1)[1].strip()
            break

    frequency = []
    re_values = []
    im_values = []

    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue

        parts = [x.strip() for x in stripped.split(",")]
        if len(parts) != 3:
            continue

        try:
            f = float(parts[0])
            re_v = float(parts[1])
            im_v = float(parts[2])
        except ValueError:
            continue

        frequency.append(f)
        re_values.append(re_v)
        im_values.append(im_v)

    frequency = np.asarray(frequency, dtype=np.float64)
    re_values = np.asarray(re_values, dtype=np.float64)
    im_values = np.asarray(im_values, dtype=np.float64)

    if len(frequency) != EXPECTED_POINTS:
        raise ValueError(
            f"{path}: expected {EXPECTED_POINTS} points, "
            f"found {len(frequency)}"
        )

    if not (
        np.isfinite(frequency).all()
        and np.isfinite(re_values).all()
        and np.isfinite(im_values).all()
    ):
        raise ValueError(f"{path}: non-finite measurement value detected")

    if len(np.unique(frequency)) != EXPECTED_POINTS:
        raise ValueError(f"{path}: duplicate frequency points detected")

    if not np.all(np.diff(frequency) > 0):
        raise ValueError(
            f"{path}: frequency values are not strictly increasing"
        )

    return {
        "channel_name": channel_name,
        "frequency_hz": frequency,
        "re": re_values,
        "im": im_values,
    }


# ============================================================
# PROCESSED OUTPUT VALIDATION
# ============================================================

def output_paths(unit: Dict) -> Dict[int, Tuple[Path, Path]]:
    # Mirror the native BioZ hierarchy:
    # BZ_01/BZ_01_HOC/Channel_1/set_01.npz
    gesture_dir = (
        PROCESSED_BIOZ
        / unit["bz_id"]
        / f"{unit['bz_id']}_{unit['gesture_code']}"
    )

    return {
        1: (
            gesture_dir / "Channel_1"
            / f"set_{unit['set_number']:02d}.npz",
            gesture_dir / "Channel_1"
            / f"set_{unit['set_number']:02d}.json",
        ),
        2: (
            gesture_dir / "Channel_2"
            / f"set_{unit['set_number']:02d}.npz",
            gesture_dir / "Channel_2"
            / f"set_{unit['set_number']:02d}.json",
        ),
    }


def validate_existing_output(
    npz_path: Path,
    json_path: Path,
    input_sha: str,
) -> bool:
    if not npz_path.exists() or not json_path.exists():
        return False

    try:
        metadata = json.loads(
            json_path.read_text(encoding="utf-8")
        )

        if metadata.get("input_sha256") != input_sha:
            return False

        with np.load(npz_path, allow_pickle=False) as data:
            required = {
                "frequency_hz",
                "re",
                "im",
                "z_magnitude",
                "z_phase_deg",
            }

            if not required.issubset(set(data.files)):
                return False

            for name in required:
                arr = np.asarray(data[name])
                if arr.shape != (EXPECTED_POINTS,):
                    return False
                if not np.isfinite(arr).all():
                    return False

        return True

    except Exception:
        return False


# ============================================================
# PROCESSING
# ============================================================

def process_channel(
    unit: Dict,
    channel: int,
    input_path: Path,
) -> Dict:
    input_sha = sha256_file(input_path)
    npz_path, json_path = output_paths(unit)[channel]

    if validate_existing_output(
        npz_path,
        json_path,
        input_sha,
    ):
        return {
            "status": "SKIPPED_EXISTING_VALID",
            "npz_path": npz_path,
            "json_path": json_path,
            "input_sha256": input_sha,
            "quality": None,
        }

    parsed = parse_spec(input_path)

    frequency = parsed["frequency_hz"]
    re_values = parsed["re"]
    im_values = parsed["im"]

    z_magnitude = np.sqrt(
        np.square(re_values) + np.square(im_values)
    )
    z_phase_deg = np.degrees(
        np.arctan2(im_values, re_values)
    )

    if not (
        np.isfinite(z_magnitude).all()
        and np.isfinite(z_phase_deg).all()
    ):
        raise ValueError(
            f"{input_path}: derived impedance contains non-finite values"
        )

    arrays = {
        "frequency_hz": frequency,
        "re": re_values,
        "im": im_values,
        "z_magnitude": z_magnitude,
        "z_phase_deg": z_phase_deg,
    }

    metadata = {
        "protocol": PROTOCOL_VERSION,
        "generated_utc": utc_now(),
        "bz_id": unit["bz_id"],
        "subject_id": unit["subject_id"],
        "gesture_code": unit["gesture_code"],
        "gesture_name": unit["gesture_name"],
        "set_number": unit["set_number"],
        "channel": channel,
        "native_channel_name": parsed["channel_name"],
        "input_relative_path": str(
            input_path.relative_to(PROJECT_ROOT)
        ),
        "input_sha256": input_sha,
        "measurement_domain": "frequency",
        "point_count": int(EXPECTED_POINTS),
        "frequency_min_hz": float(frequency.min()),
        "frequency_max_hz": float(frequency.max()),
        "operations": [
            "native .spec parsing",
            "finite-value validation",
            "frequency-order validation",
            "impedance magnitude derivation",
            "impedance phase derivation",
        ],
        "raw_data_modified": False,
    }

    atomic_save_npz(npz_path, **arrays)
    write_json(json_path, metadata)

    return {
        "status": "PROCESSED",
        "npz_path": npz_path,
        "json_path": json_path,
        "input_sha256": input_sha,
        "quality": {
            "point_count": EXPECTED_POINTS,
            "frequency_min_hz": float(frequency.min()),
            "frequency_max_hz": float(frequency.max()),
            "z_magnitude_min": float(z_magnitude.min()),
            "z_magnitude_max": float(z_magnitude.max()),
            "z_magnitude_mean": float(z_magnitude.mean()),
            "z_phase_min_deg": float(z_phase_deg.min()),
            "z_phase_max_deg": float(z_phase_deg.max()),
        },
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    start = datetime.now()

    print("=" * 72)
    print("PRE3 — BIOIMPEDANCE PREPROCESSING")
    print(PROTOCOL_VERSION)
    print("=" * 72)

    if not RAW_BIOZ.exists():
        raise FileNotFoundError(
            f"Raw BioZ directory not found:\n{RAW_BIOZ}"
        )

    QC_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_BIOZ.mkdir(parents=True, exist_ok=True)

    print(f"RAW:      {RAW_BIOZ}")
    print(f"PROCESSED: {PROCESSED_BIOZ}")
    print(f"QC:       {QC_ROOT}")
    print()

    # --------------------------------------------------------
    # 1. DISCOVERY
    # --------------------------------------------------------
    spec_files = discover_all_spec_files()

    print(
        f"BioZ native .spec files discovered: "
        f"{len(spec_files)}"
    )

    if len(spec_files) != EXPECTED_SPEC_FILES:
        raise RuntimeError(
            f"Native .spec inventory mismatch: "
            f"{len(spec_files)}/{EXPECTED_SPEC_FILES}"
        )

    inventory = build_exact_inventory(spec_files)

    # --------------------------------------------------------
    # 2. EXACT PAIRING
    # --------------------------------------------------------
    units = pair_recording_units(inventory)

    print(
        f"BioZ recording units discovered: "
        f"{len(units)}"
    )

    if len(units) != EXPECTED_UNITS:
        raise RuntimeError(
            f"Recording-unit inventory mismatch: "
            f"{len(units)}/{EXPECTED_UNITS}"
        )

    # --------------------------------------------------------
    # 3. EXACT COVERAGE
    # --------------------------------------------------------
    expected_unit_keys = {
        (bz_id, gesture_code, set_number)
        for bz_id in BIOZ_CROSSWALK
        for gesture_code in GESTURE_CODES
        for set_number in range(
            1,
            EXPECTED_SETS_PER_GESTURE + 1,
        )
    }

    actual_unit_keys = {
        (
            u["bz_id"],
            u["gesture_code"],
            u["set_number"],
        )
        for u in units
    }

    if expected_unit_keys != actual_unit_keys:
        raise RuntimeError("Exact BioZ recording coverage failed")

    # --------------------------------------------------------
    # 4. PROCESS
    # --------------------------------------------------------
    status_rows = []
    quality_rows = []
    frequency_rows = []

    processed_now = 0
    skipped = 0
    failed = 0

    for i, unit in enumerate(units, start=1):
        print(
            f"[{i:04d}/{len(units):04d}] "
            f"{unit['unit_id']}"
        )

        for channel in (1, 2):
            input_path = (
                unit["channel_1_path"]
                if channel == 1
                else unit["channel_2_path"]
            )

            input_sha = sha256_file(input_path)

            try:
                result = process_channel(
                    unit,
                    channel,
                    input_path,
                )

                if result["status"] == "SKIPPED_EXISTING_VALID":
                    skipped += 1
                else:
                    processed_now += 1

                status_rows.append({
                    "unit_id": unit["unit_id"],
                    "bz_id": unit["bz_id"],
                    "subject_id": unit["subject_id"],
                    "gesture_code": unit["gesture_code"],
                    "gesture_name": unit["gesture_name"],
                    "set_number": unit["set_number"],
                    "channel": channel,
                    "input_path": str(
                        input_path.relative_to(PROJECT_ROOT)
                    ),
                    "input_sha256": input_sha,
                    "status": result["status"],
                    "error": "",
                })

                quality = result["quality"]

                if quality is None:
                    # Re-read existing metadata for QC.
                    _, json_path = output_paths(unit)[channel]
                    meta = json.loads(
                        json_path.read_text(
                            encoding="utf-8"
                        )
                    )

                    quality = {
                        "point_count": meta["point_count"],
                        "frequency_min_hz": meta[
                            "frequency_min_hz"
                        ],
                        "frequency_max_hz": meta[
                            "frequency_max_hz"
                        ],
                        "z_magnitude_min": np.nan,
                        "z_magnitude_max": np.nan,
                        "z_magnitude_mean": np.nan,
                        "z_phase_min_deg": np.nan,
                        "z_phase_max_deg": np.nan,
                    }

                quality_rows.append({
                    "unit_id": unit["unit_id"],
                    "bz_id": unit["bz_id"],
                    "subject_id": unit["subject_id"],
                    "gesture_code": unit["gesture_code"],
                    "set_number": unit["set_number"],
                    "channel": channel,
                    **quality,
                })

                frequency_rows.append({
                    "unit_id": unit["unit_id"],
                    "bz_id": unit["bz_id"],
                    "subject_id": unit["subject_id"],
                    "gesture_code": unit["gesture_code"],
                    "set_number": unit["set_number"],
                    "channel": channel,
                    "point_count": quality["point_count"],
                    "frequency_min_hz": quality[
                        "frequency_min_hz"
                    ],
                    "frequency_max_hz": quality[
                        "frequency_max_hz"
                    ],
                    "frequency_valid": True,
                })

            except Exception as exc:
                failed += 1

                status_rows.append({
                    "unit_id": unit["unit_id"],
                    "bz_id": unit["bz_id"],
                    "subject_id": unit["subject_id"],
                    "gesture_code": unit["gesture_code"],
                    "gesture_name": unit["gesture_name"],
                    "set_number": unit["set_number"],
                    "channel": channel,
                    "input_path": str(
                        input_path.relative_to(PROJECT_ROOT)
                    ),
                    "input_sha256": input_sha,
                    "status": "FAILED",
                    "error": str(exc),
                })

                print(
                    f"    ERROR channel {channel}: {exc}"
                )

    # --------------------------------------------------------
    # 5. WRITE QC
    # --------------------------------------------------------
    write_csv(
        QC_ROOT / "PRE3_recording_processing_status.csv",
        status_rows,
        [
            "unit_id",
            "bz_id",
            "subject_id",
            "gesture_code",
            "gesture_name",
            "set_number",
            "channel",
            "input_path",
            "input_sha256",
            "status",
            "error",
        ],
    )

    write_csv(
        QC_ROOT / "BIOIMPEDANCE_signal_quality.csv",
        quality_rows,
        [
            "unit_id",
            "bz_id",
            "subject_id",
            "gesture_code",
            "set_number",
            "channel",
            "point_count",
            "frequency_min_hz",
            "frequency_max_hz",
            "z_magnitude_min",
            "z_magnitude_max",
            "z_magnitude_mean",
            "z_phase_min_deg",
            "z_phase_max_deg",
        ],
    )

    write_csv(
        QC_ROOT / "frequency_QC.csv",
        frequency_rows,
        [
            "unit_id",
            "bz_id",
            "subject_id",
            "gesture_code",
            "set_number",
            "channel",
            "point_count",
            "frequency_min_hz",
            "frequency_max_hz",
            "frequency_valid",
        ],
    )

    # --------------------------------------------------------
    # 6. PARTICIPANT INVENTORY
    # --------------------------------------------------------
    participant_rows = []

    for bz_id, subject_id in BIOZ_CROSSWALK.items():
        unit_count = sum(
            1
            for u in units
            if u["bz_id"] == bz_id
        )
        participant_rows.append({
            "bz_id": bz_id,
            "subject_id": subject_id,
            "gestures_expected": EXPECTED_GESTURES,
            "sets_per_gesture_expected":
                EXPECTED_SETS_PER_GESTURE,
            "recording_units_expected":
                EXPECTED_GESTURES
                * EXPECTED_SETS_PER_GESTURE,
            "recording_units_discovered": unit_count,
            "coverage_status": (
                "PASS"
                if unit_count
                == EXPECTED_GESTURES
                * EXPECTED_SETS_PER_GESTURE
                else "FAIL"
            ),
        })

    write_csv(
        QC_ROOT / "PRE3_BIOZ_PARTICIPANT_INVENTORY.csv",
        participant_rows,
        [
            "bz_id",
            "subject_id",
            "gestures_expected",
            "sets_per_gesture_expected",
            "recording_units_expected",
            "recording_units_discovered",
            "coverage_status",
        ],
    )

    # --------------------------------------------------------
    # 7. EXACT RECORDING COVERAGE
    # --------------------------------------------------------
    coverage_rows = []

    for bz_id in BIOZ_CROSSWALK:
        for gesture_code in GESTURE_CODES:
            for set_number in range(
                1,
                EXPECTED_SETS_PER_GESTURE + 1,
            ):
                coverage_rows.append({
                    "bz_id": bz_id,
                    "subject_id": BIOZ_CROSSWALK[bz_id],
                    "gesture_code": gesture_code,
                    "gesture_name": GESTURE_CODES[
                        gesture_code
                    ],
                    "set_number": set_number,
                    "channel_1_present": True,
                    "channel_2_present": True,
                    "unit_complete": True,
                })

    write_csv(
        QC_ROOT / "PRE3_EXACT_RECORDING_COVERAGE.csv",
        coverage_rows,
        [
            "bz_id",
            "subject_id",
            "gesture_code",
            "gesture_name",
            "set_number",
            "channel_1_present",
            "channel_2_present",
            "unit_complete",
        ],
    )

    # --------------------------------------------------------
    # 8. CHANNEL SUMMARY
    # --------------------------------------------------------
    channel_summary = []

    for channel in (1, 2):
        rows = [
            r
            for r in quality_rows
            if r["channel"] == channel
        ]

        channel_summary.append({
            "channel": channel,
            "recording_units": len(rows),
            "expected_units": EXPECTED_UNITS,
            "all_points_100": all(
                r["point_count"] == EXPECTED_POINTS
                for r in rows
            ),
            "all_frequency_valid": all(
                r["frequency_max_hz"]
                > r["frequency_min_hz"]
                for r in rows
            ),
            "min_frequency_hz": min(
                r["frequency_min_hz"] for r in rows
            ) if rows else np.nan,
            "max_frequency_hz": max(
                r["frequency_max_hz"] for r in rows
            ) if rows else np.nan,
        })

    write_csv(
        QC_ROOT / "BIOIMPEDANCE_channel_quality_summary.csv",
        channel_summary,
        [
            "channel",
            "recording_units",
            "expected_units",
            "all_points_100",
            "all_frequency_valid",
            "min_frequency_hz",
            "max_frequency_hz",
        ],
    )

    # --------------------------------------------------------
    # 9. SUMMARY
    # --------------------------------------------------------
    elapsed = (
        datetime.now() - start
    ).total_seconds()

    complete = (
        len(spec_files) == EXPECTED_SPEC_FILES
        and len(units) == EXPECTED_UNITS
        and failed == 0
        and len(status_rows) == EXPECTED_SPEC_FILES
    )

    summary = {
        "protocol": PROTOCOL_VERSION,
        "generated_utc": utc_now(),
        "expected_participants": EXPECTED_PARTICIPANTS,
        "discovered_participants": len(BIOZ_CROSSWALK),
        "expected_recording_units": EXPECTED_UNITS,
        "discovered_recording_units": len(units),
        "expected_spec_files": EXPECTED_SPEC_FILES,
        "discovered_spec_files": len(spec_files),
        "expected_points_per_channel": EXPECTED_POINTS,
        "processed_now": processed_now,
        "skipped_existing_valid": skipped,
        "failed": failed,
        "complete": complete,
        "measurement_domain": "frequency",
        "raw_data_modified": False,
        "filtering_applied": False,
        "resampling_applied": False,
        "interpolation_applied": False,
        "smoothing_applied": False,
        "normalization_applied": False,
        "feature_fitting_applied": False,
        "ml_processing_applied": False,
        "cross_modal_synchronization_applied": False,
        "runtime_seconds": elapsed,
    }

    write_csv(
        QC_ROOT / "PRE3_processing_summary.csv",
        [summary],
        list(summary.keys()),
    )

    readme = f"""PRE3 Bioimpedance Processing QC
Protocol: {PROTOCOL_VERSION}

Dataset scope:
- 18 BioZ participants
- 7 gestures
- 15 sets per gesture
- 1,890 recording units
- 3,780 native .spec channel files
- 2 channels per recording unit
- 100 frequency points per channel

Native measurement:
frequency[Hz], Re, Im

Frequency sweep:
approximately 10 Hz to 1 MHz, as present in the native files.

Processing:
- native .spec parsing
- structural/header validation
- finite-value validation
- frequency uniqueness/order validation
- impedance magnitude derivation
- impedance phase derivation

Not performed:
- time-axis reconstruction
- duration inference
- filtering
- resampling
- interpolation
- smoothing
- normalization
- ML/feature fitting
- cross-modal synchronization

Raw .spec files were not modified.

Final status:
{'COMPLETE' if complete else 'INCOMPLETE'}
"""

    atomic_write_bytes(
        QC_ROOT / "PRE3_QC_README.txt",
        readme.encode("utf-8"),
    )

    write_json(
        QC_ROOT / "PRE3_FINAL_REPORT.json",
        summary,
    )

    print()
    print("=" * 72)
    print("PRE3 FINAL SUMMARY")
    print("=" * 72)
    print(
        f"BioZ participants       "
        f"{len(BIOZ_CROSSWALK)}/{EXPECTED_PARTICIPANTS}"
    )
    print(
        f"Recording units         "
        f"{len(units)}/{EXPECTED_UNITS}"
    )
    print(
        f"Native .spec files      "
        f"{len(spec_files)}/{EXPECTED_SPEC_FILES}"
    )
    print(
        f"Processed now            {processed_now}"
    )
    print(
        f"Skipped valid            {skipped}"
    )
    print(
        f"Failed                   {failed}"
    )
    print(
        f"Points/channel           {EXPECTED_POINTS}"
    )
    print(
        f"Raw data modified        False"
    )
    print(
        f"PRE3 STATUS: "
        f"{'COMPLETE' if complete else 'FAILED'}"
    )
    print(
        f"Runtime                  {elapsed:.1f} s"
    )
    print("=" * 72)

    if not complete:
        raise RuntimeError(
            "PRE3 STATUS: FAILED"
        )


if __name__ == "__main__":
    main()
