#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================================
MODALITY VIEWS — DATASET V1.0 FINAL
====================================================================

Purpose
-------
Create publication-ready modality manifests from the already
validated raw and processed dataset.

This block is MANIFEST-ONLY.

It does NOT:
    - copy signal files
    - move signal files
    - rename signal files
    - modify raw data
    - modify processed signal data
    - rerun preprocessing
    - create annotations
    - create fusion data
    - perform feature extraction
    - perform machine learning

Final output structure
----------------------

02_MODALITY_VIEWS/
├── EEG/
│   ├── EEG_MOTOR_EXECUTION_manifest.csv
│   └── EEG_MOTOR_IMAGERY_manifest.csv
│
├── EMG/
│   └── EMG_manifest.csv
│
├── BIOIMPEDANCE/
│   └── BIOIMPEDANCE_manifest.csv
│
└── MODALITY_VIEWS_summary.csv

Design principles
-----------------
1. Raw paths are preserved exactly as stored.
2. Processed paths are preserved exactly as stored.
3. Raw files are matched by authoritative recording keys.
4. BioZ raw matching is indexed from the actual raw .spec paths;
   no assumed raw directory naming is required.
5. Parent gesture folders are preferred over filename parsing.
6. Filename parsing is used only as a fallback.
7. Duplicate raw keys are detected and reported instead of silently
   overwriting one path with another.
8. Processed files are never copied, moved, renamed, or rewritten.
9. The script exits non-zero unless every expected modality manifest
   passes its inventory and raw-link checks.
10. Only the four manifest files and one summary file are generated
    by this script.

Expected frozen inventory
-------------------------
Motor execution EEG : 840
Motor imagery EEG   : 525
Motor execution EMG : 840
BioZ processed NPZ  : 3780
BioZ raw .spec      : 3780
"""

from __future__ import annotations

import csv
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


# ====================================================================
# 1. CONFIGURATION
# ====================================================================

ROOT = Path(
    "/mnt/f/Faruk/OFS_Paper_Work/Data_Set_Paper_Work/"
    "EEG_EMG_BIOZ_DATASET"
)

RAW = ROOT / "01_RAW_DATA"
PROCESSED = ROOT / "06_PROCESSED_DATA"

RAW_EXEC = RAW / "EMG_EEG_SYNCHRONIZED"
RAW_MI = RAW / "EEG_MOTOR_IMAGERY"
RAW_BIOZ = RAW / "BIOIMPEDANCE"

PROC_EEG_EXEC = PROCESSED / "EEG_MOTOR_EXECUTION"
PROC_EEG_MI = PROCESSED / "EEG_MOTOR_IMAGERY"
PROC_EMG = PROCESSED / "EMG"
PROC_BIOZ = PROCESSED / "BIOIMPEDANCE"

MODALITY_VIEWS = ROOT / "02_MODALITY_VIEWS"

EEG_VIEW = MODALITY_VIEWS / "EEG"
EMG_VIEW = MODALITY_VIEWS / "EMG"
BIOZ_VIEW = MODALITY_VIEWS / "BIOIMPEDANCE"

PROTOCOL_VERSION = "MODALITY-VIEWS-DATASET-V1.0-FINAL"

DATASET_VERSION = "V1.0"

EXPECTED_EXECUTION = 840
EXPECTED_MI = 525
EXPECTED_EMG = 840
EXPECTED_BIOZ_NPZ = 3780
EXPECTED_BIOZ_SPEC = 3780


# ====================================================================
# 2. DATASET DEFINITIONS
# ====================================================================

GESTURE_MAP = {
    "HOC": "Hand_Open_Close",
    "TFM": "Thumb_Finger_Movement",
    "IFM": "Index_Finger_Movement",
    "MFM": "Middle_Finger_Movement",
    "RFM": "Ring_Finger_Movement",
    "LFM": "Little_Finger_Movement",
    "PH": "Pen_Holding",
}

SET_IDS = {"A", "B", "C"}

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


# ====================================================================
# 3. GENERAL UTILITIES
# ====================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def relative_path(path: Path | None) -> str:
    if path is None:
        return ""
    return str(path.relative_to(ROOT)).replace("\\", "/")


def file_exists(path: Path | None) -> str:
    return "YES" if path is not None and path.is_file() else "NO"


def pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def normalize_text(value: str) -> str:
    """Normalize a folder/file label for robust comparison."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def atomic_write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    """
    Atomically replace one manifest/summary file.

    This prevents a partially written CSV if the process is interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

        os.replace(tmp_name, path)

    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def list_files(
    root: Path,
    suffix: str,
    exclude_parts: Iterable[str] = (),
) -> list[Path]:
    excluded = set(exclude_parts)

    if not root.exists():
        return []

    return sorted(
        p for p in root.rglob(f"*{suffix}")
        if p.is_file()
        and not any(part in excluded for part in p.parts)
    )


# ====================================================================
# 4. GESTURE PARSING
# ====================================================================

GESTURE_NAME_TO_CODE = {
    normalize_text(name): code
    for code, name in GESTURE_MAP.items()
}

# Additional normalized aliases that occur naturally in filenames.
GESTURE_ALIASES = {
    "hand_open_close": "HOC",
    "thumb_finger_movement": "TFM",
    "index_finger_movement": "IFM",
    "middle_finger_movement": "MFM",
    "ring_finger_movement": "RFM",
    "little_finger_movement": "LFM",
    "pen_holding": "PH",
    "hand_open_close_hoc": "HOC",
    "thumb_finger_movement_tfm": "TFM",
    "index_finger_movement_ifm": "IFM",
    "middle_finger_movement_mfm": "MFM",
    "ring_finger_movement_rfm": "RFM",
    "little_finger_movement_lfm": "LFM",
    "pen_holding_ph": "PH",
}


def code_from_label(label: str) -> str | None:
    """
    Resolve a gesture from a directory/file label.

    Order:
      1. exact normalized full gesture name
      2. normalized alias
      3. exact code
      4. tokenized code
      5. full-name substring
    """
    norm = normalize_text(label)

    if norm in GESTURE_NAME_TO_CODE:
        return GESTURE_NAME_TO_CODE[norm]

    if norm in GESTURE_ALIASES:
        return GESTURE_ALIASES[norm]

    if norm.upper() in GESTURE_MAP:
        return norm.upper()

    for code in GESTURE_MAP:
        if re.search(
            rf"(?:^|_){re.escape(code.lower())}(?:_|$)",
            norm,
        ):
            return code

    for code, full_name in GESTURE_MAP.items():
        full_norm = normalize_text(full_name)
        if full_norm and full_norm in norm:
            return code

    return None


def gesture_from_path(
    path: Path,
) -> tuple[str | None, str | None]:
    """
    Resolve gesture code/name.

    The closest parent directory is authoritative when it resolves
    cleanly to a known gesture. Filename parsing is fallback only.
    """
    for parent in reversed(path.parents):
        code = code_from_label(parent.name)
        if code is not None:
            return code, GESTURE_MAP[code]

    code = code_from_label(path.stem)
    if code is not None:
        return code, GESTURE_MAP[code]

    return None, None


# ====================================================================
# 5. EXECUTION / MI PARSING
# ====================================================================

def execution_subject(path: Path) -> str | None:
    """
    Supports both:
        S01
        Subject_01
    in filenames or directory names.
    """
    candidates = [path.stem] + [p.name for p in path.parents]

    for text in candidates:
        match = re.search(
            r"(?:^|_)(?:S|Subject_)(\d{1,2})(?:_|$)",
            text,
            re.IGNORECASE,
        )

        if match:
            number = int(match.group(1))
            if 1 <= number <= 99:
                return f"Subject_{number:02d}"

    return None


def mi_identifier(path: Path) -> str | None:
    """
    Resolve MI identifier from MI_01 style names.

    MI IDs remain independent identifiers. They are NOT mapped to
    canonical Subject IDs in this modality-view stage.
    """
    candidates = [path.stem] + [p.name for p in path.parents]

    for text in candidates:
        match = re.search(
            r"(?:^|_)MI_(\d{1,2})(?:_|$)",
            text,
            re.IGNORECASE,
        )

        if match:
            number = int(match.group(1))
            if 1 <= number <= 99:
                return f"MI_{number:02d}"

    return None


def set_id_from_path(path: Path) -> str | None:
    """
    Supports:
        SetA / SetB / SetC
        set_A / set_B / set_C
        set_01 / set_02 ... for BioZ
    """
    for text in [path.stem] + [p.name for p in path.parents]:
        match = re.search(
            r"(?:^|[_-])Set[_-]?([ABC])(?:[_-]|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()

    return None


# ====================================================================
# 6. EXECUTION RAW INDEX
# ====================================================================

def build_execution_raw_index() -> tuple[dict, list[str]]:
    """
    Build:
        (subject, gesture_code, SetA/B/C) -> raw CSV

    Duplicate keys are reported rather than silently overwritten.
    """
    index: dict[tuple, Path] = {}
    duplicates: dict[tuple, list[Path]] = defaultdict(list)
    malformed: list[str] = []

    for path in list_files(RAW_EXEC, ".csv"):
        subject = execution_subject(path)
        gesture_code, _ = gesture_from_path(path)
        recording_set = set_id_from_path(path)

        if (
            subject is None
            or gesture_code is None
            or recording_set is None
        ):
            malformed.append(relative_path(path))
            continue

        key = (subject, gesture_code, recording_set)

        if key in index:
            duplicates[key].append(path)
        else:
            index[key] = path

    duplicate_messages = []

    for key, paths in sorted(duplicates.items()):
        all_paths = [index[key], *paths]
        duplicate_messages.append(
            f"{key}: " + " | ".join(
                relative_path(p) for p in all_paths
            )
        )

    return index, duplicate_messages + malformed


# ====================================================================
# 7. EEG MOTOR EXECUTION
# ====================================================================

def create_eeg_execution_manifest() -> dict:
    print("\n" + "=" * 72)
    print("EEG MOTOR EXECUTION MODALITY VIEW")
    print("=" * 72)

    processed_files = list_files(
        PROC_EEG_EXEC,
        ".npz",
    )

    raw_index, index_issues = build_execution_raw_index()

    rows = []
    missing_raw = []
    malformed_processed = []
    duplicate_keys = []

    seen_keys: set[tuple] = set()

    for path in processed_files:
        subject = execution_subject(path)
        gesture_code, gesture_name = gesture_from_path(path)
        recording_set = set_id_from_path(path)

        key = (subject, gesture_code, recording_set)

        if None in key:
            malformed_processed.append(relative_path(path))

        if key in seen_keys:
            duplicate_keys.append(
                f"{key}: {relative_path(path)}"
            )
        seen_keys.add(key)

        raw_path = raw_index.get(key)

        if raw_path is None:
            missing_raw.append(relative_path(path))

        rows.append({
            "dataset_version": DATASET_VERSION,
            "modality": "EEG",
            "protocol": "MOTOR_EXECUTION",
            "subject_id": subject or "",
            "gesture_code": gesture_code or "",
            "gesture_name": gesture_name or "",
            "set_id": recording_set or "",
            "sampling_rate_hz": 125,
            "n_channels": 13,
            "n_samples": 5625,
            "raw_file": relative_path(raw_path),
            "processed_file": relative_path(path),
            "raw_file_exists": file_exists(raw_path),
            "processed_file_exists": file_exists(path),
            "processing_stage": "PRE1",
            "synchronization": "SAMPLE_LEVEL_WITH_EMG",
            "raw_data_modified": "NO",
        })

    # A manifest passes only when inventory, keys and raw links all pass.
    ok = (
        len(processed_files) == EXPECTED_EXECUTION
        and len(rows) == EXPECTED_EXECUTION
        and not missing_raw
        and not malformed_processed
        and not duplicate_keys
        and not index_issues
    )

    output = EEG_VIEW / "EEG_MOTOR_EXECUTION_manifest.csv"

    fields = [
        "dataset_version",
        "modality",
        "protocol",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_id",
        "sampling_rate_hz",
        "n_channels",
        "n_samples",
        "raw_file",
        "processed_file",
        "raw_file_exists",
        "processed_file_exists",
        "processing_stage",
        "synchronization",
        "raw_data_modified",
    ]

    atomic_write_csv(output, rows, fields)

    print(f"Processed EEG files : {len(processed_files)}")
    print(f"Manifest rows        : {len(rows)}")
    print(f"Missing raw links    : {len(missing_raw)}")
    print(f"Malformed keys       : {len(malformed_processed)}")
    print(f"Duplicate keys       : {len(duplicate_keys)}")
    print(f"Raw-index issues     : {len(index_issues)}")
    print(f"STATUS               : {pass_fail(ok)}")

    return {
        "count": len(rows),
        "status": pass_fail(ok),
        "missing_raw": missing_raw,
        "issues": index_issues,
    }


# ====================================================================
# 8. EEG MOTOR IMAGERY
# ====================================================================

def build_mi_raw_index() -> tuple[dict, list[str]]:
    """
    Build:
        (MI_ID, gesture_code, SetA/B/C) -> raw CSV

    Parent gesture directory is preferred. Filename is fallback.
    """
    index: dict[tuple, Path] = {}
    duplicates: dict[tuple, list[Path]] = defaultdict(list)
    malformed: list[str] = []

    for path in list_files(RAW_MI, ".csv"):
        identifier = mi_identifier(path)
        gesture_code, _ = gesture_from_path(path)
        recording_set = set_id_from_path(path)

        if (
            identifier is None
            or gesture_code is None
            or recording_set is None
        ):
            malformed.append(relative_path(path))
            continue

        key = (identifier, gesture_code, recording_set)

        if key in index:
            duplicates[key].append(path)
        else:
            index[key] = path

    issues = []

    for key, paths in sorted(duplicates.items()):
        all_paths = [index[key], *paths]
        issues.append(
            f"DUPLICATE {key}: " + " | ".join(
                relative_path(p) for p in all_paths
            )
        )

    issues.extend(
        f"MALFORMED: {x}" for x in malformed
    )

    return index, issues


def create_eeg_mi_manifest() -> dict:
    print("\n" + "=" * 72)
    print("EEG MOTOR IMAGERY MODALITY VIEW")
    print("=" * 72)

    processed_files = list_files(
        PROC_EEG_MI,
        ".npz",
    )

    raw_index, index_issues = build_mi_raw_index()

    rows = []
    missing_raw = []
    malformed_processed = []
    duplicate_keys = []
    seen_keys: set[tuple] = set()

    for path in processed_files:
        identifier = mi_identifier(path)
        gesture_code, gesture_name = gesture_from_path(path)
        recording_set = set_id_from_path(path)

        key = (identifier, gesture_code, recording_set)

        if None in key:
            malformed_processed.append(relative_path(path))

        if key in seen_keys:
            duplicate_keys.append(
                f"{key}: {relative_path(path)}"
            )
        seen_keys.add(key)

        raw_path = raw_index.get(key)

        if raw_path is None:
            missing_raw.append(relative_path(path))

        rows.append({
            "dataset_version": DATASET_VERSION,
            "modality": "EEG",
            "protocol": "MOTOR_IMAGERY",
            "mi_id": identifier or "",
            "canonical_subject_id": "",
            "mapping_status": "AVAILABILITY_CONFIRMED_ID_PENDING",
            "gesture_code": gesture_code or "",
            "gesture_name": gesture_name or "",
            "set_id": recording_set or "",
            "sampling_rate_hz": 125,
            "n_channels": 13,
            "n_samples": 5625,
            "raw_file": relative_path(raw_path),
            "processed_file": relative_path(path),
            "raw_file_exists": file_exists(raw_path),
            "processed_file_exists": file_exists(path),
            "processing_stage": "PRE1",
            "synchronization": "EEG_ONLY",
            "raw_data_modified": "NO",
        })

    actual_mi_ids = {
        row["mi_id"]
        for row in rows
        if row["mi_id"]
    }

    ok = (
        len(processed_files) == EXPECTED_MI
        and len(rows) == EXPECTED_MI
        and len(actual_mi_ids) == 25
        and not missing_raw
        and not malformed_processed
        and not duplicate_keys
        and not index_issues
    )

    output = EEG_VIEW / "EEG_MOTOR_IMAGERY_manifest.csv"

    fields = [
        "dataset_version",
        "modality",
        "protocol",
        "mi_id",
        "canonical_subject_id",
        "mapping_status",
        "gesture_code",
        "gesture_name",
        "set_id",
        "sampling_rate_hz",
        "n_channels",
        "n_samples",
        "raw_file",
        "processed_file",
        "raw_file_exists",
        "processed_file_exists",
        "processing_stage",
        "synchronization",
        "raw_data_modified",
    ]

    atomic_write_csv(output, rows, fields)

    print(f"Processed EEG files : {len(processed_files)}")
    print(f"Manifest rows        : {len(rows)}")
    print(f"MI IDs               : {len(actual_mi_ids)}")
    print(f"Missing raw links    : {len(missing_raw)}")
    print(f"Malformed keys       : {len(malformed_processed)}")
    print(f"Duplicate keys       : {len(duplicate_keys)}")
    print(f"Raw-index issues     : {len(index_issues)}")
    print(f"STATUS               : {pass_fail(ok)}")

    return {
        "count": len(rows),
        "status": pass_fail(ok),
        "mi_ids": sorted(actual_mi_ids),
        "missing_raw": missing_raw,
        "issues": index_issues,
    }


# ====================================================================
# 9. EMG
# ====================================================================

def create_emg_manifest() -> dict:
    print("\n" + "=" * 72)
    print("EMG MODALITY VIEW")
    print("=" * 72)

    # PRE2_QC is excluded explicitly. Only processed NPZ files are
    # eligible for the EMG modality manifest.
    processed_files = list_files(
        PROC_EMG,
        ".npz",
        exclude_parts={"PRE2_QC"},
    )

    raw_index, index_issues = build_execution_raw_index()

    rows = []
    missing_raw = []
    malformed_processed = []
    duplicate_keys = []
    seen_keys: set[tuple] = set()

    for path in processed_files:
        subject = execution_subject(path)
        gesture_code, gesture_name = gesture_from_path(path)
        recording_set = set_id_from_path(path)

        key = (subject, gesture_code, recording_set)

        if None in key:
            malformed_processed.append(relative_path(path))

        if key in seen_keys:
            duplicate_keys.append(
                f"{key}: {relative_path(path)}"
            )
        seen_keys.add(key)

        raw_path = raw_index.get(key)

        if raw_path is None:
            missing_raw.append(relative_path(path))

        rows.append({
            "dataset_version": DATASET_VERSION,
            "modality": "EMG",
            "protocol": "MOTOR_EXECUTION",
            "subject_id": subject or "",
            "gesture_code": gesture_code or "",
            "gesture_name": gesture_name or "",
            "set_id": recording_set or "",
            "sampling_rate_hz": 125,
            "n_channels": 3,
            "n_samples": 5625,
            "raw_file": relative_path(raw_path),
            "processed_file": relative_path(path),
            "raw_file_exists": file_exists(raw_path),
            "processed_file_exists": file_exists(path),
            "processing_stage": "PRE2",
            "synchronization": "SAMPLE_LEVEL_WITH_EEG",
            "raw_data_modified": "NO",
        })

    ok = (
        len(processed_files) == EXPECTED_EMG
        and len(rows) == EXPECTED_EMG
        and not missing_raw
        and not malformed_processed
        and not duplicate_keys
        and not index_issues
    )

    output = EMG_VIEW / "EMG_manifest.csv"

    fields = [
        "dataset_version",
        "modality",
        "protocol",
        "subject_id",
        "gesture_code",
        "gesture_name",
        "set_id",
        "sampling_rate_hz",
        "n_channels",
        "n_samples",
        "raw_file",
        "processed_file",
        "raw_file_exists",
        "processed_file_exists",
        "processing_stage",
        "synchronization",
        "raw_data_modified",
    ]

    atomic_write_csv(output, rows, fields)

    print(f"Processed EMG files : {len(processed_files)}")
    print(f"Manifest rows        : {len(rows)}")
    print(f"Missing raw links    : {len(missing_raw)}")
    print(f"Malformed keys       : {len(malformed_processed)}")
    print(f"Duplicate keys       : {len(duplicate_keys)}")
    print(f"Raw-index issues     : {len(index_issues)}")
    print(f"STATUS               : {pass_fail(ok)}")

    return {
        "count": len(rows),
        "status": pass_fail(ok),
        "missing_raw": missing_raw,
        "issues": index_issues,
    }


# ====================================================================
# 10. BIOIMPEDANCE RAW INDEX — AUTHORITATIVE FIX
# ====================================================================

def bz_id_from_path(path: Path) -> str | None:
    for part in path.parts:
        match = re.fullmatch(
            r"BZ_(\d{1,2})",
            part,
            re.IGNORECASE,
        )
        if match:
            return f"BZ_{int(match.group(1)):02d}"

    for text in [path.stem] + [p.name for p in path.parents]:
        match = re.search(
            r"(?:^|_)BZ_(\d{1,2})(?:_|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            return f"BZ_{int(match.group(1)):02d}"

    return None


def bioz_channel_from_path(path: Path) -> int | None:
    for part in [p.name for p in path.parents] + [path.stem]:
        match = re.fullmatch(
            r"Channel[_-]?(\d+)",
            part,
            re.IGNORECASE,
        )
        if match:
            return int(match.group(1))

    return None


def bioz_measurement_set_from_path(path: Path) -> int | None:
    candidates = [path.stem] + [p.name for p in path.parents]

    for text in candidates:
        match = re.search(
            r"(?:^|[_-])set[_-]?(\d+)(?:[_-]|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            return int(match.group(1))

    return None


def build_bioz_raw_index() -> tuple[dict, list[str]]:
    """
    AUTHORITATIVE BioZ lookup.

    Key:
        (BZ_ID, gesture_code, channel, measurement_set)

    The actual raw filesystem is scanned and indexed first.
    No fixed folder spelling such as BZ_01_HOC is assumed.
    """
    index: dict[tuple, Path] = {}
    duplicates: dict[tuple, list[Path]] = defaultdict(list)
    malformed: list[str] = []

    raw_specs = list_files(RAW_BIOZ, ".spec")

    for path in raw_specs:
        bz_id = bz_id_from_path(path)
        gesture_code, _ = gesture_from_path(path)
        channel = bioz_channel_from_path(path)
        measurement_set = bioz_measurement_set_from_path(path)

        if (
            bz_id is None
            or gesture_code is None
            or channel is None
            or measurement_set is None
        ):
            malformed.append(relative_path(path))
            continue

        key = (
            bz_id,
            gesture_code,
            channel,
            measurement_set,
        )

        if key in index:
            duplicates[key].append(path)
        else:
            index[key] = path

    issues = []

    for key, paths in sorted(duplicates.items()):
        all_paths = [index[key], *paths]
        issues.append(
            f"DUPLICATE {key}: " + " | ".join(
                relative_path(p) for p in all_paths
            )
        )

    issues.extend(
        f"MALFORMED: {x}" for x in malformed
    )

    return index, issues


# ====================================================================
# 11. BIOIMPEDANCE MANIFEST
# ====================================================================

def create_bioz_manifest() -> dict:
    print("\n" + "=" * 72)
    print("BIOIMPEDANCE MODALITY VIEW")
    print("=" * 72)

    processed_files = list_files(
        PROC_BIOZ,
        ".npz",
        exclude_parts={"PRE3_QC"},
    )

    raw_files = list_files(
        RAW_BIOZ,
        ".spec",
    )

    raw_index, raw_index_issues = build_bioz_raw_index()

    rows = []
    missing_raw = []
    malformed_processed = []
    duplicate_keys = []
    seen_keys: set[tuple] = set()

    for path in processed_files:
        bz_id = bz_id_from_path(path)
        subject = BIOZ_CROSSWALK.get(bz_id or "")

        gesture_code, gesture_name = gesture_from_path(path)
        channel = bioz_channel_from_path(path)
        measurement_set = bioz_measurement_set_from_path(path)

        key = (
            bz_id,
            gesture_code,
            channel,
            measurement_set,
        )

        if None in key:
            malformed_processed.append(relative_path(path))

        if key in seen_keys:
            duplicate_keys.append(
                f"{key}: {relative_path(path)}"
            )
        seen_keys.add(key)

        raw_path = raw_index.get(key)

        if raw_path is None:
            missing_raw.append(relative_path(path))

        rows.append({
            "dataset_version": DATASET_VERSION,
            "modality": "BIOIMPEDANCE",
            "protocol": "MOTOR_EXECUTION",
            "bioz_id": bz_id or "",
            "canonical_subject_id": subject or "",
            "gesture_code": gesture_code or "",
            "gesture_name": gesture_name or "",
            "measurement_set": (
                measurement_set
                if measurement_set is not None
                else ""
            ),
            "channel": (
                channel
                if channel is not None
                else ""
            ),
            "measurement_domain": "FREQUENCY_DOMAIN",
            "points_per_channel": 100,
            "raw_file": relative_path(raw_path),
            "processed_file": relative_path(path),
            "raw_file_exists": file_exists(raw_path),
            "processed_file_exists": file_exists(path),
            "processing_stage": "PRE3",
            "synchronization": "INDEPENDENT_ACQUISITION",
            "raw_data_modified": "NO",
        })

    # Validate the frozen BioZ structure itself.
    expected_bz_ids = set(BIOZ_CROSSWALK)
    actual_bz_ids = {
        row["bioz_id"]
        for row in rows
        if row["bioz_id"]
    }

    channel_counts = defaultdict(int)
    unit_keys = set()

    for row in rows:
        if row["bioz_id"] and row["gesture_code"] and row["measurement_set"]:
            unit_keys.add(
                (
                    row["bioz_id"],
                    row["gesture_code"],
                    row["measurement_set"],
                )
            )

        if row["bioz_id"] and row["channel"] != "":
            channel_counts[
                (
                    row["bioz_id"],
                    row["gesture_code"],
                    row["channel"],
                )
            ] += 1

    exact_units = (
        len(unit_keys) == EXPECTED_BIOZ_NPZ // 2
        and all(
            count == 15
            for count in channel_counts.values()
        )
    )

    ok = (
        len(processed_files) == EXPECTED_BIOZ_NPZ
        and len(raw_files) == EXPECTED_BIOZ_SPEC
        and len(rows) == EXPECTED_BIOZ_NPZ
        and len(actual_bz_ids) == len(expected_bz_ids)
        and not missing_raw
        and not malformed_processed
        and not duplicate_keys
        and not raw_index_issues
        and exact_units
        and all(
            row["canonical_subject_id"]
            for row in rows
        )
    )

    output = BIOZ_VIEW / "BIOIMPEDANCE_manifest.csv"

    fields = [
        "dataset_version",
        "modality",
        "protocol",
        "bioz_id",
        "canonical_subject_id",
        "gesture_code",
        "gesture_name",
        "measurement_set",
        "channel",
        "measurement_domain",
        "points_per_channel",
        "raw_file",
        "processed_file",
        "raw_file_exists",
        "processed_file_exists",
        "processing_stage",
        "synchronization",
        "raw_data_modified",
    ]

    atomic_write_csv(output, rows, fields)

    print(f"Processed BioZ NPZ   : {len(processed_files)}")
    print(f"Raw BioZ .spec       : {len(raw_files)}")
    print(f"Manifest rows        : {len(rows)}")
    print(f"BZ IDs               : {len(actual_bz_ids)}")
    print(f"Missing raw links    : {len(missing_raw)}")
    print(f"Malformed keys       : {len(malformed_processed)}")
    print(f"Duplicate keys       : {len(duplicate_keys)}")
    print(f"Raw-index issues     : {len(raw_index_issues)}")
    print(f"Exact 15/channel     : {pass_fail(exact_units)}")
    print(f"STATUS               : {pass_fail(ok)}")

    return {
        "count": len(rows),
        "status": pass_fail(ok),
        "missing_raw": missing_raw,
        "issues": raw_index_issues,
    }


# ====================================================================
# 12. SUMMARY
# ====================================================================

def create_summary(
    eeg_exec: dict,
    eeg_mi: dict,
    emg: dict,
    bioz: dict,
) -> None:
    rows = [
        {
            "modality": "EEG_MOTOR_EXECUTION",
            "manifest": relative_path(
                EEG_VIEW / "EEG_MOTOR_EXECUTION_manifest.csv"
            ),
            "rows": eeg_exec["count"],
            "expected_rows": EXPECTED_EXECUTION,
            "status": eeg_exec["status"],
        },
        {
            "modality": "EEG_MOTOR_IMAGERY",
            "manifest": relative_path(
                EEG_VIEW / "EEG_MOTOR_IMAGERY_manifest.csv"
            ),
            "rows": eeg_mi["count"],
            "expected_rows": EXPECTED_MI,
            "status": eeg_mi["status"],
        },
        {
            "modality": "EMG",
            "manifest": relative_path(
                EMG_VIEW / "EMG_manifest.csv"
            ),
            "rows": emg["count"],
            "expected_rows": EXPECTED_EMG,
            "status": emg["status"],
        },
        {
            "modality": "BIOIMPEDANCE",
            "manifest": relative_path(
                BIOZ_VIEW / "BIOIMPEDANCE_manifest.csv"
            ),
            "rows": bioz["count"],
            "expected_rows": EXPECTED_BIOZ_NPZ,
            "status": bioz["status"],
        },
    ]

    atomic_write_csv(
        MODALITY_VIEWS / "MODALITY_VIEWS_summary.csv",
        rows,
        [
            "modality",
            "manifest",
            "rows",
            "expected_rows",
            "status",
        ],
    )


# ====================================================================
# 13. OUTPUT DIRECTORY SAFETY
# ====================================================================

def validate_required_paths() -> None:
    required = [
        RAW_EXEC,
        RAW_MI,
        RAW_BIOZ,
        PROC_EEG_EXEC,
        PROC_EEG_MI,
        PROC_EMG,
        PROC_BIOZ,
    ]

    missing = [str(p) for p in required if not p.is_dir()]

    if missing:
        print("\nERROR: Required dataset path(s) not found:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)


def create_output_directories() -> None:
    """
    Create only the modality-view directories required by this block.
    """
    EEG_VIEW.mkdir(parents=True, exist_ok=True)
    EMG_VIEW.mkdir(parents=True, exist_ok=True)
    BIOZ_VIEW.mkdir(parents=True, exist_ok=True)


# ====================================================================
# 14. MAIN
# ====================================================================

def main() -> None:
    print()
    print("=" * 72)
    print("MODALITY VIEWS — DATASET V1.0 FINAL")
    print("=" * 72)
    print(f"ROOT     : {ROOT}")
    print(f"VERSION  : {PROTOCOL_VERSION}")
    print(f"START    : {now_iso()}")
    print("=" * 72)

    validate_required_paths()
    create_output_directories()

    eeg_exec = create_eeg_execution_manifest()
    eeg_mi = create_eeg_mi_manifest()
    emg = create_emg_manifest()
    bioz = create_bioz_manifest()

    create_summary(
        eeg_exec,
        eeg_mi,
        emg,
        bioz,
    )

    results = {
        "EEG_MOTOR_EXECUTION": eeg_exec["status"],
        "EEG_MOTOR_IMAGERY": eeg_mi["status"],
        "EMG": emg["status"],
        "BIOIMPEDANCE": bioz["status"],
    }

    overall = all(
        value == "PASS"
        for value in results.values()
    )

    print()
    print("=" * 72)
    print("MODALITY VIEWS FINAL SUMMARY")
    print("=" * 72)

    for name, value in results.items():
        print(f"{name:30s}: {value}")

    print("-" * 72)
    print("RAW DATA MODIFICATION      : False")
    print("SIGNAL FILES COPIED        : False")
    print("SIGNAL FILES MOVED         : False")
    print("SIGNAL FILES RENAMED       : False")
    print("PREPROCESSING RERUN        : False")
    print("ANNOTATIONS CREATED        : False")
    print("FUSION DATA CREATED        : False")
    print("FEATURE EXTRACTION         : False")
    print("MACHINE LEARNING           : False")
    print("-" * 72)
    print(
        "MODALITY VIEWS STATUS      : "
        f"{'PASS' if overall else 'FAIL'}"
    )
    print(f"OUTPUT                     : {MODALITY_VIEWS}")
    print("=" * 72)

    if not overall:
        print(
            "\nManifest generation completed, "
            "but validation did not pass."
        )
        print(
            "No raw or processed signal files were modified."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
