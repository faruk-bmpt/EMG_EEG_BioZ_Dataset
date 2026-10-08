#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================================
GLOBAL DATASET QC — DATASET V1.0
====================================================================

Purpose
-------
Global integrity and cross-modality consistency validation after:

    DATA1  -> Dataset inventory
    DATA2  -> Raw-data QA
    PRE1   -> EEG preprocessing
    PRE2   -> EMG preprocessing
    PRE3   -> Bioimpedance preprocessing

This block ONLY validates existing files.

It does NOT:
    - modify raw data
    - modify processed data
    - rerun preprocessing
    - create annotations
    - create fusion data
    - perform feature extraction
    - perform machine learning

====================================================================
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


# ====================================================================
# 1. CONFIGURATION
# ====================================================================

# Resolve the dataset root from the script location so the same canonical
# script works on the frozen Windows dataset layout without hard-coded
# Linux/WSL paths.
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

if (SCRIPT_DIR / "01_RAW_DATA").exists():
    ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "01_RAW_DATA").exists():
    ROOT = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        "Cannot locate dataset root. Expected 01_RAW_DATA under the "
        "script directory or its parent."
    )

RAW = ROOT / "01_RAW_DATA"
PROCESSED = ROOT / "06_PROCESSED_DATA"
METADATA = ROOT / "03_METADATA"
QC_DIR = ROOT / "05_QC"

EXEC_RAW = RAW / "EMG_EEG_SYNCHRONIZED"
MI_RAW = RAW / "EEG_MOTOR_IMAGERY"
BIOZ_RAW = RAW / "BIOIMPEDANCE"

EXEC_EEG = PROCESSED / "EEG_MOTOR_EXECUTION"
MI_EEG = PROCESSED / "EEG_MOTOR_IMAGERY"
EMG = PROCESSED / "EMG"
BIOZ = PROCESSED / "BIOIMPEDANCE"

PROTOCOL_VERSION = "GLOBAL-QC-DATASET-V1.0"

EXEC_SUBJECTS = {
    f"Subject_{i:02d}"
    for i in range(1, 41)
}

GESTURE_MAP = {
    "HOC": "Hand_Open_Close",
    "TFM": "Thumb_Finger_Movement",
    "IFM": "Index_Finger_Movement",
    "MFM": "Middle_Finger_Movement",
    "RFM": "Ring_Finger_Movement",
    "LFM": "Little_Finger_Movement",
    "PH": "Pen_Holding",
}

GESTURE_CODES = set(GESTURE_MAP)

GESTURE_NAME_TO_CODE = {
    v.lower(): k
    for k, v in GESTURE_MAP.items()
}

SETS = {"A", "B", "C"}

EXPECTED_EXEC = 40 * 7 * 3
EXPECTED_MI = 25 * 7 * 3
EXPECTED_BIOZ_UNITS = 18 * 7 * 15
EXPECTED_BIOZ_SPEC = EXPECTED_BIOZ_UNITS * 2

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
# 2. UTILITIES
# ====================================================================

def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def write_csv(path, rows, fieldnames):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
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


def write_json(path, data):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def status(condition):
    return "PASS" if condition else "FAIL"


# ====================================================================
# 3. GESTURE PARSER
# ====================================================================

def gesture_code_from_filename_or_folder(path: Path):

    stem = path.stem.lower()

    # ---------------------------------------------------------------
    # First try abbreviated gesture codes
    # ---------------------------------------------------------------

    for code in GESTURE_CODES:

        pattern = rf"(?:^|_){code.lower()}(?:_|$)"

        if re.search(pattern, stem):
            return code

    # ---------------------------------------------------------------
    # Then try full gesture names
    # ---------------------------------------------------------------

    for full_name, code in GESTURE_NAME_TO_CODE.items():

        normalized_stem = re.sub(
            r"[^a-z0-9]+",
            "_",
            stem,
        ).strip("_")

        normalized_name = re.sub(
            r"[^a-z0-9]+",
            "_",
            full_name,
        ).strip("_")

        if normalized_name in normalized_stem:
            return code

    # ---------------------------------------------------------------
    # Finally use parent gesture directory
    # ---------------------------------------------------------------

    parent_name = path.parent.name.lower()

    if parent_name in GESTURE_NAME_TO_CODE:
        return GESTURE_NAME_TO_CODE[parent_name]

    return None


def set_from_filename(path: Path):

    m = re.search(
        r"(?:^|_)Set([ABC])(?:$|_)",
        path.stem,
        re.IGNORECASE,
    )

    if m:
        return m.group(1).upper()

    return None


def execution_subject_from_filename(path: Path):

    m = re.search(
        r"(?:^|_)S(\d{2})(?:_|$)",
        path.stem,
        re.IGNORECASE,
    )

    if not m:
        return None

    return f"Subject_{int(m.group(1)):02d}"


def mi_id_from_filename(path: Path):

    m = re.search(
        r"(?:^|_)MI_(\d{2})(?:_|$)",
        path.stem,
        re.IGNORECASE,
    )

    if not m:
        return None

    return f"MI_{int(m.group(1)):02d}"


# ====================================================================
# 4. INVENTORY
# ====================================================================

def inventory():

    print("\n" + "=" * 72)
    print("GLOBAL DATASET INVENTORY")
    print("=" * 72)

    result = {}

    targets = {
        "execution_raw_csv": (
            EXEC_RAW,
            "*.csv",
        ),
        "mi_raw_csv": (
            MI_RAW,
            "*.csv",
        ),
        "bioz_raw_spec": (
            BIOZ_RAW,
            "*.spec",
        ),
        "execution_eeg_npz": (
            EXEC_EEG,
            "*.npz",
        ),
        "mi_eeg_npz": (
            MI_EEG,
            "*.npz",
        ),
        "emg_npz": (
            EMG,
            "*.npz",
        ),
        "bioz_npz": (
            BIOZ,
            "*.npz",
        ),
    }

    for name, (folder, pattern) in targets.items():

        count = (
            len(list(folder.rglob(pattern)))
            if folder.exists()
            else 0
        )

        result[name] = count

        print(
            f"{name:25s}: {count}"
        )

    return result


# ====================================================================
# 5. EXECUTION RAW CONSISTENCY
# ====================================================================

def execution_raw_qc():

    print("\n" + "-" * 72)
    print("MOTOR EXECUTION RAW CONSISTENCY")
    print("-" * 72)

    files = sorted(
        EXEC_RAW.rglob("*.csv")
    )

    keys = []
    malformed = []

    for path in files:

        subject = execution_subject_from_filename(path)
        gesture = gesture_code_from_filename_or_folder(path)
        set_id = set_from_filename(path)

        if (
            subject is None
            or gesture is None
            or set_id is None
        ):

            malformed.append(
                str(path.relative_to(ROOT))
            )

            continue

        keys.append(
            (
                subject,
                gesture,
                set_id,
            )
        )

    counts = Counter(keys)

    expected_keys = {
        (
            subject,
            gesture,
            set_id,
        )
        for subject in EXEC_SUBJECTS
        for gesture in GESTURE_CODES
        for set_id in SETS
    }

    actual_keys = set(keys)

    missing = sorted(
        expected_keys - actual_keys
    )

    unexpected = sorted(
        actual_keys - expected_keys
    )

    duplicates = {
        str(k): v
        for k, v in counts.items()
        if v > 1
    }

    passed = (
        len(files) == EXPECTED_EXEC
        and len(actual_keys) == EXPECTED_EXEC
        and not malformed
        and not missing
        and not unexpected
        and not duplicates
    )

    print(
        f"Expected recordings : {EXPECTED_EXEC}"
    )
    print(
        f"Actual CSV files    : {len(files)}"
    )
    print(
        f"Unique keys         : {len(actual_keys)}"
    )
    print(
        f"Missing keys        : {len(missing)}"
    )
    print(
        f"Unexpected keys     : {len(unexpected)}"
    )
    print(
        f"Duplicate keys      : {len(duplicates)}"
    )
    print(
        f"Malformed filenames : {len(malformed)}"
    )
    print(
        f"STATUS              : {status(passed)}"
    )

    rows = []

    for key in sorted(expected_keys):

        subject, gesture, set_id = key

        rows.append({
            "subject": subject,
            "gesture_code": gesture,
            "set_id": set_id,
            "present": "YES" if key in actual_keys else "NO",
            "status": status(key in actual_keys),
        })

    write_csv(
        QC_DIR / "execution_global_coverage.csv",
        rows,
        [
            "subject",
            "gesture_code",
            "set_id",
            "present",
            "status",
        ],
    )

    return {
        "status": status(passed),
        "file_count": len(files),
        "unique_keys": len(actual_keys),
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "malformed": malformed,
        "keys": actual_keys,
    }


# ====================================================================
# 6. MI RAW CONSISTENCY
# ====================================================================

def mi_raw_qc():

    print("\n" + "-" * 72)
    print("MOTOR IMAGERY RAW CONSISTENCY")
    print("-" * 72)

    files = sorted(
        MI_RAW.rglob("*.csv")
    )

    keys = []
    malformed = []

    for path in files:

        mi_id = mi_id_from_filename(path)
        gesture = gesture_code_from_filename_or_folder(path)
        set_id = set_from_filename(path)

        if (
            mi_id is None
            or gesture is None
            or set_id is None
        ):

            malformed.append(
                str(path.relative_to(ROOT))
            )

            continue

        keys.append(
            (
                mi_id,
                gesture,
                set_id,
            )
        )

    counts = Counter(keys)

    actual_keys = set(keys)

    mi_ids = sorted(
        {
            key[0]
            for key in actual_keys
        }
    )

    expected_keys = {
        (
            mi_id,
            gesture,
            set_id,
        )
        for mi_id in mi_ids
        for gesture in GESTURE_CODES
        for set_id in SETS
    }

    missing = sorted(
        expected_keys - actual_keys
    )

    duplicates = {
        str(k): v
        for k, v in counts.items()
        if v > 1
    }

    passed = (
        len(files) == EXPECTED_MI
        and len(actual_keys) == EXPECTED_MI
        and len(mi_ids) == 25
        and not malformed
        and not missing
        and not duplicates
    )

    print(
        f"Expected recordings : {EXPECTED_MI}"
    )
    print(
        f"Actual CSV files    : {len(files)}"
    )
    print(
        f"Unique keys         : {len(actual_keys)}"
    )
    print(
        f"Actual MI IDs       : {len(mi_ids)}"
    )
    print(
        f"Missing keys        : {len(missing)}"
    )
    print(
        f"Duplicate keys      : {len(duplicates)}"
    )
    print(
        f"Malformed filenames : {len(malformed)}"
    )
    print(
        f"STATUS              : {status(passed)}"
    )

    write_csv(
        QC_DIR / "mi_global_coverage.csv",
        [
            {
                "mi_id": mi_id,
                "recording_count": sum(
                    1
                    for k in actual_keys
                    if k[0] == mi_id
                ),
            }
            for mi_id in mi_ids
        ],
        [
            "mi_id",
            "recording_count",
        ],
    )

    return {
        "status": status(passed),
        "file_count": len(files),
        "unique_keys": len(actual_keys),
        "mi_ids": mi_ids,
        "missing": missing,
        "duplicates": duplicates,
        "malformed": malformed,
    }


# ====================================================================
# 7. EEG–EMG PAIRING
# ====================================================================

def processed_execution_keys(folder):

    keys = set()

    for path in sorted(
        folder.rglob("*.npz")
    ):

        subject = execution_subject_from_filename(path)
        gesture = gesture_code_from_filename_or_folder(path)
        set_id = set_from_filename(path)

        if (
            subject
            and gesture
            and set_id
        ):

            keys.add(
                (
                    subject,
                    gesture,
                    set_id,
                )
            )

    return keys


def eeg_emg_pairing_qc():

    print("\n" + "-" * 72)
    print("EEG–EMG PROCESSED PAIRING")
    print("-" * 72)

    eeg_files = sorted(
        EXEC_EEG.rglob("*.npz")
    )

    emg_files = sorted(
        p
        for p in EMG.rglob("*.npz")
        if "PRE2_QC" not in p.parts
    )

    eeg_keys = processed_execution_keys(
        EXEC_EEG
    )

    emg_keys = processed_execution_keys(
        EMG
    )

    missing_emg = sorted(
        eeg_keys - emg_keys
    )

    missing_eeg = sorted(
        emg_keys - eeg_keys
    )

    passed = (
        len(eeg_files) == EXPECTED_EXEC
        and len(emg_files) == EXPECTED_EXEC
        and len(eeg_keys) == EXPECTED_EXEC
        and len(emg_keys) == EXPECTED_EXEC
        and not missing_emg
        and not missing_eeg
    )

    print(
        f"EEG recordings      : {len(eeg_files)}"
    )
    print(
        f"EMG recordings      : {len(emg_files)}"
    )
    print(
        f"EEG unique keys     : {len(eeg_keys)}"
    )
    print(
        f"EMG unique keys     : {len(emg_keys)}"
    )
    print(
        f"Missing EMG pairs   : {len(missing_emg)}"
    )
    print(
        f"Missing EEG pairs   : {len(missing_eeg)}"
    )
    print(
        f"STATUS              : {status(passed)}"
    )

    rows = []

    all_keys = sorted(
        eeg_keys | emg_keys
    )

    for subject, gesture, set_id in all_keys:

        eeg_ok = (
            (
                subject,
                gesture,
                set_id,
            )
            in eeg_keys
        )

        emg_ok = (
            (
                subject,
                gesture,
                set_id,
            )
            in emg_keys
        )

        rows.append({
            "subject": subject,
            "gesture_code": gesture,
            "set_id": set_id,
            "eeg_present": eeg_ok,
            "emg_present": emg_ok,
            "status": status(
                eeg_ok and emg_ok
            ),
        })

    write_csv(
        QC_DIR / "EEG_EMG_pairing_QC.csv",
        rows,
        [
            "subject",
            "gesture_code",
            "set_id",
            "eeg_present",
            "emg_present",
            "status",
        ],
    )

    return {
        "status": status(passed),
        "eeg_count": len(eeg_files),
        "emg_count": len(emg_files),
        "eeg_keys": eeg_keys,
        "emg_keys": emg_keys,
        "missing_eeg": missing_eeg,
        "missing_emg": missing_emg,
    }


# ====================================================================
# 8. PROCESSED DATA STRUCTURE
# ====================================================================

def processed_structure_qc():

    print("\n" + "-" * 72)
    print("PROCESSED DATA STRUCTURE")
    print("-" * 72)

    rows = []

    # ---------------------------------------------------------------
    # EEG EXECUTION
    # ---------------------------------------------------------------

    for path in sorted(
        EXEC_EEG.rglob("*.npz")
    ):

        status_value = "PASS"
        reason = ""

        try:

            # allow_pickle=True is required because timestamp
            # provenance may be stored as an object/string array.
            with np.load(
                path,
                allow_pickle=True,
            ) as d:

                required = {
                    "eeg",
                    "raw_eeg",
                    "sample_index",
                    "timestamp",
                    "time_seconds",
                    "windows",
                }

                missing = (
                    required
                    - set(d.files)
                )

                if missing:

                    status_value = "FAIL"
                    reason = (
                        f"missing_keys="
                        f"{sorted(missing)}"
                    )

                elif d["eeg"].shape != (
                    13,
                    5625,
                ):

                    status_value = "FAIL"
                    reason = (
                        f"eeg_shape="
                        f"{d['eeg'].shape}"
                    )

                elif d["raw_eeg"].shape != (
                    13,
                    5625,
                ):

                    status_value = "FAIL"
                    reason = (
                        f"raw_eeg_shape="
                        f"{d['raw_eeg'].shape}"
                    )

                elif not np.all(
                    np.isfinite(d["eeg"])
                ):

                    status_value = "FAIL"
                    reason = "eeg_nonfinite"

                elif not np.all(
                    np.isfinite(d["raw_eeg"])
                ):

                    status_value = "FAIL"
                    reason = "raw_eeg_nonfinite"

        except Exception as e:

            status_value = "FAIL"
            reason = repr(e)

        rows.append({
            "modality": "EEG_MOTOR_EXECUTION",
            "file": str(
                path.relative_to(ROOT)
            ),
            "status": status_value,
            "reason": reason,
        })

    # ---------------------------------------------------------------
    # EEG MOTOR IMAGERY
    # ---------------------------------------------------------------

    for path in sorted(
        MI_EEG.rglob("*.npz")
    ):

        status_value = "PASS"
        reason = ""

        try:

            with np.load(
                path,
                allow_pickle=True,
            ) as d:

                required = {
                    "eeg",
                    "raw_eeg",
                    "sample_index",
                    "timestamp",
                    "time_seconds",
                    "windows",
                }

                missing = (
                    required
                    - set(d.files)
                )

                if missing:

                    status_value = "FAIL"
                    reason = (
                        f"missing_keys="
                        f"{sorted(missing)}"
                    )

                elif d["eeg"].shape != (
                    13,
                    5625,
                ):

                    status_value = "FAIL"
                    reason = (
                        f"eeg_shape="
                        f"{d['eeg'].shape}"
                    )

                elif d["raw_eeg"].shape != (
                    13,
                    5625,
                ):

                    status_value = "FAIL"
                    reason = (
                        f"raw_eeg_shape="
                        f"{d['raw_eeg'].shape}"
                    )

                elif not np.all(
                    np.isfinite(d["eeg"])
                ):

                    status_value = "FAIL"
                    reason = "eeg_nonfinite"

                elif not np.all(
                    np.isfinite(d["raw_eeg"])
                ):

                    status_value = "FAIL"
                    reason = "raw_eeg_nonfinite"

        except Exception as e:

            status_value = "FAIL"
            reason = repr(e)

        rows.append({
            "modality": "EEG_MOTOR_IMAGERY",
            "file": str(
                path.relative_to(ROOT)
            ),
            "status": status_value,
            "reason": reason,
        })

    # ---------------------------------------------------------------
    # EMG
    # ---------------------------------------------------------------

    for path in sorted(
        EMG.rglob("*.npz")
    ):

        if "PRE2_QC" in path.parts:
            continue

        status_value = "PASS"
        reason = ""

        try:

            with np.load(
                path,
                allow_pickle=True,
            ) as d:

                required = {
                    "emg_raw",
                    "emg_preprocessed",
                    "sample_index",
                    "timestamp_formatted",
                }

                missing = (
                    required
                    - set(d.files)
                )

                if missing:

                    status_value = "FAIL"
                    reason = (
                        f"missing_keys="
                        f"{sorted(missing)}"
                    )

                elif d["emg_raw"].shape != (
                    5625,
                    3,
                ):

                    status_value = "FAIL"
                    reason = (
                        f"emg_raw_shape="
                        f"{d['emg_raw'].shape}"
                    )

                elif d[
                    "emg_preprocessed"
                ].shape != (
                    5625,
                    3,
                ):

                    status_value = "FAIL"
                    reason = (
                        f"emg_preprocessed_shape="
                        f"{d['emg_preprocessed'].shape}"
                    )

                elif not np.all(
                    np.isfinite(
                        d["emg_raw"]
                    )
                ):

                    status_value = "FAIL"
                    reason = "emg_raw_nonfinite"

                elif not np.all(
                    np.isfinite(
                        d["emg_preprocessed"]
                    )
                ):

                    status_value = "FAIL"
                    reason = (
                        "emg_preprocessed_nonfinite"
                    )

        except Exception as e:

            status_value = "FAIL"
            reason = repr(e)

        rows.append({
            "modality": "EMG_MOTOR_EXECUTION",
            "file": str(
                path.relative_to(ROOT)
            ),
            "status": status_value,
            "reason": reason,
        })

    # ---------------------------------------------------------------
    # BIOIMPEDANCE
    # ---------------------------------------------------------------

    for path in sorted(
        BIOZ.rglob("*.npz")
    ):

        if "PRE3_QC" in path.parts:
            continue

        status_value = "PASS"
        reason = ""

        try:

            with np.load(
                path,
                allow_pickle=False,
            ) as d:

                required = {
                    "frequency_hz",
                    "re",
                    "im",
                    "z_magnitude",
                    "z_phase_deg",
                }

                missing = (
                    required
                    - set(d.files)
                )

                if missing:

                    status_value = "FAIL"
                    reason = (
                        f"missing_keys="
                        f"{sorted(missing)}"
                    )

                else:

                    for key in required:

                        if d[key].shape != (
                            100,
                        ):

                            status_value = "FAIL"
                            reason = (
                                f"{key}_shape="
                                f"{d[key].shape}"
                            )
                            break

                        if not np.all(
                            np.isfinite(d[key])
                        ):

                            status_value = "FAIL"
                            reason = (
                                f"{key}_nonfinite"
                            )
                            break

        except Exception as e:

            status_value = "FAIL"
            reason = repr(e)

        rows.append({
            "modality": "BIOIMPEDANCE",
            "file": str(
                path.relative_to(ROOT)
            ),
            "status": status_value,
            "reason": reason,
        })

    write_csv(
        QC_DIR / "processed_data_integrity.csv",
        rows,
        [
            "modality",
            "file",
            "status",
            "reason",
        ],
    )

    failures = sum(
        1
        for r in rows
        if r["status"] != "PASS"
    )

    print(
        f"Files checked : {len(rows)}"
    )
    print(
        f"Failures      : {failures}"
    )
    print(
        f"STATUS        : {status(failures == 0)}"
    )

    return {
        "files_checked": len(rows),
        "failures": failures,
        "status": status(
            failures == 0
        ),
    }


# ====================================================================
# 9. BIOZ CROSSWALK
# ====================================================================

def bioz_crosswalk_qc():

    print("\n" + "-" * 72)
    print("BIOIMPEDANCE CROSSWALK")
    print("-" * 72)

    rows = []
    overall = True

    for bz_id, subject in BIOZ_CROSSWALK.items():

        folder = BIOZ_RAW / bz_id

        count = (
            len(list(folder.rglob("*.spec")))
            if folder.exists()
            else 0
        )

        ok = count == 210

        if not ok:
            overall = False

        print(
            f"{bz_id:6s} -> "
            f"{subject:10s} | "
            f"{count:3d}/210 | "
            f"{status(ok)}"
        )

        rows.append({
            "bioz_id": bz_id,
            "canonical_subject": subject,
            "expected_spec_files": 210,
            "actual_spec_files": count,
            "status": status(ok),
        })

    write_csv(
        QC_DIR / "bioz_crosswalk_QC.csv",
        rows,
        [
            "bioz_id",
            "canonical_subject",
            "expected_spec_files",
            "actual_spec_files",
            "status",
        ],
    )

    print(
        f"STATUS: {status(overall)}"
    )

    return {
        "status": status(overall),
        "rows": rows,
    }


# ====================================================================
# 10. DUPLICATE CONTENT
# ====================================================================

def duplicate_content_qc():

    print("\n" + "-" * 72)
    print("PROCESSED FILE DUPLICATE CONTENT CHECK")
    print("-" * 72)

    targets = []

    for folder in [
        EXEC_EEG,
        MI_EEG,
        EMG,
        BIOZ,
    ]:

        if not folder.exists():
            continue

        for path in folder.rglob("*.npz"):

            if "PRE2_QC" in path.parts:
                continue

            if "PRE3_QC" in path.parts:
                continue

            targets.append(path)

    hashes = defaultdict(list)

    for i, path in enumerate(
        targets,
        1,
    ):

        digest = sha256_file(path)

        hashes[digest].append(
            str(path.relative_to(ROOT))
        )

        if (
            i % 500 == 0
            or i == len(targets)
        ):

            print(
                f"\rHashed "
                f"{i}/{len(targets)}",
                end="",
                flush=True,
            )

    print()

    duplicate_groups = {
        digest: paths
        for digest, paths
        in hashes.items()
        if len(paths) > 1
    }

    rows = []

    for digest, paths in duplicate_groups.items():

        for path in paths:

            rows.append({
                "sha256": digest,
                "file": path,
                "group_size": len(paths),
            })

    write_csv(
        QC_DIR / "duplicate_content_QC.csv",
        rows,
        [
            "sha256",
            "file",
            "group_size",
        ],
    )

    ok = len(
        duplicate_groups
    ) == 0

    print(
        f"Files hashed       : {len(targets)}"
    )
    print(
        f"Duplicate groups   : "
        f"{len(duplicate_groups)}"
    )
    print(
        f"STATUS             : "
        f"{status(ok)}"
    )

    return {
        "files_hashed": len(targets),
        "duplicate_groups": len(
            duplicate_groups
        ),
        "status": status(ok),
    }


# ====================================================================
# 11. METADATA
# ====================================================================

def metadata_qc():

    print("\n" + "-" * 72)
    print("METADATA PRESENCE")
    print("-" * 72)

    required = [
        "participants.csv",
        "subject_modality_availability.csv",
        "cohort_definition.csv",
        "recording_inventory.csv",
        "recording_parameters.csv",
        "acquisition_parameters.csv",
        "gesture_dictionary.csv",
        "data_provenance.csv",
        "exclusion_log.csv",
    ]

    rows = []

    for name in required:

        exists = (
            METADATA / name
        ).exists()

        print(
            f"{name:40s}: "
            f"{status(exists)}"
        )

        rows.append({
            "file": name,
            "exists": status(exists),
        })

    ok = all(
        r["exists"] == "PASS"
        for r in rows
    )

    write_csv(
        QC_DIR / "metadata_presence_QC.csv",
        rows,
        [
            "file",
            "exists",
        ],
    )

    print(
        f"STATUS: {status(ok)}"
    )

    return {
        "status": status(ok),
    }


# ====================================================================
# 12. FINAL GLOBAL REPORT
# ====================================================================

def final_report(
    inventory_data,
    execution,
    pairing,
    structure,
    bioz,
    mi,
    duplicates,
    metadata,
):

    checks = {
        "execution_raw": execution["status"],
        "eeg_emg_pairing": pairing["status"],
        "processed_structure": structure["status"],
        "bioz_crosswalk": bioz["status"],
        "motor_imagery_inventory": mi["status"],
        "duplicate_content": duplicates["status"],
        "metadata_presence": metadata["status"],
    }

    overall = all(
        value == "PASS"
        for value in checks.values()
    )

    report = {

        "dataset_version": "V1.0",

        "protocol_version": (
            PROTOCOL_VERSION
        ),

        "generated_at": now_iso(),

        "dataset_root": str(ROOT),

        "raw_data_modified": False,

        "inventory": inventory_data,

        "checks": checks,

        "cohort": {

            "motor_execution_subjects": 40,

            "motor_execution_recordings": 840,

            "motor_imagery_ids": mi["mi_ids"],

            "motor_imagery_participants_or_ids": len(
                mi["mi_ids"]
            ),

            "bioimpedance_participants": 18,

            "excluded_subject": "Subject_41",
        },

        "synchronization": {

            "eeg_emg": (
                "sample-level synchronized "
                "through combined acquisition"
            ),

            "bioimpedance": (
                "independent acquisition; "
                "no sample-level synchronization"
            ),
        },

        "preprocessing_status": {

            "PRE1_EEG": "COMPLETE",

            "PRE2_EMG_MOTOR_EXECUTION": "COMPLETE",

            "PRE3_BIOIMPEDANCE": "COMPLETE",
        },

        "annotations_created": False,

        "fusion_created": False,

        "overall_status": (
            "PASS"
            if overall
            else "FAIL"
        ),
    }

    write_json(
        QC_DIR / "DATASET_GLOBAL_QC_REPORT.json",
        report,
    )

    return report


# ====================================================================
# 13. MAIN
# ====================================================================

def main():

    print("\n")
    print("=" * 72)
    print(
        "GLOBAL DATASET QC — DATASET V1.0"
    )
    print("=" * 72)

    print(
        f"ROOT     : {ROOT}"
    )

    print(
        f"VERSION  : {PROTOCOL_VERSION}"
    )

    print(
        f"START    : {now_iso()}"
    )

    print("=" * 72)

    if not ROOT.exists():

        print(
            "\nERROR: Dataset root does not exist."
        )

        sys.exit(1)

    QC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    inv = inventory()

    execution = execution_raw_qc()

    pairing = eeg_emg_pairing_qc()

    structure = processed_structure_qc()

    bioz = bioz_crosswalk_qc()

    mi = mi_raw_qc()

    duplicates = duplicate_content_qc()

    metadata = metadata_qc()

    report = final_report(
        inv,
        execution,
        pairing,
        structure,
        bioz,
        mi,
        duplicates,
        metadata,
    )

    print("\n")
    print("=" * 72)
    print(
        "GLOBAL DATASET QC FINAL SUMMARY"
    )
    print("=" * 72)

    for name, value in report[
        "checks"
    ].items():

        print(
            f"{name:30s}: {value}"
        )

    print("-" * 72)

    print(
        "RAW DATA MODIFICATION      : "
        f"{report['raw_data_modified']}"
    )

    print(
        "GLOBAL DATASET QC STATUS    : "
        f"{report['overall_status']}"
    )

    print(
        "REPORT                      : "
        f"{QC_DIR / 'DATASET_GLOBAL_QC_REPORT.json'}"
    )

    print("=" * 72)

    if report[
        "overall_status"
    ] != "PASS":

        sys.exit(2)


if __name__ == "__main__":
    main()
