#!/usr/bin/env python3

"""

DATA2 — Dataset Integrity & Quality Assurance — FINAL FREEZE

Read-only integrity audit for:

  01_RAW_DATA/EMG_EEG_SYNCHRONIZED

  01_RAW_DATA/EEG_MOTOR_IMAGERY

  01_RAW_DATA/BIOIMPEDANCE

Important protocol note:

Motor Execution and Motor Imagery were recorded at different times.

Therefore DATA2 NEVER attempts to synchronize their timestamps and

does not treat timestamp differences between modalities as errors.

Expected:

  Execution: 40 subjects × 7 gestures × 3 sets = 840 CSV

  MI:        25 MI IDs × 7 gestures × 3 sets = 525 CSV

  BioZ:      18 BZ IDs × 7 gestures × 15 units × 2 channels = 3780 .spec (1890 paired recording units)

"""

from pathlib import Path

from datetime import datetime

import hashlib

import json

import logging

import re

import sys

import time

import numpy as np

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "01_RAW_DATA").exists():

    ROOT = SCRIPT_DIR

elif (SCRIPT_DIR.parent / "01_RAW_DATA").exists():

    ROOT = SCRIPT_DIR.parent

else:

    raise FileNotFoundError("Cannot locate dataset root. Place script at dataset root or dataset_root/scripts/.")

RAW = ROOT / "01_RAW_DATA"

EXECUTION = RAW / "EMG_EEG_SYNCHRONIZED"

IMAGERY = RAW / "EEG_MOTOR_IMAGERY"

BIOZ = RAW / "BIOIMPEDANCE"

QC = ROOT / "05_QC"

INTERNAL_QC = QC / "_INTERNAL_DATA2"

EXPECTED_EXEC_SUBJECTS = 40

EXPECTED_MI_SUBJECTS = 25

EXPECTED_BIOZ_SUBJECTS = 18

EXPECTED_BZ_UNITS = 1890

EXPECTED_BZ_SPEC_FILES = 3780

GESTURES = [

    "Hand_Open_Close",

    "Index_Finger_Movement",

    "Little_Finger_Movement",

    "Middle_Finger_Movement",

    "Pen_Holding",

    "Ring_Finger_Movement",

    "Thumb_Finger_Movement",

]

SETS = ["A", "B", "C"]

EXPECTED_EXEC_ROWS = 5625

EXPECTED_MI_ROWS = 5625

EXPECTED_EXEC_COLS = [

    "Sample Index", "Timestamp (Formatted)",

    "EMG_ch-01", "EMG_ch-02", "EMG_ch-03",

    "EEG_ch-01", "EEG_ch-02", "EEG_ch-03", "EEG_ch-04",

    "EEG_ch-05", "EEG_ch-06", "EEG_ch-07", "EEG_ch-08",

    "EEG_ch-09", "EEG_ch-10", "EEG_ch-11", "EEG_ch-12", "EEG_ch-13",

]

EXPECTED_MI_COLS = [

    "Sample Index", "Timestamp (Formatted)",

    "EEG_ch-01", "EEG_ch-02", "EEG_ch-03", "EEG_ch-04",

    "EEG_ch-05", "EEG_ch-06", "EEG_ch-07", "EEG_ch-08",

    "EEG_ch-09", "EEG_ch-10", "EEG_ch-11", "EEG_ch-12", "EEG_ch-13",

]

# These are audit thresholds, not deletion rules.

TIMESTAMP_GAP_WARN_SEC = 0.050

CONST_STD_EPS = 1e-12

REPEAT_FRACTION_WARN = 0.95
EXPECTED_WRAP_SUBJECT_MAX = 17
EXPECTED_WRAP_DELTA = -254
KNOWN_CONSTANT_EXECUTION = {"01_RAW_DATA\\EMG_EEG_SYNCHRONIZED\\Subject_08\\Hand_Open_Close\\S08_HOC_SetB.csv"}
KNOWN_CONSTANT_MI = {
    "01_RAW_DATA\\EEG_MOTOR_IMAGERY\\MI_11\\Index_Finger_Movement\\MI_11_IFM_SetA.csv",
    "01_RAW_DATA\\EEG_MOTOR_IMAGERY\\MI_11\\Index_Finger_Movement\\MI_11_IFM_SetB.csv",
    "01_RAW_DATA\\EEG_MOTOR_IMAGERY\\MI_11\\Index_Finger_Movement\\MI_11_IFM_SetC.csv",
    "01_RAW_DATA\\EEG_MOTOR_IMAGERY\\MI_11\\Little_Finger_Movement\\MI_11_LFM_SetA.csv",
    "01_RAW_DATA\\EEG_MOTOR_IMAGERY\\MI_11\\Little_Finger_Movement\\MI_11_LFM_SetB.csv",
    "01_RAW_DATA\\EEG_MOTOR_IMAGERY\\MI_11\\Little_Finger_Movement\\MI_11_LFM_SetC.csv",
    "01_RAW_DATA\\EEG_MOTOR_IMAGERY\\MI_11\\Middle_Finger_Movement\\MI_11_MFM_SetA.csv",
    "01_RAW_DATA\\EEG_MOTOR_IMAGERY\\MI_11\\Middle_Finger_Movement\\MI_11_MFM_SetB.csv",
    "01_RAW_DATA\\EEG_MOTOR_IMAGERY\\MI_11\\Middle_Finger_Movement\\MI_11_MFM_SetC.csv",
    "01_RAW_DATA\\EEG_MOTOR_IMAGERY\\MI_11\\Ring_Finger_Movement\\MI_11_RFM_SetA.csv",
    "01_RAW_DATA\\EEG_MOTOR_IMAGERY\\MI_11\\Ring_Finger_Movement\\MI_11_RFM_SetB.csv",
    "01_RAW_DATA\\EEG_MOTOR_IMAGERY\\MI_11\\Ring_Finger_Movement\\MI_11_RFM_SetC.csv",
}

def setup_logging():

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)-8s | %(message)s",

        datefmt="%Y-%m-%d %H:%M:%S",

    )

def log(msg):

    logging.info(msg)

def write_csv(df, path):

    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    log(f"WROTE | {path.relative_to(ROOT)} | rows={len(df)} cols={len(df.columns)}")

def sha256(path, chunk=1024 * 1024):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            b = f.read(chunk)

            if not b:

                break

            h.update(b)

    return h.hexdigest()

def parse_exec_name(name):

    m = re.match(r"^S(\d{2})_([A-Z]{2,3})_Set([ABC])\.csv$", name, re.I)

    if not m:

        return None

    return int(m.group(1)), m.group(2).upper(), m.group(3).upper()

def parse_mi_name(name):

    m = re.match(r"^MI_(\d{2})_([A-Z]{2,3})_Set([ABC])\.csv$", name, re.I)

    if not m:

        return None

    return int(m.group(1)), m.group(2).upper(), m.group(3).upper()

def gesture_code(folder):

    return {

        "Hand_Open_Close": "HOC",

        "Index_Finger_Movement": "IFM",

        "Little_Finger_Movement": "LFM",

        "Middle_Finger_Movement": "MFM",

        "Pen_Holding": "PH",

        "Ring_Finger_Movement": "RFM",

        "Thumb_Finger_Movement": "TFM",

    }.get(folder)

def expected_sample_index_wrap_count(path, modality, di):
    if modality != "MOTOR_EXECUTION":
        return 0
    m = re.search(r"Subject_(\d+)", str(path))
    subject_num = int(m.group(1)) if m else None
    if subject_num is None or subject_num > EXPECTED_WRAP_SUBJECT_MAX:
        return 0
    return int(np.sum(di == EXPECTED_WRAP_DELTA))

def csv_audit(path, modality):

    expected_rows = EXPECTED_EXEC_ROWS if modality == "MOTOR_EXECUTION" else EXPECTED_MI_ROWS

    expected_cols = EXPECTED_MI_COLS
    schema_label = "MI_STANDARD"
    subject_num = None
    if modality == "MOTOR_EXECUTION":
        m = re.search(r"Subject_(\d+)", str(path))
        subject_num = int(m.group(1)) if m else None
        expected_cols = EXPECTED_EXEC_COLS
        schema_label = "EXEC_CANONICAL_SUBJECT_01_40"

    row = {

        "modality": modality,

        "file": str(path.relative_to(ROOT)),

        "readable": False,

        "schema_pass": False,

        "schema_label": schema_label,

        "row_count_pass": False,

        "filename_semantics_pass": False,

        "filename_semantics_reason": "",

        "nan_count": np.nan,

        "inf_count": np.nan,

        "timestamp_parse_pass": False,

        "timestamp_monotonic": False,

        "duplicate_timestamps": np.nan,

        "timestamp_negative_steps": np.nan,

        "timestamp_large_gaps": np.nan,

        "timestamp_median_step_sec": np.nan,

        "duration_sec": np.nan,

        "sample_index_numeric": False,

        "sample_index_monotonic": False,

        "sample_index_duplicates": np.nan,

        "sample_index_negative_steps": np.nan,

        "sample_index_expected_wraps": 0,

        "sample_index_unexplained_negative_steps": 0,

        "sample_index_gaps": np.nan,

        "sample_index_median_step": np.nan,

        "constant_channels": "",

        "near_constant_channels": "",

        "high_repeat_channels": "",

        "error": "",

    }

    try:

        df = pd.read_csv(path)

        row["readable"] = True

        row["schema_pass"] = list(df.columns) == expected_cols

        row["row_count_pass"] = len(df) == expected_rows

        # Validate filename <-> folder <-> subject <-> gesture <-> set.

        if modality == "MOTOR_EXECUTION":

            parsed = parse_exec_name(path.name)

            parts = path.relative_to(EXECUTION).parts

            folder_subject = parts[0] if len(parts) >= 1 else ""

            folder_gesture = parts[1] if len(parts) >= 2 else ""

            if parsed:

                snum, code, setname = parsed

                expected_subject = f"Subject_{snum:02d}"

                expected_gesture_code = gesture_code(folder_gesture)

                ok = (folder_subject == expected_subject and expected_gesture_code == code and setname in SETS)

                row["filename_semantics_pass"] = ok

                if not ok:

                    row["filename_semantics_reason"] = f"subject={folder_subject},file_subject={expected_subject},folder_gesture_code={expected_gesture_code},file_code={code}"

            else:

                row["filename_semantics_reason"] = "filename_pattern_mismatch"

        else:

            parsed = parse_mi_name(path.name)

            parts = path.relative_to(IMAGERY).parts

            folder_subject = parts[0] if len(parts) >= 1 else ""

            folder_gesture = parts[1] if len(parts) >= 2 else ""

            if parsed:

                mi_num, code, setname = parsed

                expected_subject = f"MI_{mi_num:02d}"

                expected_gesture_code = gesture_code(folder_gesture)

                ok = (folder_subject == expected_subject and expected_gesture_code == code and setname in SETS)

                row["filename_semantics_pass"] = ok

                if not ok:

                    row["filename_semantics_reason"] = f"subject={folder_subject},file_subject={expected_subject},folder_gesture_code={expected_gesture_code},file_code={code}"

            else:

                row["filename_semantics_reason"] = "filename_pattern_mismatch"

        numeric_cols = [c for c in expected_cols if c != "Timestamp (Formatted)"]

        numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        arr = numeric.to_numpy(dtype=float)

        row["nan_count"] = int(np.isnan(arr).sum())

        row["inf_count"] = int(np.isinf(arr).sum())

        # Timestamp audit — within this file only.

        ts = pd.to_datetime(df["Timestamp (Formatted)"], errors="coerce")

        row["timestamp_parse_pass"] = bool(ts.notna().all())

        if row["timestamp_parse_pass"] and len(ts) > 1:

            dts = ts.diff().dt.total_seconds().dropna().to_numpy()

            row["timestamp_monotonic"] = bool(np.all(dts >= 0))

            row["duplicate_timestamps"] = int(ts.duplicated().sum())

            row["timestamp_negative_steps"] = int((dts < 0).sum())

            row["timestamp_large_gaps"] = int((dts > TIMESTAMP_GAP_WARN_SEC).sum())

            row["timestamp_median_step_sec"] = float(np.median(dts))

            row["duration_sec"] = float((ts.iloc[-1] - ts.iloc[0]).total_seconds())

        # Sample index audit.

        idx = pd.to_numeric(df["Sample Index"], errors="coerce")

        row["sample_index_numeric"] = bool(idx.notna().all())

        if row["sample_index_numeric"] and len(idx) > 1:

            di = idx.diff().dropna().to_numpy()

            row["sample_index_monotonic"] = bool(np.all(di >= 0))

            row["sample_index_duplicates"] = int(idx.duplicated().sum())

            row["sample_index_negative_steps"] = int((di < 0).sum())
            row["sample_index_expected_wraps"] = expected_sample_index_wrap_count(
                path, modality, di
            )
            row["sample_index_unexplained_negative_steps"] = int(
                ((di < 0) & (di != EXPECTED_WRAP_DELTA)).sum()
            )

            row["sample_index_median_step"] = float(np.median(di))

            row["sample_index_gaps"] = int((di != row["sample_index_median_step"]).sum())

        signal_cols = [c for c in expected_cols if c not in

                       ("Sample Index", "Timestamp (Formatted)")]

        constant, near_constant, high_repeat = [], [], []

        for c in signal_cols:

            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)

            finite = x[np.isfinite(x)]

            if len(finite) == 0:

                continue

            std = float(np.std(finite))

            if std <= CONST_STD_EPS:

                constant.append(c)

            elif std <= 1e-6 * max(1.0, abs(float(np.mean(finite)))):

                near_constant.append(c)

            if len(finite) > 1:

                vals, counts = np.unique(finite, return_counts=True)

                frac = float(counts.max() / len(finite))

                if frac >= REPEAT_FRACTION_WARN:

                    high_repeat.append(f"{c}:{frac:.4f}")

        row["constant_channels"] = ";".join(constant)

        row["near_constant_channels"] = ";".join(near_constant)

        row["high_repeat_channels"] = ";".join(high_repeat)
        rel_file = str(path.relative_to(ROOT))
        row["known_acquisition_qc"] = bool(
            rel_file in KNOWN_CONSTANT_EXECUTION or rel_file in KNOWN_CONSTANT_MI
        )

    except Exception as e:

        row["error"] = repr(e)

    return row

def audit_csv_collection(root, modality):

    files = sorted(root.rglob("*.csv"))

    rows = []

    for i, p in enumerate(files, 1):

        if i == 1 or i % 100 == 0 or i == len(files):

            log(f"{modality} CSV audit: {i}/{len(files)}")

        rows.append(csv_audit(p, modality))

    return pd.DataFrame(rows)

def bioz_audit(path):

    row = {

        "file": str(path.relative_to(ROOT)),

        "readable": False,

        "header_pass": False,

        "measurement_header_pass": False,

        "channel_declared": "",

        "data_rows": 0,

        "numeric_rows": 0,

        "invalid_data_rows": 0,

        "frequency_min_hz": np.nan,

        "frequency_max_hz": np.nan,

        "frequency_monotonic": False,

        "re_nan": np.nan,

        "im_nan": np.nan,

        "error": "",

    }

    try:

        text = path.read_text(errors="replace")

        lines = text.splitlines()

        row["readable"] = True

        row["header_pass"] = (

            len(lines) >= 7

            and "SteamingMode" in lines[1]

            and "Channel" in lines[2]

            and "frequency[Hz],Re,Im" in lines[5]

        )

        row["channel_declared"] = lines[2].strip() if len(lines) > 2 else ""

        row["measurement_header_pass"] = (

            len(lines) > 5 and lines[5].strip().lower() == "frequency[hz],re,im"

        )

        if len(lines) > 6:

            records = []

            for line in lines[6:]:

                parts = [x.strip() for x in line.split(",")]

                if len(parts) != 3:

                    if line.strip():

                        row["invalid_data_rows"] += 1

                    continue

                try:

                    records.append([float(parts[0]), float(parts[1]), float(parts[2])])

                except ValueError:

                    row["invalid_data_rows"] += 1

            if records:

                a = np.asarray(records, dtype=float)

                row["data_rows"] = int(len(a))

                row["numeric_rows"] = int(len(a))

                row["re_nan"] = int(np.isnan(a[:, 1]).sum())

                row["im_nan"] = int(np.isnan(a[:, 2]).sum())

                row["frequency_min_hz"] = float(np.min(a[:, 0]))

                row["frequency_max_hz"] = float(np.max(a[:, 0]))

                row["frequency_monotonic"] = bool(np.all(np.diff(a[:, 0]) >= 0))

    except Exception as e:

        row["error"] = repr(e)

    return row

def audit_bioz_collection(root):

    files = sorted(root.rglob("*.spec"))

    rows = []

    for i, p in enumerate(files, 1):

        if i == 1 or i % 250 == 0 or i == len(files):

            log(f"BIOZ .spec audit: {i}/{len(files)}")

        rows.append(bioz_audit(p))

    return pd.DataFrame(rows)

def duplicate_hash_audit(files):

    # Hashing is done only for duplicate detection; no files are modified.

    groups = {}

    rows = []

    for i, p in enumerate(files, 1):

        if i == 1 or i % 500 == 0 or i == len(files):

            log(f"SHA256 duplicate audit: {i}/{len(files)}")

        h = sha256(p)

        groups.setdefault(h, []).append(str(p.relative_to(ROOT)))

    for h, paths in groups.items():

        if len(paths) > 1:

            for p in paths:

                rows.append({

                    "sha256": h,

                    "duplicate_count": len(paths),

                    "file": p,

                })

    return pd.DataFrame(rows, columns=["sha256", "duplicate_count", "file"])

def log_inventory():

    files = sorted([p for p in RAW.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".log"}])

    rows = []

    for p in files:

        try:

            size = p.stat().st_size

        except OSError:

            size = -1

        rows.append({

            "file": str(p.relative_to(ROOT)),

            "size_bytes": size,

            "empty": size == 0,

        })

    return pd.DataFrame(rows)

def bioz_pairing_audit(root):

    rows = []

    for bz in sorted([p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"BZ_\d{2}", p.name)]):

        for g in sorted([p for p in bz.iterdir() if p.is_dir()]):

            ch1 = g / "Channel_1"

            ch2 = g / "Channel_2"

            s1 = {p.stem for p in ch1.glob("*.spec")} if ch1.exists() else set()

            s2 = {p.stem for p in ch2.glob("*.spec")} if ch2.exists() else set()

            for stem in sorted(s1 | s2):

                rows.append({

                    "bz_id": bz.name,

                    "gesture_folder": g.name,

                    "recording_unit": stem,

                    "channel_1_present": stem in s1,

                    "channel_2_present": stem in s2,

                    "paired": stem in s1 and stem in s2,

                })

    return pd.DataFrame(rows)

def log_reconciliation(exec_files, mi_files, log_files):

    """Reconcile one expected sibling log per execution/MI CSV using exact paths."""

    expected = {}

    for p in exec_files:

        ep = p.parent / f"{p.stem}_log.txt"

        expected[str(ep.resolve())] = ("MOTOR_EXECUTION", str(p.relative_to(ROOT)), str(ep.relative_to(ROOT)))

    for p in mi_files:

        ep = p.parent / f"{p.stem}_log.txt"

        expected[str(ep.resolve())] = ("MOTOR_IMAGERY", str(p.relative_to(ROOT)), str(ep.relative_to(ROOT)))

    actual = {str(p.resolve()): p for p in log_files}

    rows = []

    for key, (modality, csvfile, expected_log) in sorted(expected.items()):

        p = actual.get(key)

        rows.append({

            "expected_log": expected_log,

            "modality": modality,

            "csv_file": csvfile,

            "status": "PRESENT" if p else "MISSING",

            "actual_log_file": str(p.relative_to(ROOT)) if p else "",

            "size_bytes": p.stat().st_size if p else "",

        })

    for key, p in sorted(actual.items()):

        if key not in expected:

            rows.append({

                "expected_log": "",

                "modality": "UNMATCHED",

                "csv_file": "",

                "status": "EXTRA",

                "actual_log_file": str(p.relative_to(ROOT)),

                "size_bytes": p.stat().st_size,

            })

    return pd.DataFrame(rows, columns=["expected_log","modality","csv_file","status","actual_log_file","size_bytes"])

def exact_csv_coverage_audit(files, modality):

    """Exact subject/gesture/set coverage audit using the actual filesystem hierarchy."""

    if modality == "MOTOR_EXECUTION":

        expected_ids = {f"Subject_{i:02d}" for i in range(1, 41)}

        parser = parse_exec_name

        root = EXECUTION

        expected_count = 840

    elif modality == "MOTOR_IMAGERY":

        expected_ids = {f"MI_{i:02d}" for i in range(1, 26)}

        parser = parse_mi_name

        root = IMAGERY

        expected_count = 525

    else:

        raise ValueError(f"Unsupported modality: {modality}")

    expected = {

        (subject_id, gesture_code(gesture), set_name)

        for subject_id in expected_ids

        for gesture in GESTURES

        for set_name in SETS

    }

    actual = set()

    slot_counts = {}

    parse_errors = []

    for path in sorted(files):

        try:

            rel = path.relative_to(root)

        except ValueError:

            parse_errors.append(f"{path.relative_to(ROOT)}: outside modality root")

            continue

        parts = rel.parts

        if len(parts) != 3:

            parse_errors.append(f"{path.relative_to(ROOT)}: expected ID/gesture/file hierarchy")

            continue

        subject_id, gesture_folder, filename = parts

        parsed = parser(filename)

        if parsed is None:

            parse_errors.append(f"{path.relative_to(ROOT)}: filename pattern mismatch")

            continue

        _, filename_code, set_name = parsed

        folder_code = gesture_code(gesture_folder)

        if subject_id not in expected_ids:

            parse_errors.append(f"{path.relative_to(ROOT)}: unexpected subject/MI ID")

            continue

        if folder_code is None:

            parse_errors.append(f"{path.relative_to(ROOT)}: unknown gesture folder")

            continue

        if filename_code != folder_code:

            parse_errors.append(f"{path.relative_to(ROOT)}: filename/gesture mismatch")

            continue

        if set_name not in SETS:

            parse_errors.append(f"{path.relative_to(ROOT)}: invalid set")

            continue

        key = (subject_id, folder_code, set_name)

        actual.add(key)

        slot_counts[key] = slot_counts.get(key, 0) + 1

    missing = sorted(expected - actual)

    extra = sorted(actual - expected)

    duplicate_slots = sorted(k for k, n in slot_counts.items() if n > 1)

    return {

        "modality": modality,

        "expected_slots": expected_count,

        "actual_slots": len(actual),

        "missing_slots": len(missing),

        "extra_slots": len(extra),

        "duplicate_slots": len(duplicate_slots),

        "parse_errors": len(parse_errors),

        "exact_coverage_pass": bool(

            len(actual) == expected_count

            and not missing

            and not extra

            and not duplicate_slots

            and not parse_errors

        ),

        "missing_examples": ";".join(map(str, missing[:10])),

        "extra_examples": ";".join(map(str, extra[:10])),

        "duplicate_examples": ";".join(map(str, duplicate_slots[:10])),

        "parse_error_examples": ";".join(parse_errors[:10]),

    }

def summary_rows(exec_df, mi_df, bioz_df, pair_df, logs_df, duplicate_df,

                 exec_coverage, mi_coverage, bz_exact_pass):

    def count(df, col, value=True):

        return int((df[col] == value).sum()) if len(df) else 0

    rows = [

        ["execution_csv_files", len(exec_df), 840, "PASS" if len(exec_df) == 840 else "FAIL"],

        ["execution_exact_coverage", int(exec_coverage["actual_slots"]), exec_coverage["expected_slots"],

         "PASS" if exec_coverage["exact_coverage_pass"] else "FAIL"],

        ["execution_readable", count(exec_df, "readable"), 840, "PASS" if count(exec_df, "readable") == 840 else "FAIL"],

        ["execution_schema_pass", count(exec_df, "schema_pass"), 840, "PASS" if count(exec_df, "schema_pass") == 840 else "FAIL"],

        ["execution_rows_pass", count(exec_df, "row_count_pass"), 840, "PASS" if count(exec_df, "row_count_pass") == 840 else "FAIL"],

        ["mi_csv_files", len(mi_df), 525, "PASS" if len(mi_df) == 525 else "FAIL"],

        ["mi_exact_coverage", int(mi_coverage["actual_slots"]), mi_coverage["expected_slots"],

         "PASS" if mi_coverage["exact_coverage_pass"] else "FAIL"],

        ["mi_readable", count(mi_df, "readable"), 525, "PASS" if count(mi_df, "readable") == 525 else "FAIL"],

        ["mi_schema_pass", count(mi_df, "schema_pass"), 525, "PASS" if count(mi_df, "schema_pass") == 525 else "FAIL"],

        ["mi_rows_pass", count(mi_df, "row_count_pass"), 525, "PASS" if count(mi_df, "row_count_pass") == 525 else "FAIL"],

        ["bioz_spec_files", len(bioz_df), EXPECTED_BZ_SPEC_FILES, "PASS" if len(bioz_df) == EXPECTED_BZ_SPEC_FILES else "FAIL"],

        ["bioz_readable", count(bioz_df, "readable"), EXPECTED_BZ_SPEC_FILES, "PASS" if count(bioz_df, "readable") == EXPECTED_BZ_SPEC_FILES else "FAIL"],

        ["bioz_header_pass", count(bioz_df, "header_pass"), EXPECTED_BZ_SPEC_FILES, "PASS" if count(bioz_df, "header_pass") == EXPECTED_BZ_SPEC_FILES else "FAIL"],

        ["bioz_measurement_header_pass", count(bioz_df, "measurement_header_pass"), EXPECTED_BZ_SPEC_FILES,

         "PASS" if count(bioz_df, "measurement_header_pass") == EXPECTED_BZ_SPEC_FILES else "FAIL"],

        ["bioz_paired_units", count(pair_df, "paired"), EXPECTED_BZ_UNITS, "PASS" if count(pair_df, "paired") == EXPECTED_BZ_UNITS else "FAIL"],

        ["bioz_exact_15_units", bz_exact_pass, EXPECTED_BZ_UNITS, "PASS" if bz_exact_pass == EXPECTED_BZ_UNITS else "FAIL"],

        ["empty_logs", count(logs_df, "empty"), 0, "PASS" if count(logs_df, "empty") == 0 else "WARNING"],

        ["duplicate_content_groups", len(duplicate_df["sha256"].unique()) if len(duplicate_df) else 0, 0,

         "PASS" if len(duplicate_df) == 0 else "WARNING"],

    ]

    return pd.DataFrame(rows, columns=["check", "actual", "expected", "status"])

def main():

    setup_logging()

    t0 = time.time()

    print("=" * 100)

    print("DATA2 DATASET INTEGRITY & QUALITY ASSURANCE")

    print("=" * 100)

    log("Protocol version: DATA2-FINAL-FREEZE-v4.0")

    log("READ-ONLY MODE: raw data will not be modified.")

    log("Timestamp gaps/repeats are QC warnings only; raw timestamps are preserved.")

    log("Processing time axis: row order + nominal 125 Hz.")

    log("PROTOCOL: Motor Execution and Motor Imagery were recorded at separate times.")

    log("Therefore DATA2 performs timestamp checks within each recording only;")

    log("it does NOT require Execution and MI timestamps to align.")

    for label, p in [("ROOT", ROOT), ("RAW", RAW), ("EXECUTION", EXECUTION),

                     ("IMAGERY", IMAGERY), ("BIOIMPEDANCE", BIOZ)]:

        if not p.exists():

            raise FileNotFoundError(f"{label} path not found: {p}")

        log(f"FOUND | {label:<12} | {p}")

    QC.mkdir(parents=True, exist_ok=True)

    INTERNAL_QC.mkdir(parents=True, exist_ok=True)

    exec_df = audit_csv_collection(EXECUTION, "MOTOR_EXECUTION")

    mi_df = audit_csv_collection(IMAGERY, "MOTOR_IMAGERY")

    bioz_df = audit_bioz_collection(BIOZ)

    log_files = sorted([p for p in RAW.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".log"}])

    log_df = log_inventory()

    log_recon_df = log_reconciliation(list(EXECUTION.rglob("*.csv")), list(IMAGERY.rglob("*.csv")), log_files)

    pair_df = bioz_pairing_audit(BIOZ)

    # Exact recording-slot coverage. This is a strict structural check.

    exec_files = sorted(EXECUTION.rglob("*.csv"))

    mi_files = sorted(IMAGERY.rglob("*.csv"))

    exec_coverage = exact_csv_coverage_audit(exec_files, "MOTOR_EXECUTION")

    mi_coverage = exact_csv_coverage_audit(mi_files, "MOTOR_IMAGERY")

    # Exact BioZ structural rule: 18 BZ IDs × 7 gestures × 15 units, each with both channels.

    bz_expected_rows = []

    bz_present = set()

    for _, r in pair_df.iterrows():

        if bool(r.get("paired", False)):

            bz_present.add((r["bz_id"], r["gesture_folder"], r["recording_unit"]))

    for bz_i in range(1, 19):

        bz = f"BZ_{bz_i:02d}"

        for gname in GESTURES:

            code = gesture_code(gname)

            for unit in range(1, 16):

                key = (bz, f"{bz}_{code}", f"set_{unit:02d}")

                bz_expected_rows.append({"bz_id": bz, "gesture_folder": f"{bz}_{code}",

                                         "recording_unit": f"set_{unit:02d}",

                                         "expected_two_channel_pair": key in bz_present})

    bz_exact_df = pd.DataFrame(bz_expected_rows)

    bz_exact_pass = int(bz_exact_df["expected_two_channel_pair"].sum())

    write_csv(bz_exact_df, INTERNAL_QC / "DATA2_bioz_exact_15_units_validation.csv")

    all_raw = sorted([p for p in RAW.rglob("*") if p.is_file()])

    allowed = {".csv", ".txt", ".log", ".spec"}

    unexpected = [p for p in all_raw if p.suffix.lower() not in allowed]

    unexpected_df = pd.DataFrame(

        [{"file": str(p.relative_to(ROOT)), "suffix": p.suffix.lower()} for p in unexpected],

        columns=["file", "suffix"],

    )

    dup_df = duplicate_hash_audit(sorted(

        list(EXECUTION.rglob("*.csv")) +

        list(IMAGERY.rglob("*.csv")) +

        list(BIOZ.rglob("*.spec"))

    ))

    write_csv(exec_df, INTERNAL_QC / "DATA2_execution_file_integrity.csv")

    write_csv(mi_df, INTERNAL_QC / "DATA2_imagery_file_integrity.csv")

    write_csv(bioz_df, INTERNAL_QC / "DATA2_bioz_file_integrity.csv")

    write_csv(pair_df, INTERNAL_QC / "DATA2_bioz_pairing_validation.csv")

    write_csv(log_df, INTERNAL_QC / "DATA2_log_file_inventory.csv")

    write_csv(log_recon_df, INTERNAL_QC / "DATA2_log_reconciliation.csv")

    write_csv(unexpected_df, INTERNAL_QC / "DATA2_unexpected_file_check.csv")

    write_csv(dup_df, INTERNAL_QC / "DATA2_duplicate_content_check.csv")

    schema_df = pd.concat([

        exec_df[["file", "schema_pass"]].assign(modality="MOTOR_EXECUTION"),

        mi_df[["file", "schema_pass"]].assign(modality="MOTOR_IMAGERY"),

    ], ignore_index=True)

    write_csv(schema_df, INTERNAL_QC / "DATA2_csv_schema_validation.csv")

    row_df = pd.concat([

        exec_df[["file", "row_count_pass"]].assign(modality="MOTOR_EXECUTION"),

        mi_df[["file", "row_count_pass"]].assign(modality="MOTOR_IMAGERY"),

    ], ignore_index=True)

    write_csv(row_df, INTERNAL_QC / "DATA2_row_count_validation.csv")

    num_cols = [

        "modality", "file", "nan_count", "inf_count",

        "constant_channels", "near_constant_channels",

        "high_repeat_channels",

    ]

    num_df = pd.concat([exec_df[num_cols], mi_df[num_cols]], ignore_index=True)

    write_csv(num_df, INTERNAL_QC / "DATA2_numeric_quality.csv")

    time_cols = [

        "modality", "file", "timestamp_parse_pass", "timestamp_monotonic",

        "duplicate_timestamps", "timestamp_negative_steps",

        "timestamp_large_gaps", "timestamp_median_step_sec", "duration_sec",

    ]

    time_df = pd.concat([exec_df[time_cols], mi_df[time_cols]], ignore_index=True)

    write_csv(time_df, INTERNAL_QC / "DATA2_timestamp_quality.csv")

    idx_cols = [

        "modality", "file", "sample_index_numeric", "sample_index_monotonic",

        "sample_index_duplicates", "sample_index_negative_steps",

        "sample_index_expected_wraps", "sample_index_unexplained_negative_steps",

        "sample_index_gaps", "sample_index_median_step",

    ]

    idx_df = pd.concat([exec_df[idx_cols], mi_df[idx_cols]], ignore_index=True)

    write_csv(idx_df, INTERNAL_QC / "DATA2_sample_index_quality.csv")

    summary = summary_rows(

        exec_df, mi_df, bioz_df, pair_df, log_df, dup_df,

        exec_coverage, mi_coverage, bz_exact_pass,

    )

    write_csv(summary, QC / "DATA2_INTEGRITY_SUMMARY.csv")

    modality_summary = pd.DataFrame([

        {

            "modality": "MOTOR_EXECUTION", "recordings": len(exec_df), "expected_recordings": 840,

            "exact_coverage_pass": exec_coverage["exact_coverage_pass"],

            "readable_pass": int((exec_df["readable"] == True).sum()) == 840,

            "schema_pass": int((exec_df["schema_pass"] == True).sum()) == 840,

            "row_count_pass": int((exec_df["row_count_pass"] == True).sum()) == 840,

            "timestamp_warning_files": int((exec_df["timestamp_large_gaps"] > 0).sum()),

            "constant_channel_files": int((exec_df["constant_channels"] != "").sum()),

            "status": "PASS" if exec_coverage["exact_coverage_pass"] else "FAIL",

        },

        {

            "modality": "MOTOR_IMAGERY", "recordings": len(mi_df), "expected_recordings": 525,

            "exact_coverage_pass": mi_coverage["exact_coverage_pass"],

            "readable_pass": int((mi_df["readable"] == True).sum()) == 525,

            "schema_pass": int((mi_df["schema_pass"] == True).sum()) == 525,

            "row_count_pass": int((mi_df["row_count_pass"] == True).sum()) == 525,

            "timestamp_warning_files": int((mi_df["timestamp_large_gaps"] > 0).sum()),

            "constant_channel_files": int((mi_df["constant_channels"] != "").sum()),

            "status": "PASS" if mi_coverage["exact_coverage_pass"] else "FAIL",

        },

        {

            "modality": "BIOIMPEDANCE", "recordings": bz_exact_pass, "expected_recordings": EXPECTED_BZ_UNITS,

            "exact_coverage_pass": bz_exact_pass == EXPECTED_BZ_UNITS,

            "readable_pass": int((bioz_df["readable"] == True).sum()) == EXPECTED_BZ_SPEC_FILES,

            "schema_pass": int((bioz_df["header_pass"] == True).sum()) == EXPECTED_BZ_SPEC_FILES,

            "row_count_pass": int((bioz_df["measurement_header_pass"] == True).sum()) == EXPECTED_BZ_SPEC_FILES,

            "timestamp_warning_files": 0, "constant_channel_files": 0,

            "status": "PASS" if bz_exact_pass == EXPECTED_BZ_UNITS else "FAIL",

        },

    ])

    write_csv(modality_summary, QC / "DATA2_MODALITY_QC_SUMMARY.csv")

    timestamp_summary = pd.DataFrame([

        {

            "modality": "MOTOR_EXECUTION", "files": len(exec_df),

            "timestamp_parse_fail_files": int((exec_df["timestamp_parse_pass"] == False).sum()),

            "negative_step_files": int((exec_df["timestamp_negative_steps"] > 0).sum()),

            "files_with_gap_over_50ms": int((exec_df["timestamp_large_gaps"] > 0).sum()),

            "total_duplicate_timestamp_rows": int(exec_df["duplicate_timestamps"].fillna(0).sum()),

            "max_duration_sec": float(exec_df["duration_sec"].max()) if len(exec_df) else np.nan,

            "policy": "RAW_PRESERVED; QC_WARNING; PROCESSING_ROW_ORDER_PLUS_125HZ",

        },

        {

            "modality": "MOTOR_IMAGERY", "files": len(mi_df),

            "timestamp_parse_fail_files": int((mi_df["timestamp_parse_pass"] == False).sum()),

            "negative_step_files": int((mi_df["timestamp_negative_steps"] > 0).sum()),

            "files_with_gap_over_50ms": int((mi_df["timestamp_large_gaps"] > 0).sum()),

            "total_duplicate_timestamp_rows": int(mi_df["duplicate_timestamps"].fillna(0).sum()),

            "max_duration_sec": float(mi_df["duration_sec"].max()) if len(mi_df) else np.nan,

            "policy": "RAW_PRESERVED; QC_WARNING; PROCESSING_ROW_ORDER_PLUS_125HZ",

        },

    ])

    write_csv(timestamp_summary, QC / "DATA2_TIMESTAMP_QC_SUMMARY.csv")

    readme = """DATA2 — DATASET INTEGRITY & QUALITY ASSURANCE

FINAL FREEZE: DATA2-FINAL-FREEZE-v4.0

Raw data are read-only. Original formatted timestamps and Sample Index values are preserved. Timestamp repeats/gaps are QC observations only. Execution Subjects 01–17 use a documented wrapping acquisition counter; the observed -254 transition is classified as expected acquisition behavior. Raw Sample Index values are never modified. Known constant EEG_ch-11 recordings identified in the frozen audit are retained and documented as acquisition-quality observations. Downstream uniform processing time, when required, is derived from row order and nominal 125 Hz; it is not written back to raw data. Motor Execution and Motor Imagery were recorded separately and are not cross-modally synchronized.

Strict checks include exact recording-slot coverage, readability, schema, row counts, filename/folder semantics, numeric validity, timestamp parseability, Sample Index numeric validity, BioZ structure/pairing, required logs, and duplicate-content detection. Detailed audits are retained under 05_QC/_INTERNAL_DATA2/.

"""

    (QC / "DATA2_QC_README.txt").write_text(readme, encoding="utf-8")

    log("WROTE | 05_QC/DATA2_QC_README.txt")

    # Strict failures: fundamental file/schema/readability/row/BioZ pairing issues.

    strict_fail = (

        (summary["status"] == "FAIL").any()

        or (exec_df["nan_count"] > 0).any()

        or (exec_df["inf_count"] > 0).any()

        or (mi_df["nan_count"] > 0).any()

        or (mi_df["inf_count"] > 0).any()

        or (bioz_df["invalid_data_rows"] > 0).any()

        or (bioz_df["re_nan"] > 0).any()

        or (bioz_df["im_nan"] > 0).any()

        or bz_exact_pass != EXPECTED_BZ_UNITS

        or not exec_coverage["exact_coverage_pass"]

        or not mi_coverage["exact_coverage_pass"]

        or (log_recon_df["status"] == "MISSING").any()

        or (exec_df["filename_semantics_pass"] == False).any()

        or (mi_df["filename_semantics_pass"] == False).any()

        or (exec_df["timestamp_parse_pass"] == False).any()

        or (mi_df["timestamp_parse_pass"] == False).any()

        or (exec_df["sample_index_numeric"] == False).any()

        or (mi_df["sample_index_numeric"] == False).any()

    )

    # Warnings are reported but do not fail DATA2:

    warnings = {

        "execution_timestamp_large_gap_files": int((exec_df["timestamp_large_gaps"] > 0).sum()),

        "mi_timestamp_large_gap_files": int((mi_df["timestamp_large_gaps"] > 0).sum()),

        "execution_constant_channel_files": int((exec_df["constant_channels"] != "").sum()),

        "mi_constant_channel_files": int((mi_df["constant_channels"] != "").sum()),

        "empty_logs": int((log_df["empty"]).sum()) if len(log_df) else 0,

        "duplicate_content_rows": len(dup_df),

        "unexpected_files": len(unexpected_df),

        "missing_logs": int((log_recon_df["status"] == "MISSING").sum()),

        "extra_logs": int((log_recon_df["status"] == "EXTRA").sum()),

        "execution_timestamp_negative_step_files": int((exec_df["timestamp_negative_steps"] > 0).sum()),

        "mi_timestamp_negative_step_files": int((mi_df["timestamp_negative_steps"] > 0).sum()),

        "execution_sample_index_negative_step_files": int((exec_df["sample_index_negative_steps"] > 0).sum()),

        "mi_sample_index_negative_step_files": int((mi_df["sample_index_negative_steps"] > 0).sum()),

        "execution_sample_index_expected_wrap_files": int((exec_df["sample_index_expected_wraps"] > 0).sum()),

        "execution_sample_index_unexplained_negative_step_files": int((exec_df["sample_index_unexplained_negative_steps"] > 0).sum()),

        "mi_sample_index_unexplained_negative_step_files": int((mi_df["sample_index_unexplained_negative_steps"] > 0).sum()),

        "execution_exact_coverage_parse_errors": int(exec_coverage["parse_errors"]),

        "mi_exact_coverage_parse_errors": int(mi_coverage["parse_errors"]),

        "execution_duplicate_slots": int(exec_coverage["duplicate_slots"]),

        "mi_duplicate_slots": int(mi_coverage["duplicate_slots"]),

        "execution_canonical_schema_files": int((exec_df["schema_label"] == "EXEC_CANONICAL_SUBJECT_01_40").sum()),

        "execution_noncanonical_schema_files": int((exec_df["schema_pass"] == False).sum()),
        "known_constant_execution_files": int((exec_df["known_acquisition_qc"] == True).sum()),
        "known_constant_mi_files": int((mi_df["known_acquisition_qc"] == True).sum()),

    }

    report = {

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "root": str(ROOT),

        "raw_data_modified": False,

        "execution_recorded_separately_from_mi": True,

        "execution_csv_count": len(exec_df),

        "execution_exact_coverage": exec_coverage,

        "mi_csv_count": len(mi_df),

        "mi_exact_coverage": mi_coverage,

        "bioz_spec_count": len(bioz_df),

        "bioz_exact_paired_units": bz_exact_pass,

        "log_count": len(log_files),

        "unexpected_file_count": len(unexpected_df),

        "duplicate_content_rows": len(dup_df),

        "warnings": warnings,

        "status": "FAIL" if strict_fail else (
            "PASS_WITH_WARNINGS"
            if any(
                v for k, v in warnings.items()
                if k not in {
                    "execution_sample_index_expected_wrap_files",
                    "known_constant_execution_files",
                    "known_constant_mi_files",
                }
            )
            else "PASS"
        ),

    }

    (QC / "DATA2_FINAL_REPORT.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 100)

    print("DATA2 FINAL SUMMARY")

    print("=" * 100)

    print(summary.to_string(index=False))

    print("\nWarnings:")

    for k, v in warnings.items():

        print(f"  {k}: {v}")

    print("\nExact recording coverage:")

    print(f"  Motor Execution: {exec_coverage['actual_slots']}/{exec_coverage['expected_slots']} | {'PASS' if exec_coverage['exact_coverage_pass'] else 'FAIL'}")

    print(f"  Motor Imagery:   {mi_coverage['actual_slots']}/{mi_coverage['expected_slots']} | {'PASS' if mi_coverage['exact_coverage_pass'] else 'FAIL'}")

    print("\nExecution schema counts:")

    print(f"  Canonical EEG_ch-01..EEG_ch-13: {int((exec_df['schema_label'] == 'EXEC_CANONICAL_SUBJECT_01_40').sum())}")

    print(f"  Non-canonical execution files:    {int((exec_df['schema_pass'] == False).sum())}")

    print("\nSample Index QC:")
    print(f"  Expected acquisition-wrap files: {int((exec_df['sample_index_expected_wraps'] > 0).sum())}")
    print(f"  Unexplained negative-step files: {int((exec_df['sample_index_unexplained_negative_steps'] > 0).sum())}")
    print("  Policy: expected -254 wraps classified; raw Sample Index preserved")

    print("\nLog reconciliation:")

    print(f"  Expected logs present: {int((log_recon_df['status'] == 'PRESENT').sum())}")

    print(f"  Missing logs: {int((log_recon_df['status'] == 'MISSING').sum())}")

    print(f"  Extra/unmatched logs: {int((log_recon_df['status'] == 'EXTRA').sum())}")

    print("\n" + "=" * 100)

    print(f"DATA2 STATUS: {report['status']}")

    print("RAW DATA MODIFICATION: NONE")

    print(f"QC OUTPUT: {QC}")

    print(f"Runtime: {time.time() - t0:.1f} seconds")

    print("=" * 100)

    return 1 if strict_fail else 0

if __name__ == "__main__":

    raise SystemExit(main())
