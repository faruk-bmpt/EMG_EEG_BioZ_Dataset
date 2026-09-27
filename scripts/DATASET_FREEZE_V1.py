#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DATASET V1.0 FREEZE GATE
===============================================================================

Purpose
-------
Final validation gate before SHA256 checksum generation and release packaging.

This stage is READ-ONLY with respect to raw and processed signal data.

It validates:
    01_RAW_DATA
    02_MODALITY_VIEWS
    03_METADATA
    04_ANNOTATIONS
    05_QC
    06_PROCESSED_DATA

It does NOT:
    - modify raw data
    - modify processed signals
    - rerun preprocessing
    - perform feature extraction
    - perform machine learning
    - create inferred annotations
    - create event/repetition annotations
    - create fusion data
    - rename recordings
    - rename raw folders
===============================================================================
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# =============================================================================
# DATASET ROOT
# =============================================================================

ROOT = Path(
    "/mnt/f/Faruk/OFS_Paper_Work/Data_Set_Paper_Work/"
    "EEG_EMG_BIOZ_DATASET"
).resolve()

RAW = ROOT / "01_RAW_DATA"
MODALITY = ROOT / "02_MODALITY_VIEWS"
META = ROOT / "03_METADATA"
ANNOTATIONS = ROOT / "04_ANNOTATIONS"
QC = ROOT / "05_QC"
PROCESSED = ROOT / "06_PROCESSED_DATA"

FREEZE_QC = QC / "DATASET_FREEZE"

REPORT_JSON = FREEZE_QC / "DATASET_FREEZE_FINAL_REPORT.json"
MANIFEST_CSV = FREEZE_QC / "DATASET_FREEZE_MANIFEST.csv"
SUMMARY_TXT = FREEZE_QC / "DATASET_FREEZE_SUMMARY.txt"
SENTINEL = FREEZE_QC / "_DATASET_V1.0_FROZEN.json"


# =============================================================================
# FROZEN CONTRACT
# =============================================================================

DATASET_VERSION = "V1.0"
PROTOCOL_VERSION = "DATASET-FREEZE-V1.0-FINAL"

EXPECTED_EXECUTION = 840
EXPECTED_MI = 525
EXPECTED_BIOZ_UNITS = 1890
EXPECTED_BIOZ_FILES = 3780

EXPECTED_EXECUTION_EEG_NPZ = 840
EXPECTED_MI_EEG_NPZ = 525
EXPECTED_EMG_NPZ = 840
EXPECTED_BIOZ_NPZ = 3780

EXPECTED_TOTAL_PROCESSED_NPZ = 5985
EXPECTED_ANNOTATIONS = 3255

EXPECTED_RETAINED_SUBJECTS = 40
EXPECTED_EXCLUDED_SUBJECTS = 1


# =============================================================================
# UTILITY
# =============================================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def list_files(root: Path, pattern: str):
    if not root.exists():
        return []
    return sorted(root.rglob(pattern))


def add_check(
    records,
    category,
    item,
    path,
    passed,
    details,
):
    records.append(
        {
            "category": category,
            "item": item,
            "path": path,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


# =============================================================================
# 1. STRUCTURE
# =============================================================================

def validate_structure(records):

    required = [
        ROOT / "01_RAW_DATA",
        ROOT / "02_MODALITY_VIEWS",
        ROOT / "03_METADATA",
        ROOT / "04_ANNOTATIONS",
        ROOT / "05_QC",
        ROOT / "06_PROCESSED_DATA",

        RAW / "BIOIMPEDANCE",
        RAW / "EEG_MOTOR_IMAGERY",
        RAW / "EMG_EEG_SYNCHRONIZED",

        MODALITY / "EEG",
        MODALITY / "EMG",
        MODALITY / "BIOIMPEDANCE",

        PROCESSED / "BIOIMPEDANCE",
        PROCESSED / "EEG_MOTOR_EXECUTION",
        PROCESSED / "EEG_MOTOR_IMAGERY",
        PROCESSED / "EMG",
        PROCESSED / "PRE1_QC",
    ]

    passed = True

    for path in required:
        ok = path.exists()

        add_check(
            records,
            "structure",
            f"Required directory: {path.relative_to(ROOT)}",
            str(path.relative_to(ROOT)),
            ok,
            "exists" if ok else "missing",
        )

        passed &= ok

    return passed


# =============================================================================
# 2. RAW INVENTORY
# =============================================================================

def validate_raw_inventory(records):

    execution = list_files(
        RAW / "EMG_EEG_SYNCHRONIZED",
        "*.csv",
    )

    imagery = list_files(
        RAW / "EEG_MOTOR_IMAGERY",
        "*.csv",
    )

    bioz = list_files(
        RAW / "BIOIMPEDANCE",
        "*.spec",
    )

    checks = [
        (
            "Motor execution raw CSV",
            len(execution),
            EXPECTED_EXECUTION,
            "01_RAW_DATA/EMG_EEG_SYNCHRONIZED",
        ),
        (
            "Motor imagery raw CSV",
            len(imagery),
            EXPECTED_MI,
            "01_RAW_DATA/EEG_MOTOR_IMAGERY",
        ),
        (
            "BioZ native .spec",
            len(bioz),
            EXPECTED_BIOZ_FILES,
            "01_RAW_DATA/BIOIMPEDANCE",
        ),
    ]

    passed = True

    for name, actual, expected, path in checks:

        ok = actual == expected

        add_check(
            records,
            "raw_inventory",
            name,
            path,
            ok,
            f"actual={actual}, expected={expected}",
        )

        passed &= ok

    return passed


# =============================================================================
# 3. PROCESSED INVENTORY
# =============================================================================

def validate_processed_inventory(records):

    checks = [
        (
            "Execution EEG NPZ",
            PROCESSED / "EEG_MOTOR_EXECUTION",
            EXPECTED_EXECUTION_EEG_NPZ,
        ),
        (
            "Motor imagery EEG NPZ",
            PROCESSED / "EEG_MOTOR_IMAGERY",
            EXPECTED_MI_EEG_NPZ,
        ),
        (
            "EMG NPZ",
            PROCESSED / "EMG",
            EXPECTED_EMG_NPZ,
        ),
        (
            "BioZ NPZ",
            PROCESSED / "BIOIMPEDANCE",
            EXPECTED_BIOZ_NPZ,
        ),
    ]

    passed = True
    total = 0

    for name, path, expected in checks:

        actual = count_files(path, "*.npz")
        total += actual

        ok = actual == expected

        add_check(
            records,
            "processed_inventory",
            name,
            str(path.relative_to(ROOT)),
            ok,
            f"actual={actual}, expected={expected}",
        )

        passed &= ok

    total_ok = total == EXPECTED_TOTAL_PROCESSED_NPZ

    add_check(
        records,
        "processed_inventory",
        "Total processed NPZ",
        "06_PROCESSED_DATA",
        total_ok,
        f"actual={total}, expected={EXPECTED_TOTAL_PROCESSED_NPZ}",
    )

    passed &= total_ok

    return passed


# =============================================================================
# 4. MODALITY VIEWS
# =============================================================================

def validate_modality_views(records):

    expected = [
        (
            MODALITY / "EEG" / "EEG_MOTOR_EXECUTION_manifest.csv",
            EXPECTED_EXECUTION,
        ),
        (
            MODALITY / "EEG" / "EEG_MOTOR_IMAGERY_manifest.csv",
            EXPECTED_MI,
        ),
        (
            MODALITY / "EMG" / "EMG_manifest.csv",
            EXPECTED_EXECUTION,
        ),
        (
            MODALITY / "BIOIMPEDANCE" / "BIOIMPEDANCE_manifest.csv",
            EXPECTED_BIOZ_FILES,
        ),
    ]

    passed = True

    for path, expected_rows in expected:

        if not path.exists():

            add_check(
                records,
                "modality_views",
                path.name,
                str(path.relative_to(ROOT)),
                False,
                "file missing",
            )

            passed = False
            continue

        try:
            df = pd.read_csv(path)
            actual_rows = len(df)

            ok = actual_rows == expected_rows

            add_check(
                records,
                "modality_views",
                path.name,
                str(path.relative_to(ROOT)),
                ok,
                f"rows={actual_rows}, expected={expected_rows}",
            )

            passed &= ok

        except Exception as exc:

            add_check(
                records,
                "modality_views",
                path.name,
                str(path.relative_to(ROOT)),
                False,
                f"read error: {exc}",
            )

            passed = False

    return passed


# =============================================================================
# 5. METADATA
# =============================================================================

def validate_metadata(records):

    required = [
        "participants.csv",
        "subject_crosswalk.csv",
        "subject_modality_availability.csv",
        "recording_inventory.csv",
        "recording_parameters.csv",
        "acquisition_parameters.csv",
        "gesture_dictionary.csv",
        "data_provenance.csv",
        "cohort_definition.csv",
        "exclusion_log.csv",
    ]

    passed = True

    for filename in required:

        path = META / filename
        ok = path.exists()

        add_check(
            records,
            "metadata",
            filename,
            str(path.relative_to(ROOT)),
            ok,
            "exists" if ok else "missing",
        )

        passed &= ok

    return passed


# =============================================================================
# 6. COHORT
# =============================================================================

def validate_cohort(records):

    path = META / "participants.csv"

    if not path.exists():

        add_check(
            records,
            "cohort",
            "participants.csv",
            str(path.relative_to(ROOT)),
            False,
            "missing",
        )

        return False

    try:

        df = pd.read_csv(path)

        if "cohort_status" not in df.columns:
            add_check(
                records,
                "cohort",
                "cohort_status field",
                str(path.relative_to(ROOT)),
                False,
                "column missing",
            )
            return False

        status = (
            df["cohort_status"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        retained = int((status == "retained").sum())
        excluded = int((status == "excluded").sum())

        retained_ok = retained == EXPECTED_RETAINED_SUBJECTS
        excluded_ok = excluded == EXPECTED_EXCLUDED_SUBJECTS

        add_check(
            records,
            "cohort",
            "Retained participants",
            str(path.relative_to(ROOT)),
            retained_ok,
            f"actual={retained}, expected={EXPECTED_RETAINED_SUBJECTS}",
        )

        add_check(
            records,
            "cohort",
            "Excluded participants",
            str(path.relative_to(ROOT)),
            excluded_ok,
            f"actual={excluded}, expected={EXPECTED_EXCLUDED_SUBJECTS}",
        )

        passed = retained_ok and excluded_ok

        # Subject 41 must remain explicitly excluded.
        if "subject_id" in df.columns:

            row = df[
                df["subject_id"].astype(str).str.strip()
                == "Subject_41"
            ]

            subject41_ok = len(row) == 1

            if subject41_ok:
                subject41_status = (
                    str(row.iloc[0]["cohort_status"])
                    .strip()
                    .lower()
                )

                subject41_ok = (
                    subject41_status == "excluded"
                )

                details = (
                    f"cohort_status={subject41_status}"
                )
            else:
                details = (
                    f"records found={len(row)}"
                )

            add_check(
                records,
                "cohort",
                "Subject_41 exclusion",
                str(path.relative_to(ROOT)),
                subject41_ok,
                details,
            )

            passed &= subject41_ok

        return passed

    except Exception as exc:

        add_check(
            records,
            "cohort",
            "Cohort validation",
            str(path.relative_to(ROOT)),
            False,
            f"error: {exc}",
        )

        return False


# =============================================================================
# 7. ANNOTATION MANIFEST
# =============================================================================

def validate_annotation_manifest(records):

    manifest = (
        ANNOTATIONS
        / "RECORDING_LEVEL_ANNOTATION_MANIFEST.csv"
    )

    required = [
        "RECORDING_LEVEL_ANNOTATION_MANIFEST.csv",
        "ANNOTATION_POLICY.md",
        "ANNOTATION_SCHEMA.csv",
        "ANNOTATION_POLICY_FINAL_REPORT.json",
    ]

    passed = True

    for filename in required:

        path = ANNOTATIONS / filename
        ok = path.exists()

        add_check(
            records,
            "annotations",
            filename,
            str(path.relative_to(ROOT)),
            ok,
            "exists" if ok else "missing",
        )

        passed &= ok

    if not manifest.exists():
        return False

    try:

        df = pd.read_csv(manifest)

        checks = []

        checks.append(
            (
                "Annotation rows",
                len(df) == EXPECTED_ANNOTATIONS,
                f"actual={len(df)}, expected={EXPECTED_ANNOTATIONS}",
            )
        )

        unique_ids = (
            df["annotation_id"].nunique()
            if "annotation_id" in df.columns
            else -1
        )

        checks.append(
            (
                "Unique annotation IDs",
                unique_ids == EXPECTED_ANNOTATIONS,
                f"actual={unique_ids}, expected={EXPECTED_ANNOTATIONS}",
            )
        )

        for column, label in [
            (
                "event_annotation_available",
                "Event annotations",
            ),
            (
                "trigger_annotation_available",
                "Trigger annotations",
            ),
            (
                "repetition_annotation_available",
                "Repetition annotations",
            ),
            (
                "annotation_inferred",
                "Annotation inference",
            ),
        ]:

            if column not in df.columns:

                checks.append(
                    (
                        label,
                        False,
                        f"column missing: {column}",
                    )
                )

            else:

                values = (
                    df[column]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                checks.append(
                    (
                        label,
                        values.eq("NO").all(),
                        "all NO",
                    )
                )

        for column, label in [
            ("event_id", "Event IDs fabricated"),
            ("repetition_id", "Repetition IDs fabricated"),
        ]:

            if column not in df.columns:

                checks.append(
                    (
                        label,
                        False,
                        f"column missing: {column}",
                    )
                )

            else:

                values = (
                    df[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                checks.append(
                    (
                        label,
                        values.eq("").all(),
                        "all blank",
                    )
                )

        for name, ok, details in checks:

            add_check(
                records,
                "annotations",
                name,
                str(manifest.relative_to(ROOT)),
                ok,
                details,
            )

            passed &= ok

        return passed

    except Exception as exc:

        add_check(
            records,
            "annotations",
            "Annotation manifest validation",
            str(manifest.relative_to(ROOT)),
            False,
            f"error: {exc}",
        )

        return False


# =============================================================================
# 8. ANNOTATION POLICY JSON
# =============================================================================

def validate_annotation_policy_json(records):

    path = (
        ANNOTATIONS
        / "ANNOTATION_POLICY_FINAL_REPORT.json"
    )

    if not path.exists():
        return False

    try:

        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        checks = [
            (
                "Annotation policy type",
                report.get("annotation_policy")
                == "RECORDING_LEVEL_GESTURE_ONLY",
                str(report.get("annotation_policy")),
            ),
            (
                "Motor execution annotation rows",
                report.get("motor_execution_annotation_rows")
                == 840,
                str(report.get("motor_execution_annotation_rows")),
            ),
            (
                "Motor imagery annotation rows",
                report.get("motor_imagery_annotation_rows")
                == 525,
                str(report.get("motor_imagery_annotation_rows")),
            ),
            (
                "BioZ annotation rows",
                report.get("bioimpedance_annotation_rows")
                == 1890,
                str(report.get("bioimpedance_annotation_rows")),
            ),
            (
                "Total annotation rows",
                report.get("total_annotation_rows")
                == 3255,
                str(report.get("total_annotation_rows")),
            ),
            (
                "Event annotations unavailable",
                report.get("event_annotations_available")
                is False,
                str(report.get("event_annotations_available")),
            ),
            (
                "Trigger annotations unavailable",
                report.get("trigger_annotations_available")
                is False,
                str(report.get("trigger_annotations_available")),
            ),
            (
                "Repetition annotations unavailable",
                report.get("repetition_annotations_available")
                is False,
                str(report.get("repetition_annotations_available")),
            ),
            (
                "Annotation inference disabled",
                report.get("annotation_inferred")
                is False,
                str(report.get("annotation_inferred")),
            ),
            (
                "Event IDs not fabricated",
                report.get("event_ids_fabricated")
                is False,
                str(report.get("event_ids_fabricated")),
            ),
            (
                "Repetition IDs not fabricated",
                report.get("repetition_ids_fabricated")
                is False,
                str(report.get("repetition_ids_fabricated")),
            ),
            (
                "Raw data unmodified",
                report.get("raw_data_modified")
                is False,
                str(report.get("raw_data_modified")),
            ),
            (
                "Processed data unmodified",
                report.get("processed_signal_data_modified")
                is False,
                str(report.get("processed_signal_data_modified")),
            ),
            (
                "Preprocessing not rerun",
                report.get("preprocessing_rerun")
                is False,
                str(report.get("preprocessing_rerun")),
            ),
            (
                "Feature extraction not performed",
                report.get("feature_extraction_performed")
                is False,
                str(report.get("feature_extraction_performed")),
            ),
            (
                "Machine learning not performed",
                report.get("machine_learning_performed")
                is False,
                str(report.get("machine_learning_performed")),
            ),
        ]

        passed = True

        for name, ok, details in checks:

            add_check(
                records,
                "annotation_policy",
                name,
                str(path.relative_to(ROOT)),
                ok,
                details,
            )

            passed &= ok

        return passed

    except Exception as exc:

        add_check(
            records,
            "annotation_policy",
            "Annotation policy JSON validation",
            str(path.relative_to(ROOT)),
            False,
            f"error: {exc}",
        )

        return False


# =============================================================================
# 9. GLOBAL QC
# =============================================================================

def validate_global_qc(records):

    path = QC / "DATASET_GLOBAL_QC_REPORT.json"

    if not path.exists():

        add_check(
            records,
            "global_qc",
            "Global QC report",
            str(path.relative_to(ROOT)),
            False,
            "missing",
        )

        return False

    try:

        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        # EXACT FIELD USED BY THE ACTUAL GLOBAL QC REPORT.
        overall_status = str(
            report.get("overall_status", "")
        ).upper()

        status_ok = overall_status == "PASS"

        add_check(
            records,
            "global_qc",
            "Global QC overall status",
            str(path.relative_to(ROOT)),
            status_ok,
            f"overall_status={overall_status}",
        )

        inventory = report.get("inventory", {})

        inventory_checks = [
            (
                "Execution raw inventory",
                inventory.get("execution_raw_csv") == 840,
                inventory.get("execution_raw_csv"),
            ),
            (
                "MI raw inventory",
                inventory.get("mi_raw_csv") == 525,
                inventory.get("mi_raw_csv"),
            ),
            (
                "BioZ raw inventory",
                inventory.get("bioz_raw_spec") == 3780,
                inventory.get("bioz_raw_spec"),
            ),
            (
                "Execution EEG NPZ inventory",
                inventory.get("execution_eeg_npz") == 840,
                inventory.get("execution_eeg_npz"),
            ),
            (
                "MI EEG NPZ inventory",
                inventory.get("mi_eeg_npz") == 525,
                inventory.get("mi_eeg_npz"),
            ),
            (
                "EMG NPZ inventory",
                inventory.get("emg_npz") == 840,
                inventory.get("emg_npz"),
            ),
            (
                "BioZ NPZ inventory",
                inventory.get("bioz_npz") == 3780,
                inventory.get("bioz_npz"),
            ),
        ]

        inventory_pass = True

        for name, ok, actual in inventory_checks:

            add_check(
                records,
                "global_qc",
                name,
                str(path.relative_to(ROOT)),
                ok,
                f"actual={actual}",
            )

            inventory_pass &= ok

        checks = report.get("checks", {})

        required_checks = [
            "execution_raw",
            "eeg_emg_pairing",
            "processed_structure",
            "bioz_crosswalk",
            "motor_imagery_inventory",
            "duplicate_content",
            "metadata_presence",
        ]

        component_pass = True

        for key in required_checks:

            value = str(
                checks.get(key, "")
            ).upper()

            ok = value == "PASS"

            add_check(
                records,
                "global_qc",
                f"Global QC component: {key}",
                str(path.relative_to(ROOT)),
                ok,
                f"status={value}",
            )

            component_pass &= ok

        raw_modified_ok = (
            report.get("raw_data_modified") is False
        )

        add_check(
            records,
            "global_qc",
            "Global QC raw-data modification",
            str(path.relative_to(ROOT)),
            raw_modified_ok,
            f"raw_data_modified={report.get('raw_data_modified')}",
        )

        return (
            status_ok
            and inventory_pass
            and component_pass
            and raw_modified_ok
        )

    except Exception as exc:

        add_check(
            records,
            "global_qc",
            "Global QC report validation",
            str(path.relative_to(ROOT)),
            False,
            f"error: {exc}",
        )

        return False


# =============================================================================
# 10. ANNOTATION AUDIT
# =============================================================================
def validate_annotation_audit(records):

    path = (
        QC
        / "ANNOTATION_AUDIT"
        / "ANNOTATION_AUDIT_FINAL_REPORT.json"
    )

    if not path.exists():

        add_check(
            records,
            "annotation_audit",
            "Annotation audit report",
            str(path.relative_to(ROOT)),
            False,
            "missing",
        )

        return False

    try:

        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        # ---------------------------------------------------------
        # Validate the actual annotation-audit schema
        # ---------------------------------------------------------

        positive_records = report.get("positive_records", None)
        error_records = report.get("error_records", None)

        raw_csv_annotation_fields = report.get(
            "raw_csv_annotation_fields",
            None,
        )

        processed_npz_annotation_keys = report.get(
            "processed_npz_annotation_keys",
            None,
        )

        annotations_created = report.get(
            "annotations_created",
            None,
        )

        repetition_events_inferred = report.get(
            "repetition_events_inferred",
            None,
        )

        event_ids_fabricated = report.get(
            "event_ids_fabricated",
            None,
        )

        # ---------------------------------------------------------
        # Scientific acceptance criteria
        #
        # No event/trigger/repetition evidence was found.
        # No annotations were created or inferred.
        # No event IDs were fabricated.
        # ---------------------------------------------------------

        ok = (
            positive_records == 0
            and error_records == 0
            and raw_csv_annotation_fields == 0
            and processed_npz_annotation_keys == 0
            and annotations_created is False
            and repetition_events_inferred is False
            and event_ids_fabricated is False
        )

        add_check(
            records,
            "annotation_audit",
            "Annotation audit integrity",
            str(path.relative_to(ROOT)),
            ok,
            (
                f"positive_records={positive_records}; "
                f"error_records={error_records}; "
                f"raw_csv_annotation_fields={raw_csv_annotation_fields}; "
                f"processed_npz_annotation_keys={processed_npz_annotation_keys}; "
                f"annotations_created={annotations_created}; "
                f"repetition_events_inferred={repetition_events_inferred}; "
                f"event_ids_fabricated={event_ids_fabricated}"
            ),
        )

        return ok

    except Exception as exc:

        add_check(
            records,
            "annotation_audit",
            "Annotation audit report",
            str(path.relative_to(ROOT)),
            False,
            f"error: {exc}",
        )

        return False

# =============================================================================
# 11. TEMP FILE CHECK
# =============================================================================

def validate_temporary_files(records):

    patterns = [
        "*.tmp",
        "*.partial",
        "*.part",
        "*~",
    ]

    found = []

    for pattern in patterns:
        found.extend(ROOT.rglob(pattern))

    found = [
        p for p in found
        if FREEZE_QC not in p.parents
    ]

    ok = len(found) == 0

    details = (
        "none found"
        if ok
        else "; ".join(
            str(p.relative_to(ROOT))
            for p in found[:20]
        )
    )

    add_check(
        records,
        "file_integrity",
        "Temporary files",
        ".",
        ok,
        details,
    )

    return ok


# =============================================================================
# 12. WRITE OUTPUTS
# =============================================================================

def write_outputs(records, overall_pass, runtime):

    FREEZE_QC.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(records)

    df.to_csv(
        MANIFEST_CSV,
        index=False,
        encoding="utf-8",
    )

    passed = int(
        (df["status"] == "PASS").sum()
    )

    failed = int(
        (df["status"] == "FAIL").sum()
    )

    summary = [
        "=" * 78,
        "DATASET V1.0 FREEZE SUMMARY",
        "=" * 78,
        f"Dataset version       : {DATASET_VERSION}",
        f"Protocol version      : {PROTOCOL_VERSION}",
        f"Freeze timestamp UTC  : {utc_now()}",
        f"Dataset root          : {ROOT}",
        "",
        "FROZEN DATASET COUNTS",
        "-" * 78,
        f"Motor execution CSV   : {EXPECTED_EXECUTION}",
        f"Motor imagery CSV     : {EXPECTED_MI}",
        f"BioZ measurement units: {EXPECTED_BIOZ_UNITS}",
        f"BioZ native .spec     : {EXPECTED_BIOZ_FILES}",
        "",
        f"Execution EEG NPZ     : {EXPECTED_EXECUTION_EEG_NPZ}",
        f"Motor imagery EEG NPZ : {EXPECTED_MI_EEG_NPZ}",
        f"EMG NPZ               : {EXPECTED_EMG_NPZ}",
        f"BioZ NPZ              : {EXPECTED_BIOZ_NPZ}",
        f"Total processed NPZ   : {EXPECTED_TOTAL_PROCESSED_NPZ}",
        "",
        f"Annotation rows       : {EXPECTED_ANNOTATIONS}",
        f"Retained subjects     : {EXPECTED_RETAINED_SUBJECTS}",
        f"Excluded subjects     : {EXPECTED_EXCLUDED_SUBJECTS}",
        "",
        "ANNOTATION POLICY",
        "-" * 78,
        "Event annotations     : NOT AVAILABLE",
        "Trigger annotations   : NOT AVAILABLE",
        "Repetition annotations: NOT AVAILABLE",
        "Annotation inference  : NO",
        "Event IDs fabricated  : NO",
        "Repetition IDs fabricated: NO",
        "",
        "FREEZE POLICY",
        "-" * 78,
        "Raw data modification : NONE",
        "Preprocessing rerun   : NO",
        "Feature extraction    : NO",
        "Machine learning      : NO",
        "Fusion generation     : NO",
        "Raw structure changed : NO",
        "Raw recordings renamed: NO",
        "",
        f"Freeze checks          : {len(records)}",
        f"Checks passed          : {passed}",
        f"Checks failed          : {failed}",
        f"Runtime seconds        : {runtime:.2f}",
        "",
        f"DATASET V1.0 FREEZE STATUS: "
        f"{'PASS' if overall_pass else 'FAIL'}",
        "=" * 78,
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    report = {
        "dataset_version": DATASET_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "freeze_timestamp_utc": utc_now(),
        "dataset_root": str(ROOT),

        "overall_status": (
            "PASS"
            if overall_pass
            else "FAIL"
        ),

        "checks": {
            "total": len(records),
            "passed": passed,
            "failed": failed,
        },

        "inventory": {
            "execution_raw_csv": EXPECTED_EXECUTION,
            "mi_raw_csv": EXPECTED_MI,
            "bioz_raw_spec": EXPECTED_BIOZ_FILES,
            "execution_eeg_npz": EXPECTED_EXECUTION_EEG_NPZ,
            "mi_eeg_npz": EXPECTED_MI_EEG_NPZ,
            "emg_npz": EXPECTED_EMG_NPZ,
            "bioz_npz": EXPECTED_BIOZ_NPZ,
            "total_processed_npz": EXPECTED_TOTAL_PROCESSED_NPZ,
            "annotation_rows": EXPECTED_ANNOTATIONS,
        },

        "cohort": {
            "retained_subjects": EXPECTED_RETAINED_SUBJECTS,
            "excluded_subjects": EXPECTED_EXCLUDED_SUBJECTS,
            "excluded_subject": "Subject_41",
        },

        "annotation_policy": {
            "event_annotations": "NOT_AVAILABLE",
            "trigger_annotations": "NOT_AVAILABLE",
            "repetition_annotations": "NOT_AVAILABLE",
            "annotation_inference": False,
            "event_ids_fabricated": False,
            "repetition_ids_fabricated": False,
        },

        "freeze_policy": {
            "raw_data_modified": False,
            "processed_data_modified": False,
            "preprocessing_rerun": False,
            "feature_extraction": False,
            "machine_learning": False,
            "fusion_generation": False,
        },

        "next_stage": (
            "SHA256_CHECKSUM_GENERATION"
            if overall_pass
            else "CORRECT_FREEZE_FAILURES"
        ),

        "python": sys.version,
        "platform": platform.platform(),
        "runtime_seconds": round(runtime, 3),

        "records": records,
    }

    with REPORT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    if overall_pass:

        sentinel = {
            "dataset_version": DATASET_VERSION,
            "status": "FROZEN",
            "protocol_version": PROTOCOL_VERSION,
            "frozen_at_utc": utc_now(),
            "raw_data_modified": False,
            "processed_data_modified": False,
            "global_qc": "PASS",
            "annotation_policy": "PASS",
            "next_stage": "SHA256_CHECKSUM_GENERATION",
        }

        with SENTINEL.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                sentinel,
                f,
                indent=2,
            )


# =============================================================================
# MAIN
# =============================================================================

def main():

    start = time.time()

    print()
    print("=" * 78)
    print("DATASET V1.0 FREEZE GATE")
    print("=" * 78)
    print(f"ROOT     : {ROOT}")
    print(f"VERSION  : {DATASET_VERSION}")
    print(f"PROTOCOL : {PROTOCOL_VERSION}")
    print(f"START    : {utc_now()}")
    print("=" * 78)

    if not ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {ROOT}"
        )

    records = []

    print("\n[1/9] Validating dataset structure...")
    structure_pass = validate_structure(records)
    print(f"       {'PASS' if structure_pass else 'FAIL'}")

    print("\n[2/9] Validating raw inventory...")
    raw_pass = validate_raw_inventory(records)
    print(f"       {'PASS' if raw_pass else 'FAIL'}")

    print("\n[3/9] Validating processed inventory...")
    processed_pass = validate_processed_inventory(records)
    print(f"       {'PASS' if processed_pass else 'FAIL'}")

    print("\n[4/9] Validating modality views...")
    modality_pass = validate_modality_views(records)
    print(f"       {'PASS' if modality_pass else 'FAIL'}")

    print("\n[5/9] Validating metadata/cohort...")
    metadata_pass = validate_metadata(records)
    cohort_pass = validate_cohort(records)
    print(f"       Metadata : {'PASS' if metadata_pass else 'FAIL'}")
    print(f"       Cohort   : {'PASS' if cohort_pass else 'FAIL'}")

    print("\n[6/9] Validating annotation manifest...")
    annotation_pass = validate_annotation_manifest(records)
    print(f"       Manifest : {'PASS' if annotation_pass else 'FAIL'}")

    print("\n[7/9] Validating annotation policy and audit...")
    policy_pass = validate_annotation_policy_json(records)
    audit_pass = validate_annotation_audit(records)
    print(f"       Policy   : {'PASS' if policy_pass else 'FAIL'}")
    print(f"       Audit    : {'PASS' if audit_pass else 'FAIL'}")

    print("\n[8/9] Validating Global QC...")
    global_qc_pass = validate_global_qc(records)
    print(f"       Global QC: {'PASS' if global_qc_pass else 'FAIL'}")

    print("\n[9/9] Validating file integrity...")
    temp_pass = validate_temporary_files(records)
    print(f"       Temp files: {'PASS' if temp_pass else 'FAIL'}")

    overall_pass = all(
        [
            structure_pass,
            raw_pass,
            processed_pass,
            modality_pass,
            metadata_pass,
            cohort_pass,
            annotation_pass,
            policy_pass,
            audit_pass,
            global_qc_pass,
            temp_pass,
        ]
    )

    runtime = time.time() - start

    write_outputs(
        records,
        overall_pass,
        runtime,
    )

    print()
    print("=" * 78)
    print("DATASET V1.0 FREEZE FINAL SUMMARY")
    print("=" * 78)

    print(
        f"Motor execution raw       : {EXPECTED_EXECUTION}"
    )
    print(
        f"Motor imagery raw         : {EXPECTED_MI}"
    )
    print(
        f"BioZ measurement units    : {EXPECTED_BIOZ_UNITS}"
    )
    print(
        f"BioZ native .spec files   : {EXPECTED_BIOZ_FILES}"
    )

    print(
        f"Execution EEG NPZ         : {EXPECTED_EXECUTION_EEG_NPZ}"
    )
    print(
        f"Motor imagery EEG NPZ     : {EXPECTED_MI_EEG_NPZ}"
    )
    print(
        f"EMG NPZ                   : {EXPECTED_EMG_NPZ}"
    )
    print(
        f"BioZ NPZ                  : {EXPECTED_BIOZ_NPZ}"
    )
    print(
        f"Total processed NPZ       : {EXPECTED_TOTAL_PROCESSED_NPZ}"
    )

    print(
        f"Annotation rows           : {EXPECTED_ANNOTATIONS}"
    )

    print("-" * 78)

    print(
        "Event annotations         : NOT AVAILABLE"
    )
    print(
        "Trigger annotations       : NOT AVAILABLE"
    )
    print(
        "Repetition annotations    : NOT AVAILABLE"
    )
    print(
        "Annotation inference      : NO"
    )
    print(
        "Event IDs fabricated      : NO"
    )
    print(
        "Repetition IDs fabricated : NO"
    )

    print("-" * 78)

    print(
        "RAW DATA MODIFICATION     : False"
    )
    print(
        "PREPROCESSING RERUN       : False"
    )
    print(
        "FEATURE EXTRACTION        : False"
    )
    print(
        "MACHINE LEARNING          : False"
    )
    print(
        "FUSION GENERATION         : False"
    )

    print("-" * 78)

    print(
        "DATASET V1.0 FREEZE STATUS: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    if overall_pass:
        print(
            "NEXT STAGE                : SHA256 CHECKSUM GENERATION"
        )
    else:
        print(
            "NEXT STAGE                : FIX FREEZE FAILURES"
        )

    print(
        f"Runtime                   : {runtime:.2f} s"
    )

    print(
        f"Report                    : {REPORT_JSON}"
    )

    print(
        f"Manifest                  : {MANIFEST_CSV}"
    )

    print(
        f"Summary                   : {SUMMARY_TXT}"
    )

    if overall_pass:
        print(
            f"Freeze sentinel           : {SENTINEL}"
        )

    print("=" * 78)

    if not overall_pass:
        print(
            "\nDATASET V1.0 FREEZE STATUS: FAIL"
        )
        print(
            "Do NOT proceed to checksum generation."
        )
        sys.exit(2)

    print(
        "\nDATASET V1.0 FREEZE STATUS: PASS"
    )
    print(
        "Dataset state is formally frozen."
    )


if __name__ == "__main__":
    main()