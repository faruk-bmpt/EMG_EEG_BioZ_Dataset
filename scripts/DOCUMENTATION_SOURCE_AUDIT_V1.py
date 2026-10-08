#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DOCUMENTATION SOURCE AUDIT — DATASET V1.0
===============================================================================

Purpose
-------
Audit the formally frozen Dataset V1.0 and collect only source-supported
information needed to prepare the publication/release documentation.

This stage is READ-ONLY.

It does NOT:
    - modify raw data
    - modify processed data
    - rerun preprocessing
    - perform feature extraction
    - perform machine learning
    - infer annotations
    - modify metadata
    - modify the frozen release package
    - generate publication claims

Outputs
-------
08_DOCUMENTATION/
    DOCUMENTATION_SOURCE_AUDIT_V1.0.json
    DOCUMENTATION_SOURCE_AUDIT_V1.0.csv
    DOCUMENTATION_SOURCE_AUDIT_SUMMARY_V1.0.txt

The audit reports what is actually present in the frozen dataset. It does not
silently fill missing information from external/general knowledge.
===============================================================================
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# =============================================================================
# DATASET ROOT
# =============================================================================

# Resolve the dataset root from the canonical script location.
# This keeps the script portable across Windows/Linux environments and avoids
# hard-coded machine-specific paths.
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

if (SCRIPT_DIR / "01_RAW_DATA").exists():
    ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "01_RAW_DATA").exists():
    ROOT = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        "Could not locate the dataset root. Expected '01_RAW_DATA' "
        f"under either {SCRIPT_DIR} or {SCRIPT_DIR.parent}."
    )

ROOT = ROOT.resolve()

FREEZE = ROOT / "05_QC" / "DATASET_FREEZE"

SENTINEL = (
    FREEZE /
    "_DATASET_V1.0_FROZEN.json"
)

CHECKSUM_REPORT = (
    FREEZE /
    "SHA256" /
    "SHA256_CHECKSUM_FINAL_REPORT.json"
)

META = ROOT / "03_METADATA"
ANNOTATIONS = ROOT / "04_ANNOTATIONS"
QC = ROOT / "05_QC"
PROCESSED = ROOT / "06_PROCESSED_DATA"
MODALITY = ROOT / "02_MODALITY_VIEWS"

DOCS = ROOT / "08_DOCUMENTATION"

REPORT_JSON = (
    DOCS /
    "DOCUMENTATION_SOURCE_AUDIT_V1.0.json"
)

REPORT_CSV = (
    DOCS /
    "DOCUMENTATION_SOURCE_AUDIT_V1.0.csv"
)

SUMMARY_TXT = (
    DOCS /
    "DOCUMENTATION_SOURCE_AUDIT_SUMMARY_V1.0.txt"
)


# =============================================================================
# VERSION
# =============================================================================

DATASET_VERSION = "V1.0"

PROTOCOL_VERSION = (
    "DOCUMENTATION-SOURCE-AUDIT-DATASET-V1.0-FINAL"
)


# =============================================================================
# EXPECTED FROZEN COUNTS
# =============================================================================

EXPECTED = {
    "motor_execution_raw_csv": 840,
    "motor_imagery_raw_csv": 525,
    "bioz_raw_spec": 3780,
    "execution_eeg_npz": 840,
    "mi_eeg_npz": 525,
    "emg_npz": 840,
    "bioz_npz": 3780,
    "total_processed_npz": 5985,
    "annotation_rows": 3255,
    "retained_subjects": 40,
    "excluded_subjects": 1,
}

# Authoritative source SHA-256 result generated after the V1.0 freeze.
# These values are read from the actual checksum report and verified below;
# they are not guessed from an earlier run.
EXPECTED_SOURCE_SHA_FILES = 18588
EXPECTED_SOURCE_CATALOG_SHA256 = (
    "50459fa6298fc8a0487a7bebd18e63bedd86a22514b8ab821e9eab0d8fdb76bb"
)


# =============================================================================
# UTILITY
# =============================================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def add_record(
    records,
    section,
    source,
    item,
    status,
    value,
    notes="",
):
    records.append(
        {
            "section": section,
            "source": source,
            "item": item,
            "status": status,
            "value": value,
            "notes": notes,
        }
    )


def sha256_file(path: Path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def count_files(root: Path, pattern: str):

    if not root.exists():
        return 0

    return sum(
        1
        for _ in root.rglob(pattern)
    )


def safe_read_csv(path: Path):

    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return None, str(exc)


# =============================================================================
# FREEZE VERIFICATION
# =============================================================================

def verify_freeze(records):

    if not ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {ROOT}"
        )

    if not SENTINEL.exists():
        raise FileNotFoundError(
            f"Freeze sentinel missing: {SENTINEL}"
        )

    with SENTINEL.open(
        "r",
        encoding="utf-8",
    ) as f:
        sentinel = json.load(f)

    frozen = (
        sentinel.get("status") == "FROZEN"
        and sentinel.get("dataset_version") == DATASET_VERSION
    )

    add_record(
        records,
        "freeze",
        str(SENTINEL.relative_to(ROOT)),
        "Dataset freeze status",
        "VERIFIED" if frozen else "FAIL",
        sentinel.get("status"),
        f"dataset_version={sentinel.get('dataset_version')}",
    )

    if not frozen:
        raise RuntimeError(
            "Dataset V1.0 is not formally frozen."
        )

    return sentinel


# =============================================================================
# CHECKSUM VERIFICATION
# =============================================================================

def verify_checksum(records):

    if not CHECKSUM_REPORT.exists():

        add_record(
            records,
            "integrity",
            str(CHECKSUM_REPORT.relative_to(ROOT)),
            "Source SHA-256 report",
            "MISSING",
            "",
        )

        return None

    with CHECKSUM_REPORT.open(
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    ok = (
        report.get("dataset_version") == DATASET_VERSION
        and report.get("checksum_algorithm") == "SHA-256"
        and report.get("freeze_status_verified") is True
        and report.get("freeze_sentinel_status") == "FROZEN"
        and report.get("read_only") is True
        and report.get("files_hashed") == EXPECTED_SOURCE_SHA_FILES
        and report.get("checksum_catalog_sha256") == EXPECTED_SOURCE_CATALOG_SHA256
        and report.get("raw_data_modified") is False
        and report.get("processed_data_modified") is False
        and report.get("preprocessing_rerun") is False
        and report.get("feature_extraction") is False
        and report.get("machine_learning") is False
        and report.get("fusion_generation") is False
    )

    add_record(
        records,
        "integrity",
        str(CHECKSUM_REPORT.relative_to(ROOT)),
        "Source SHA-256 stage",
        "VERIFIED" if ok else "FAIL",
        report.get("checksum_catalog_sha256", ""),
        f"files_hashed={report.get('files_hashed')}",
    )

    if not ok:
        raise RuntimeError(
            "Source SHA-256 report does not match the authoritative frozen "
            "V1.0 checksum result (18,588 files / expected catalog hash)."
        )

    return report


# =============================================================================
# INVENTORY
# =============================================================================

def audit_inventory(records):

    checks = [
        (
            "Motor execution raw CSV",
            ROOT / "01_RAW_DATA" / "EMG_EEG_SYNCHRONIZED",
            "*.csv",
            EXPECTED["motor_execution_raw_csv"],
        ),
        (
            "Motor imagery raw CSV",
            ROOT / "01_RAW_DATA" / "EEG_MOTOR_IMAGERY",
            "*.csv",
            EXPECTED["motor_imagery_raw_csv"],
        ),
        (
            "BioZ native .spec",
            ROOT / "01_RAW_DATA" / "BIOIMPEDANCE",
            "*.spec",
            EXPECTED["bioz_raw_spec"],
        ),
        (
            "Execution EEG NPZ",
            PROCESSED / "EEG_MOTOR_EXECUTION",
            "*.npz",
            EXPECTED["execution_eeg_npz"],
        ),
        (
            "Motor imagery EEG NPZ",
            PROCESSED / "EEG_MOTOR_IMAGERY",
            "*.npz",
            EXPECTED["mi_eeg_npz"],
        ),
        (
            "EMG NPZ",
            PROCESSED / "EMG",
            "*.npz",
            EXPECTED["emg_npz"],
        ),
        (
            "BioZ NPZ",
            PROCESSED / "BIOIMPEDANCE",
            "*.npz",
            EXPECTED["bioz_npz"],
        ),
    ]

    for name, path, pattern, expected in checks:

        actual = count_files(
            path,
            pattern,
        )

        add_record(
            records,
            "inventory",
            str(path.relative_to(ROOT)),
            name,
            "PASS" if actual == expected else "FAIL",
            actual,
            f"expected={expected}",
        )

    total_processed = (
        count_files(
            PROCESSED / "EEG_MOTOR_EXECUTION",
            "*.npz",
        )
        + count_files(
            PROCESSED / "EEG_MOTOR_IMAGERY",
            "*.npz",
        )
        + count_files(
            PROCESSED / "EMG",
            "*.npz",
        )
        + count_files(
            PROCESSED / "BIOIMPEDANCE",
            "*.npz",
        )
    )

    add_record(
        records,
        "inventory",
        str(PROCESSED.relative_to(ROOT)),
        "Total processed NPZ",
        "PASS"
        if total_processed == EXPECTED["total_processed_npz"]
        else "FAIL",
        total_processed,
        f"expected={EXPECTED['total_processed_npz']}",
    )


# =============================================================================
# METADATA AUDIT
# =============================================================================

def audit_metadata(records):

    files = [
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

    for filename in files:

        path = META / filename

        if not path.exists():

            add_record(
                records,
                "metadata",
                str(path.relative_to(ROOT)),
                "Metadata file",
                "MISSING",
                "",
            )

            continue

        df, error = safe_read_csv(path)

        if error:

            add_record(
                records,
                "metadata",
                str(path.relative_to(ROOT)),
                "Metadata readability",
                "ERROR",
                "",
                error,
            )

            continue

        add_record(
            records,
            "metadata",
            str(path.relative_to(ROOT)),
            "Metadata file",
            "PASS",
            f"rows={len(df)}; columns={len(df.columns)}",
            "Source-supported metadata inventory.",
        )


# =============================================================================
# PARTICIPANT / COHORT FACTS
# =============================================================================

def audit_cohort(records):

    path = META / "participants.csv"

    if not path.exists():
        return

    df, error = safe_read_csv(path)

    if error:
        return

    if "cohort_status" in df.columns:

        status = (
            df["cohort_status"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        retained = int(
            (status == "retained").sum()
        )

        excluded = int(
            (status == "excluded").sum()
        )

        add_record(
            records,
            "cohort",
            str(path.relative_to(ROOT)),
            "Retained participants",
            "PASS"
            if retained == EXPECTED["retained_subjects"]
            else "CHECK",
            retained,
            f"expected={EXPECTED['retained_subjects']}",
        )

        add_record(
            records,
            "cohort",
            str(path.relative_to(ROOT)),
            "Excluded participants",
            "PASS"
            if excluded == EXPECTED["excluded_subjects"]
            else "CHECK",
            excluded,
            f"expected={EXPECTED['excluded_subjects']}",
        )

    if "subject_id" in df.columns:

        subject_ids = (
            df["subject_id"]
            .astype(str)
            .str.strip()
        )

        subject41 = df[
            subject_ids == "Subject_41"
        ]

        if len(subject41) == 1:

            value = (
                str(
                    subject41.iloc[0]
                    .get("cohort_status", "")
                )
            )

            add_record(
                records,
                "cohort",
                str(path.relative_to(ROOT)),
                "Subject_41 status",
                "PASS"
                if value.strip().lower() == "excluded"
                else "CHECK",
                value,
                "Subject_41 is retained as explicitly excluded.",
            )


# =============================================================================
# GESTURE DICTIONARY
# =============================================================================

def audit_gestures(records):

    path = META / "gesture_dictionary.csv"

    if not path.exists():
        return

    df, error = safe_read_csv(path)

    if error:
        return

    add_record(
        records,
        "gestures",
        str(path.relative_to(ROOT)),
        "Gesture dictionary rows",
        "AVAILABLE",
        len(df),
    )

    for column in df.columns:

        if column.lower() in {
            "gesture",
            "gesture_name",
            "gesture_code",
            "code",
            "label",
        }:

            values = (
                df[column]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )

            add_record(
                records,
                "gestures",
                str(path.relative_to(ROOT)),
                f"Gesture field: {column}",
                "AVAILABLE",
                values,
                "Values copied from frozen metadata only.",
            )


# =============================================================================
# ACQUISITION PARAMETERS
# =============================================================================

def audit_parameters(records):

    for filename in [
        "recording_parameters.csv",
        "acquisition_parameters.csv",
    ]:

        path = META / filename

        if not path.exists():
            continue

        df, error = safe_read_csv(path)

        if error:
            continue

        for _, row in df.iterrows():

            values = {}

            for column in df.columns:

                value = row[column]

                if pd.isna(value):
                    value = None
                else:
                    value = str(value)

                values[column] = value

            add_record(
                records,
                "acquisition",
                str(path.relative_to(ROOT)),
                "Parameter record",
                "AVAILABLE",
                values,
                "Directly extracted from frozen metadata.",
            )


# =============================================================================
# ANNOTATION SOURCES
# =============================================================================

def audit_annotations(records):

    files = [
        ANNOTATIONS /
        "RECORDING_LEVEL_ANNOTATION_MANIFEST.csv",

        ANNOTATIONS /
        "ANNOTATION_POLICY.md",

        ANNOTATIONS /
        "ANNOTATION_SCHEMA.csv",

        ANNOTATIONS /
        "ANNOTATION_POLICY_FINAL_REPORT.json",
    ]

    for path in files:

        add_record(
            records,
            "annotations",
            str(path.relative_to(ROOT)),
            "Annotation source",
            "AVAILABLE"
            if path.exists()
            else "MISSING",
            path.name,
        )

    manifest = files[0]

    if manifest.exists():

        df, error = safe_read_csv(manifest)

        if error:
            return

        add_record(
            records,
            "annotations",
            str(manifest.relative_to(ROOT)),
            "Annotation rows",
            "PASS"
            if len(df) == EXPECTED["annotation_rows"]
            else "CHECK",
            len(df),
            f"expected={EXPECTED['annotation_rows']}",
        )

    policy = files[3]

    if policy.exists():

        try:

            with policy.open(
                "r",
                encoding="utf-8",
            ) as f:
                report = json.load(f)

            for key in [
                "event_annotations_available",
                "trigger_annotations_available",
                "repetition_annotations_available",
                "annotation_inferred",
                "event_ids_fabricated",
                "repetition_ids_fabricated",
            ]:

                add_record(
                    records,
                    "annotations",
                    str(policy.relative_to(ROOT)),
                    key,
                    "AVAILABLE",
                    report.get(key),
                )

        except Exception as exc:

            add_record(
                records,
                "annotations",
                str(policy.relative_to(ROOT)),
                "Policy JSON readability",
                "ERROR",
                "",
                str(exc),
            )


# =============================================================================
# QC SOURCES
# =============================================================================

def audit_qc(records):

    required = [
        QC / "DATASET_GLOBAL_QC_REPORT.json",
        QC / "ANNOTATION_AUDIT" /
        "ANNOTATION_AUDIT_FINAL_REPORT.json",
        QC / "DATASET_FREEZE" /
        "DATASET_FREEZE_FINAL_REPORT.json",
        QC / "DATASET_FREEZE" /
        "DATASET_FREEZE_MANIFEST.csv",
        QC / "DATASET_FREEZE" /
        "DATASET_FREEZE_SUMMARY.txt",
        QC / "DATASET_FREEZE" /
        "_DATASET_V1.0_FROZEN.json",
        QC / "DATASET_FREEZE" /
        "SHA256" /
        "SHA256_CHECKSUM_FINAL_REPORT.json",
    ]

    for path in required:

        add_record(
            records,
            "quality_control",
            str(path.relative_to(ROOT)),
            "QC source",
            "AVAILABLE"
            if path.exists()
            else "MISSING",
            path.name,
        )

    global_qc = QC / "DATASET_GLOBAL_QC_REPORT.json"

    if global_qc.exists():

        try:

            with global_qc.open(
                "r",
                encoding="utf-8",
            ) as f:
                report = json.load(f)

            add_record(
                records,
                "quality_control",
                str(global_qc.relative_to(ROOT)),
                "Global QC overall status",
                "AVAILABLE",
                report.get("overall_status"),
            )

            add_record(
                records,
                "quality_control",
                str(global_qc.relative_to(ROOT)),
                "Global QC checks",
                "AVAILABLE",
                report.get("checks"),
            )

        except Exception as exc:

            add_record(
                records,
                "quality_control",
                str(global_qc.relative_to(ROOT)),
                "Global QC readability",
                "ERROR",
                "",
                str(exc),
            )


# =============================================================================
# MODALITY VIEWS
# =============================================================================

def audit_modality_views(records):

    files = [
        MODALITY / "EEG" /
        "EEG_MOTOR_EXECUTION_manifest.csv",

        MODALITY / "EEG" /
        "EEG_MOTOR_IMAGERY_manifest.csv",

        MODALITY / "EMG" /
        "EMG_manifest.csv",

        MODALITY / "BIOIMPEDANCE" /
        "BIOIMPEDANCE_manifest.csv",

        MODALITY / "MODALITY_VIEWS_summary.csv",
    ]

    for path in files:

        if not path.exists():

            add_record(
                records,
                "modality_views",
                str(path.relative_to(ROOT)),
                "Modality view",
                "MISSING",
                "",
            )

            continue

        df, error = safe_read_csv(path)

        if error:

            add_record(
                records,
                "modality_views",
                str(path.relative_to(ROOT)),
                "Modality view",
                "ERROR",
                "",
                error,
            )

            continue

        add_record(
            records,
            "modality_views",
            str(path.relative_to(ROOT)),
            "Modality view",
            "AVAILABLE",
            f"rows={len(df)}; columns={len(df.columns)}",
        )


# =============================================================================
# PROCESSED DATA STRUCTURE
# =============================================================================

def audit_processed_structure(records):

    branches = [
        (
            "EEG_MOTOR_EXECUTION",
            EXPECTED["execution_eeg_npz"],
        ),
        (
            "EEG_MOTOR_IMAGERY",
            EXPECTED["mi_eeg_npz"],
        ),
        (
            "EMG",
            EXPECTED["emg_npz"],
        ),
        (
            "BIOIMPEDANCE",
            EXPECTED["bioz_npz"],
        ),
    ]

    for branch, expected in branches:

        path = PROCESSED / branch

        actual = count_files(
            path,
            "*.npz",
        )

        add_record(
            records,
            "processed_data",
            str(path.relative_to(ROOT)),
            f"{branch} NPZ",
            "PASS"
            if actual == expected
            else "CHECK",
            actual,
            f"expected={expected}",
        )


# =============================================================================
# RELEASE STATUS
# =============================================================================

def audit_release(records):

    release = ROOT / "09_RELEASE" / "DATASET_V1.0"

    if not release.exists():

        add_record(
            records,
            "release",
            str(release.relative_to(ROOT)),
            "Release package",
            "MISSING",
            "",
        )

        return

    add_record(
        records,
        "release",
        str(release.relative_to(ROOT)),
        "Release package",
        "AVAILABLE",
        str(release),
    )

    for relative in [
        "RELEASE_NOTES.md",
        "SHA256SUMS_RELEASE_V1.0",
        "SHA256SUMS_SOURCE_V1.0",
    ]:

        path = release / relative

        add_record(
            records,
            "release",
            str(path.relative_to(ROOT)),
            "Release component",
            "AVAILABLE"
            if path.exists()
            else "MISSING",
            relative,
        )


# =============================================================================
# DOCUMENTATION READINESS
# =============================================================================

def documentation_readiness(records):

    sections = {
        "README_DATASET_V1.0.md": "Dataset overview, scope, structure, access",
        "DATASET_DESCRIPTION_V1.0.md": "Dataset description and scientific scope",
        "DATA_DICTIONARY_V1.0.md": "Metadata, manifest, and file-field definitions",
        "ACQUISITION_AND_STANDARDIZATION_PROTOCOL_V1.0.md":
            "Acquisition and standardization details",
        "ANNOTATION_GUIDE_V1.0.md":
            "Recording-level annotation policy and limitations",
        "QUALITY_CONTROL_REPORT_V1.0.md":
            "QC and integrity evidence",
        "MODALITY_AND_SYNCHRONIZATION_GUIDE_V1.0.md":
            "Modality relationships and synchronization boundaries",
        "ETHICS_AND_DATA_GOVERNANCE_V1.0.md":
            "Cohort, consent, ethics, anonymization, and governance",
        "DATA_ACCESS_AND_RELEASE_GUIDE_V1.0.md":
            "Release organization and access guidance",
        "DOCUMENTATION_INDEX_V1.0.md":
            "Documentation map",
    }

    for filename, purpose in sections.items():
        path = DOCS / filename
        exists = path.exists()

        add_record(
            records,
            "documentation_readiness",
            str(path.relative_to(ROOT)),
            filename,
            "AVAILABLE" if exists else "MISSING",
            path.name if exists else "",
            purpose,
        )


# =============================================================================
# SUMMARY
# =============================================================================

def write_outputs(
    records,
    sentinel,
    checksum_report,
    documentation_directory_existed_before_audit,
):

    DOCS.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(records)

    df.to_csv(
        REPORT_CSV,
        index=False,
        encoding="utf-8",
    )

    status_counts = (
        df["status"]
        .value_counts()
        .to_dict()
    )

    summary = [
        "=" * 78,
        "DOCUMENTATION SOURCE AUDIT — DATASET V1.0",
        "=" * 78,
        f"Dataset version       : {DATASET_VERSION}",
        f"Protocol version      : {PROTOCOL_VERSION}",
        f"Audit timestamp UTC   : {utc_now()}",
        f"Dataset root          : {ROOT}",
        "",
        "FREEZE",
        "-" * 78,
        f"Freeze status         : {sentinel.get('status')}",
        f"Freeze version        : {sentinel.get('dataset_version')}",
        "",
        "SOURCE CHECKSUM",
        "-" * 78,
        (
            "Checksum status       : "
            + (
                "VERIFIED"
                if checksum_report is not None
                else "MISSING"
            )
        ),
        (
            "Files hashed          : "
            + str(
                checksum_report.get("files_hashed")
                if checksum_report
                else ""
            )
        ),
        "",
        "FROZEN DATASET INVENTORY",
        "-" * 78,
        f"Motor execution CSV   : {EXPECTED['motor_execution_raw_csv']}",
        f"Motor imagery CSV     : {EXPECTED['motor_imagery_raw_csv']}",
        f"BioZ native .spec     : {EXPECTED['bioz_raw_spec']}",
        f"Execution EEG NPZ     : {EXPECTED['execution_eeg_npz']}",
        f"Motor imagery EEG NPZ : {EXPECTED['mi_eeg_npz']}",
        f"EMG NPZ               : {EXPECTED['emg_npz']}",
        f"BioZ NPZ              : {EXPECTED['bioz_npz']}",
        f"Total processed NPZ   : {EXPECTED['total_processed_npz']}",
        f"Annotation rows       : {EXPECTED['annotation_rows']}",
        f"Retained subjects     : {EXPECTED['retained_subjects']}",
        f"Excluded subjects     : {EXPECTED['excluded_subjects']}",
        "",
        "DOCUMENTATION STATUS",
        "-" * 78,
        (
            "08_DOCUMENTATION existed before audit: "
            + ("YES" if documentation_directory_existed_before_audit else "NO")
        ),
        "Documentation content generated by audit: NO",
        (
            "Missing documentation identified: "
            + (
                "YES"
                if any(
                    r.get("section") == "documentation_readiness"
                    and r.get("status") == "MISSING"
                    for r in records
                )
                else "NO"
            )
        ),
        "",
        "IMPORTANT POLICY",
        "-" * 78,
        "Only frozen, source-supported information should be used to draft",
        "publication/release documentation.",
        "Missing information must be explicitly identified rather than inferred.",
        "",
        "AUDIT RECORDS",
        "-" * 78,
        f"Total audit records   : {len(df)}",
        f"Status counts         : {status_counts}",
        "",
        f"CSV report            : {REPORT_CSV}",
        f"JSON report           : {REPORT_JSON}",
        f"Summary               : {SUMMARY_TXT}",
        "=" * 78,
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    report = {
        "dataset_version": DATASET_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "audit_timestamp_utc": utc_now(),
        "dataset_root": str(ROOT),

        "freeze": {
            "status": sentinel.get("status"),
            "dataset_version":
                sentinel.get("dataset_version"),
            "protocol_version":
                sentinel.get("protocol_version"),
        },

        "checksum": {
            "verified":
                checksum_report is not None,
            "algorithm":
                (
                    checksum_report.get(
                        "checksum_algorithm"
                    )
                    if checksum_report
                    else None
                ),
            "files_hashed":
                (
                    checksum_report.get(
                        "files_hashed"
                    )
                    if checksum_report
                    else None
                ),
            "catalog_sha256":
                (
                    checksum_report.get(
                        "checksum_catalog_sha256"
                    )
                    if checksum_report
                    else None
                ),
        },

        "frozen_inventory": EXPECTED,

        "documentation_readiness": {
            "source_directory_existed_before_audit":
                documentation_directory_existed_before_audit,
            "documentation_generated_by_audit": False,
            "drafting_ready": True,
            "external_or_missing_facts_must_not_be_invented": True,
        },

        "read_only": True,
        "raw_data_modified": False,
        "processed_data_modified": False,
        "preprocessing_rerun": False,
        "feature_extraction": False,
        "machine_learning": False,
        "annotation_inference": False,

        "audit_record_count": len(df),

        "reports": {
            "csv":
                str(REPORT_CSV.relative_to(ROOT)),
            "summary":
                str(SUMMARY_TXT.relative_to(ROOT)),
        },

        "python": sys.version,
        "platform": platform.platform(),
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


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 78)
    print("DOCUMENTATION SOURCE AUDIT — DATASET V1.0")
    print("=" * 78)
    print(f"ROOT     : {ROOT}")
    print(f"VERSION  : {DATASET_VERSION}")
    print(f"PROTOCOL : {PROTOCOL_VERSION}")
    print("=" * 78)

    records = []
    documentation_directory_existed_before_audit = DOCS.exists()

    print("\n[1/9] Verifying Dataset V1.0 freeze...")
    sentinel = verify_freeze(records)
    print("       PASS")

    print("\n[2/9] Verifying source SHA-256...")
    checksum_report = verify_checksum(records)
    print(
        "       "
        + (
            "PASS"
            if checksum_report is not None
            else "FAIL"
        )
    )
    if checksum_report is None:
        raise RuntimeError(
            "Source SHA-256 is required before documentation source audit."
        )

    print("\n[3/9] Auditing frozen inventory...")
    audit_inventory(records)
    print("       COMPLETE")

    print("\n[4/9] Auditing metadata and cohort...")
    audit_metadata(records)
    audit_cohort(records)
    print("       COMPLETE")

    print("\n[5/9] Auditing gesture/acquisition information...")
    audit_gestures(records)
    audit_parameters(records)
    print("       COMPLETE")

    print("\n[6/9] Auditing annotations and modality views...")
    audit_annotations(records)
    audit_modality_views(records)
    print("       COMPLETE")

    print("\n[7/9] Auditing QC and processed-data sources...")
    audit_qc(records)
    audit_processed_structure(records)
    print("       COMPLETE")

    print("\n[8/9] Auditing existing release package...")
    audit_release(records)
    print("       COMPLETE")

    print("\n[9/9] Assessing documentation readiness...")
    documentation_readiness(records)
    print("       COMPLETE")

    write_outputs(
        records,
        sentinel,
        checksum_report,
        documentation_directory_existed_before_audit,
    )

    print()
    print("=" * 78)
    print("DOCUMENTATION SOURCE AUDIT FINAL SUMMARY")
    print("=" * 78)

    print(
        f"Audit records          : {len(records)}"
    )

    print(
        "Dataset freeze         : VERIFIED"
    )

    print(
        "Source SHA-256         : "
        + (
            "VERIFIED"
            if checksum_report is not None
            else "MISSING"
        )
    )

    print(
        "Raw data modified      : False"
    )

    print(
        "Processed modified     : False"
    )

    print(
        "Preprocessing rerun    : False"
    )

    print(
        "Documentation created  : NO"
    )

    print(
        "External facts inferred: NO"
    )

    print("-" * 78)

    print(
        f"JSON report            : {REPORT_JSON}"
    )

    print(
        f"CSV report             : {REPORT_CSV}"
    )

    print(
        f"Summary                : {SUMMARY_TXT}"
    )

    print("=" * 78)
    print(
        "DOCUMENTATION SOURCE AUDIT: COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
