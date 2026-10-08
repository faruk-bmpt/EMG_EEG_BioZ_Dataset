#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
DATASET V1.0 FINAL RELEASE PACKAGER
===============================================================================

Purpose
-------
Build the final publication/release directory for the formally frozen
EEG + EMG + Bioimpedance Dataset V1.0.

This stage:
    - verifies the V1.0 freeze sentinel
    - verifies the SHA-256 checksum stage
    - copies frozen dataset content into 09_RELEASE/DATASET_V1.0
    - preserves the raw and processed hierarchy without renaming raw folders
    - copies modality views, metadata, annotations, selected QC evidence,
      documentation, and checksum records
    - creates release notes
    - generates release-specific SHA-256 checksums

This stage does NOT:
    - modify 01_RAW_DATA
    - modify 06_PROCESSED_DATA
    - rerun preprocessing
    - perform feature extraction
    - perform machine learning
    - infer annotations
    - create event/repetition annotations
    - generate fusion data
    - rename raw recordings
    - rename raw folders

The source dataset remains untouched. The release directory is a copy.

===============================================================================
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# =============================================================================
# DATASET ROOT
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if (SCRIPT_DIR / "01_RAW_DATA").is_dir():
    ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "01_RAW_DATA").is_dir():
    ROOT = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        "Cannot locate dataset root. Expected 01_RAW_DATA beside "
        "the script or in its parent directory."
    )
ROOT = ROOT.resolve()

RELEASE_ROOT = (
    ROOT /
    "09_RELEASE" /
    "DATASET_V1.0"
)

FREEZE_DIR = (
    ROOT /
    "05_QC" /
    "DATASET_FREEZE"
)

FREEZE_SENTINEL = (
    FREEZE_DIR /
    "_DATASET_V1.0_FROZEN.json"
)

CHECKSUM_DIR = (
    FREEZE_DIR /
    "SHA256"
)

CHECKSUM_REPORT = (
    CHECKSUM_DIR /
    "SHA256_CHECKSUM_FINAL_REPORT.json"
)

SOURCE_CHECKSUM_TXT = (
    CHECKSUM_DIR /
    "SHA256SUMS_V1.0.txt"
)


# =============================================================================
# VERSION
# =============================================================================

DATASET_VERSION = "V1.0"

PROTOCOL_VERSION = (
    "DATASET-RELEASE-PACKAGING-V1.0-FINAL"
)


# =============================================================================
# FROZEN COUNTS
# =============================================================================

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


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def copy_tree(source: Path, destination: Path):

    if not source.exists():
        raise FileNotFoundError(
            f"Required source directory missing: {source}"
        )

    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
    )


def copy_file(source: Path, destination: Path):

    if not source.exists():
        raise FileNotFoundError(
            f"Required source file missing: {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )


# =============================================================================
# VERIFY FREEZE
# =============================================================================

def verify_freeze():

    if not ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {ROOT}"
        )

    if not FREEZE_SENTINEL.exists():
        raise FileNotFoundError(
            f"Freeze sentinel not found: {FREEZE_SENTINEL}"
        )

    with FREEZE_SENTINEL.open(
        "r",
        encoding="utf-8",
    ) as f:
        sentinel = json.load(f)

    if sentinel.get("status") != "FROZEN":
        raise RuntimeError(
            "Dataset is not formally frozen. "
            "Release packaging aborted."
        )

    if sentinel.get("dataset_version") != DATASET_VERSION:
        raise RuntimeError(
            "Freeze sentinel dataset version does not match V1.0."
        )

    return sentinel


# =============================================================================
# VERIFY CHECKSUM STAGE
# =============================================================================

def verify_checksum_stage():

    if not CHECKSUM_REPORT.exists():
        raise FileNotFoundError(
            f"Checksum report missing: {CHECKSUM_REPORT}"
        )

    if not SOURCE_CHECKSUM_TXT.exists():
        raise FileNotFoundError(
            f"Checksum manifest missing: {SOURCE_CHECKSUM_TXT}"
        )

    with CHECKSUM_REPORT.open(
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    if report.get("dataset_version") != DATASET_VERSION:
        raise RuntimeError(
            "Checksum report dataset version mismatch."
        )

    if report.get("checksum_algorithm") != "SHA-256":
        raise RuntimeError(
            "Checksum algorithm is not SHA-256."
        )

    if not report.get("freeze_status_verified", False):
        raise RuntimeError(
            "Checksum stage does not verify frozen state."
        )

    if not report.get("read_only", False):
        raise RuntimeError(
            "Checksum stage was not marked read-only."
        )

    if report.get("files_hashed") != 18588:
        raise RuntimeError(
            "Source SHA-256 file count mismatch: "
            f"{report.get('files_hashed')!r}; expected 18588."
        )

    expected_catalog = (
        "50459fa6298fc8a0487a7bebd18e63bedd86a22514b8ab821e9eab0d8fdb76bb"
    )
    if report.get("checksum_catalog_sha256") != expected_catalog:
        raise RuntimeError(
            "Source SHA-256 catalog hash mismatch: "
            f"{report.get('checksum_catalog_sha256')!r}; "
            f"expected {expected_catalog}."
        )

    for key in (
        "raw_data_modified",
        "processed_data_modified",
        "preprocessing_rerun",
        "feature_extraction",
        "machine_learning",
        "fusion_generation",
    ):
        if report.get(key) is not False:
            raise RuntimeError(
                f"Source checksum report indicates unexpected {key}="
                f"{report.get(key)!r}."
            )

    return report


# =============================================================================
# REQUIRED SOURCE STRUCTURE
# =============================================================================

def verify_sources():

    required_dirs = [
        ROOT / "01_RAW_DATA",
        ROOT / "02_MODALITY_VIEWS",
        ROOT / "03_METADATA",
        ROOT / "04_ANNOTATIONS",
        ROOT / "05_QC",
        ROOT / "06_PROCESSED_DATA",
    ]

    required_files = [
        ROOT / "02_MODALITY_VIEWS" /
        "EEG" /
        "EEG_MOTOR_EXECUTION_manifest.csv",

        ROOT / "02_MODALITY_VIEWS" /
        "EEG" /
        "EEG_MOTOR_IMAGERY_manifest.csv",

        ROOT / "02_MODALITY_VIEWS" /
        "EMG" /
        "EMG_manifest.csv",

        ROOT / "02_MODALITY_VIEWS" /
        "BIOIMPEDANCE" /
        "BIOIMPEDANCE_manifest.csv",

        ROOT / "03_METADATA" / "participants.csv",
        ROOT / "03_METADATA" / "subject_crosswalk.csv",
        ROOT / "03_METADATA" / "subject_modality_availability.csv",
        ROOT / "03_METADATA" / "recording_inventory.csv",
        ROOT / "03_METADATA" / "recording_parameters.csv",
        ROOT / "03_METADATA" / "acquisition_parameters.csv",
        ROOT / "03_METADATA" / "gesture_dictionary.csv",
        ROOT / "03_METADATA" / "data_provenance.csv",
        ROOT / "03_METADATA" / "cohort_definition.csv",
        ROOT / "03_METADATA" / "exclusion_log.csv",

        ROOT / "04_ANNOTATIONS" /
        "RECORDING_LEVEL_ANNOTATION_MANIFEST.csv",

        ROOT / "04_ANNOTATIONS" /
        "ANNOTATION_POLICY.md",

        ROOT / "04_ANNOTATIONS" /
        "ANNOTATION_SCHEMA.csv",

        ROOT / "04_ANNOTATIONS" /
        "ANNOTATION_POLICY_FINAL_REPORT.json",

        ROOT / "05_QC" /
        "DATASET_GLOBAL_QC_REPORT.json",

        ROOT / "05_QC" /
        "ANNOTATION_AUDIT" /
        "ANNOTATION_AUDIT_FINAL_REPORT.json",

        ROOT / "05_QC" /
        "DATASET_FREEZE" /
        "DATASET_FREEZE_FINAL_REPORT.json",

        ROOT / "05_QC" /
        "DATASET_FREEZE" /
        "DATASET_FREEZE_MANIFEST.csv",

        ROOT / "05_QC" /
        "DATASET_FREEZE" /
        "DATASET_FREEZE_SUMMARY.txt",

        ROOT / "05_QC" /
        "DATASET_FREEZE" /
        "_DATASET_V1.0_FROZEN.json",

        ROOT / "05_QC" /
        "DATASET_FREEZE" /
        "SHA256" /
        "SHA256SUMS_V1.0.csv",

        ROOT / "05_QC" /
        "DATASET_FREEZE" /
        "SHA256" /
        "SHA256SUMS_V1.0.txt",

        ROOT / "05_QC" /
        "DATASET_FREEZE" /
        "SHA256" /
        "SHA256_CHECKSUM_FINAL_REPORT.json",
    ]

    missing = []

    for path in required_dirs:
        if not path.exists():
            missing.append(
                str(path.relative_to(ROOT))
            )

    for path in required_files:
        if not path.exists():
            missing.append(
                str(path.relative_to(ROOT))
            )

    if missing:

        message = "\n".join(
            f"  - {item}"
            for item in missing
        )

        raise RuntimeError(
            "Required release source(s) missing:\n"
            + message
        )


# =============================================================================
# COPY RELEASE CONTENT
# =============================================================================

def build_release():

    # ------------------------------------------------------------
    # Safety: release directory may be recreated, but source data
    # are never deleted or modified.
    # ------------------------------------------------------------

    RELEASE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # DATA
    #
    # Preserve the frozen raw folder names exactly.
    # ------------------------------------------------------------

    data_root = RELEASE_ROOT / "DATA"

    copy_tree(
        ROOT / "01_RAW_DATA",
        data_root / "01_RAW_DATA",
    )

    copy_tree(
        ROOT / "06_PROCESSED_DATA",
        data_root / "06_PROCESSED_DATA",
    )

    # ------------------------------------------------------------
    # MODALITY VIEWS
    # ------------------------------------------------------------

    copy_tree(
        ROOT / "02_MODALITY_VIEWS",
        RELEASE_ROOT / "MODALITY_VIEWS",
    )

    # ------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------

    copy_tree(
        ROOT / "03_METADATA",
        RELEASE_ROOT / "METADATA",
    )

    # ------------------------------------------------------------
    # ANNOTATIONS
    # ------------------------------------------------------------

    copy_tree(
        ROOT / "04_ANNOTATIONS",
        RELEASE_ROOT / "ANNOTATIONS",
    )

    # ------------------------------------------------------------
    # QC
    #
    # Copy the existing QC evidence. The generated checksum files
    # inside DATASET_FREEZE are retained as provenance evidence.
    # ------------------------------------------------------------

    qc_out = RELEASE_ROOT / "QC"

    copy_file(
        ROOT / "05_QC" /
        "DATASET_GLOBAL_QC_REPORT.json",
        qc_out / "DATASET_GLOBAL_QC_REPORT.json",
    )

    copy_tree(
        ROOT / "05_QC" /
        "ANNOTATION_AUDIT",
        qc_out / "ANNOTATION_AUDIT",
    )

    copy_tree(
        ROOT / "05_QC" /
        "DATASET_FREEZE",
        qc_out / "DATASET_FREEZE",
    )

    # ------------------------------------------------------------
    # DOCUMENTATION
    #
    # Documentation was generated in the preceding stage and is
    # required for the final V1.0 release package.
    # ------------------------------------------------------------

    documentation_source = ROOT / "08_DOCUMENTATION"

    if not documentation_source.is_dir():
        raise FileNotFoundError(
            f"Required documentation source directory missing: "
            f"{documentation_source}"
        )

    required_documentation = [
        "README_DATASET_V1.0.md",
        "DATASET_DESCRIPTION_V1.0.md",
        "DATA_DICTIONARY_V1.0.md",
        "ACQUISITION_AND_STANDARDIZATION_PROTOCOL_V1.0.md",
        "ANNOTATION_GUIDE_V1.0.md",
        "QUALITY_CONTROL_REPORT_V1.0.md",
        "MODALITY_AND_SYNCHRONIZATION_GUIDE_V1.0.md",
        "ETHICS_AND_DATA_GOVERNANCE_V1.0.md",
        "DATA_ACCESS_AND_RELEASE_GUIDE_V1.0.md",
        "DOCUMENTATION_INDEX_V1.0.md",
        "DOCUMENTATION_MANIFEST_V1.0.csv",
        "DOCUMENTATION_GENERATION_SUMMARY_V1.0.txt",
        "DOCUMENTATION_GENERATION_FINAL_REPORT.json",
    ]

    for filename in required_documentation:
        if not (documentation_source / filename).is_file():
            raise FileNotFoundError(
                f"Required documentation file missing: "
                f"{documentation_source / filename}"
            )

    copy_tree(
        documentation_source,
        RELEASE_ROOT / "DOCUMENTATION",
    )

    documentation_status = "COPIED"

    # ------------------------------------------------------------
    # SOURCE CHECKSUMS
    # ------------------------------------------------------------

    checksum_out = (
        RELEASE_ROOT /
        "SHA256SUMS_SOURCE_V1.0"
    )

    checksum_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    copy_file(
        SOURCE_CHECKSUM_TXT,
        checksum_out / "SHA256SUMS_V1.0.txt",
    )

    copy_file(
        CHECKSUM_REPORT,
        checksum_out /
        "SHA256_CHECKSUM_FINAL_REPORT.json",
    )


# =============================================================================
# RELEASE NOTES
# =============================================================================

def write_release_notes(
    sentinel,
    checksum_report,
    documentation_status,
):

    notes = f"""# EEG–EMG–Bioimpedance Dataset V1.0
## Release Notes

**Dataset version:** {DATASET_VERSION}

**Release packaging protocol:** {PROTOCOL_VERSION}

**Release generated (UTC):** {utc_now()}

### Frozen inventory

| Component | Count |
|---|---:|
| Motor-execution raw CSV | {EXPECTED_EXECUTION} |
| Motor-imagery raw CSV | {EXPECTED_MI} |
| Bioimpedance measurement units | {EXPECTED_BIOZ_UNITS} |
| Bioimpedance native `.spec` files | {EXPECTED_BIOZ_FILES} |
| Execution EEG processed NPZ | {EXPECTED_EXECUTION_EEG_NPZ} |
| Motor-imagery EEG processed NPZ | {EXPECTED_MI_EEG_NPZ} |
| EMG processed NPZ | {EXPECTED_EMG_NPZ} |
| Bioimpedance processed NPZ | {EXPECTED_BIOZ_NPZ} |
| Total processed NPZ | {EXPECTED_TOTAL_PROCESSED_NPZ} |
| Recording-level annotation rows | {EXPECTED_ANNOTATIONS} |
| Retained participants | {EXPECTED_RETAINED_SUBJECTS} |
| Excluded participants | {EXPECTED_EXCLUDED_SUBJECTS} |

### Annotation policy

- Event annotations: NOT AVAILABLE
- Trigger annotations: NOT AVAILABLE
- Repetition annotations: NOT AVAILABLE
- Annotation inference: NO
- Event IDs fabricated: NO
- Repetition IDs fabricated: NO

The dataset provides recording-level gesture labels. Event-level and
repetition-level boundaries were not inferred or fabricated.

### Data integrity

The dataset passed the formal Dataset V1.0 Freeze Gate before release
packaging.

Freeze sentinel status: **{sentinel.get("status")}**

SHA-256 checksum stage: **VERIFIED**

Files included in the source checksum catalog:
**{checksum_report.get("files_hashed")}**

Source checksum catalog SHA-256:

`{checksum_report.get("checksum_catalog_sha256")}`

### Source-data preservation

Raw data are preserved under their frozen V1.0 hierarchy.

No raw recordings or raw folder names were renamed during release packaging.

The release package is a copy of the frozen dataset; the source dataset was
not modified by this packaging stage.

### Documentation

Documentation source status:
**{documentation_status}**

No documentation content is generated or invented by this packaging script.

### Known dataset characteristics

- Motor execution uses synchronized EEG and EMG acquired from the same
  OpenBCI stream.
- Motor imagery contains EEG-only recordings.
- Bioimpedance is an independent frequency-domain measurement modality and
  is not sample-synchronized with EEG/EMG.
- Repetition/event annotations are unavailable.
- Subject_41 remains explicitly excluded from the retained cohort.

### Release contents

- `DATA/`
- `MODALITY_VIEWS/`
- `METADATA/`
- `ANNOTATIONS/`
- `QC/`
- `DOCUMENTATION/` when present in the source dataset
- `SHA256SUMS_SOURCE_V1.0/`
- `RELEASE_NOTES.md`

### Release status

**DATASET V1.0 RELEASE PACKAGE GENERATED**

"""

    (RELEASE_ROOT / "RELEASE_NOTES.md").write_text(
        notes,
        encoding="utf-8",
    )


# =============================================================================
# RELEASE CHECKSUMS
# =============================================================================

def generate_release_checksums():

    output_dir = RELEASE_ROOT / "SHA256SUMS_RELEASE_V1.0"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_output = (
        output_dir /
        "SHA256SUMS_RELEASE_V1.0.csv"
    )

    txt_output = (
        output_dir /
        "SHA256SUMS_RELEASE_V1.0.txt"
    )

    files = []

    for path in RELEASE_ROOT.rglob("*"):

        if not path.is_file():
            continue

        relative = path.relative_to(
            RELEASE_ROOT
        )

        # Exclude release checksum outputs from the catalog to avoid
        # self-reference.
        if (
            relative == output_dir.relative_to(RELEASE_ROOT)
            or output_dir.relative_to(RELEASE_ROOT)
            in relative.parents
        ):
            continue

        files.append(
            (
                relative.as_posix(),
                path,
            )
        )

    files.sort(
        key=lambda x: x[0]
    )

    rows = []

    for relative, path in files:

        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    with csv_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_path",
                "size_bytes",
                "sha256",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    with txt_output.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                f"{row['sha256']}  "
                f"{row['relative_path']}\n"
            )

    catalog_hash = hashlib.sha256()

    for row in rows:

        catalog_hash.update(
            (
                f"{row['sha256']}  "
                f"{row['relative_path']}\n"
            ).encode("utf-8")
        )

    report = {
        "dataset_version": DATASET_VERSION,
        "protocol_version":
            "SHA256-RELEASE-CHECKSUM-DATASET-V1.0-FINAL",
        "generated_at_utc": utc_now(),
        "release_root": str(RELEASE_ROOT),
        "checksum_algorithm": "SHA-256",
        "files_hashed": len(rows),
        "catalog_sha256": catalog_hash.hexdigest(),
        "read_only_source": True,
        "source_dataset_modified": False,
        "preprocessing_rerun": False,
        "feature_extraction": False,
        "machine_learning": False,
        "fusion_generation": False,
        "release_checksum_csv":
            str(csv_output.relative_to(RELEASE_ROOT)),
        "release_checksum_txt":
            str(txt_output.relative_to(RELEASE_ROOT)),
    }

    report_output = (
        output_dir /
        "SHA256_RELEASE_CHECKSUM_FINAL_REPORT.json"
    )

    with report_output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return rows, catalog_hash.hexdigest(), report_output


# =============================================================================
# FINAL INVENTORY
# =============================================================================

def release_inventory():

    counts = {
        "release_files": 0,
        "raw_csv": 0,
        "mi_csv": 0,
        "bioz_spec": 0,
        "processed_npz": 0,
    }

    for path in RELEASE_ROOT.rglob("*"):

        if not path.is_file():
            continue

        counts["release_files"] += 1

        rel = path.relative_to(
            RELEASE_ROOT
        ).as_posix()

        if "DATA/01_RAW_DATA/EMG_EEG_SYNCHRONIZED/" in rel:
            if path.suffix.lower() == ".csv":
                counts["raw_csv"] += 1

        if "DATA/01_RAW_DATA/EEG_MOTOR_IMAGERY/" in rel:
            if path.suffix.lower() == ".csv":
                counts["mi_csv"] += 1

        if "DATA/01_RAW_DATA/BIOIMPEDANCE/" in rel:
            if path.suffix.lower() == ".spec":
                counts["bioz_spec"] += 1

        if "DATA/06_PROCESSED_DATA/" in rel:
            if path.suffix.lower() == ".npz":
                counts["processed_npz"] += 1

    return counts


# =============================================================================
# MAIN
# =============================================================================

def main():

    start = time.time()

    print()
    print("=" * 78)
    print("DATASET V1.0 FINAL RELEASE PACKAGER")
    print("=" * 78)
    print(f"Dataset root   : {ROOT}")
    print(f"Release root   : {RELEASE_ROOT}")
    print(f"Protocol       : {PROTOCOL_VERSION}")
    print("=" * 78)

    # ------------------------------------------------------------
    # 1. Verify frozen state
    # ------------------------------------------------------------

    print("\n[1/6] Verifying Dataset V1.0 freeze...")
    sentinel = verify_freeze()
    print("       PASS")

    # ------------------------------------------------------------
    # 2. Verify checksum stage
    # ------------------------------------------------------------

    print("\n[2/6] Verifying SHA-256 checksum stage...")
    checksum_report = verify_checksum_stage()
    print(
        f"       PASS | files={checksum_report.get('files_hashed')}"
    )

    # ------------------------------------------------------------
    # 3. Verify source structure
    # ------------------------------------------------------------

    print("\n[3/6] Verifying release source structure...")
    verify_sources()
    print("       PASS")

    # ------------------------------------------------------------
    # 4. Build release
    # ------------------------------------------------------------

    print("\n[4/6] Building release package...")
    build_release()

    documentation_source = ROOT / "08_DOCUMENTATION"

    documentation_status = (
        "COPIED"
        if documentation_source.exists()
        else "NOT_PRESENT_IN_SOURCE_DATASET"
    )

    print("       PASS")

    # ------------------------------------------------------------
    # 5. Release notes
    # ------------------------------------------------------------

    print("\n[5/6] Writing release notes...")
    write_release_notes(
        sentinel,
        checksum_report,
        documentation_status,
    )
    print("       PASS")

    # ------------------------------------------------------------
    # 6. Release-specific checksums
    # ------------------------------------------------------------

    print("\n[6/6] Generating release-specific SHA-256 checksums...")

    rows, catalog_hash, release_report = (
        generate_release_checksums()
    )

    print(
        f"       PASS | files={len(rows)}"
    )

    # ------------------------------------------------------------
    # Final inventory
    # ------------------------------------------------------------

    counts = release_inventory()

    expected_counts = {
        "raw_csv": EXPECTED_EXECUTION,
        "mi_csv": EXPECTED_MI,
        "bioz_spec": EXPECTED_BIOZ_FILES,
        "processed_npz": EXPECTED_TOTAL_PROCESSED_NPZ,
    }
    for key, expected in expected_counts.items():
        if counts[key] != expected:
            raise RuntimeError(
                f"Final release inventory mismatch for {key}: "
                f"{counts[key]} != expected {expected}"
            )

    runtime = time.time() - start

    print()
    print("=" * 78)
    print("DATASET V1.0 FINAL RELEASE SUMMARY")
    print("=" * 78)

    print(
        f"Release files             : "
        f"{counts['release_files']}"
    )

    print(
        f"Motor execution raw CSV   : "
        f"{counts['raw_csv']}"
    )

    print(
        f"Motor imagery raw CSV     : "
        f"{counts['mi_csv']}"
    )

    print(
        f"BioZ native .spec files   : "
        f"{counts['bioz_spec']}"
    )

    print(
        f"Processed NPZ             : "
        f"{counts['processed_npz']}"
    )

    print("-" * 78)

    print(
        "Source dataset modified   : False"
    )

    print(
        "Preprocessing rerun       : False"
    )

    print(
        "Feature extraction        : False"
    )

    print(
        "Machine learning          : False"
    )

    print(
        "Fusion generation         : False"
    )

    print("-" * 78)

    print(
        f"Documentation status      : "
        f"{documentation_status}"
    )

    print(
        f"Release checksum files    : "
        f"{len(rows)}"
    )

    print(
        f"Release catalog SHA-256   : "
        f"{catalog_hash}"
    )

    print(
        f"Release root              : "
        f"{RELEASE_ROOT}"
    )

    print(
        f"Release checksum report   : "
        f"{release_report}"
    )

    print(
        f"Runtime                   : "
        f"{runtime:.2f} s"
    )

    print("=" * 78)
    print(
        "DATASET V1.0 RELEASE PACKAGE: COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
