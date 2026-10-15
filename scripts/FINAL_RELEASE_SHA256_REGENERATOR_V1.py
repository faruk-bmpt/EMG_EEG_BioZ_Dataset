#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final release SHA-256 generation and verification for Dataset V1.0."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

if (SCRIPT_DIR / "01_RAW_DATA").is_dir():
    ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "01_RAW_DATA").is_dir():
    ROOT = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        "Cannot locate dataset root. Expected 01_RAW_DATA "
        "in the script directory or its parent directory."
    )

ROOT = ROOT.resolve()

RELEASE = ROOT / "09_RELEASE" / "DATASET_V1.0"
FREEZE = ROOT / "05_QC" / "DATASET_FREEZE"
SOURCE_SHA = FREEZE / "SHA256"
DOC = RELEASE / "DOCUMENTATION"
R_SHA = RELEASE / "SHA256SUMS_RELEASE_V1.0"

PROTOCOL = "FINAL-RELEASE-SHA256-DATASET-V1.0-FINAL"
DATASET_VERSION = "V1.0"

# Preserve the established frozen-source checksum contract.
EXPECTED_SOURCE_FILES = 18588
EXPECTED_SOURCE_CATALOG_SHA256 = (
    "50459fa6298fc8a0487a7bebd18e63bedd86a22514b8ab821e9eab0d8fdb76bb"
)

EXPECTED_DOC_FILES = [
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


# ============================================================
# 2. UTILITIES
# ============================================================

def die(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def require(path: Path, label: str) -> None:
    if not path.exists():
        die(f"Missing {label}: {path}")


def read_json(path: Path) -> dict:
    require(path, "JSON file")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        die(f"Cannot read valid JSON from {path}: {exc}")

    if not isinstance(data, dict):
        die(f"Expected a JSON object in {path}")

    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# 3. VERIFY DATASET FREEZE
# ============================================================

def verify_freeze() -> None:
    sentinel_path = FREEZE / "_DATASET_V1.0_FROZEN.json"
    report_path = FREEZE / "DATASET_FREEZE_FINAL_REPORT.json"

    sentinel = read_json(sentinel_path)
    report = read_json(report_path)

    combined = (
        json.dumps(sentinel) + " " + json.dumps(report)
    ).upper()

    if sentinel.get("frozen") is not True and not any(
        marker in combined
        for marker in ('"PASS"', '"FROZEN"', '"VERIFIED"')
    ):
        die(
            "Dataset freeze could not be verified from "
            "the freeze report and sentinel."
        )


# ============================================================
# 4. VERIFY FROZEN SOURCE CHECKSUMS
# ============================================================

def verify_source_checksums() -> dict:
    report_path = SOURCE_SHA / "SHA256_CHECKSUM_FINAL_REPORT.json"
    report = read_json(report_path)

    expected = {
        "freeze_status_verified": True,
        "freeze_sentinel_status": "FROZEN",
        "checksum_algorithm": "SHA-256",
        "files_hashed": EXPECTED_SOURCE_FILES,
        "read_only": True,
        "raw_data_modified": False,
        "processed_data_modified": False,
        "preprocessing_rerun": False,
        "feature_extraction": False,
        "machine_learning": False,
        "fusion_generation": False,
    }

    for key, value in expected.items():
        if report.get(key) != value:
            die(
                f"Source SHA report {key}={report.get(key)!r}; "
                f"expected {value!r}"
            )

    if (
        report.get("checksum_catalog_sha256")
        != EXPECTED_SOURCE_CATALOG_SHA256
    ):
        die(
            "Source checksum catalog hash differs from "
            "the frozen Dataset V1.0 contract."
        )

    return report


# ============================================================
# 5. VERIFY RELEASE STRUCTURE
# ============================================================

def verify_release_structure() -> None:
    required_components = [
        "DATA",
        "ANNOTATIONS",
        "METADATA",
        "MODALITY_VIEWS",
        "QC",
        "DOCUMENTATION",
        "RELEASE_NOTES.md",
        "SHA256SUMS_SOURCE_V1.0",
    ]

    for relative in required_components:
        require(RELEASE / relative, f"release component {relative}")


# ============================================================
# 6. VERIFY DOCUMENTATION
# ============================================================

def verify_documentation(source_report: dict) -> dict:
    for filename in EXPECTED_DOC_FILES:
        require(DOC / filename, f"release documentation file {filename}")

    report = read_json(
        DOC / "DOCUMENTATION_GENERATION_FINAL_REPORT.json"
    )

    expected_flags = {
        "external_facts_inferred": False,
        "raw_data_modified": False,
        "processed_data_modified": False,
        "preprocessing_rerun": False,
        "feature_extraction_performed": False,
        "machine_learning_performed": False,
        "fusion_generation_performed": False,
    }

    for key, value in expected_flags.items():
        if report.get(key) is not value:
            die(
                f"Documentation report {key}={report.get(key)!r}; "
                f"expected {value!r}"
            )

    if report.get("dataset_version") != DATASET_VERSION:
        die("Documentation report dataset_version must be V1.0.")

    if (
        report.get("source_sha256_files_hashed")
        != source_report.get("files_hashed")
    ):
        die(
            "Documentation source checksum file count does not "
            "match the frozen source report."
        )

    if (
        report.get("source_checksum_catalog_sha256")
        != source_report.get("checksum_catalog_sha256")
    ):
        die(
            "Documentation source checksum catalog does not "
            "match the frozen source report."
        )

    # Confirm that the ethics document has the expected approval details.
    ethics_path = DOC / "ETHICS_AND_DATA_GOVERNANCE_V1.0.md"
    ethics = ethics_path.read_text(encoding="utf-8").casefold()

    required_ethics = [
        "323/biol. scs.",
        "12 october 2025",
        "approved",
    ]

    missing_ethics = [
        term for term in required_ethics if term not in ethics
    ]

    if missing_ethics:
        die(
            "Ethics documentation requires correction. Missing: "
            + ", ".join(missing_ethics)
        )

    # Verify DOI and license statements in the access guide.
    access_path = DOC / "DATA_ACCESS_AND_RELEASE_GUIDE_V1.0.md"
    access = access_path.read_text(encoding="utf-8").casefold()

    if "10.6084/m9.figshare.34421721" not in access:
        die("Data access guide must include the reserved Figshare DOI.")

    if "cc by 4.0" not in access and "cc-by-4.0" not in access:
        die("Data access guide must state the CC BY 4.0 license.")

    if "reserved" not in access:
        die(
            "Data access guide must identify the DOI as reserved "
            "until publication."
        )

    return report


# ============================================================
# 7. CLEAR ONLY THIS STAGE'S OLD OUTPUTS
# ============================================================

def clear_previous_generated_catalog(directory: Path) -> None:
    """Remove only known outputs from this checksum stage."""

    directory.mkdir(parents=True, exist_ok=True)

    known_outputs = [
        "SHA256SUMS_RELEASE_V1.0.csv",
        "SHA256SUMS_RELEASE_V1.0.txt",
        "RELEASE_SHA256_FINAL_REPORT.json",
    ]

    for name in known_outputs:
        path = directory / name

        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


# ============================================================
# 8. GENERATE AND VERIFY RELEASE CHECKSUMS
# ============================================================

def main() -> None:
    print("=" * 72)
    print("FINAL RELEASE SHA-256 REGENERATION — DATASET V1.0")
    print("=" * 72)
    print("Dataset root :", ROOT)
    print("Release root :", RELEASE)
    print("Protocol     :", PROTOCOL)

    verify_freeze()
    print("[1/6] Verifying Dataset V1.0 freeze... PASS")

    source_report = verify_source_checksums()
    print(
        f"[2/6] Verifying frozen source SHA-256... "
        f"PASS | files={EXPECTED_SOURCE_FILES}"
    )

    verify_release_structure()
    print("[3/6] Verifying final release structure... PASS")

    verify_documentation(source_report)
    print(
        f"[4/6] Verifying documentation and publication metadata... "
        f"PASS | {len(EXPECTED_DOC_FILES)} required files"
    )

    # Remove previous outputs from this stage only.
    clear_previous_generated_catalog(R_SHA)

    entries = []

    for path in sorted(RELEASE.rglob("*")):
        if not path.is_file():
            continue

        try:
            path.relative_to(R_SHA)
            continue
        except ValueError:
            pass

        entries.append(
            (
                sha256_file(path),
                path.relative_to(RELEASE).as_posix(),
            )
        )

    entries.sort(key=lambda item: item[1])

    csv_path = R_SHA / "SHA256SUMS_RELEASE_V1.0.csv"
    txt_path = R_SHA / "SHA256SUMS_RELEASE_V1.0.txt"
    report_path = R_SHA / "RELEASE_SHA256_FINAL_REPORT.json"

    with csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sha256", "relative_path"])
        writer.writerows(entries)

    with txt_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        for digest, relative_path in entries:
            handle.write(f"{digest}  {relative_path}\n")

    catalog_hash = sha256_file(csv_path)

    report = {
        "dataset_version": DATASET_VERSION,
        "protocol_version": PROTOCOL,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "release_root": str(RELEASE),
        "source_sha256_report": str(
            (
                SOURCE_SHA / "SHA256_CHECKSUM_FINAL_REPORT.json"
            ).relative_to(ROOT)
        ),
        "source_freeze_verified": True,
        "source_sha256_verified": True,
        "source_sha256_files_hashed": source_report["files_hashed"],
        "source_checksum_catalog_sha256": (
            source_report["checksum_catalog_sha256"]
        ),
        "checksum_algorithm": "SHA-256",
        "files_hashed": len(entries),
        "release_checksum_csv": (
            "SHA256SUMS_RELEASE_V1.0/"
            "SHA256SUMS_RELEASE_V1.0.csv"
        ),
        "release_checksum_text": (
            "SHA256SUMS_RELEASE_V1.0/"
            "SHA256SUMS_RELEASE_V1.0.txt"
        ),
        "release_checksum_catalog_sha256": catalog_hash,
        "excluded_paths": ["SHA256SUMS_RELEASE_V1.0/"],
        "raw_data_modified": False,
        "processed_data_modified": False,
        "preprocessing_rerun": False,
        "feature_extraction": False,
        "machine_learning": False,
        "fusion_generation": False,
        "documentation_included": True,
        "physiological_signal_values_modified": False,
        "sample_index_metadata_corrected_in_prior_stage": True,
    }

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    # Verify all catalog entries against the files on disk.
    with csv_path.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != len(entries):
        die("Release checksum catalog row count mismatch.")

    for row in rows:
        file_path = RELEASE / row["relative_path"]

        if not file_path.is_file():
            die(
                "Catalog references a missing file: "
                + row["relative_path"]
            )

        if sha256_file(file_path) != row["sha256"]:
            die(
                "Release checksum mismatch: "
                + row["relative_path"]
            )

    if sha256_file(csv_path) != catalog_hash:
        die("Release checksum catalog self-hash mismatch.")

    print(
        f"[5/6] Generating release SHA-256 catalog... "
        f"PASS | files={len(entries)}"
    )
    print("[6/6] Verifying release SHA-256 catalog... PASS")

    print("\n" + "=" * 72)
    print("FINAL RELEASE SHA-256: COMPLETE")
    print("=" * 72)
    print("Release files hashed       :", len(entries))
    print("Checksum algorithm         : SHA-256")
    print("Catalog SHA-256            :", catalog_hash)
    print("Documentation included     : True")
    print("Raw data modified          : False")
    print("Processed data modified    : False")
    print("Preprocessing rerun        : False")
    print("Feature extraction         : False")
    print("Machine learning           : False")
    print("Fusion generation          : False")
    print("=" * 72)
    print("DATASET V1.0 RELEASE INTEGRITY STAGE: COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()