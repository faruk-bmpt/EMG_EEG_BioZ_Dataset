
###################################################################
"""
====================================================================
ANNOTATION AUDIT — DATASET V1.0
====================================================================

Purpose
-------
Audit the frozen dataset for EXISTING annotation/event information.

This block DOES NOT:
    - create annotations
    - modify raw data
    - modify processed data
    - create repetition IDs
    - infer repetition boundaries
    - fabricate event markers
    - modify signal files
    - perform preprocessing
    - perform feature extraction
    - perform machine learning

The audit searches for evidence of:
    1. Raw CSV annotation/event columns
    2. Trigger/event columns
    3. Repetition-related fields
    4. Separate annotation/event files
    5. Processed NPZ annotation/event arrays
    6. Existing annotation directories
    7. Existing annotation metadata

Output
------
05_QC/
└── ANNOTATION_AUDIT/
    ├── ANNOTATION_AUDIT_REPORT.csv
    ├── ANNOTATION_AUDIT_FILE_INVENTORY.csv
    ├── ANNOTATION_AUDIT_SUMMARY.txt
    └── ANNOTATION_AUDIT_FINAL_REPORT.json

No signal data are written or modified.
====================================================================
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


# ====================================================================
# 1. CONFIGURATION
# ====================================================================

ROOT = Path(
    "/mnt/f/Faruk/OFS_Paper_Work/Data_Set_Paper_Work/"
    "EEG_EMG_BIOZ_DATASET"
)

RAW = ROOT / "01_RAW_DATA"
PROCESSED = ROOT / "06_PROCESSED_DATA"

OUTPUT = ROOT / "05_QC" / "ANNOTATION_AUDIT"

RAW_EXEC = RAW / "EMG_EEG_SYNCHRONIZED"
RAW_MI = RAW / "EEG_MOTOR_IMAGERY"
RAW_BIOZ = RAW / "BIOIMPEDANCE"

PROC_EEG_EXEC = PROCESSED / "EEG_MOTOR_EXECUTION"
PROC_EEG_MI = PROCESSED / "EEG_MOTOR_IMAGERY"
PROC_EMG = PROCESSED / "EMG"
PROC_BIOZ = PROCESSED / "BIOIMPEDANCE"

DATASET_VERSION = "V1.0"
PROTOCOL_VERSION = "ANNOTATION-AUDIT-DATASET-V1.0"


# ====================================================================
# 2. SEARCH TERMS
# ====================================================================

ANNOTATION_TERMS = [
    "annotation",
    "annotations",
    "annot",
    "event",
    "events",
    "trigger",
    "triggers",
    "marker",
    "markers",
    "stimulus",
    "stim",
    "repetition",
    "repeat",
    "repetitions",
    "trial",
    "trials",
    "event_id",
    "eventid",
    "event_type",
    "eventtype",
    "trigger_id",
    "triggerid",
    "marker_id",
    "markerid",
    "onset",
    "offset",
    "duration",
    "epoch",
    "segment",
]

EVENT_TERMS = {
    "event",
    "events",
    "event_id",
    "eventid",
    "event_type",
    "eventtype",
}

TRIGGER_TERMS = {
    "trigger",
    "triggers",
    "trigger_id",
    "triggerid",
}

MARKER_TERMS = {
    "marker",
    "markers",
    "marker_id",
    "markerid",
}

REPETITION_TERMS = {
    "repetition",
    "repetitions",
    "repeat",
    "repeats",
    "trial",
    "trials",
    "rep",
}

ANNOTATION_FILE_TERMS = [
    "annotation",
    "annot",
    "event",
    "trigger",
    "marker",
    "repetition",
    "trial",
]


# ====================================================================
# 3. UTILITIES
# ====================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def normalize(text: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(text).lower()
    ).strip("_")


def contains_term(text: str, terms: set[str] | list[str]) -> bool:
    normalized = normalize(text)

    for term in terms:
        term_norm = normalize(term)

        if (
            normalized == term_norm
            or normalized.startswith(term_norm + "_")
            or normalized.endswith("_" + term_norm)
            or f"_{term_norm}_" in normalized
        ):
            return True

    return False


def annotation_category(text: str) -> str:
    normalized = normalize(text)

    if contains_term(normalized, EVENT_TERMS):
        return "EVENT"

    if contains_term(normalized, TRIGGER_TERMS):
        return "TRIGGER"

    if contains_term(normalized, MARKER_TERMS):
        return "MARKER"

    if contains_term(normalized, REPETITION_TERMS):
        return "REPETITION"

    return ""


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:

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


# ====================================================================
# 4. DIRECTORY AUDIT
# ====================================================================

def audit_directories() -> list[dict]:

    rows = []

    annotation_like_dirs = []

    for path in ROOT.rglob("*"):

        if not path.is_dir():
            continue

        name = path.name.lower()

        if any(
            term in name
            for term in ANNOTATION_FILE_TERMS
        ):

            annotation_like_dirs.append(path)

    for path in sorted(annotation_like_dirs):

        rows.append({
            "type": "DIRECTORY",
            "path": relative(path),
            "extension": "",
            "evidence": path.name,
            "annotation_category":
                annotation_category(path.name),
        })

    return rows


# ====================================================================
# 5. FILE-NAME AUDIT
# ====================================================================

def audit_annotation_like_files() -> list[dict]:

    rows = []

    for path in ROOT.rglob("*"):

        if not path.is_file():
            continue

        name = path.name.lower()

        if any(
            term in name
            for term in ANNOTATION_FILE_TERMS
        ):

            rows.append({
                "type": "FILE_NAME",
                "path": relative(path),
                "extension": path.suffix.lower(),
                "evidence": path.name,
                "annotation_category":
                    annotation_category(path.name),
            })

    return rows


# ====================================================================
# 6. RAW CSV HEADER AUDIT
# ====================================================================

def audit_csv_headers() -> list[dict]:

    rows = []

    csv_files = []

    csv_files.extend(
        RAW_EXEC.rglob("*.csv")
        if RAW_EXEC.exists()
        else []
    )

    csv_files.extend(
        RAW_MI.rglob("*.csv")
        if RAW_MI.exists()
        else []
    )

    for path in sorted(csv_files):

        try:

            with open(
                path,
                "r",
                encoding="utf-8-sig",
                errors="replace",
                newline="",
            ) as f:

                reader = csv.reader(f)
                header = next(reader, [])

        except Exception as exc:

            rows.append({
                "type": "CSV_HEADER_ERROR",
                "path": relative(path),
                "extension": ".csv",
                "evidence": str(exc),
                "annotation_category": "ERROR",
            })

            continue

        annotation_columns = []

        for column in header:

            category = annotation_category(column)

            if (
                category
                or any(
                    term in normalize(column)
                    for term in ANNOTATION_TERMS
                )
            ):

                annotation_columns.append(
                    f"{column} [{category or 'POTENTIAL'}]"
                )

        rows.append({
            "type": "RAW_CSV_HEADER",
            "path": relative(path),
            "extension": ".csv",
            "evidence":
                " | ".join(annotation_columns)
                if annotation_columns
                else "NO_ANNOTATION_LIKE_COLUMNS",
            "annotation_category":
                (
                    ",".join(
                        sorted(
                            set(
                                annotation_category(c)
                                for c in header
                                if annotation_category(c)
                            )
                        )
                    )
                    or "NONE"
                ),
        })

    return rows


# ====================================================================
# 7. PROCESSED NPZ KEY AUDIT
# ====================================================================

def audit_npz_keys() -> list[dict]:

    rows = []

    processed_roots = [
        PROC_EEG_EXEC,
        PROC_EEG_MI,
        PROC_EMG,
        PROC_BIOZ,
    ]

    for proc_root in processed_roots:

        if not proc_root.exists():
            continue

        for path in sorted(
            proc_root.rglob("*.npz")
        ):

            if (
                "PRE2_QC" in path.parts
                or "PRE3_QC" in path.parts
            ):
                continue

            try:

                with np.load(
                    path,
                    allow_pickle=True,
                ) as data:

                    keys = list(data.files)

            except Exception as exc:

                rows.append({
                    "type": "NPZ_KEY_ERROR",
                    "path": relative(path),
                    "extension": ".npz",
                    "evidence": str(exc),
                    "annotation_category": "ERROR",
                })

                continue

            annotation_keys = []

            for key in keys:

                category = annotation_category(key)

                if (
                    category
                    or any(
                        term in normalize(key)
                        for term in ANNOTATION_TERMS
                    )
                ):

                    annotation_keys.append(
                        f"{key} [{category or 'POTENTIAL'}]"
                    )

            rows.append({
                "type": "PROCESSED_NPZ_KEYS",
                "path": relative(path),
                "extension": ".npz",
                "evidence":
                    " | ".join(annotation_keys)
                    if annotation_keys
                    else "NO_ANNOTATION_LIKE_KEYS",
                "annotation_category":
                    (
                        ",".join(
                            sorted(
                                set(
                                    annotation_category(k)
                                    for k in keys
                                    if annotation_category(k)
                                )
                            )
                        )
                        or "NONE"
                    ),
            })

    return rows


# ====================================================================
# 8. METADATA / CSV FILE AUDIT
# ====================================================================

def audit_metadata_files() -> list[dict]:

    rows = []

    metadata_roots = [
        ROOT / "03_METADATA",
        ROOT / "04_ANNOTATIONS",
        ROOT / "05_QC",
        ROOT / "08_DOCUMENTATION",
        ROOT / "09_RELEASE",
    ]

    for root in metadata_roots:

        if not root.exists():
            continue

        for path in sorted(root.rglob("*")):

            if not path.is_file():
                continue

            name = path.name.lower()

            if any(
                term in name
                for term in ANNOTATION_FILE_TERMS
            ):

                rows.append({
                    "type": "METADATA_ANNOTATION_FILE",
                    "path": relative(path),
                    "extension": path.suffix.lower(),
                    "evidence": path.name,
                    "annotation_category":
                        annotation_category(path.name),
                })

    return rows


# ====================================================================
# 9. SIGNAL-LEVEL COLUMN VALUE AUDIT
# ====================================================================

def audit_special_raw_columns() -> list[dict]:

    rows = []

    csv_files = []

    if RAW_EXEC.exists():
        csv_files.extend(RAW_EXEC.rglob("*.csv"))

    if RAW_MI.exists():
        csv_files.extend(RAW_MI.rglob("*.csv"))

    for path in sorted(csv_files):

        try:

            with open(
                path,
                "r",
                encoding="utf-8-sig",
                errors="replace",
                newline="",
            ) as f:

                reader = csv.DictReader(f)

                if not reader.fieldnames:
                    continue

                annotation_columns = [
                    c
                    for c in reader.fieldnames
                    if (
                        annotation_category(c)
                        or any(
                            term in normalize(c)
                            for term in ANNOTATION_TERMS
                        )
                    )
                ]

                if not annotation_columns:
                    continue

                counters = {
                    column: Counter()
                    for column in annotation_columns
                }

                row_count = 0

                for row in reader:

                    row_count += 1

                    for column in annotation_columns:

                        value = row.get(column, "")

                        counters[column][
                            str(value)
                        ] += 1

                        # Prevent an enormous audit file.
                        if len(counters[column]) > 100:
                            break

                for column in annotation_columns:

                    values = counters[column]

                    rows.append({
                        "type": "RAW_ANNOTATION_COLUMN_VALUES",
                        "path": relative(path),
                        "extension": ".csv",
                        "evidence": (
                            f"column={column}; "
                            f"rows={row_count}; "
                            f"unique_values={len(values)}; "
                            f"top_values="
                            f"{values.most_common(10)}"
                        ),
                        "annotation_category":
                            annotation_category(column)
                            or "POTENTIAL",
                    })

        except Exception as exc:

            rows.append({
                "type": "RAW_COLUMN_AUDIT_ERROR",
                "path": relative(path),
                "extension": ".csv",
                "evidence": str(exc),
                "annotation_category": "ERROR",
            })

    return rows


# ====================================================================
# 10. REPORT
# ====================================================================

def create_reports(
    rows: list[dict],
) -> None:

    report_csv = OUTPUT / "ANNOTATION_AUDIT_REPORT.csv"

    fields = [
        "type",
        "path",
        "extension",
        "evidence",
        "annotation_category",
    ]

    write_csv(
        report_csv,
        rows,
        fields,
    )

    # ---------------------------------------------------------------
    # Summary counts
    # ---------------------------------------------------------------

    category_counts = Counter(
        row["annotation_category"]
        for row in rows
    )

    type_counts = Counter(
        row["type"]
        for row in rows
    )

    positive_rows = [
        row
        for row in rows
        if row["annotation_category"]
        in {
            "EVENT",
            "TRIGGER",
            "MARKER",
            "REPETITION",
            "POTENTIAL",
        }
    ]

    error_rows = [
        row
        for row in rows
        if row["annotation_category"] == "ERROR"
    ]

    # ---------------------------------------------------------------
    # Explicit evidence classification
    # ---------------------------------------------------------------

    raw_annotation_columns = [
        row
        for row in rows
        if row["type"] == "RAW_CSV_HEADER"
        and row["annotation_category"] != "NONE"
    ]

    npz_annotation_keys = [
        row
        for row in rows
        if row["type"] == "PROCESSED_NPZ_KEYS"
        and row["annotation_category"] != "NONE"
    ]

    annotation_named_files = [
        row
        for row in rows
        if row["type"] in {
            "FILE_NAME",
            "METADATA_ANNOTATION_FILE",
        }
    ]

    # This is an AUDIT classification, not an inference.
    if raw_annotation_columns:
        annotation_status = (
            "EXISTING_RAW_ANNOTATION_FIELDS_FOUND"
        )

    elif npz_annotation_keys:
        annotation_status = (
            "EXISTING_PROCESSED_ANNOTATION_FIELDS_FOUND"
        )

    elif annotation_named_files:
        annotation_status = (
            "ANNOTATION_NAMED_FILES_OR_DIRECTORIES_FOUND"
        )

    else:
        annotation_status = (
            "NO_EXISTING_ANNOTATION_EVIDENCE_FOUND"
        )

    summary = [
        f"ANNOTATION AUDIT — DATASET V1.0",
        f"Audit time: {now_iso()}",
        f"Protocol: {PROTOCOL_VERSION}",
        "",
        "AUDIT SCOPE",
        "------------",
        "Raw execution CSV files",
        "Raw motor-imagery CSV files",
        "Processed EEG NPZ files",
        "Processed EMG NPZ files",
        "Processed BioZ NPZ files",
        "Metadata/QC/documentation filenames",
        "Existing annotation-like directories/files",
        "",
        "RESULT",
        "------",
        f"Annotation status: {annotation_status}",
        "",
        "COUNTS",
        "------",
        f"Total audit records: {len(rows)}",
        f"Potential positive records: {len(positive_rows)}",
        f"Errors: {len(error_rows)}",
        "",
        "Raw CSV annotation-like headers:",
        f"  {len(raw_annotation_columns)}",
        "",
        "Processed NPZ annotation-like keys:",
        f"  {len(npz_annotation_keys)}",
        "",
        "Annotation/event/repetition named files/directories:",
        f"  {len(annotation_named_files)}",
        "",
        "CATEGORY COUNTS",
        "---------------",
    ]

    for category, count in sorted(
        category_counts.items()
    ):
        summary.append(
            f"{category}: {count}"
        )

    summary.extend([
        "",
        "RECORD TYPE COUNTS",
        "------------------",
    ])

    for record_type, count in sorted(
        type_counts.items()
    ):
        summary.append(
            f"{record_type}: {count}"
        )

    summary.extend([
        "",
        "IMPORTANT POLICY",
        "-----------------",
        "This audit does not create repetition/event annotations.",
        "No repetition boundaries are inferred from signal morphology.",
        "No Event_ID values are fabricated.",
        "No trigger timing is reconstructed unless explicitly present",
        "in the authoritative source data.",
        "Recording-level gesture labels are not treated as repetition-level",
        "event markers.",
        "",
        "RAW DATA MODIFICATION: False",
        "PROCESSED SIGNAL MODIFICATION: False",
        "ANNOTATIONS CREATED: False",
    ])

    summary_path = (
        OUTPUT / "ANNOTATION_AUDIT_SUMMARY.txt"
    )

    summary_path.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # JSON report
    # ---------------------------------------------------------------

    final_report = {
        "dataset_version": DATASET_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "audit_time": now_iso(),
        "annotation_status": annotation_status,
        "total_audit_records": len(rows),
        "positive_records": len(positive_rows),
        "error_records": len(error_rows),
        "raw_csv_annotation_fields": len(
            raw_annotation_columns
        ),
        "processed_npz_annotation_keys": len(
            npz_annotation_keys
        ),
        "annotation_named_files_or_directories": len(
            annotation_named_files
        ),
        "category_counts": dict(
            category_counts
        ),
        "record_type_counts": dict(
            type_counts
        ),
        "raw_data_modified": False,
        "processed_signal_data_modified": False,
        "annotations_created": False,
        "repetition_events_inferred": False,
        "event_ids_fabricated": False,
    }

    json_path = (
        OUTPUT / "ANNOTATION_AUDIT_FINAL_REPORT.json"
    )

    json_path.write_text(
        json.dumps(
            final_report,
            indent=2,
        ),
        encoding="utf-8",
    )


# ====================================================================
# 11. MAIN
# ====================================================================

def main():

    print()
    print("=" * 72)
    print("ANNOTATION AUDIT — DATASET V1.0")
    print("=" * 72)
    print(f"ROOT     : {ROOT}")
    print(f"VERSION  : {PROTOCOL_VERSION}")
    print(f"START    : {now_iso()}")
    print("=" * 72)

    if not ROOT.exists():
        print(
            f"ERROR: Dataset root not found:\n{ROOT}"
        )
        raise SystemExit(1)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[1/6] Auditing directories...")
    directory_rows = audit_directories()
    print(
        f"       Annotation-like directories: "
        f"{len(directory_rows)}"
    )

    print("\n[2/6] Auditing annotation-like filenames...")
    filename_rows = audit_annotation_like_files()
    print(
        f"       Annotation-like files: "
        f"{len(filename_rows)}"
    )

    print("\n[3/6] Auditing raw CSV headers...")
    csv_rows = audit_csv_headers()
    print(
        f"       CSV files audited: "
        f"{len(csv_rows)}"
    )

    print("\n[4/6] Auditing processed NPZ keys...")
    npz_rows = audit_npz_keys()
    print(
        f"       NPZ files audited: "
        f"{len(npz_rows)}"
    )

    print("\n[5/6] Auditing metadata/QC/documentation...")
    metadata_rows = audit_metadata_files()
    print(
        f"       Annotation-like metadata items: "
        f"{len(metadata_rows)}"
    )

    print("\n[6/6] Auditing annotation-like column values...")
    value_rows = audit_special_raw_columns()
    print(
        f"       Annotation-like value audits: "
        f"{len(value_rows)}"
    )

    all_rows = (
        directory_rows
        + filename_rows
        + csv_rows
        + npz_rows
        + metadata_rows
        + value_rows
    )

    create_reports(all_rows)

    print()
    print("=" * 72)
    print("ANNOTATION AUDIT COMPLETE")
    print("=" * 72)

    print(
        f"Audit records : {len(all_rows)}"
    )

    print(
        "RAW DATA MODIFICATION      : False"
    )

    print(
        "PROCESSED DATA MODIFICATION: False"
    )

    print(
        "ANNOTATIONS CREATED        : False"
    )

    print(
        f"OUTPUT                     : {OUTPUT}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
