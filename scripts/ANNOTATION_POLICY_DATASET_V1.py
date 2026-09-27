#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================================
ANNOTATION POLICY — DATASET V1.0 FINAL
====================================================================

Purpose
-------
Formalize the annotation structure actually supported by the
frozen dataset.

This stage:
    - creates recording-level annotation metadata
    - preserves existing gesture labels
    - documents the absence of repetition/event markers
    - does NOT infer repetition boundaries
    - does NOT fabricate Event_ID values
    - does NOT create trigger timestamps
    - does NOT modify raw signal files
    - does NOT modify processed signal files
    - does NOT rerun preprocessing
    - does NOT create fusion data
    - does NOT perform feature extraction or ML

Annotation principle
--------------------
The authoritative class label is assigned at the recording level.

For motor execution:
    Subject × Gesture × Set

For motor imagery:
    MI_ID × Gesture × Set

For BioZ:
    BZ_ID × Gesture × Measurement Set

No repetition-level annotations are currently available.

Output
------
04_ANNOTATIONS/
├── RECORDING_LEVEL_ANNOTATION_MANIFEST.csv
├── ANNOTATION_POLICY.md
├── ANNOTATION_SCHEMA.csv
└── ANNOTATION_POLICY_FINAL_REPORT.json
====================================================================
"""

from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


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

OUTPUT = ROOT / "04_ANNOTATIONS"

DATASET_VERSION = "V1.0"
PROTOCOL_VERSION = "ANNOTATION-POLICY-DATASET-V1.0-FINAL"

EXPECTED_EXECUTION = 840
EXPECTED_MI = 525
EXPECTED_BIOZ = 1890


# ====================================================================
# 2. GESTURE DEFINITIONS
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

GESTURE_ALIASES = {
    "hand_open_close": "HOC",
    "thumb_finger_movement": "TFM",
    "index_finger_movement": "IFM",
    "middle_finger_movement": "MFM",
    "ring_finger_movement": "RFM",
    "little_finger_movement": "LFM",
    "pen_holding": "PH",
}


# ====================================================================
# 3. BIOZ CROSSWALK
# ====================================================================

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
# 4. UTILITIES
# ====================================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def normalize(text: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(text).lower(),
    ).strip("_")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(text)

        # open(fd, ...) is not valid on every Python build, so this
        # branch is retained only as a safety fallback.
    except TypeError:
        import os

        os.close(fd)

        with open(
            tmp_name,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(text)

    import os

    if os.path.exists(tmp_name):
        os.replace(tmp_name, path)


def atomic_write_csv(
    path: Path,
    rows: list[dict],
    fields: list[str],
) -> None:

    import os

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
                fieldnames=fields,
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


def list_csv(root: Path) -> list[Path]:
    if not root.exists():
        return []

    return sorted(
        p
        for p in root.rglob("*.csv")
        if p.is_file()
    )


def list_spec(root: Path) -> list[Path]:
    if not root.exists():
        return []

    return sorted(
        p
        for p in root.rglob("*.spec")
        if p.is_file()
    )


# ====================================================================
# 5. PARSERS
# ====================================================================

def gesture_from_path(
    path: Path,
) -> tuple[str | None, str | None]:
    """
    Resolve gesture code/name robustly.

    Priority:
      1. Exact gesture directory name.
      2. Gesture code embedded in a directory name, e.g. BZ_01_HOC.
      3. Full gesture name embedded in a directory name.
      4. Filename fallback.

    This is especially important for BioZ paths such as:

        BZ_01/BZ_01_HOC/Channel_1/set_01.spec
    """

    # ---------------------------------------------------------------
    # 1. Inspect parent directories from nearest to farthest.
    # ---------------------------------------------------------------

    for parent in path.parents:

        label = normalize(parent.name)

        # Exact full gesture name.
        for code, full_name in GESTURE_MAP.items():

            if label == normalize(full_name):
                return code, full_name

        # Exact gesture code.
        if label.upper() in GESTURE_MAP:
            code = label.upper()
            return code, GESTURE_MAP[code]

        # Gesture code embedded in directory name.
        # Example: BZ_01_HOC -> HOC
        for code, full_name in GESTURE_MAP.items():

            if re.search(
                rf"(?:^|_){re.escape(code.lower())}(?:_|$)",
                label,
            ):
                return code, full_name

        # Full gesture name embedded in directory name.
        for code, full_name in GESTURE_MAP.items():

            full_norm = normalize(full_name)

            if (
                full_norm
                and full_norm in label
            ):
                return code, full_name

    # ---------------------------------------------------------------
    # 2. Filename fallback.
    # ---------------------------------------------------------------

    normalized = normalize(path.stem)

    for code, full_name in GESTURE_MAP.items():

        if re.search(
            rf"(?:^|_){re.escape(code.lower())}(?:_|$)",
            normalized,
        ):
            return code, full_name

        if (
            normalize(full_name)
            and normalize(full_name) in normalized
        ):
            return code, full_name

    return None, None


def execution_subject(path: Path) -> str | None:

    candidates = [
        path.stem,
        *[p.name for p in path.parents],
    ]

    for text in candidates:

        match = re.search(
            r"(?:^|_)(?:S|Subject_)(\d{1,2})(?:_|$)",
            text,
            re.IGNORECASE,
        )

        if match:
            return f"Subject_{int(match.group(1)):02d}"

    return None


def mi_id(path: Path) -> str | None:

    candidates = [
        path.stem,
        *[p.name for p in path.parents],
    ]

    for text in candidates:

        match = re.search(
            r"(?:^|_)MI_(\d{1,2})(?:_|$)",
            text,
            re.IGNORECASE,
        )

        if match:
            return f"MI_{int(match.group(1)):02d}"

    return None


def execution_set(path: Path) -> str | None:

    candidates = [
        path.stem,
        *[p.name for p in path.parents],
    ]

    for text in candidates:

        match = re.search(
            r"(?:^|[_-])Set[_-]?([ABC])(?:[_-]|$)",
            text,
            re.IGNORECASE,
        )

        if match:
            return match.group(1).upper()

    return None


def bioz_id(path: Path) -> str | None:

    for part in path.parts:

        match = re.fullmatch(
            r"BZ_(\d{1,2})",
            part,
            re.IGNORECASE,
        )

        if match:
            return f"BZ_{int(match.group(1)):02d}"

    return None


def bioz_set(path: Path) -> int | None:

    candidates = [
        path.stem,
        *[p.name for p in path.parents],
    ]

    for text in candidates:

        match = re.search(
            r"(?:^|[_-])set[_-]?(\d+)(?:[_-]|$)",
            text,
            re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

    return None


# ====================================================================
# 6. EXECUTION RECORDING INVENTORY
# ====================================================================

def build_execution_inventory() -> list[dict]:

    rows = []
    seen = set()

    for path in list_csv(RAW_EXEC):

        subject = execution_subject(path)
        gesture_code, gesture_name = gesture_from_path(path)
        recording_set = execution_set(path)

        if (
            subject is None
            or gesture_code is None
            or recording_set is None
        ):
            continue

        key = (
            subject,
            gesture_code,
            recording_set,
        )

        if key in seen:
            continue

        seen.add(key)

        rows.append({
            "annotation_id":
                f"EXEC_{subject}_{gesture_code}_Set{recording_set}",

            "annotation_level":
                "RECORDING",

            "protocol":
                "MOTOR_EXECUTION",

            "modality":
                "EEG_EMG_SYNCHRONIZED",

            "subject_id":
                subject,

            "mi_id":
                "",

            "bioz_id":
                "",

            "gesture_code":
                gesture_code,

            "gesture_name":
                gesture_name,

            "set_id":
                recording_set,

            "measurement_set":
                "",

            "class_label":
                gesture_code,

            "class_label_source":
                "RECORDING_LEVEL_GESTURE",

            "event_annotation_available":
                "NO",

            "trigger_annotation_available":
                "NO",

            "repetition_annotation_available":
                "NO",

            "event_id":
                "",

            "repetition_id":
                "",

            "onset_sample":
                "",

            "offset_sample":
                "",

            "onset_time_seconds":
                "",

            "offset_time_seconds":
                "",

            "annotation_inferred":
                "NO",

            "raw_file":
                relative(path),

            "notes":
                (
                    "Entire continuous recording is assigned one "
                    "recording-level gesture class. No repetition "
                    "or event markers are available."
                ),
        })

    return sorted(
        rows,
        key=lambda r: (
            r["subject_id"],
            r["gesture_code"],
            r["set_id"],
        ),
    )


# ====================================================================
# 7. MOTOR IMAGERY RECORDING INVENTORY
# ====================================================================

def build_mi_inventory() -> list[dict]:

    rows = []
    seen = set()

    for path in list_csv(RAW_MI):

        identifier = mi_id(path)
        gesture_code, gesture_name = gesture_from_path(path)
        recording_set = execution_set(path)

        if (
            identifier is None
            or gesture_code is None
            or recording_set is None
        ):
            continue

        key = (
            identifier,
            gesture_code,
            recording_set,
        )

        if key in seen:
            continue

        seen.add(key)

        rows.append({
            "annotation_id":
                f"MI_{identifier}_{gesture_code}_Set{recording_set}",

            "annotation_level":
                "RECORDING",

            "protocol":
                "MOTOR_IMAGERY",

            "modality":
                "EEG",

            "subject_id":
                "",

            "mi_id":
                identifier,

            "bioz_id":
                "",

            "gesture_code":
                gesture_code,

            "gesture_name":
                gesture_name,

            "set_id":
                recording_set,

            "measurement_set":
                "",

            "class_label":
                gesture_code,

            "class_label_source":
                "RECORDING_LEVEL_GESTURE",

            "event_annotation_available":
                "NO",

            "trigger_annotation_available":
                "NO",

            "repetition_annotation_available":
                "NO",

            "event_id":
                "",

            "repetition_id":
                "",

            "onset_sample":
                "",

            "offset_sample":
                "",

            "onset_time_seconds":
                "",

            "offset_time_seconds":
                "",

            "annotation_inferred":
                "NO",

            "raw_file":
                relative(path),

            "notes":
                (
                    "MI recording identifier is independent of the "
                    "canonical execution Subject ID. Entire recording "
                    "is assigned one recording-level gesture class."
                ),
        })

    return sorted(
        rows,
        key=lambda r: (
            r["mi_id"],
            r["gesture_code"],
            r["set_id"],
        ),
    )


# ====================================================================
# 8. BIOZ MEASUREMENT INVENTORY
# ====================================================================

def build_bioz_inventory() -> list[dict]:

    rows = []
    seen = set()

    for path in list_spec(RAW_BIOZ):

        identifier = bioz_id(path)
        gesture_code, gesture_name = gesture_from_path(path)
        measurement_set = bioz_set(path)

        if (
            identifier is None
            or gesture_code is None
            or measurement_set is None
        ):
            continue

        key = (
            identifier,
            gesture_code,
            measurement_set,
        )

        if key in seen:
            continue

        seen.add(key)

        rows.append({
            "annotation_id":
                (
                    f"BIOZ_{identifier}_"
                    f"{gesture_code}_Set{measurement_set:02d}"
                ),

            "annotation_level":
                "MEASUREMENT_UNIT",

            "protocol":
                "MOTOR_EXECUTION",

            "modality":
                "BIOIMPEDANCE",

            "subject_id":
                BIOZ_CROSSWALK.get(identifier, ""),

            "mi_id":
                "",

            "bioz_id":
                identifier,

            "gesture_code":
                gesture_code,

            "gesture_name":
                gesture_name,

            "set_id":
                "",

            "measurement_set":
                measurement_set,

            "class_label":
                gesture_code,

            "class_label_source":
                "MEASUREMENT_LEVEL_GESTURE",

            "event_annotation_available":
                "NO",

            "trigger_annotation_available":
                "NO",

            "repetition_annotation_available":
                "NO",

            "event_id":
                "",

            "repetition_id":
                "",

            "onset_sample":
                "",

            "offset_sample":
                "",

            "onset_time_seconds":
                "",

            "offset_time_seconds":
                "",

            "annotation_inferred":
                "NO",

            "raw_file":
                relative(path),

            "notes":
                (
                    "Bioimpedance is an independent frequency-domain "
                    "measurement. No sample-level temporal event or "
                    "repetition annotation is available."
                ),
        })

    return sorted(
        rows,
        key=lambda r: (
            r["bioz_id"],
            r["gesture_code"],
            r["measurement_set"],
        ),
    )


# ====================================================================
# 9. POLICY DOCUMENT
# ====================================================================

def create_policy_document(
    exec_count: int,
    mi_count: int,
    bioz_count: int,
) -> None:

    text = f"""# Annotation Policy — Dataset V1.0

## 1. Purpose

This document defines the annotation structure supported by the
frozen Dataset V1.0.

The policy is based on the completed annotation audit and does not
introduce inferred or fabricated event information.

## 2. Authoritative annotation level

The dataset uses **recording-level gesture labels**.

For motor execution:

`Subject × Gesture × Set`

For motor imagery:

`MI_ID × Gesture × Set`

For Bioimpedance:

`BZ_ID × Gesture × Measurement Set`

The recording-level gesture label identifies the instructed gesture
represented by the recording or measurement unit.

## 3. Motor execution

Motor-execution recordings contain synchronized EEG and EMG acquired
from the same OpenBCI recording stream.

Each recording corresponds to one:

- participant
- gesture
- set

The complete retained continuous recording is assigned one gesture
class label.

EEG and EMG therefore share the same recording-level class label.

Current inventory:

- 40 retained participants
- 7 gesture classes
- 3 sets per gesture
- 840 recording-level annotation rows

## 4. Motor imagery

Motor-imagery recordings contain EEG only.

MI identifiers are maintained independently from the canonical
motor-execution Subject IDs.

No unverified MI_ID-to-Subject_ID mapping is introduced.

Current inventory:

- 25 MI identifiers
- 7 gesture classes
- 3 sets per gesture
- 525 recording-level annotation rows

## 5. Bioimpedance

Bioimpedance measurements are independent frequency-domain
measurements.

The annotation unit is the measurement unit:

`BZ_ID × Gesture × Measurement Set`

Bioimpedance is not treated as sample-level synchronized with
EEG/EMG.

Current inventory:

- 18 BioZ identifiers
- 7 gesture classes
- 15 measurement units per gesture
- 1,890 measurement-level annotation rows

## 6. Repetition annotations

No repetition-level annotation source was identified during the
annotation audit.

Therefore:

- repetition boundaries are not provided;
- repetition IDs are not assigned;
- repetition onset/offset samples are not assigned;
- repetition timestamps are not reconstructed;
- signal morphology is not used to infer repetitions.

## 7. Event and trigger annotations

No event, trigger, or marker fields were identified in the audited
raw CSV headers or processed NPZ keys.

Therefore:

- no Event_ID values are fabricated;
- no trigger timestamps are reconstructed;
- no event boundaries are inferred.

## 8. Window-level labels

If a future analysis windows a continuous recording, each derived
window may inherit the recording-level gesture label from its source
recording.

This inherited label must be documented as:

`RECORDING_LEVEL_GESTURE`

It must not be represented as an independently observed
repetition-level event.

## 9. Missing annotation fields

The following fields are intentionally empty in the manifest because
no authoritative source exists:

- event_id
- repetition_id
- onset_sample
- offset_sample
- onset_time_seconds
- offset_time_seconds

Empty fields do not indicate missing processing. They indicate that
the corresponding annotation information was not available in the
source dataset.

## 10. Annotation inference policy

For Dataset V1.0:

`annotation_inferred = NO`

No automated or manual signal-based repetition segmentation is
included in the frozen release.

## 11. Synchronization policy

Motor-execution EEG and EMG originate from the same OpenBCI stream
and are therefore sample-level synchronized.

Motor-imagery EEG is EEG-only.

Bioimpedance is an independent acquisition and is not assigned
sample-level temporal synchronization with EEG/EMG.

## 12. Data integrity

This annotation stage:

- does not modify raw files;
- does not modify processed signal files;
- does not rename files;
- does not move files;
- does not copy signal files;
- does not rerun preprocessing;
- does not perform feature extraction;
- does not perform machine learning.

## 13. Dataset V1.0 annotation status

The annotation layer is **recording-level / measurement-level only**.

No repetition-level or event-level annotation is included in the
frozen Dataset V1.0 release.
"""


    (OUTPUT / "ANNOTATION_POLICY.md").write_text(
        text,
        encoding="utf-8",
    )


# ====================================================================
# 10. ANNOTATION SCHEMA
# ====================================================================

def create_schema() -> None:

    rows = [
        {
            "field": "annotation_id",
            "type": "string",
            "required": "YES",
            "description":
                "Unique identifier for the recording or BioZ measurement annotation row.",
        },
        {
            "field": "annotation_level",
            "type": "categorical",
            "required": "YES",
            "description":
                "RECORDING for EEG/EMG; MEASUREMENT_UNIT for BioZ.",
        },
        {
            "field": "protocol",
            "type": "categorical",
            "required": "YES",
            "description":
                "MOTOR_EXECUTION or MOTOR_IMAGERY.",
        },
        {
            "field": "modality",
            "type": "categorical",
            "required": "YES",
            "description":
                "EEG, EEG_EMG_SYNCHRONIZED, or BIOIMPEDANCE.",
        },
        {
            "field": "subject_id",
            "type": "string",
            "required": "CONDITIONAL",
            "description":
                "Canonical execution Subject ID where verified.",
        },
        {
            "field": "mi_id",
            "type": "string",
            "required": "CONDITIONAL",
            "description":
                "Independent motor-imagery identifier.",
        },
        {
            "field": "bioz_id",
            "type": "string",
            "required": "CONDITIONAL",
            "description":
                "Bioimpedance acquisition identifier.",
        },
        {
            "field": "gesture_code",
            "type": "categorical",
            "required": "YES",
            "description":
                "Seven-class gesture code.",
        },
        {
            "field": "gesture_name",
            "type": "string",
            "required": "YES",
            "description":
                "Full gesture name.",
        },
        {
            "field": "set_id",
            "type": "categorical",
            "required": "CONDITIONAL",
            "description":
                "Motor-execution or motor-imagery set A/B/C.",
        },
        {
            "field": "measurement_set",
            "type": "integer",
            "required": "CONDITIONAL",
            "description":
                "Bioimpedance measurement unit number.",
        },
        {
            "field": "class_label",
            "type": "categorical",
            "required": "YES",
            "description":
                "Recording-level or measurement-level gesture label.",
        },
        {
            "field": "class_label_source",
            "type": "categorical",
            "required": "YES",
            "description":
                "Identifies the source level of the class label.",
        },
        {
            "field": "event_annotation_available",
            "type": "boolean",
            "required": "YES",
            "description":
                "NO for Dataset V1.0 based on annotation audit.",
        },
        {
            "field": "trigger_annotation_available",
            "type": "boolean",
            "required": "YES",
            "description":
                "NO for Dataset V1.0 based on annotation audit.",
        },
        {
            "field": "repetition_annotation_available",
            "type": "boolean",
            "required": "YES",
            "description":
                "NO for Dataset V1.0 based on annotation audit.",
        },
        {
            "field": "event_id",
            "type": "string",
            "required": "NO",
            "description":
                "Intentionally empty because no event markers exist.",
        },
        {
            "field": "repetition_id",
            "type": "string",
            "required": "NO",
            "description":
                "Intentionally empty because no repetition markers exist.",
        },
        {
            "field": "onset_sample",
            "type": "integer",
            "required": "NO",
            "description":
                "Not provided because no event/repetition onset markers exist.",
        },
        {
            "field": "offset_sample",
            "type": "integer",
            "required": "NO",
            "description":
                "Not provided because no event/repetition offset markers exist.",
        },
        {
            "field": "onset_time_seconds",
            "type": "float",
            "required": "NO",
            "description":
                "Not provided because no event/repetition timing markers exist.",
        },
        {
            "field": "offset_time_seconds",
            "type": "float",
            "required": "NO",
            "description":
                "Not provided because no event/repetition timing markers exist.",
        },
        {
            "field": "annotation_inferred",
            "type": "boolean",
            "required": "YES",
            "description":
                "NO; no annotation was inferred from signal morphology.",
        },
        {
            "field": "raw_file",
            "type": "relative_path",
            "required": "YES",
            "description":
                "Relative path to the authoritative raw source file.",
        },
        {
            "field": "notes",
            "type": "string",
            "required": "NO",
            "description":
                "Annotation interpretation and provenance note.",
        },
    ]

    atomic_write_csv(
        OUTPUT / "ANNOTATION_SCHEMA.csv",
        rows,
        [
            "field",
            "type",
            "required",
            "description",
        ],
    )


# ====================================================================
# 11. FINAL REPORT
# ====================================================================

def create_final_report(
    exec_rows: list[dict],
    mi_rows: list[dict],
    bioz_rows: list[dict],
) -> None:

    report = {
        "dataset_version": DATASET_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": now_iso(),

        "annotation_policy":
            "RECORDING_LEVEL_GESTURE_ONLY",

        "event_annotations_available": False,
        "trigger_annotations_available": False,
        "repetition_annotations_available": False,

        "annotation_inferred": False,
        "event_ids_fabricated": False,
        "repetition_ids_fabricated": False,
        "repetition_boundaries_inferred": False,

        "motor_execution_annotation_rows":
            len(exec_rows),

        "motor_imagery_annotation_rows":
            len(mi_rows),

        "bioimpedance_annotation_rows":
            len(bioz_rows),

        "total_annotation_rows":
            (
                len(exec_rows)
                + len(mi_rows)
                + len(bioz_rows)
            ),

        "expected_motor_execution_rows":
            EXPECTED_EXECUTION,

        "expected_motor_imagery_rows":
            EXPECTED_MI,

        "expected_bioimpedance_rows":
            EXPECTED_BIOZ,

        "raw_data_modified": False,
        "processed_signal_data_modified": False,
        "preprocessing_rerun": False,
        "feature_extraction_performed": False,
        "machine_learning_performed": False,
    }

    (OUTPUT / "ANNOTATION_POLICY_FINAL_REPORT.json").write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )


# ====================================================================
# 12. MAIN
# ====================================================================

def main():

    print()
    print("=" * 72)
    print("ANNOTATION POLICY — DATASET V1.0 FINAL")
    print("=" * 72)
    print(f"ROOT     : {ROOT}")
    print(f"VERSION  : {PROTOCOL_VERSION}")
    print(f"START    : {now_iso()}")
    print("=" * 72)

    required = [
        RAW_EXEC,
        RAW_MI,
        RAW_BIOZ,
        PROC_EEG_EXEC,
        PROC_EEG_MI,
        PROC_EMG,
        PROC_BIOZ,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:

        print("\nERROR: Required dataset paths missing:")

        for path in missing:
            print(f"  {path}")

        sys.exit(1)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[1/4] Building motor-execution annotation manifest...")
    exec_rows = build_execution_inventory()
    print(
        f"       Rows: {len(exec_rows)} "
        f"(expected {EXPECTED_EXECUTION})"
    )

    print("\n[2/4] Building motor-imagery annotation manifest...")
    mi_rows = build_mi_inventory()
    print(
        f"       Rows: {len(mi_rows)} "
        f"(expected {EXPECTED_MI})"
    )

    print("\n[3/4] Building BioZ annotation manifest...")
    bioz_rows = build_bioz_inventory()

    # BioZ raw files have two channels per measurement. Therefore
    # 3780 .spec files correspond to 1890 measurement-level rows.
    print(
        f"       Rows: {len(bioz_rows)} "
        f"(expected {EXPECTED_BIOZ})"
    )

    print("\n[4/4] Writing policy and schema...")
    all_rows = exec_rows + mi_rows + bioz_rows

    fields = [
        "annotation_id",
        "annotation_level",
        "protocol",
        "modality",
        "subject_id",
        "mi_id",
        "bioz_id",
        "gesture_code",
        "gesture_name",
        "set_id",
        "measurement_set",
        "class_label",
        "class_label_source",
        "event_annotation_available",
        "trigger_annotation_available",
        "repetition_annotation_available",
        "event_id",
        "repetition_id",
        "onset_sample",
        "offset_sample",
        "onset_time_seconds",
        "offset_time_seconds",
        "annotation_inferred",
        "raw_file",
        "notes",
    ]

    atomic_write_csv(
        OUTPUT / "RECORDING_LEVEL_ANNOTATION_MANIFEST.csv",
        all_rows,
        fields,
    )

    create_schema()

    create_policy_document(
        len(exec_rows),
        len(mi_rows),
        len(bioz_rows),
    )

    create_final_report(
        exec_rows,
        mi_rows,
        bioz_rows,
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    unique_ids = {
        row["annotation_id"]
        for row in all_rows
    }

    all_gesture_labels_present = all(
        row["gesture_code"]
        and row["gesture_name"]
        and row["class_label"]
        for row in all_rows
    )

    no_inference = all(
        row["annotation_inferred"] == "NO"
        for row in all_rows
    )

    no_event_ids = all(
        row["event_id"] == ""
        for row in all_rows
    )

    no_repetition_ids = all(
        row["repetition_id"] == ""
        for row in all_rows
    )

    no_event_annotations = all(
        row["event_annotation_available"] == "NO"
        for row in all_rows
    )

    no_trigger_annotations = all(
        row["trigger_annotation_available"] == "NO"
        for row in all_rows
    )

    no_repetition_annotations = all(
        row["repetition_annotation_available"] == "NO"
        for row in all_rows
    )

    inventory_pass = (
        len(exec_rows) == EXPECTED_EXECUTION
        and len(mi_rows) == EXPECTED_MI
        and len(bioz_rows) == EXPECTED_BIOZ
    )

    validation_pass = all([
        inventory_pass,
        len(unique_ids) == len(all_rows),
        all_gesture_labels_present,
        no_inference,
        no_event_ids,
        no_repetition_ids,
        no_event_annotations,
        no_trigger_annotations,
        no_repetition_annotations,
    ])

    print()
    print("=" * 72)
    print("ANNOTATION POLICY FINAL SUMMARY")
    print("=" * 72)

    print(
        f"Motor execution rows       : "
        f"{len(exec_rows)}/{EXPECTED_EXECUTION}"
    )

    print(
        f"Motor imagery rows         : "
        f"{len(mi_rows)}/{EXPECTED_MI}"
    )

    print(
        f"BioZ measurement rows      : "
        f"{len(bioz_rows)}/{EXPECTED_BIOZ}"
    )

    print(
        f"Total annotation rows      : "
        f"{len(all_rows)}"
    )

    print(
        f"Unique annotation IDs      : "
        f"{len(unique_ids)}"
    )

    print(
        "Event annotations          : NOT AVAILABLE"
    )

    print(
        "Trigger annotations        : NOT AVAILABLE"
    )

    print(
        "Repetition annotations     : NOT AVAILABLE"
    )

    print(
        "Annotation inference       : NO"
    )

    print(
        "Event IDs fabricated       : NO"
    )

    print(
        "Repetition IDs fabricated  : NO"
    )

    print("-" * 72)

    print(
        "RAW DATA MODIFICATION      : False"
    )

    print(
        "PROCESSED DATA MODIFICATION: False"
    )

    print(
        "PREPROCESSING RERUN        : False"
    )

    print(
        "FEATURE EXTRACTION         : False"
    )

    print(
        "MACHINE LEARNING           : False"
    )

    print("-" * 72)

    print(
        "ANNOTATION POLICY STATUS   : "
        f"{'PASS' if validation_pass else 'FAIL'}"
    )

    print(
        f"OUTPUT                     : {OUTPUT}"
    )

    print("=" * 72)

    if not validation_pass:
        sys.exit(2)


if __name__ == "__main__":
    main()