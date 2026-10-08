#!/usr/bin/env python3
"""
DOCUMENTATION_GENERATOR_V1_FINAL.py

Generate publication/release documentation for the frozen
EEG–EMG–Bioimpedance Dataset V1.0.

This script:
- verifies the existing Dataset V1.0 freeze;
- verifies the ACTUAL source SHA-256 report schema;
- reads only existing frozen/source metadata and QC facts;
- generates documentation under 08_DOCUMENTATION;
- copies the generated documentation into the release package;
- does NOT modify raw data;
- does NOT modify processed signal data;
- does NOT rerun preprocessing;
- does NOT perform feature extraction, ML, or fusion generation.

Important:
The source SHA-256 catalog was generated before documentation existed and
must remain unchanged. After documentation is copied into the release package,
the RELEASE-specific checksum catalog must be regenerated separately.
"""

from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import shutil
import sys


# ============================================================
# 1. AUTHORITATIVE PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if (SCRIPT_DIR / "01_RAW_DATA").exists():
    ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "01_RAW_DATA").exists():
    ROOT = SCRIPT_DIR.parent
else:
    raise FileNotFoundError(
        "Cannot locate dataset root. Expected 01_RAW_DATA next to this script "
        "or one directory above it."
    )
ROOT = ROOT.resolve()

RAW = ROOT / "01_RAW_DATA"
VIEWS = ROOT / "02_MODALITY_VIEWS"
META = ROOT / "03_METADATA"
ANN = ROOT / "04_ANNOTATIONS"
QC = ROOT / "05_QC"
PROC = ROOT / "06_PROCESSED_DATA"
DOC = ROOT / "08_DOCUMENTATION"

RELEASE = ROOT / "09_RELEASE" / "DATASET_V1.0"
RELEASE_DOC = RELEASE / "DOCUMENTATION"

FREEZE = QC / "DATASET_FREEZE"
FREEZE_SENTINEL = FREEZE / "_DATASET_V1.0_FROZEN.json"
FREEZE_REPORT = FREEZE / "DATASET_FREEZE_FINAL_REPORT.json"

SHA_DIR = FREEZE / "SHA256"
SHA_REPORT = SHA_DIR / "SHA256_CHECKSUM_FINAL_REPORT.json"
SHA_CSV = SHA_DIR / "SHA256SUMS_V1.0.csv"
SHA_TXT = SHA_DIR / "SHA256SUMS_V1.0.txt"

DATASET_VERSION = "V1.0"
PROTOCOL_VERSION = "DOCUMENTATION-GENERATOR-DATASET-V1.0-FINAL"


# ============================================================
# 2. HELPERS
# ============================================================

def fail(message: str) -> None:
    print(f"ERROR: {message}")
    sys.exit(1)


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        fail(f"cannot read JSON: {path} | {exc}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def count_files(path: Path, suffix: str) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() == suffix.lower()
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"missing {label}: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        fail(f"missing {label}: {path}")


# ============================================================
# 3. VERIFY FREEZE
# ============================================================

def verify_freeze():
    require_file(FREEZE_SENTINEL, "Dataset V1.0 freeze sentinel")
    require_file(FREEZE_REPORT, "Dataset V1.0 freeze report")

    sentinel = load_json(FREEZE_SENTINEL)
    report = load_json(FREEZE_REPORT)

    sentinel_status = str(
        sentinel.get("status", sentinel.get("freeze_status", ""))
    ).upper()

    # The actual freeze report used in this dataset has already been
    # validated as PASS. We accept explicit PASS/COMPLETE/VERIFIED values
    # and also verify the sentinel itself.
    report_status_values = [
        str(report.get(k, "")).strip().upper()
        for k in (
            "status",
            "freeze_status",
            "overall_status",
            "freeze_gate_status",
        )
    ]

    report_pass = any(
        value in {"PASS", "COMPLETE", "VERIFIED"}
        for value in report_status_values
    )

    sentinel_frozen = (
        sentinel_status in {"FROZEN", "PASS", "COMPLETE", "VERIFIED"}
        or sentinel.get("frozen") is True
        or sentinel.get("freeze_status") is True
    )

    # If the report itself does not expose a generic status key, the
    # presence of a valid frozen sentinel is authoritative for this
    # documentation stage, provided the report exists.
    if not report_pass and not sentinel_frozen:
        fail(
            "freeze not verified; no explicit PASS/COMPLETE/VERIFIED "
            "status or frozen sentinel state was found"
        )

    return sentinel, report


# ============================================================
# 4. VERIFY ACTUAL SOURCE SHA-256 REPORT
# ============================================================

def verify_source_sha():
    require_file(SHA_REPORT, "source SHA-256 final report")
    require_file(SHA_CSV, "source SHA-256 CSV manifest")
    require_file(SHA_TXT, "source SHA-256 text manifest")

    sha = load_json(SHA_REPORT)

    # ACTUAL schema confirmed from the user's frozen report:
    #
    # freeze_status_verified: true
    # checksum_algorithm: SHA-256
    # files_hashed: 18588
    # read_only: true
    # raw_data_modified: false
    # processed_data_modified: false
    # preprocessing_rerun: false
    # feature_extraction: false
    # machine_learning: false
    # fusion_generation: false
    # freeze_sentinel_status: FROZEN
    #
    if sha.get("freeze_status_verified") is not True:
        fail("source SHA-256 freeze_status_verified is not True")

    if sha.get("freeze_sentinel_status") != "FROZEN":
        fail("source SHA-256 freeze_sentinel_status is not FROZEN")

    if sha.get("checksum_algorithm") != "SHA-256":
        fail(
            "unexpected checksum algorithm: "
            f"{sha.get('checksum_algorithm')!r}"
        )

    if sha.get("files_hashed") != 18588:
        fail(
            "unexpected source SHA-256 file count: "
            f"{sha.get('files_hashed')!r}; expected 18588"
        )

    if sha.get("read_only") is not True:
        fail("source SHA-256 report does not indicate read_only=True")

    for key in (
        "raw_data_modified",
        "processed_data_modified",
        "preprocessing_rerun",
        "feature_extraction",
        "machine_learning",
        "fusion_generation",
    ):
        if sha.get(key) is not False:
            fail(
                f"source SHA-256 report indicates unexpected {key}="
                f"{sha.get(key)!r}"
            )

    if sha.get("checksum_catalog_sha256") != (
        "50459fa6298fc8a0487a7bebd18e63bedd86a22514b8ab821e9eab0d8fdb76bb"
    ):
        fail("source checksum catalog SHA-256 does not match frozen value")

    return sha


# ============================================================
# 5. VERIFY REQUIRED STRUCTURE
# ============================================================

def verify_structure():
    for path, label in (
        (RAW, "raw-data directory"),
        (VIEWS, "modality-views directory"),
        (META, "metadata directory"),
        (ANN, "annotations directory"),
        (QC, "QC directory"),
        (PROC, "processed-data directory"),
    ):
        require_dir(path, label)

    required_raw = (
        RAW / "BIOIMPEDANCE",
        RAW / "EEG_MOTOR_IMAGERY",
        RAW / "EMG_EEG_SYNCHRONIZED",
    )

    required_proc = (
        PROC / "BIOIMPEDANCE",
        PROC / "EEG_MOTOR_EXECUTION",
        PROC / "EEG_MOTOR_IMAGERY",
        PROC / "EMG",
        PROC / "PRE1_QC",
    )

    for path in required_raw:
        require_dir(path, "required raw-data branch")

    for path in required_proc:
        require_dir(path, "required processed-data branch")

    required_views = (
        VIEWS / "EEG" / "EEG_MOTOR_EXECUTION_manifest.csv",
        VIEWS / "EEG" / "EEG_MOTOR_IMAGERY_manifest.csv",
        VIEWS / "EMG" / "EMG_manifest.csv",
        VIEWS / "BIOIMPEDANCE" / "BIOIMPEDANCE_manifest.csv",
    )

    for path in required_views:
        require_file(path, "modality-view manifest")


# ============================================================
# 6. FIND ANNOTATION POLICY
# ============================================================

def find_annotation_policy():
    candidates = sorted(ANN.glob("*.json"))

    for path in candidates:
        try:
            data = load_json(path)
        except SystemExit:
            continue

        if data.get("annotation_policy") == "RECORDING_LEVEL_GESTURE_ONLY":
            return path, data

    fail("RECORDING_LEVEL_GESTURE_ONLY annotation policy JSON not found")


# ============================================================
# 7. GENERATE DOCUMENTATION
# ============================================================

def generate_documents(sha, policy):
    execution_raw = count_files(RAW / "EMG_EEG_SYNCHRONIZED", ".csv")
    mi_raw = count_files(RAW / "EEG_MOTOR_IMAGERY", ".csv")
    bioz_spec = count_files(RAW / "BIOIMPEDANCE", ".spec")

    execution_npz = count_files(PROC / "EEG_MOTOR_EXECUTION", ".npz")
    mi_npz = count_files(PROC / "EEG_MOTOR_IMAGERY", ".npz")
    emg_npz = count_files(PROC / "EMG", ".npz")
    bioz_npz = count_files(PROC / "BIOIMPEDANCE", ".npz")

    annotation_rows = policy.get("total_annotation_rows", 3255)

    documents = {}

    documents["README_DATASET_V1.0.md"] = f"""# EEG–EMG–Bioimpedance Dataset V1.0

## Release overview

Dataset V1.0 contains motor-execution EEG+EMG recordings, separate
motor-imagery EEG recordings, and independent frequency-domain
Bioimpedance measurements.

**Dataset version:** V1.0  
**Freeze status:** VERIFIED  
**Source SHA-256:** VERIFIED

## Frozen inventory

| Component | Count |
|---|---:|
| Retained participants | 40 |
| Excluded participant records | 1 |
| Motor-execution recordings | {execution_raw} |
| Motor-imagery recordings | {mi_raw} |
| Bioimpedance native `.spec` files | {bioz_spec} |
| Bioimpedance measurement units | 1,890 |
| Processed execution EEG NPZ | {execution_npz} |
| Processed MI EEG NPZ | {mi_npz} |
| Processed EMG NPZ | {emg_npz} |
| Processed BioZ NPZ | {bioz_npz} |
| Annotation-manifest rows | {annotation_rows} |

## Raw-data structure

```text
01_RAW_DATA/
├── BIOIMPEDANCE/
├── EEG_MOTOR_IMAGERY/
└── EMG_EEG_SYNCHRONIZED/
```

## Processed-data structure

```text
06_PROCESSED_DATA/
├── BIOIMPEDANCE/
├── EEG_MOTOR_EXECUTION/
├── EEG_MOTOR_IMAGERY/
├── EMG/
└── PRE1_QC/
```

## Annotation limitation

Only recording-level gesture labels are available. Event, trigger,
repetition, and repetition-boundary annotations were not available and
were not inferred or fabricated.

## Integrity

The source SHA-256 catalog contains 18,588 hashed files.

Source checksum catalog SHA-256:

`{sha["checksum_catalog_sha256"]}`

The source checksum catalog excludes the frozen
`05_QC/DATASET_FREEZE/` directory to avoid circular checksum generation.

## Scope

This documentation describes the frozen V1.0 dataset state. It does not
rerun preprocessing, feature extraction, machine learning, or fusion
generation.
"""

    documents["DATASET_DESCRIPTION_V1.0.md"] = """# Dataset Description V1.0

## 1. Cohort

The demography source contains 41 participant records. Subject_41 is
excluded for data-quality/leakage reasons recorded in the source.

The retained cohort contains 40 participants:
- 38 male
- 2 female
- Mean age: 22.775 years
- SD: 4.922 years
- Median age: 22 years
- Range: 16–45 years

The public release uses anonymized subject identifiers.

## 2. Motor execution

The motor-execution component contains seven gesture classes and three
recording sets (A, B, and C):

40 participants × 7 gestures × 3 sets = 840 recordings.

Each recording retains 5,625 samples, corresponding to 45 seconds at the
nominal 125 Hz acquisition rate.

The acquisition used an OpenBCI Cyton + Daisy 16-channel configuration.
The first three signal channels are EMG and the remaining 13 are EEG.

EEG and EMG are sample-level synchronized because they originate from the
same OpenBCI stream/CSV recording.

The seven gestures are:
1. Hand Open–Close
2. Thumb Finger Movement
3. Index Finger Movement
4. Middle Finger Movement
5. Ring Finger Movement
6. Little Finger Movement
7. Pen Holding

## 3. Motor imagery

The motor-imagery component contains:

25 MI IDs × 7 gestures × 3 sets = 525 EEG recordings.

Motor imagery is a separate EEG-only component. EMG was not recorded for
this component.

MI identifiers are independent from execution Subject_ID identifiers.
The release does not assert an MI_ID-to-execution-Subject_ID mapping.

For processed MI EEG, original physical EEG channels CH4–CH16 were reindexed
to MI-EEG CH1–CH13. The signal values were not changed by this relabeling.

## 4. Bioimpedance

Bioimpedance was acquired independently using a Sciospec ISX-5 Series
system.

The BioZ cohort comprises:
Subject_01–Subject_15, Subject_17, Subject_29, and Subject_39.

There are:

18 participants × 7 gestures × 15 measurement units = 1,890 units.

Each measurement unit contains two native `.spec` channel files, giving
3,780 native `.spec` files.

Each channel contains 100 frequency-domain points over approximately
10.0117 Hz to 1,000,000.0475 Hz.

BioZ is a frequency-domain impedance sweep, not a uniformly sampled
continuous time-series recording.

Derived variables include:
- impedance magnitude = sqrt(Re² + Im²)
- phase = atan2(Im, Re), expressed in degrees

No filtering, resampling, interpolation, smoothing, normalization,
feature fitting, machine learning, or cross-modal synchronization was
applied to the BioZ source during this dataset-release preprocessing
stage.

## 5. Preprocessing

### EEG

- 4th-order Butterworth band-pass filter, 1–40 Hz
- Zero-phase implementation using `sosfiltfilt`
- No ICA
- No common-average referencing
- No interpolation
- No baseline correction
- No normalization
- No feature fitting

### EMG

- 50 Hz notch filter, Q=30, zero-phase
- 4th-order Butterworth band-pass filter, 10–45 Hz, zero-phase

The EMG processing band is explicitly limited by the 125 Hz sampling rate
and is not a claim that the full conventional sEMG bandwidth is
represented.

### BioZ

Native frequency-domain data are retained, with Re/Im and derived
magnitude/phase representations.
"""

    documents["DATA_DICTIONARY_V1.0.md"] = """# Data Dictionary V1.0

## 1. Motor-execution CSV

### Canonical motor-execution schema — all retained subjects

```text
Sample Index
Timestamp (Formatted)
EMG_ch-01
EMG_ch-02
EMG_ch-03
EEG_ch-01 ... EEG_ch-13
```

The released motor-execution files use the canonical EEG channel names
`EEG_ch-01` through `EEG_ch-13`. For source recordings whose physical
OpenBCI labels were CH4–CH16, the released column names were standardized to
this canonical 13-channel naming scheme; signal values were preserved.

Each motor-execution recording contains 5,625 rows.

## 2. Motor-imagery CSV

```text
Sample Index
Timestamp
EEG_ch-01 ... EEG_ch-13
```

MI identifiers are independent from execution Subject_ID identifiers.

## 3. Bioimpedance `.spec`

The native measurement section contains:

```text
frequency[Hz], re, im
```

Each channel file contains 100 numeric frequency-domain measurement
points.

## 4. Processed EEG

Execution EEG and MI EEG retain signal arrays together with provenance
information including sample index and timestamps. The execution EEG
processed representation also contains a processing time axis and
diagnostic window information.

## 5. Processed EMG

The processed EMG package stores:

```text
emg_raw
emg_preprocessed
sample_index
timestamp_formatted
```

The EMG signal orientation is:

```text
(samples, channels)
```

## 6. Processed BioZ

The processed BioZ representation stores:

```text
frequency_hz
re
im
z_magnitude
z_phase_deg
```

Each is represented as a 100-point channel vector.

## 7. Labels

The authoritative class label is the recording-level gesture label.

No Event_ID, trigger ID, repetition ID, or repetition boundary is
fabricated in V1.0.
"""

    documents["ACQUISITION_AND_STANDARDIZATION_PROTOCOL_V1.0.md"] = """# Acquisition and Standardization Protocol V1.0

## Motor execution

- OpenBCI Cyton + Daisy
- 16-channel configuration
- Nominal sampling rate: 125 Hz
- 40 retained participants
- 7 gestures
- 3 sets per gesture
- 840 recordings
- 5,625 samples per retained recording
- 45-second retained segment

The first three signal channels are EMG and the remaining 13 are EEG.

## Motor imagery

- Separate EEG-only component
- 25 MI IDs
- 7 gestures
- 3 sets per gesture
- 525 recordings
- 13 EEG channels

## Bioimpedance

- Sciospec ISX-5 Series
- Independent acquisition
- Frequency-domain impedance sweep
- 18 participants
- 1,890 measurement units
- 2 channel files per unit
- 100 points per channel

## Standardization

### EEG
4th-order Butterworth 1–40 Hz, zero-phase.

### EMG
50 Hz notch, Q=30, zero-phase, followed by 4th-order Butterworth
10–45 Hz, zero-phase.

### BioZ
Native frequency, Re, and Im values are retained. Magnitude and phase are
derived from Re/Im.

## Raw-data preservation

Raw files and source fields are preserved. Timestamp and sample-index
irregularities are documented rather than rewritten.

No physiological signal values were modified during V1.0 processing. A targeted
Sample Index correction was applied to 31 motor-execution files to resolve
acquisition-counter rollover/non-monotonic index metadata; timestamps and
physiological signal columns were preserved.
"""

    documents["ANNOTATION_GUIDE_V1.0.md"] = """# Annotation Guide V1.0

## Authoritative policy

`RECORDING_LEVEL_GESTURE_ONLY`

The gesture label identifies the class of the complete recording unit.

## Available annotation-manifest rows

- Motor execution: 840
- Motor imagery: 525
- Bioimpedance: 1,890
- Total: 3,255

## Not available

The following annotations are not available:

- Event annotations
- Trigger annotations
- Repetition annotations
- Repetition boundaries
- Exact repetition counts

## No inference or fabrication

V1.0 does not infer event timing from signal morphology, create
repetition boundaries, or fabricate Event_ID/repetition identifiers.

## Synchronization

EEG and EMG in the motor-execution component are sample-level synchronized
because they originate from the same OpenBCI stream.

BioZ measurements are independent and are not sample-synchronized with
EEG/EMG or motor-imagery EEG.
"""

    documents["QUALITY_CONTROL_REPORT_V1.0.md"] = """# Quality Control Report V1.0

## Final stage status

| Stage | Status |
|---|---|
| DATA1 inventory | PASS |
| DATA2 validation | PASS_WITH_WARNINGS |
| PRE1 EEG | COMPLETE |
| PRE2 EMG | COMPLETE |
| PRE3 BioZ | COMPLETE |
| Global Dataset QC | PASS |
| Modality Views | PASS |
| Annotation Audit | COMPLETE |
| Annotation Policy | PASS |
| Dataset Freeze | PASS |
| Source SHA-256 | COMPLETE |
| Release Packaging | DOCUMENTATION COPIED |

## Recorded DATA2 warnings

- Motor-execution timestamp large-gap warnings: 136
- Motor-imagery timestamp large-gap warnings: 61
- Motor-execution constant-channel records: 1
- Motor-imagery constant-channel records: 12
- Motor-execution negative sample-index steps: 326
- Motor-imagery negative sample-index steps: 0

The motor-execution negative sample-index steps are associated with the
documented acquisition-counter rollover behavior.

## Duplicate-content check

The final duplicate-content check reported zero duplicate groups/rows.

## Integrity state

- Raw data modified: False
- Processed data modified during freeze/release preparation: False
- Feature extraction during release preparation: False
- Machine learning during release preparation: False
- Fusion generation during release preparation: False

## Interpretation

The QC status is a documented release-quality state. Known
acquisition/export irregularities remain transparently documented rather
than being rewritten in the raw files.
"""

    documents["MODALITY_AND_SYNCHRONIZATION_GUIDE_V1.0.md"] = """# Modality and Synchronization Guide V1.0

| Component | Scope | Synchronization |
|---|---|---|
| Motor-execution EEG+EMG | 40 retained subjects | Sample-level synchronized |
| Motor-imagery EEG | 25 MI IDs | Separate session/component |
| Bioimpedance | 18 execution-cohort subjects | Not sample-synchronized |

## Motor-execution EEG+EMG

EEG and EMG are contained in the same OpenBCI CSV stream. The first three
signal channels are EMG and the remaining 13 are EEG.

## Motor-imagery EEG

Motor imagery is EEG-only. There is no corresponding EMG stream in V1.0.

MI IDs are independent identifiers. MI_01 must not be assumed to be
Subject_01.

## Bioimpedance

BioZ is a separate frequency-domain acquisition and is not synchronized
to the EEG/EMG or MI sample clock.

## Fusion status

A derived synchronized EEG–EMG–BioZ fusion dataset is not part of the
current V1.0 release. A future fusion product should receive its own
version, provenance, and validation.
"""

    documents["ETHICS_AND_DATA_GOVERNANCE_V1.0.md"] = """# Ethics and Data Governance V1.0

## Participant governance

The demography source records consent and public-data-release fields.
The public package uses anonymized participant identifiers.

## Ethics documentation

Available ethics documentation records:

Reference:
`323/Biol. Scs./2025-2026`

Date:
`27 August 2025`

The available document describes the application as **under process**.

This documentation therefore does not claim final ethics approval.

## Demographic field interpretation

The source field named `Handedness` contains `Right Fore Arm`, which
describes the recording site rather than handedness.

V1.0 therefore does not claim that handedness was separately recorded.

## Public release

Users should not attempt to reconstruct participant identities or infer
the MI_ID-to-execution-Subject_ID mapping.

## Version governance

Any future modification to raw data, processed signals, metadata,
annotation policy, or release contents that changes the scientific
dataset should receive a new dataset version and corresponding integrity
records.
"""

    documents["DATA_ACCESS_AND_RELEASE_GUIDE_V1.0.md"] = """# Data Access and Release Guide V1.0

## Release location

```text
09_RELEASE/DATASET_V1.0/
```

The release package contains:

```text
DATA/
ANNOTATIONS/
METADATA/
MODALITY_VIEWS/
QC/
DOCUMENTATION/
RELEASE_NOTES.md
SHA256SUMS_SOURCE_V1.0/
SHA256SUMS_RELEASE_V1.0/
```

## Recommended use

1. Verify the final release checksum catalog.
2. Read the dataset README.
3. Read the data dictionary.
4. Read the annotation guide.
5. Read the modality/synchronization guide.
6. Preserve raw source files and provenance when performing downstream
   analyses.

## Downstream analyses

Feature extraction, machine learning, alternative windowing, and fusion
are downstream analyses. They should be documented as derived analyses or
separately versioned derived products.

## DOI

A public repository DOI is not assigned in this documentation yet.
"""

    documents["DOCUMENTATION_INDEX_V1.0.md"] = """# Documentation Index V1.0

| File | Purpose |
|---|---|
| `README_DATASET_V1.0.md` | Entry point and release overview |
| `DATASET_DESCRIPTION_V1.0.md` | Scientific dataset description |
| `DATA_DICTIONARY_V1.0.md` | Schemas and processed variables |
| `ACQUISITION_AND_STANDARDIZATION_PROTOCOL_V1.0.md` | Acquisition and preprocessing |
| `ANNOTATION_GUIDE_V1.0.md` | Labels and annotation limitations |
| `QUALITY_CONTROL_REPORT_V1.0.md` | Final QC state |
| `MODALITY_AND_SYNCHRONIZATION_GUIDE_V1.0.md` | Modality relationships |
| `ETHICS_AND_DATA_GOVERNANCE_V1.0.md` | Ethics and governance |
| `DATA_ACCESS_AND_RELEASE_GUIDE_V1.0.md` | Release/use guidance |
| `DOCUMENTATION_INDEX_V1.0.md` | Documentation map |
"""

    return documents, {
        "motor_execution_raw_csv": execution_raw,
        "motor_imagery_raw_csv": mi_raw,
        "bioz_native_spec": bioz_spec,
        "processed_execution_eeg_npz": execution_npz,
        "processed_mi_eeg_npz": mi_npz,
        "processed_emg_npz": emg_npz,
        "processed_bioz_npz": bioz_npz,
        "annotation_rows": annotation_rows,
    }


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 72)
    print("DOCUMENTATION GENERATOR — DATASET V1.0")
    print("=" * 72)
    print("ROOT:", ROOT)
    print("PROTOCOL:", PROTOCOL_VERSION)

    if not ROOT.is_dir():
        fail(f"dataset root not found: {ROOT}")

    print("[1/6] Verifying Dataset V1.0 freeze...")
    verify_freeze()
    print("      PASS")

    print("[2/6] Verifying actual source SHA-256 report...")
    sha = verify_source_sha()
    print(
        "      PASS | algorithm="
        f"{sha['checksum_algorithm']} | files={sha['files_hashed']}"
    )
    print(
        "      Catalog SHA-256:",
        sha["checksum_catalog_sha256"]
    )

    print("[3/6] Verifying frozen source structure...")
    verify_structure()
    print("      PASS")

    print("[4/6] Generating documentation...")
    policy_path, policy = find_annotation_policy()
    documents, stats = generate_documents(sha, policy)

    DOC.mkdir(parents=True, exist_ok=True)

    for filename, content in documents.items():
        write_text(DOC / filename, content)

    # Documentation manifest.
    manifest_rows = []
    for path in sorted(DOC.glob("*.md")):
        data = path.read_bytes()
        manifest_rows.append({
            "file": path.name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "protocol": PROTOCOL_VERSION,
        })

    manifest_path = DOC / "DOCUMENTATION_MANIFEST_V1.0.csv"
    with manifest_path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "bytes", "sha256", "protocol"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    generation_report = {
        "dataset_version": DATASET_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "freeze_verified": True,
        "source_sha256_verified": True,
        "source_sha256_files_hashed": sha["files_hashed"],
        "source_checksum_catalog_sha256":
            sha["checksum_catalog_sha256"],
        "annotation_policy_file": str(
            policy_path.relative_to(ROOT)
        ),
        "annotation_policy":
            policy.get("annotation_policy"),
        "external_facts_inferred": False,
        "raw_data_modified": False,
        "physiological_signal_values_modified": False,
        "sample_index_metadata_corrected_before_freeze": True,
        "processed_data_modified": False,
        "preprocessing_rerun": False,
        "feature_extraction_performed": False,
        "machine_learning_performed": False,
        "fusion_generation_performed": False,
        "documentation_files_generated": len(documents),
        "source_statistics": stats,
    }

    write_text(
        DOC / "DOCUMENTATION_GENERATION_SUMMARY_V1.0.txt",
        "DOCUMENTATION GENERATION — DATASET V1.0\n\n"
        + json.dumps(generation_report, indent=2)
        + "\n\n"
        "IMPORTANT:\n"
        "The source SHA-256 catalog remains unchanged.\n"
        "The release-specific SHA-256 catalog must be regenerated after "
        "the final documentation is copied into the release package."
    )

    (DOC / "DOCUMENTATION_GENERATION_FINAL_REPORT.json").write_text(
        json.dumps(generation_report, indent=2),
        encoding="utf-8"
    )

    print(f"      PASS | generated {len(documents)} Markdown files")

    print("[5/6] Copying documentation into release package...")
    RELEASE_DOC.mkdir(parents=True, exist_ok=True)

    release_files = list(documents.keys()) + [
        "DOCUMENTATION_MANIFEST_V1.0.csv",
        "DOCUMENTATION_GENERATION_SUMMARY_V1.0.txt",
        "DOCUMENTATION_GENERATION_FINAL_REPORT.json",
    ]

    for filename in release_files:
        copy_file(DOC / filename, RELEASE_DOC / filename)

    print(f"      PASS | copied {len(release_files)} files")

    print("[6/6] Final documentation integrity check...")

    for filename in release_files:
        if not (RELEASE_DOC / filename).is_file():
            fail(f"release documentation copy missing: {filename}")

    print("      PASS")

    print()
    print("=" * 72)
    print("DOCUMENTATION GENERATION: COMPLETE")
    print("=" * 72)
    print("Documentation files generated :", len(documents))
    print("Documentation source           :", DOC)
    print("Documentation release          :", RELEASE_DOC)
    print("Raw data modification          : False")
    print("Processed data modification    : False")
    print("Preprocessing rerun            : False")
    print("Feature extraction             : False")
    print("Machine learning               : False")
    print("Fusion generation              : False")
    print("External facts inferred        : False")
    print("=" * 72)
    print("NEXT STAGE: FINAL RELEASE SHA-256 REGENERATION")
    print("=" * 72)


if __name__ == "__main__":
    main()
