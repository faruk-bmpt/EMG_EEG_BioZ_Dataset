
#!/usr/bin/env python3
"""
DATA1_MASTER_INVENTORY.py
Stage 1: Master Dataset Inventory & Cross-Modality Mapping

Canonical identity source:
    Data set Demography (3).xlsx

ME:
    Subject_01–Subject_40

MI:
    MI_01–MI_25 mapped using the workbook's MI_ID column.

BioZ:
    BZ_01–BZ_18 mapped using the workbook's BZ_ID column.

Raw recordings are never renamed, moved, or modified.
"""

from __future__ import annotations

import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(
    r"F:\Faruk\OFS_Paper_Work\Data_Set_Paper_Work\EEG_EMG_BIOZ_DATASET"
).resolve()

RAW = ROOT / "01_RAW_DATA"
EXEC = RAW / "EMG_EEG_SYNCHRONIZED"
MI = RAW / "EEG_MOTOR_IMAGERY"
BIOZ = RAW / "BIOIMPEDANCE"

META = ROOT / "03_METADATA"
QC = ROOT / "05_QC"
LOGS = ROOT / "logs"

DEMOGRAPHY_FILENAMES = (
    "Data set Demography.xlsx",
    "Data set Demography .xlsx",
)

DEMOGRAPHY_CANDIDATES = [
    parent / filename
    for filename in DEMOGRAPHY_FILENAMES
    for parent in (ROOT, ROOT.parent, ROOT.parent.parent)
]

GESTURES = [
    ("HOC", "Hand_Open_Close", "Hand Open–Close"),
    ("TFM", "Thumb_Finger_Movement", "Thumb Finger Movement"),
    ("IFM", "Index_Finger_Movement", "Index Finger Movement"),
    ("MFM", "Middle_Finger_Movement", "Middle Finger Movement"),
    ("RFM", "Ring_Finger_Movement", "Ring Finger Movement"),
    ("LFM", "Little_Finger_Movement", "Little Finger Movement"),
    ("PH", "Pen_Holding", "Pen Holding"),
]

SETS = ("A", "B", "C")


# ============================================================
# LOGGING AND UTILITIES
# ============================================================

def logger() -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)

    lg = logging.getLogger("DATA1")
    lg.handlers.clear()
    lg.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    lg.addHandler(console)

    file_handler = logging.FileHandler(
        LOGS / "DATA1_MASTER_INVENTORY.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    lg.addHandler(file_handler)

    return lg


L = logger()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    L.info(
        "WROTE | %s | rows=%d | columns=%d",
        rel(path),
        len(df),
        len(df.columns),
    )


def validate_roots() -> bool:
    L.info("=" * 80)
    L.info("DATA1 MASTER DATASET INVENTORY")
    L.info("=" * 80)

    roots = [
        ("ROOT", ROOT),
        ("RAW", RAW),
        ("EXECUTION", EXEC),
        ("IMAGERY", MI),
        ("BIOIMPEDANCE", BIOZ),
    ]

    for label, path in roots:
        if path.is_dir():
            L.info("FOUND | %-13s | %s", label, path)
        else:
            L.error("MISSING | %-12s | %s", label, path)

    return all(path.is_dir() for _, path in roots)


# ============================================================
# DEMOGRAPHIC WORKBOOK
# ============================================================

def find_demography_workbook() -> Optional[Path]:
    for path in DEMOGRAPHY_CANDIDATES:
        if path.is_file():
            return path

    search_root = ROOT.parent
    matches = []

    try:
        for pattern in (
            "Data set Demography*.xlsx",
            "Data Set Demography*.xlsx",
        ):
            matches.extend(search_root.rglob(pattern))
    except OSError as exc:
        L.warning("Workbook fallback search failed: %s", exc)

    matches = sorted(
        {
            path.resolve()
            for path in matches
            if path.is_file() and RAW not in path.parents
        },
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if matches:
        L.warning(
            "Using most recently modified matching workbook: %s",
            matches[0],
        )
        return matches[0]

    return None


def clean_demography_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Subject Name": "subject_name",
        "Participant_ID": "subject_id",
        "MI_ID": "mi_id",
        "BZ_ID": "bioz_id",
        "Gender ": "gender",
        "Age /Year": "age_years",
        "Height ft/cm": "height_raw",
        "Weight /kg": "weight_raw",
        "Handedness": "recording_site",
        "Fore Arm Lenght /cm": "forearm_length_raw",
        "Fore Arm Circumference /cm": "forearm_circumference_raw",
        "BCI_Experience": "bci_experience",
        "Inclusion_Criteria_Met": "inclusion_criteria_met",
        "Neurological_Exclusion": "neurological_exclusion",
        "Musculoskeletal_Exclusion": "musculoskeletal_exclusion",
        "Study Consent": "study_consent",
        "Eligible_for_Study": "eligible_for_study",
        "Public_Data_Release_Consent": "public_data_release_consent",
        "Participant_Status": "participant_status",
        "Data Packet Loss": "data_packet_loss",
        "Live Recording Status ": "live_recording_status",
        "OpenBCI Recording (EEG, Executoin + EMG)": (
            "openbci_execution_emg_recording"
        ),
        "OpenBCI Recoriding (EEG, Imagery  only)": (
            "openbci_motor_imagery_recording"
        ),
        "Bioimpedence data set ": "bioimpedance_dataset_contribution",
        "Data Contribution Modalities ": "data_contribution_modalities",
    }

    df = df.rename(
        columns={
            column: rename.get(column, str(column).strip())
            for column in df.columns
        }
    )

    return df.dropna(axis=1, how="all").copy()


def load_demography() -> tuple[pd.DataFrame, Path]:
    path = find_demography_workbook()

    if path is None:
        raise FileNotFoundError(
            "The demographic workbook was not found. Expected one of: "
            + ", ".join(DEMOGRAPHY_FILENAMES)
        )

    excel = pd.ExcelFile(path)
    df = None
    selected_sheet = None

    for sheet in excel.sheet_names:
        candidate = pd.read_excel(path, sheet_name=sheet)
        candidate = candidate.dropna(how="all")

        if not candidate.empty:
            df = candidate
            selected_sheet = sheet
            break

    if df is None:
        raise ValueError("No non-empty worksheet was found.")

    df = clean_demography_columns(df)

    required_columns = {"subject_id", "mi_id", "bioz_id"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required demographic columns: "
            f"{sorted(missing_columns)}"
        )

    df["subject_id"] = df["subject_id"].astype(str).str.strip()

    if df["subject_id"].duplicated().any():
        duplicates = sorted(
            df.loc[
                df["subject_id"].duplicated(keep=False),
                "subject_id",
            ].unique()
        )
        raise ValueError(
            f"Duplicate canonical participant IDs: {duplicates}"
        )

    for column, pattern, expected_count in (
        ("mi_id", r"MI_\d{2}", 25),
        ("bioz_id", r"BZ_\d{2}", 18),
    ):
        df[column] = df[column].fillna("").astype(str).str.strip()
        df.loc[
            df[column].str.lower().isin({"nan", "none"}),
            column,
        ] = ""

        populated = df.loc[df[column].ne(""), column]

        invalid = sorted(
            populated.loc[
                ~populated.str.fullmatch(pattern, case=False)
            ].unique()
        )

        if invalid:
            raise ValueError(
                f"Invalid values in {column}: {invalid}"
            )

        if populated.duplicated().any():
            duplicates = sorted(
                populated.loc[
                    populated.duplicated(keep=False)
                ].unique()
            )
            raise ValueError(
                f"Duplicate values in {column}: {duplicates}"
            )

        if len(populated) != expected_count:
            raise ValueError(
                f"{column}: expected {expected_count} mappings, "
                f"found {len(populated)}."
            )

    expected_subjects = {
        f"Subject_{i:02d}" for i in range(1, 42)
    }
    actual_subjects = set(df["subject_id"])

    if actual_subjects != expected_subjects:
        raise ValueError(
            "Demographic participant IDs do not match the expected "
            "41-row source. "
            f"Missing={sorted(expected_subjects - actual_subjects)}; "
            f"unexpected={sorted(actual_subjects - expected_subjects)}"
        )

    expected_mi_subjects = (
        {"Subject_01", "Subject_02", "Subject_05"}
        | {f"Subject_{i:02d}" for i in range(18, 25)}
        | {f"Subject_{i:02d}" for i in range(26, 41)}
    )

    actual_mi_subjects = set(
        df.loc[df["mi_id"].ne(""), "subject_id"]
    )

    if actual_mi_subjects != expected_mi_subjects:
        raise ValueError(
            "MI cohort mismatch. "
            f"Missing={sorted(expected_mi_subjects - actual_mi_subjects)}; "
            f"unexpected={sorted(actual_mi_subjects - expected_mi_subjects)}"
        )

    expected_bioz_subjects = (
        {f"Subject_{i:02d}" for i in range(1, 16)}
        | {"Subject_17", "Subject_29", "Subject_39"}
    )

    actual_bioz_subjects = set(
        df.loc[df["bioz_id"].ne(""), "subject_id"]
    )

    if actual_bioz_subjects != expected_bioz_subjects:
        raise ValueError(
            "BioZ cohort mismatch. "
            f"Missing={sorted(expected_bioz_subjects - actual_bioz_subjects)}; "
            f"unexpected={sorted(actual_bioz_subjects - expected_bioz_subjects)}"
        )

    df["_order"] = df["subject_id"].str.extract(r"(\d+)$")[0].astype(int)
    df = (
        df.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    df.attrs["source_sheet"] = selected_sheet

    L.info(
        "DEMOGRAPHY | source=%s | sheet=%s | rows=%d",
        path,
        selected_sheet,
        len(df),
    )
    L.info("MI canonical cohort: PASS (%d participants)", len(actual_mi_subjects))
    L.info("BioZ canonical cohort: PASS (%d participants)", len(actual_bioz_subjects))

    return df, path


# ============================================================
# CANONICAL PARTICIPANT METADATA
# ============================================================

def build_participants() -> tuple[pd.DataFrame, Path]:
    demo, source_path = load_demography()

    source_sheet = demo.attrs.get("source_sheet", "unknown")

    if "participant_status" in demo.columns:
        status = demo["participant_status"].astype(str).str.strip().str.lower()
        demo["cohort_status"] = status.map(
            lambda value: "retained" if value == "included" else "excluded"
        )
    else:
        demo["cohort_status"] = "retained"

    demo.loc[
        demo["subject_id"].eq("Subject_41"),
        "cohort_status",
    ] = "excluded"

    demo["exclusion_reason"] = ""
    demo.loc[
        demo["subject_id"].eq("Subject_41"),
        "exclusion_reason",
    ] = "data quality, leakage"

    demo["demographic_source_file"] = source_path.name
    demo["demographic_source_sheet"] = source_sheet
    demo["demographic_merge_status"] = "MERGED"

    first_columns = [
        "subject_id",
        "mi_id",
        "bioz_id",
        "cohort_status",
        "exclusion_reason",
        "subject_name",
        "gender",
        "age_years",
        "height_raw",
        "weight_raw",
        "recording_site",
        "forearm_length_raw",
        "forearm_circumference_raw",
        "bci_experience",
        "inclusion_criteria_met",
        "neurological_exclusion",
        "musculoskeletal_exclusion",
        "study_consent",
        "eligible_for_study",
        "public_data_release_consent",
        "participant_status",
        "data_packet_loss",
        "live_recording_status",
        "openbci_execution_emg_recording",
        "openbci_motor_imagery_recording",
        "bioimpedance_dataset_contribution",
        "data_contribution_modalities",
        "demographic_source_file",
        "demographic_source_sheet",
        "demographic_merge_status",
    ]

    ordered = (
        [column for column in first_columns if column in demo.columns]
        + [column for column in demo.columns if column not in first_columns]
    )

    return demo[ordered], source_path


def build_crosswalk(
    participants: pd.DataFrame,
    discovered_mi_ids: set[str],
    discovered_bioz_ids: set[str],
) -> pd.DataFrame:
    rows = []

    for _, row in participants.iterrows():
        canonical_id = str(row["subject_id"]).strip()
        mi_id = str(row["mi_id"]).strip()
        bioz_id = str(row["bioz_id"]).strip()

        mi_id = "" if mi_id.lower() in {"", "nan", "none"} else mi_id
        bioz_id = "" if bioz_id.lower() in {"", "nan", "none"} else bioz_id

        rows.append(
            {
                "canonical_subject_id": canonical_id,
                "execution_id": (
                    canonical_id
                    if canonical_id in {
                        f"Subject_{i:02d}" for i in range(1, 41)
                    }
                    else ""
                ),
                "mi_id": mi_id,
                "bioz_id": bioz_id,
                "execution_available": canonical_id != "Subject_41",
                "mi_available": bool(
                    mi_id and mi_id.upper() in discovered_mi_ids
                ),
                "bioz_available": bool(
                    bioz_id and bioz_id.upper() in discovered_bioz_ids
                ),
                "mi_mapping_status": (
                    "MAPPED"
                    if mi_id and mi_id.upper() in discovered_mi_ids
                    else "NOT_AVAILABLE"
                ),
                "bioz_mapping_status": (
                    "MAPPED"
                    if bioz_id and bioz_id.upper() in discovered_bioz_ids
                    else "NOT_AVAILABLE"
                ),
                "notes": (
                    "Canonical correspondence sourced from the demographic "
                    "workbook; modality-specific identifiers are preserved."
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MOTOR EXECUTION / MOTOR IMAGERY INVENTORY
# ============================================================

def inventory_csv_protocol(
    root: Path,
    protocol: str,
    local_to_canonical: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not root.is_dir():
        raise FileNotFoundError(f"Protocol directory not found: {root}")

    if protocol == "MOTOR_EXECUTION":
        subject_pattern = re.compile(r"Subject_\d{2}", re.I)
        prefix = "S"
    else:
        subject_pattern = re.compile(r"MI_\d{2}", re.I)
        prefix = "MI"

    subjects = sorted(
        [
            path for path in root.iterdir()
            if path.is_dir() and subject_pattern.fullmatch(path.name)
        ],
        key=lambda path: int(re.search(r"(\d+)$", path.name).group(1)),
    )

    L.info("%s folders discovered: %d", protocol, len(subjects))

    inventory = []
    completeness = []

    for subject_dir in subjects:
        number = int(re.search(r"(\d+)$", subject_dir.name).group(1))

        for gesture_code, gesture_folder, gesture_label in GESTURES:
            gesture_dir = subject_dir / gesture_folder

            for recording_set in SETS:
                if protocol == "MOTOR_EXECUTION":
                    filename = (
                        f"S{number:02d}_{gesture_code}_Set{recording_set}.csv"
                    )
                else:
                    filename = (
                        f"MI_{number:02d}_{gesture_code}_Set{recording_set}.csv"
                    )

                log_filename = filename[:-4] + "_log.txt"
                csv_path = gesture_dir / filename
                log_path = gesture_dir / log_filename

                if not csv_path.is_file() and gesture_dir.is_dir():
                    matches = [
                        path for path in gesture_dir.iterdir()
                        if path.is_file()
                        and path.suffix.lower() == ".csv"
                        and path.name.lower() == filename.lower()
                    ]
                    if matches:
                        csv_path = matches[0]

                if not log_path.is_file() and gesture_dir.is_dir():
                    matches = [
                        path for path in gesture_dir.iterdir()
                        if path.is_file()
                        and path.name.lower() == log_filename.lower()
                    ]
                    if matches:
                        log_path = matches[0]

                rows = ""
                columns = ""
                column_names = ""
                read_status = "NOT_READ"

                if csv_path.is_file():
                    try:
                        signal_df = pd.read_csv(csv_path)
                        rows, columns = signal_df.shape
                        column_names = " | ".join(map(str, signal_df.columns))
                        read_status = "PASS"
                    except Exception as exc:
                        read_status = f"READ_ERROR:{type(exc).__name__}"

                complete = csv_path.is_file() and log_path.is_file()

                if protocol == "MOTOR_EXECUTION":
                    canonical_id = subject_dir.name
                else:
                    canonical_id = (local_to_canonical or {}).get(
                        subject_dir.name.upper(), ""
                    )

                inventory.append(
                    {
                        "modality": (
                            "EEG+EMG"
                            if protocol == "MOTOR_EXECUTION"
                            else "EEG"
                        ),
                        "protocol": protocol,
                        "subject_id": subject_dir.name,
                        "canonical_subject_id": canonical_id,
                        "gesture_code": gesture_code,
                        "gesture": gesture_label,
                        "gesture_folder": gesture_folder,
                        "set": recording_set,
                        "csv_file": (
                            csv_path.name if csv_path.is_file() else ""
                        ),
                        "csv_relative_path": (
                            rel(csv_path) if csv_path.is_file() else ""
                        ),
                        "log_file": (
                            log_path.name if log_path.is_file() else ""
                        ),
                        "log_relative_path": (
                            rel(log_path) if log_path.is_file() else ""
                        ),
                        "csv_exists": csv_path.is_file(),
                        "log_exists": log_path.is_file(),
                        "csv_rows": rows,
                        "csv_columns": columns,
                        "csv_column_names": column_names,
                        "csv_read_status": read_status,
                        "complete_slot": complete,
                    }
                )

                completeness.append(
                    {
                        "protocol": protocol,
                        "subject_id": subject_dir.name,
                        "canonical_subject_id": canonical_id,
                        "gesture_code": gesture_code,
                        "gesture": gesture_label,
                        "set": recording_set,
                        "csv_exists": csv_path.is_file(),
                        "log_exists": log_path.is_file(),
                        "complete_slot": complete,
                        "status": (
                            "PASS"
                            if complete
                            else (
                                "MISSING_CSV"
                                if not csv_path.is_file()
                                else "MISSING_LOG"
                            )
                        ),
                    }
                )

    return pd.DataFrame(inventory), pd.DataFrame(completeness)


# ============================================================
# BIOIMPEDANCE INVENTORY
# ============================================================

def inventory_bioz(
    bioz_to_canonical: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(
        [
            path for path in BIOZ.iterdir()
            if path.is_dir()
            and re.fullmatch(r"BZ_\d{2}", path.name, re.I)
        ],
        key=lambda path: int(path.name[-2:]),
    )

    L.info("BioZ participant folders discovered: %d", len(subjects))

    inventory = []
    completeness = []

    for subject_dir in subjects:
        canonical_id = bioz_to_canonical.get(subject_dir.name.upper(), "")
        groups = defaultdict(list)

        for spec_path in subject_dir.rglob("*.spec"):
            path_parts = {part.lower() for part in spec_path.parts}
            gesture_match = None

            for code, folder, label in GESTURES:
                if folder.lower() in path_parts:
                    gesture_match = (code, folder, label)
                    break

            if gesture_match is None:
                path_text = str(spec_path).lower()

                for code, folder, label in GESTURES:
                    if folder.lower() in path_text or code.lower() in path_text:
                        gesture_match = (code, folder, label)
                        break

            if gesture_match:
                code, folder, label = gesture_match
                groups[(code, spec_path.stem)].append(spec_path)

        for code, folder, label in GESTURES:
            entries = sorted(
                [
                    (stem, files)
                    for (group_code, stem), files in groups.items()
                    if group_code == code
                ],
                key=lambda item: item[0],
            )

            for index, (stem, files) in enumerate(entries, start=1):
                channel_names = []

                for file_path in files:
                    match = re.search(
                        r"Channel[_ -]?(\d+)",
                        str(file_path),
                        re.I,
                    )
                    if match:
                        channel_names.append(f"Channel_{match.group(1)}")

                channel_set = set(channel_names)
                has_channel_1 = "Channel_1" in channel_set
                has_channel_2 = "Channel_2" in channel_set

                inventory.append(
                    {
                        "modality": "BIOIMPEDANCE",
                        "protocol": "MOTOR_EXECUTION",
                        "subject_id": subject_dir.name,
                        "canonical_subject_id": canonical_id,
                        "gesture_code": code,
                        "gesture": label,
                        "gesture_folder": folder,
                        "recording_index_within_gesture": index,
                        "recording_stem": stem,
                        "native_spec_file_count": len(files),
                        "channel_names": " | ".join(sorted(channel_set)),
                        "native_spec_relative_paths": " | ".join(
                            sorted(rel(file_path) for file_path in files)
                        ),
                        "has_channel_1": has_channel_1,
                        "has_channel_2": has_channel_2,
                        "two_channel_complete": (
                            has_channel_1
                            and has_channel_2
                            and len(files) == 2
                        ),
                    }
                )

            gesture_entries = [
                files for (group_code, _), files in groups.items()
                if group_code == code
            ]
            spec_count = sum(len(files) for files in gesture_entries)

            complete_units = sum(
                1
                for files in gesture_entries
                if len(files) == 2
                and any(
                    re.search(r"Channel[_ -]?1", str(path), re.I)
                    for path in files
                )
                and any(
                    re.search(r"Channel[_ -]?2", str(path), re.I)
                    for path in files
                )
            )

            completeness.append(
                {
                    "protocol": "MOTOR_EXECUTION",
                    "modality": "BIOIMPEDANCE",
                    "bioz_id": subject_dir.name,
                    "canonical_subject_id": canonical_id,
                    "gesture_code": code,
                    "gesture": label,
                    "expected_recordings": 15,
                    "actual_recordings": len(gesture_entries),
                    "recording_count_pass": len(gesture_entries) == 15,
                    "expected_native_spec_files": 30,
                    "actual_native_spec_files": spec_count,
                    "native_spec_count_pass": spec_count == 30,
                    "two_channel_recordings": complete_units,
                }
            )

    return pd.DataFrame(inventory), pd.DataFrame(completeness)


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    started = datetime.now()

    if not validate_roots():
        return 2

    participants, demographic_source = build_participants()

    mi_to_canonical = {
        str(row["mi_id"]).upper(): row["subject_id"]
        for _, row in participants.loc[
            participants["mi_id"].ne("")
        ].iterrows()
    }

    bioz_to_canonical = {
        str(row["bioz_id"]).upper(): row["subject_id"]
        for _, row in participants.loc[
            participants["bioz_id"].ne("")
        ].iterrows()
    }

    execution_inventory, execution_qc = inventory_csv_protocol(
        EXEC,
        "MOTOR_EXECUTION",
    )

    mi_inventory, mi_qc = inventory_csv_protocol(
        MI,
        "MOTOR_IMAGERY",
        mi_to_canonical,
    )

    bioz_inventory, bioz_qc = inventory_bioz(bioz_to_canonical)

    META.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Canonical participant metadata and crosswalk
    # --------------------------------------------------------

    write(participants, META / "participants.csv")

    crosswalk = build_crosswalk(
        participants,
        set(mi_inventory["subject_id"].str.upper())
        if not mi_inventory.empty else set(),
        set(bioz_inventory["subject_id"].str.upper())
        if not bioz_inventory.empty else set(),
    )

    write(crosswalk, META / "subject_crosswalk.csv")
    write(crosswalk.copy(), META / "subject_modality_availability.csv")

    # --------------------------------------------------------
    # Master recording inventory
    # --------------------------------------------------------

    master_tables = []

    for table in (execution_inventory, mi_inventory):
        if not table.empty:
            master_tables.append(
                table.assign(
                    recording_unit=True,
                    native_spec_file_count="",
                )
            )

    if not bioz_inventory.empty:
        bioz_master = bioz_inventory.copy()
        bioz_master["set"] = ""
        bioz_master["csv_file"] = ""
        bioz_master["csv_relative_path"] = ""
        bioz_master["log_file"] = ""
        bioz_master["log_relative_path"] = ""
        bioz_master["csv_exists"] = False
        bioz_master["log_exists"] = False
        bioz_master["csv_rows"] = ""
        bioz_master["csv_columns"] = ""
        bioz_master["csv_column_names"] = ""
        bioz_master["csv_read_status"] = "NATIVE_SPEC"
        bioz_master["complete_slot"] = bioz_master["two_channel_complete"]
        bioz_master["recording_unit"] = True

        master_tables.append(bioz_master)

    master_inventory = (
        pd.concat(master_tables, ignore_index=True, sort=False)
        if master_tables else pd.DataFrame()
    )

    write(master_inventory, META / "recording_inventory.csv")

    # --------------------------------------------------------
    # Recording parameters
    # --------------------------------------------------------

    recording_parameters = pd.DataFrame(
        [
            {
                "parameter": "execution_recordings",
                "value": 840,
                "unit": "recordings",
                "scope": "MOTOR_EXECUTION",
            },
            {
                "parameter": "imagery_recordings",
                "value": 525,
                "unit": "recordings",
                "scope": "MOTOR_IMAGERY",
            },
            {
                "parameter": "bioz_recordings",
                "value": 1890,
                "unit": "recording units",
                "scope": "BIOIMPEDANCE",
            },
            {
                "parameter": "bioz_native_spec_files",
                "value": 3780,
                "unit": "channel files",
                "scope": "BIOIMPEDANCE",
            },
        ]
    )

    write(recording_parameters, META / "recording_parameters.csv")

    # --------------------------------------------------------
    # Acquisition parameters
    # --------------------------------------------------------

    acquisition_parameters = pd.DataFrame(
        [
            {
                "modality": "EEG",
                "protocol": "MOTOR_EXECUTION",
                "system": "OpenBCI Cyton + Daisy",
                "channels": 13,
                "sampling_rate_hz": 125,
                "channel_mapping": "EEG_ch-01 through EEG_ch-13",
            },
            {
                "modality": "EMG",
                "protocol": "MOTOR_EXECUTION",
                "system": "OpenBCI Cyton + Daisy",
                "channels": 3,
                "sampling_rate_hz": 125,
                "channel_mapping": "Physical acquisition channels 1–3",
            },
            {
                "modality": "EEG",
                "protocol": "MOTOR_IMAGERY",
                "system": "OpenBCI Cyton + Daisy",
                "channels": 13,
                "sampling_rate_hz": 125,
                "channel_mapping": "EEG_ch-01 through EEG_ch-13",
            },
            {
                "modality": "BIOIMPEDANCE",
                "protocol": "SEPARATE_ACQUISITION",
                "system": "Sciospec ISX-5 Series",
                "channels": 2,
                "sampling_rate_hz": "",
                "channel_mapping": "Channel_1 and Channel_2",
            },
        ]
    )

    write(acquisition_parameters, META / "acquisition_parameters.csv")

    gesture_dictionary = pd.DataFrame(
        [
            {
                "gesture_code": code,
                "gesture_folder": folder,
                "gesture_label": label,
                "sets": "A;B;C",
            }
            for code, folder, label in GESTURES
        ]
    )

    write(gesture_dictionary, META / "gesture_dictionary.csv")

    # --------------------------------------------------------
    # Provenance and cohort definition
    # --------------------------------------------------------

    provenance = pd.DataFrame(
        [
            {
                "dataset_root": str(ROOT),
                "demographic_source_file": demographic_source.name,
                "raw_data_modified": "NO",
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "notes": (
                    "DATA1 inventory only. Canonical mappings are sourced "
                    "from the demographic workbook. Raw data remain unchanged."
                ),
            }
        ]
    )

    write(provenance, META / "data_provenance.csv")

    cohort_definition = pd.DataFrame(
        [
            {
                "cohort": "Motor execution",
                "participants": 40,
                "status": "retained",
                "notes": "Subject_01–Subject_40",
            },
            {
                "cohort": "Motor imagery",
                "participants": 25,
                "status": "mapped",
                "notes": "MI_01–MI_25 mapped to canonical Subject_ID",
            },
            {
                "cohort": "Bioimpedance",
                "participants": 18,
                "status": "mapped",
                "notes": "BZ_01–BZ_18 mapped to canonical Subject_ID",
            },
            {
                "cohort": "Excluded",
                "participants": 1,
                "status": "excluded",
                "notes": "Subject_41 — data quality and leakage",
            },
        ]
    )

    write(cohort_definition, META / "cohort_definition.csv")

    write(
        pd.DataFrame(
            [
                {
                    "subject_id": "Subject_41",
                    "status": "excluded",
                    "reason": "data quality, leakage",
                }
            ]
        ),
        META / "exclusion_log.csv",
    )

    # --------------------------------------------------------
    # Quality-control reports
    # --------------------------------------------------------

    write(
        execution_qc,
        QC / "motor_execution_completeness.csv",
    )
    write(
        mi_qc,
        QC / "motor_imagery_completeness.csv",
    )
    write(
        bioz_qc,
        QC / "bioimpedance_completeness.csv",
    )

    structure_rows = []

    for name, table in (
        ("MOTOR_EXECUTION", execution_inventory),
        ("MOTOR_IMAGERY", mi_inventory),
    ):
        readable = (
            table.loc[table["csv_read_status"].eq("PASS")]
            if not table.empty
            else table
        )

        structure_rows.append(
            {
                "protocol": name,
                "csv_files": (
                    int(table["csv_exists"].sum())
                    if not table.empty else 0
                ),
                "readable_csv_files": len(readable),
                "unique_row_counts": (
                    ";".join(
                        map(
                            str,
                            sorted(
                                set(readable["csv_rows"].astype(int))
                            ),
                        )
                    )
                    if len(readable) else ""
                ),
                "unique_column_counts": (
                    ";".join(
                        map(
                            str,
                            sorted(
                                set(readable["csv_columns"].astype(int))
                            ),
                        )
                    )
                    if len(readable) else ""
                ),
            }
        )

    write(
        pd.DataFrame(structure_rows),
        QC / "csv_structure_summary.csv",
    )

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    execution_csv_count = (
        int(execution_inventory["csv_exists"].sum())
        if not execution_inventory.empty else 0
    )
    execution_complete = (
        int(execution_inventory["complete_slot"].sum())
        if not execution_inventory.empty else 0
    )

    mi_subject_count = (
        mi_inventory["subject_id"].nunique()
        if not mi_inventory.empty else 0
    )
    mi_csv_count = (
        int(mi_inventory["csv_exists"].sum())
        if not mi_inventory.empty else 0
    )
    mi_complete = (
        int(mi_inventory["complete_slot"].sum())
        if not mi_inventory.empty else 0
    )

    bioz_subject_count = (
        bioz_inventory["subject_id"].nunique()
        if not bioz_inventory.empty else 0
    )
    bioz_units = len(bioz_inventory)
    bioz_spec_count = (
        int(bioz_inventory["native_spec_file_count"].sum())
        if not bioz_inventory.empty else 0
    )
    bioz_complete_units = (
        int(bioz_inventory["two_channel_complete"].sum())
        if not bioz_inventory.empty else 0
    )

    summary_rows = [
        (
            "MOTOR_EXECUTION",
            "subjects",
            execution_inventory["subject_id"].nunique(),
            40,
        ),
        ("MOTOR_EXECUTION", "csv_recordings", execution_csv_count, 840),
        (
            "MOTOR_EXECUTION",
            "complete_csv_log_slots",
            execution_complete,
            840,
        ),
        ("MOTOR_IMAGERY", "subjects", mi_subject_count, 25),
        ("MOTOR_IMAGERY", "csv_recordings", mi_csv_count, 525),
        ("MOTOR_IMAGERY", "complete_csv_log_slots", mi_complete, 525),
        ("BIOIMPEDANCE", "participants", bioz_subject_count, 18),
        ("BIOIMPEDANCE", "recording_units", bioz_units, 1890),
        ("BIOIMPEDANCE", "native_spec_files", bioz_spec_count, 3780),
        (
            "BIOIMPEDANCE",
            "two_channel_recording_units",
            bioz_complete_units,
            1890,
        ),
    ]

    summary = pd.DataFrame(
        [
            {
                "component": component,
                "metric": metric,
                "actual": actual,
                "expected": expected,
                "status": (
                    "PASS" if actual == expected else "CHECK_REQUIRED"
                ),
            }
            for component, metric, actual, expected in summary_rows
        ]
    )

    write(summary, QC / "dataset_inventory_summary.csv")

    # --------------------------------------------------------
    # Demographic merge QC
    # --------------------------------------------------------

    retained_count = int(
        participants["cohort_status"].eq("retained").sum()
    )
    excluded_count = int(
        participants["cohort_status"].eq("excluded").sum()
    )

    demographic_qc = pd.DataFrame(
        [
            {
                "metric": "demographic_source_found",
                "actual": True,
                "expected": True,
                "status": "PASS",
                "source": str(demographic_source),
            },
            {
                "metric": "demographic_rows",
                "actual": len(participants),
                "expected": 41,
                "status": (
                    "PASS" if len(participants) == 41
                    else "CHECK_REQUIRED"
                ),
                "source": demographic_source.name,
            },
            {
                "metric": "retained_participants",
                "actual": retained_count,
                "expected": 40,
                "status": (
                    "PASS" if retained_count == 40
                    else "CHECK_REQUIRED"
                ),
                "source": demographic_source.name,
            },
            {
                "metric": "excluded_participants",
                "actual": excluded_count,
                "expected": 1,
                "status": (
                    "PASS" if excluded_count == 1
                    else "CHECK_REQUIRED"
                ),
                "source": demographic_source.name,
            },
        ]
    )

    write(demographic_qc, QC / "demographic_merge_QC.csv")

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    mi_mapping_count = int(participants["mi_id"].ne("").sum())
    bioz_mapping_count = int(participants["bioz_id"].ne("").sum())

    mi_mapped_folders = (
        int(
            mi_inventory["canonical_subject_id"]
            .astype(str)
            .str.startswith("Subject_")
            .sum()
        )
        if not mi_inventory.empty else 0
    )

    bioz_mapped_folders = (
        int(
            bioz_inventory["canonical_subject_id"]
            .astype(str)
            .str.startswith("Subject_")
            .sum()
        )
        if not bioz_inventory.empty else 0
    )

    L.info("=" * 80)
    L.info("DATA1 FINAL SUMMARY")
    L.info("=" * 80)

    L.info(
        "Execution CSV files: %d / 840",
        execution_csv_count,
    )
    L.info(
        "Execution complete slots: %d / 840",
        execution_complete,
    )
    L.info("MI folders discovered: %d / 25", mi_subject_count)
    L.info("MI CSV files: %d / 525", mi_csv_count)
    L.info("MI complete slots: %d / 525", mi_complete)
    L.info("BioZ folders discovered: %d / 18", bioz_subject_count)
    L.info("BioZ recording units: %d / 1890", bioz_units)
    L.info("BioZ native .spec files: %d / 3780", bioz_spec_count)
    L.info(
        "BioZ complete two-channel units: %d / 1890",
        bioz_complete_units,
    )
    L.info("Demographic rows: %d / 41", len(participants))
    L.info("Retained participants: %d / 40", retained_count)
    L.info("Excluded participants: %d / 1", excluded_count)
    L.info("MI mappings in workbook: %d / 25", mi_mapping_count)
    L.info("BioZ mappings in workbook: %d / 18", bioz_mapping_count)
    L.info("MI inventory rows mapped: %d", mi_mapped_folders)
    L.info("BioZ inventory rows mapped: %d", bioz_mapped_folders)
    L.info("RAW DATA MODIFICATION: NONE")

    demographic_pass = (
        len(participants) == 41
        and retained_count == 40
        and excluded_count == 1
    )

    all_pass = (
        execution_csv_count == 840
        and execution_complete == 840
        and mi_subject_count == 25
        and mi_csv_count == 525
        and mi_complete == 525
        and bioz_subject_count == 18
        and bioz_units == 1890
        and bioz_spec_count == 3780
        and bioz_complete_units == bioz_units
        and demographic_pass
        and mi_mapping_count == 25
        and bioz_mapping_count == 18
        and mi_mapped_folders == 525
        and bioz_mapped_folders == bioz_units
    )

    L.info(
        "DATA1 INVENTORY STATUS: %s",
        "PASS" if all_pass else "CHECK_REQUIRED",
    )
    L.info("Metadata directory: %s", META)
    L.info("QC directory: %s", QC)
    L.info("Runtime: %s", datetime.now() - started)

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
