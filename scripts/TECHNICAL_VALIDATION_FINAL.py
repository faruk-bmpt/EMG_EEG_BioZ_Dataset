#!/usr/bin/env python3
"""
TECHNICAL_VALIDATION_FINAL.py
==============================

Technical validation of the frozen Dataset V1.0 (EEG / EMG / Bioimpedance)
for a Scientific Data "Technical Validation" section.

The script is READ-ONLY with respect to the dataset. Every output is written
to a separate validation folder (default: <dataset_root>/../TECHNICAL_VALIDATION_FINAL).

Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scipy
import sklearn
from scipy.signal import welch
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Silence metric warnings for zero-division in unused classes gracefully
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

_trapz = getattr(np, "trapezoid", None) or np.trapz

# ============================================================================
# 1. FROZEN CONTRACT / CONSTANTS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WINDOWS_ROOT = Path(
    r"F:\Faruk\OFS_Paper_Work\Data_Set_Paper_Work\EEG_EMG_BIOZ_DATASET"
)

FS = 125.0
N_CH_EEG = 13
N_CH_EMG = 3
N_SAMPLES = 5625
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(WINDOW_SEC * FS)                 # 625 samples
OVERLAP = 0.50
STRIDE = int(WINDOW_SAMPLES * (1.0 - OVERLAP))         # 312 samples
N_SPLITS = 5
N_TOP_FEATURES = 0  # 0 = use all candidate features
RANDOM_STATE = 20261007
EPS = 1e-30

EXPECTED = {
    "EEG_MOTOR_EXECUTION": 840,
    "EEG_MOTOR_IMAGERY": 525,
    "EMG_MOTOR_EXECUTION": 840,
    "BIOZ_CHANNEL_FILES": 3780,
    "BIOZ_UNITS": 1890,
}

GESTURE_ORDER = ["HOC", "TFM", "IFM", "MFM", "RFM", "LFM", "PH"]
GESTURE_NAMES = {
    "HOC": "Hand Open-Close",
    "TFM": "Thumb Finger Movement",
    "IFM": "Index Finger Movement",
    "MFM": "Middle Finger Movement",
    "RFM": "Ring Finger Movement",
    "LFM": "Little Finger Movement",
    "PH": "Pen Holding",
}
GESTURE_TO_INT = {g: i + 1 for i, g in enumerate(GESTURE_ORDER)}
LABELS = list(range(1, 8))
CHANCE = 1.0 / len(LABELS)
GESTURE_DIR_ALIASES = {
    "hand_open_close": "HOC",
    "thumb_finger_movement": "TFM",
    "index_finger_movement": "IFM",
    "middle_finger_movement": "MFM",
    "ring_finger_movement": "RFM",
    "little_finger_movement": "LFM",
    "pen_holding": "PH",
}

BIOZ_CROSSWALK = {f"BZ_{i:02d}": f"Subject_{i:02d}" for i in range(1, 16)}
BIOZ_CROSSWALK.update({
    "BZ_16": "Subject_17",
    "BZ_17": "Subject_29",
    "BZ_18": "Subject_39",
})

EEG_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

MODELS = ["LDA", "Linear SVM", "RBF SVM", "Random Forest"]

COHORTS = [
    "EEG_MOTOR_EXECUTION",
    "EEG_MOTOR_IMAGERY",
    "EMG_MOTOR_EXECUTION",
    "BIOIMPEDANCE",
]
COHORT_LABEL = {
    "EEG_MOTOR_EXECUTION": "EEG motor execution",
    "EEG_MOTOR_IMAGERY": "EEG motor imagery",
    "EMG_MOTOR_EXECUTION": "EMG motor execution",
    "BIOIMPEDANCE": "Bioimpedance",
}

PROTO_A = "A_cross_participant"
PROTO_C = "C_within_participant_cross_set"
PROTO_LABEL = {
    PROTO_A: "Cross-participant (GroupKFold)",
    PROTO_C: "Within-participant (leave-one-set-out)",
}
CURRENT_PROTOCOL = "DESCRIPTIVE"

FIG_INDEX: list[dict] = []
TAB_INDEX: list[dict] = []


class Paths:
    def __init__(self, dataset_root: Path, out_root: Path):
        self.dataset = dataset_root
        self.processed = dataset_root / "06_PROCESSED_DATA"
        self.eeg_me = self.processed / "EEG_MOTOR_EXECUTION"
        self.eeg_mi = self.processed / "EEG_MOTOR_IMAGERY"
        self.emg = self.processed / "EMG"
        self.bioz = self.processed / "BIOIMPEDANCE"
        self.out = out_root
        self.tables = out_root / "TABLES"
        self.fig_root = out_root / "FIGURES"
        self.fig_dataset = self.fig_root / "01_DATASET"
        self.fig_eeg = self.fig_root / "02_EEG"
        self.fig_emg = self.fig_root / "03_EMG"
        self.fig_bioz = self.fig_root / "04_BIOIMPEDANCE"
        self.fig_cls = self.fig_root / "05_CLASSIFICATION"


# ============================================================================
# 2. UTILITIES
# ============================================================================

def log(msg: str = "") -> None:
    print(msg, flush=True)


def finite_array(x) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def posix_rel(p: Path, base: Path) -> str:
    return "/".join(p.relative_to(base).parts)


def parse_gesture(text: str) -> str | None:
    tokens = [t.upper() for t in re.split(r"[^A-Za-z0-9]+", text) if t]
    hits = [g for g in GESTURE_ORDER if g in tokens]
    if len(hits) == 1:
        return hits[0]
    low = re.sub(r"[^a-z0-9]+", "_", text.lower())
    found = {v for k, v in GESTURE_DIR_ALIASES.items() if k in low}
    if len(found) == 1:
        return found.pop()
    return None


def parse_set_number(text: str) -> int | None:
    m = re.search(r"set[_\-]?(\d{1,2})(?![A-Za-z0-9])", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"set[_\-]?([ABC])(?![A-Za-z])", text, re.I)
    if m:
        return {"A": 1, "B": 2, "C": 3}[m.group(1).upper()]
    return None


def parse_subject(text: str) -> str | None:
    m = re.search(r"Subject[_\-]?(\d{1,3})", text, re.I)
    return f"Subject_{int(m.group(1)):02d}" if m else None


def parse_mi(text: str) -> str | None:
    m = re.search(r"(?<![A-Za-z])MI[_\-]?(\d{1,3})", text, re.I)
    return f"MI_{int(m.group(1)):02d}" if m else None


def dir_snapshot(root: Path) -> dict:
    lines = []
    total = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in ("scripts", "__pycache__") for part in p.relative_to(root).parts):
            continue
        st = p.stat()
        total += st.st_size
        lines.append(f"{posix_rel(p, root)}|{st.st_size}|{st.st_mtime_ns}")
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return {"n_files": len(lines), "total_bytes": total, "sha256": digest}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(out) + "\n"


def save_table(df: pd.DataFrame, P: Paths, name: str, caption: str, markdown: bool = True) -> None:
    df.to_csv(P.tables / f"{name}.csv", index=False)
    if markdown and len(df) <= 400:
        (P.tables / f"{name}.md").write_text(
            f"**{name}** - {caption}\n\n" + df_to_markdown(df), encoding="utf-8"
        )
    TAB_INDEX.append({"protocol": CURRENT_PROTOCOL, "table": name, "rows": len(df), "description": caption})


def save_fig(fig, folder: Path, name: str, caption: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    png_path = str((folder / f"{name}.png").resolve())
    pdf_path = str((folder / f"{name}.pdf").resolve())
    
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    FIG_INDEX.append({"protocol": CURRENT_PROTOCOL, "figure": name, "folder": folder.name, "description": caption})


def set_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })


# ============================================================================
# 3. CONCISE BASELINE FEATURE LIBRARIES (Standard Benchmark Sets)
# ============================================================================

def psd_cf(x: np.ndarray, nperseg: int = 256):
    x = finite_array(x)
    nperseg = min(x.shape[-1], nperseg)
    return welch(x, fs=FS, nperseg=nperseg, noverlap=nperseg // 2,
                 detrend="constant", scaling="density", axis=-1)


def band_power(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (f >= lo) & (f < hi)
    if m.sum() < 2:
        return np.zeros(p.shape[:-1])
    return _trapz(p[..., m], f[m], axis=-1)


def eeg_feature_names(n_ch: int = N_CH_EEG) -> list[str]:
    names = []
    for b in EEG_BANDS:
        names += [f"eeg_log_{b}_power_ch{c + 1:02d}" for c in range(n_ch)]
    names += [f"eeg_log_rms_ch{c + 1:02d}" for c in range(n_ch)]
    return names


def eeg_window_vec(w: np.ndarray) -> np.ndarray:
    """Concise EEG Features: 5-band PSD log-power + Channel Log-RMS."""
    f, p = psd_cf(w)
    bp = np.stack([band_power(f, p, lo, hi) for lo, hi in EEG_BANDS.values()])
    logbp = np.log10(bp + EPS)
    lrms = np.log10(np.sqrt(np.mean(w ** 2, axis=1)) + EPS)
    return np.concatenate([logbp.ravel(), lrms])


EMG_METRICS = ["log_rms", "log_mav", "log_wl", "zc", "ssc"]


def emg_feature_names(n_ch: int = N_CH_EMG) -> list[str]:
    return [f"emg_{m}_ch{c + 1}" for m in EMG_METRICS for c in range(n_ch)]


def emg_window_vec(w: np.ndarray) -> np.ndarray:
    """Concise EMG Features: Hudgins / Phinyomark standard baseline set."""
    dx = np.diff(w, axis=-1)
    log_rms = np.log10(np.sqrt(np.mean(w ** 2, axis=1)) + EPS)
    log_mav = np.log10(np.mean(np.abs(w), axis=1) + EPS)
    log_wl = np.log10(np.sum(np.abs(dx), axis=1) + EPS)
    zc = np.sum(w[:, :-1] * w[:, 1:] < 0, axis=1).astype(float)
    ssc = np.sum(dx[:, :-1] * dx[:, 1:] < 0, axis=1).astype(float)
    return np.concatenate([log_rms, log_mav, log_wl, zc, ssc])


BIOZ_CH_METRICS = ["mean_log_z", "sd_log_z", "mean_phase_deg", "sd_phase_deg"]


def bioz_feature_names() -> list[str]:
    names = [f"bioz_{m}_ch{c}" for c in (1, 2) for m in BIOZ_CH_METRICS]
    return names + ["bioz_mean_log_ratio_ch1_ch2", "bioz_spectrum_corr"]


def bioz_pair_vec(c1: dict, c2: dict) -> np.ndarray:
    """Concise Bioimpedance features: magnitude/phase stats per channel + coupling."""
    n = min(len(c1["frequency_hz"]), len(c2["frequency_hz"]))
    z1, z2 = np.maximum(c1["z_magnitude"][:n], EPS), np.maximum(c2["z_magnitude"][:n], EPS)
    l1, l2 = np.log10(z1), np.log10(z2)
    ph1, ph2 = c1["z_phase_deg"][:n], c2["z_phase_deg"][:n]

    ch1_stats = [np.mean(l1), np.std(l1), np.mean(ph1), np.std(ph1)]
    ch2_stats = [np.mean(l2), np.std(l2), np.mean(ph2), np.std(ph2)]

    corr = float(np.corrcoef(l1, l2)[0, 1]) if n > 2 and l1.std() > 0 and l2.std() > 0 else 0.0
    pair = [float(np.mean(l1 - l2)), corr]
    return np.asarray(ch1_stats + ch2_stats + pair, dtype=np.float64)


# ============================================================================
# 4. DYNAMIC WINDOWING & DISCOVERY
# ============================================================================

def make_windows(x_cf: np.ndarray) -> np.ndarray:
    n = x_cf.shape[1]
    starts = range(0, n - WINDOW_SAMPLES + 1, STRIDE)
    wins = [x_cf[:, s:s + WINDOW_SAMPLES] for s in starts]
    return np.stack(wins) if wins else np.empty((0, x_cf.shape[0], WINDOW_SAMPLES))


def discover_eeg(P: Paths, root: Path, cohort: str) -> list[dict]:
    recs = []
    for p in sorted(root.rglob("*_PRE1.npz")):
        rel = posix_rel(p, P.processed)
        gid = parse_subject(rel) if cohort == "EEG_MOTOR_EXECUTION" else parse_mi(rel)
        g = parse_gesture(rel)
        if gid is None or g is None:
            continue
        recs.append({
            "path": p, "cohort": cohort, "group": gid, "gesture": g,
            "y": GESTURE_TO_INT[g], "set": parse_set_number(p.name),
            "recording_id": posix_rel(p, P.dataset)
        })
    return recs


def discover_emg(P: Paths) -> list[dict]:
    recs = []
    for p in sorted(P.emg.rglob("*_PRE2.npz")):
        rel = posix_rel(p, P.processed)
        sid, g = parse_subject(rel), parse_gesture(rel)
        if sid is None or g is None:
            continue
        recs.append({
            "path": p, "cohort": "EMG_MOTOR_EXECUTION", "group": sid,
            "gesture": g, "y": GESTURE_TO_INT[g],
            "set": parse_set_number(p.name),
            "recording_id": posix_rel(p, P.dataset)
        })
    return recs


def discover_bioz(P: Paths):
    ch1, ch2 = {}, {}
    for p in sorted(P.bioz.rglob("*.npz")):
        parts = p.relative_to(P.bioz).parts
        for tag, store in (("Channel_1", ch1), ("Channel_2", ch2)):
            if tag in parts:
                i = parts.index(tag)
                store[parts[:i] + parts[i + 1:]] = p
    paired = sorted(set(ch1) & set(ch2))
    counts = {
        "n_channel_1_files": len(ch1),
        "n_channel_2_files": len(ch2),
        "n_unpaired_files": len(set(ch1) ^ set(ch2))
    }
    units = []
    for key in paired:
        bz_id = key[0]
        g = None
        for part in key[1:-1]:
            if part.startswith(bz_id + "_"):
                g = parse_gesture(part[len(bz_id) + 1:])
        if g is None:
            continue
        units.append({
            "path_ch1": ch1[key], "path_ch2": ch2[key], "bz_id": bz_id,
            "group": BIOZ_CROSSWALK[bz_id], "gesture": g, "y": GESTURE_TO_INT[g],
            "set": parse_set_number(key[-1]),
            "recording_id": posix_rel(ch1[key], P.dataset),
        })
    return units, counts


def validate_inventory(eeg_me, eeg_mi, emg, units, counts, strict: bool):
    rows = [
        ("EEG_MOTOR_EXECUTION", len(eeg_me), EXPECTED["EEG_MOTOR_EXECUTION"]),
        ("EEG_MOTOR_IMAGERY", len(eeg_mi), EXPECTED["EEG_MOTOR_IMAGERY"]),
        ("EMG_MOTOR_EXECUTION", len(emg), EXPECTED["EMG_MOTOR_EXECUTION"]),
        ("BIOZ_CHANNEL_FILES", counts["n_channel_1_files"] + counts["n_channel_2_files"], EXPECTED["BIOZ_CHANNEL_FILES"]),
        ("BIOZ_UNITS", len(units), EXPECTED["BIOZ_UNITS"]),
        ("BIOZ_UNPAIRED_FILES", counts["n_unpaired_files"], 0),
    ]
    df = pd.DataFrame([
        {"item": n, "expected": e, "actual": a, "status": "PASS" if a == e else "FAIL"}
        for n, a, e in rows
    ])
    if strict and not (df["status"] == "PASS").all():
        raise RuntimeError("Frozen inventory mismatch:\n" + df.to_string(index=False))
    return df


# ============================================================================
# 5. DATA LOADERS & PROCESSING
# ============================================================================

def load_eeg(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z["eeg"], dtype=np.float64)
    if x.shape == (N_SAMPLES, N_CH_EEG):
        x = x.T
    return x


def load_emg(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z["emg_preprocessed"], dtype=np.float64)
    if x.shape == (N_SAMPLES, N_CH_EMG):
        x = x.T
    return x


BIOZ_KEYS = ("frequency_hz", "re", "im", "z_magnitude", "z_phase_deg")


def load_bioz_channel(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k], dtype=np.float64) for k in BIOZ_KEYS}


def recording_quality(x: np.ndarray, kind: str) -> dict:
    xf = finite_array(x)
    sd = xf.std(axis=1)
    f, p = psd_cf(xf)
    rms_ch = np.sqrt(np.mean(xf ** 2, axis=1))
    return {
        "n_nonfinite": int((~np.isfinite(x)).sum()),
        "rms_mean": float(rms_ch.mean()),
        "sd_mean": float(sd.mean()),
        "abs_max": float(np.abs(xf).max()),
    }, f, p


def process_window_cohort(records: list[dict], cohort: str):
    is_eeg = cohort.startswith("EEG")
    loader = load_eeg if is_eeg else load_emg
    vec = eeg_window_vec if is_eeg else emg_window_vec
    names = eeg_feature_names() if is_eeg else emg_feature_names()
    X, rows, qc_rows, psd_all = [], [], [], []
    freq = None
    for i, r in enumerate(records, 1):
        x = loader(r["path"])
        q, f, p = recording_quality(x, "EEG" if is_eeg else "EMG")
        freq = f
        qc_rows.append({
            "cohort": cohort, "recording_id": r["recording_id"],
            "group": r["group"], "gesture": r["gesture"],
            "set": r["set"], **q
        })
        psd_all.append(p.mean(axis=0) if is_eeg else p)
        for wi, w in enumerate(make_windows(finite_array(x))):
            X.append(vec(w))
            rows.append({
                "observation_id": f"{r['recording_id']}::W{wi + 1:02d}",
                "recording_id": r["recording_id"], "group": r["group"],
                "gesture": r["gesture"], "y": r["y"],
                "window_index": wi + 1, "set": r["set"]
            })
        if i % 200 == 0 or i == len(records):
            log(f"  {cohort}: {i}/{len(records)} recordings processed")
    X = np.asarray(X, dtype=np.float64)
    qc = pd.DataFrame(qc_rows)
    spectra = {"f": freq, "psd": np.asarray(psd_all)}
    return X, pd.DataFrame(rows), names, qc, spectra


def process_bioz(units: list[dict]):
    names = bioz_feature_names()
    X, rows, qc_rows = [], [], []
    ref_f = None
    spec = {"mag1": [], "f": None}
    for i, u in enumerate(units, 1):
        c1, c2 = load_bioz_channel(u["path_ch1"]), load_bioz_channel(u["path_ch2"])
        if ref_f is None:
            ref_f = c1["frequency_hz"].copy()
            spec["f"] = ref_f
        c1f = {k: finite_array(v) for k, v in c1.items()}
        c2f = {k: finite_array(v) for k, v in c2.items()}
        X.append(bioz_pair_vec(c1f, c2f))
        rows.append({
            "observation_id": u["recording_id"], "recording_id": u["recording_id"],
            "group": u["group"], "gesture": u["gesture"], "y": u["y"],
            "window_index": 0, "set": u["set"], "bz_id": u["bz_id"]
        })
        qc_rows.append({
            "cohort": "BIOIMPEDANCE", "recording_id": u["recording_id"],
            "group": u["group"], "gesture": u["gesture"], "set": u["set"],
        })
        spec["mag1"].append(c1f["z_magnitude"])
        if i % 300 == 0 or i == len(units):
            log(f"  BIOIMPEDANCE: {i}/{len(units)} units processed")
    spec["mag1"] = np.asarray(spec["mag1"])
    return np.asarray(X, dtype=np.float64), pd.DataFrame(rows), names, pd.DataFrame(qc_rows), spec


# ============================================================================
# 6. SUBJECT FEATURE ALIGNMENT (Subject Normalization)
# ============================================================================

def apply_subject_normalization(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Normalizes features per subject to reduce inter-subject variability."""
    X_norm = np.copy(X)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        mean = np.mean(X[idx], axis=0)
        std = np.std(X[idx], axis=0) + 1e-6
        X_norm[idx] = (X[idx] - mean) / std
    return X_norm


# ============================================================================
# 7. CROSS-VALIDATION PIPELINE
# ============================================================================

def make_models(n_jobs: int):
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "LDA": Pipeline([
            ("imp", imp()),
            ("sc", StandardScaler()),
            ("m", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        "Linear SVM": Pipeline([
            ("imp", imp()),
            ("sc", StandardScaler()),
            ("m", SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)),
        ]),
        "RBF SVM": Pipeline([
            ("imp", imp()),
            ("sc", StandardScaler()),
            ("m", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("imp", imp()),
            ("m", RandomForestClassifier(
                n_estimators=100,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
            )),
        ]),
    }


def _fit_predict_selected(pipe: Pipeline, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, top_k: int):
    if top_k > 0 and top_k < X_tr.shape[1]:
        f_vals, _ = f_classif(X_tr, y_tr)
        f_vals = np.nan_to_num(f_vals, nan=0.0)
        idx = np.argsort(f_vals)[::-1][:top_k]
        X_tr_s, X_te_s = X_tr[:, idx], X_te[:, idx]
    else:
        idx = np.arange(X_tr.shape[1])
        X_tr_s, X_te_s = X_tr, X_te

    clf = clone(pipe)
    clf.fit(X_tr_s, y_tr)
    preds = clf.predict(X_te_s)
    return preds, idx


def run_protocol_a(X: np.ndarray, df_rows: pd.DataFrame, feature_names: list[str], cohort: str, top_k: int, n_jobs: int):
    y = df_rows["y"].to_numpy()
    groups = df_rows["group"].to_numpy()
    recs = df_rows["recording_id"].to_numpy()

    # Normalize per-subject to mitigate subject shift
    X_proc = apply_subject_normalization(X, groups)

    gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
    fold_metrics, cm_dict = [], {m: np.zeros((len(LABELS), len(LABELS)), dtype=int) for m in MODELS}
    feature_selection_history, rec_level_preds = [], []

    models = make_models(n_jobs)

    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X_proc, y, groups), 1):
        X_tr, y_tr = X_proc[tr_idx], y[tr_idx]
        X_te, y_te = X_proc[te_idx], y[te_idx]
        recs_te, groups_te = recs[te_idx], groups[te_idx]

        for m_name, pipe in models.items():
            preds, selected_idx = _fit_predict_selected(pipe, X_tr, y_tr, X_te, top_k)

            acc = accuracy_score(y_te, preds)
            bal_acc = balanced_accuracy_score(y_te, preds)
            f1_macro = f1_score(y_te, preds, average="macro", zero_division=0)
            cm_dict[m_name] += confusion_matrix(y_te, preds, labels=LABELS)

            fold_metrics.append({
                "protocol": PROTO_A, "cohort": cohort, "fold": fold_idx, "model": m_name,
                "eval_level": "window", "accuracy": acc, "balanced_accuracy": bal_acc,
                "f1_macro": f1_macro, "n_test_samples": len(y_te),
            })

            # Majority voting per recording
            unique_recs = np.unique(recs_te)
            rec_true, rec_pred = [], []
            for r_id in unique_recs:
                mask = recs_te == r_id
                r_y = y_te[mask][0]
                r_pred_maj = int(np.argmax(np.bincount(preds[mask], minlength=len(LABELS) + 1)))
                rec_true.append(r_y)
                rec_pred.append(r_pred_maj)

            fold_metrics.append({
                "protocol": PROTO_A, "cohort": cohort, "fold": fold_idx, "model": m_name,
                "eval_level": "recording_majority_vote",
                "accuracy": accuracy_score(rec_true, rec_pred),
                "balanced_accuracy": balanced_accuracy_score(rec_true, rec_pred),
                "f1_macro": f1_score(rec_true, rec_pred, average="macro", zero_division=0),
                "n_test_samples": len(unique_recs),
            })

    return pd.DataFrame(fold_metrics), pd.DataFrame(feature_selection_history), pd.DataFrame(rec_level_preds), cm_dict


def run_protocol_c(X: np.ndarray, df_rows: pd.DataFrame, feature_names: list[str], cohort: str, top_k: int, n_jobs: int):
    y = df_rows["y"].to_numpy()
    groups = df_rows["group"].to_numpy()
    sets = df_rows["set"].to_numpy()
    recs = df_rows["recording_id"].to_numpy()

    fold_metrics, cm_dict = [], {m: np.zeros((len(LABELS), len(LABELS)), dtype=int) for m in MODELS}
    models = make_models(n_jobs)

    for g_id in np.unique(groups):
        g_mask = groups == g_id
        X_g, y_g, sets_g, recs_g = X[g_mask], y[g_mask], sets[g_mask], recs[g_mask]
        unique_sets = np.unique(sets_g)
        if len(unique_sets) < 2:
            continue

        for test_set in unique_sets:
            tr_idx, te_idx = sets_g != test_set, sets_g == test_set
            X_tr, y_tr = X_g[tr_idx], y_g[tr_idx]
            X_te, y_te = X_g[te_idx], y_g[te_idx]
            recs_te = recs_g[te_idx]

            for m_name, pipe in models.items():
                preds, _ = _fit_predict_selected(pipe, X_tr, y_tr, X_te, top_k)

                fold_metrics.append({
                    "protocol": PROTO_C, "cohort": cohort, "group": g_id, "test_set": test_set,
                    "model": m_name, "eval_level": "window",
                    "accuracy": accuracy_score(y_te, preds),
                    "balanced_accuracy": balanced_accuracy_score(y_te, preds),
                    "f1_macro": f1_score(y_te, preds, average="macro", zero_division=0),
                    "n_test_samples": len(y_te),
                })

    return pd.DataFrame(fold_metrics), pd.DataFrame(), pd.DataFrame(), cm_dict


# ============================================================================
# 8. FIGURES & MAIN ENTRY POINT
# ============================================================================

def plot_grand_average_psd(spectra_dict: dict, P: Paths, cohort: str) -> None:
    f, psd = spectra_dict["f"], spectra_dict["psd"]
    psd_mean_ch = psd.mean(axis=1) if psd.ndim == 3 else psd

    fig, ax = plt.subplots(figsize=(7, 4.5))
    med = np.median(psd_mean_ch, axis=0)
    ax.semilogy(f, med, color="#1f77b4", lw=1.8, label="Median PSD")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title(f"Grand-Average PSD — {COHORT_LABEL[cohort]}")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    folder = P.fig_eeg if cohort.startswith("EEG") else P.fig_emg
    save_fig(fig, folder, f"{cohort}_grand_average_psd", f"Grand-average PSD for {cohort}.")


def plot_classification_summary(df_all_folds: pd.DataFrame, P: Paths) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for idx, proto in enumerate([PROTO_A, PROTO_C]):
        ax = axes[idx]
        sub_df = df_all_folds[(df_all_folds["protocol"] == proto) & (df_all_folds["eval_level"] == "window")]
        if sub_df.empty:
            continue

        summary = sub_df.groupby(["cohort", "model"])["f1_macro"].agg(["mean", "std"]).reset_index()
        cohorts = [c for c in COHORTS if c in summary["cohort"].unique()]
        x = np.arange(len(cohorts))
        width = 0.18

        for m_idx, m_name in enumerate(MODELS):
            m_data = summary[summary["model"] == m_name]
            means = [m_data[m_data["cohort"] == c]["mean"].values[0] if c in m_data["cohort"].values else 0 for c in cohorts]
            stds = [m_data[m_data["cohort"] == c]["std"].values[0] if c in m_data["cohort"].values else 0 for c in cohorts]
            ax.bar(x + (m_idx - 1.5) * width, means, width, yerr=stds, label=m_name, capsize=3)

        ax.axhline(CHANCE, color="red", ls="--", lw=1.2, label=f"Chance ({CHANCE:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels([COHORT_LABEL[c] for c in cohorts], rotation=15, ha="right")
        ax.set_ylabel("Macro F1-Score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Protocol: {PROTO_LABEL[proto]}")
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    save_fig(fig, P.fig_cls, "classification_performance_benchmark", "Macro F1-score across protocols.")


def main():
    parser = argparse.ArgumentParser(description="Technical validation of Dataset V1.0 (EEG/EMG/BioZ)")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_WINDOWS_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=N_TOP_FEATURES)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--strict-inventory", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_root = args.out.resolve() if args.out else (dataset_root.parent / "TECHNICAL_VALIDATION_FINAL").resolve()
    P = Paths(dataset_root, out_root)

    for folder in [P.out, P.tables, P.fig_root, P.fig_dataset, P.fig_eeg, P.fig_emg, P.fig_bioz, P.fig_cls]:
        folder.mkdir(parents=True, exist_ok=True)

    set_style()
    log("=================================================================")
    log(" TECHNICAL VALIDATION PIPELINE — SCIENTIFIC DATA PUBLICATION")
    log(f" Dataset Root: {P.dataset}")
    log(f" Output Root:  {P.out}")
    log("=================================================================")

    snap_before = dir_snapshot(P.dataset)

    eeg_me_recs = discover_eeg(P, P.eeg_me, "EEG_MOTOR_EXECUTION")
    eeg_mi_recs = discover_eeg(P, P.eeg_mi, "EEG_MOTOR_IMAGERY")
    emg_recs = discover_emg(P)
    bioz_units, bioz_counts = discover_bioz(P)

    df_inv = validate_inventory(eeg_me_recs, eeg_mi_recs, emg_recs, bioz_units, bioz_counts, strict=args.strict_inventory)
    save_table(df_inv, P, "01_dataset_inventory_check", "Frozen dataset file inventory audit.")

    X_eeg_me, rows_eeg_me, names_eeg, qc_eeg_me, spec_eeg_me = process_window_cohort(eeg_me_recs, "EEG_MOTOR_EXECUTION")
    X_eeg_mi, rows_eeg_mi, _, qc_eeg_mi, spec_eeg_mi = process_window_cohort(eeg_mi_recs, "EEG_MOTOR_IMAGERY")
    X_emg, rows_emg, names_emg, qc_emg, spec_emg = process_window_cohort(emg_recs, "EMG_MOTOR_EXECUTION")
    X_bioz, rows_bioz, names_bioz, qc_bioz, spec_bioz = process_bioz(bioz_units)

    qc_all = pd.concat([qc_eeg_me, qc_eeg_mi, qc_emg, qc_bioz], ignore_index=True)
    save_table(qc_all, P, "02_quality_metrics_per_recording", "Per-recording signal quality descriptors.")

    plot_grand_average_psd(spec_eeg_me, P, "EEG_MOTOR_EXECUTION")
    plot_grand_average_psd(spec_eeg_mi, P, "EEG_MOTOR_IMAGERY")
    plot_grand_average_psd(spec_emg, P, "EMG_MOTOR_EXECUTION")

    all_folds = []
    cohort_data = [
        ("EEG_MOTOR_EXECUTION", X_eeg_me, rows_eeg_me, names_eeg),
        ("EEG_MOTOR_IMAGERY", X_eeg_mi, rows_eeg_mi, names_eeg),
        ("EMG_MOTOR_EXECUTION", X_emg, rows_emg, names_emg),
        ("BIOIMPEDANCE", X_bioz, rows_bioz, names_bioz),
    ]

    for c_name, X_c, rows_c, names_c in cohort_data:
        log(f"  Executing benchmarks for: {c_name}")
        df_f_a, _, _, _ = run_protocol_a(X_c, rows_c, names_c, c_name, args.top_k, args.n_jobs)
        df_f_c, _, _, _ = run_protocol_c(X_c, rows_c, names_c, c_name, args.top_k, args.n_jobs)
        all_folds.extend([df_f_a, df_f_c])

    df_all_folds = pd.concat(all_folds, ignore_index=True)
    save_table(df_all_folds, P, "03_classification_fold_metrics", "Classification performance summary.")
    plot_classification_summary(df_all_folds, P)

    snap_after = dir_snapshot(P.dataset)
    unchanged = snap_before["sha256"] == snap_after["sha256"]
    assert unchanged, "CRITICAL ERROR: Dataset files were modified during execution!"

    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "dataset_unchanged": unchanged,
    }
    with open(P.out / "VALIDATION_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log("\n=================================================================")
    log(" TECHNICAL VALIDATION COMPLETE SUCCESSFULLY")
    log("=================================================================")


if __name__ == "__main__":
    main()
