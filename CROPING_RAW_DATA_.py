
update croping subject 01-40.py
100%
import os
import pandas as pd
from typing import Dict, Tuple

# --- 1. PARAMETERS ---
fs = 125
skip_seconds = 15
keep_seconds = 45  
tolerance = 0.5    

# --- 2. SUBJECT + PATHS BASE TEMPLATES ---
BASE_INPUT_ROOT = r"F:\MS BMPT DU\BMPT-513\BMPT-513\Open BCI Data\OpenBCI Perfect Raw Data"
BASE_OUTPUT = r"F:\BCI Research\Raw_Data"

# Updated to process all subjects from 01 to 40
SUBJECT_IDS = [str(i).zfill(2) for i in range(1, 40)]

# --- 3. MAPPING DICTIONARIES ---
gesture_map: Dict[str, Tuple[str, str]] = {
    "index finger": ("Index_Finger_Movement", "IFM"),
    "little finger": ("Little_Finger_Movement", "LFM"),
    "middle finger": ("Middle_Finger_Movement", "MFM"),
    "ring finger": ("Ring_Finger_Movement", "RFM"),
    "thumb finger": ("Thumb_Finger_Movement", "TFM"),
    "open close": ("Hand_Open_Close", "HOC"),
    "pen holding": ("Pen_Holding", "PH"),
}

set_map: Dict[str, str] = {
    "set 1": "SetA",
    "set 2": "SetB",
    "set 3": "SetC",
}

# --- 4. CORE CROPPING FUNCTION ---
def crop_dataset(input_path, output_base_dir, filename, fs, skip_seconds, keep_seconds, gesture_name, set_label, subject_id, tolerance=0.5):
    try:
        df = pd.read_csv(input_path, header=0, engine='python')
    except Exception as e:
        print(f"FATAL ERROR: Could not load {input_path}. Error: {e}")
        return

    start_idx = skip_seconds * fs
    end_idx = start_idx + keep_seconds * fs
    expected_rows = keep_seconds * fs
    cropped_df = df.iloc[start_idx:end_idx].copy()

    # --- TIMESTAMP ADJUSTMENT ---
    actual_duration = keep_seconds
    try:
        ts_column = [col for col in df.columns if 'timestamp' in col.lower()][0]
        ts = pd.to_datetime(cropped_df[ts_column])
        actual_duration = (ts.iloc[-1] - ts.iloc[0]).total_seconds()

        if abs(actual_duration - keep_seconds) > tolerance:
            target_end_time = ts.iloc[0] + pd.Timedelta(seconds=keep_seconds)
            closest_idx = (ts - target_end_time).abs().idxmin()
            cropped_df = cropped_df.loc[cropped_df.index[0]:closest_idx].copy()
            actual_duration = (pd.to_datetime(cropped_df[ts_column]).iloc[-1] - ts.iloc[0]).total_seconds()
    except:
        pass

    # Verification String for Log
    verification_msg = f"✅ SAMPLE COUNT VERIFIED: Dataset has exactly {len(cropped_df)} rows ({len(cropped_df)} samples)." if len(cropped_df) == expected_rows else f"⚠️ SAMPLE COUNT MISMATCH: {len(cropped_df)} rows found."

    # --- SAVE DATA AND LOGS FOR ALL 3 CATEGORIES ---
    # subject_id is now passed dynamically
    projects = {
        "EMG_EEG_Combine_Gesture": cropped_df,
        "EMG_Gesture": cropped_df.iloc[:, [0, 1, 2, 3, 4]],
        "EEG_Gesture": cropped_df.iloc[:, [0, 1] + list(range(5, 18))]
    }

    for proj_name, data in projects.items():
        save_dir = os.path.join(output_base_dir, proj_name, f"Subject_{subject_id}", gesture_name)
        os.makedirs(save_dir, exist_ok=True)
        
        csv_path = os.path.join(save_dir, filename)
        log_path = os.path.join(save_dir, filename.replace(".csv", "_log.txt"))
        
        # Save CSV
        data.to_csv(csv_path, index=False)
        
        # Write Log
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"--- Segment Log for {set_label} ---\n")
            f.write(f"Subject ID: {subject_id}\n")
            f.write(f"Segment Start Skip: {skip_seconds} seconds\n")
            f.write(f"Time Window: {skip_seconds}-{skip_seconds+keep_seconds} sec\n")
            f.write(f"Sample Indices: {start_idx}-{end_idx} | Expected rows: {expected_rows}\n")
            f.write(f"Initial cropped rows: {len(cropped_df)}\n")
            f.write(f"{verification_msg}\n")
            f.write(f"Timestamps span: {actual_duration:.3f} sec (expected ~{keep_seconds})\n")
            f.write(f"✅ Saved {proj_name.split('_')[0]} ({len(data.columns)} cols): {csv_path}")

    print(f"✅ Processed: Subject {subject_id} | {filename}")

# --- 5. SUBJECT PROCESSING ---
def process_subject(subject_id: str):
    print(f"\n--- SCANNING SUBJECT {subject_id} ---")
    try:
        subject_folder = None
        # Handle variations in folder naming (e.g., "Subject 01", "Subject 01(Name)")
        for folder in os.listdir(BASE_INPUT_ROOT):
            if folder.startswith(f"Subject {subject_id}"):
                subject_folder = os.path.join(BASE_INPUT_ROOT, folder)
                break
        
        if not subject_folder:
            print(f"Skipping Subject {subject_id}: Folder not found.")
            return

        for folder in os.listdir(subject_folder):
            folder_path = os.path.join(subject_folder, folder)
            if not os.path.isdir(folder_path): continue

            # Extract info from folder names
            gesture_info = next(((v[0], v[1]) for k, v in gesture_map.items() if k in folder.lower()), None)
            set_name = next((v for k, v in set_map.items() if k in folder.lower()), None)

            if gesture_info and set_name:
                input_file = os.path.join(folder_path, "combined_dataset.csv")
                if os.path.exists(input_file):
                    fname = f"S{subject_id}_{gesture_info[1]}_{set_name}.csv"
                    # Pass subject_id to the function
                    crop_dataset(input_file, BASE_OUTPUT, fname, fs, skip_seconds, keep_seconds, gesture_info[0], set_name, subject_id, tolerance)
    except Exception as e:
        print(f"🚨 ERROR in Subject {subject_id}: {e}")

if __name__ == '__main__':
    for sid in SUBJECT_IDS:
        process_subject(sid)
Displaying update croping subject 01-40.py.
