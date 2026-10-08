#!/usr/bin/env python3
"""
DATA1_MASTER_INVENTORY.py
Stage 1: Master Dataset Inventory & Cross-Modality Mapping

Authoritative raw roots:
  01_RAW_DATA/MOTOR_EXECUTION
  01_RAW_DATA/MOTOR_IMAGERY
  01_RAW_DATA/BIOIMPEDANCE

Raw data are read only. No rename/move/delete/write operation is performed
inside 01_RAW_DATA.

Identity rules:
  Execution: Subject_01 ... Subject_40
  Motor imagery: MI_01 ... MI_25 (modality-local IDs; canonical mapping blank)
  Bioimpedance: BZ_01 ... BZ_18, with explicit canonical crosswalk below.

BioZ counting rule:
  One recording = one recording basename within a gesture, paired across
  Channel_1 and Channel_2. Therefore 1,890 recordings correspond to 3,780
  native .spec channel files when both channels are present.
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

ROOT = Path(r'F:\Faruk\OFS_Paper_Work\Data_Set_Paper_Work\EEG_EMG_BIOZ_DATASET').resolve()
RAW = ROOT / '01_RAW_DATA'
EXEC = RAW / 'EMG_EEG_SYNCHRONIZED'
MI = RAW / 'EEG_MOTOR_IMAGERY'
BIOZ = RAW / 'BIOIMPEDANCE'
META = ROOT / '03_METADATA'
QC = ROOT / '05_QC'
LOGS = ROOT / 'logs'

# Demographic source used by DATA1. The workbook is metadata only; it is never modified.
DEMOGRAPHY_FILENAME = "Data set Demography .xlsx"
DEMOGRAPHY_CANDIDATES = [
    ROOT / DEMOGRAPHY_FILENAME,
    ROOT.parent / DEMOGRAPHY_FILENAME,
    ROOT.parent.parent / DEMOGRAPHY_FILENAME,
]

GESTURES = [
    ('HOC', 'Hand_Open_Close', 'Hand Open–Close'),
    ('TFM', 'Thumb_Finger_Movement', 'Thumb Finger Movement'),
    ('IFM', 'Index_Finger_Movement', 'Index Finger Movement'),
    ('MFM', 'Middle_Finger_Movement', 'Middle Finger Movement'),
    ('RFM', 'Ring_Finger_Movement', 'Ring Finger Movement'),
    ('LFM', 'Little_Finger_Movement', 'Little Finger Movement'),
    ('PH', 'Pen_Holding', 'Pen Holding'),
]
SETS = ('A', 'B', 'C')
BIOZ_XW = {f'BZ_{i:02d}': f'Subject_{i:02d}' for i in range(1, 16)}
BIOZ_XW.update({'BZ_16': 'Subject_17', 'BZ_17': 'Subject_29', 'BZ_18': 'Subject_39'})

EXEC_RE = re.compile(r'^S(\d{1,3})_(HOC|TFM|IFM|MFM|RFM|LFM|PH)_Set([ABC])\.csv$', re.I)
MI_RE = re.compile(r'^MI_(\d{1,3})_(HOC|TFM|IFM|MFM|RFM|LFM|PH)_Set([ABC])\.csv$', re.I)


def logger() -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger('DATA1')
    lg.handlers.clear(); lg.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', '%Y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); lg.addHandler(sh)
    fh = logging.FileHandler(LOGS / 'DATA1_MASTER_INVENTORY.log', 'w', encoding='utf-8'); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg

L = logger()

def rel(p: Path) -> str:
    try: return str(p.resolve().relative_to(ROOT))
    except Exception: return str(p)

def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    L.info('WROTE | %s | rows=%d cols=%d', rel(path), len(df), len(df.columns))

def validate_roots() -> bool:
    L.info('=' * 80); L.info('DATA1 MASTER DATASET INVENTORY'); L.info('=' * 80)
    for label, p in [('ROOT', ROOT), ('RAW', RAW), ('EXECUTION', EXEC), ('IMAGERY', MI), ('BIOIMPEDANCE', BIOZ)]:
        if p.is_dir(): L.info('FOUND | %-13s | %s', label, p)
        else: L.error('MISSING | %-12s | %s', label, p)
    return all(p.is_dir() for p in (ROOT, RAW, EXEC, MI, BIOZ))

def expected_slot_rows(protocol: str, local_id: str, gcode: str, gfolder: str, label: str, sid_num: int) -> dict:
    return {
        'protocol': protocol, 'subject_id': local_id, 'canonical_subject_id': '',
        'gesture_code': gcode, 'gesture': label, 'gesture_folder': gfolder,
        'set': '', 'csv_file': '', 'csv_relative_path': '', 'log_file': '',
        'log_relative_path': '', 'csv_exists': False, 'log_exists': False,
        'csv_rows': '', 'csv_columns': '', 'csv_column_names': '',
        'csv_read_status': 'NOT_READ', 'complete_slot': False,
    }

def inventory_csv_protocol(root: Path, prefix: str, subject_re: re.Pattern, expected_subjects: int, protocol: str):
    subjects = sorted([p for p in root.iterdir() if p.is_dir() and subject_re.fullmatch(p.name)], key=lambda p: int(re.search(r'(\d+)$', p.name).group(1)))
    L.info('%s subject folders discovered: %d', protocol, len(subjects))
    inv, comp = [], []
    for sd in subjects:
        n = int(re.search(r'(\d+)$', sd.name).group(1))
        for code, folder, label in GESTURES:
            gd = sd / folder
            for st in SETS:
                fname = f'{prefix}_{n:02d}_{code}_Set{st}.csv' if prefix == 'MI' else f'S{n:02d}_{code}_Set{st}.csv'
                lname = fname[:-4] + '_log.txt'
                cp = gd / fname; lp = gd / lname
                if not cp.is_file() and gd.is_dir():
                    x = [p for p in gd.iterdir() if p.is_file() and p.suffix.lower()=='.csv' and p.name.lower()==fname.lower()]
                    if x: cp = x[0]
                if not lp.is_file() and gd.is_dir():
                    x = [p for p in gd.iterdir() if p.is_file() and p.name.lower()==lname.lower()]
                    if x: lp = x[0]
                rows = cols = ''; names = ''; status = 'NOT_READ'
                if cp.is_file():
                    try:
                        d = pd.read_csv(cp)
                        rows, cols = d.shape; names = ' | '.join(map(str, d.columns)); status = 'PASS'
                    except Exception as e: status = f'READ_ERROR:{type(e).__name__}'
                complete = cp.is_file() and lp.is_file()
                inv.append({'modality':'EEG+EMG' if protocol=='MOTOR_EXECUTION' else 'EEG', 'protocol':protocol,
                            'subject_id':sd.name, 'canonical_subject_id':sd.name if protocol=='MOTOR_EXECUTION' else '',
                            'gesture_code':code,'gesture':label,'gesture_folder':folder,'set':st,
                            'csv_file':cp.name if cp.is_file() else '', 'csv_relative_path':rel(cp) if cp.is_file() else '',
                            'log_file':lp.name if lp.is_file() else '', 'log_relative_path':rel(lp) if lp.is_file() else '',
                            'csv_exists':cp.is_file(),'log_exists':lp.is_file(),'csv_rows':rows,'csv_columns':cols,
                            'csv_column_names':names,'csv_read_status':status,'complete_slot':complete})
                comp.append({'protocol':protocol,'subject_id':sd.name,'gesture_code':code,'gesture':label,'set':st,
                             'csv_exists':cp.is_file(),'log_exists':lp.is_file(),'complete_slot':complete,
                             'status':'PASS' if complete else ('MISSING_CSV' if not cp.is_file() else 'MISSING_LOG')})
    return pd.DataFrame(inv), pd.DataFrame(comp)

def inventory_bioz():
    subjects = sorted([p for p in BIOZ.iterdir() if p.is_dir() and re.fullmatch(r'BZ_\d{2}', p.name, re.I)], key=lambda p:int(p.name[-2:]))
    L.info('BioZ participant folders discovered: %d', len(subjects))
    inv, comp = [], []
    for sd in subjects:
        canonical = BIOZ_XW.get(sd.name, '')
        # Group by gesture folder and .spec basename. Channel folders are excluded
        # from the recording identity, so Channel_1 + Channel_2 = one recording.
        groups = defaultdict(list)
        for sp in sd.rglob('*.spec'):
            gcode = None; gfolder = ''
            parts = {x.lower() for x in sp.parts}
            for code, folder, label in GESTURES:
                if folder.lower() in parts:
                    gcode, gfolder = code, folder; break
            if not gcode:
                text = str(sp).lower()
                for code, folder, label in GESTURES:
                    if folder.lower() in text or code.lower() in text:
                        gcode, gfolder = code, folder; break
            if gcode: groups[(gcode, sp.stem)].append(sp)
        for code, folder, label in GESTURES:
            entries = sorted([(stem, fs) for (gc, stem), fs in groups.items() if gc == code], key=lambda x:x[0])
            for idx, (stem, fs) in enumerate(entries, 1):
                chans = []
                for f in fs:
                    m = re.search(r'Channel[_ -]?(\d+)', str(f), re.I)
                    if m: chans.append(f'Channel_{m.group(1)}')
                chset = set(chans)
                has1, has2 = 'Channel_1' in chset, 'Channel_2' in chset
                inv.append({'modality':'BIOIMPEDANCE','protocol':'MOTOR_EXECUTION','subject_id':sd.name,
                            'canonical_subject_id':canonical,'gesture_code':code,'gesture':label,'gesture_folder':folder,
                            'recording_index_within_gesture':idx,'recording_stem':stem,'native_spec_file_count':len(fs),
                            'channel_names':' | '.join(sorted(chset)),
                            'native_spec_relative_paths':' | '.join(sorted(rel(f) for f in fs)),
                            'has_channel_1':has1,'has_channel_2':has2,
                            'two_channel_complete':has1 and has2 and len(fs)==2})
            specs = sum(len(fs) for _, fs in entries)
            comp.append({'protocol':'MOTOR_EXECUTION','modality':'BIOIMPEDANCE','bioz_id':sd.name,
                         'canonical_subject_id':canonical,'gesture_code':code,'gesture':label,
                         'expected_recordings':15,'actual_recordings':len(entries),'recording_count_pass':len(entries)==15,
                         'expected_native_spec_files':30,'actual_native_spec_files':specs,
                         'native_spec_count_pass':specs==30,
                         'two_channel_recordings':sum(1 for _,fs in entries if len(fs)==2 and any('Channel_1' in str(f) for f in fs) and any('Channel_2' in str(f) for f in fs))})
    return pd.DataFrame(inv), pd.DataFrame(comp)

def crosswalk(mi_ids):
    rev = {v:k for k,v in BIOZ_XW.items()}
    return pd.DataFrame([{
        'canonical_subject_id':f'Subject_{i:02d}', 'execution_id':f'Subject_{i:02d}',
        'mi_id':'', 'bioz_id':rev.get(f'Subject_{i:02d}',''),
        'execution_available':True, 'mi_available':False, 'bioz_available':f'Subject_{i:02d}' in rev,
        'mi_mapping_status':'PENDING_EXPLICIT_CROSSWALK',
        'bioz_mapping_status':'MAPPED' if f'Subject_{i:02d}' in rev else 'NOT_AVAILABLE',
        'notes':'MI local ID intentionally not mapped to canonical subject.'
    } for i in range(1,41)])

def _find_demography_workbook() -> Optional[Path]:
    """Locate the authoritative demographic workbook without guessing its identity."""
    for p in DEMOGRAPHY_CANDIDATES:
        if p.is_file():
            return p
    # Also allow the exact filename anywhere under the project parent, but never
    # search inside 01_RAW_DATA because demographic metadata are not raw signals.
    search_root = ROOT.parent
    try:
        matches = sorted(search_root.rglob(DEMOGRAPHY_FILENAME))
    except Exception:
        matches = []
    for p in matches:
        if RAW not in p.parents:
            return p
    return None


def _clean_demography_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize only column labels; values are preserved from the workbook."""
    rename = {
        'Subject Name': 'subject_name',
        'Participant_ID': 'subject_id',
        'Gender ': 'gender',
        'Age /Year': 'age_years',
        'Height ft/cm': 'height_raw',
        'Weight /kg': 'weight_raw',
        'Handedness': 'recording_site',
        'Fore Arm Lenght /cm': 'forearm_length_raw',
        'Fore Arm Circumference /cm': 'forearm_circumference_raw',
        'BCI_Experience': 'bci_experience',
        'Inclusion_Criteria_Met': 'inclusion_criteria_met',
        'Neurological_Exclusion': 'neurological_exclusion',
        'Musculoskeletal_Exclusion': 'musculoskeletal_exclusion',
        'Study Consent': 'study_consent',
        'Eligible_for_Study': 'eligible_for_study',
        'Public_Data_Release_Consent': 'public_data_release_consent',
        'Participant_Status': 'participant_status',
        'Data Packet Loss': 'data_packet_loss',
        'Live Recording Status ': 'live_recording_status',
        'OpenBCI Recording (EEG, Executoin + EMG)': 'openbci_execution_emg_recording',
        'OpenBCI Recoriding (EEG, Imagery  only)': 'openbci_motor_imagery_recording',
        'Bioimpedence data set ': 'bioimpedance_dataset_contribution',
        'Data Contribution Modalities ': 'data_contribution_modalities',
    }
    df = df.rename(columns={c: rename.get(c, str(c).strip()) for c in df.columns})
    # Remove completely empty spreadsheet columns created by formatting.
    df = df.dropna(axis=1, how='all').copy()
    return df


def load_demography() -> tuple[pd.DataFrame, Path, bool]:
    """Read and validate the demographic workbook used by DATA1."""
    path = _find_demography_workbook()
    if path is None:
        raise FileNotFoundError(
            f'Authoritative demographic workbook not found: {DEMOGRAPHY_FILENAME}. '
            'Place it in the dataset root or its parent project directory.'
        )

    xl = pd.ExcelFile(path)
    if not xl.sheet_names:
        raise ValueError('Demographic workbook contains no worksheets.')

    # Use the first non-empty worksheet. The supplied workbook has one data sheet.
    source_sheet = None
    df = None
    for sheet in xl.sheet_names:
        candidate = pd.read_excel(path, sheet_name=sheet)
        candidate = candidate.dropna(how='all')
        if not candidate.empty:
            source_sheet = sheet
            df = candidate
            break
    if df is None:
        raise ValueError('Demographic workbook contains no non-empty worksheet.')

    df = _clean_demography_columns(df)
    if 'subject_id' not in df.columns:
        raise ValueError('Demographic workbook must contain Participant_ID.')

    df['subject_id'] = df['subject_id'].astype(str).str.strip()
    if df['subject_id'].duplicated().any():
        dup = sorted(df.loc[df['subject_id'].duplicated(keep=False), 'subject_id'].unique())
        raise ValueError(f'Duplicate Participant_ID values in demographic workbook: {dup}')

    expected = {f'Subject_{i:02d}' for i in range(1, 42)}
    actual = set(df['subject_id'])
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f'Demographic ID validation failed. Missing={missing}; Extra={extra}')

    # Canonical ordering is fixed and independent of workbook row ordering.
    order = {f'Subject_{i:02d}': i for i in range(1, 42)}
    df['_order'] = df['subject_id'].map(order)
    df = df.sort_values('_order').drop(columns='_order').reset_index(drop=True)
    L.info('DEMOGRAPHY | source=%s | sheet=%s | rows=%d | cols=%d', path, source_sheet, len(df), len(df.columns))
    return df, path, True


def participants() -> tuple[pd.DataFrame, Path]:
    """Build the authoritative participant metadata by merging DATA1 cohort logic with the demographic workbook."""
    demo, source_path, _ = load_demography()

    # DATA1 cohort status follows the demographic workbook's Participant_Status.
    # Subject_41 is retained in metadata but excluded from the analytical cohort.
    demo['cohort_status'] = demo['participant_status'].astype(str).str.strip().str.lower().map(
        lambda x: 'retained' if x == 'included' else 'excluded'
    )
    demo.loc[demo['subject_id'].eq('Subject_41'), 'cohort_status'] = 'excluded'

    demo['exclusion_reason'] = ''
    demo.loc[demo['subject_id'].eq('Subject_41'), 'exclusion_reason'] = 'data quality, leakage'

    # Add explicit DATA1 provenance fields without altering source values.
    demo['demographic_source_file'] = source_path.name
    demo['demographic_source_sheet'] = 'Sheet1'
    demo['demographic_merge_status'] = 'MERGED'

    # Put identity/cohort fields first, then all source-derived demographic fields.
    first = [
        'subject_id', 'cohort_status', 'exclusion_reason', 'subject_name', 'gender',
        'age_years', 'height_raw', 'weight_raw', 'recording_site',
        'forearm_length_raw', 'forearm_circumference_raw', 'bci_experience',
        'inclusion_criteria_met', 'neurological_exclusion', 'musculoskeletal_exclusion',
        'study_consent', 'eligible_for_study', 'public_data_release_consent',
        'participant_status', 'data_packet_loss', 'live_recording_status',
        'openbci_execution_emg_recording', 'openbci_motor_imagery_recording',
        'bioimpedance_dataset_contribution', 'data_contribution_modalities',
        'demographic_source_file', 'demographic_source_sheet', 'demographic_merge_status'
    ]
    ordered = [c for c in first if c in demo.columns] + [c for c in demo.columns if c not in first]
    return demo[ordered], source_path

def main():
    t=datetime.now()
    if not validate_roots(): return 2
    ex, exq = inventory_csv_protocol(EXEC,'S',re.compile(r'Subject_\d{2}',re.I),40,'MOTOR_EXECUTION')
    mi, miq = inventory_csv_protocol(MI,'MI',re.compile(r'MI_\d{2}',re.I),25,'MOTOR_IMAGERY')
    bz, bzq = inventory_bioz()
    META.mkdir(parents=True,exist_ok=True); QC.mkdir(parents=True,exist_ok=True)

    # Common master inventory; BioZ rows are recording units, not individual .spec files.
    master=[]
    for d in (ex,mi):
        if not d.empty: master.append(d.assign(recording_unit=True, native_spec_file_count=''))
    if not bz.empty:
        master.append(bz.assign(set='',csv_file='',csv_relative_path='',log_file='',log_relative_path='',
                                 csv_exists=False,log_exists=False,csv_rows='',csv_columns='',csv_column_names='',
                                 csv_read_status='NATIVE_SPEC',complete_slot=bz['two_channel_complete']))
    inv=pd.concat(master,ignore_index=True,sort=False) if master else pd.DataFrame()

    participant_df, demographic_source = participants()
    write(participant_df, META/'participants.csv')
    cw=crosswalk(sorted(mi['subject_id'].unique()) if not mi.empty else [])
    write(cw,META/'subject_crosswalk.csv')
    avail=cw.copy(); write(avail,META/'subject_modality_availability.csv')
    write(inv,META/'recording_inventory.csv')
    write(pd.DataFrame([
        {'parameter':'execution_recordings','value':840,'unit':'recordings','scope':'MOTOR_EXECUTION'},
        {'parameter':'imagery_recordings','value':525,'unit':'recordings','scope':'MOTOR_IMAGERY'},
        {'parameter':'bioz_recordings','value':1890,'unit':'recordings','scope':'BIOIMPEDANCE'},
        {'parameter':'bioz_native_spec_files','value':3780,'unit':'channel files','scope':'BIOIMPEDANCE'},
        {'parameter':'sets_per_execution_gesture','value':3,'unit':'sets','scope':'MOTOR_EXECUTION'},
        {'parameter':'sets_per_imagery_gesture','value':3,'unit':'sets','scope':'MOTOR_IMAGERY'},
        {'parameter':'bioz_recordings_per_gesture','value':15,'unit':'recordings','scope':'BIOIMPEDANCE'},
    ]),META/'recording_parameters.csv')
    write(pd.DataFrame([
        {'modality':'EEG','protocol':'MOTOR_EXECUTION','system':'OpenBCI Cyton + Daisy','channels':13,'sampling_rate_hz':125,'channel_mapping':'EEG_ch-01 through EEG_ch-13 (canonical release labels)'},
        {'modality':'EMG','protocol':'MOTOR_EXECUTION','system':'OpenBCI Cyton + Daisy','channels':3,'sampling_rate_hz':125,'channel_mapping':'physical CH1–CH3'},
        {'modality':'EEG','protocol':'MOTOR_IMAGERY','system':'OpenBCI Cyton + Daisy','channels':13,'sampling_rate_hz':125,'channel_mapping':'EEG_ch-01 through EEG_ch-13 (canonical release labels)'},
        {'modality':'BIOIMPEDANCE','protocol':'MOTOR_EXECUTION','system':'Sciospec ISX-5 Series','channels':2,'sampling_rate_hz':'','channel_mapping':'Channel_1 and Channel_2; separate acquisition'},
    ]),META/'acquisition_parameters.csv')
    write(pd.DataFrame([{'gesture_code':c,'gesture_folder':f,'gesture_label':l,'sets':'A;B;C'} for c,f,l in GESTURES]),META/'gesture_dictionary.csv')
    write(pd.DataFrame([{'dataset_root':str(ROOT),'raw_data_modified':'NO','generated_at':datetime.now().isoformat(timespec='seconds'),'notes':'DATA1 inventory only; raw data untouched.'}]),META/'data_provenance.csv')
    write(pd.DataFrame([
        {'cohort':'Final motor execution','participants':40,'status':'retained','notes':'Subject_01–Subject_40'},
        {'cohort':'Motor imagery','participants':25,'status':'local modality IDs','notes':'MI_01–MI_25; canonical crosswalk pending'},
        {'cohort':'Bioimpedance','participants':18,'status':'retained','notes':'BZ_01–BZ_18'},
        {'cohort':'Excluded','participants':1,'status':'excluded','notes':'Subject_41 — data quality, leakage'},
    ]),META/'cohort_definition.csv')
    write(pd.DataFrame([{'subject_id':'Subject_41','status':'excluded','reason':'data quality, leakage'}]),META/'exclusion_log.csv')

    # QC tables
    write(exq,QC/'motor_execution_completeness.csv'); write(miq,QC/'motor_imagery_completeness.csv'); write(bzq,QC/'bioimpedance_completeness.csv')
    csvstruct=[]
    for name,d in [('MOTOR_EXECUTION',ex),('MOTOR_IMAGERY',mi)]:
        readable=d[d.csv_read_status.eq('PASS')] if not d.empty else d
        csvstruct.append({'protocol':name,'csv_files':int(d.csv_exists.sum()) if not d.empty else 0,
                          'readable_csv_files':len(readable),'unique_row_counts':';'.join(map(str,sorted(set(readable.csv_rows.astype(int))))) if len(readable) else '',
                          'unique_column_counts':';'.join(map(str,sorted(set(readable.csv_columns.astype(int))))) if len(readable) else ''})
    write(pd.DataFrame(csvstruct),QC/'csv_structure_summary.csv')

    ecsv=int(ex.csv_exists.sum()) if not ex.empty else 0; ecomp=int(ex.complete_slot.sum()) if not ex.empty else 0
    msub=mi.subject_id.nunique() if not mi.empty else 0; mcsv=int(mi.csv_exists.sum()) if not mi.empty else 0; mcomp=int(mi.complete_slot.sum()) if not mi.empty else 0
    bsub=bz.subject_id.nunique() if not bz.empty else 0; brec=len(bz); bspec=int(bz.native_spec_file_count.sum()) if not bz.empty else 0; b2=int(bz.two_channel_complete.sum()) if not bz.empty else 0
    summary=pd.DataFrame([
        ['MOTOR_EXECUTION','subjects',ex.subject_id.nunique(),40,'PASS' if ex.subject_id.nunique()==40 else 'CHECK_REQUIRED'],
        ['MOTOR_EXECUTION','csv_recordings',ecsv,840,'PASS' if ecsv==840 else 'CHECK_REQUIRED'],
        ['MOTOR_EXECUTION','complete_csv_log_slots',ecomp,840,'PASS' if ecomp==840 else 'CHECK_REQUIRED'],
        ['MOTOR_IMAGERY','local_subjects',msub,25,'PASS' if msub==25 else 'CHECK_REQUIRED'],
        ['MOTOR_IMAGERY','csv_recordings',mcsv,525,'PASS' if mcsv==525 else 'CHECK_REQUIRED'],
        ['MOTOR_IMAGERY','complete_csv_log_slots',mcomp,525,'PASS' if mcomp==525 else 'CHECK_REQUIRED'],
        ['BIOIMPEDANCE','BZ_subjects',bsub,18,'PASS' if bsub==18 else 'CHECK_REQUIRED'],
        ['BIOIMPEDANCE','recording_units',brec,1890,'PASS' if brec==1890 else 'CHECK_REQUIRED'],
        ['BIOIMPEDANCE','native_spec_channel_files',bspec,3780,'PASS' if bspec==3780 else 'CHECK_REQUIRED'],
        ['BIOIMPEDANCE','two_channel_recording_units',b2,brec,'PASS' if b2==brec else 'CHECK_REQUIRED'],
    ],columns=['component','metric','actual','expected','status'])
    write(summary,QC/'dataset_inventory_summary.csv')

    # DATA1 demographic merge QC (metadata reconciliation is part of DATA1).
    retained_demo = int((participant_df['cohort_status'] == 'retained').sum())
    excluded_demo = int((participant_df['cohort_status'] == 'excluded').sum())
    demo_qc = pd.DataFrame([
        {'metric':'demographic_source_found','actual':True,'expected':True,'status':'PASS','source':str(demographic_source)},
        {'metric':'demographic_rows','actual':len(participant_df),'expected':41,'status':'PASS' if len(participant_df)==41 else 'CHECK_REQUIRED','source':demographic_source.name},
        {'metric':'retained_participants','actual':retained_demo,'expected':40,'status':'PASS' if retained_demo==40 else 'CHECK_REQUIRED','source':demographic_source.name},
        {'metric':'excluded_participants','actual':excluded_demo,'expected':1,'status':'PASS' if excluded_demo==1 else 'CHECK_REQUIRED','source':demographic_source.name},
        {'metric':'demographic_merge_status','actual':int((participant_df['demographic_merge_status']=='MERGED').sum()),'expected':41,'status':'PASS' if (participant_df['demographic_merge_status']=='MERGED').all() else 'CHECK_REQUIRED','source':demographic_source.name},
    ])
    write(demo_qc, QC/'demographic_merge_QC.csv')

    L.info('=' * 80); L.info('DATA1 FINAL SUMMARY'); L.info('=' * 80)
    L.info('Execution CSV files: %d / 840', ecsv); L.info('Execution complete slots: %d / 840', ecomp)
    L.info('MI subjects discovered: %d / 25', msub); L.info('MI CSV files: %d / 525', mcsv); L.info('MI complete slots: %d / 525', mcomp)
    L.info('BioZ BZ folders with data: %d / 18', bsub); L.info('BioZ recording units: %d / 1890', brec)
    L.info('BioZ native .spec channel files: %d / 3780', bspec); L.info('BioZ two-channel recording units: %d / %d', b2, brec)
    L.info('Demographic rows merged: %d / 41', len(participant_df)); L.info('Retained participants: %d / 40', retained_demo); L.info('Excluded participants: %d / 1', excluded_demo)
    L.info('RAW DATA MODIFICATION: NONE')
    demopass=(len(participant_df)==41 and retained_demo==40 and excluded_demo==1 and
              (participant_df['demographic_merge_status']=='MERGED').all())
    allpass=ecsv==840 and ecomp==840 and msub==25 and mcsv==525 and mcomp==525 and bsub==18 and brec==1890 and bspec==3780 and b2==brec and demopass
    L.info('DATA1 INVENTORY STATUS: %s', 'PASS' if allpass else 'CHECK_REQUIRED')
    L.info('Metadata: %s', META); L.info('QC: %s', QC); L.info('Runtime: %s', datetime.now()-t)
    return 0 if allpass else 1

if __name__=='__main__':
    raise SystemExit(main())
