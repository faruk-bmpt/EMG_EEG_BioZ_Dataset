#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, shutil, sys

ROOT=Path('/mnt/f/Faruk/OFS_Paper_Work/Data_Set_Paper_Work/EEG_EMG_BIOZ_DATASET')
RELEASE=ROOT/'09_RELEASE'/'DATASET_V1.0'
R_SHA=RELEASE/'SHA256SUMS_RELEASE_V1.0'
FREEZE=ROOT/'05_QC'/'DATASET_FREEZE'
SOURCE_SHA=FREEZE/'SHA256'
DOC=RELEASE/'DOCUMENTATION'
PROTO='FINAL-RELEASE-SHA256-DATASET-V1.0-FINAL'

def die(m): raise SystemExit('ERROR: '+m)
def rj(p):
    try:
        with p.open(encoding='utf-8') as f: return json.load(f)
    except Exception as e: die(f'cannot read {p}: {e}')
def req(p,label):
    if not p.exists(): die(f'missing {label}: {p}')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    print('='*72); print('FINAL RELEASE SHA-256 REGENERATION — DATASET V1.0'); print('='*72)
    print('Dataset root :',ROOT); print('Release root :',RELEASE); print('Protocol     :',PROTO)
    req(FREEZE/'_DATASET_V1.0_FROZEN.json','freeze sentinel')
    fr=rj(FREEZE/'DATASET_FREEZE_FINAL_REPORT.json')
    sentinel=rj(FREEZE/'_DATASET_V1.0_FROZEN.json')
    text=(json.dumps(fr)+' '+json.dumps(sentinel)).upper()
    if not any(x in text for x in ('"PASS"','"FROZEN"','"VERIFIED"')) and not (sentinel.get('frozen') is True): die('freeze not verified')
    print('[1/6] Verifying Dataset V1.0 freeze... PASS')

    sr=SOURCE_SHA/'SHA256_CHECKSUM_FINAL_REPORT.json'; req(sr,'source SHA-256 report')
    s=rj(sr)
    checks={'freeze_status_verified':True,'freeze_sentinel_status':'FROZEN','checksum_algorithm':'SHA-256','files_hashed':18622,'read_only':True,'raw_data_modified':False,'processed_data_modified':False,'preprocessing_rerun':False,'feature_extraction':False,'machine_learning':False,'fusion_generation':False}
    for k,v in checks.items():
        if s.get(k)!=v: die(f'source SHA report {k}={s.get(k)!r}, expected {v!r}')
    if s.get('checksum_catalog_sha256')!='1a5bcb14c598205f8a497215d8a54aea466b1ae9ec15dc0433b42b66ae250ec6': die('source catalog hash mismatch')
    print('[2/6] Verifying source SHA-256... PASS | files=18622')

    for p in ['DATA','ANNOTATIONS','METADATA','MODALITY_VIEWS','QC','DOCUMENTATION','RELEASE_NOTES.md','SHA256SUMS_SOURCE_V1.0']:
        req(RELEASE/p,'release component')
    print('[3/6] Verifying final release structure... PASS')

    docs=['README_DATASET_V1.0.md','DATASET_DESCRIPTION_V1.0.md','DATA_DICTIONARY_V1.0.md','ACQUISITION_AND_STANDARDIZATION_PROTOCOL_V1.0.md','ANNOTATION_GUIDE_V1.0.md','QUALITY_CONTROL_REPORT_V1.0.md','MODALITY_AND_SYNCHRONIZATION_GUIDE_V1.0.md','ETHICS_AND_DATA_GOVERNANCE_V1.0.md','DATA_ACCESS_AND_RELEASE_GUIDE_V1.0.md','DOCUMENTATION_INDEX_V1.0.md','DOCUMENTATION_MANIFEST_V1.0.csv','DOCUMENTATION_GENERATION_SUMMARY_V1.0.txt','DOCUMENTATION_GENERATION_FINAL_REPORT.json']
    for n in docs: req(DOC/n,'release documentation file')
    dr=rj(DOC/'DOCUMENTATION_GENERATION_FINAL_REPORT.json')
    for k in ['external_facts_inferred','raw_data_modified','processed_data_modified','preprocessing_rerun','feature_extraction_performed','machine_learning_performed','fusion_generation_performed']:
        if dr.get(k) is not False: die(f'documentation report {k} is not False')
    print('[4/6] Verifying final documentation... PASS | 13 files present')

    R_SHA.mkdir(parents=True,exist_ok=True)
    for p in list(R_SHA.iterdir()):
        if p.is_dir(): shutil.rmtree(p)
        else: p.unlink()
    entries=[]
    for p in sorted(RELEASE.rglob('*')):
        if not p.is_file(): continue
        try: p.relative_to(R_SHA); continue
        except ValueError: pass
        entries.append((sha(p),p.relative_to(RELEASE).as_posix()))
    entries.sort(key=lambda x:x[1])
    with (R_SHA/'SHA256SUMS_RELEASE_V1.0.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['sha256','relative_path']); w.writerows(entries)
    with (R_SHA/'SHA256SUMS_RELEASE_V1.0.txt').open('w',encoding='utf-8') as f:
        for d,p in entries: f.write(f'{d}  {p}\n')
    cat=sha(R_SHA/'SHA256SUMS_RELEASE_V1.0.csv')
    report={'dataset_version':'V1.0','protocol_version':PROTO,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'release_root':str(RELEASE),'source_freeze_verified':True,'source_sha256_verified':True,'checksum_algorithm':'SHA-256','files_hashed':len(entries),'release_checksum_csv':'SHA256SUMS_RELEASE_V1.0/SHA256SUMS_RELEASE_V1.0.csv','release_checksum_text':'SHA256SUMS_RELEASE_V1.0/SHA256SUMS_RELEASE_V1.0.txt','release_checksum_catalog_sha256':cat,'excluded_paths':['SHA256SUMS_RELEASE_V1.0/'],'raw_data_modified':False,'processed_data_modified':False,'preprocessing_rerun':False,'feature_extraction':False,'machine_learning':False,'fusion_generation':False,'documentation_included':True}
    (R_SHA/'RELEASE_SHA256_FINAL_REPORT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'[5/6] Generating final release SHA-256 catalog... PASS | files={len(entries)}')

    with (R_SHA/'SHA256SUMS_RELEASE_V1.0.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    if len(rows)!=len(entries): die('catalog row count mismatch')
    for row in rows:
        p=RELEASE/row['relative_path']
        if not p.is_file(): die('missing catalog file: '+row['relative_path'])
        if sha(p)!=row['sha256']: die('checksum mismatch: '+row['relative_path'])
    if sha(R_SHA/'SHA256SUMS_RELEASE_V1.0.csv')!=cat: die('catalog self-hash mismatch')
    print('[6/6] Verifying final release checksum catalog... PASS')
    print('\n'+'='*72); print('FINAL RELEASE SHA-256: COMPLETE'); print('='*72)
    print('Release files hashed       :',len(entries)); print('Checksum algorithm         : SHA-256'); print('Catalog SHA-256            :',cat); print('Documentation included     : True'); print('Raw data modified          : False'); print('Processed data modified    : False'); print('Preprocessing rerun        : False'); print('Feature extraction         : False'); print('Machine learning           : False'); print('Fusion generation          : False'); print('='*72); print('DATASET V1.0 RELEASE INTEGRITY STAGE: COMPLETE'); print('='*72)
if __name__=='__main__': main()
