import os
import sys
import pandas as pd
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.subject_registry import get_subject_label_mapping

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def parse_clinical_data(excel_dir=None, cache_dir=None, mri_zip=None, spect_zip=None):
    """
    Parses clinical CSVs to extract raw baseline tabular features.
    ONLY extracting: [NP1RTOT, NP2PTOT, MCATOT]
    Saves to clinical_raw.pt dict without imputation or normalization.
    """
    excel_dir = excel_dir or os.path.join(PROJECT_ROOT, "excel")
    cache_dir = cache_dir or os.path.join(PROJECT_ROOT, "cache_processed")
    subject_map = get_subject_label_mapping(mri_zip, spect_zip)
    subjects = sorted(list(subject_map.keys()))
    
    # Initialize feature dict with NaNs
    clinical_features = {sub: {'np1': np.nan, 'np2': np.nan, 'moca': np.nan} for sub in subjects}
    
    # 1. NP1RTOT (UPDRS Part I - Non-motor)
    try:
        df1 = pd.read_csv(os.path.join(excel_dir, "MDS-UPDRS_Part_I_10Aug2026.csv"), low_memory=False)
        # Handle baseline (BL or SC)
        df1_bl = df1[df1['EVENT_ID'].isin(['BL', 'SC'])]
        # Sort so BL comes before SC, then drop duplicates by PATNO keeping first (BL)
        df1_bl = df1_bl.sort_values('EVENT_ID').drop_duplicates('PATNO')
        for _, row in df1_bl.iterrows():
            sub = str(row['PATNO'])
            if sub in clinical_features and not pd.isna(row.get('NP1RTOT')):
                clinical_features[sub]['np1'] = float(row['NP1RTOT'])
    except Exception as e:
        print(f"Warning: UPDRS 1 parse error: {e}")

    # 2. NP2PTOT (UPDRS Part II - Motor ADL, patient-reported)
    try:
        df2 = pd.read_csv(os.path.join(excel_dir, "MDS_UPDRS_Part_II__Patient_Questionnaire_10Aug2026.csv"), low_memory=False)
        df2_bl = df2[df2['EVENT_ID'].isin(['BL', 'SC'])]
        df2_bl = df2_bl.sort_values('EVENT_ID').drop_duplicates('PATNO')
        for _, row in df2_bl.iterrows():
            sub = str(row['PATNO'])
            if sub in clinical_features and not pd.isna(row.get('NP2PTOT')):
                clinical_features[sub]['np2'] = float(row['NP2PTOT'])
    except Exception as e:
        print(f"Warning: UPDRS 2 parse error: {e}")

    # 3. MCATOT (MoCA)
    try:
        df_moca = pd.read_csv(os.path.join(excel_dir, "Montreal_Cognitive_Assessment__MoCA__10Aug2026.csv"), low_memory=False)
        df_moca_bl = df_moca[df_moca['EVENT_ID'].isin(['BL', 'SC'])]
        df_moca_bl = df_moca_bl.sort_values('EVENT_ID').drop_duplicates('PATNO')
        for _, row in df_moca_bl.iterrows():
            sub = str(row['PATNO'])
            if sub in clinical_features and not pd.isna(row.get('MCATOT')):
                clinical_features[sub]['moca'] = float(row['MCATOT'])
    except Exception as e:
        print(f"Warning: MoCA parse error: {e}")

    # Save to dictionary mapping subject_id to array of shape (3,) with possible NaNs
    clinical_tensors = {}
    for sub in subjects:
        d = clinical_features[sub]
        arr = np.array([d['np1'], d['np2'], d['moca']], dtype=np.float32)
        clinical_tensors[sub] = arr
        
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, "clinical_raw.pt")
    torch.save(clinical_tensors, out_path)
    print(f"Saved RAW clinical features for {len(subjects)} subjects to {out_path}")

if __name__ == "__main__":
    parse_clinical_data()
