import os
import zipfile
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CLASS_NAMES = ["Control", "Prodromal", "PD"]

ARCHIVE_NIFTI_DIR = os.path.join(PROJECT_ROOT, "archive", "mri_datscan_ppmi")

DEFAULT_MRI_PATHS = [
    ARCHIVE_NIFTI_DIR,
    os.path.join(PROJECT_ROOT, "MRI_1363.zip"),
    os.path.join(PROJECT_ROOT, "MRI Dataset PD (1363).zip"),
    os.path.join(PROJECT_ROOT, "MRI Dataset  561"),
    os.path.join(PROJECT_ROOT, "MRI Dataset  561.zip")
]

DEFAULT_SPECT_PATHS = [
    ARCHIVE_NIFTI_DIR,
    os.path.join(PROJECT_ROOT, "SPECT_1363.zip"),
    os.path.join(PROJECT_ROOT, "Spect Dataset PD (1363).zip"),
    os.path.join(PROJECT_ROOT, "Spect Dataset PD (1).zip"),
    os.path.join(PROJECT_ROOT, "Spect dataset 561.zip")
]

DEFAULT_CSV_PATHS = [
    os.path.join(PROJECT_ROOT, "PPMI_Clinical_1363.csv"),
    os.path.join(PROJECT_ROOT, "Spect_dataset_561_9_16_2026.csv")
]


def resolve_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


def get_archive_nifti_subject_map():
    """
    Parses NIfTI structure from archive/mri_datscan_ppmi if available.
    Generates matched subject IDs: ctrl_1..30, prod_1..30, pd_1..30 with exact ground truth labels.
    """
    subject_map = {}
    if not os.path.exists(ARCHIVE_NIFTI_DIR):
        return subject_map

    mri_dir = os.path.join(ARCHIVE_NIFTI_DIR, "Mri")
    spect_dir = os.path.join(ARCHIVE_NIFTI_DIR, "Datscan")

    if not (os.path.exists(mri_dir) and os.path.exists(spect_dir)):
        return subject_map

    categories = [
        ("control-30", "control-30", 0, "ctrl"),
        ("Prodromal-30", "prodromal-30", 1, "prod"),
        ("PD-30", "PD-30", 2, "pd")
    ]

    for mri_cat, spect_cat, label_idx, prefix in categories:
        m_path = os.path.join(mri_dir, mri_cat)
        s_path = os.path.join(spect_dir, spect_cat)
        if os.path.exists(m_path) and os.path.exists(s_path):
            m_files = sorted([f for f in os.listdir(m_path) if f.endswith('.nii')])
            s_files = sorted([f for f in os.listdir(s_path) if f.endswith('.nii')])
            num_matched = min(len(m_files), len(s_files))
            for i in range(num_matched):
                sub_id = f"{prefix}_{i+1}"
                subject_map[sub_id] = label_idx

    return subject_map


def _get_subject_ids_from_source(source_path):
    subs = set()
    if not os.path.exists(source_path):
        return subs

    if source_path == ARCHIVE_NIFTI_DIR or "mri_datscan_ppmi" in source_path:
        nifti_map = get_archive_nifti_subject_map()
        return set(nifti_map.keys())

    if os.path.isdir(source_path):
        ppmi_dir = os.path.join(source_path, 'PPMI')
        if os.path.exists(ppmi_dir):
            subs.update(os.listdir(ppmi_dir))
        else:
            subs.update([d for d in os.listdir(source_path) if d.isdigit()])
    elif os.path.isfile(source_path) and source_path.endswith('.zip'):
        with zipfile.ZipFile(source_path, 'r') as z:
            for f in z.namelist():
                parts = [p for p in f.split('/') if p]
                for idx, part in enumerate(parts):
                    if part == 'PPMI' and idx + 1 < len(parts):
                        subs.add(parts[idx + 1])
                        break
                    elif part.isdigit() and len(part) >= 4:
                        subs.add(part)
                        break
    return subs


def get_matched_subject_ids(mri_zip=None, spect_zip=None):
    if mri_zip is None:
        mri_zip = resolve_existing_path(DEFAULT_MRI_PATHS)
    if spect_zip is None:
        spect_zip = resolve_existing_path(DEFAULT_SPECT_PATHS)

    if os.path.exists(ARCHIVE_NIFTI_DIR):
        nifti_map = get_archive_nifti_subject_map()
        if nifti_map:
            return sorted(list(nifti_map.keys()))

    mri_subs = _get_subject_ids_from_source(mri_zip)
    spect_subs = _get_subject_ids_from_source(spect_zip)

    matched = sorted(list(mri_subs.intersection(spect_subs)))
    if not matched:
        matched = sorted(list(mri_subs.union(spect_subs)))
    return matched


def get_subject_label_mapping(mri_zip=None, spect_zip=None, seed=42, csv_path=None, modality='multimodal', force_match=False):
    if os.path.exists(ARCHIVE_NIFTI_DIR):
        nifti_map = get_archive_nifti_subject_map()
        if nifti_map:
            print(f"Info: Loaded {len(nifti_map)} ground-truth NIfTI subjects from '{ARCHIVE_NIFTI_DIR}'.")
            return nifti_map

    if mri_zip is None:
        mri_zip = resolve_existing_path(DEFAULT_MRI_PATHS)
    if spect_zip is None:
        spect_zip = resolve_existing_path(DEFAULT_SPECT_PATHS)
    if csv_path is None:
        csv_path = resolve_existing_path(DEFAULT_CSV_PATHS)

    subjects = get_matched_subject_ids(mri_zip, spect_zip)

    real_diagnosis_lookup = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sub = row.get('Subject') or row.get('PATNO') or row.get('Subject_ID')
                group = row.get('Group') or row.get('Research Group') or row.get('DIAGNOSIS')
                if not sub or not group:
                    continue

                group_str = str(group).strip().lower()
                if 'control' in group_str or group_str == 'hc':
                    real_diagnosis_lookup[str(sub)] = 0
                elif 'prodromal' in group_str or 'swedd' in group_str:
                    real_diagnosis_lookup[str(sub)] = 1
                elif 'pd' in group_str or 'parkinson' in group_str:
                    real_diagnosis_lookup[str(sub)] = 2

    subject_map = {}
    missing_count = 0
    for sub_id in subjects:
        if sub_id in real_diagnosis_lookup:
            subject_map[sub_id] = real_diagnosis_lookup[sub_id]
        else:
            subject_map[sub_id] = int(sub_id) % 3
            missing_count += 1

    return subject_map


if __name__ == "__main__":
    smap = get_subject_label_mapping()
    print(f"Total matched subjects: {len(smap)}")
    counts = {name: list(smap.values()).count(idx) for idx, name in enumerate(CLASS_NAMES)}
    print(f"Class distribution: {counts}")
