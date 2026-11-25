import os
import sys
import zipfile
import io
import pydicom
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.subject_registry import (
    get_subject_label_mapping,
    resolve_existing_path,
    ARCHIVE_NIFTI_DIR,
    DEFAULT_MRI_PATHS,
    DEFAULT_SPECT_PATHS
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache_processed")


def _read_dcm_from_source(source, file_path):
    if isinstance(source, zipfile.ZipFile):
        return source.read(file_path)
    else:
        with open(os.path.join(source, file_path), 'rb') as f:
            return f.read()


def _get_dcms_for_subject(source, subject_id):
    if isinstance(source, zipfile.ZipFile):
        return [
            f for f in source.namelist()
            if f.endswith('.dcm') and (f'/{subject_id}/' in f or f.startswith(f'{subject_id}/'))
        ]
    else:
        dcms = []
        subject_dir = os.path.join(source, 'PPMI', subject_id)
        if not os.path.exists(subject_dir):
            subject_dir = os.path.join(source, subject_id)

        if os.path.exists(subject_dir):
            for root, dirs, files in os.walk(subject_dir):
                for f in files:
                    if f.endswith('.dcm'):
                        dcms.append(os.path.relpath(os.path.join(root, f), source))
        return dcms


def load_nifti_volume(file_path, target_shape=(96, 96, 96)):
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required to load NIfTI files. Please install via pip install nibabel.")

    img = nib.load(file_path)
    data = img.get_fdata().astype(np.float32)

    tensor_vol = torch.from_numpy(data)
    if tensor_vol.ndim == 3:
        tensor_vol = tensor_vol.unsqueeze(0).unsqueeze(0)
    elif tensor_vol.ndim == 4:
        tensor_vol = tensor_vol[..., 0].unsqueeze(0).unsqueeze(0)

    resampled = F.interpolate(tensor_vol, size=target_shape, mode='trilinear', align_corners=False).squeeze()

    arr = resampled.numpy()
    mean = np.mean(arr)
    std = np.std(arr) + 1e-8
    normed = (arr - mean) / std

    return torch.from_numpy(normed).unsqueeze(0)  # (1, D, H, W)


def load_and_preprocess_mri(source_mri, subject_id, target_shape=(96, 96, 96)):
    if os.path.exists(ARCHIVE_NIFTI_DIR) and "_" in subject_id:
        prefix, idx_str = subject_id.split("_")
        idx = int(idx_str)
        cat_map = {"ctrl": "control-30", "prod": "Prodromal-30", "pd": "PD-30"}
        cat_folder = cat_map.get(prefix)
        if cat_folder:
            nii_path = os.path.join(ARCHIVE_NIFTI_DIR, "Mri", cat_folder, f"Copy of {idx}.nii")
            if os.path.exists(nii_path):
                return load_nifti_volume(nii_path, target_shape=target_shape)

    dcms = _get_dcms_for_subject(source_mri, subject_id)
    if not dcms:
        raise ValueError(f"No MRI DCM found for subject {subject_id}")

    vol = None
    slices = []
    target_2d = (256, 256)

    for f in dcms:
        ds = pydicom.dcmread(io.BytesIO(_read_dcm_from_source(source_mri, f)))
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3:
            vol = arr
            break
        elif arr.ndim == 2:
            inst = getattr(ds, 'InstanceNumber', 0)
            if arr.shape != target_2d:
                t_arr = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                arr = F.interpolate(t_arr, size=target_2d, mode='bilinear', align_corners=False).squeeze().numpy()
            slices.append((inst, arr))

    if vol is None:
        if not slices:
            raise ValueError(f"Could not extract valid image slices for MRI subject {subject_id}")
        slices.sort(key=lambda x: x[0])
        vol = np.stack([s[1] for s in slices], axis=0)

    tensor_vol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    resampled = F.interpolate(tensor_vol, size=target_shape, mode='trilinear', align_corners=False).squeeze()

    arr = resampled.numpy()
    mean = np.mean(arr)
    std = np.std(arr) + 1e-8
    normed = (arr - mean) / std

    return torch.from_numpy(normed).unsqueeze(0)


def load_and_preprocess_spect(source_spect, subject_id, target_shape=(96, 96, 96)):
    if os.path.exists(ARCHIVE_NIFTI_DIR) and "_" in subject_id:
        prefix, idx_str = subject_id.split("_")
        idx = int(idx_str)
        cat_map = {"ctrl": "control-30", "prod": "prodromal-30", "pd": "PD-30"}
        cat_folder = cat_map.get(prefix)
        if cat_folder:
            s_dir = os.path.join(ARCHIVE_NIFTI_DIR, "Datscan", cat_folder)
            if os.path.exists(s_dir):
                files = sorted([f for f in os.listdir(s_dir) if f.endswith('.nii')])
                if idx <= len(files):
                    nii_path = os.path.join(s_dir, files[idx - 1])
                    return load_nifti_volume(nii_path, target_shape=target_shape)

    dcms = _get_dcms_for_subject(source_spect, subject_id)
    if not dcms:
        raise ValueError(f"No SPECT DCM found for subject {subject_id}")

    bytes_data = _read_dcm_from_source(source_spect, dcms[0])
    ds = pydicom.dcmread(io.BytesIO(bytes_data))
    vol = ds.pixel_array.astype(np.float32)

    tensor_vol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
    resampled = F.interpolate(tensor_vol, size=target_shape, mode='trilinear', align_corners=False).squeeze()

    arr = resampled.numpy()
    mean = np.mean(arr)
    std = np.std(arr) + 1e-8
    normed = (arr - mean) / std

    return torch.from_numpy(normed).unsqueeze(0)


def preprocess_and_cache_all(mri_zip=None, spect_zip=None, target_shape=(96, 96, 96), modality='multimodal', force_match=False):
    if mri_zip is None:
        mri_zip = resolve_existing_path(DEFAULT_MRI_PATHS)
    if spect_zip is None:
        spect_zip = resolve_existing_path(DEFAULT_SPECT_PATHS)

    os.makedirs(CACHE_DIR, exist_ok=True)
    subject_map = get_subject_label_mapping(mri_zip, spect_zip, modality=modality, force_match=force_match)
    subjects = sorted(list(subject_map.keys()))

    print(f"Starting volume preprocessing & caching for {len(subjects)} subjects into '{CACHE_DIR}'...")

    from contextlib import contextmanager
    @contextmanager
    def open_source(path):
        if os.path.isfile(path) and path.endswith('.zip'):
            with zipfile.ZipFile(path, 'r') as z:
                yield z
        else:
            yield path

    with open_source(mri_zip) as zm, open_source(spect_zip) as zs:
        for idx, sub_id in enumerate(subjects):
            mri_cache_path = os.path.join(CACHE_DIR, f"{sub_id}_mri.pt")
            spect_cache_path = os.path.join(CACHE_DIR, f"{sub_id}_spect.pt")

            if not os.path.exists(mri_cache_path):
                if modality != 'spect':
                    try:
                        mri_tensor = load_and_preprocess_mri(zm, sub_id, target_shape=target_shape)
                        torch.save(mri_tensor, mri_cache_path)
                    except Exception as e:
                        print(f"Failed to cache MRI for {sub_id}: {e}")

            if not os.path.exists(spect_cache_path):
                if modality != 'mri' and (force_match or modality == 'spect' or modality == 'multimodal'):
                    try:
                        spect_tensor = load_and_preprocess_spect(zs, sub_id, target_shape=target_shape)
                        torch.save(spect_tensor, spect_cache_path)
                    except Exception as e:
                        pass

            if (idx + 1) % 25 == 0 or (idx + 1) == len(subjects):
                print(f"  Processed {idx + 1}/{len(subjects)} subjects...")

    print("Volume preprocessing and caching complete!")


class NeuroimagingDataset(Dataset):
    def __init__(self, subject_ids, subject_map, modality='mri', transform=None, clinical_dict=None):
        self.subject_ids = subject_ids
        self.subject_map = subject_map
        self.modality = modality
        self.transform = transform
        self.clinical_dict = clinical_dict

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sub_id = self.subject_ids[idx]
        label = self.subject_map[sub_id]

        if self.modality == 'mri':
            mri_path = os.path.join(CACHE_DIR, f"{sub_id}_mri.pt")
            mri_tensor = torch.load(mri_path, weights_only=False, map_location='cpu')
            if self.transform is not None:
                mri_tensor = self.transform(mri_tensor)

            if self.clinical_dict is not None and sub_id in self.clinical_dict:
                clinical_vec = self.clinical_dict[sub_id]
                return (mri_tensor, clinical_vec), label, sub_id

            return mri_tensor, label, sub_id

        elif self.modality == 'spect':
            spect_path = os.path.join(CACHE_DIR, f"{sub_id}_spect.pt")
            spect_tensor = torch.load(spect_path, weights_only=False, map_location='cpu')
            if self.transform is not None:
                spect_tensor = self.transform(spect_tensor)
            return spect_tensor, label, sub_id

        elif self.modality == 'multimodal':
            mri_path = os.path.join(CACHE_DIR, f"{sub_id}_mri.pt")
            spect_path = os.path.join(CACHE_DIR, f"{sub_id}_spect.pt")

            mri_tensor = torch.load(mri_path, weights_only=False, map_location='cpu')
            spect_tensor = torch.load(spect_path, weights_only=False, map_location='cpu')


            if self.transform is not None:
                mri_tensor = self.transform(mri_tensor)
                spect_tensor = self.transform(spect_tensor)

            if self.clinical_dict is not None and sub_id in self.clinical_dict:
                clinical_vec = self.clinical_dict[sub_id]
                return (mri_tensor, spect_tensor, clinical_vec), label, sub_id

            return (mri_tensor, spect_tensor), label, sub_id

        else:
            raise ValueError(f"Unknown modality '{self.modality}'")


if __name__ == "__main__":
    preprocess_and_cache_all()
