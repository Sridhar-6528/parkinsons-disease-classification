"""
================================================================================
 DICOM TO NIFTI / PYTORCH CONVERTER UTILITY FOR LONI IDA DOWNLOADS
================================================================================
This script converts downloaded DICOM (.dcm) series folders from LONI IDA / PPMI
into standardized 3D NIfTI (.nii) and PyTorch volume tensor (.pt) files ready for
diagnostic inference.
================================================================================
"""

import os
import sys
import glob
import zipfile
import numpy as np
import torch
import torch.nn.functional as F

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def load_dicom_stack(dicom_dir):
    """
    Reads a directory containing DICOM (.dcm) files, orders them by InstanceNumber or SliceLocation,
    and returns a 3D float32 numpy volume array [Depth, Height, Width].
    """
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required. Install via: pip install pydicom")

    dcm_paths = []
    for root, dirs, files in os.walk(dicom_dir):
        for f in files:
            if f.lower().endswith('.dcm') or f.lower().endswith('.ima') or '.' not in f:
                dcm_paths.append(os.path.join(root, f))

    if not dcm_paths:
        raise FileNotFoundError(f"No DICOM (.dcm) files found in: '{dicom_dir}'")

    slices = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=False)
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise ValueError(f"Could not parse any valid DICOM files in: '{dicom_dir}'")

    # Sort slices by InstanceNumber or SliceLocation or ImagePositionPatient[2]
    def sort_key(s):
        if hasattr(s, 'InstanceNumber'):
            return int(s.InstanceNumber)
        elif hasattr(s, 'SliceLocation'):
            return float(s.SliceLocation)
        elif hasattr(s, 'ImagePositionPatient'):
            return float(s.ImagePositionPatient[2])
        return 0

    slices.sort(key=sort_key)

    # Stack slice pixel arrays into 3D volume [D, H, W]
    vol_list = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        # Apply RescaleSlope and RescaleIntercept if present
        slope = getattr(s, 'RescaleSlope', 1.0)
        intercept = getattr(s, 'RescaleIntercept', 0.0)
        arr = arr * float(slope) + float(intercept)
        vol_list.append(arr)

    vol = np.stack(vol_list, axis=0) # [Depth, Height, Width]
    return vol


def convert_dicom_dir_to_nifti_and_pt(input_dir_or_zip, output_dir="predict_data", base_name="unseen_subject"):
    """
    Converts a DICOM folder or ZIP archive into standardized 3D NIfTI (.nii) and PyTorch (.pt) tensors.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Handle zip files
    temp_dir = None
    if input_dir_or_zip.endswith('.zip'):
        import tempfile
        temp_dir = tempfile.mkdtemp()
        print(f"Extracting ZIP archive '{input_dir_or_zip}'...")
        with zipfile.ZipFile(input_dir_or_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        target_dir = temp_dir
    else:
        target_dir = input_dir_or_zip

    print(f"Reading DICOM series stack from '{target_dir}'...")
    vol_3d = load_dicom_stack(target_dir)
    vol_3d = np.squeeze(vol_3d) # Squeeze any 1-length dimensions to get 3D array [D, H, W]
    print(f"Original DICOM 3D Volume Shape: {vol_3d.shape}")

    # Resample to standard (128, 128, 128) shape
    tensor_vol = torch.from_numpy(vol_3d).unsqueeze(0).unsqueeze(0) # [1, 1, D, H, W]
    if tensor_vol.ndim > 5:
        tensor_vol = tensor_vol.squeeze()
        tensor_vol = tensor_vol.unsqueeze(0).unsqueeze(0)

    resampled = F.interpolate(tensor_vol, size=(128, 128, 128), mode='trilinear', align_corners=False)

    # Z-score normalization
    arr_resampled = resampled.squeeze().numpy()
    mean = np.mean(arr_resampled)
    std = np.std(arr_resampled) + 1e-8
    normed = (arr_resampled - mean) / std
    normed_tensor = torch.from_numpy(normed).unsqueeze(0).unsqueeze(0) # [1, 1, 128, 128, 128]

    # Save PyTorch Tensor (.pt)
    pt_path = os.path.join(output_dir, f"{base_name}.pt")
    torch.save(normed_tensor, pt_path)
    print(f"Saved PyTorch volume tensor to: '{pt_path}'")

    # Save NIfTI File (.nii)
    if HAS_NIBABEL:
        nii_path = os.path.join(output_dir, f"{base_name}.nii")
        nii_img = nib.Nifti1Image(normed.astype(np.float32), np.eye(4))
        nib.save(nii_img, nii_path)
        print(f"Saved NIfTI volume to: '{nii_path}'")

    # Clean up temp dir if created
    if temp_dir and os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("Conversion completed successfully!")
    return pt_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_dicom_to_nii.py <path_to_dicom_folder_or_zip> [output_dir] [base_name]")
        print("Example: python convert_dicom_to_nii.py \"Spect Dataset PD (1).zip\" predict_data/spect Subject_54144_spect")
    else:
        in_path = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "predict_data"
        name = sys.argv[3] if len(sys.argv) > 3 else "converted_dicom"
        convert_dicom_dir_to_nifti_and_pt(in_path, out_dir, name)
