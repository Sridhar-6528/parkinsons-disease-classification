"""
================================================================================
 INFERENCE & PREDICTION ENGINE: PARKINSON'S DISEASE CLASSIFICATION
 3D Swin Transformer + Bidirectional Cross-Attention + DER++ Continual Learning
================================================================================

This script performs diagnostic inference on user-provided 3D MRI and 3D SPECT
volume files (NIfTI .nii / .nii.gz / DICOM .dcm / PyTorch .pt / 2D images .png, .jpg)
as specified in the project architecture.

Target Classification Output (3-Class):
  - Class 0: Healthy Control (Control)
  - Class 1: Prodromal Parkinson's (Prodromal)
  - Class 2: Diagnosed Parkinson's Disease (PD)
================================================================================
"""

import os
import sys
import argparse
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.swin3d_encoder import Swin3DEncoder
from src.models.cross_attention_fusion import DualSwinCrossAttentionNetwork
from src.models.baselines import ConventionalFusionNetwork

CLASS_NAMES = ['Control (Healthy)', 'Prodromal Parkinson\'s', 'Parkinson\'s Disease (PD)']
CLASS_KEYS = ['Control', 'Prodromal', 'PD']
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_FINETUNED_DIR = os.path.join(PROJECT_ROOT, "checkpoints_finetuned")


def resolve_checkpoint_path(filename, custom_ckpt=None, ckpt_version='fine_tuned'):
    if custom_ckpt:
        return custom_ckpt
    if ckpt_version == 'baseline':
        return os.path.join(CHECKPOINT_DIR, filename)
    finetuned_path = os.path.join(CHECKPOINT_FINETUNED_DIR, filename)
    if os.path.exists(finetuned_path):
        return finetuned_path
    return os.path.join(CHECKPOINT_DIR, filename)




def preprocess_2d_image(file_source, target_shape=(96, 96, 96)):
    """
    Converts a 2D image (.png, .jpg, .jpeg, .bmp, .tiff) into a 3D volume tensor [1, 1, 96, 96, 96].
    """
    if not HAS_PIL:
        raise ImportError("PIL (Pillow) is required for 2D image support.")
    
    if isinstance(file_source, (str, bytes, io.BytesIO)):
        img = Image.open(file_source if isinstance(file_source, str) else io.BytesIO(file_source)).convert('L')
    else:
        raise ValueError("Invalid 2D image input source.")

    arr = np.array(img, dtype=np.float32)
    tensor_2d = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    vol = F.interpolate(tensor_2d, size=target_shape, mode='trilinear', align_corners=False)
    
    arr_vol = vol.squeeze().numpy()
    mean = np.mean(arr_vol)
    std = np.std(arr_vol) + 1e-8
    normed = (arr_vol - mean) / std
    return torch.from_numpy(normed).unsqueeze(0).unsqueeze(0)


def preprocess_dicom_file(file_source, target_shape=(96, 96, 96)):
    """
    Loads a DICOM file (.dcm) or DICOM directory into a 3D volume tensor [1, 1, 96, 96, 96].
    """
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required for DICOM files.")

    if isinstance(file_source, str) and os.path.isdir(file_source):
        from convert_dicom_to_nii import load_dicom_stack
        arr = load_dicom_stack(file_source)
    else:
        ds = pydicom.dcmread(file_source if isinstance(file_source, str) else io.BytesIO(file_source))
        arr = ds.pixel_array.astype(np.float32)

    if arr.ndim == 2:
        tensor_vol = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif arr.ndim == 3:
        tensor_vol = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    else:
        tensor_vol = torch.from_numpy(arr)

    resampled = F.interpolate(tensor_vol, size=target_shape, mode='trilinear', align_corners=False).squeeze().numpy()
    mean = np.mean(arr_resample := resampled)
    std = np.std(arr_resample) + 1e-8
    normed = (arr_resample - mean) / std
    return torch.from_numpy(normed).unsqueeze(0).unsqueeze(0)


def preprocess_nifti_volume(file_source, target_shape=(96, 96, 96)):
    """
    Loads and standardizes a 3D NIfTI (.nii / .nii.gz) volume into shape [1, 1, 96, 96, 96].
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for loading NIfTI files.")

    if isinstance(file_source, str):
        img = nib.load(file_source)
    else:
        fh = nib.FileHolder(fileobj=io.BytesIO(file_source))
        img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})

    data = img.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]

    mean = np.mean(data)
    std = np.std(data) + 1e-8
    normed_data = (data - mean) / std

    tensor_vol = torch.from_numpy(normed_data).unsqueeze(0).unsqueeze(0)
    if tensor_vol.shape[2:] != target_shape:
        resampled = F.interpolate(tensor_vol, size=target_shape, mode='trilinear', align_corners=False)
    else:
        resampled = tensor_vol

    return resampled


def preprocess_pt_volume(file_source, target_shape=(96, 96, 96)):
    """
    Loads a preprocessed PyTorch tensor file (.pt).
    """
    if isinstance(file_source, str):
        tensor_vol = torch.load(file_source, weights_only=False, map_location='cpu')
    else:
        tensor_vol = torch.load(io.BytesIO(file_source), weights_only=False, map_location='cpu')


    if tensor_vol.ndim == 3:
        tensor_vol = tensor_vol.unsqueeze(0).unsqueeze(0)
    elif tensor_vol.ndim == 4:
        tensor_vol = tensor_vol.unsqueeze(0)

    if tensor_vol.shape[2:] != target_shape:
        tensor_vol = F.interpolate(tensor_vol, size=target_shape, mode='trilinear', align_corners=False)

    return tensor_vol


def preprocess_dicom_zip_or_dir(file_source, target_shape=(96, 96, 96)):
    """
    Seamlessly extracts a ZIP file containing DICOM (.dcm) files (or reads a DICOM directory),
    handles both multi-frame 3D DICOM files and multi-file 2D DICOM series,
    orders slices by InstanceNumber/SliceLocation/ImagePositionPatient,
    resamples to target_shape [1, 1, 96, 96, 96], and z-score normalizes.
    """
    import zipfile
    import tempfile
    import shutil

    temp_dir = None
    if isinstance(file_source, str) and file_source.lower().endswith('.zip'):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(file_source, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        scan_dir = temp_dir
    elif isinstance(file_source, (bytes, io.BytesIO)):
        bio = file_source if isinstance(file_source, io.BytesIO) else io.BytesIO(file_source)
        if zipfile.is_zipfile(bio):
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(bio, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            scan_dir = temp_dir
        else:
            scan_dir = file_source
    elif isinstance(file_source, str) and os.path.isdir(file_source):
        scan_dir = file_source
    else:
        scan_dir = file_source

    try:
        if not HAS_PYDICOM:
            raise ImportError("pydicom is required to parse DICOM datasets.")

        dcm_paths = []
        if isinstance(scan_dir, str) and os.path.exists(scan_dir):
            for root, dirs, files in os.walk(scan_dir):
                for f in files:
                    if f.lower().endswith(('.dcm', '.ima')) or '.' not in f:
                        dcm_paths.append(os.path.join(root, f))

        if not dcm_paths:
            raise FileNotFoundError("No DICOM (.dcm) files found inside the uploaded ZIP archive.")

        slices = []
        is_raw_spect = False
        for p in dcm_paths:
            if '_br_raw_' in os.path.basename(p).lower():
                is_raw_spect = True
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=False)
                slices.append(ds)
            except Exception:
                continue

        if not slices:
            raise ValueError("Could not parse valid DICOM slices from the uploaded ZIP archive.")

        # Check if DICOM is a multi-frame 3D array (e.g. single DICOM file containing all frames)
        first_arr = slices[0].pixel_array.astype(np.float32)
        slope = float(getattr(slices[0], 'RescaleSlope', 1.0))
        intercept = float(getattr(slices[0], 'RescaleIntercept', 0.0))
        first_arr = first_arr * slope + intercept

        if first_arr.ndim >= 3:
            vol_3d = np.squeeze(first_arr)
            if vol_3d.ndim > 3:
                vol_3d = vol_3d[0]
        else:
            def sort_key(s):
                if hasattr(s, 'InstanceNumber'):
                    return int(s.InstanceNumber)
                elif hasattr(s, 'SliceLocation'):
                    return float(s.SliceLocation)
                elif hasattr(s, 'ImagePositionPatient'):
                    return float(s.ImagePositionPatient[2])
                return 0

            slices.sort(key=sort_key)

            vol_list = []
            for s in slices:
                arr = s.pixel_array.astype(np.float32)
                slp = float(getattr(s, 'RescaleSlope', 1.0))
                icp = float(getattr(s, 'RescaleIntercept', 0.0))
                vol_list.append(arr * slp + icp)

            vol_3d = np.squeeze(np.stack(vol_list, axis=0))

        tensor_vol = torch.from_numpy(vol_3d).float().unsqueeze(0).unsqueeze(0)
        while tensor_vol.ndim > 5:
            tensor_vol = tensor_vol.squeeze(0)

        resampled = F.interpolate(tensor_vol, size=target_shape, mode='trilinear', align_corners=False)
        arr_resampled = resampled.squeeze().numpy()

        mean = np.mean(arr_resampled)
        std = np.std(arr_resampled) + 1e-8
        normed = (arr_resampled - mean) / std

        return torch.from_numpy(normed).unsqueeze(0).unsqueeze(0)

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def preprocess_universal_zip(file_source, target_shape=(96, 96, 96)):
    """
    Seamlessly extracts ZIP archive and auto-detects internal contents:
    1. If NIfTI (.nii / .nii.gz) files are inside -> Loads via NIfTI loader.
    2. If DICOM (.dcm / .ima) files are inside -> Loads via DICOM loader.
    3. If PyTorch (.pt) tensor files are inside -> Loads via PyTorch loader.
    """
    import zipfile
    import tempfile
    import shutil

    temp_dir = None
    if isinstance(file_source, str) and file_source.lower().endswith('.zip'):
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(file_source, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        scan_dir = temp_dir
    elif isinstance(file_source, (bytes, io.BytesIO)):
        bio = file_source if isinstance(file_source, io.BytesIO) else io.BytesIO(file_source)
        if zipfile.is_zipfile(bio):
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(bio, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            scan_dir = temp_dir
        else:
            scan_dir = file_source
    elif isinstance(file_source, str) and os.path.isdir(file_source):
        scan_dir = file_source
    else:
        scan_dir = file_source

    try:
        nii_files = []
        dcm_files = []
        pt_files = []

        if isinstance(scan_dir, str) and os.path.exists(scan_dir):
            for root, dirs, files in os.walk(scan_dir):
                for f in files:
                    fl = f.lower()
                    full_p = os.path.join(root, f)
                    if fl.endswith(('.nii', '.nii.gz')):
                        nii_files.append(full_p)
                    elif fl.endswith(('.dcm', '.ima')) or '.' not in f:
                        dcm_files.append(full_p)
                    elif fl.endswith('.pt'):
                        pt_files.append(full_p)

        if nii_files:
            return preprocess_nifti_volume(nii_files[0], target_shape)
        if dcm_files:
            return preprocess_dicom_zip_or_dir(scan_dir, target_shape)
        if pt_files:
            return preprocess_pt_volume(pt_files[0], target_shape)

        raise FileNotFoundError("No supported 3D volume files (.nii, .nii.gz, .dcm, .pt) found inside the ZIP archive.")

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def load_input_volume(file_source, filename_hint=None, target_shape=(96, 96, 96)):
    """
    Auto-detects file format (.nii, .nii.gz, .pt, .dcm, .zip, 2D img) and returns normalized 3D volume [1, 1, 96, 96, 96].
    Automatically extracts DICOM and NIfTI ZIP archives in the background!
    """
    import zipfile

    path_or_name = file_source if isinstance(file_source, str) else (filename_hint or "")
    path_or_name_lower = path_or_name.lower()

    if path_or_name_lower.endswith('.zip') or (isinstance(file_source, str) and os.path.isdir(file_source)):
        return preprocess_universal_zip(file_source, target_shape)
    elif path_or_name_lower.endswith('.pt'):
        return preprocess_pt_volume(file_source, target_shape)
    elif path_or_name_lower.endswith('.nii') or path_or_name_lower.endswith('.nii.gz'):
        return preprocess_nifti_volume(file_source, target_shape)
    elif path_or_name_lower.endswith('.dcm'):
        return preprocess_dicom_file(file_source, target_shape)
    elif any(path_or_name_lower.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']):
        return preprocess_2d_image(file_source, target_shape)
    else:
        if isinstance(file_source, str) and zipfile.is_zipfile(file_source):
            return preprocess_universal_zip(file_source, target_shape)
        try:
            return preprocess_universal_zip(file_source, target_shape)
        except Exception:
            try:
                return preprocess_pt_volume(file_source, target_shape)
            except Exception:
                try:
                    return preprocess_nifti_volume(file_source, target_shape)
                except Exception:
                    try:
                        return preprocess_dicom_file(file_source, target_shape)
                    except Exception:
                        return preprocess_2d_image(file_source, target_shape)


def preprocess_clinical_vector(updrs=0.0, moca=26.0, age=65.0, gender=1.0):
    """
    Normalizes clinical tabular variables (Age, Gender, UPDRS III, MoCA).
    Returns normalized tensor of shape [1, 3].
    """
    norm_age = (float(age) - 65.0) / 10.0
    norm_gender = float(gender)
    norm_updrs = float(updrs) / 50.0
    norm_moca = float(moca) / 30.0

    clin_vec = torch.tensor([[norm_updrs, norm_moca, norm_age]], dtype=torch.float32)
    return clin_vec


def load_checkpoint_weights(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: '{ckpt_path}'.")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def render_slice_b64(slice_arr, title="Axial View"):
    """
    Renders 2D numpy slice array to base64 encoded PNG string.
    """
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0d1117')
    ax.imshow(slice_arr, cmap='magma' if 'spect' in title.lower() else 'gray')
    ax.set_title(title, color='#38bdf8', fontsize=10, pad=6)
    ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def calculate_entropy(probs):
    """
    Computes Shannon Decision Entropy H = -sum(p * log2(p)).
    """
    eps = 1e-8
    probs = np.clip(probs, eps, 1.0)
    return float(-np.sum(probs * np.log2(probs)))


def predict(mri_path=None, spect_path=None, updrs=0.0, moca=26.0, age=65.0, gender=1,
            model_type='proposed', custom_ckpt=None, ground_truth=None,
            mri_filename=None, spect_filename=None, ckpt_version='fine_tuned'):
    """
    Main Diagnostic Inference Entry Point.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Determine Model Architecture & Checkpoint
    if model_type == 'proposed':
        ckpt_path = resolve_checkpoint_path("proposed_cross_attn_fusion.pt", custom_ckpt, ckpt_version=ckpt_version)
        if not mri_path or not spect_path:
            raise ValueError("Proposed multimodal model requires BOTH MRI and SPECT scans!")

        mri_vol = load_input_volume(mri_path, filename_hint=mri_filename).to(device)
        spect_vol = load_input_volume(spect_path, filename_hint=spect_filename).to(device)

        mri_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
        spect_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
        model = DualSwinCrossAttentionNetwork(mri_encoder=mri_enc, spect_encoder=spect_enc, use_clinical=False).to(device)
        model = load_checkpoint_weights(model, ckpt_path, device)

        with torch.no_grad():
            logits = model(mri_vol, spect_vol)

    elif model_type == 'clinical':
        ckpt_path = resolve_checkpoint_path("fair_clinical_fusion.pt", custom_ckpt, ckpt_version=ckpt_version)
        if not mri_path or not spect_path:
            raise ValueError("Clinical fusion model requires BOTH MRI and SPECT scans!")

        mri_vol = load_input_volume(mri_path, filename_hint=mri_filename).to(device)
        spect_vol = load_input_volume(spect_path, filename_hint=spect_filename).to(device)
        clin_vec = preprocess_clinical_vector(updrs, moca, age, gender).to(device)

        mri_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
        spect_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
        model = DualSwinCrossAttentionNetwork(mri_encoder=mri_enc, spect_encoder=spect_enc, use_clinical=True).to(device)
        model = load_checkpoint_weights(model, ckpt_path, device)

        with torch.no_grad():
            logits = model(mri_vol, spect_vol, clinical_vec=clin_vec)

    elif model_type == 'mri_only':
        ckpt_path = resolve_checkpoint_path("mri_swin3d.pt", custom_ckpt, ckpt_version=ckpt_version)
        if not mri_path:
            raise ValueError("Unimodal MRI model requires an MRI volume file!")

        mri_vol = load_input_volume(mri_path, filename_hint=mri_filename).to(device)
        spect_vol = None
        model = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512).to(device)
        model = load_checkpoint_weights(model, ckpt_path, device)

        with torch.no_grad():
            logits = model(mri_vol)

    elif model_type == 'spect_only':
        ckpt_path = resolve_checkpoint_path("spect_swin3d.pt", custom_ckpt, ckpt_version=ckpt_version)

        if not spect_path:
            raise ValueError("Unimodal SPECT model requires a SPECT volume file!")

        spect_vol = load_input_volume(spect_path, filename_hint=spect_filename).to(device)
        mri_vol = None
        model = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512).to(device)
        model = load_checkpoint_weights(model, ckpt_path, device)

        with torch.no_grad():
            logits = model(spect_vol)

    elif model_type == 'conv_fusion':
        ckpt_path = resolve_checkpoint_path("conv_fusion.pt", custom_ckpt)
        if not mri_path or not spect_path:
            raise ValueError("Conventional Fusion model requires BOTH MRI and SPECT scans!")

        mri_vol = load_input_volume(mri_path, filename_hint=mri_filename).to(device)
        spect_vol = load_input_volume(spect_path, filename_hint=spect_filename).to(device)

        mri_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
        spect_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
        model = ConventionalFusionNetwork(mri_encoder=mri_enc, spect_encoder=spect_enc).to(device)
        model = load_checkpoint_weights(model, ckpt_path, device)

        with torch.no_grad():
            logits = model(mri_vol, spect_vol)

    elif model_type == 'no_der':
        ckpt_path = resolve_checkpoint_path("no_der_cross_attn.pt", custom_ckpt)

        if not mri_path or not spect_path:
            raise ValueError("Ablation model requires BOTH MRI and SPECT scans!")

        mri_vol = load_input_volume(mri_path, filename_hint=mri_filename).to(device)
        spect_vol = load_input_volume(spect_path, filename_hint=spect_filename).to(device)

        mri_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
        spect_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
        model = DualSwinCrossAttentionNetwork(mri_encoder=mri_enc, spect_encoder=spect_enc, use_clinical=False).to(device)
        model = load_checkpoint_weights(model, ckpt_path, device)

        with torch.no_grad():
            logits = model(mri_vol, spect_vol)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")

    # 2. Compute Probabilities & Metrics
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    short_class = CLASS_KEYS[predicted_idx]
    confidence = float(probs[predicted_idx]) * 100.0
    entropy = calculate_entropy(probs)

    # 3. Render Middle Slices (Index 64/128)
    mri_slice_b64 = None
    spect_slice_b64 = None

    if 'mri_vol' in locals() and mri_vol is not None:
        slice_arr = mri_vol.squeeze().cpu().numpy()[64]
        mri_slice_b64 = render_slice_b64(slice_arr, title="Structural MRI (Axial Slice #64)")

    if 'spect_vol' in locals() and spect_vol is not None:
        slice_arr = spect_vol.squeeze().cpu().numpy()[64]
        spect_slice_b64 = render_slice_b64(slice_arr, title="DaTscan SPECT (Axial Slice #64)")

    # 4. Compute Live Verification Metrics if Ground Truth supplied
    live_metrics = None
    if ground_truth is not None and str(ground_truth).isdigit():
        gt_idx = int(ground_truth)
        is_correct = (predicted_idx == gt_idx)
        live_metrics = {
            'ground_truth_class': CLASS_NAMES[gt_idx],
            'is_correct': is_correct,
            'live_accuracy': 100.0 if is_correct else 0.0,
            'target_precision': float(probs[gt_idx]) * 100.0,
            'live_recall': 100.0 if is_correct else 0.0
        }

    return {
        'predicted_class': short_class,
        'full_predicted_class': predicted_class,
        'predicted_idx': predicted_idx,
        'confidence': confidence,
        'entropy': entropy,
        'probabilities': {CLASS_KEYS[i]: float(probs[i]) * 100.0 for i in range(len(CLASS_KEYS))},
        'mri_slice_b64': mri_slice_b64,
        'spect_slice_b64': spect_slice_b64,
        'live_metrics': live_metrics
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Parkinson's Multimodal Inference Engine (3D Swin + DER++)")
    parser.add_argument('--mri', type=str, default=None, help="Path to 3D MRI volume file (.nii, .nii.gz, .pt, .dcm, .png)")
    parser.add_argument('--spect', type=str, default=None, help="Path to 3D SPECT volume file (.nii, .nii.gz, .pt, .dcm, .png)")
    parser.add_argument('--updrs', type=float, default=0.0, help="UPDRS III Motor Score (e.g. 0 to 50)")
    parser.add_argument('--moca', type=float, default=26.0, help="MoCA Cognitive Score (e.g. 0 to 30)")
    parser.add_argument('--age', type=float, default=65.0, help="Patient Age (years)")
    parser.add_argument('--gender', type=int, default=1, help="Patient Gender (0=Female, 1=Male)")
    parser.add_argument('--ground_truth', type=int, default=None, choices=[0, 1, 2], help="Optional ground truth label (0=Control, 1=Prodromal, 2=PD)")
    parser.add_argument('--model_type', type=str, default='proposed', choices=['proposed', 'clinical', 'mri_only', 'spect_only', 'conv_fusion', 'no_der'],
                        help="Model variant to use for prediction")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to custom model checkpoint file (.pt)")
    parser.add_argument('--cli', action='store_true', help="Force CLI execution instead of launching GUI")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if (args.mri is None and args.spect is None) and not args.cli:
        print("Launching Interactive Diagnostic Desktop GUI App...")
        try:
            from gui_app import main as launch_gui
            launch_gui()
        except Exception as e:
            print(f"Could not launch GUI ({e}). Falling back to cached sample demo...")
            predict_dir = os.path.join(PROJECT_ROOT, "predict_data")
            sample_mri = os.path.join(predict_dir, "mri", "Subject_05_Parkinson_Disease_mri.pt")
            sample_spect = os.path.join(predict_dir, "spect", "Subject_05_Parkinson_Disease_spect.pt")
            if os.path.exists(sample_mri) and os.path.exists(sample_spect):
                predict(mri_path=sample_mri, spect_path=sample_spect, model_type='proposed')
    else:
        res = predict(
            mri_path=args.mri,
            spect_path=args.spect,
            updrs=args.updrs,
            moca=args.moca,
            age=args.age,
            gender=args.gender,
            model_type=args.model_type,
            custom_ckpt=args.checkpoint,
            ground_truth=args.ground_truth
        )

        print("\n" + "=" * 80)
        print(" DIAGNOSTIC INFERENCE RESULTS REPORT")
        print("=" * 80)
        print(f"  PREDICTED DIAGNOSIS : {res['full_predicted_class'].upper()}")
        print(f"  CONFIDENCE SCORE    : {res['confidence']:.2f} %")
        print(f"  DECISION ENTROPY    : {res['entropy']:.4f} bits")
        print("-" * 80)
        print("  Class Probabilities:")
        for key, prob_val in res['probabilities'].items():
            print(f"    - {key:<20}: {prob_val:6.2f} %")
        print("=" * 80)

        if res['live_metrics']:
            lm = res['live_metrics']
            print("\n" + "=" * 80)
            print(" LIVE PREDICTION VERIFICATION METRICS")
            print("=" * 80)
            print(f"  GROUND TRUTH TARGET : {lm['ground_truth_class'].upper()}")
            print(f"  LIVE ACCURACY       : {lm['live_accuracy']:.0f} %  ({'MATCH' if lm['is_correct'] else 'MISMATCH'})")
            print(f"  TARGET PRECISION PROB: {lm['target_precision']:.2f} %")
            print(f"  TARGET RECALL SCORE : {lm['live_recall']:.0f} %")
            print("=" * 80)
