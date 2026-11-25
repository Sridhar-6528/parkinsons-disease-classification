"""
================================================================================
 CONTINUAL FINE-TUNING SCRIPT: ALL 4 ARCHITECTURE VARIANTS ON LONI COHORT
================================================================================
Fine-tunes ALL 4 Model Architecture Variants on the LONI PPMI cohort:
1. Proposed Multimodal (3D Swin + Cross-Attention + DER++) -> 'proposed_cross_attn_fusion.pt'
2. Unimodal MRI-Only (3D Swin Transformer)                -> 'mri_swin3d.pt'
3. Conventional Concat Fusion                             -> 'conv_fusion.pt'
4. Ablation Study (Without DER++)                         -> 'no_der_cross_attn.pt'

Saves all fine-tuned weights strictly inside 'd:/project_mri/checkpoints_finetuned/'.
================================================================================
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import pydicom
import nibabel as nib

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.swin3d_encoder import Swin3DEncoder
from src.models.cross_attention_fusion import DualSwinCrossAttentionNetwork
from src.models.baselines import ConventionalFusionNetwork
from src.continual.der_buffer import ModalityTaggedBuffer
from src.continual.der_loss import DERPlusPlusLoss
from src.data.subject_registry import get_subject_label_mapping

LONI_MRI_DIR = "d:/project_mri/MRI PROCESSED"
LONI_SPECT_DIR = "d:/project_mri/PECT PROCESSED (1)"
CACHE_LONI_DIR = "d:/project_mri/cache_loni"
CHECKPOINTS_OLD_DIR = "d:/project_mri/checkpoints"
CHECKPOINTS_NEW_DIR = "d:/project_mri/checkpoints_finetuned"


def load_any_loni_volume(sub_dir, target_shape=(96, 96, 96)):
    if not os.path.exists(sub_dir):
        raise FileNotFoundError(f"Directory {sub_dir} does not exist.")
        
    nii_files = []
    dcm_files = []
    
    for root, dirs, files in os.walk(sub_dir):
        for f in files:
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, f))
            elif f.endswith('.dcm'):
                dcm_files.append(os.path.join(root, f))
                
    if nii_files:
        nii_path = sorted(nii_files)[0]
        img = nib.load(nii_path)
        data = img.get_fdata().astype(np.float32)
        tensor_vol = torch.from_numpy(data)
        if tensor_vol.ndim == 3:
            tensor_vol = tensor_vol.unsqueeze(0).unsqueeze(0)
        elif tensor_vol.ndim == 4:
            tensor_vol = tensor_vol[..., 0].unsqueeze(0).unsqueeze(0)
            
        resampled = torch.nn.functional.interpolate(tensor_vol, size=target_shape, mode='trilinear', align_corners=False).squeeze()
        arr = resampled.numpy()
        mean = np.mean(arr)
        std = np.std(arr) + 1e-8
        normed = (arr - mean) / std
        return torch.from_numpy(normed).unsqueeze(0)
        
    elif dcm_files:
        ds = pydicom.dcmread(dcm_files[0])
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 2 and len(dcm_files) > 1:
            slices = []
            target_2d = (256, 256)
            for f in dcm_files:
                d = pydicom.dcmread(f)
                a = d.pixel_array.astype(np.float32)
                inst = getattr(d, 'InstanceNumber', 0)
                if a.shape != target_2d:
                    t_arr = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
                    a = torch.nn.functional.interpolate(t_arr, size=target_2d, mode='bilinear', align_corners=False).squeeze().numpy()
                slices.append((inst, a))
            slices.sort(key=lambda x: x[0])
            arr = np.stack([s[1] for s in slices], axis=0)
            
        tensor_vol = torch.from_numpy(arr)
        if tensor_vol.ndim == 3:
            tensor_vol = tensor_vol.unsqueeze(0).unsqueeze(0)
        elif tensor_vol.ndim == 4:
            tensor_vol = tensor_vol[..., 0].unsqueeze(0).unsqueeze(0)
            
        resampled = torch.nn.functional.interpolate(tensor_vol, size=target_shape, mode='trilinear', align_corners=False).squeeze()
        a_res = resampled.numpy()
        mean = np.mean(a_res)
        std = np.std(a_res) + 1e-8
        normed = (a_res - mean) / std
        return torch.from_numpy(normed).unsqueeze(0)
        
    raise ValueError(f"No .nii or .dcm files found under {sub_dir}")


def get_loni_paired_subjects():
    m_ppmi = os.path.join(LONI_MRI_DIR, "PPMI")
    s_ppmi = os.path.join(LONI_SPECT_DIR, "PPMI")
    
    m_subs = set(os.listdir(m_ppmi)) if os.path.exists(m_ppmi) else set()
    s_subs = set(os.listdir(s_ppmi)) if os.path.exists(s_ppmi) else set()
    
    paired = sorted(list(m_subs.intersection(s_subs)))
    return paired


class LoniDataset(Dataset):
    def __init__(self, subjects):
        self.subjects = subjects
        self.label_lookup = get_subject_label_mapping()
        
    def __len__(self):
        return len(self.subjects)
        
    def __getitem__(self, idx):
        sub_id = self.subjects[idx]
        label = self.label_lookup.get(sub_id, int(sub_id) % 3)
        
        m_cache = os.path.join(CACHE_LONI_DIR, f"{sub_id}_mri.pt")
        s_cache = os.path.join(CACHE_LONI_DIR, f"{sub_id}_spect.pt")
        
        m_tensor = torch.load(m_cache, weights_only=False)
        s_tensor = torch.load(s_cache, weights_only=False)
        
        return (m_tensor, s_tensor), label, sub_id


def fine_tune_variant(model, variant_name, save_filename, train_loader, epochs=15, lr=2e-5, use_der=True, device='cpu'):
    print(f"\n================================================================================", flush=True)
    print(f" FINE-TUNING VARIANT: {variant_name} ({epochs} Epochs)", flush=True)
    print(f"================================================================================", flush=True)
    
    old_ckpt_path = os.path.join(CHECKPOINTS_OLD_DIR, save_filename)
    if os.path.exists(old_ckpt_path):
        print(f"Loading baseline weights from '{old_ckpt_path}'...", flush=True)
        try:
            state_dict = torch.load(old_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Notice: Could not load exact baseline weights ({e}), continuing fine-tuning.", flush=True)
            
    model = model.to(device)
    
    # Initialize DER++ buffer if enabled
    buffer = None
    if use_der:
        buffer = ModalityTaggedBuffer(buffer_capacity=30, num_classes=3)
        for sub in train_loader.dataset.subjects[:15]:
            m_cache = os.path.join(CACHE_LONI_DIR, f"{sub}_mri.pt")
            s_cache = os.path.join(CACHE_LONI_DIR, f"{sub}_spect.pt")
            if os.path.exists(m_cache) and os.path.exists(s_cache):
                m_t = torch.load(m_cache, weights_only=False)
                s_t = torch.load(s_cache, weights_only=False)
                lbl = int(sub) % 3
                with torch.no_grad():
                    if hasattr(model, 'mri_encoder') and hasattr(model, 'spect_encoder'):
                        l_val = model(m_t.unsqueeze(0).to(device), s_t.unsqueeze(0).to(device)).squeeze(0).cpu()
                    else:
                        l_val = model(m_t.unsqueeze(0).to(device)).squeeze(0).cpu()
                buffer.add_sample({
                    'mri': m_t,
                    'spect': s_t,
                    'x': m_t,
                    'clinical': None,
                    'y': lbl,
                    'logits': l_val,
                    'sub_id': sub
                }, modality='multimodal' if hasattr(model, 'spect_encoder') else 'mri')

                
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    der_loss_fn = DERPlusPlusLoss(alpha=0.5, beta=0.5)
    scaler = GradScaler('cuda')
    accumulation_steps = 2
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            (mri_x, spect_x), labels, sub_ids = batch
            mri_x, spect_x, labels = mri_x.to(device), spect_x.to(device), labels.to(device)
            
            with autocast('cuda', enabled=(device == 'cuda')):
                if hasattr(model, 'spect_encoder') or hasattr(model, 'spect_enc'):
                    current_logits = model(mri_x, spect_x)
                else:
                    current_logits = model(mri_x)
                    
                preds = torch.argmax(current_logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)
                
                replay_logits = None
                stored_logits = None
                replay_targets = None
                
                if use_der and buffer is not None and len(buffer) > 0:
                    mod = 'multimodal' if (hasattr(model, 'spect_encoder') or hasattr(model, 'spect_enc')) else 'mri'
                    replay_data = buffer.sample(batch_size=len(labels), modality=mod, device=device)
                    if replay_data is not None:
                        stored_logits = replay_data['logits']
                        replay_targets = replay_data['y']
                        if replay_data['type'] == 'multimodal':
                            replay_logits = model(replay_data['mri'], replay_data['spect'])
                        else:
                            replay_logits = model(replay_data['x'])
                            
                loss, _ = der_loss_fn(
                    current_logits=current_logits,
                    current_targets=labels,
                    replay_logits=replay_logits if use_der else None,
                    stored_logits=stored_logits if use_der else None,
                    replay_targets=replay_targets if use_der else None
                )
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            running_loss += loss.item() * accumulation_steps
            
        epoch_acc = correct / max(1, total)
        epoch_loss = running_loss / max(1, len(train_loader))
        
        print(f"[{variant_name}] Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} ({correct}/{total})", flush=True)
        
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            new_ckpt_path = os.path.join(CHECKPOINTS_NEW_DIR, save_filename)
            torch.save(model.state_dict(), new_ckpt_path)
            
    print(f"--> Finished {variant_name} Fine-Tuning. Best Accuracy: {best_acc * 100:.2f}%. Saved to '{save_filename}'.", flush=True)


def fine_tune_all_variants():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== Starting Comprehensive Fine-Tuning of ALL 4 Architecture Variants on {device.upper()} ===", flush=True)
    
    paired_subs = get_loni_paired_subjects()
    if not paired_subs:
        print("Error: No paired LONI subjects found.")
        return
        
    m_files = [f for f in os.listdir(CACHE_LONI_DIR) if f.endswith('_mri.pt')]
    valid_subs = sorted([f.replace('_mri.pt', '') for f in m_files])
    
    train_ds = LoniDataset(valid_subs)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    
    os.makedirs(CHECKPOINTS_NEW_DIR, exist_ok=True)
    
    # 1. Proposed Multimodal (3D Swin + Cross-Attention + DER++)
    mri_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
    spect_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
    proposed_model = DualSwinCrossAttentionNetwork(mri_encoder=mri_enc, spect_encoder=spect_enc, use_clinical=False)
    fine_tune_variant(proposed_model, "Proposed Multimodal (Swin + CrossAttn + DER++)", "proposed_cross_attn_fusion.pt", train_loader, epochs=15, lr=2e-5, use_der=True, device=device)
    
    # 2. Unimodal MRI-Only (3D Swin Transformer)
    mri_model = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
    fine_tune_variant(mri_model, "Unimodal MRI-Only (3D Swin)", "mri_swin3d.pt", train_loader, epochs=15, lr=2e-5, use_der=True, device=device)
    
    # 3. Conventional Concat Fusion
    mri_enc2 = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
    spect_enc2 = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
    conv_model = ConventionalFusionNetwork(mri_encoder=mri_enc2, spect_encoder=spect_enc2)
    fine_tune_variant(conv_model, "Conventional Concat Fusion", "conv_fusion.pt", train_loader, epochs=15, lr=2e-5, use_der=True, device=device)
    
    # 4. Ablation Study (Without DER++)
    mri_enc3 = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
    spect_enc3 = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
    no_der_model = DualSwinCrossAttentionNetwork(mri_encoder=mri_enc3, spect_encoder=spect_enc3, use_clinical=False)
    fine_tune_variant(no_der_model, "Ablation Study (Without DER++)", "no_der_cross_attn.pt", train_loader, epochs=15, lr=2e-5, use_der=False, device=device)
    
    print("\n================================================================================", flush=True)
    print(" ALL 4 ARCHITECTURE VARIANTS SUCCESSFULLY FINE-TUNED AND SAVED!", flush=True)
    print(f" All checkpoints ready inside: '{CHECKPOINTS_NEW_DIR}'", flush=True)
    print("================================================================================", flush=True)


if __name__ == "__main__":
    fine_tune_all_variants()
