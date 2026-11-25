"""
================================================================================
 LOCAL GPU TRAINING LAUNCHER: NOVEL PARKINSON'S DISEASE CLASSIFICATION SYSTEM
 3D ResNet-10 + Bidirectional Cross-Attention + DER++ Continual Learning
================================================================================
Targeted for Local Execution on GTX 1650 (4GB VRAM) / NVIDIA CUDA GPU.
Auto-detects dataset zips, extracted PPMI folders, and Kaggle DAT/MRI folders.
"""

import os
import sys
import zipfile
import io
import csv
import math
import numpy as np
import pandas as pd
from copy import deepcopy

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, brier_score_loss, roc_auc_score, precision_recall_curve, auc

try:
    import pydicom
except ImportError:
    print("Installing pydicom...")
    os.system("pip install pydicom")
    import pydicom

try:
    from monai.networks.nets import resnet10 as monai_resnet10
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from src.data.subject_registry import get_subject_label_mapping, resolve_existing_path
from src.data.dataset import preprocess_and_cache_all
from src.data.clinical_parser import parse_clinical_data
from src.data.dataloader import create_dataloaders
from src.models.swin3d_encoder import Swin3DEncoder
from src.models.cross_attention_fusion import DualSwinCrossAttentionNetwork
from src.models.baselines import ConventionalFusionNetwork
from src.continual.der_buffer import ModalityTaggedBuffer
from src.continual.train_session import train_unimodal_session, train_multimodal_session
from src.evaluation.forgetting_eval import evaluate_model_on_loader


def main():
    print("=" * 80)
    print(" MULTIMODAL NEUROIMAGING (MRI + SPECT) PARKINSON'S DISEASE CLASSIFICATION SYSTEM")
    print(" 3D Swin Transformer + Bidirectional Cross-Attention + DER++ Continual Learning")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Compute Device: {device.upper()}")
    if device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Capacity: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Stage 0: Data Preprocessing & Caching
    print("\n--- STAGE 0: Data Preprocessing & Caching ---")
    preprocess_and_cache_all()
    print("\n--- STAGE 0: Parsing Clinical Tabular Data ---")
    parse_clinical_data()

    # Create stratified subject-level dataloaders (batch_size=2 for 4GB VRAM GTX 1650)
    print("\n--- STAGE 0: Creating Stratified Subject-Level Data Loaders ---")
    mri_train, mri_val, mri_test, splits = create_dataloaders(modality='mri', batch_size=2, seed=42)
    spect_train, spect_val, spect_test, _ = create_dataloaders(modality='spect', batch_size=2, seed=42)
    mm_train, mm_val, mm_test, _ = create_dataloaders(modality='multimodal', batch_size=2, seed=42)
    mm_clin_train, mm_clin_val, mm_clin_test, _ = create_dataloaders(modality='multimodal_clinical', batch_size=2, seed=42)

    train_subs, val_subs, test_subs = splits
    print(f"Dataset Split -> Train: {len(train_subs)} subjects, Val: {len(val_subs)} subjects, Test: {len(test_subs)} subjects")

    buffer = ModalityTaggedBuffer(buffer_capacity=30, num_classes=3)
    results = {}

    checkpoints_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # ---------------------------------------------------------
    # SESSION 1: Unimodal MRI-only Model (Variant 1)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" SESSION 1: Base Session — 3D Swin Transformer (MRI-only)")
    print("=" * 60)
    mri_model = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
    mri_model, _ = train_unimodal_session(mri_model, mri_train, mri_val, modality='mri', epochs=35, buffer=None, device=device)
    mri_metrics = evaluate_model_on_loader(mri_model, mri_test, modality='mri', device=device)
    results['Variant 1: Unimodal MRI-only'] = mri_metrics

    # Save MRI model checkpoint
    torch.save(mri_model.state_dict(), os.path.join(checkpoints_dir, 'mri_swin3d.pt'))
    print("Saved MRI 3D Swin Transformer checkpoint to 'checkpoints/mri_swin3d.pt'")

    # Populate buffer with MRI samples
    buffer.populate_from_loader(mri_model, mri_train, modality='mri', device=device, samples_per_class=8)
    print(f"Session 1 Complete! Buffer size: {len(buffer)} entries.")

    # ---------------------------------------------------------
    # SESSION 2: Unimodal SPECT-only Model (Variant 2)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" SESSION 2: Incremental Session — 3D Swin Transformer (SPECT-only + DER++ Replay)")
    print("=" * 60)
    spect_model = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
    spect_model, _ = train_unimodal_session(spect_model, spect_train, spect_val, modality='spect', epochs=35, buffer=buffer, device=device)
    spect_metrics = evaluate_model_on_loader(spect_model, spect_test, modality='spect', device=device)
    results['Variant 2: Unimodal SPECT-only'] = spect_metrics

    # Save SPECT model checkpoint
    torch.save(spect_model.state_dict(), os.path.join(checkpoints_dir, 'spect_swin3d.pt'))
    print("Saved SPECT 3D Swin Transformer checkpoint to 'checkpoints/spect_swin3d.pt'")

    # Retest MRI performance post-Session 2 to measure retention
    post_s2_mri_metrics = evaluate_model_on_loader(spect_model, mri_test, modality='mri', device=device)
    print(f"Retention Check post-Session 2 -> MRI Accuracy: {post_s2_mri_metrics['accuracy']:.4f}")

    # Populate buffer with SPECT samples
    buffer.populate_from_loader(spect_model, spect_train, modality='spect', device=device, samples_per_class=8)
    print(f"Session 2 Complete! Buffer size: {len(buffer)} entries.")

    # ---------------------------------------------------------
    # ABLATION VARIANT 3: Conventional Multimodal Fusion (Variant 3)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" ABLATION VARIANT 3: Conventional Multimodal Fusion (Concat, Trained from Scratch)")
    print("=" * 60)
    conv_model = ConventionalFusionNetwork(num_classes=3, embed_dim=512, use_clinical=False)
    conv_model, _ = train_multimodal_session(conv_model, mm_train, mm_val, epochs=35, buffer=None, device=device)
    conv_metrics = evaluate_model_on_loader(conv_model, mm_test, modality='multimodal', device=device)
    results['Variant 3: Conventional Fusion (Concat)'] = conv_metrics

    # Save Conventional Fusion checkpoint
    torch.save(conv_model.state_dict(), os.path.join(checkpoints_dir, 'conv_fusion.pt'))

    # ---------------------------------------------------------
    # ABLATION VARIANT 4: Cross-Attention Fusion without DER++ (Variant 4)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" ABLATION VARIANT 4: Cross-Attention Fusion without DER++ (Naive Fine-tuning)")
    print("=" * 60)
    no_der_model = DualSwinCrossAttentionNetwork(mri_encoder=mri_model, spect_encoder=spect_model, use_clinical=False)
    no_der_model, _ = train_multimodal_session(no_der_model, mm_train, mm_val, epochs=35, buffer=None, device=device)
    no_der_metrics = evaluate_model_on_loader(no_der_model, mm_test, modality='multimodal', device=device)
    results['Variant 4: Cross-Attention (No DER++)'] = no_der_metrics

    # Save Cross-Attention without DER++ checkpoint
    torch.save(no_der_model.state_dict(), os.path.join(checkpoints_dir, 'no_der_cross_attn.pt'))

    # ---------------------------------------------------------
    # SESSION 3: Proposed Framework (Cross-Attention + DER++) (Variant 5)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" SESSION 3: Dual 3D Swin Transformer + Cross-Attention + DER++ (Imaging Only)")
    print("=" * 60)
    proposed_model = DualSwinCrossAttentionNetwork(mri_encoder=mri_model, spect_encoder=spect_model, use_clinical=False)
    proposed_model, _ = train_multimodal_session(proposed_model, mm_train, mm_val, epochs=35, buffer=buffer, device=device)
    proposed_metrics = evaluate_model_on_loader(proposed_model, mm_test, modality='multimodal', device=device)
    results['Variant 5: Proposed (Swin + CrossAttn + DER++)'] = proposed_metrics

    # Save Proposed Model checkpoint
    torch.save(proposed_model.state_dict(), os.path.join(checkpoints_dir, 'proposed_cross_attn_fusion.pt'))
    print("Saved Proposed Multimodal Model checkpoint to 'checkpoints/proposed_cross_attn_fusion.pt'")

    # ---------------------------------------------------------
    # SESSION 4 / VARIANT 6: Imaging + Fair Clinical Fusion (Variant 6)
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print(" SESSION 4 / VARIANT 6: Imaging + Fair Clinical Fusion")
    print("=" * 60)
    fair_model = DualSwinCrossAttentionNetwork(mri_encoder=mri_model, spect_encoder=spect_model, use_clinical=True)
    fair_model, _ = train_multimodal_session(fair_model, mm_clin_train, mm_clin_val, epochs=35, buffer=buffer, device=device)
    fair_metrics = evaluate_model_on_loader(fair_model, mm_clin_test, modality='multimodal_clinical', device=device)
    results['Variant 6: Imaging + Fair Clinical Fusion'] = fair_metrics

    # Save Fair Model checkpoint
    torch.save(fair_model.state_dict(), os.path.join(checkpoints_dir, 'fair_clinical_fusion.pt'))
    print("Saved Fair Clinical Fusion checkpoint to 'checkpoints/fair_clinical_fusion.pt'")

    # Print Comprehensive Benchmark Report Table
    print("\n" + "=" * 90)
    print(" COMPREHENSIVE EXPERIMENTAL BENCHMARK REPORT (LOCAL GPU)")
    print("=" * 90)

    summary_rows = []
    for model_name, m in results.items():
        summary_rows.append({
            'Model Variant': model_name,
            'Accuracy': f"{m['accuracy']:.4f}",
            'Bal Acc': f"{m['balanced_accuracy']:.4f}",
            'Macro F1': f"{m['macro_f1']:.4f}",
            'Kappa': f"{m['cohen_kappa']:.4f}",
            'MCC': f"{m['mcc']:.4f}",
            'Brier': f"{m['brier_score']:.4f}",
            'ROC-AUC': f"{m['roc_auc']:.4f}",
            'PR-AUC': f"{m['pr_auc']:.4f}"
        })

    df_summary = pd.DataFrame(summary_rows)
    if HAS_TABULATE:
        print(tabulate(df_summary, headers='keys', tablefmt='github', showindex=False))
    else:
        print(df_summary.to_string(index=False))

    print("\nCatastrophic Forgetting Evaluation:")
    print(f"  Session 1 MRI Test Accuracy: {mri_metrics['accuracy']:.4f}")
    print(f"  Post-Session 2 MRI Retention Accuracy: {post_s2_mri_metrics['accuracy']:.4f}")
    print(f"  Final Proposed Multimodal Accuracy: {proposed_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
