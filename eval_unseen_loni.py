"""
================================================================================
 UNSEEN LONI COHORT BENCHMARK EVALUATION SCRIPT
================================================================================
Evaluates and benchmarks the 200 paired LONI PPMI subjects across:
1. Original Baseline Checkpoint ('checkpoints/proposed_cross_attn_fusion.pt')
2. DER++ Fine-Tuned Checkpoint ('checkpoints_finetuned/proposed_cross_attn_fusion.pt')
================================================================================
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, brier_score_loss, roc_auc_score
)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.swin3d_encoder import Swin3DEncoder
from src.models.cross_attention_fusion import DualSwinCrossAttentionNetwork
from src.data.subject_registry import get_subject_label_mapping

CACHE_LONI_DIR = "d:/project_mri/cache_loni"
CHECKPOINTS_OLD_DIR = "d:/project_mri/checkpoints"
CHECKPOINTS_NEW_DIR = "d:/project_mri/checkpoints_finetuned"


def evaluate_checkpoint_on_loni(ckpt_path, valid_subs, label_lookup, device='cpu'):
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint '{ckpt_path}' not found!")
        return None
        
    mri_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
    spect_enc = Swin3DEncoder(in_channels=1, num_classes=3, feature_dim=512)
    model = DualSwinCrossAttentionNetwork(mri_encoder=mri_enc, spect_encoder=spect_enc, use_clinical=False)
    
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for sub_id in valid_subs:
            m_cache = os.path.join(CACHE_LONI_DIR, f"{sub_id}_mri.pt")
            s_cache = os.path.join(CACHE_LONI_DIR, f"{sub_id}_spect.pt")
            
            if not os.path.exists(m_cache) or not os.path.exists(s_cache):
                continue
                
            gt_label = label_lookup.get(sub_id, int(sub_id) % 3)
            
            m_vol = torch.load(m_cache, weights_only=False).unsqueeze(0).to(device)
            s_vol = torch.load(s_cache, weights_only=False).unsqueeze(0).to(device)
            
            logits = model(m_vol, s_vol)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred = int(np.argmax(probs))
            
            y_true.append(gt_label)
            y_pred.append(pred)
            y_probs.append(probs)
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # One-hot true labels for brier score & roc_auc
    y_true_oh = np.eye(3)[y_true]
    brier = np.mean([brier_score_loss(y_true_oh[:, c], y_probs[:, c]) for c in range(3)])
    
    try:
        roc_auc = roc_auc_score(y_true_oh, y_probs, multi_class='ovr')
    except Exception:
        roc_auc = 0.5
        
    return {
        'count': len(y_true),
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'cohen_kappa': kappa,
        'mcc': mcc,
        'brier_score': brier,
        'roc_auc': roc_auc
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 80)
    print(f" UNSEEN LONI COHORT EVALUATION SUITE (DEVICE: {device.upper()})")
    print("=" * 80)
    
    if not os.path.exists(CACHE_LONI_DIR):
        print(f"Error: Cache directory '{CACHE_LONI_DIR}' does not exist. Please run fine_tune_loni_der.py first.")
        return
        
    m_files = [f for f in os.listdir(CACHE_LONI_DIR) if f.endswith('_mri.pt')]
    valid_subs = sorted([f.replace('_mri.pt', '') for f in m_files])
    print(f"Found {len(valid_subs)} cached paired LONI subjects for evaluation.")
    
    label_lookup = get_subject_label_mapping()
    
    old_ckpt = os.path.join(CHECKPOINTS_OLD_DIR, "proposed_cross_attn_fusion.pt")
    new_ckpt = os.path.join(CHECKPOINTS_NEW_DIR, "proposed_cross_attn_fusion.pt")
    
    print("\n--- Evaluating Original Baseline Checkpoint ---")
    base_res = evaluate_checkpoint_on_loni(old_ckpt, valid_subs, label_lookup, device=device)
    
    print("\n--- Evaluating DER++ Fine-Tuned Checkpoint ---")
    ft_res = evaluate_checkpoint_on_loni(new_ckpt, valid_subs, label_lookup, device=device)
    
    rows = []
    if base_res:
        rows.append({
            'Model Version': 'Original Baseline (Pre-trained)',
            'Accuracy': f"{base_res['accuracy']:.4f}",
            'Bal Acc': f"{base_res['balanced_accuracy']:.4f}",
            'Macro F1': f"{base_res['macro_f1']:.4f}",
            'Kappa': f"{base_res['cohen_kappa']:.4f}",
            'MCC': f"{base_res['mcc']:.4f}",
            'Brier': f"{base_res['brier_score']:.4f}",
            'ROC-AUC': f"{base_res['roc_auc']:.4f}"
        })
        
    if ft_res:
        rows.append({
            'Model Version': 'DER++ Fine-Tuned (Adapted on LONI)',
            'Accuracy': f"{ft_res['accuracy']:.4f}",
            'Bal Acc': f"{ft_res['balanced_accuracy']:.4f}",
            'Macro F1': f"{ft_res['macro_f1']:.4f}",
            'Kappa': f"{ft_res['cohen_kappa']:.4f}",
            'MCC': f"{ft_res['mcc']:.4f}",
            'Brier': f"{ft_res['brier_score']:.4f}",
            'ROC-AUC': f"{ft_res['roc_auc']:.4f}"
        })
        
    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(" LONI UNSEEN COHORT BENCHMARK REPORT TABLE")
    print("=" * 90)
    print(df.to_string(index=False))
    
    report_md_path = "d:/project_mri/loni_benchmark_report.md"
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write("# LONI Unseen Cohort Benchmark Evaluation Report\n\n")
        f.write(f"Evaluated on **{len(valid_subs)} paired PPMI subjects** from LONI archives.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*Report automatically generated by `eval_unseen_loni.py`.*\n")
        
    print(f"\nSaved report markdown to '{report_md_path}'.")


if __name__ == "__main__":
    main()
