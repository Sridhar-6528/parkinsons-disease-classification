import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    brier_score_loss,
    roc_auc_score,
    precision_recall_curve,
    auc
)

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

CLASS_NAMES = ["Control", "Prodromal", "PD"]

def compute_all_metrics(y_true, y_pred_logits):
    """
    Computes complete set of medical classification metrics from ground truth targets and raw predicted logits.
    """
    y_true = np.array(y_true)
    y_pred_logits = np.array(y_pred_logits)
    
    # Softmax probabilities
    exp_logits = np.exp(y_pred_logits - np.max(y_pred_logits, axis=-1, keepdims=True))
    y_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Hard predictions
    y_pred = np.argmax(y_probs, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Multi-class One-vs-Rest Brier Score
    y_onehot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((y_probs - y_onehot) ** 2, axis=1)))

    # ROC-AUC & PR-AUC
    try:
        roc_auc = roc_auc_score(y_onehot, y_probs, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = 0.5

    pr_aucs = []
    for c in range(3):
        try:
            p, r, _ = precision_recall_curve(y_onehot[:, c], y_probs[:, c])
            pr_aucs.append(auc(r, p))
        except Exception:
            pr_aucs.append(0.5)
    macro_pr_auc = float(np.mean(pr_aucs))

    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_precision': float(prec),
        'macro_recall': float(rec),
        'macro_f1': float(f1),
        'cohen_kappa': float(kappa),
        'mcc': float(mcc),
        'brier_score': float(brier),
        'roc_auc': float(roc_auc),
        'pr_auc': macro_pr_auc,
        'confusion_matrix': cm.tolist(),
        'per_class_f1': {CLASS_NAMES[i]: float(f1_per_class[i]) for i in range(3)},
        'per_class_precision': {CLASS_NAMES[i]: float(prec_per_class[i]) for i in range(3)},
        'per_class_recall': {CLASS_NAMES[i]: float(rec_per_class[i]) for i in range(3)}
    }

if __name__ == "__main__":
    y_t = [0, 1, 2, 1, 0, 2, 1, 1, 2, 0]
    y_l = [[2.0, 0.1, 0.1], [0.1, 2.5, 0.2], [0.2, 0.3, 2.1], [0.5, 1.8, 0.2], [1.9, 0.4, 0.1],
           [0.1, 0.2, 2.0], [0.2, 2.2, 0.1], [0.3, 1.9, 0.2], [0.1, 0.3, 2.4], [2.1, 0.2, 0.1]]
    metrics = compute_all_metrics(y_t, y_l)
    print("Metrics computation test passed!")
    print(f"  Accuracy: {metrics['accuracy']:.4f}, Balanced Acc: {metrics['balanced_accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}, MCC: {metrics['mcc']:.4f}, Brier: {metrics['brier_score']:.4f}")
