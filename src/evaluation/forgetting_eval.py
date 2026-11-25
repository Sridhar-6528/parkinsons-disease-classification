import os
import sys
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.metrics import compute_all_metrics

def evaluate_model_on_loader(model, dataloader, modality='mri', device='cpu'):
    """
    Evaluates model on given dataloader and modality ('mri', 'spect', or 'multimodal').
    """
    model.eval()
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            if modality == 'multimodal' or modality == 'multimodal_clinical':
                if len(batch[0]) == 3:
                    (mri_x, spect_x, clin_x), labels, _ = batch
                    mri_x = mri_x.to(device)
                    spect_x = spect_x.to(device)
                    clin_x = clin_x.to(device)
                    logits = model(mri_x, spect_x, clinical_vec=clin_x)
                else:
                    (mri_x, spect_x), labels, _ = batch
                    mri_x = mri_x.to(device)
                    spect_x = spect_x.to(device)
                    logits = model(mri_x, spect_x)
            else:
                x, labels, _ = batch
                x = x.to(device)
                logits = model(x)

            all_targets.extend(labels.numpy().tolist())
            all_logits.extend(logits.cpu().numpy().tolist())

    return compute_all_metrics(all_targets, all_logits)

def calculate_backward_transfer(acc_matrix):
    """
    Computes Backward Transfer (BWT) metric from an R matrix of shape (S, S),
    where R[i, j] is the test accuracy on task j after training on session i.
    """
    R = np.array(acc_matrix)
    S = R.shape[0]
    if S <= 1:
        return 0.0
    
    bwt_sum = 0.0
    for j in range(S - 1):
        bwt_sum += (R[S - 1, j] - R[j, j])
    
    return float(bwt_sum / (S - 1))
