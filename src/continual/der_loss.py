import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        # alpha is expected to be a tensor of class weights
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DERPlusPlusLoss(nn.Module):
    """
    Dark Experience Replay++ (DER++) Loss.
    Combines:
      1. Current Task Focal Loss
      2. Replay Logit Distillation Loss (MSE between current logits on replayed samples and stored logits)
      3. Replay Classification Loss (Focal Loss on replayed samples vs ground-truth targets)
    """
    def __init__(self, alpha=0.5, beta=0.5, class_weights=None):
        super().__init__()
        self.alpha = alpha # Weight for logit distillation loss
        self.beta = beta   # Weight for replay classification loss
        
        self.ce_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        self.mse_loss = nn.MSELoss()

    def forward(self, current_logits, current_targets, replay_logits=None, stored_logits=None, replay_targets=None):
        # Move weights to the correct device
        if self.ce_loss.alpha is not None and self.ce_loss.alpha.device != current_logits.device:
            self.ce_loss.alpha = self.ce_loss.alpha.to(current_logits.device)
            self.ce_loss.ce.weight = self.ce_loss.alpha
            
        loss_ce = self.ce_loss(current_logits, current_targets)
        
        if replay_logits is None or stored_logits is None or replay_targets is None:
            return loss_ce, {'loss_ce': loss_ce.item(), 'loss_distill': 0.0, 'loss_replay_ce': 0.0}

        loss_distill = self.mse_loss(replay_logits, stored_logits)
        loss_replay_ce = self.ce_loss(replay_logits, replay_targets)

        total_loss = loss_ce + (self.alpha * loss_distill) + (self.beta * loss_replay_ce)
        
        return total_loss, {
            'loss_ce': loss_ce.item(),
            'loss_distill': loss_distill.item(),
            'loss_replay_ce': loss_replay_ce.item(),
            'total_loss': total_loss.item()
        }

if __name__ == "__main__":
    der_loss = DERPlusPlusLoss(alpha=0.5, beta=0.5)
    cur_logits = torch.randn(4, 3)
    cur_targets = torch.tensor([0, 1, 2, 1])
    rep_logits = torch.randn(2, 3)
    sto_logits = torch.randn(2, 3)
    rep_targets = torch.tensor([0, 2])

    loss, details = der_loss(cur_logits, cur_targets, rep_logits, sto_logits, rep_targets)
    print("DER++ Loss test passed! Total loss:", loss.item())
    print("  Loss details:", details)
