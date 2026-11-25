import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pretrained.pretrained_swin import PretrainedSwinEncoder
from src.models.cross_attention_fusion import FairClinicalMLP

class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional Cross-Attention module where MRI tokens attend to SPECT tokens and vice versa.
    """
    def __init__(self, dim=768, num_heads=8, drop=0.1):
        super().__init__()
        self.mri_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=drop)
        self.spect_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=drop)
        
        self.norm_mri = nn.LayerNorm(dim)
        self.norm_spect = nn.LayerNorm(dim)

    def forward(self, mri_tokens, spect_tokens):
        # mri_tokens: (B, N, C), spect_tokens: (B, N, C)
        
        # MRI queries SPECT (MRI attends to SPECT)
        mri_cross, _ = self.mri_attn(query=mri_tokens, key=spect_tokens, value=spect_tokens)
        fused_mri = self.norm_mri(mri_tokens + mri_cross)

        # SPECT queries MRI (SPECT attends to MRI)
        spect_cross, _ = self.spect_attn(query=spect_tokens, key=mri_tokens, value=mri_tokens)
        fused_spect = self.norm_spect(spect_tokens + spect_cross)

        return fused_mri, fused_spect

class PretrainedDualSwinCrossAttentionNetwork(nn.Module):
    """
    Dual Pre-trained MONAI SwinUNETR + Bidirectional Cross-Attention Fusion Network.
    """
    def __init__(self, mri_encoder=None, spect_encoder=None, num_classes=3, embed_dim=768, num_heads=8, drop_rate=0.5, use_clinical=False):
        super().__init__()
        self.use_clinical = use_clinical
        if mri_encoder is not None:
            self.mri_encoder = mri_encoder
        else:
            self.mri_encoder = PretrainedSwinEncoder(img_size=(96, 96, 96), num_classes=num_classes)

        if spect_encoder is not None:
            self.spect_encoder = spect_encoder
        else:
            self.spect_encoder = PretrainedSwinEncoder(img_size=(96, 96, 96), num_classes=num_classes)

        self.cross_attn = BidirectionalCrossAttention(dim=embed_dim, num_heads=num_heads, drop=drop_rate)
        
        # Fusion head receives [pooled_fused_mri; pooled_fused_spect; pooled_orig_mri; pooled_orig_spect]
        fusion_dim = embed_dim * 4
        
        if self.use_clinical:
            self.clinical_encoder = FairClinicalMLP(in_features=3, out_dim=32)
            fusion_dim += 32
            
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, mri_vol, spect_vol, clinical_vec=None, return_features=False):
        _, mri_tokens, mri_pooled = self.mri_encoder(mri_vol, return_features=True)
        _, spect_tokens, spect_pooled = self.spect_encoder(spect_vol, return_features=True)

        fused_mri_tokens, fused_spect_tokens = self.cross_attn(mri_tokens, spect_tokens)
        
        fused_mri_pooled = fused_mri_tokens.mean(dim=1)
        fused_spect_pooled = fused_spect_tokens.mean(dim=1)

        feats = [fused_mri_pooled, fused_spect_pooled, mri_pooled, spect_pooled]
        
        if self.use_clinical and clinical_vec is not None:
            clin_pooled = self.clinical_encoder(clinical_vec)
            feats.append(clin_pooled)

        joint_repr = torch.cat(feats, dim=-1)
        logits = self.fusion_head(joint_repr)

        if return_features:
            return logits, joint_repr
        return logits
