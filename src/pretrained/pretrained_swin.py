import os
import sys
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

class PretrainedSwinEncoder(nn.Module):
    """
    Wrapper around MONAI's SwinUNETR to extract the SwinViT encoder and use it for classification.
    """
    def __init__(self, img_size=(96, 96, 96), in_chans=1, num_classes=3, pretrained_weights_path="models/model_swinvit.pt", freeze_encoder=False):
        super().__init__()
        
        # Initialize full SwinUNETR (feature_size=48 is standard for the pre-trained weights)
        self.swin_unetr = SwinUNETR(
            in_channels=in_chans,
            out_channels=14,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            spatial_dims=3
        )
        
        # Load pre-trained weights into the encoder if available
        if os.path.exists(pretrained_weights_path):
            print(f"Loading pre-trained weights from {pretrained_weights_path}")
            weight = torch.load(pretrained_weights_path, weights_only=False, map_location='cpu')
            self.swin_unetr.load_from(weights=weight)
        else:
            print(f"Warning: Pre-trained weights not found at {pretrained_weights_path}. Initializing from scratch.")
            
        self.encoder = self.swin_unetr.swinViT
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # The output of the encoder at the bottleneck is of dimension feature_size * 2**4 = 48 * 16 = 768
        embed_dim = 768
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward_features(self, x):
        # The encoder returns a tuple of hidden states: (x0, x1, x2, x3, x4)
        hidden_states = self.encoder(x)
        bottleneck = hidden_states[-1] # Shape: (B, 768, 3, 3, 3)
        
        B, C, D, H, W = bottleneck.shape
        # Flatten spatial dimensions into N tokens
        tokens = bottleneck.view(B, C, -1).transpose(1, 2) # Shape: (B, N, C) -> (B, 27, 768)
        
        # Global average pooling over token sequence
        pooled = tokens.mean(dim=1) # Shape: (B, C)
        
        pooled = self.norm(pooled)
        return tokens, pooled

    def forward(self, x, return_features=False):
        tokens, pooled = self.forward_features(x)
        x = self.dropout(pooled)
        logits = self.head(x)
        
        if return_features:
            return logits, tokens, pooled
        return logits

if __name__ == "__main__":
    model = PretrainedSwinEncoder()
    dummy_x = torch.randn(2, 1, 96, 96, 96)
    logits = model(dummy_x)
    print("Dummy input shape:", dummy_x.shape)
    print("Logits shape:", logits.shape)
