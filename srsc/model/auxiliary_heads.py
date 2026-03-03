"""
Auxiliary Task Heads for Multi-Task Learning
=============================================

This module contains the prediction heads for:
1. RelationHead: Predicts multi-label relation map [B, 7, H, W]
   - Uses GlobalContextBlock for long-range dependencies

These heads are attached to the encoder output and trained with
auxiliary losses alongside the main recognition task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GlobalContextBlock(nn.Module):
    """
    Global Context (GC) Block for lightweight global context modeling.

    Mechanism:
    1. Context Modeling: Conv 1x1 + Softmax → global context vector
    2. Transform: Bottleneck FC layers
    3. Broadcast: Add global vector to each pixel

    Reference: GCNet (https://arxiv.org/abs/1904.11492)
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()

        mid_channels = max(in_channels // reduction_ratio, 16)

        self.context_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.LayerNorm([mid_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.context_conv.weight, mode='fan_in')
        nn.init.zeros_(self.context_conv.bias)

        for m in self.transform.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        context_weights = self.context_conv(x).view(B, 1, H * W)
        context_weights = F.softmax(context_weights, dim=-1)

        x_flat = x.view(B, C, H * W)
        global_context = torch.bmm(x_flat, context_weights.transpose(1, 2))
        global_context = global_context.view(B, C, 1, 1)

        global_context = self.transform(global_context)

        return x + global_context


class RelationHead(nn.Module):
    def __init__(self, d_model: int = 256, hidden_dim: int = 128, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        
        self.gc_block = GlobalContextBlock(hidden_dim, reduction_ratio=4)
        
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
        initial_weights = torch.tensor([0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.channel_weights = nn.Parameter(initial_weights)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        x = self.gc_block(x)
        x = self.refine(x)
        return self.output(x)
    
    def compute_guidance(self, relation_map: torch.Tensor) -> torch.Tensor:
        if relation_map.shape[1] == 1:
            return relation_map
        
        weights = F.softplus(self.channel_weights)
        mask = torch.ones_like(weights)
        mask[0] = 0.0
        weights = weights * mask
        weights = weights / (weights.sum() + 1e-8)
        weights = weights.view(1, -1, 1, 1)
        
        num_channels = min(relation_map.shape[1], len(self.channel_weights))
        guidance = (relation_map[:, :num_channels] * weights[:, :num_channels]).sum(dim=1, keepdim=True)
        
        return guidance


def compute_relation_loss(
    pred: torch.Tensor,   # [B, C, H, W]
    target: torch.Tensor, # [B, C, H, W]
    mask: torch.Tensor = None,  # [B, H, W]
    ignore_class_0: bool = True  # Ignore NONE class
) -> torch.Tensor:
    """
    Compute multi-label relation loss using Binary Cross-Entropy.
    
    Each channel is treated as an independent binary classification:
    - Channel k: Is this pixel part of relation k?
    
    This allows multiple channels to be active for the same pixel.
    """
    if ignore_class_0:
        # Skip channel 0 (NONE)
        pred = pred[:, 1:, :, :]
        target = target[:, 1:, :, :]
    
    if mask is not None:
        # Apply mask to ignore padding regions
        mask = mask.unsqueeze(1).expand_as(pred)  # [B, C, H, W]
        
        # Compute BCE only on valid pixels
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        bce = bce * mask
        loss = bce.sum() / mask.sum().clamp(min=1)
        return loss
    else:
        return F.binary_cross_entropy(pred, target)


def get_max_relation_map(relation_pred: torch.Tensor) -> torch.Tensor:
    """
    Get maximum relation probability across all channels.
    
    This is used in Guided Coverage Attention:
        max_k R̂_{k,i} = maximum probability at position i
    
    Args:
        relation_pred: [B, C, H, W] multi-label predictions
        
    Returns:
        max_relation: [B, H*W] flattened max values
    """
    # Take max across channel dimension, ignoring channel 0 (NONE)
    max_relation = relation_pred[:, 1:, :, :].max(dim=1)[0]  # [B, H, W]
    
    # Flatten for attention computation
    B, H, W = max_relation.shape
    return max_relation.view(B, H * W)  # [B, H*W]


if __name__ == '__main__':
    # Test the heads
    batch_size = 2
    H, W, D = 4, 8, 256
    
    features = torch.randn(batch_size, H, W, D)
    
    print("Testing RelationHead")
    print("=" * 50)
    print(f"Input features: {features.shape}")
    
    head = RelationHead(d_model=D)
    relation_pred = head(features)
    
    print(f"\nRelation prediction: {relation_pred.shape}")
    print(f"  Range: [{relation_pred.min():.4f}, {relation_pred.max():.4f}]")
    
    # Test max relation for guided coverage
    max_rel = get_max_relation_map(relation_pred)
    print(f"\nMax relation map: {max_rel.shape}")
    print(f"  For guided coverage attention")
