"""
Auxiliary Task Heads for Multi-Task Learning
=============================================

This module contains the prediction heads for:
1. RelationHead: Predicts multi-label relation map [B, 7, H, W]
   - Uses LightweightASPP for multi-scale context
   - Uses GlobalContextBlock for long-range dependencies
   - Returns raw logits (use sigmoid externally when probabilities needed)

These heads are attached to the encoder output and trained with
auxiliary losses alongside the main recognition task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module to guide the network to focus on informative regions.
    Uses channel-wise max and mean pooling followed by a convolution to generate a 2D spatial weight map.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='sigmoid')

    def forward(self, x):
        # x shape: [B, C, H, W]
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        mean_pool = torch.mean(x, dim=1, keepdim=True)    # [B, 1, H, W]
        
        attention = torch.cat([max_pool, mean_pool], dim=1) # [B, 2, H, W]
        attention = self.conv(attention)                    # [B, 1, H, W]
        attention = self.sigmoid(attention)
        
        return x * attention


class LightweightASPP(nn.Module):
    """
    Lightweight Atrous Spatial Pyramid Pooling for multi-scale context.
    
    Three parallel branches capture features at different scales:
    - Branch 1 (1×1): local/point-wise features
    - Branch 2 (3×3, dilation=2): medium-range (superscript/subscript scale)
    - Branch 3 (3×3, dilation=4): long-range (fraction/sqrt structure)
    
    All branches are fused via concatenation + 1×1 projection.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        
        branch_channels = out_channels // 3
        remainder = out_channels - branch_channels * 3
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, 
                      padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels + remainder, kernel_size=3,
                      padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(branch_channels + remainder),
            nn.ReLU(inplace=True),
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return self.fuse(torch.cat([b1, b2, b3], dim=1))


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
    """
    Predicts multi-label structural relation maps from encoder features.
    
    Architecture: ASPP (multi-scale) → GC Block (global) → Refine (local) → Output
    
    Returns raw logits [B, num_classes, H, W].
    Apply torch.sigmoid() externally when probabilities are needed.
    """
    
    def __init__(self, d_model: int = 256, hidden_dim: int = 128, 
                 num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Multi-scale context aggregation
        self.aspp = LightweightASPP(d_model, hidden_dim, dropout=dropout)
        
        # Global context
        self.gc_block = GlobalContextBlock(hidden_dim, reduction_ratio=4)
        
        # Local refinement with residual connection
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Explicit spatial focus before output
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # Output: raw logits (no sigmoid)
        self.output = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        
        # Channel interaction to model relation dependencies (Above<->Below, etc.)

        self.channel_interaction = nn.Sequential(
            nn.Conv2d(num_classes, 14, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(14, num_classes, kernel_size=1, bias=False)
        )
        
        
        
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
        """
        Args:
            features: [B, H, W, D] encoder output (channel-last)
        Returns:
            logits: [B, num_classes, H, W] raw logits
        """
        x = features.permute(0, 3, 1, 2).contiguous()
        x = self.aspp(x)
        x = self.gc_block(x)
        x = x + self.refine(x)  # residual connection
        
        # Apply Spatial Attention
        x = self.spatial_attention(x)
        
        # Generate Logits
        logits = self.output(x)
        
        # Apply Channel Interaction (Residual)
        logits = logits + self.channel_interaction(logits)
        return logits


if __name__ == '__main__':
    batch_size = 2
    H, W, D = 4, 8, 256
    
    features = torch.randn(batch_size, H, W, D)
    
    print("Testing RelationHead")
    print("=" * 50)
    print(f"Input features: {features.shape}")
    
    head = RelationHead(d_model=D)
    logits = head(features)
    probs = torch.sigmoid(logits)
    
    print(f"\nLogits: {logits.shape}")
    print(f"  Range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"Probabilities (after sigmoid): [{probs.min():.4f}, {probs.max():.4f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in head.parameters())
    print(f"\nTotal parameters: {total_params:,}")
