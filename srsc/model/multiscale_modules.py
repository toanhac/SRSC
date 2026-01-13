"""
Multi-scale Feature Modules for Lightweight Auxiliary Heads
============================================================

This module contains:
1. AddCoords / CoordConv: Adds coordinate channels for position awareness
2. LightweightFeatureFusion: Fuses F_16x and F_8x features
3. GlobalContextBlock: Lightweight global context modeling

Reference: Design Specification for Lightweight Multi-scale Auxiliary Heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AddCoords(nn.Module):
    """
    Adds 2 coordinate channels (normalized i, j) to input tensor.
    
    CoordConv paper: https://arxiv.org/abs/1807.03247
    
    Input: [B, C, H, W]
    Output: [B, C+2, H, W] with added (x_coord, y_coord) channels
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.shape
        device = x.device
        dtype = x.dtype
        
        # Create coordinate grids normalized to [-1, 1]
        y_coords = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        
        # Create meshgrid
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Expand to batch size: [B, 1, H, W]
        xx = xx.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        yy = yy.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Concatenate with input
        return torch.cat([x, xx, yy], dim=1)


class CoordConv(nn.Module):
    """
    CoordConv: Conv2d with coordinate channels prepended.
    
    Helps the network learn position-dependent features,
    useful for spatial localization tasks like heatmap prediction.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        self.add_coords = AddCoords()
        # +2 for the coordinate channels
        self.conv = nn.Conv2d(
            in_channels + 2, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_coords(x)
        return self.conv(x)


class LightweightFeatureFusion(nn.Module):
    """
    Lightweight Feature Fusion Module.
    
    Fuses deep features (stride 16) with shallow features (stride 8)
    to restore spatial resolution while maintaining semantic richness.
    
    Design principles:
    - Channel reduction via 1x1 conv (bottleneck)
    - Bilinear upsampling for scale matching
    - Element-wise sum for feature aggregation
    """
    
    def __init__(
        self, 
        deep_channels: int,    # F_16x channels
        shallow_channels: int, # F_8x channels  
        out_channels: int = 128
    ):
        super().__init__()
        
        # Channel reduction (bottleneck design)
        self.deep_reduce = nn.Sequential(
            nn.Conv2d(deep_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.shallow_reduce = nn.Sequential(
            nn.Conv2d(shallow_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # Refinement after fusion
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        f_deep: torch.Tensor,   # [B, C_deep, H/16, W/16]
        f_shallow: torch.Tensor # [B, C_shallow, H/8, W/8]
    ) -> torch.Tensor:
        """
        Returns:
            f_fused: [B, out_channels, H/8, W/8] - fused features at stride 8
        """
        # Channel reduction
        f_deep = self.deep_reduce(f_deep)
        f_shallow = self.shallow_reduce(f_shallow)
        
        # Upsample deep features 2x to match shallow
        f_deep = F.interpolate(
            f_deep, 
            size=f_shallow.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Element-wise sum
        f_fused = f_deep + f_shallow
        
        # Refinement
        f_fused = self.refine(f_fused)
        
        return f_fused


class GlobalContextBlock(nn.Module):
    """
    Global Context (GC) Block for lightweight global context modeling.
    
    Provides global receptive field without the cost of full self-attention.
    
    Mechanism:
    1. Context Modeling: Conv 1x1 + Softmax → global context vector
    2. Transform: Bottleneck FC layers
    3. Broadcast: Add global vector to each pixel
    
    Reference: GCNet (https://arxiv.org/abs/1904.11492)
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()
        
        mid_channels = max(in_channels // reduction_ratio, 16)
        
        # Context modeling: spatial attention to create global vector
        self.context_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
        # Transform: bottleneck to reduce parameters
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
        """
        Args:
            x: [B, C, H, W]
            
        Returns:
            out: [B, C, H, W] - input + global context
        """
        B, C, H, W = x.shape
        
        # Context modeling: compute attention weights
        # [B, 1, H, W] → [B, 1, H*W]
        context_weights = self.context_conv(x).view(B, 1, H * W)
        context_weights = F.softmax(context_weights, dim=-1)
        
        # Weighted sum over spatial dimensions to get global vector
        # x: [B, C, H*W], weights: [B, 1, H*W] → [B, C, 1]
        x_flat = x.view(B, C, H * W)
        global_context = torch.bmm(x_flat, context_weights.transpose(1, 2))
        global_context = global_context.view(B, C, 1, 1)
        
        # Transform
        global_context = self.transform(global_context)
        
        # Broadcast and add
        return x + global_context


# ============================================================================
# Test code
# ============================================================================

if __name__ == '__main__':
    print("Testing Multi-scale Modules")
    print("=" * 60)
    
    batch_size = 2
    
    # Test AddCoords
    print("\n1. AddCoords")
    add_coords = AddCoords()
    x = torch.randn(batch_size, 64, 8, 16)
    out = add_coords(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    assert out.shape == (batch_size, 66, 8, 16), "AddCoords shape mismatch"
    print("   ✓ Coords range:", out[:, -2:].min().item(), "to", out[:, -2:].max().item())
    
    # Test CoordConv
    print("\n2. CoordConv")
    coord_conv = CoordConv(64, 128, kernel_size=3, padding=1)
    x = torch.randn(batch_size, 64, 8, 16)
    out = coord_conv(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    assert out.shape == (batch_size, 128, 8, 16), "CoordConv shape mismatch"
    print("   ✓ Pass")
    
    # Test LightweightFeatureFusion
    print("\n3. LightweightFeatureFusion")
    fusion = LightweightFeatureFusion(
        deep_channels=256,
        shallow_channels=128,
        out_channels=128
    )
    f_deep = torch.randn(batch_size, 256, 4, 8)    # stride 16
    f_shallow = torch.randn(batch_size, 128, 8, 16)  # stride 8
    out = fusion(f_deep, f_shallow)
    print(f"   F_deep: {f_deep.shape}, F_shallow: {f_shallow.shape}")
    print(f"   → Fused: {out.shape}")
    assert out.shape == (batch_size, 128, 8, 16), "Fusion shape mismatch"
    print("   ✓ Pass")
    
    # Test GlobalContextBlock
    print("\n4. GlobalContextBlock")
    gc_block = GlobalContextBlock(in_channels=256, reduction_ratio=4)
    x = torch.randn(batch_size, 256, 8, 16)
    out = gc_block(x)
    print(f"   Input: {x.shape} → Output: {out.shape}")
    assert out.shape == x.shape, "GC Block shape mismatch"
    print("   ✓ Pass")
    
    # Test gradient flow
    print("\n5. Gradient Flow Test")
    x = torch.randn(batch_size, 256, 8, 16, requires_grad=True)
    gc_block = GlobalContextBlock(256)
    out = gc_block(x)
    loss = out.sum()
    loss.backward()
    print(f"   Gradient norm: {x.grad.norm().item():.4f}")
    print("   ✓ Gradients flowing")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
