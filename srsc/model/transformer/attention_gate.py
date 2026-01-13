import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class AttentionGate(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.85)
    
    def forward(
        self,
        spatial_map: Optional[Tensor],
        relation_map: Optional[Tensor],
        coverage: Tensor,
        h: int,
        w: int,
    ) -> Tensor:
        b = coverage.shape[0]
        
        features = []
        
        if spatial_map is not None:
            spatial_resized = self._resize_map(spatial_map, h, w)
            spatial_flat = spatial_resized.view(b, -1, 1)
            features.append(spatial_flat)
        else:
            features.append(torch.zeros(b, h * w, 1, device=coverage.device))
        
        if relation_map is not None:
            if relation_map.shape[1] > 1:
                relation_max = relation_map[:, 1:, :, :].max(dim=1, keepdim=True)[0]
            else:
                relation_max = relation_map
            relation_resized = self._resize_map(relation_max, h, w)
            relation_flat = relation_resized.view(b, -1, 1)
            features.append(relation_flat)
        else:
            features.append(torch.zeros(b, h * w, 1, device=coverage.device))
        
        coverage_flat = coverage.view(b, -1, 1)
        features.append(coverage_flat)
        
        gate_input = torch.cat(features, dim=2)
        gate = torch.sigmoid(self.gate_net(gate_input))
        
        return gate.squeeze(2)
    
    def _resize_map(self, feature_map: Tensor, h: int, w: int) -> Tensor:
        if feature_map.shape[2:] != (h, w):
            feature_map = F.interpolate(feature_map, size=(h, w), mode='bilinear', align_corners=False)
        return feature_map
