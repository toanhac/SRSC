"""
Relation Query Module (RQM)

At each decoder step, projects the decoder hidden state h_t to a soft
distribution over K relation types (relation query q_rel), then computes
a spatial structural prior by weighting the relation probability maps with
q_rel.  The prior is added (in log-domain) to cross-attention logits,
guiding the decoder to attend to structurally appropriate image regions.

Complementary to ARM/RMC:
  ARM/RMC — WHERE NOT to re-attend  (negative coverage feedback)
  RQM     — WHERE to attend          (positive structural guidance)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class RelationQueryModule(nn.Module):
    """
    Parameters
    ----------
    d_model : int
        Decoder hidden dimension.
    num_relations : int
        Number of spatial relation classes K (default 7).
    """

    def __init__(self, d_model: int, num_relations: int = 7):
        super().__init__()
        self.num_relations = num_relations

        # W_q : D → K  projects hidden state to a relation query
        self.query_proj = nn.Linear(d_model, num_relations)

        # Scalar mixing weight between structural prior and uniform fallback.
        # prior_scale = 0  →  α = sigmoid(0) = 0.5 initially.
        self.prior_scale = nn.Parameter(torch.zeros(1))

        # Zero-init: at step 0, q_rel = softmax(0) = uniform(1/K)
        # → prior ≈ mean(R, over K)  →  weak, non-biasing structural signal.
        nn.init.zeros_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

    def forward(
        self,
        tgt: Tensor,
        relation_probs: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        tgt : [T, B, D]
            Decoder hidden states after self-attention + norm.
        relation_probs : [B, K, H, W]
            Sigmoid-activated relation probability maps (detached).
        memory_key_padding_mask : [B, H*W], optional
            True at padding positions that should be masked to -inf.

        Returns
        -------
        log_prior : [B, T, H*W]
            Additive log-domain structural attention bias.
        """
        T, B, D = tgt.shape
        # relation_probs may have same batch dim as tgt (B_doubled = 2B in bidirectional)
        B_rel, K, H, W = relation_probs.shape
        HW = H * W

        # 1. Relation query: [T, B, K] → softmax over K
        q_rel = torch.softmax(self.query_proj(tgt), dim=-1)   # [T, B, K]

        # Reshape for einsum: [B, T, K] and [B, K, H*W]
        q_rel = q_rel.permute(1, 0, 2)                        # [B, T, K]
        r_flat = relation_probs.view(B_rel, K, HW)             # [B_rel, K, HW]

        # 2. Spatial prior: weighted sum over relation channels
        prior = torch.bmm(q_rel, r_flat)                       # [B_rel, T, HW]

        # 3. Mix with uniform fallback to prevent log(0) = -inf
        #    α starts at 0.5 and is learned; uniform = 1/HW
        alpha = torch.sigmoid(self.prior_scale)
        prior_mixed = (1.0 - alpha) / HW + alpha * prior       # [B_rel, T, HW]

        # 4. Log-domain — additive bias to QK^T before softmax
        log_prior = torch.log(prior_mixed.clamp(min=1e-9))     # [B_rel, T, HW]

        # 5. Mask padding positions with -inf so they cannot be attended
        if memory_key_padding_mask is not None:
            # mask: [B_rel, HW], True = padding
            mask_inf = memory_key_padding_mask.unsqueeze(1).float() * (-1e9)
            log_prior = log_prior + mask_inf                   # [B_rel, T, HW]

        return log_prior
