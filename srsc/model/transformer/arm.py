import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm1d
from typing import Optional


class MaskBatchNorm2d(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = BatchNorm1d(num_features)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = rearrange(x, "b d h w -> b h w d")
        mask = mask.squeeze(1)
        not_mask = ~mask
        flat_x = x[not_mask, :]
        if flat_x.numel() > 0:
            flat_x = self.bn(flat_x)
            x[not_mask, :] = flat_x
        x = rearrange(x, "b h w d -> b d h w")
        return x


class RelationModulator(nn.Module):
    """
    Computes a per-channel spatial gate from predicted relation probability maps.

    For each coverage channel (attention head), learns which combination of
    relation types (horizontal, above, below, superscript, subscript, inside)
    should boost coverage sensitivity at each spatial position.

    Gate equation:
        gate[h,w] = sigmoid(W_rel · relation_probs[h,w])  ∈ [0, 1]^coverage_chs
        coverage_modulated = coverage * (1 + gate)

    This differs from naive concatenation: relation information multiplicatively
    scales coverage rather than being treated as independent input features.
    """

    def __init__(self, num_relation_classes: int, coverage_chs: int):
        super().__init__()
        self.proj = nn.Conv2d(num_relation_classes, coverage_chs, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, relation_probs: Tensor) -> Tensor:
        """
        Args:
            relation_probs: [B*T, n_rel, H, W]  (already repeated over time)
        Returns:
            gate: [B*T, coverage_chs, H, W]  in (0, 1) range
        """
        return torch.sigmoid(self.proj(relation_probs))


class AttentionRefinementModule(nn.Module):
    """
    Coverage-based attention bias module (ARM).

    BTTR baseline (num_relation_classes=0):
        Computes cumulative cross- and/or self-attention coverage maps,
        then projects them to a per-head spatial bias added to cross-attention
        logits before softmax.

    Relation-Modulated Coverage / RMC (num_relation_classes>0):
        Before the Conv5x5 projection, the coverage maps are multiplicatively
        scaled by a learned gate derived from the RelationHead's predicted
        relation probabilities:

            gate = sigmoid(Conv1x1(relation_probs))
            coverage_modulated = coverage * (1 + gate)

        Structural regions (e.g., fractions, subscripts) boost their own
        coverage sensitivity, while background regions remain unaffected.
        Zero-initialisation of the gate projection ensures the module starts
        as pure BTTR coverage and learns the relation modulation gradually.
    """

    def __init__(
        self,
        nhead: int,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        num_relation_classes: int = 0,
    ):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage

        self.coverage_chs = (2 if cross_coverage and self_coverage else 1) * nhead

        self.relation_modulator: Optional[RelationModulator] = None
        if num_relation_classes > 0:
            self.relation_modulator = RelationModulator(num_relation_classes, self.coverage_chs)

        self.conv = nn.Conv2d(self.coverage_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
        self.post_norm = MaskBatchNorm2d(nhead)

    def forward(
        self,
        prev_attn: Tensor,
        key_padding_mask: Tensor,
        h: int,
        curr_attn: Tensor,
        relation_probs: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            prev_attn:          [B*nhead, T, HW]  cross-attention from previous layer
            key_padding_mask:   [B, HW]            True = padding
            h:                  int                spatial height of feature map
            curr_attn:          [B*nhead, T, HW]  self-attention from current layer
            relation_probs:     [B, n_rel, H, W]  sigmoid outputs of RelationHead, or None
        Returns:
            coverage bias:      [B*nhead, T, HW]  subtracted from attention logits
        """
        t = curr_attn.shape[1]
        b = key_padding_mask.shape[0]

        mask = repeat(key_padding_mask, "b (h w) -> (b t) () h w", h=h, t=t)

        curr_attn = rearrange(curr_attn, "(b n) t l -> b n t l", n=self.nhead)
        prev_attn = rearrange(prev_attn, "(b n) t l -> b n t l", n=self.nhead)

        attns = []
        if self.cross_coverage:
            attns.append(prev_attn)
        if self.self_coverage:
            attns.append(curr_attn)
        attns = torch.cat(attns, dim=1)

        # Cumulative coverage: how much each position has been attended to so far
        attns = attns.cumsum(dim=2) - attns
        coverage = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)

        # Relation-Modulated Coverage: multiplicatively gate coverage by structural priors
        if self.relation_modulator is not None and relation_probs is not None:
            relation_probs_t = repeat(relation_probs, "b c h w -> (b t) c h w", t=t)
            gate = self.relation_modulator(relation_probs_t)  # [B*T, coverage_chs, H, W]
            coverage = coverage * (1.0 + gate)

        cov = self.conv(coverage)
        cov = self.act(cov)
        cov = cov.masked_fill(mask, 0.0)
        cov = self.proj(cov)
        cov = self.post_norm(cov, mask)

        cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)

        return cov
