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


class AttentionRefinementModule(nn.Module):
    def __init__(
        self,
        nhead: int,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        num_relation_classes: int = 0,
        d_model: int = 0,
    ):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage
        self.num_relation_classes = num_relation_classes

        coverage_chs = (2 if cross_coverage and self_coverage else 1) * nhead
        n_enc_chs = num_relation_classes if (d_model > 0 and num_relation_classes > 0) else 0
        in_chs = coverage_chs + n_enc_chs
        self.coverage_chs = coverage_chs

        self.conv = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
        self.post_norm = MaskBatchNorm2d(nhead)

        if n_enc_chs > 0:
            self.enc_proj = nn.Conv2d(d_model, n_enc_chs, kernel_size=1)
            with torch.no_grad():
                self.conv.weight[:, coverage_chs:].zero_()

    def forward(
        self,
        prev_attn: Tensor,
        key_padding_mask: Tensor,
        h: int,
        curr_attn: Tensor,
        encoder_memory: Optional[Tensor] = None,
    ) -> Tensor:
        t = curr_attn.shape[1]
        b = key_padding_mask.shape[0]
        w = key_padding_mask.shape[1] // h

        mask = repeat(key_padding_mask, "b (h w) -> (b t) () h w", h=h, t=t)

        curr_attn = rearrange(curr_attn, "(b n) t l -> b n t l", n=self.nhead)
        prev_attn = rearrange(prev_attn, "(b n) t l -> b n t l", n=self.nhead)

        attns = []
        if self.cross_coverage:
            attns.append(prev_attn)
        if self.self_coverage:
            attns.append(curr_attn)
        attns = torch.cat(attns, dim=1)

        attns = attns.cumsum(dim=2) - attns
        attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)

        if hasattr(self, 'enc_proj') and encoder_memory is not None:
            r = self.enc_proj(encoder_memory)       # [B, n_enc_chs, H, W]
            r = repeat(r, "b c h w -> (b t) c h w", t=t)
            attns = torch.cat([attns, r], dim=1)

        cov = self.conv(attns)
        cov = self.act(cov)
        cov = cov.masked_fill(mask, 0.0)
        cov = self.proj(cov)
        cov = self.post_norm(cov, mask)

        cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)

        return cov
