import copy
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .arm import AttentionRefinementModule
from .attention import MultiheadAttention


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers: int,
        arm: Optional[AttentionRefinementModule],
        norm=None,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.arm = arm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        height: int,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        relation_flat: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        arm = None

        for i, mod in enumerate(self.layers):
            output, attn = mod(
                output,
                memory,
                arm,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                relation_flat=relation_flat,
            )

            if i != len(self.layers) - 1 and self.arm is not None:
                arm = partial(
                    self.arm,
                    attn,
                    memory_key_padding_mask,
                    height,
                )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_relation_classes: int = 7,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu
        self.nhead = nhead

        self.rel_q_proj = nn.Linear(d_model, num_relation_classes, bias=False)
        self.rel_scale = nn.Parameter(torch.zeros(1))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def _compute_relation_bias(
        self, tgt: Tensor, relation_flat: Tensor
    ) -> Tensor:
        # tgt: [T, B, d_model], relation_flat: [B, H*W, num_rel]
        tgt_t = rearrange(tgt, "t b d -> b t d")
        gate = torch.sigmoid(self.rel_q_proj(tgt_t))           # [B, T, num_rel]
        r_attn = torch.bmm(gate, relation_flat.transpose(1, 2)) # [B, T, H*W]
        r_attn = self.rel_scale * r_attn

        B, T, L = r_attn.shape
        r_bias = r_attn.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        return r_bias.reshape(B * self.nhead, T, L)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        arm: Optional[AttentionRefinementModule],
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        relation_flat: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        r_bias = None
        if relation_flat is not None:
            r_bias = self._compute_relation_bias(tgt, relation_flat)

        tgt2, attn = self.multihead_attn(
            tgt,
            memory,
            memory,
            arm=arm,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            r_bias=r_bias,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn
