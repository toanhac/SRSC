import copy
from functools import partial
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from .arm import AttentionRefinementModule
from .attention import MultiheadAttention


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class RASAModule(nn.Module):
    """
    Relation-Aware Self-Attention (RASA) module.

    Each token predicts a soft relation type vector.
    The outer product of query/key relation vectors forms a relation bias
    that is added to self-attention scores before softmax.

    This captures structural relationships between decoded tokens
    (e.g. superscript/subscript patterns) in the self-attention path.

    Args:
        d_model: model dimension
        num_relation_classes: number of relation types (7 for SRSC)
        num_heads: number of attention heads (used for scaling)
    """

    def __init__(self, d_model: int, num_relation_classes: int = 7, num_heads: int = 8):
        super().__init__()
        self.num_relation_classes = num_relation_classes
        self.num_heads = num_heads

        self.rel_proj_q = nn.Linear(d_model, num_relation_classes)
        self.rel_proj_k = nn.Linear(d_model, num_relation_classes)

        # alpha_raw initialized to -4 so sigmoid(-4) ≈ 0.018: relation bias starts very small
        self.alpha_raw = nn.Parameter(torch.tensor(-4.0))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.rel_proj_q.weight)
        nn.init.zeros_(self.rel_proj_q.bias)
        nn.init.xavier_uniform_(self.rel_proj_k.weight)
        nn.init.zeros_(self.rel_proj_k.bias)

    def forward(self, tgt: Tensor) -> Tensor:
        """
        Args:
            tgt: [T, B, d_model] token hidden states (post-norm, before self-attention)
        Returns:
            rel_bias: [B * num_heads, T, T] relation bias to add to self-attention scores
        """
        T, B, _ = tgt.shape

        tgt_bt = tgt.transpose(0, 1)  # [B, T, d_model]

        g_q = torch.sigmoid(self.rel_proj_q(tgt_bt))  # [B, T, C]
        g_k = torch.sigmoid(self.rel_proj_k(tgt_bt))  # [B, T, C]

        rel_bias = torch.bmm(g_q, g_k.transpose(1, 2))  # [B, T, T]

        # Scale: sigmoid(alpha_raw)/sqrt(num_heads) — grows gradually from near-zero
        scale = torch.sigmoid(self.alpha_raw) / (self.num_heads ** 0.5)
        rel_bias = scale * rel_bias

        # Broadcast to all heads: [B, T, T] -> [B * num_heads, T, T]
        rel_bias = rel_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        rel_bias = rel_bias.reshape(B * self.num_heads, T, T)

        return rel_bias


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
        relation_pos: Optional[Tensor] = None,
        relation_flat: Optional[Tensor] = None,
        return_coverage: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        output = tgt

        arm = None
        all_gate_values = []
        all_attns = []
        
        for i, mod in enumerate(self.layers):
            output, attn, gate_t = mod(
                output,
                memory,
                arm,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                relation_pos=relation_pos,
                relation_flat=relation_flat,
            )
            all_gate_values.append(gate_t)
            all_attns.append(attn)
            
            if i != len(self.layers) - 1 and self.arm is not None:
                arm = partial(
                    self.arm, 
                    attn, 
                    memory_key_padding_mask, 
                    height,
                )

        if self.norm is not None:
            output = self.norm(output)

        coverage_T = None
        if return_coverage and len(all_attns) > 0 and len(all_gate_values) > 0:
            coverage_T = self._compute_coverage_T(
                all_attns, all_gate_values, memory_key_padding_mask, height
            )

        return output, coverage_T
    
    def _compute_coverage_T(
        self,
        all_attns,
        all_gate_values,
        memory_key_padding_mask,
        height,
    ):
        last_attn = all_attns[-1]
        last_gate = all_gate_values[-1]
        
        if last_gate is None:
            return None
        
        bsz_heads, T, N = last_attn.shape
        nhead = self.layers[0].multihead_attn.num_heads
        bsz = bsz_heads // nhead
        
        attn_mean = rearrange(last_attn, "(b n) t l -> b n t l", n=nhead).mean(dim=1)
        
        coverage_T = torch.einsum('btn,btc->bnc', attn_mean, last_gate)
        coverage_T = coverage_T.clamp(max=1.0)
        return coverage_T


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_relation_classes: int = 7,
        use_rasa: bool = False,
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

        self.relation_gate = nn.Linear(d_model, num_relation_classes)
        self.nhead = nhead

        self.use_rasa = use_rasa
        if use_rasa:
            self.rasa = RASAModule(
                d_model=d_model,
                num_relation_classes=num_relation_classes,
                num_heads=nhead,
            )

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        arm: Optional[AttentionRefinementModule],
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        relation_pos: Optional[Tensor] = None,
        relation_flat: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        gate_t = None

        if self.use_rasa:
            rasa_bias = self.rasa(tgt)  # [B*H, T, T]
            if tgt_mask is not None and tgt_mask.dtype == torch.bool:
                self_attn_mask = rasa_bias.masked_fill(
                    tgt_mask.unsqueeze(0), float('-inf')
                )
            else:
                self_attn_mask = rasa_bias
            tgt2 = self.self_attn(
                tgt, tgt, tgt, attn_mask=self_attn_mask, key_padding_mask=tgt_key_padding_mask
            )[0]
        else:
            tgt2 = self.self_attn(
                tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
            )[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if relation_flat is not None:
            tgt_for_gate = rearrange(tgt, "t b d -> b t d")
            gate_t = torch.sigmoid(self.relation_gate(tgt_for_gate))

        tgt2, attn = self.multihead_attn(
            tgt,
            memory,
            memory,
            arm=arm,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            k_pos=relation_pos,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn, gate_t
