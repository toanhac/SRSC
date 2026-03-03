import copy
from functools import partial
from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from .arm import AttentionRefinementModule
from .attention import MultiheadAttention

# Type alias for per-layer cache: (self_attn_kv, cross_attn_kv)
LayerCache = Tuple[Optional[Tuple[Tensor, Tensor]], Optional[Tuple[Tensor, Tensor]]]


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
        return_coverage: bool = False,
        use_cache: bool = False,
        past_key_values: Optional[List[LayerCache]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[List[LayerCache]]]:
        output = tgt

        arm = None
        all_gate_values = []
        all_attns = []
        present_key_values: List[LayerCache] = []
        
        for i, mod in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            output, attn, gate_t, present_kv = mod(
                output,
                memory,
                arm,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                relation_flat=relation_flat,
                use_cache=use_cache,
                past_self_kv=layer_past[0] if layer_past is not None else None,
                past_cross_kv=layer_past[1] if layer_past is not None else None,
            )
            all_gate_values.append(gate_t)
            all_attns.append(attn)
            if use_cache:
                present_key_values.append(present_kv)
            
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

        return output, coverage_T, present_key_values if use_cache else None
    
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
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, num_relation_classes=7):
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
        self.alpha_rel_bias = nn.Parameter(torch.tensor(-3.0))
        self.nhead = nhead

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
        relation_flat: Optional[Tensor] = None,
        use_cache: bool = False,
        past_self_kv: Optional[Tuple[Tensor, Tensor]] = None,
        past_cross_kv: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[LayerCache]]:
        
        # Self-attention with KV-cache
        tgt2 = self.norm1(tgt)
        tgt2, _, present_self_kv = self.self_attn(
            tgt2, tgt2, tgt2, 
            attn_mask=tgt_mask, 
            key_padding_mask=tgt_key_padding_mask,
            past_key_value=past_self_kv,
            use_cache=use_cache,
        )
        tgt = tgt + self.dropout1(tgt2)
        
        # Relation bias (computed for query tokens only)
        relation_bias = None
        gate_t = None
        
        if relation_flat is not None:
            tgt_for_gate = rearrange(tgt, "t b d -> b t d")
            gate_t = torch.sigmoid(self.relation_gate(tgt_for_gate))

            rel_bias = torch.bmm(gate_t, relation_flat.transpose(1, 2))
            rel_bias = rel_bias / math.sqrt(self.nhead)
            alpha = torch.sigmoid(self.alpha_rel_bias)
            rel_bias = alpha * rel_bias
            
            relation_bias = repeat(rel_bias, "b t n -> (b nh) t n", nh=self.nhead)
        
        # Cross-attention with KV-cache
        tgt2 = self.norm2(tgt)
        tgt2, attn, present_cross_kv = self.multihead_attn(
            tgt2,
            memory,
            memory,
            arm=arm,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            relation_bias=relation_bias,
            past_key_value=past_cross_kv,
            use_cache=use_cache,
        )
        tgt = tgt + self.dropout2(tgt2)
        
        # FFN
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
        present_kv: Optional[LayerCache] = None
        if use_cache:
            present_kv = (present_self_kv, present_cross_kv)
        
        return tgt, attn, gate_t, present_kv

