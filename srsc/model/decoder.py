from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import FloatTensor, LongTensor

from srsc.datamodule import vocab, vocab_size
from srsc.model.pos_enc import WordPosEnc
from srsc.model.transformer.arm import AttentionRefinementModule
from srsc.model.transformer.transformer_decoder import (
    LayerCache,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from srsc.utils.generation_utils import DecodeModel
from srsc.utils.utils import Hypothesis


def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    dc: int,
    cross_coverage: bool,
    self_coverage: bool,
    num_relation_classes: int = 7,
) -> TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_relation_classes=num_relation_classes,
    )
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(
            nhead, 
            dc, 
            cross_coverage, 
            self_coverage,
        )
    else:
        arm = None

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, arm)
    return decoder


class Decoder(DecodeModel):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        num_relation_classes: int = 7,
    ):
        super().__init__()

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )

        self.pos_enc = WordPosEnc(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.num_relation_classes = num_relation_classes

        self.relation_proj = nn.Sequential(
            nn.Conv2d(num_relation_classes, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self.proj = nn.Linear(d_model, vocab_size)
        
        # Beam search state
        self._cached_relation_map = None
        # KV-cache state (populated during beam search)
        self._kv_cache: Optional[List[LayerCache]] = None
        self._cached_memory: Optional[FloatTensor] = None
        self._cached_memory_mask: Optional[LongTensor] = None
        self._cached_r_flat: Optional[FloatTensor] = None
        self._cached_height: Optional[int] = None

    def _build_attention_mask(self, length):
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)
        return mask

    def _prepare_relation(self, relation_map, h, w):
        if relation_map is None:
            return None, None
        
        if relation_map.shape[2] != h or relation_map.shape[3] != w:
            relation_map_resized = F.interpolate(
                relation_map, size=(h, w), mode='bilinear', align_corners=False
            )
        else:
            relation_map_resized = relation_map
        
        r_proj = self.relation_proj(relation_map_resized)
        r_proj = rearrange(r_proj, "b d h w -> (h w) b d")
        
        r_flat = rearrange(relation_map_resized, "b c h w -> b (h w) c")
        
        return r_proj, r_flat

    def forward(
        self, 
        src: FloatTensor, 
        src_mask: LongTensor, 
        tgt: LongTensor,
        relation_map: Optional[FloatTensor] = None,
        return_coverage: bool = False,
    ) -> Tuple[FloatTensor, Optional[FloatTensor]]:
        """Training forward pass (no KV-cache)."""
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt = self.word_embed(tgt)
        tgt = self.pos_enc(tgt)
        tgt = self.norm(tgt)

        h = src.shape[1]
        w = src.shape[2]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        r_proj, r_flat = self._prepare_relation(relation_map, h, w)
        
        if r_proj is not None:
            src = src + r_proj

        out, coverage_T, _ = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
            relation_flat=r_flat,
            return_coverage=return_coverage,
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)

        if return_coverage:
            return out, coverage_T
        return out, None

    def transform(
        self, 
        src: List[FloatTensor], 
        src_mask: List[LongTensor], 
        input_ids: LongTensor,
    ) -> FloatTensor:
        """
        Incremental decode for beam search with KV-cache.
        
        First call: full forward + populate caches.
        Subsequent calls: process only the new (last) token.
        """
        assert len(src) == 1 and len(src_mask) == 1
        
        batch_size = input_ids.shape[0]
        relation_map = self._cached_relation_map
        if relation_map is not None and relation_map.shape[0] != batch_size:
            relation_map = relation_map.repeat(
                batch_size // relation_map.shape[0] + 1, 1, 1, 1
            )[:batch_size]
        
        # First call: no cache yet, do full forward and populate cache
        if self._kv_cache is None:
            # Prepare encoder memory (cached for all future calls)
            s = src[0]
            sm = src_mask[0]
            h = s.shape[1]
            w = s.shape[2]
            memory = rearrange(s, "b h w d -> (h w) b d")
            memory_mask = rearrange(sm, "b h w -> b (h w)")
            
            r_proj, r_flat = self._prepare_relation(relation_map, h, w)
            if r_proj is not None:
                memory = memory + r_proj
            
            # Cache encoder outputs
            self._cached_memory = memory
            self._cached_memory_mask = memory_mask
            self._cached_r_flat = r_flat
            self._cached_height = h
            
            # Full forward with cache enabled
            _, l = input_ids.size()
            tgt_mask = self._build_attention_mask(l)
            tgt_pad_mask = input_ids == vocab.PAD_IDX
            
            tgt = self.word_embed(input_ids)
            tgt = self.pos_enc(tgt)
            tgt = self.norm(tgt)
            tgt = rearrange(tgt, "b l d -> l b d")
            
            out, _, present_kv = self.model(
                tgt=tgt,
                memory=memory,
                height=h,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=memory_mask,
                relation_flat=r_flat,
                use_cache=True,
            )
            
            self._kv_cache = present_kv
            
            out = rearrange(out, "l b d -> b l d")
            return self.proj(out)
        
        # Subsequent calls: incremental decode (only process last token)
        new_token = input_ids[:, -1:]  # [B, 1]
        seq_len = input_ids.shape[1]
        
        tgt = self.word_embed(new_token)
        tgt = self.pos_enc(tgt, offset=seq_len - 1)
        tgt = self.norm(tgt)
        tgt = rearrange(tgt, "b l d -> l b d")  # [1, B, D]
        
        # No causal mask needed for single token (it attends to all cached tokens)
        out, _, present_kv = self.model(
            tgt=tgt,
            memory=self._cached_memory,
            height=self._cached_height,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=self._cached_memory_mask,
            relation_flat=self._cached_r_flat,
            use_cache=True,
            past_key_values=self._kv_cache,
        )
        
        self._kv_cache = present_kv
        
        out = rearrange(out, "l b d -> b l d")  # [B, 1, D]
        
        # Return full sequence shape for compatibility: [B, seq_len, V]
        # But only the last token logits matter (beam search takes [:, -1, :])
        word_out = self.proj(out)
        return word_out

    def _reorder_cache(self, beam_idx: LongTensor):
        """Reorder KV-cache entries when beam indices change."""
        if self._kv_cache is None:
            return
        
        new_cache = []
        for layer_cache in self._kv_cache:
            self_kv, cross_kv = layer_cache
            
            # Reorder self-attention cache
            new_self_kv = None
            if self_kv is not None:
                # self_kv shape: (bsz * num_heads, seq_len, head_dim)
                # beam_idx shape: (bsz,)
                # Need to expand beam_idx for num_heads
                num_heads = self_kv[0].shape[0] // beam_idx.shape[0]
                expanded_idx = beam_idx.unsqueeze(1).repeat(1, num_heads).view(-1)
                new_self_kv = (
                    self_kv[0].index_select(0, expanded_idx),
                    self_kv[1].index_select(0, expanded_idx),
                )
            
            # Reorder cross-attention cache
            new_cross_kv = None
            if cross_kv is not None:
                num_heads = cross_kv[0].shape[0] // beam_idx.shape[0]
                expanded_idx = beam_idx.unsqueeze(1).repeat(1, num_heads).view(-1)
                new_cross_kv = (
                    cross_kv[0].index_select(0, expanded_idx),
                    cross_kv[1].index_select(0, expanded_idx),
                )
            
            new_cache.append((new_self_kv, new_cross_kv))
        
        self._kv_cache = new_cache

    def _clear_cache(self):
        """Clear all beam search caches."""
        self._kv_cache = None
        self._cached_memory = None
        self._cached_memory_mask = None
        self._cached_r_flat = None
        self._cached_height = None

    def beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        relation_map: Optional[FloatTensor] = None,
    ) -> List[Hypothesis]:
        self._cached_relation_map = relation_map
        self._clear_cache()
        try:
            result = super().beam_search(
                src, src_mask, beam_size, max_len, alpha, early_stopping, temperature
            )
        finally:
            self._cached_relation_map = None
            self._clear_cache()
        return result

