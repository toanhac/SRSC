from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor

from srsc.datamodule import vocab, vocab_size
from srsc.model.pos_enc import WordPosEnc
from srsc.model.transformer.arm import AttentionRefinementModule
from srsc.model.transformer.transformer_decoder import (
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
    use_guided_coverage: bool = False,
    spatial_scale: float = 1.0,
    alpha_spatial: float = 0.3,
    alpha_relation: float = 0.2,
    coverage_aware_w1: float = 2.0,
    coverage_aware_w2: float = 1.0,
) -> nn.TransformerDecoder:
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(
            nhead, 
            dc, 
            cross_coverage, 
            self_coverage,
            use_guided_coverage=use_guided_coverage,
            spatial_scale=spatial_scale,
            alpha_spatial=alpha_spatial,
            alpha_relation=alpha_relation,
            coverage_aware_w1=coverage_aware_w1,
            coverage_aware_w2=coverage_aware_w2,
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
        use_guided_coverage: bool = False,
        spatial_scale: float = 1.0,
        alpha_spatial: float = 0.3,
        alpha_relation: float = 0.2,
        coverage_aware_w1: float = 2.0,
        coverage_aware_w2: float = 1.0,
    ):
        super().__init__()

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )

        self.pos_enc = WordPosEnc(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            use_guided_coverage=use_guided_coverage,
            spatial_scale=spatial_scale,
            alpha_spatial=alpha_spatial,
            alpha_relation=alpha_relation,
            coverage_aware_w1=coverage_aware_w1,
            coverage_aware_w2=coverage_aware_w2,
        )

        self.proj = nn.Linear(d_model, vocab_size)
        self._cached_spatial_map = None
        self._cached_relation_map = None

    def _build_attention_mask(self, length):
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)
        return mask

    def forward(
        self, 
        src: FloatTensor, 
        src_mask: LongTensor, 
        tgt: LongTensor,
        spatial_map: Optional[FloatTensor] = None,
        relation_map: Optional[FloatTensor] = None,
        epoch_idx: int = -1,
    ) -> FloatTensor:
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt = self.word_embed(tgt)
        tgt = self.pos_enc(tgt)
        tgt = self.norm(tgt)

        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        out = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
            spatial_map=spatial_map,
            relation_map=relation_map,
            epoch_idx=epoch_idx,
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)

        return out

    def transform(
        self, 
        src: List[FloatTensor], 
        src_mask: List[LongTensor], 
        input_ids: LongTensor,
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        spatial_map = self._cached_spatial_map
        relation_map = self._cached_relation_map
        
        batch_size = input_ids.shape[0]
        
        if spatial_map is not None:
            if spatial_map.shape[0] != batch_size:
                spatial_map = spatial_map.repeat(batch_size // spatial_map.shape[0] + 1, 1, 1, 1)[:batch_size]
        
        if relation_map is not None:
            if relation_map.shape[0] != batch_size:
                relation_map = relation_map.repeat(batch_size // relation_map.shape[0] + 1, 1, 1, 1)[:batch_size]
        
        word_out = self.forward(
            src[0], src_mask[0], input_ids, 
            spatial_map=spatial_map,
            relation_map=relation_map,
        )
        return word_out

    def beam_search(
        self,
        src: List[FloatTensor],
        src_mask: List[LongTensor],
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        spatial_map: Optional[FloatTensor] = None,
        relation_map: Optional[FloatTensor] = None,
    ) -> List[Hypothesis]:
        self._cached_spatial_map = spatial_map
        self._cached_relation_map = relation_map
        try:
            result = super().beam_search(
                src, src_mask, beam_size, max_len, alpha, early_stopping, temperature
            )
        finally:
            self._cached_spatial_map = None
            self._cached_relation_map = None
        return result

