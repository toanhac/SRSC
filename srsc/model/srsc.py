from typing import List, Tuple, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from srsc.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder
from .auxiliary_heads import RelationHead


class SRSC(pl.LightningModule):
    
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        use_relation_aux: bool = False,
        relation_hidden_channels: int = 128,
        num_relation_classes: int = 7,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, 
            growth_rate=growth_rate, 
            num_layers=num_layers,
        )
        
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            num_relation_classes=num_relation_classes,
        )
        
        self.use_relation_aux = use_relation_aux
        
        self.relation_head = None
        
        if use_relation_aux:
            self.relation_head = RelationHead(
                d_model=d_model,
                hidden_dim=relation_hidden_channels,
                num_classes=num_relation_classes,
            )

    def forward(
        self, 
        img: FloatTensor, 
        img_mask: LongTensor, 
        tgt: LongTensor,
        relation_map_gt: Optional[FloatTensor] = None,
        return_relation: bool = False,
        return_coverage: bool = False,
    ) -> Union[FloatTensor, Tuple[FloatTensor, ...]]:
        feature_16x, mask_16x = self.encoder(img, img_mask)
        
        relation_pred = None
        
        if self.use_relation_aux and self.relation_head is not None:
            relation_pred = self.relation_head(feature_16x)
        
        relation_for_decoder = None
        if relation_map_gt is not None:
            relation_for_decoder = relation_map_gt
        elif relation_pred is not None:
            relation_for_decoder = relation_pred.detach()
        
        feature_doubled = torch.cat((feature_16x, feature_16x), dim=0)
        mask_doubled = torch.cat((mask_16x, mask_16x), dim=0)
        
        if relation_for_decoder is not None:
            relation_decoder_doubled = torch.cat((relation_for_decoder, relation_for_decoder), dim=0)
        else:
            relation_decoder_doubled = None
        
        out, coverage_T = self.decoder(
            feature_doubled, mask_doubled, tgt, 
            relation_map=relation_decoder_doubled,
            return_coverage=return_coverage,
        )
        
        results = [out]
        if return_relation:
            results.append(relation_pred)
        if return_coverage:
            results.append(coverage_T)
        
        if len(results) == 1:
            return results[0]
        return tuple(results)

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        relation_map: Optional[FloatTensor] = None,
        **kwargs,
    ) -> List[Hypothesis]:
        feature_16x, mask_16x = self.encoder(img, img_mask)
        
        relation_for_decoder = None
        
        if relation_map is not None:
            relation_for_decoder = relation_map
        elif self.use_relation_aux and self.relation_head is not None:
            with torch.no_grad():
                relation_pred = self.relation_head(feature_16x)
                relation_for_decoder = relation_pred
        
        return self.decoder.beam_search(
            [feature_16x], [mask_16x], beam_size, max_len, alpha, early_stopping, temperature,
            relation_map=relation_for_decoder,
        )
    
    def predict_relation(self, img: FloatTensor, img_mask: LongTensor) -> Optional[FloatTensor]:
        if not self.use_relation_aux or self.relation_head is None:
            return None
        feature_16x, _ = self.encoder(img, img_mask)
        return self.relation_head(feature_16x)
