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
        return_relation: bool = False,
        return_coverage: bool = False,
    ) -> Union[FloatTensor, Tuple[FloatTensor, ...]]:
        feature_16x, mask_16x = self.encoder(img, img_mask)

        relation_pred = None
        if self.use_relation_aux and self.relation_head is not None:
            relation_pred = self.relation_head(feature_16x)

        feature_doubled = torch.cat((feature_16x, feature_16x), dim=0)
        mask_doubled = torch.cat((mask_16x, mask_16x), dim=0)

        out, coverage_T = self.decoder(
            feature_doubled, mask_doubled, tgt,
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
        **kwargs,
    ) -> List[Hypothesis]:
        feature_16x, mask_16x = self.encoder(img, img_mask)

        return self.decoder.beam_search(
            [feature_16x], [mask_16x], beam_size, max_len, alpha, early_stopping, temperature
        )

    def predict_relation(self, img: FloatTensor, img_mask: LongTensor) -> Optional[FloatTensor]:
        if not self.use_relation_aux or self.relation_head is None:
            return None
        feature_16x, _ = self.encoder(img, img_mask)
        return torch.sigmoid(self.relation_head(feature_16x))
