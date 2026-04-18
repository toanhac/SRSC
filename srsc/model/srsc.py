from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
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
        use_rqm: bool = False,
        use_rmc: bool = True,
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
            num_relation_classes=num_relation_classes if use_relation_aux else 0,
            use_rqm=use_rqm,
            use_rmc=use_rmc,
        )

        self.use_relation_aux = use_relation_aux
        self.use_rqm = use_rqm
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
    ) -> Union[FloatTensor, Dict]:
        feature_16x, mask_16x = self.encoder(img, img_mask)

        relation_pred = None
        relation_probs = None
        if self.use_relation_aux and self.relation_head is not None:
            relation_pred = self.relation_head(feature_16x)
            # Detach so ARM/RQM gradients don't flow back into RelationHead
            relation_probs = torch.sigmoid(relation_pred).detach()

        feature_doubled = torch.cat((feature_16x, feature_16x), dim=0)
        mask_doubled = torch.cat((mask_16x, mask_16x), dim=0)

        relation_probs_doubled = None
        if relation_probs is not None:
            relation_probs_doubled = torch.cat((relation_probs, relation_probs), dim=0)

        out = self.decoder(
            feature_doubled, mask_doubled, tgt,
            relation_probs=relation_probs_doubled,
        )

        if not return_relation:
            return out

        result: Dict = {"logits": out}
        result["relation_pred"] = relation_pred
        return result

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

        relation_probs = None
        if self.use_relation_aux and self.relation_head is not None:
            relation_probs = torch.sigmoid(self.relation_head(feature_16x)).detach()

        return self.decoder.beam_search(
            [feature_16x],
            [mask_16x],
            beam_size,
            max_len,
            alpha,
            early_stopping,
            temperature,
            relation_probs=relation_probs,
        )

