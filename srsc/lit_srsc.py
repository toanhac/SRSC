import zipfile
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from torch import FloatTensor, LongTensor

from srsc.datamodule import Batch, vocab
from srsc.model.srsc import SRSC
from srsc.utils.utils import (ExpRateRecorder, Hypothesis, ce_loss,
                               to_bi_tgt_out)


class LitSRSC(pl.LightningModule):
    
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
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        learning_rate: float,
        patience: int,
        use_relation_aux: bool = False,
        relation_loss_weight: float = 0.3,
        coverage_loss_weight: float = 0.1,
        relation_hidden_channels: int = 128,
        num_relation_classes: int = 7,
        optimizer_type: str = 'sgd',
    ):
        super().__init__()
        self.save_hyperparameters()
        self.srsc_model = SRSC(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            use_relation_aux=use_relation_aux,
        )

        self.exprate_recorder = ExpRateRecorder()
        
        self.use_relation_aux = use_relation_aux
        self.relation_loss_weight = relation_loss_weight
        self.coverage_loss_weight = coverage_loss_weight
        self.num_relation_classes = num_relation_classes

    def forward(
        self, 
        img: FloatTensor, 
        img_mask: LongTensor, 
        tgt: LongTensor,
        relation_map_gt: Optional[FloatTensor] = None,
        return_relation: bool = False,
        return_coverage: bool = False,
        epoch_idx: int = -1,
    ) -> FloatTensor:
        return self.srsc_model(
            img, img_mask, tgt, 
            relation_map_gt=relation_map_gt,
            return_relation=return_relation,
            return_coverage=return_coverage,
        )

    def compute_relation_loss(
        self,
        relation_logits: FloatTensor,
        relation_gt: FloatTensor,
        ignore_class_0: bool = True,
        focal_gamma: float = 2.0,
    ) -> FloatTensor:
        """Focal BCE loss on raw logits for numerical stability."""
        if relation_logits.shape[2:] != relation_gt.shape[2:]:
            relation_gt = F.interpolate(
                relation_gt,
                size=relation_logits.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        if ignore_class_0 and relation_logits.shape[1] > 1:
            relation_logits = relation_logits[:, 1:, :, :]
            relation_gt = relation_gt[:, 1:, :, :]
        
        relation_gt = relation_gt.clamp(0, 1).float()
        
        # Numerically stable focal BCE using logits
        bce = F.binary_cross_entropy_with_logits(
            relation_logits, relation_gt, reduction='none'
        )
        
        # Focal weighting
        with torch.no_grad():
            p = torch.sigmoid(relation_logits)
            p_t = p * relation_gt + (1 - p) * (1 - relation_gt)
            focal_weight = (1 - p_t) ** focal_gamma
        
        focal_loss = focal_weight * bce
        return focal_loss.mean()

    def compute_coverage_penalty(
        self,
        coverage_T: FloatTensor,
        relation_gt: FloatTensor,
    ) -> FloatTensor:
        B = relation_gt.shape[0]
        H, W = relation_gt.shape[2], relation_gt.shape[3]
        N = coverage_T.shape[1]
        
        if N != H * W:
            cov_h = int(N ** 0.5)
            cov_w = N // cov_h if cov_h > 0 else N
            relation_gt_resized = F.interpolate(
                relation_gt, size=(cov_h, cov_w), mode='bilinear', align_corners=False
            )
        else:
            relation_gt_resized = relation_gt
        
        relation_flat = rearrange(relation_gt_resized, "b c h w -> b (h w) c")
        
        C_r = coverage_T.shape[2]
        if relation_flat.shape[2] != C_r:
            relation_flat = relation_flat[:, :, :C_r]
        
        if coverage_T.shape[0] != B:
            coverage_T = coverage_T[:B]
        
        penalty = F.relu(coverage_T - relation_flat)
        return penalty.mean()

    def _get_auxiliary_gt(self, batch: Batch):
        relation_gt = None
        
        if hasattr(batch, 'relation_map') and batch.relation_map is not None:
            relation_gt = batch.relation_map.to(self.device)
        
        return relation_gt

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        relation_gt = self._get_auxiliary_gt(batch)
        
        need_relation = self.use_relation_aux and relation_gt is not None
        need_coverage = need_relation and self.coverage_loss_weight > 0
        
        outputs = self.srsc_model(
            batch.imgs, batch.mask, tgt,
            relation_map_gt=relation_gt,
            return_relation=need_relation,
            return_coverage=need_coverage,
        )
        
        if need_relation and need_coverage:
            out_hat, relation_pred, coverage_T = outputs
        elif need_relation:
            out_hat, relation_pred = outputs
            coverage_T = None
        else:
            out_hat = outputs
            relation_pred = None
            coverage_T = None
        
        rec_loss = ce_loss(out_hat, out)
        total_loss = rec_loss
        
        self.log("train_rec_loss", rec_loss, on_step=False, on_epoch=True, sync_dist=True)
    
        if need_relation and relation_pred is not None:
            relation_loss = self.compute_relation_loss(relation_pred, relation_gt)
            total_loss = total_loss + self.relation_loss_weight * relation_loss
            self.log("train_relation_loss", relation_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        if need_coverage and coverage_T is not None and relation_gt is not None:
            cov_penalty = self.compute_coverage_penalty(coverage_T, relation_gt)
            total_loss = total_loss + self.coverage_loss_weight * cov_penalty
            self.log("train_coverage_loss", cov_penalty, on_step=False, on_epoch=True, sync_dist=True)
        
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        relation_gt = self._get_auxiliary_gt(batch)
        
        need_relation = self.use_relation_aux and relation_gt is not None
        
        outputs = self.srsc_model(
            batch.imgs, batch.mask, tgt,
            relation_map_gt=relation_gt,
            return_relation=need_relation,
        )
        
        if need_relation:
            out_hat, relation_pred = outputs
        else:
            out_hat = outputs
            relation_pred = None
        
        rec_loss = ce_loss(out_hat, out)
        total_loss = rec_loss
        
        if need_relation and relation_pred is not None:
            relation_loss = self.compute_relation_loss(relation_pred, relation_gt)
            total_loss = total_loss + self.relation_loss_weight * relation_loss
            self.log("val_relation_loss", relation_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        self.log(
            "val_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        hyps = self.approximate_joint_search(batch.imgs, batch.mask, relation_gt)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
        relation_gt = self._get_auxiliary_gt(batch)
        hyps = self.approximate_joint_search(batch.imgs, batch.mask, relation_gt)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        return batch.img_bases, [vocab.indices2label(h.seq) for h in hyps]

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")

        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds in test_outputs:
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)

    def approximate_joint_search(
        self, 
        img: FloatTensor, 
        mask: LongTensor,
        relation_map: Optional[FloatTensor] = None,
    ) -> List[Hypothesis]:
        return self.srsc_model.beam_search(
            img, mask, 
            relation_map=relation_map,
            **self.hparams
        )

    def configure_optimizers(self):
        opt_type = self.hparams.get('optimizer_type', 'sgd').lower()
        
        if opt_type == 'adamw':
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
        else:  # sgd (default)
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
