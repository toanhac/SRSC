import zipfile
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor, LongTensor

from srsc.datamodule import Batch, vocab
from srsc.model.srsc import SRSC
from srsc.utils.utils import (ExpRateRecorder, Hypothesis, ce_loss,
                               to_bi_tgt_out)


class LitSRSC(pl.LightningModule):
    """
    PyTorch Lightning module for SRSC with Multi-Task Learning.
    
    Supports:
    1. Main task: HMER (sequence recognition)
    2. Auxiliary task 1: Spatial distribution recovery
    3. Auxiliary task 2: Structural relation prediction
    4. Guided coverage attention using spatial and relation maps
    
    Loss function (from temp-2.pdf):
    L_total = L_rec + λ_s · L_spatial + λ_r · L_relation
    """
    
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
        fusion_out_channels: int = 128,
        use_spatial_aux: bool = False,
        use_relation_aux: bool = False,
        spatial_loss_weight: float = 0.5,
        relation_loss_weight: float = 0.3,
        use_guided_coverage: bool = False,
        alpha_spatial: float = 0.3,
        alpha_relation: float = 0.2,
        coverage_aware_w1: float = 2.0,
        coverage_aware_w2: float = 1.0,
        spatial_hidden_channels: int = 256,
        relation_hidden_channels: int = 128,
        spatial_scale: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        use_guided_coverage = use_guided_coverage or use_spatial_aux or use_relation_aux
        
        if use_guided_coverage:
            if not use_spatial_aux:
                use_spatial_aux = True
            if not use_relation_aux:
                use_relation_aux = True
        
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
            fusion_out_channels=fusion_out_channels,
            use_spatial_aux=use_spatial_aux,
            use_relation_aux=use_relation_aux,
            spatial_hidden_channels=spatial_hidden_channels,
            relation_hidden_channels=relation_hidden_channels,
            spatial_scale=spatial_scale,
            use_guided_coverage=use_guided_coverage,
            alpha_spatial=alpha_spatial,
            alpha_relation=alpha_relation,
            coverage_aware_w1=coverage_aware_w1,
            coverage_aware_w2=coverage_aware_w2,
        )

        self.exprate_recorder = ExpRateRecorder()
        
        # Multi-task config
        self.use_spatial_aux = use_spatial_aux
        self.use_relation_aux = use_relation_aux
        self.use_guided_coverage = use_guided_coverage
        self.spatial_loss_weight = spatial_loss_weight
        self.relation_loss_weight = relation_loss_weight

    def forward(
        self, 
        img: FloatTensor, 
        img_mask: LongTensor, 
        tgt: LongTensor,
        spatial_map_gt: Optional[FloatTensor] = None,
        relation_map_gt: Optional[FloatTensor] = None,
        return_spatial: bool = False,
        return_relation: bool = False,
        epoch_idx: int = -1,
    ) -> FloatTensor:
        return self.srsc_model(
            img, img_mask, tgt, 
            spatial_map_gt=spatial_map_gt,
            relation_map_gt=relation_map_gt,
            return_spatial=return_spatial,
            return_relation=return_relation,
            epoch_idx=epoch_idx,
        )

    def compute_spatial_loss(
        self, 
        spatial_pred: FloatTensor, 
        spatial_gt: FloatTensor
    ) -> FloatTensor:
        """Compute spatial recovery loss using BCE + Dice Loss."""
        if spatial_gt.dim() == 3:
            spatial_gt = spatial_gt.unsqueeze(1)
        if spatial_pred.shape[2:] != spatial_gt.shape[2:]:
            spatial_gt = F.interpolate(
                spatial_gt, 
                size=spatial_pred.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Clamp GT to [0, 1] for BCE
        spatial_gt = spatial_gt.clamp(0, 1)
        
        # BCE Loss
        bce_loss = F.binary_cross_entropy(spatial_pred, spatial_gt)
        
        # Dice Loss for better segmentation
        smooth = 1e-5
        pred_flat = spatial_pred.view(-1)
        gt_flat = spatial_gt.view(-1)
        intersection = (pred_flat * gt_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
        
        return 0.5 * bce_loss + 0.5 * dice_loss

    def compute_relation_loss(
        self,
        relation_pred: FloatTensor,
        relation_gt: FloatTensor,
        ignore_class_0: bool = True,
        focal_gamma: float = 2.0,
    ) -> FloatTensor:
        """
        Compute multi-label relation loss using Focal Loss.
        
        Focal Loss helps with class imbalance by down-weighting easy examples.
        """
        if relation_pred.shape[2:] != relation_gt.shape[2:]:
            relation_gt = F.interpolate(
                relation_gt,
                size=relation_pred.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Skip channel 0 (NONE) if specified
        if ignore_class_0 and relation_pred.shape[1] > 1:
            relation_pred = relation_pred[:, 1:, :, :]
            relation_gt = relation_gt[:, 1:, :, :]
        
        # Clamp values
        relation_gt = relation_gt.clamp(0, 1)
        relation_pred = relation_pred.clamp(1e-7, 1 - 1e-7)
        
        # Focal Loss: FL(p) = -alpha * (1-p)^gamma * log(p)
        # For binary: p_t = p if y=1 else 1-p
        p_t = relation_pred * relation_gt + (1 - relation_pred) * (1 - relation_gt)
        focal_weight = (1 - p_t) ** focal_gamma
        
        bce = -relation_gt * torch.log(relation_pred) - (1 - relation_gt) * torch.log(1 - relation_pred)
        focal_loss = focal_weight * bce
        
        return focal_loss.mean()

    def _get_auxiliary_gt(self, batch: Batch):
        """Extract spatial and relation GT from batch."""
        spatial_gt = None
        relation_gt = None
        
        if hasattr(batch, 'spatial_map') and batch.spatial_map is not None:
            spatial_gt = batch.spatial_map.to(self.device)
        
        if hasattr(batch, 'relation_map') and batch.relation_map is not None:
            relation_gt = batch.relation_map.to(self.device)
        
        return spatial_gt, relation_gt

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        spatial_gt, relation_gt = self._get_auxiliary_gt(batch)
        
        # Determine what auxiliary outputs we need
        need_spatial = self.use_spatial_aux and spatial_gt is not None
        need_relation = self.use_relation_aux and relation_gt is not None
        
        # Forward pass
        outputs = self.srsc_model(
            batch.imgs, batch.mask, tgt,
            spatial_map_gt=spatial_gt,
            relation_map_gt=relation_gt,
            return_spatial=need_spatial,
            return_relation=need_relation,
            epoch_idx=self.current_epoch,
        )
        
        # Unpack outputs
        if need_spatial and need_relation:
            out_hat, spatial_pred, relation_pred = outputs
        elif need_spatial:
            out_hat, spatial_pred = outputs
            relation_pred = None
        elif need_relation:
            out_hat, relation_pred = outputs
            spatial_pred = None
        else:
            out_hat = outputs
            spatial_pred = None
            relation_pred = None
        
        # Compute losses
        rec_loss = ce_loss(out_hat, out)
        total_loss = rec_loss
        
        self.log("train_rec_loss", rec_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        if need_spatial and spatial_pred is not None:
            spatial_loss = self.compute_spatial_loss(spatial_pred, spatial_gt)
            total_loss = total_loss + self.spatial_loss_weight * spatial_loss
            self.log("train_spatial_loss", spatial_loss, on_step=False, on_epoch=True, sync_dist=True)
    
        if need_relation and relation_pred is not None:
            relation_loss = self.compute_relation_loss(relation_pred, relation_gt)
            total_loss = total_loss + self.relation_loss_weight * relation_loss
            self.log("train_relation_loss", relation_loss, on_step=False, on_epoch=True, sync_dist=True)
        
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        spatial_gt, relation_gt = self._get_auxiliary_gt(batch)
        
        need_spatial = self.use_spatial_aux and spatial_gt is not None
        need_relation = self.use_relation_aux and relation_gt is not None
        
        outputs = self.srsc_model(
            batch.imgs, batch.mask, tgt,
            spatial_map_gt=spatial_gt,
            relation_map_gt=relation_gt,
            return_spatial=need_spatial,
            return_relation=need_relation,
        )
        
        if need_spatial and need_relation:
            out_hat, spatial_pred, relation_pred = outputs
        elif need_spatial:
            out_hat, spatial_pred = outputs
            relation_pred = None
        elif need_relation:
            out_hat, relation_pred = outputs
            spatial_pred = None
        else:
            out_hat = outputs
            spatial_pred = None
            relation_pred = None
        
        rec_loss = ce_loss(out_hat, out)
        total_loss = rec_loss
        
        if need_spatial and spatial_pred is not None:
            spatial_loss = self.compute_spatial_loss(spatial_pred, spatial_gt)
            total_loss = total_loss + self.spatial_loss_weight * spatial_loss
            self.log("val_spatial_loss", spatial_loss, on_step=False, on_epoch=True, sync_dist=True)
        
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

        hyps = self.approximate_joint_search(batch.imgs, batch.mask, spatial_gt, relation_gt)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
        spatial_gt, relation_gt = self._get_auxiliary_gt(batch)
        hyps = self.approximate_joint_search(batch.imgs, batch.mask, spatial_gt, relation_gt)
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
        spatial_map: Optional[FloatTensor] = None,
        relation_map: Optional[FloatTensor] = None,
    ) -> List[Hypothesis]:
        return self.srsc_model.beam_search(
            img, mask, 
            spatial_map=spatial_map,
            relation_map=relation_map,
            **self.hparams
        )

    def configure_optimizers(self):
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

