#!/usr/bin/env python3
"""
Visualize Relation Head Predictions vs Ground Truth
===================================================

This script loads a trained SRSC model and compares its multi-label relation 
predictions against the ground truth and the original base image.

Features:
1. Loads the latest checkpoint from `lightning_logs/` (or a specific one).
2. Uses the CROHMEDatamodule to load validation or test sequences.
3. Obtains model predictions via `LitSRSC.srsc_model.predict_relation()`.
4. Visualizes per-channel overlays comparing Image, GT, and Prediction.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srsc.lit_srsc import LitSRSC
from srsc.datamodule.datamodule import CROHMEDatamodule
from srsc.datamodule.multilabel_relation_gt import RelationType

CHANNEL_NAMES = RelationType.names()
CHANNEL_COLORS = RelationType.colors()


def find_latest_checkpoint(log_dir: str = "lightning_logs") -> str:
    """Finds the latest checkpoint in the lightning_logs directory."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return None

    versions = [v for v in log_path.iterdir() if v.is_dir() and v.name.startswith("version_")]
    if not versions:
        return None
    
    # Sort by version number
    try:
        versions.sort(key=lambda v: int(v.name.split('_')[1]))
    except ValueError:
        pass
        
    latest_version = versions[-1]
    ckpt_dir = latest_version / "checkpoints"
    
    if not ckpt_dir.exists():
        return None
        
    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
        
    # Exclude last.ckpt if possible and sort by name to get best/latest
    # lightning typically names as epoch=...
    return str(checkpoints[0])


def upsample_map(feature_map: torch.Tensor, target_h: int, target_w: int) -> np.ndarray:
    """Upsample relation map to target size using bilinear interpolation."""
    # map: (7, H', W')
    t = feature_map.float().unsqueeze(0)  # (1, 7, H', W')
    upsampled = F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return upsampled[0].detach().cpu().numpy()  # (7, H, W)


def load_gt_and_image(data_dir: Path, cache_dir: Path, split: str, sample_name: str):
    npz_path = cache_dir / split / f"{sample_name}_gt.npz"
    img_path = data_dir / split / "img" / f"{sample_name}.bmp"
    
    if not npz_path.exists():
        return None, None, None
    
    data = np.load(str(npz_path), allow_pickle=True)
    relation_map = data['relation_map']  # (7, H', W')
    latex = str(data['latex']) if 'latex' in data else ""
    
    img = None
    if img_path.exists():
        img = np.array(Image.open(img_path).convert('L'))
    
    return relation_map, img, latex


def visualize_comparison(img: np.ndarray, gt_map: np.ndarray, pred_map: np.ndarray, 
                         latex: str, sample_name: str, output_path: str):
    """
    Creates a grid visualization comparing Ground Truth and Predictions
    for the 6 meaningful channels.
    """
    H, W = img.shape[:2]
    gt_up = upsample_map(torch.from_numpy(gt_map), H, W)
    pred_up = upsample_map(torch.from_numpy(pred_map), H, W)
    
    # Create a 6x3 grid (6 channels, 3 columns: Base Image, GT, Pred)
    fig, axes = plt.subplots(6, 3, figsize=(18, 24))
    fig.suptitle(f"Relation Head Prediction: {sample_name}\nLaTeX: {latex[:100]}", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    col_titles = ["Base Image", "Ground Truth", "Prediction"]
    for i in range(3):
        axes[0, i].set_title(col_titles[i], fontsize=12, fontweight='bold')
    
    for idx, ch in enumerate(range(1, 7)):
        # Column 0: Base Image
        ax_img = axes[idx, 0]
        ax_img.imshow(img, cmap='gray')
        ax_img.set_ylabel(f"{CHANNEL_NAMES[ch]}", fontsize=12, fontweight='bold')
        ax_img.axis('auto')
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        # Column 1: Ground Truth
        ax_gt = axes[idx, 1]
        ax_gt.imshow(img, cmap='gray', alpha=1.0)
        gt_data = gt_up[ch]
        if gt_data.max() > 0.01:
            masked_gt = np.ma.masked_where(gt_data < 0.05, gt_data)
            ax_gt.imshow(masked_gt, alpha=0.6, cmap='jet', vmin=0, vmax=1)
            ax_gt.text(5, 15, f"active={(gt_data > 0.1).sum()}, max={gt_data.max():.2f}", 
                       color='lime', backgroundcolor='black', fontsize=10)
        else:
            ax_gt.text(5, 15, "empty", color='gray', backgroundcolor='black', fontsize=10)
        ax_gt.axis('auto')
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        
        # Column 2: Prediction
        ax_pred = axes[idx, 2]
        ax_pred.imshow(img, cmap='gray', alpha=1.0)
        pred_data = pred_up[ch]
        # Binarize output roughly with 0.1 threshold to show active region
        if pred_data.max() > 0.05:
            masked_pred = np.ma.masked_where(pred_data < 0.1, pred_data)
            ax_pred.imshow(masked_pred, alpha=0.6, cmap='jet', vmin=0, vmax=1)
            ax_pred.text(5, 15, f"active={(pred_data > 0.1).sum()}, max={pred_data.max():.2f}", 
                         color='cyan', backgroundcolor='black', fontsize=10)
        else:
            ax_pred.text(5, 15, "empty or weak", color='gray', backgroundcolor='black', fontsize=10)
        ax_pred.axis('auto')
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Relation Head Predictions')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint. If None, uses latest in lightning_logs/')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--cache_dir', type=str, default='data/cached_maps', help='Path to cached GT maps')
    parser.add_argument('--split', type=str, default='2014', help='Dataset split to use (e.g., 2014, train)')
    parser.add_argument('--max_samples', type=int, default=10, help='Max number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./visualize/relation_preds', help='Output directory')
    args = parser.parse_args()

    # Find checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint()
    
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"Error: Could not find checkpoint: {ckpt_path}")
        return
        
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LitSRSC.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nVisualizing predictions...")
    
    from srsc.datamodule.datamodule import CROHMEDatamodule
    # Removed MultiTaskCollator import from multilabel_relation_gt as it's not needed directly here
    
    # We will instantiate the datamodule just to get the dataset
    dm = CROHMEDatamodule(
        zipfile_path="data.zip", 
        test_year=args.split,
        eval_batch_size=1,
        use_relation_maps=True,
        gt_cache_dir=args.cache_dir
    )
    dm.setup(stage='test')
    
    if args.split == 'train':
        dl = dm.train_dataloader()
    elif args.split in ['2014', '2016', '2019', '2023']:
        dl = dm.test_dataloader()
    else:
        dl = dm.test_dataloader() # Default to test
        
    count = 0
    with torch.no_grad():
        for batch in dl:
            if count >= args.max_samples:
                break
                
            batch = batch.to(device)
            # imgs: (B, 1, H, W)
            # mask: (B, H, W)
            imgs = batch.imgs
            mask = batch.mask
            
            # Predict
            # LitSRSC.srsc_model is the SRSC model
            preds_prob = model.srsc_model.predict_relation(imgs, mask) # Returns probabilities (B, 7, H_feat, W_feat)
            
            if preds_prob is None:
                print("Model does not predict relation maps (use_relation_aux=False).")
                return
                
            # Iterate through batch
            for b_idx in range(len(batch.img_bases)):
                sample_name = batch.img_bases[b_idx]
                
                print(f"[{count+1}/{args.max_samples}] Processing: {sample_name}")
                
                gt_map, img_np, latex = load_gt_and_image(Path(args.data_dir), Path(args.cache_dir), args.split, sample_name)
                
                if img_np is None or gt_map is None:
                    print(f"  ⚠ Skipping {sample_name}: Image or GT not found (check config dirs)")
                    continue
                    
                pred_map_np = preds_prob[b_idx].detach().cpu().numpy()
                
                out_path = output_dir / f"{sample_name}_pred.png"
                visualize_comparison(img_np, gt_map, pred_map_np, latex, sample_name, str(out_path))
                
                count += 1
                if count >= args.max_samples:
                    break

    print(f"\n✓ Completed. Output saved to {output_dir}")

if __name__ == '__main__':
    main()
