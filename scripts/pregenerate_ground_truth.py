#!/usr/bin/env python3
"""
Pre-generate Ground Truth Maps for SRSC Multi-Task Training
=============================================================

This script pre-generates both types of ground truth automatically:
1. Spatial maps (S): Binary masks showing symbol regions [1, H', W']
2. Relation maps (R): Multi-label multi-channel structure maps [7, H', W']

Size is automatically computed based on encoder 16x downsampling.

Usage:
    # Generate for training set only (val/test don't need GT)
    python scripts/pregenerate_ground_truth.py --data_zip data.zip --split train
    
    # Process with max samples limit (for testing)
    python scripts/pregenerate_ground_truth.py --data_zip data.zip --split train --max_samples 100
"""

import os
import sys
import argparse
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srsc.datamodule.spatial_gt import SpatialMapGenerator
from srsc.datamodule.multilabel_relation_gt import (
    MultiLabelRelationMapGenerator as RelationMapGenerator,
    RelationType
)

# Encoder downsampling: 8x after Feature Fusion (16x→8x)
ENCODER_DOWNSAMPLE_FACTOR = 8

# Fixed DPI for LaTeX rendering
DEFAULT_DPI = 200


def compute_encoder_output_size(img_height: int, img_width: int) -> tuple:
    """
    Compute encoder output size after 8x downsampling (with Feature Fusion).
    
    DenseNet encoder with Feature Fusion:
    1. conv1 (stride=2): ceil(H/2), ceil(W/2)
    2. max_pool2d (2): ceil(H/4), ceil(W/4)  
    3. trans1 avg_pool2d (2): ceil(H/8), ceil(W/8) ← F_8x (output after fusion)
    
    Note: F_16x is fused back to F_8x resolution via LightweightFeatureFusion
    """
    h, w = img_height, img_width
    for _ in range(3):  # Only 3 downsampling stages (8x total)
        h = (h + 1) // 2
        w = (w + 1) // 2
    return max(h, 1), max(w, 1)


def extract_samples_from_zip(archive: ZipFile, folder: str):
    """Extract samples from a zip archive."""
    samples = []
    try:
        caption_path = f"data/{folder}/caption.txt"
        with archive.open(caption_path, 'r') as f:
            captions = f.readlines()
        
        for line in captions:
            line = line.decode().strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            img_name = parts[0]
            formula = ' '.join(parts[1:])
            img_path = f"data/{folder}/img/{img_name}.bmp"
            try:
                with archive.open(img_path, 'r') as f:
                    img = Image.open(f).copy()
                    if img.mode != 'L':
                        img = img.convert('L')
                samples.append((img_name, img, formula))
            except KeyError:
                continue
    except KeyError:
        print(f"Warning: Could not find {folder} in archive")
        return []
    return samples


def process_single_sample(args):
    """Process a single sample to generate both spatial and relation maps."""
    img_name, img_pil, latex, output_dir = args
    
    try:
        img_w, img_h = img_pil.size
        
        # Compute target size based on encoder output
        target_h, target_w = compute_encoder_output_size(img_h, img_w)
        
        img_np = np.array(img_pil).astype(np.float32)
        img_tensor = torch.from_numpy(img_np)
        
        result = {
            'img_height': img_h,
            'img_width': img_w,
            'target_height': target_h,
            'target_width': target_w,
            'latex': latex,
        }
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate SPATIAL map
        spatial_gen = SpatialMapGenerator(
            img_height=img_h,
            img_width=img_w,
            target_height=target_h,
            target_width=target_w
        )
        spatial_map = spatial_gen(img_tensor).numpy()
        result['spatial_map'] = spatial_map  # (1, H, W)
        
        # Generate RELATION map (multi-label)
        relation_gen = RelationMapGenerator(
            target_height=target_h,
            target_width=target_w,
            dpi=DEFAULT_DPI,
            use_gaussian=True
        )
        relation_map, symbol_infos, _, _ = relation_gen.generate(latex)
        result['relation_map'] = relation_map  # (7, H, W)
        result['num_symbols'] = len(symbol_infos)
        
        # Save to npz file
        npz_path = output_dir / f"{img_name}_gt.npz"
        np.savez_compressed(str(npz_path), **result)
        
        return (img_name, True, None, (img_h, img_w, target_h, target_w))
    except Exception as e:
        import traceback
        return (img_name, False, str(e) + "\n" + traceback.format_exc(), None)


def main():
    parser = argparse.ArgumentParser(
        description='Pre-generate ground truth maps for SRSC multi-task training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate for training set only (recommended)
    python scripts/pregenerate_ground_truth.py --data_zip data.zip --split train
    
    # Generate for all splits
    python scripts/pregenerate_ground_truth.py --data_zip data.zip --split all
    
    # Test with limited samples and multiple workers
    python scripts/pregenerate_ground_truth.py --data_zip data.zip --split train --max_samples 100 --workers 4
        """
    )
    parser.add_argument('--data_zip', type=str, default='data.zip', 
                        help='Path to data.zip')
    parser.add_argument('--output_dir', type=str, default='data/cached_maps', 
                        help='Output directory')
    parser.add_argument('--split', type=str, default='train', 
                        choices=['train', '2014', '2016', '2019', 'all'],
                        help='Dataset split to process (default: train)')
    parser.add_argument('--max_samples', type=int, default=-1, 
                        help='Max samples to process (-1 for all)')
    parser.add_argument('--workers', type=int, default=1, 
                        help='Number of parallel workers (default: 1)')
    
    args = parser.parse_args()
    
    data_zip_path = Path(args.data_zip)
    if not data_zip_path.exists():
        print(f"Error: data.zip not found at {data_zip_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PRE-GENERATING GROUND TRUTH MAPS")
    print("=" * 70)
    print(f"Data zip: {data_zip_path}")
    print(f"Output directory: {output_dir}")
    print(f"Split: {args.split}")
    print(f"Workers: {args.workers}")
    print(f"Encoder downsample: {ENCODER_DOWNSAMPLE_FACTOR}x")
    print(f"Rendering DPI: {DEFAULT_DPI}")
    print()
    print("Generating both:")
    print("  - Spatial maps (S): Binary symbol masks [1, H', W']")
    print(f"  - Relation maps (R): Multi-label structure [7, H', W']")
    print(f"    Classes: {', '.join(RelationType.names())}")
    print("=" * 70)
    
    # Determine splits to process
    if args.split == 'all':
        splits = ['train', '2014', '2016', '2019']
    else:
        splits = [args.split]
    
    total_success = 0
    total_errors = 0
    all_sizes = []
    
    with ZipFile(data_zip_path, 'r') as archive:
        for split in splits:
            print(f"\nProcessing split: {split}")
            samples = extract_samples_from_zip(archive, split)
            if not samples:
                print(f"No samples found")
                continue
            
            if args.max_samples > 0:
                samples = samples[:args.max_samples]
            
            print(f"Found {len(samples)} samples")
            
            split_output_dir = output_dir / split
            split_output_dir.mkdir(parents=True, exist_ok=True)
            
            process_args = [
                (img_name, img_pil, latex, split_output_dir)
                for img_name, img_pil, latex in samples
            ]
            
            # Use multiprocessing if workers > 1
            if args.workers > 1:
                from multiprocessing import Pool
                with Pool(args.workers) as pool:
                    results = list(tqdm(
                        pool.imap(process_single_sample, process_args),
                        total=len(process_args),
                        desc=f"{split}"
                    ))
            else:
                # Single process
                results = []
                for proc_args in tqdm(process_args, desc=f"{split}"):
                    results.append(process_single_sample(proc_args))
            
            success = sum(1 for _, ok, _, _ in results if ok)
            errors = len(results) - success
            total_success += success
            total_errors += errors
            
            for _, ok, _, sizes in results:
                if ok and sizes:
                    all_sizes.append(sizes)
            
            print(f"Success: {success}/{len(results)}")
            if errors > 0:
                print(f"Errors: {errors}")
                error_count = 0
                for name, ok, err, _ in results:
                    if not ok:
                        print(f"  - {name}: {err[:150]}...")
                        error_count += 1
                        if error_count >= 3:
                            print(f"  ... and {errors - 3} more errors")
                        break
    
    if all_sizes:
        img_heights = [s[0] for s in all_sizes]
        img_widths = [s[1] for s in all_sizes]
        target_heights = [s[2] for s in all_sizes]
        target_widths = [s[3] for s in all_sizes]
        
        print("\n" + "=" * 70)
        print("SIZE STATISTICS")
        print("=" * 70)
        print(f"Image sizes:")
        print(f"  Height - min: {min(img_heights):4d}, max: {max(img_heights):4d}, avg: {np.mean(img_heights):.1f}")
        print(f"  Width  - min: {min(img_widths):4d}, max: {max(img_widths):4d}, avg: {np.mean(img_widths):.1f}")
        print(f"Ground truth sizes (encoder output):")
        print(f"  Height - min: {min(target_heights):4d}, max: {max(target_heights):4d}, avg: {np.mean(target_heights):.1f}")
        print(f"  Width  - min: {min(target_widths):4d}, max: {max(target_widths):4d}, avg: {np.mean(target_widths):.1f}")
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"Total success: {total_success}")
    print(f"Total errors: {total_errors}")
    if args.split == 'all':
        print(f"Cached maps saved to:")
        for s in splits:
            print(f"  - {output_dir / s}")
    else:
        print(f"Cached maps saved to: {output_dir / args.split}")
    print("=" * 70)
    
    print("\nExample usage to load cached ground truth:")
    print("```python")
    print("import numpy as np")
    example_split = 'train' if args.split == 'all' else args.split
    print(f"data = np.load('{output_dir / example_split}/sample_gt.npz')")
    print("spatial_map = data['spatial_map']   # Shape: (1, H', W')")
    print("relation_map = data['relation_map'] # Shape: (7, H', W') - MULTI-LABEL")
    print("```")


if __name__ == '__main__':
    main()
