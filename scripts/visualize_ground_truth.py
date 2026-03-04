#!/usr/bin/env python3
"""
Visualize Ground Truth Maps for SRSC
======================================

Kiểm tra chất lượng Ground Truth bằng cách:
1. Per-channel overlay: Upsample relation map lên kích thước gốc, overlay lên ảnh
2. Bounding box inspection: Re-render LaTeX có màu, vẽ bounding box
3. Relation accumulation: Kiểm tra multi-label tại các cấu trúc lồng nhau
4. Multi-sample grid: Xuất nhiều sample cùng lúc
5. Statistics: Thống kê kênh active và cảnh báo bất thường

Usage:
    # Kiểm tra 5 mẫu ngẫu nhiên
    python scripts/visualize_ground_truth.py --data_dir data --split train --max_samples 5

    # Chỉ kiểm tra các mẫu phức tạp (có frac/sqrt)
    python scripts/visualize_ground_truth.py --data_dir data --split train --complex_only --max_samples 10

    # Kiểm tra mẫu cụ thể
    python scripts/visualize_ground_truth.py --data_dir data --split train --sample_name 200922-1017-140
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from srsc.datamodule.multilabel_relation_gt import (
    MultiLabelRelationMapGenerator as RelationMapGenerator,
    RelationType,
    MultiLabelLaTeXParser,
    ColorCodedLatexRenderer,
    detect_symbol_regions,
)

# Channel names (skip NONE=0 for visualization, channels 1-6 are meaningful)
CHANNEL_NAMES = RelationType.names()  # ['None', 'Horizontal', 'Above', 'Below', 'Superscript', 'Subscript', 'Inside']
CHANNEL_COLORS = RelationType.colors()


def load_gt_and_image(data_dir: Path, cache_dir: Path, split: str, sample_name: str):
    """Load ground truth .npz and the corresponding .bmp image."""
    npz_path = cache_dir / split / f"{sample_name}_gt.npz"
    img_path = data_dir / split / "img" / f"{sample_name}.bmp"
    
    if not npz_path.exists():
        print(f"  ⚠ GT file not found: {npz_path}")
        return None, None, None
    
    data = np.load(str(npz_path), allow_pickle=True)
    relation_map = data['relation_map']  # (7, H', W')
    latex = str(data['latex']) if 'latex' in data else ""
    
    img = None
    if img_path.exists():
        img = np.array(Image.open(img_path).convert('L'))
    else:
        print(f"  ⚠ Image not found: {img_path}")
    
    return relation_map, img, latex


def upsample_relation_map(relation_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Upsample relation map to target size using bilinear interpolation."""
    import torch
    import torch.nn.functional as F
    
    # relation_map: (7, H', W')
    t = torch.from_numpy(relation_map).float().unsqueeze(0)  # (1, 7, H', W')
    upsampled = F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return upsampled[0].numpy()  # (7, H, W)


def visualize_per_channel_overlay(relation_map: np.ndarray, img: np.ndarray, 
                                   latex: str, sample_name: str, output_path: str):
    """
    Bước 1+2: Upsample relation map, overlay từng kênh lên ảnh gốc.
    Xuất grid 2×3 (6 kênh: Horizontal, Above, Below, Superscript, Subscript, Inside).
    """
    H, W = img.shape[:2]
    r_up = upsample_relation_map(relation_map, H, W)  # (7, H, W)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Per-Channel Overlay: {sample_name}\nLaTeX: {latex[:80]}{'...' if len(latex) > 80 else ''}", 
                 fontsize=12, fontweight='bold')
    
    # Channels 1-6 (skip NONE=0)
    for idx, ch in enumerate(range(1, 7)):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img, cmap='gray', alpha=1.0)
        
        channel_data = r_up[ch]
        if channel_data.max() > 0.01:
            # Overlay heatmap
            masked = np.ma.masked_where(channel_data < 0.05, channel_data)
            ax.imshow(masked, alpha=0.6, cmap='jet', vmin=0, vmax=1)
            active_pixels = (channel_data > 0.1).sum()
            max_val = channel_data.max()
            ax.set_title(f"{CHANNEL_NAMES[ch]} (active={active_pixels}, max={max_val:.2f})", 
                        fontsize=10, color='green')
        else:
            ax.set_title(f"{CHANNEL_NAMES[ch]} (empty)", fontsize=10, color='gray')
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved per-channel overlay: {output_path}")


def visualize_bounding_boxes(latex: str, sample_name: str, output_path: str, dpi: int = 200):
    """
    Bước 3: Re-render LaTeX color-coded, detect symbol regions, vẽ bounding box.
    """
    tokens = latex.split()
    parser = MultiLabelLaTeXParser(tokens)
    symbol_infos = parser.parse()
    
    if not symbol_infos:
        print(f"  ⚠ No symbols parsed from LaTeX: {latex[:60]}")
        return
    
    renderer = ColorCodedLatexRenderer(dpi=dpi)
    colored_img, gray_img = renderer.render_colored(latex, symbol_infos)
    
    if colored_img is None:
        print(f"  ⚠ Rendering failed for: {latex[:60]}")
        return
    
    regions = detect_symbol_regions(colored_img, symbol_infos)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Bounding Box Inspection: {sample_name}\nLaTeX: {latex[:80]}{'...' if len(latex) > 80 else ''}", 
                 fontsize=11, fontweight='bold')
    
    # Left: Color-coded render with bounding boxes
    axes[0].imshow(colored_img)
    axes[0].set_title("Color-coded + BBox", fontsize=10)
    
    for info in symbol_infos:
        if info.index in regions:
            mask = regions[info.index]
            ys, xs = np.where(mask)
            if len(ys) > 0:
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                linewidth=1.5, edgecolor='lime', facecolor='none')
                axes[0].add_patch(rect)
                # Label
                rels = ','.join([r.name[0] for r in info.relations])  # First letter
                axes[0].text(x1, y1 - 2, f"{info.token}[{rels}]", 
                           fontsize=6, color='lime', backgroundcolor='black')
    axes[0].axis('off')
    
    # Right: Grayscale render
    axes[1].imshow(gray_img, cmap='gray')
    axes[1].set_title("Grayscale render", fontsize=10)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved bounding box visualization: {output_path}")


def check_relation_accumulation(relation_map: np.ndarray, latex: str, sample_name: str):
    """
    Bước 4: Kiểm tra sự tích lũy quan hệ.
    In ra các vị trí có nhiều kênh active đồng thời.
    """
    # Count how many channels are active per pixel (skip NONE channel 0)
    active_channels = (relation_map[1:] > 0.1).sum(axis=0)  # (H', W')
    max_overlap = active_channels.max()
    
    print(f"\n  📊 Relation Accumulation for: {sample_name}")
    print(f"     LaTeX: {latex[:80]}{'...' if len(latex) > 80 else ''}")
    print(f"     Relation map shape: {relation_map.shape}")
    print(f"     Max channel overlap: {max_overlap}")
    
    if max_overlap >= 2:
        # Find which channels overlap at the max overlap location
        max_y, max_x = np.unravel_index(active_channels.argmax(), active_channels.shape)
        active_at_max = []
        for ch in range(1, 7):
            if relation_map[ch, max_y, max_x] > 0.1:
                active_at_max.append(f"{CHANNEL_NAMES[ch]}({relation_map[ch, max_y, max_x]:.2f})")
        print(f"     ✓ Multi-label found at ({max_y}, {max_x}): {', '.join(active_at_max)}")
    
    # Per-channel statistics
    for ch in range(1, 7):
        channel_data = relation_map[ch]
        active = (channel_data > 0.1).sum()
        if active > 0:
            print(f"     Ch {ch} ({CHANNEL_NAMES[ch]:12s}): {active:4d} px, "
                  f"max={channel_data.max():.3f}, mean(active)={channel_data[channel_data > 0.1].mean():.3f}")
    
    return max_overlap


def print_statistics_summary(stats: list):
    """In tổng kết thống kê cho tất cả sample."""
    print("\n" + "=" * 70)
    print("STATISTICS SUMMARY")
    print("=" * 70)
    
    total = len(stats)
    if total == 0:
        print("No samples processed.")
        return
    
    # Count channels usage
    channel_counts = {name: 0 for name in CHANNEL_NAMES[1:]}
    empty_gt_count = 0
    multi_label_count = 0
    
    for s in stats:
        rmap = s['relation_map']
        has_any = False
        for ch in range(1, 7):
            if (rmap[ch] > 0.1).any():
                channel_counts[CHANNEL_NAMES[ch]] += 1
                has_any = True
        if not has_any:
            empty_gt_count += 1
        
        # Check multi-label
        active_per_pixel = (rmap[1:] > 0.1).sum(axis=0)
        if active_per_pixel.max() >= 2:
            multi_label_count += 1
    
    print(f"\nTotal samples: {total}")
    print(f"Empty GT (no active channels): {empty_gt_count}")
    print(f"Samples with multi-label: {multi_label_count}")
    print(f"\nChannel usage across samples:")
    for name, count in channel_counts.items():
        bar = "█" * int(count / total * 40) if total > 0 else ""
        print(f"  {name:12s}: {count:4d}/{total} ({count/total*100:5.1f}%) {bar}")
    
    # Warnings
    warnings = []
    if empty_gt_count > total * 0.1:
        warnings.append(f"⚠ {empty_gt_count}/{total} samples have completely empty GT!")
    if channel_counts.get('Horizontal', 0) < total * 0.5:
        warnings.append(f"⚠ Horizontal channel is active in less than 50% of samples")
    
    if warnings:
        print(f"\n⚠ WARNINGS:")
        for w in warnings:
            print(f"  {w}")
    else:
        print(f"\n✓ No anomalies detected")


def get_sample_names(data_dir: Path, cache_dir: Path, split: str, 
                     max_samples: int = -1, complex_only: bool = False,
                     sample_name: str = None):
    """Get list of sample names to process."""
    if sample_name:
        return [sample_name]
    
    # Read caption.txt to get sample names and formulas
    caption_path = data_dir / split / "caption.txt"
    if not caption_path.exists():
        print(f"Error: caption.txt not found at {caption_path}")
        return []
    
    samples = []
    with open(caption_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t') if '\t' in line else line.split(None, 1)
            if len(parts) < 2:
                continue
            name = parts[0]
            formula = parts[1]
            
            # Check if GT exists
            npz_path = cache_dir / split / f"{name}_gt.npz"
            if not npz_path.exists():
                continue
            
            if complex_only:
                # Filter for complex formulas (frac, sqrt, superscript, subscript)
                if any(kw in formula for kw in ['\\frac', '\\sqrt', '^', '_']):
                    samples.append(name)
            else:
                samples.append(name)
    
    if max_samples > 0 and len(samples) > max_samples:
        # Take evenly spaced samples for diversity
        indices = np.linspace(0, len(samples) - 1, max_samples, dtype=int)
        samples = [samples[i] for i in indices]
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Ground Truth maps for SRSC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 5 mẫu ngẫu nhiên
    python scripts/visualize_ground_truth.py --data_dir data --split train --max_samples 5

    # Mẫu phức tạp
    python scripts/visualize_ground_truth.py --data_dir data --split train --complex_only --max_samples 10

    # Mẫu cụ thể
    python scripts/visualize_ground_truth.py --data_dir data --split train --sample_name 200922-1017-140
        """
    )
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--cache_dir', type=str, default='data/cached_maps', help='Path to cached GT maps')
    parser.add_argument('--split', type=str, default='train', 
                        choices=['train', '2014', '2016', '2019'])
    parser.add_argument('--output_dir', type=str, default='/tmp/gt_viz', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=5, help='Max samples (-1 for all)')
    parser.add_argument('--complex_only', action='store_true', help='Only process complex formulas')
    parser.add_argument('--sample_name', type=str, default=None, help='Specific sample name')
    parser.add_argument('--skip_bbox', action='store_true', help='Skip bounding box render (needs xelatex)')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for LaTeX rendering')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GROUND TRUTH VISUALIZATION")
    print("=" * 70)
    print(f"Data dir:   {data_dir}")
    print(f"Cache dir:  {cache_dir}")
    print(f"Split:      {args.split}")
    print(f"Output:     {output_dir}")
    print(f"Complex only: {args.complex_only}")
    print("=" * 70)
    
    # Get sample names
    sample_names = get_sample_names(
        data_dir, cache_dir, args.split, 
        args.max_samples, args.complex_only, args.sample_name
    )
    
    if not sample_names:
        print("No samples found!")
        return
    
    print(f"\nProcessing {len(sample_names)} samples...\n")
    
    all_stats = []
    
    for i, name in enumerate(sample_names):
        print(f"\n[{i+1}/{len(sample_names)}] Processing: {name}")
        print("-" * 50)
        
        relation_map, img, latex = load_gt_and_image(data_dir, cache_dir, args.split, name)
        
        if relation_map is None:
            continue
        
        all_stats.append({
            'name': name,
            'relation_map': relation_map,
            'latex': latex,
        })
        
        # 1+2: Per-channel overlay
        if img is not None:
            overlay_path = output_dir / f"{name}_overlay.png"
            visualize_per_channel_overlay(relation_map, img, latex, name, str(overlay_path))
        else:
            print(f"  ⚠ Skipping overlay (no image)")
        
        # 3: Bounding box inspection
        if not args.skip_bbox and latex:
            bbox_path = output_dir / f"{name}_bbox.png"
            try:
                visualize_bounding_boxes(latex, name, str(bbox_path), dpi=args.dpi)
            except Exception as e:
                print(f"  ⚠ BBox visualization failed: {e}")
        
        # 4: Relation accumulation check
        check_relation_accumulation(relation_map, latex, name)
    
    # 5: Statistics summary
    print_statistics_summary(all_stats)
    
    print(f"\n✓ Output saved to: {output_dir}")
    print(f"  Files generated:")
    output_files = sorted(output_dir.glob("*.png"))
    for f in output_files:
        print(f"    {f.name}")


if __name__ == '__main__':
    main()
