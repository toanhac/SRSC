import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from srsc.datamodule.dataset import CROHMEDataset
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader

from .vocab import vocab
from .spatial_gt import AuxiliaryTargetGenerator

Data = List[Tuple[str, Image.Image, List[str]]]

import cv2

cv2.setNumThreads(0)

MAX_SIZE = 32e5  # Increased from 32e4 to allow larger batches
ENCODER_DOWNSAMPLE_FACTOR = 16


def compute_encoder_output_size(img_height, img_width, factor=ENCODER_DOWNSAMPLE_FACTOR):
    h = img_height
    w = img_width
    for _ in range(4):
        h = (h + 1) // 2
        w = (w + 1) // 2
    return h, w


def data_iterator(
    data: Data,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE,
    maxlen: int = 200,
    maxImagesize: int = MAX_SIZE,
):
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    data.sort(key=lambda x: x[1].size[0] * x[1].size[1])

    i = 0
    for fname, fea, lab in data:
        size = fea.size[0] * fea.size[1]
        fea = np.array(fea)
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size > maxImagesize:
            print(
                f"image: {fname} size: {fea.shape[0]} x {fea.shape[1]} =  bigger than {maxImagesize}, ignore"
            )
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:
                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                fname_batch = []
                feature_batch = []
                label_batch = []
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    
    # Debug: Print batch size distribution
    batch_sizes = [len(b) for b in feature_total]
    print(f"total {len(feature_total)} batches loaded")
    print(f"  Batch size distribution: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={sum(batch_sizes)/len(batch_sizes):.1f}")
    print(f"  Batches with size 1: {sum(1 for s in batch_sizes if s == 1)}")
    
    return list(zip(fname_total, feature_total, label_total))


def extract_data(archive: ZipFile, dir_name: str) -> Data:
    with archive.open(f"data/{dir_name}/caption.txt", "r") as f:
        captions = f.readlines()
    data = []
    for line in captions:
        tmp = line.decode().strip().split()
        img_name = tmp[0]
        formula = tmp[1:]
        with archive.open(f"data/{dir_name}/img/{img_name}.bmp", "r") as f:
            img = Image.open(f).copy()
        data.append((img_name, img, formula))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return data


@dataclass
class Batch:
    img_bases: List[str]
    imgs: FloatTensor
    mask: LongTensor
    indices: List[List[int]]
    spatial_map: Optional[FloatTensor] = None
    relation_map: Optional[FloatTensor] = None  # Multi-label relation map [B, 7, H, W]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        spatial = self.spatial_map.to(device) if self.spatial_map is not None else None
        relation = self.relation_map.to(device) if self.relation_map is not None else None
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
            spatial_map=spatial,
            relation_map=relation,
        )



def collate_fn(batch):
    assert len(batch) == 1
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x) for x in batch[2]]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

    return Batch(fnames, x, x_mask, seqs_y, None)


class MultiTaskCollator:
    """
    Collator that loads both spatial and relation ground truth maps.
    
    Supports two modes:
    1. Load from cached .npz files (pregenerated) - faster
    2. Generate on-the-fly (slower, only for spatial)
    
    Ground truth files should be named: {img_name}_gt.npz
    With keys: 'spatial_map', 'relation_map'
    
    Uses in-memory caching to avoid repeated I/O.
    """
    
    NUM_RELATION_CLASSES = 7
    
    # Class-level cache shared across instances
    _gt_cache = {}
    _file_exists_cache = {}  # Cache for file existence checks
    
    def __init__(
        self, 
        cache_dir: Optional[str] = None, 
        generate_gt_on_fly: bool = True,
        use_relation: bool = True,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.generate_gt_on_fly = generate_gt_on_fly
        self.use_relation = use_relation
        
        # Preload file existence info to avoid repeated .exists() calls
        if self.cache_dir is not None and str(self.cache_dir) not in MultiTaskCollator._file_exists_cache:
            existing_files = set()
            if self.cache_dir.exists():
                for f in self.cache_dir.iterdir():
                    existing_files.add(f.name)
            MultiTaskCollator._file_exists_cache[str(self.cache_dir)] = existing_files
            print(f"  Preloaded {len(existing_files)} cache files from {self.cache_dir}")
    
    def __call__(self, batch):
        assert len(batch) == 1
        batch = batch[0]
        fnames = batch[0]
        images_x = batch[1]
        seqs_y = [vocab.words2indices(x) for x in batch[2]]

        heights_x = [s.size(1) for s in images_x]
        widths_x = [s.size(2) for s in images_x]

        n_samples = len(heights_x)
        max_height_x = max(heights_x)
        max_width_x = max(widths_x)

        x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
        x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
        
        enc_h, enc_w = compute_encoder_output_size(max_height_x, max_width_x)
        spatial_maps = torch.zeros(n_samples, 1, enc_h, enc_w)
        relation_maps = torch.zeros(n_samples, self.NUM_RELATION_CLASSES, enc_h, enc_w)
        
        for idx, s_x in enumerate(images_x):
            h, w = heights_x[idx], widths_x[idx]
            x[idx, :, :h, :w] = s_x
            x_mask[idx, :h, :w] = 0
            
            spatial_map = None
            relation_map = None
            fname = fnames[idx]
            
            # Check in-memory cache first
            cache_key = f"{self.cache_dir}_{fname}" if self.cache_dir else fname
            if cache_key in MultiTaskCollator._gt_cache:
                cached = MultiTaskCollator._gt_cache[cache_key]
                spatial_map = cached.get('spatial')
                if self.use_relation:
                    relation_map = cached.get('relation')
            elif self.cache_dir is not None:
                # Use preloaded file existence cache (fast set lookup instead of .exists())
                cache_dir_key = str(self.cache_dir)
                existing_files = MultiTaskCollator._file_exists_cache.get(cache_dir_key, set())
                
                gt_filename = f"{fname}_gt.npz"
                if gt_filename in existing_files:
                    cache_path = self.cache_dir / gt_filename
                    try:
                        data = np.load(cache_path, allow_pickle=True)
                        cache_entry = {}
                        if 'spatial_map' in data.files:
                            spatial_map = torch.from_numpy(data['spatial_map'].astype(np.float32))
                            cache_entry['spatial'] = spatial_map
                        if 'relation_map' in data.files:
                            relation_map = torch.from_numpy(data['relation_map'].astype(np.float32))
                            cache_entry['relation'] = relation_map
                        MultiTaskCollator._gt_cache[cache_key] = cache_entry
                    except Exception as e:
                        pass
                else:
                    # Try old format: _spatial.npz
                    old_filename = f"{fname}_spatial.npz"
                    if old_filename in existing_files:
                        old_cache_path = self.cache_dir / old_filename
                        try:
                            data = np.load(old_cache_path)
                            spatial_map = torch.from_numpy(data['spatial_map'].astype(np.float32))
                            MultiTaskCollator._gt_cache[cache_key] = {'spatial': spatial_map}
                        except:
                            pass
            
            # Generate spatial on-the-fly if needed (relation requires pregeneration)
            if spatial_map is None and self.generate_gt_on_fly:
                target_h, target_w = compute_encoder_output_size(h, w)
                target_h = max(target_h, 1)
                target_w = max(target_w, 1)
                generator = AuxiliaryTargetGenerator(
                    img_height=h,
                    img_width=w,
                    target_height=target_h,
                    target_width=target_w,
                )
                img_tensor = s_x.squeeze(0)
                spatial_map = generator(img_tensor)
            
            # Place spatial map in batch tensor
            if spatial_map is not None:
                sm_h, sm_w = spatial_map.shape[1], spatial_map.shape[2]
                if sm_h <= enc_h and sm_w <= enc_w:
                    spatial_maps[idx, :, :sm_h, :sm_w] = spatial_map
                else:
                    resized = F.interpolate(
                        spatial_map.unsqueeze(0), 
                        size=(enc_h, enc_w), 
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    spatial_maps[idx] = resized
            
            # Place relation map in batch tensor
            if relation_map is not None:
                rm_h, rm_w = relation_map.shape[1], relation_map.shape[2]
                if rm_h <= enc_h and rm_w <= enc_w:
                    relation_maps[idx, :, :rm_h, :rm_w] = relation_map
                else:
                    resized = F.interpolate(
                        relation_map.unsqueeze(0), 
                        size=(enc_h, enc_w), 
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    relation_maps[idx] = resized

        return Batch(fnames, x, x_mask, seqs_y, spatial_maps, relation_maps)


# Backward compatibility alias
SpatialCollator = MultiTaskCollator


def build_dataset(archive, folder: str, batch_size: int):
    data = extract_data(archive, folder)
    return data_iterator(data, batch_size)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data.zip",
        test_year: str = "2014",
        train_batch_size: int = 8,
        eval_batch_size: int = 4,
        num_workers: int = 5,
        scale_aug: bool = False,
        # Multi-task learning options
        use_spatial_maps: bool = False,
        use_relation_maps: bool = False,
        gt_cache_dir: Optional[str] = None,  # Unified cache dir for both spatial and relation
        generate_gt_on_fly: bool = True,  # Renamed from generate_spatial_on_fly
        # Legacy options (backward compatibility)
        spatial_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        self.zipfile_path = zipfile_path
        self.test_year = test_year
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug
        
        # Multi-task configuration
        self.use_spatial_maps = use_spatial_maps
        self.use_relation_maps = use_relation_maps
        self.gt_cache_dir = gt_cache_dir or spatial_cache_dir  # Use unified or legacy
        self.generate_gt_on_fly = generate_gt_on_fly

        print(f"Load data from: {self.zipfile_path}")
        if use_spatial_maps or use_relation_maps:
            print(f"Multi-task learning enabled:")
            print(f"  Spatial maps: {use_spatial_maps}")
            print(f"  Relation maps: {use_relation_maps}")
            print(f"  Cache dir: {self.gt_cache_dir}")
            print(f"  Generate GT on-the-fly: {self.generate_gt_on_fly}")

    def setup(self, stage: Optional[str] = None) -> None:
        with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                self.train_dataset = CROHMEDataset(
                    build_dataset(archive, "train", self.train_batch_size),
                    True,
                    self.scale_aug,
                )
                self.val_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size),
                    False,
                    self.scale_aug,
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    build_dataset(archive, self.test_year, self.eval_batch_size),
                    False,
                    self.scale_aug,
                )

    def _get_collate_fn(self, split: str = "train"):
        if self.use_spatial_maps or self.use_relation_maps:
            cache_dir = None
            if self.gt_cache_dir:
                cache_dir = Path(self.gt_cache_dir) / split
            return MultiTaskCollator(
                cache_dir=cache_dir,
                generate_gt_on_fly=self.generate_gt_on_fly,
                use_relation=self.use_relation_maps,
            )
        return collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn("train"),
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(self.test_year),
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(self.test_year),
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
