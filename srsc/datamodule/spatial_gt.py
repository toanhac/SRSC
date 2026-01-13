import torch
import numpy as np
import cv2
from typing import List, Dict
import warnings

try:
    import skimage.filters as filters
    SAUVOLA_AVAILABLE = True
except ImportError:
    SAUVOLA_AVAILABLE = False
    warnings.warn("skimage not available. Using adaptive thresholding instead of Sauvola.")


class SpatialMapGenerator:
    def __init__(self, 
                 img_height: int = 256,
                 img_width: int = 512,
                 target_height: int = 32,
                 target_width: int = 64,
                 min_component_size: int = 5):
        self.img_height = img_height
        self.img_width = img_width
        self.target_height = target_height
        self.target_width = target_width
        self.min_component_size = min_component_size
        self.scale_y = target_height / img_height
        self.scale_x = target_width / img_width
    
    def extract_symbol_components(self, image: np.ndarray) -> List[Dict]:
        image_mean = np.mean(image)
        is_inverted = image_mean < 128
        
        if SAUVOLA_AVAILABLE:
            threshold = filters.threshold_sauvola(image, 43, 0.043)
            if is_inverted:
                binary = (image > threshold).astype(np.uint8) * 255
            else:
                binary = (image < threshold).astype(np.uint8) * 255
        else:
            if is_inverted:
                binary = cv2.adaptiveThreshold(
                    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
            else:
                binary = cv2.adaptiveThreshold(
                    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        components = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < self.min_component_size:
                continue
            component_mask = (labels[y:y+h, x:x+w] == i).astype(np.float32)
            cx, cy = centroids[i]
            components.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'cx': cx, 'cy': cy,
                'area': area, 'mask': component_mask,
            })
        
        components.sort(key=lambda c: c['area'], reverse=True)
        return components
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.squeeze(0)
        
        img_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        components = self.extract_symbol_components(img_np)
        
        if len(components) == 0:
            return torch.zeros(1, self.target_height, self.target_width, dtype=torch.float32)
        
        h, w = img_np.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.float32)
        
        for comp in components:
            x, y = comp['x'], comp['y']
            comp_w, comp_h = comp['w'], comp['h']
            component_mask = comp['mask']
            
            x_end = min(x + comp_w, w)
            y_end = min(y + comp_h, h)
            actual_w = x_end - x
            actual_h = y_end - y
            
            if actual_w > 0 and actual_h > 0:
                mask_portion = component_mask[:actual_h, :actual_w]
                full_mask[y:y_end, x:x_end] = np.maximum(
                    full_mask[y:y_end, x:x_end],
                    mask_portion
                )
        
        spatial_map = cv2.resize(
            full_mask, 
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA
        )
        
        spatial_map = (spatial_map > 0.1).astype(np.float32)
        
        return torch.from_numpy(spatial_map[np.newaxis, ...]).float()


class AuxiliaryTargetGenerator:
    def __init__(self,
                 img_height: int = 256,
                 img_width: int = 512,
                 target_height: int = 32,
                 target_width: int = 64,
                 device: str = 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.img_height = img_height
        self.img_width = img_width
        self.target_height = target_height
        self.target_width = target_width
        self.spatial_gen = SpatialMapGenerator(
            img_height, img_width, target_height, target_width
        )
    
    def __call__(self, image: torch.Tensor, latex: str = None) -> torch.Tensor:
        return self.spatial_gen(image)
    
    def batch_generate(self, images: torch.Tensor, latex_strings: List[str] = None) -> torch.Tensor:
        B = images.size(0)
        if images.dim() == 4:
            images = images.squeeze(1)
        
        S_list = []
        for b in range(B):
            S = self(images[b])
            S_list.append(S)
        
        return torch.stack(S_list, dim=0)
