"""
Multi-Label Relation Ground Truth Generator
============================================

This module generates MULTI-LABEL relation ground truth where each symbol
can have MULTIPLE relation labels simultaneously.

Example: \sqrt { \frac { g } { \frac { L } { j ^ { n } } } }
  - 'n' has labels: [INSIDE, BELOW, BELOW, SUPERSCRIPT]
    - INSIDE: inside sqrt
    - BELOW: denominator of outer frac
    - BELOW: denominator of inner frac  
    - SUPERSCRIPT: superscript of j

This is essential for accurate multi-channel relation maps.
"""

import os
import tempfile
import subprocess
import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Set
from enum import IntEnum
from dataclasses import dataclass, field
from pathlib import Path
import colorsys


class RelationType(IntEnum):
    """Relation types following the specification."""
    NONE = 0
    HORIZONTAL = 1
    ABOVE = 2
    BELOW = 3
    SUPERSCRIPT = 4
    SUBSCRIPT = 5
    INSIDE = 6
    
    @classmethod
    def num_classes(cls) -> int:
        return 7
    
    @classmethod
    def names(cls) -> List[str]:
        return ['None', 'Horizontal', 'Above', 'Below', 'Superscript', 'Subscript', 'Inside']
    
    @classmethod
    def colors(cls) -> List[Tuple[int, int, int]]:
        return [
            (0, 0, 0),        # NONE - black
            (52, 152, 219),   # HORIZONTAL - blue
            (243, 156, 18),   # ABOVE - orange
            (155, 89, 182),   # BELOW - purple
            (231, 76, 60),    # SUPERSCRIPT - red
            (46, 204, 113),   # SUBSCRIPT - green
            (26, 188, 156),   # INSIDE - teal
        ]


@dataclass
class MultiLabelSymbolInfo:
    """Symbol info with MULTIPLE relation labels."""
    token: str
    relations: Set[RelationType]  # Set of relations (multi-label)
    index: int
    color_rgb: Tuple[int, int, int] = None
    
    def add_relation(self, rel: RelationType):
        """Add a relation to this symbol."""
        self.relations.add(rel)
    
    def has_relation(self, rel: RelationType) -> bool:
        return rel in self.relations
    
    @property
    def relation_names(self) -> List[str]:
        return [RelationType(r).name for r in sorted(self.relations)]


class MultiLabelLaTeXParser:
    """
    Parse LaTeX and track ALL hierarchical relations for each symbol.
    
    Key insight: Relations are ACCUMULATED as we descend the parse tree.
    When entering a structure (frac, sqrt, script), we ADD to the context.
    """
    
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0
        self.symbol_index = 0
        
    def parse(self) -> List[MultiLabelSymbolInfo]:
        """Parse tokens into list of multi-label symbol infos."""
        infos = []
        # Start with empty context set
        initial_context = {RelationType.HORIZONTAL}
        self._parse_sequence(infos, initial_context)
        return infos
    
    def _parse_sequence(self, infos: List[MultiLabelSymbolInfo], context: Set[RelationType]):
        """
        Parse a sequence of tokens.
        
        Args:
            context: Set of relations that apply to all symbols in this sequence
        """
        while self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            
            if token == '}':
                return
            
            if token == '{':
                self.pos += 1
                self._parse_sequence(infos, context)
                if self.pos < len(self.tokens) and self.tokens[self.pos] == '}':
                    self.pos += 1
                continue
            
            if token == '^':
                self.pos += 1
                # Add SUPERSCRIPT to context (inherit existing + add new)
                new_context = context.copy()
                new_context.discard(RelationType.HORIZONTAL)  # Remove baseline
                new_context.add(RelationType.SUPERSCRIPT)
                self._parse_script(infos, new_context)
                continue
                
            if token == '_':
                self.pos += 1
                # Add SUBSCRIPT to context
                new_context = context.copy()
                new_context.discard(RelationType.HORIZONTAL)
                new_context.add(RelationType.SUBSCRIPT)
                self._parse_script(infos, new_context)
                continue
            
            if token == '\\frac':
                self.pos += 1
                self._parse_frac(infos, context)
                continue
            
            if token == '\\sqrt':
                self.pos += 1
                self._parse_sqrt(infos, context)
                continue
            
            # Skip non-rendering tokens
            if token in ['\\limits', '\\left', '\\right', '\\!', '\\,', '\\ ', '\\;', 
                        '\\quad', '\\qquad', '\\lim', '\\displaystyle', '\\textstyle',
                        '\\bigl', '\\bigr', '\\Bigl', '\\Bigr']:
                self.pos += 1
                continue
            
            # Regular symbol - gets ALL relations from context
            infos.append(MultiLabelSymbolInfo(
                token=token, 
                relations=context.copy(), 
                index=self.symbol_index
            ))
            self.symbol_index += 1
            self.pos += 1
    
    def _parse_script(self, infos: List[MultiLabelSymbolInfo], context: Set[RelationType]):
        """Parse superscript or subscript with inherited context."""
        if self.pos >= len(self.tokens):
            return
        
        token = self.tokens[self.pos]
        
        if token == '{':
            self.pos += 1
            self._parse_sequence(infos, context)
            if self.pos < len(self.tokens) and self.tokens[self.pos] == '}':
                self.pos += 1
        else:
            infos.append(MultiLabelSymbolInfo(
                token=token, 
                relations=context.copy(), 
                index=self.symbol_index
            ))
            self.symbol_index += 1
            self.pos += 1
    
    def _parse_frac(self, infos: List[MultiLabelSymbolInfo], parent_context: Set[RelationType]):
        """
        Parse fraction.
        
        Numerator gets ABOVE added to context.
        Denominator gets BELOW added to context.
        """
        # Numerator context
        num_context = parent_context.copy()
        num_context.discard(RelationType.HORIZONTAL)
        num_context.add(RelationType.ABOVE)
        
        if self.pos < len(self.tokens):
            if self.tokens[self.pos] == '{':
                self.pos += 1
                self._parse_sequence(infos, num_context)
                if self.pos < len(self.tokens) and self.tokens[self.pos] == '}':
                    self.pos += 1
            else:
                infos.append(MultiLabelSymbolInfo(
                    token=self.tokens[self.pos], 
                    relations=num_context.copy(), 
                    index=self.symbol_index
                ))
                self.symbol_index += 1
                self.pos += 1
        
        # Denominator context
        denom_context = parent_context.copy()
        denom_context.discard(RelationType.HORIZONTAL)
        denom_context.add(RelationType.BELOW)
        
        if self.pos < len(self.tokens):
            if self.tokens[self.pos] == '{':
                self.pos += 1
                self._parse_sequence(infos, denom_context)
                if self.pos < len(self.tokens) and self.tokens[self.pos] == '}':
                    self.pos += 1
            else:
                infos.append(MultiLabelSymbolInfo(
                    token=self.tokens[self.pos], 
                    relations=denom_context.copy(), 
                    index=self.symbol_index
                ))
                self.symbol_index += 1
                self.pos += 1
    
    def _parse_sqrt(self, infos: List[MultiLabelSymbolInfo], parent_context: Set[RelationType]):
        """
        Parse sqrt.
        
        All content gets INSIDE added to context.
        HORIZONTAL is removed since sqrt content is not on the baseline.
        """
        # Sqrt content context - add INSIDE, remove HORIZONTAL
        sqrt_context = parent_context.copy()
        sqrt_context.discard(RelationType.HORIZONTAL)  # Not on baseline
        sqrt_context.add(RelationType.INSIDE)
        
        # Skip optional [n] for n-th root
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '[':
            self.pos += 1
            while self.pos < len(self.tokens) and self.tokens[self.pos] != ']':
                infos.append(MultiLabelSymbolInfo(
                    token=self.tokens[self.pos], 
                    relations=sqrt_context.copy(), 
                    index=self.symbol_index
                ))
                self.symbol_index += 1
                self.pos += 1
            if self.pos < len(self.tokens) and self.tokens[self.pos] == ']':
                self.pos += 1
        
        if self.pos < len(self.tokens):
            if self.tokens[self.pos] == '{':
                self.pos += 1
                self._parse_sequence(infos, sqrt_context)
                if self.pos < len(self.tokens) and self.tokens[self.pos] == '}':
                    self.pos += 1
            else:
                infos.append(MultiLabelSymbolInfo(
                    token=self.tokens[self.pos], 
                    relations=sqrt_context.copy(), 
                    index=self.symbol_index
                ))
                self.symbol_index += 1
                self.pos += 1


class ColorCodedLatexRenderer:
    """Render LaTeX with each symbol colored uniquely."""
    
    TEMPLATE = r'''
\documentclass[crop]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{fontspec,unicode-math}
\usepackage{xcolor}
\setmathfont{%s}
\begin{document}
\thispagestyle{empty}
$\displaystyle %s$
\end{document}
'''
    
    def __init__(self, dpi: int = 200, font: str = 'Latin Modern Math'):
        self.dpi = dpi
        self.font = font
    
    def _generate_unique_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n distinct RGB colors."""
        colors = []
        for i in range(n):
            hue = i / max(n, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors
    
    def _rgb_to_latex_color(self, rgb: Tuple[int, int, int]) -> str:
        r, g, b = rgb
        return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"
    
    def _colorize_latex(self, tokens: List[str], symbol_infos: List[MultiLabelSymbolInfo]) -> str:
        result_tokens = []
        symbol_idx = 0
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token in ['{', '}', '^', '_', '\\frac', '\\sqrt', '\\limits', 
                        '\\left', '\\right', '\\!', '\\,', '\\ ', '\\;', 
                        '\\quad', '\\qquad', '\\lim', '\\displaystyle', 
                        '\\bigl', '\\bigr', '\\Bigl', '\\Bigr']:
                result_tokens.append(token)
                i += 1
                continue
            
            if symbol_idx < len(symbol_infos):
                info = symbol_infos[symbol_idx]
                rgb = info.color_rgb
                color_def = self._rgb_to_latex_color(rgb)
                
                if token.startswith('\\'):
                    colored = f"{{\\color{color_def}{token}}}"
                else:
                    colored = f"{{\\color{color_def}{{{token}}}}}"
                
                result_tokens.append(colored)
                symbol_idx += 1
            else:
                result_tokens.append(token)
            
            i += 1
        
        return ' '.join(result_tokens)
    
    def render_colored(
        self, 
        latex: str, 
        symbol_infos: List[MultiLabelSymbolInfo]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Render LaTeX with colored symbols."""
        tokens = latex.split()
        
        colors = self._generate_unique_colors(len(symbol_infos))
        for i, info in enumerate(symbol_infos):
            info.color_rgb = colors[i]
        
        colorized_latex = self._colorize_latex(tokens, symbol_infos)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, 'formula.tex')
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(self.TEMPLATE % (self.font, colorized_latex))
            
            try:
                subprocess.run(
                    ['xelatex', '-interaction=nonstopmode', '-output-directory', tmpdir, tex_path],
                    capture_output=True, timeout=30
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return None, None
            
            pdf_path = os.path.join(tmpdir, 'formula.pdf')
            if not os.path.exists(pdf_path):
                return None, None
            
            png_path = os.path.join(tmpdir, 'formula.png')
            try:
                subprocess.run(
                    ['convert', '-density', str(self.dpi), pdf_path, '-quality', '100', png_path],
                    capture_output=True, timeout=30
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return None, None
            
            if not os.path.exists(png_path):
                return None, None
            
            colored_img = cv2.imread(png_path, cv2.IMREAD_COLOR)
            if colored_img is None:
                return None, None
            
            colored_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)
            colored_img = self._crop_whitespace(colored_img)
            gray_img = cv2.cvtColor(colored_img, cv2.COLOR_RGB2GRAY)
            
            return colored_img, gray_img
    
    def _crop_whitespace(self, img: np.ndarray, threshold: int = 250) -> np.ndarray:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        mask = gray < threshold
        if not mask.any():
            return img
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        pad = 5
        rmin = max(0, rmin - pad)
        rmax = min(img.shape[0], rmax + pad + 1)
        cmin = max(0, cmin - pad)
        cmax = min(img.shape[1], cmax + pad + 1)
        
        return img[rmin:rmax, cmin:cmax]


def detect_symbol_regions(
    colored_img: np.ndarray, 
    symbol_infos: List[MultiLabelSymbolInfo],
    tolerance: int = 30
) -> Dict[int, np.ndarray]:
    """Detect regions for each symbol based on their colors."""
    regions = {}
    
    for info in symbol_infos:
        if info.color_rgb is None:
            continue
        
        target_r, target_g, target_b = info.color_rgb
        
        r_match = np.abs(colored_img[:, :, 0].astype(np.int32) - target_r) < tolerance
        g_match = np.abs(colored_img[:, :, 1].astype(np.int32) - target_g) < tolerance
        b_match = np.abs(colored_img[:, :, 2].astype(np.int32) - target_b) < tolerance
        
        mask = r_match & g_match & b_match
        
        if mask.any():
            regions[info.index] = mask
    
    return regions


def gaussian_2d(h: int, w: int) -> np.ndarray:
    """Generate 2D Gaussian blob."""
    if h < 1: h = 1
    if w < 1: w = 1
    
    sigma_x, sigma_y = w / 4.0, h / 4.0
    y = np.arange(h, dtype=np.float32) - (h - 1) / 2.0
    x = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
    xx, yy = np.meshgrid(x, y)
    
    if sigma_x > 0 and sigma_y > 0:
        g = np.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
    else:
        g = np.ones((h, w), dtype=np.float32)
    
    return g / (g.max() + 1e-8)


class MultiLabelRelationMapGenerator:
    """
    Generate MULTI-LABEL relation ground truth.
    
    Each symbol can activate MULTIPLE channels in the relation map,
    reflecting all its hierarchical relations.
    
    Output: [7, H, W] tensor where multiple channels can be active
    for the same spatial location.
    """
    
    def __init__(self,
                 target_height: int = 32,
                 target_width: int = 64,
                 dpi: int = 200,
                 font: str = 'Latin Modern Math',
                 use_gaussian: bool = True):
        self.target_height = target_height
        self.target_width = target_width
        self.use_gaussian = use_gaussian
        self.num_classes = RelationType.num_classes()
        
        self.renderer = ColorCodedLatexRenderer(dpi=dpi, font=font)
    
    def generate(
        self,
        latex: str,
    ) -> Tuple[np.ndarray, List[MultiLabelSymbolInfo], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate multi-label relation map.
        
        Returns:
            relation_map: [7, H, W] multi-channel map (multiple channels can be active)
            symbol_infos: List of symbol info with multi-label relations
            colored_img: Color-coded rendered image
            gray_img: Grayscale rendered image
        """
        # Parse LaTeX with multi-label
        tokens = latex.split()
        parser = MultiLabelLaTeXParser(tokens)
        symbol_infos = parser.parse()
        
        if not symbol_infos:
            return np.zeros((self.num_classes, self.target_height, self.target_width), dtype=np.float32), symbol_infos, None, None
        
        # Render with colors
        colored_img, gray_img = self.renderer.render_colored(latex, symbol_infos)
        
        if colored_img is None:
            return np.zeros((self.num_classes, self.target_height, self.target_width), dtype=np.float32), symbol_infos, None, None
        
        # Detect symbol regions
        regions = detect_symbol_regions(colored_img, symbol_infos)
        
        # Scale factors
        h, w = colored_img.shape[:2]
        scale_y = self.target_height / h
        scale_x = self.target_width / w
        
        # Generate multi-label relation map
        relation_map = np.zeros((self.num_classes, self.target_height, self.target_width), dtype=np.float32)
        
        for info in symbol_infos:
            if info.index not in regions:
                continue
            
            mask = regions[info.index]
            
            # Find bounding box of mask
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            
            # Scale to target size
            ty1 = max(0, int(y1 * scale_y))
            ty2 = min(self.target_height, max(ty1 + 1, int(y2 * scale_y)))
            tx1 = max(0, int(x1 * scale_x))
            tx2 = min(self.target_width, max(tx1 + 1, int(x2 * scale_x)))
            
            if ty2 <= ty1 or tx2 <= tx1:
                continue
            
            # Fill ALL relation channels for this symbol
            for rel in info.relations:
                if self.use_gaussian:
                    g = gaussian_2d(ty2 - ty1, tx2 - tx1)
                    relation_map[int(rel), ty1:ty2, tx1:tx2] = np.maximum(
                        relation_map[int(rel), ty1:ty2, tx1:tx2],
                        g
                    )
                else:
                    relation_map[int(rel), ty1:ty2, tx1:tx2] = 1.0
        
        return relation_map, symbol_infos, colored_img, gray_img
    
    def __call__(self, latex: str) -> torch.Tensor:
        relation_map, _, _, _ = self.generate(latex)
        return torch.from_numpy(relation_map).float()


def test_multilabel_parser():
    """Test multi-label parsing."""
    test_cases = [
        ('\\frac { a } { b }', 
         {'a': ['ABOVE'], 'b': ['BELOW']}),
        
        ('\\sqrt { x ^ { 2 } }',
         {'x': ['INSIDE'], '2': ['INSIDE', 'SUPERSCRIPT']}),
        
        ('\\sqrt { \\frac { g } { \\frac { L } { j ^ { n } } } }',
         {'g': ['INSIDE', 'ABOVE'],
          'L': ['INSIDE', 'BELOW', 'ABOVE'],
          'j': ['INSIDE', 'BELOW', 'BELOW'],
          'n': ['INSIDE', 'BELOW', 'BELOW', 'SUPERSCRIPT']}),
        
        ('x _ { i } ^ { 2 }',
         {'x': ['HORIZONTAL'], 'i': ['SUBSCRIPT'], '2': ['SUPERSCRIPT']}),
    ]
    
    print("Testing Multi-Label LaTeX Parser")
    print("=" * 70)
    
    for latex, expected in test_cases:
        print(f"\nLaTeX: {latex}")
        print("-" * 50)
        
        tokens = latex.split()
        parser = MultiLabelLaTeXParser(tokens)
        infos = parser.parse()
        
        print("Parsed symbols:")
        for info in infos:
            relations_str = ', '.join(sorted([r.name for r in info.relations]))
            print(f"  '{info.token}' → [{relations_str}]")
        
        # Verify expected
        if expected:
            for token, exp_rels in expected.items():
                matching = [i for i in infos if i.token == token]
                if matching:
                    actual_rels = set(r.name for r in matching[0].relations)
                    exp_set = set(exp_rels)
                    if actual_rels == exp_set:
                        print(f"  ✓ '{token}' correct")
                    else:
                        print(f"  ✗ '{token}' expected {exp_set}, got {actual_rels}")


def test_multilabel_generator():
    """Test multi-label generator."""
    generator = MultiLabelRelationMapGenerator(dpi=200)
    
    latex = '\\sqrt { \\frac { g } { \\frac { L } { j ^ { n } } } }'
    
    print("\n" + "=" * 70)
    print("Testing Multi-Label Generator")
    print("=" * 70)
    print(f"LaTeX: {latex}\n")
    
    relation_map, symbol_infos, colored_img, gray_img = generator.generate(latex)
    
    if colored_img is not None:
        print(f"Colored image: {colored_img.shape}")
        print(f"Relation map: {relation_map.shape}")
        
        print("\nSymbol → Relations:")
        for info in symbol_infos:
            rels = ', '.join(sorted([r.name for r in info.relations]))
            print(f"  '{info.token}' → [{rels}]")
        
        print("\nActive channels in relation map:")
        for i in range(1, relation_map.shape[0]):
            active = (relation_map[i] > 0.1).sum()
            if active > 0:
                print(f"  {RelationType.names()[i]}: {active} pixels")
    else:
        print("✗ Rendering failed")


if __name__ == '__main__':
    test_multilabel_parser()
    test_multilabel_generator()
