"""
mapgen/map_loader.py — Load a Nav2 occupancy map into a GridMap dataclass.

Nav2 coordinate convention (critical, often confused):
    - `origin` in the YAML = world (x, y) of the LOWER-LEFT corner of the grid.
    - PGM rows are stored TOP-to-BOTTOM (row 0 = highest y in world).
    - So: world_y of PGM row r = origin_y + (H - 1 - r) * res

Correct conversions:
    world → cell:  col = int((wx - ox) / res)
                   row = H - 1 - int((wy - oy) / res)

    cell → world:  wx  = ox + (col + 0.5) * res
                   wy  = oy + (H - 1 - row + 0.5) * res

Bugs in older code used row = int((wy - oy) / res) which is correct for
ROS OccupancyGrid (costmap) but WRONG for PGM files.  GridMap always uses
the PGM convention because all offline map work starts from the YAML + PGM.

Usage:
    gmap = GridMap.load('maps/house_my1_map/house_my1_map.yaml')
    cell = gmap.world_to_cell(3.0, -1.5)   # (row, col) or None if OOB
    wx, wy = gmap.cell_to_world(120, 220)  # cell-centre in metres
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import binary_erosion


# Default inflation radius — robot half-width (0.20 m) + 0.10 m navigation margin.
DEFAULT_INFLATION_M = 0.30


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha1(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _disk_struct(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


# ── GridMap ───────────────────────────────────────────────────────────────────

@dataclass
class GridMap:
    """
    Binary free-space grid loaded from a Nav2 map YAML + PGM.

    Attributes:
        free      : (H, W) bool array; True = navigable after inflation.
        res       : metres per cell.
        origin_x  : world x of the LEFT edge of the left column.
        origin_y  : world y of the BOTTOM edge of the bottom row.
        sha1      : SHA1 of the source PGM — used for cache validation.
    """
    free: np.ndarray
    res: float
    origin_x: float
    origin_y: float
    sha1: str

    @property
    def height(self) -> int:
        return int(self.free.shape[0])

    @property
    def width(self) -> int:
        return int(self.free.shape[1])

    # ── Coordinate conversions ────────────────────────────────────────────────

    def world_to_cell(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        """
        World (x, y) → grid (row, col).
        Row 0 = top of PGM = highest y in world.
        Returns None if the point is outside the grid.
        """
        col = int((wx - self.origin_x) / self.res)
        # y increases upward in world; row increases downward in PGM.
        row = self.height - 1 - int((wy - self.origin_y) / self.res)
        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col
        return None

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """
        Grid (row, col) → world (x, y) at cell centre.
        """
        wx = self.origin_x + (col + 0.5) * self.res
        wy = self.origin_y + (self.height - 1 - row + 0.5) * self.res
        return wx, wy

    def is_free(self, wx: float, wy: float) -> bool:
        """True if the world point is inside the grid and on a free cell."""
        cell = self.world_to_cell(wx, wy)
        if cell is None:
            return False
        return bool(self.free[cell])

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        yaml_path: str,
        inflation_radius_m: float = DEFAULT_INFLATION_M,
    ) -> 'GridMap':
        """
        Load map from a Nav2 YAML + PGM file pair.

        Args:
            yaml_path         : path to the .yaml map metadata file.
            inflation_radius_m: obstacles are inflated by this radius before
                                the free grid is returned.  Use the robot's
                                worst-case half-extent so sampled paths stay
                                clear of walls.
        """
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        pgm_path = cfg['image']
        if not os.path.isabs(pgm_path):
            pgm_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), pgm_path)

        res        = float(cfg['resolution'])
        origin_x   = float(cfg['origin'][0])
        origin_y   = float(cfg['origin'][1])
        negate     = int(cfg.get('negate', 0))
        free_thresh = float(cfg.get('free_thresh', 0.25))

        pgm_sha1 = _sha1(pgm_path)
        img = np.array(Image.open(pgm_path).convert('L'), dtype=np.float32)
        occ = img / 255.0
        if negate:
            occ = 1.0 - occ

        # Cells below free_thresh are known-free; others (inflation band,
        # unknown, occupied) are treated as obstacles for planning.
        free = occ < free_thresh

        # Inflate obstacles by eroding the free grid.
        r_cells = int(math.ceil(inflation_radius_m / res))
        if r_cells > 0:
            free = binary_erosion(free, structure=_disk_struct(r_cells), border_value=0)

        return cls(
            free=free.astype(bool),
            res=res,
            origin_x=origin_x,
            origin_y=origin_y,
            sha1=pgm_sha1,
        )

    # ── Debug ──────────────────────────────────────────────────────────────────

    def save_debug_png(self, out_path: str) -> None:
        """Save the free grid as a greyscale PNG (white=free, black=obstacle)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.free, cmap='gray', origin='upper',
                  extent=[self.origin_x,
                          self.origin_x + self.width  * self.res,
                          self.origin_y,
                          self.origin_y + self.height * self.res])
        ax.set_xlabel('world x (m)')
        ax.set_ylabel('world y (m)')
        ax.set_title(f'Free grid  {self.width}×{self.height}  res={self.res} m')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Saved debug PNG: {out_path}')
