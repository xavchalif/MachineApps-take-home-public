"""
Palletizing Grid Calculations
============================

Calculate place positions for boxes in an N×M grid pattern.
"""

from typing import List, Tuple


def calculate_place_positions(
    rows: int,
    cols: int,
    box_size_mm: Tuple[float, float, float],
    pallet_origin_mm: Tuple[float, float, float],
    spacing_mm: float = 10.0,
) -> List[Tuple[float, float, float]]:
    """
    Calculate TCP positions for placing boxes in a grid pattern.
    
    Args:
        rows: Number of rows (N)
        cols: Number of columns (M)
        box_size_mm: (width, depth, height) of each box in mm
        pallet_origin_mm: (x, y, z) position of the first box placement
        spacing_mm: Gap between adjacent boxes (default 10mm)
    
    Returns:
        List of (x, y, z) TCP target positions, ordered for row-by-row filling.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if spacing_mm < 0:
        raise ValueError("spacing_mm must be non-negative")
    width, depth, height = box_size_mm
    if width <= 0 or depth <= 0 or height <= 0:
        raise ValueError("all box dimensions must be positive")

    origin_x, origin_y, origin_z = pallet_origin_mm
    positions: List[Tuple[float, float, float]] = []
    for row in range(rows):
        for col in range(cols):
            positions.append((
                origin_x + col * (width + spacing_mm),
                origin_y + row * (depth + spacing_mm),
                origin_z,
            ))
    return positions
