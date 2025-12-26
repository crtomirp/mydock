"""docking.geometry.distances

Dependency-free geometry helpers.
"""

from __future__ import annotations

import math


def distance(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def within_cutoff(r: float, cutoff: float) -> bool:
    return (r > 1e-12) and (r <= cutoff)
