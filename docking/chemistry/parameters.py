"""docking.chemistry.parameters

Minimal Lennard-Jones parameter table (element-based).
Values are placeholders for MVP development.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .atom_types import normalize_element


@dataclass(frozen=True)
class LJParams:
    sigma: float
    epsilon: float


LJ_TABLE: Dict[str, LJParams] = {
    "H":  LJParams(sigma=1.20, epsilon=0.015),
    "C":  LJParams(sigma=3.40, epsilon=0.055),
    "N":  LJParams(sigma=3.30, epsilon=0.070),
    "O":  LJParams(sigma=3.00, epsilon=0.120),
    "S":  LJParams(sigma=3.60, epsilon=0.200),
    "P":  LJParams(sigma=3.75, epsilon=0.200),
    "F":  LJParams(sigma=3.10, epsilon=0.060),
    "Cl": LJParams(sigma=3.80, epsilon=0.100),
    "Br": LJParams(sigma=4.10, epsilon=0.120),
    "I":  LJParams(sigma=4.50, epsilon=0.150),
    "X":  LJParams(sigma=3.50, epsilon=0.050),
}


def get_lj_params(element: str) -> LJParams:
    return LJ_TABLE.get(normalize_element(element), LJ_TABLE["X"])
