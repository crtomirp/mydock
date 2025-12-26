"""docking.chemistry.vdw_radii

Simple element-based van der Waals radii (Å).
Used for distance cutoffs in piecewise linear potentials.
"""

from __future__ import annotations


VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "X": 1.70,
}


def _norm_el(e: str) -> str:
    e = (e or "").strip()
    if not e:
        return "X"
    if len(e) == 1:
        return e.upper()
    return e[0].upper() + e[1:].lower()


def vdw_radius(element: str) -> float:
    return float(VDW_RADII.get(_norm_el(element), VDW_RADII["X"]))
