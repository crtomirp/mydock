"""docking.scoring.hydrogen_bonds

Heuristic (distance-only) hydrogen bond bonus term for MVP.
Angle criteria can be added later.
"""

from __future__ import annotations

from ..geometry.distances import distance

HB_ELEMENTS = {"N", "O", "S"}


def hydrogen_bond_score_atoms(receptor_atoms, ligand_atoms, dist_cutoff: float = 3.5, bonus: float = -1.5) -> float:
    e = 0.0
    for ra in receptor_atoms:
        if getattr(ra, "element", "X") not in HB_ELEMENTS:
            continue
        for la in ligand_atoms:
            if getattr(la, "element", "X") not in HB_ELEMENTS:
                continue
            r = distance(ra.x, ra.y, ra.z, la.x, la.y, la.z)
            if 1e-12 < r <= dist_cutoff:
                e += bonus
    return e
