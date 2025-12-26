"""docking.scoring.vdw

Lennard-Jones (12-6) van der Waals scoring term.
Mixing rules: Lorentz-Berthelot.
"""

from __future__ import annotations

import math

from ..chemistry.parameters import get_lj_params
from ..geometry.distances import distance, within_cutoff


def lj_energy(r: float, sigma: float, epsilon: float) -> float:
    if r <= 1e-12:
        return 0.0
    sr6 = (sigma / r) ** 6
    sr12 = sr6 * sr6
    return 4.0 * epsilon * (sr12 - sr6)


def vdw_score_atoms(receptor_atoms, ligand_atoms, cutoff: float = 9.0) -> float:
    e = 0.0
    for ra in receptor_atoms:
        pi = get_lj_params(getattr(ra, "element", "X"))
        for la in ligand_atoms:
            pj = get_lj_params(getattr(la, "element", "X"))
            r = distance(ra.x, ra.y, ra.z, la.x, la.y, la.z)
            if not within_cutoff(r, cutoff):
                continue
            sigma = 0.5 * (pi.sigma + pj.sigma)
            epsilon = math.sqrt(pi.epsilon * pj.epsilon)
            e += lj_energy(r, sigma, epsilon)
    return e
