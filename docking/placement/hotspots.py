from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Hotspot:
    kind: str   # "HBA", "HBD", "LIPO", "PI"
    x: float
    y: float
    z: float
    # for PI
    nx: float = 0.0
    ny: float = 0.0
    nz: float = 1.0


def _fit_normal(coords: np.ndarray) -> np.ndarray:
    c = coords.mean(axis=0)
    X = coords - c
    cov = X.T @ X
    w, v = np.linalg.eigh(cov)
    n = v[:, 0]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n.astype(np.float32)


def detect_receptor_hotspots(receptor_atoms) -> List[Hotspot]:
    """
    Cheap hotspots:
      - HBA: ASP/GLU oxygens (+ backbone O proxy via element O)
      - HBD: LYS/ARG nitrogens (+ backbone N proxy via element N)
      - LIPO: C/S atoms proxy
      - PI: aromatic residue centroids (PHE/TYR/TRP/HIS) + plane normal (PCA)
    """
    hotspots: List[Hotspot] = []

    arom_res = {"PHE", "TYR", "TRP", "HIS"}
    arom_groups = {}

    for a in receptor_atoms:
        res = str(getattr(a, "resname", "")).upper()
        name = str(getattr(a, "name", "")).upper()
        el = str(getattr(a, "element", "C")).upper()

        if res in ("ASP", "GLU") and name.startswith("O"):
            hotspots.append(Hotspot("HBA", float(a.x), float(a.y), float(a.z)))

        if res in ("LYS", "ARG") and name.startswith("N"):
            hotspots.append(Hotspot("HBD", float(a.x), float(a.y), float(a.z)))

        if el in ("C", "S"):
            hotspots.append(Hotspot("LIPO", float(a.x), float(a.y), float(a.z)))

        if res in arom_res:
            key = (getattr(a, "chain", ""), int(getattr(a, "resseq", 0)), res)
            arom_groups.setdefault(key, []).append(a)

    # PI hotspots: per aromatic residue centroid + plane normal
    for _, atoms in arom_groups.items():
        coords = np.array([[at.x, at.y, at.z] for at in atoms], dtype=np.float32)
        if coords.shape[0] < 3:
            continue
        c = coords.mean(axis=0)
        n = _fit_normal(coords)
        hotspots.append(Hotspot("PI", float(c[0]), float(c[1]), float(c[2]), float(n[0]), float(n[1]), float(n[2])))

    return hotspots
