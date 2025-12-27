from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from rdkit.Chem import Lipinski


@dataclass
class Anchor:
    kind: str                 # "HBA", "HBD", "PI"
    atom_indices: List[int]
    cx: float
    cy: float
    cz: float
    nx: float = 0.0
    ny: float = 0.0
    nz: float = 1.0


def _ring_normal(coords: np.ndarray) -> np.ndarray:
    c = coords.mean(axis=0)
    X = coords - c
    cov = X.T @ X
    w, v = np.linalg.eigh(cov)
    n = v[:, 0]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n.astype(np.float32)


def detect_ligand_anchors(mol) -> List[Anchor]:
    anchors: List[Anchor] = []
    if mol is None or mol.GetNumConformers() == 0:
        return anchors

    conf = mol.GetConformer()

    # HBD/HBA from RDKit Lipinski: returns counts; we need indices
    # We'll implement a simple proxy: any N/O/S atom that has H neighbors is donor; N/O/S is acceptor
    for a in mol.GetAtoms():
        el = a.GetSymbol()
        if el not in ("N", "O", "S"):
            continue
        idx = a.GetIdx()
        p = conf.GetAtomPosition(idx)
        # acceptor proxy
        anchors.append(Anchor("HBA", [idx], float(p.x), float(p.y), float(p.z)))
        # donor proxy
        if any(nb.GetSymbol() == "H" for nb in a.GetNeighbors()):
            anchors.append(Anchor("HBD", [idx], float(p.x), float(p.y), float(p.z)))

    # PI anchors: aromatic rings
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                           for i in ring], dtype=np.float32)
        if coords.shape[0] < 3:
            continue
        c = coords.mean(axis=0)
        n = _ring_normal(coords)
        anchors.append(Anchor("PI", list(ring), float(c[0]), float(c[1]), float(c[2]), float(n[0]), float(n[1]), float(n[2])))

    return anchors
