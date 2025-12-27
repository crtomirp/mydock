from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def prepare_receptor_arrays(receptor_atoms):
    """
    Convert receptor atom objects -> numpy arrays for Numba kernels.
    Returns:
      R_xyz (float32, [nR,3])
      R_q   (float32, [nR])
      R_vdw (float32, [nR])
      R_hba (uint8,   [nR])  acceptor flag
      R_hbd (uint8,   [nR])  donor flag
      R_lipo(uint8,   [nR])  lipophilic atom flag (carbon/sulfur proxy)
    """
    n = len(receptor_atoms)
    R_xyz = np.empty((n, 3), dtype=np.float32)
    R_q = np.empty((n,), dtype=np.float32)
    R_vdw = np.empty((n,), dtype=np.float32)
    R_hba = np.zeros((n,), dtype=np.uint8)
    R_hbd = np.zeros((n,), dtype=np.uint8)
    R_lipo = np.zeros((n,), dtype=np.uint8)

    for i, a in enumerate(receptor_atoms):
        R_xyz[i, 0] = float(a.x)
        R_xyz[i, 1] = float(a.y)
        R_xyz[i, 2] = float(a.z)
        R_q[i] = float(getattr(a, "charge", 0.0))
        R_vdw[i] = float(getattr(a, "vdw_radius", 1.7))
        el = str(getattr(a, "element", "C")).upper()

        if el in ("O", "N", "S"):
            R_hba[i] = 1  # proxy; refined selection can be done in python engine
        if el in ("N",):
            R_hbd[i] = 1  # proxy
        if el in ("C", "S"):
            R_lipo[i] = 1

    return R_xyz, R_q, R_vdw, R_hba, R_hbd, R_lipo


def prepare_ligand_static(ligand_atoms, ligand_rdkit_mol=None):
    """
    Ligand static arrays (charges, vdw, flags) that don't change with pose.
    Returns:
      L_q, L_vdw, L_hba, L_hbd, L_lipo
    """
    n = len(ligand_atoms)
    L_q = np.empty((n,), dtype=np.float32)
    L_vdw = np.empty((n,), dtype=np.float32)
    L_hba = np.zeros((n,), dtype=np.uint8)
    L_hbd = np.zeros((n,), dtype=np.uint8)
    L_lipo = np.zeros((n,), dtype=np.uint8)

    for i, a in enumerate(ligand_atoms):
        L_q[i] = float(getattr(a, "charge", 0.0))
        L_vdw[i] = float(getattr(a, "vdw_radius", 1.7))
        el = str(getattr(a, "element", "C")).upper()

        if el in ("O", "N", "S"):
            L_hba[i] = 1
        if el in ("N",):
            L_hbd[i] = 1
        if el in ("C", "S"):
            L_lipo[i] = 1

    return L_q, L_vdw, L_hba, L_hbd, L_lipo


def _ring_normal_and_centroid(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # coords: (k,3)
    c = coords.mean(axis=0)
    X = coords - c
    # normal from smallest eigenvector of covariance (plane normal)
    cov = X.T @ X
    w, v = np.linalg.eigh(cov)
    n = v[:, 0]
    # normalize
    n = n / (np.linalg.norm(n) + 1e-12)
    return n.astype(np.float32), c.astype(np.float32)


def prepare_pi_features(receptor_atoms, ligand_rdkit_mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute aromatic ring centroids and normals for receptor (by aromatic residues)
    and ligand (RDKit rings).

    Returns:
      R_pi_c (float32, [nRp,3])
      R_pi_n (float32, [nRp,3])
      L_pi_c (float32, [nLp,3])
      L_pi_n (float32, [nLp,3])
    """
    # --- receptor: group by (chain, resseq, resname) for aromatic residues
    arom_res = {"PHE", "TYR", "TRP", "HIS"}
    groups = {}
    for a in receptor_atoms:
        if str(getattr(a, "resname", "")).upper() not in arom_res:
            continue
        key = (getattr(a, "chain", ""), int(getattr(a, "resseq", 0)), getattr(a, "resname", ""))
        groups.setdefault(key, []).append(a)

    R_centroids = []
    R_normals = []
    for _, atoms in groups.items():
        coords = np.array([[at.x, at.y, at.z] for at in atoms], dtype=np.float32)
        if coords.shape[0] < 3:
            continue
        n, c = _ring_normal_and_centroid(coords)
        R_centroids.append(c)
        R_normals.append(n)

    if R_centroids:
        R_pi_c = np.stack(R_centroids).astype(np.float32)
        R_pi_n = np.stack(R_normals).astype(np.float32)
    else:
        R_pi_c = np.zeros((0, 3), dtype=np.float32)
        R_pi_n = np.zeros((0, 3), dtype=np.float32)

    # --- ligand: RDKit rings
    L_centroids = []
    L_normals = []
    if ligand_rdkit_mol is not None and ligand_rdkit_mol.GetNumConformers() > 0:
        conf = ligand_rdkit_mol.GetConformer()
        ring_info = ligand_rdkit_mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            # use only aromatic rings
            if not all(ligand_rdkit_mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                continue
            coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                               for i in ring], dtype=np.float32)
            if coords.shape[0] < 3:
                continue
            n, c = _ring_normal_and_centroid(coords)
            L_centroids.append(c)
            L_normals.append(n)

    if L_centroids:
        L_pi_c = np.stack(L_centroids).astype(np.float32)
        L_pi_n = np.stack(L_normals).astype(np.float32)
    else:
        L_pi_c = np.zeros((0, 3), dtype=np.float32)
        L_pi_n = np.zeros((0, 3), dtype=np.float32)

    return R_pi_c, R_pi_n, L_pi_c, L_pi_n
