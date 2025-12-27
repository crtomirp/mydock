"""docking.geometry.pocket

Binding-site selection utilities.

Pocket modes:
- centroid: receptor atoms within radius from ligand centroid
- atoms:    receptor atoms within radius from ANY ligand atom (union of spheres)
"""

from __future__ import annotations

from typing import List, Tuple


def ligand_centroid(ligand_atoms) -> Tuple[float, float, float]:
    if not ligand_atoms:
        return (0.0, 0.0, 0.0)
    sx = sy = sz = 0.0
    n = 0
    for a in ligand_atoms:
        sx += float(a.x)
        sy += float(a.y)
        sz += float(a.z)
        n += 1
    return (sx / n, sy / n, sz / n)


def select_receptor_atoms_sphere(receptor_atoms, center_xyz: Tuple[float, float, float], radius: float) -> List:
    cx, cy, cz = center_xyz
    r2 = radius * radius
    sel = []
    for ra in receptor_atoms:
        dx = ra.x - cx
        dy = ra.y - cy
        dz = ra.z - cz
        if (dx * dx + dy * dy + dz * dz) <= r2:
            sel.append(ra)
    return sel


def select_receptor_atoms_union_spheres(receptor_atoms, ligand_atoms, radius: float) -> List:
    """Union of spheres around each ligand atom (fast O(N*M) baseline)."""
    r2 = radius * radius
    sel = []
    for ra in receptor_atoms:
        keep = False
        for la in ligand_atoms:
            dx = ra.x - la.x
            dy = ra.y - la.y
            dz = ra.z - la.z
            if (dx * dx + dy * dy + dz * dz) <= r2:
                keep = True
                break
        if keep:
            sel.append(ra)
    return sel


def define_pocket_from_ligand(
    receptor_atoms,
    ligand_atoms,
    pocket_radius: float = 12.0,
    pocket_mode: str = "centroid",
) -> Tuple[List, Tuple[float, float, float]]:
    """
    Returns:
      pocket_atoms, center_xyz (center is ligand centroid, even for atoms-mode)
    """
    center = ligand_centroid(ligand_atoms)

    mode = (pocket_mode or "centroid").strip().lower()
    if mode == "centroid":
        pocket_atoms = select_receptor_atoms_sphere(receptor_atoms, center, pocket_radius)
    elif mode == "atoms":
        pocket_atoms = select_receptor_atoms_union_spheres(receptor_atoms, ligand_atoms, pocket_radius)
    else:
        raise ValueError(f"Unknown pocket_mode={pocket_mode!r}. Use centroid|atoms")

    return pocket_atoms, center
