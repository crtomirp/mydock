"""docking.scoring.electrostatics

Simple Coulomb electrostatics term for MVP.

- Ligand charges: RDKit Gasteiger (from read_sdf.py)
- Receptor charges: heuristic (from io/read_pdb.py via protein_charges.py)

Includes debug utilities to list strongest pair interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..geometry.distances import distance, within_cutoff


def coulomb_energy(r: float, q1: float, q2: float, k: float = 332.0636, eps: float = 10.0) -> float:
    """Coulomb energy in kcal/mol if k=332.0636, r in Å, q in e."""
    if r <= 1e-12:
        return 0.0
    return k * (q1 * q2) / (eps * r)


def electrostatics_score_atoms(
    receptor_atoms,
    ligand_atoms,
    cutoff: float = 12.0,
    k: float = 332.0636,
    eps: float = 10.0,
) -> float:
    e = 0.0
    for ra in receptor_atoms:
        q1 = float(getattr(ra, "charge", 0.0))
        if abs(q1) < 1e-12:
            continue
        for la in ligand_atoms:
            q2 = float(getattr(la, "charge", 0.0))
            if abs(q2) < 1e-12:
                continue
            r = distance(ra.x, ra.y, ra.z, la.x, la.y, la.z)
            if not within_cutoff(r, cutoff):
                continue
            e += coulomb_energy(r, q1, q2, k=k, eps=eps)
    return e


@dataclass(frozen=True)
class ElePair:
    ra_serial: int
    ra_name: str
    ra_resname: str
    ra_chain: str
    ra_resseq: int
    ra_element: str
    ra_charge: float

    la_idx: int
    la_element: str
    la_charge: float

    r: float
    e: float


def top_electrostatic_pairs(
    receptor_atoms,
    ligand_atoms,
    cutoff: float = 12.0,
    k: float = 332.0636,
    eps: float = 10.0,
    top_n: int = 20,
) -> Tuple[List[ElePair], List[ElePair]]:
    """
    Return (most_attractive, most_repulsive) lists of ElePair.

    - most_attractive: lowest (most negative) energies
    - most_repulsive:  highest (most positive) energies
    """
    pairs: List[ElePair] = []

    for ra in receptor_atoms:
        q1 = float(getattr(ra, "charge", 0.0))
        if abs(q1) < 1e-12:
            continue
        for la in ligand_atoms:
            q2 = float(getattr(la, "charge", 0.0))
            if abs(q2) < 1e-12:
                continue
            r = distance(ra.x, ra.y, ra.z, la.x, la.y, la.z)
            if not within_cutoff(r, cutoff):
                continue
            e = coulomb_energy(r, q1, q2, k=k, eps=eps)
            pairs.append(
                ElePair(
                    ra_serial=int(getattr(ra, "serial", -1)),
                    ra_name=str(getattr(ra, "name", "?")),
                    ra_resname=str(getattr(ra, "resname", "?")),
                    ra_chain=str(getattr(ra, "chain", "?")),
                    ra_resseq=int(getattr(ra, "resseq", -1)),
                    ra_element=str(getattr(ra, "element", "?")),
                    ra_charge=q1,
                    la_idx=int(getattr(la, "idx", -1)),
                    la_element=str(getattr(la, "element", "?")),
                    la_charge=q2,
                    r=float(r),
                    e=float(e),
                )
            )

    most_attractive = sorted(pairs, key=lambda p: p.e)[: max(0, top_n)]
    most_repulsive = sorted(pairs, key=lambda p: p.e, reverse=True)[: max(0, top_n)]
    return most_attractive, most_repulsive
