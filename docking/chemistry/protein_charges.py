"""
docking.chemistry.protein_charges

Very simple residue-aware partial charge assignment for protein atoms.

Goal:
- Provide non-zero receptor charges for MVP electrostatics (Coulomb term)
- Keep it deterministic, dependency-free, and easy to replace later with
  proper force-field charges (AMBER/CHARMM/OpenMM).

Important:
- These are NOT force-field accurate atomic partial charges.
- They are heuristic charges suitable for early-stage scoring development.
"""

from __future__ import annotations


BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def _normalize_element(element: str) -> str:
    e = (element or "").strip()
    if not e:
        return "X"
    if len(e) == 1:
        return e.upper()
    return e[0].upper() + e[1:].lower()


def _is_backbone(atom_name: str) -> bool:
    return atom_name.strip().upper() in BACKBONE_ATOMS


def _backbone_baseline(atom_name: str, element: str) -> float:
    """
    Small backbone baseline charges to introduce polarity.
    Keep small so sidechain net charges dominate where appropriate.
    """
    an = atom_name.strip().upper()
    el = _normalize_element(element)

    if an == "O" and el == "O":
        return -0.20
    if an == "N" and el == "N":
        return +0.10
    return 0.0


def _sidechain_charge(resname: str, atom_name: str, element: str) -> float:
    """
    Heuristic distribution of net charges across key atoms.

    ASP/GLU: negative charge on OD*/OE* oxygens.
    LYS:     positive charge on NZ.
    ARG:     positive charge on NE/NH1/NH2.
    HIS:     mild positive on ND1/NE2.
    """
    r = (resname or "").strip().upper()
    an = (atom_name or "").strip().upper()
    el = _normalize_element(element)

    if r == "ASP":
        if an in {"OD1", "OD2"} or (el == "O" and an.startswith("OD")):
            return -0.50
        if an == "CG" and el == "C":
            return +0.10
        return 0.0

    if r == "GLU":
        if an in {"OE1", "OE2"} or (el == "O" and an.startswith("OE")):
            return -0.50
        if an == "CD" and el == "C":
            return +0.10
        return 0.0

    if r == "LYS":
        if an == "NZ" and el == "N":
            return +0.70
        if an == "CE" and el == "C":
            return -0.10
        return 0.0

    if r == "ARG":
        if an in {"NE", "NH1", "NH2"} and el == "N":
            return +0.35
        if an == "CZ" and el == "C":
            return -0.05
        return 0.0

    if r == "HIS":
        if an in {"ND1", "NE2"} and el == "N":
            return +0.25
        return 0.0

    return 0.0


def guess_protein_atom_charge(
    resname: str,
    atom_name: str,
    element: str,
    is_n_terminus: bool = False,
    is_c_terminus: bool = False,
) -> float:
    """
    Assign a heuristic partial charge to a protein atom.

    Termini:
    - N-terminus: add +0.60 on backbone N
    - C-terminus: add -0.60 on OXT / terminal O

    Returns:
        float charge (unitless; used in Coulomb term).
    """
    an = (atom_name or "").strip().upper()

    q = 0.0

    if _is_backbone(an):
        q += _backbone_baseline(an, element)

    q += _sidechain_charge(resname, atom_name, element)

    if is_n_terminus and an == "N":
        q += +0.60

    if is_c_terminus and an in {"OXT", "O"}:
        q += -0.60

    return q
