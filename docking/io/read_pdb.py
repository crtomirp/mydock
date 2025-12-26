"""docking.io.read_pdb

Minimal PDB parser for receptor atoms (ATOM/HETATM).

Returns a list of AtomRecord objects containing:
- coordinates
- residue info
- element (inferred if missing)
- heuristic partial charge for protein ATOM records
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from docking.chemistry.protein_charges import guess_protein_atom_charge


@dataclass(frozen=True)
class AtomRecord:
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    element: str
    is_hetatm: bool
    charge: float


def _infer_element(atom_name: str, element_field: str) -> str:
    el = (element_field or "").strip()
    if el:
        return el.capitalize()
    a = atom_name.strip()
    if not a:
        return "X"
    if a[0].isdigit() and len(a) > 1:
        a = a[1:]
    return a[0].upper()


def read_pdb_atoms(
    pdb_path: str,
    keep_hetatm: bool = True,
    drop_waters: bool = True,
    allowed_altloc: Tuple[str, ...] = ("", "A"),
) -> List[AtomRecord]:
    """Read atoms from a PDB file."""
    atoms: List[AtomRecord] = []

    # --------
    # Pass 1: find termini (first/last residue per chain), using ATOM records only
    # --------
    residues_by_chain: Dict[str, List[int]] = {}

    with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM  "):
                continue

            resname = line[17:20].strip()
            if drop_waters and resname in {"HOH", "WAT", "H2O"}:
                continue

            chain = line[21:22].strip() or " "
            resseq = int(line[22:26].strip() or "0")
            residues_by_chain.setdefault(chain, []).append(resseq)

    termini_by_chain: Dict[str, Tuple[int, int]] = {}
    for ch, lst in residues_by_chain.items():
        if lst:
            termini_by_chain[ch] = (min(lst), max(lst))

    # --------
    # Pass 2: parse atoms + assign heuristic charges
    # --------
    with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue

            is_het = line.startswith("HETATM")
            if is_het and not keep_hetatm:
                continue

            atom_name = line[12:16]
            altloc = line[16:17].strip()
            if altloc not in allowed_altloc:
                continue

            resname = line[17:20].strip()
            if drop_waters and resname in {"HOH", "WAT", "H2O"}:
                continue

            serial = int(line[6:11].strip() or "0")
            chain = line[21:22].strip() or " "
            resseq = int(line[22:26].strip() or "0")
            icode = line[26:27].strip()

            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())

            element_field = line[76:78] if len(line) >= 78 else ""
            element = _infer_element(atom_name, element_field)

            # Only assign protein charges for ATOM records (not HETATM ligands/cofactors)
            q = 0.0
            if line.startswith("ATOM  "):
                is_n_term = False
                is_c_term = False
                if chain in termini_by_chain:
                    first_res, last_res = termini_by_chain[chain]
                    is_n_term = (resseq == first_res)
                    is_c_term = (resseq == last_res)

                q = guess_protein_atom_charge(
                    resname=resname,
                    atom_name=atom_name.strip(),
                    element=element,
                    is_n_terminus=is_n_term,
                    is_c_terminus=is_c_term,
                )

            atoms.append(
                AtomRecord(
                    serial=serial,
                    name=atom_name.strip(),
                    resname=resname,
                    chain=chain,
                    resseq=resseq,
                    icode=icode,
                    x=x,
                    y=y,
                    z=z,
                    element=element,
                    is_hetatm=is_het,
                    charge=q,
                )
            )

    return atoms
