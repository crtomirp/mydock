"""docking.io.read_sdf

Read ligand structures from SDF using RDKit.

We also keep the RDKit Mol object for angle-based H-bond scoring in ChemPLP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None


@dataclass(frozen=True)
class LigandAtom:
    idx: int
    element: str
    x: float
    y: float
    z: float
    charge: float


@dataclass(frozen=True)
class LigandMol:
    name: str
    atoms: List[LigandAtom]
    rdkit_mol: object | None = None  # RDKit Mol (kept for angle-based HBonds)


def read_sdf_first_mol(
    sdf_path: str,
    add_hs: bool = True,
    compute_gasteiger: bool = True,
) -> LigandMol:
    if Chem is None:
        raise RuntimeError("RDKit is required for read_sdf_first_mol but is not installed.")

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=not add_hs)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"No valid molecules found in SDF: {sdf_path}")

    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
        AllChem.UFFOptimizeMolecule(mol)

    if compute_gasteiger:
        AllChem.ComputeGasteigerCharges(mol)

    conf = mol.GetConformer()
    atoms: List[LigandAtom] = []

    for a in mol.GetAtoms():
        idx = a.GetIdx()
        pos = conf.GetAtomPosition(idx)
        q = 0.0
        if compute_gasteiger and a.HasProp("_GasteigerCharge"):
            try:
                q = float(a.GetProp("_GasteigerCharge"))
            except Exception:
                q = 0.0
        atoms.append(
            LigandAtom(
                idx=idx,
                element=a.GetSymbol(),
                x=float(pos.x),
                y=float(pos.y),
                z=float(pos.z),
                charge=q,
            )
        )

    name = mol.GetProp("_Name") if mol.HasProp("_Name") else "ligand"
    return LigandMol(name=name, atoms=atoms, rdkit_mol=mol)
