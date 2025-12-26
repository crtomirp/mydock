"""docking.scoring.total_score

Combine scoring terms.
Includes:
- simple score (vdw + ele + hbond)
- chemplp score (ChemPLP-like with advanced options)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .precision import merge_precision

from .vdw import vdw_score_atoms
from .electrostatics import electrostatics_score_atoms
from .hydrogen_bonds import hydrogen_bond_score_atoms

from .chemplp import chemplp_score_atoms, ChemPLPBreakdown


@dataclass(frozen=True)
class ScoreBreakdown:
    vdw: float
    ele: float
    hbond: float

    @property
    def total(self) -> float:
        return self.vdw + self.ele + self.hbond


def score_complex(receptor_atoms, ligand_atoms) -> ScoreBreakdown:
    vdw = vdw_score_atoms(receptor_atoms, ligand_atoms)
    ele = electrostatics_score_atoms(receptor_atoms, ligand_atoms)
    hbond = hydrogen_bond_score_atoms(receptor_atoms, ligand_atoms)
    return ScoreBreakdown(vdw=vdw, ele=ele, hbond=hbond)


def score_complex_chemplp(
    receptor_atoms,
    ligand_atoms,
    ligand_rdkit_mol=None,
    params: Optional[Dict] = None,
    *,
    precision: str = "SP",
    engine: str = "auto",
    prepared: Optional[Dict] = None,
) -> ChemPLPBreakdown:
    p = merge_precision(params, precision=precision)
    return chemplp_score_atoms(
        receptor_atoms=receptor_atoms,
        ligand_atoms=ligand_atoms,
        ligand_rdkit_mol=ligand_rdkit_mol,
        params=p,
        engine=engine,
        prepared=prepared,
    )
