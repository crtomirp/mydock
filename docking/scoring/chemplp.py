"""docking.scoring.chemplp

ChemPLP-like scoring with:
- Steric: piecewise linear + clash hard-cap explosion
- HBond: distance term + optional angle term (donor–H–acceptor), when H coordinates available
- Lipophilic: contact-count with saturation

Also includes debug helpers:
- top_hbond_contacts(...)
- top_clash_pairs(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..geometry.distances import distance, within_cutoff
from ..chemistry.vdw_radii import vdw_radius

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


HB_ELEMENTS = {"N", "O", "S"}
LIPO_ELEMENTS = {"C", "S"}


def _norm_el(e: str) -> str:
    e = (e or "").strip()
    if not e:
        return "X"
    if len(e) == 1:
        return e.upper()
    return e[0].upper() + e[1:].lower()


def _dot(ax, ay, az, bx, by, bz) -> float:
    return ax * bx + ay * by + az * bz


def _norm(ax, ay, az) -> float:
    return (ax * ax + ay * ay + az * az) ** 0.5


def angle_deg(p1, p2, p3) -> float:
    """Angle at p2: p1 - p2 - p3 in degrees."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    v2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])
    n1 = _norm(*v1)
    n2 = _norm(*v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    c = _dot(*v1, *v2) / (n1 * n2)
    c = max(-1.0, min(1.0, c))
    import math
    return float(math.degrees(math.acos(c)))


def piecewise_linear_contact(
    r: float,
    r0: float,
    slope_in: float,
    slope_out: float,
    width_in: float,
    width_out: float,
) -> float:
    """
    Generic piecewise linear around r0:
    - r < r0: overlap penalty (positive)
    - r0 <= r <= r0+width_out: attraction (negative) fading to 0
    - else: 0
    """
    if r <= 1e-12:
        return 0.0

    if r < r0:
        dr = r0 - r
        if width_in > 0.0:
            dr = min(dr, width_in)
        return +slope_in * dr

    dr = r - r0
    if dr <= width_out:
        return slope_out * (1.0 - dr / width_out)
    return 0.0


def is_hbond_capable(element: str) -> bool:
    return _norm_el(element) in HB_ELEMENTS


def is_lipophilic(element: str) -> bool:
    return _norm_el(element) in LIPO_ELEMENTS


@dataclass(frozen=True)
class ChemPLPBreakdown:
    steric: float
    # Optional terms (enabled via params/precision). Defaults keep backward compatibility
    # with earlier versions that only had steric/hbond/lipo/clashes.
    ele: float = 0.0
    hbond: float = 0.0
    lipo: float = 0.0
    pi: float = 0.0
    clashes: float = 0.0

    @property
    def total(self) -> float:
        return self.steric + self.ele + self.hbond + self.lipo + self.pi + self.clashes


@dataclass(frozen=True)
class HBondContact:
    # receptor atom
    ra_serial: int
    ra_resname: str
    ra_chain: str
    ra_resseq: int
    ra_name: str
    ra_element: str
    # ligand atom
    la_idx: int
    la_element: str
    # geometry
    r: float
    angle: Optional[float]  # D-H-A if available (ligand donor)
    # score parts
    e_dist: float
    e_total: float


@dataclass(frozen=True)
class ClashPair:
    ra_serial: int
    ra_resname: str
    ra_chain: str
    ra_resseq: int
    ra_name: str
    ra_element: str

    la_idx: int
    la_element: str

    r: float
    r0: float
    overlap: float
    penalty: float


def _ligand_hbond_donors_with_h_coords(ligand_rdkit_mol) -> Dict[int, List[Tuple[float, float, float]]]:
    """
    donor_atom_idx -> list of H coords bonded to donor
    Works if ligand has explicit H atoms + conformer.
    """
    donors: Dict[int, List[Tuple[float, float, float]]] = {}
    if Chem is None or ligand_rdkit_mol is None:
        return donors
    if ligand_rdkit_mol.GetNumConformers() == 0:
        return donors

    conf = ligand_rdkit_mol.GetConformer()
    for a in ligand_rdkit_mol.GetAtoms():
        el = a.GetSymbol()
        if el not in ("N", "O", "S"):
            continue
        h_coords = []
        for nb in a.GetNeighbors():
            if nb.GetSymbol() == "H":
                p = conf.GetAtomPosition(nb.GetIdx())
                h_coords.append((float(p.x), float(p.y), float(p.z)))
        if h_coords:
            donors[a.GetIdx()] = h_coords
    return donors


def _default_params(params: Optional[Dict]) -> Dict:
    p = dict(params or {})
    # defaults
    p.setdefault("steric_cutoff", 10.0)
    p.setdefault("w_steric", 1.0)
    p.setdefault("w_hbond", 1.0)
    p.setdefault("w_lipo", 1.0)
    p.setdefault("w_clash", 1.0)
    p.setdefault("w_ele", 1.0)
    p.setdefault("ele_eps", 10.0)
    p.setdefault("w_pi", 0.0)

    p.setdefault("steric_slope_in", 5.0)
    p.setdefault("steric_slope_out", -1.0)
    p.setdefault("steric_width_in", 0.6)
    p.setdefault("steric_width_out", 1.5)

    p.setdefault("clash_overlap_hard", 0.8)
    p.setdefault("clash_k", 2500.0)

    p.setdefault("hbond_r0", 2.7)
    p.setdefault("hbond_width_out", 0.8)
    p.setdefault("hbond_strength", -2.0)
    p.setdefault("hbond_angle_min", 120.0)
    p.setdefault("hbond_angle_strength", 1.0)
    p.setdefault("hbond_require_angle", False)

    p.setdefault("lipo_r_min", 3.4)
    p.setdefault("lipo_r_max", 5.0)
    p.setdefault("lipo_contact_weight", 1.0)
    p.setdefault("lipo_sat_max", 10.0)
    p.setdefault("lipo_sat_k", 6.0)
    return p


def chemplp_score_atoms(
    receptor_atoms,
    ligand_atoms,
    ligand_rdkit_mol=None,
    params: Optional[Dict] = None,
    *,
    engine: str = "auto",
    prepared: Optional[Dict] = None,
) -> ChemPLPBreakdown:
    p = _default_params(params)

    # --- engine dispatch
    eng = (engine or "auto").lower()
    if eng == "auto":
        # prefer numba if available
        try:
            import numba  # noqa: F401
            eng = "numba"
        except Exception:
            eng = "python"

    if eng == "numba":
        try:
            from .numba import prepare_receptor_arrays, prepare_ligand_static, prepare_pi_features
            from .numba import chemplp_numba_score
        except Exception:
            eng = "python"

    if eng == "numba":
        # prepared may include cached receptor and ligand static arrays
        prep = prepared or {}
        if "R_arrays" in prep:
            R_xyz, R_q, R_vdw, R_hba, R_hbd, R_lipo = prep["R_arrays"]
        else:
            R_xyz, R_q, R_vdw, R_hba, R_hbd, R_lipo = prepare_receptor_arrays(receptor_atoms)

        if "L_static" in prep:
            L_q, L_vdw, L_hba, L_hbd, L_lipo = prep["L_static"]
        else:
            L_q, L_vdw, L_hba, L_hbd, L_lipo = prepare_ligand_static(ligand_atoms, ligand_rdkit_mol)

        # ligand xyz must be provided by caller in prepared for speed, else build from atom objects
        if "L_xyz" in prep:
            L_xyz = prep["L_xyz"]
        else:
            import numpy as np
            L_xyz = np.array([[a.x, a.y, a.z] for a in ligand_atoms], dtype=np.float32)

        # pi features (optional)
        R_pi_c = R_pi_n = L_pi_c = L_pi_n = None
        if float(p.get("w_pi", 0.0)) != 0.0 and ligand_rdkit_mol is not None:
            if "PI" in prep:
                R_pi_c, R_pi_n, L_pi_c, L_pi_n = prep["PI"]
            else:
                R_pi_c, R_pi_n, L_pi_c, L_pi_n = prepare_pi_features(receptor_atoms, ligand_rdkit_mol)

        bd = chemplp_numba_score(
            R_xyz, R_q, R_vdw, R_hba, R_hbd, R_lipo,
            L_xyz, L_q, L_vdw, L_hba, L_hbd, L_lipo,
            params=p,
            R_pi_c=R_pi_c, R_pi_n=R_pi_n, L_pi_c=L_pi_c, L_pi_n=L_pi_n
        )

        return ChemPLPBreakdown(
            steric=bd["steric"],
            ele=bd.get("ele", 0.0),
            hbond=bd["hbond"],
            lipo=bd["lipo"],
            pi=bd.get("pi", 0.0),
            clashes=bd["clashes"],
        )

    steric_cutoff = float(p["steric_cutoff"])
    w_steric = float(p["w_steric"])
    w_hbond = float(p["w_hbond"])
    w_lipo = float(p["w_lipo"])
    w_clash = float(p["w_clash"])
    w_ele = float(p.get("w_ele", 1.0))
    ele_eps = float(p.get("ele_eps", 10.0))
    w_pi = float(p.get("w_pi", 0.0))

    steric_slope_in = float(p["steric_slope_in"])
    steric_slope_out = float(p["steric_slope_out"])
    steric_width_in = float(p["steric_width_in"])
    steric_width_out = float(p["steric_width_out"])

    clash_overlap_hard = float(p["clash_overlap_hard"])
    clash_k = float(p["clash_k"])

    hbond_r0 = float(p["hbond_r0"])
    hbond_width_out = float(p["hbond_width_out"])
    hbond_strength = float(p["hbond_strength"])
    hbond_angle_min = float(p["hbond_angle_min"])
    hbond_angle_strength = float(p["hbond_angle_strength"])
    hbond_require_angle = bool(p["hbond_require_angle"])

    lipo_r_min = float(p["lipo_r_min"])
    lipo_r_max = float(p["lipo_r_max"])
    lipo_contact_weight = float(p["lipo_contact_weight"])
    lipo_sat_max = float(p["lipo_sat_max"])
    lipo_sat_k = float(p["lipo_sat_k"])

    steric = 0.0
    hbond = 0.0
    clashes = 0.0
    lipo_contacts = 0.0

    ligand_donor_H = _ligand_hbond_donors_with_h_coords(ligand_rdkit_mol)
    lig_xyz_by_idx = {int(a.idx): (float(a.x), float(a.y), float(a.z)) for a in ligand_atoms}

    for ra in receptor_atoms:
        el_r = _norm_el(getattr(ra, "element", "X"))
        rr = vdw_radius(el_r)
        ra_xyz = (float(ra.x), float(ra.y), float(ra.z))

        for la in ligand_atoms:
            el_l = _norm_el(getattr(la, "element", "X"))
            rl = vdw_radius(el_l)

            r = distance(ra.x, ra.y, ra.z, la.x, la.y, la.z)
            if not within_cutoff(r, steric_cutoff):
                continue

            r0 = rr + rl

            # Steric (piecewise)
            steric += piecewise_linear_contact(
                r=r,
                r0=r0,
                slope_in=steric_slope_in,
                slope_out=steric_slope_out,
                width_in=steric_width_in,
                width_out=steric_width_out,
            )

            # Clash hard-cap explosion
            overlap = r0 - r
            if overlap > clash_overlap_hard:
                d = overlap - clash_overlap_hard
                clashes += clash_k * (d * d)

            # HBond (distance + optional angle)
            if is_hbond_capable(el_r) and is_hbond_capable(el_l):
                e_dist = piecewise_linear_contact(
                    r=r,
                    r0=hbond_r0,
                    slope_in=0.0,
                    slope_out=hbond_strength,
                    width_in=0.0,
                    width_out=hbond_width_out,
                )

                angle_ok = False
                la_idx = int(getattr(la, "idx", -1))
                if la_idx in ligand_donor_H:
                    D = lig_xyz_by_idx.get(la_idx)
                    if D is not None:
                        for H in ligand_donor_H[la_idx]:
                            A = ra_xyz
                            ang = angle_deg(D, H, A)  # D-H-A
                            if ang >= hbond_angle_min:
                                angle_ok = True
                                break

                if hbond_require_angle:
                    if angle_ok:
                        hbond += e_dist * hbond_angle_strength
                else:
                    hbond += (e_dist * hbond_angle_strength) if angle_ok else e_dist

            # Lipophilic contact count (then saturate)
            if is_lipophilic(el_r) and is_lipophilic(el_l):
                if (r >= lipo_r_min) and (r <= lipo_r_max):
                    lipo_contacts += 1.0

    # Saturating lipophilic term (negative is favorable)
    raw = lipo_contact_weight * lipo_contacts
    if raw <= 1e-12:
        lipo = 0.0
    else:
        lipo = -lipo_sat_max * (raw / (raw + lipo_sat_k))


    # Pi stacking (python engine): use centroid/normal features if available
    if w_pi != 0.0 and ligand_rdkit_mol is not None:
        try:
            import numpy as np
            from .numba.adapters import prepare_pi_features
            R_pi_c, R_pi_n, L_pi_c, L_pi_n = prepare_pi_features(receptor_atoms, ligand_rdkit_mol)
            if R_pi_c.shape[0] and L_pi_c.shape[0]:
                # vectorized-ish small loops
                for i in range(R_pi_c.shape[0]):
                    for j in range(L_pi_c.shape[0]):
                        d = float(np.linalg.norm(R_pi_c[i] - L_pi_c[j]))
                        if d < 3.5 or d > 6.0:
                            continue
                        cosang = abs(float(np.dot(R_pi_n[i], L_pi_n[j])))
                        if cosang >= 0.7 or cosang <= 0.3:
                            mid = 4.75
                            span = 1.25
                            t = (d - mid) / span
                            w = 1.0 - t*t
                            if w > 0.0:
                                pi += (-w_pi * w)
        except Exception:
            pass

    return ChemPLPBreakdown(
        steric=w_steric * steric,
        ele=w_ele * ele,
        hbond=w_hbond * hbond,
        lipo=w_lipo * lipo,
        pi=pi,
        clashes=w_clash * clashes,
    )


# -------------------------
# DEBUG HELPERS
# -------------------------

def top_hbond_contacts(
    receptor_atoms,
    ligand_atoms,
    ligand_rdkit_mol=None,
    params: Optional[Dict] = None,
    top_n: int = 20,
) -> List[HBondContact]:
    """
    Return strongest (most negative) Hbond contacts based on the ChemPLP Hbond term.
    Angle is reported when donor is on ligand and H coordinates are available.
    """
    p = _default_params(params)
    steric_cutoff = float(p["steric_cutoff"])

    hbond_r0 = float(p["hbond_r0"])
    hbond_width_out = float(p["hbond_width_out"])
    hbond_strength = float(p["hbond_strength"])
    hbond_angle_min = float(p["hbond_angle_min"])
    hbond_angle_strength = float(p["hbond_angle_strength"])
    hbond_require_angle = bool(p["hbond_require_angle"])

    ligand_donor_H = _ligand_hbond_donors_with_h_coords(ligand_rdkit_mol)
    lig_xyz_by_idx = {int(a.idx): (float(a.x), float(a.y), float(a.z)) for a in ligand_atoms}

    out: List[HBondContact] = []

    for ra in receptor_atoms:
        el_r = _norm_el(getattr(ra, "element", "X"))
        if el_r not in HB_ELEMENTS:
            continue
        ra_xyz = (float(ra.x), float(ra.y), float(ra.z))

        for la in ligand_atoms:
            el_l = _norm_el(getattr(la, "element", "X"))
            if el_l not in HB_ELEMENTS:
                continue

            r = distance(ra.x, ra.y, ra.z, la.x, la.y, la.z)
            if not within_cutoff(r, steric_cutoff):
                continue

            e_dist = piecewise_linear_contact(
                r=r,
                r0=hbond_r0,
                slope_in=0.0,
                slope_out=hbond_strength,
                width_in=0.0,
                width_out=hbond_width_out,
            )
            if abs(e_dist) < 1e-12:
                continue

            la_idx = int(getattr(la, "idx", -1))
            ang_val: Optional[float] = None
            angle_ok = False
            if la_idx in ligand_donor_H:
                D = lig_xyz_by_idx.get(la_idx)
                if D is not None:
                    # pick best angle among attached H's
                    best = -1.0
                    for H in ligand_donor_H[la_idx]:
                        ang = angle_deg(D, H, ra_xyz)
                        if ang > best:
                            best = ang
                    if best >= 0:
                        ang_val = best
                        angle_ok = (best >= hbond_angle_min)

            if hbond_require_angle and not angle_ok:
                continue

            e_total = (e_dist * hbond_angle_strength) if angle_ok else e_dist

            out.append(
                HBondContact(
                    ra_serial=int(getattr(ra, "serial", -1)),
                    ra_resname=str(getattr(ra, "resname", "?")),
                    ra_chain=str(getattr(ra, "chain", "?")),
                    ra_resseq=int(getattr(ra, "resseq", -1)),
                    ra_name=str(getattr(ra, "name", "?")),
                    ra_element=str(getattr(ra, "element", "?")),
                    la_idx=la_idx,
                    la_element=str(getattr(la, "element", "?")),
                    r=float(r),
                    angle=ang_val,
                    e_dist=float(e_dist),
                    e_total=float(e_total),
                )
            )

    out.sort(key=lambda x: x.e_total)  # most negative first
    return out[: max(0, top_n)]


def top_clash_pairs(
    receptor_atoms,
    ligand_atoms,
    params: Optional[Dict] = None,
    top_n: int = 20,
) -> List[ClashPair]:
    """
    Return strongest clashes (largest penalty), using overlap hard-cap explosion.
    """
    p = _default_params(params)
    steric_cutoff = float(p["steric_cutoff"])
    clash_overlap_hard = float(p["clash_overlap_hard"])
    clash_k = float(p["clash_k"])

    out: List[ClashPair] = []

    for ra in receptor_atoms:
        el_r = _norm_el(getattr(ra, "element", "X"))
        rr = vdw_radius(el_r)

        for la in ligand_atoms:
            el_l = _norm_el(getattr(la, "element", "X"))
            rl = vdw_radius(el_l)

            r = distance(ra.x, ra.y, ra.z, la.x, la.y, la.z)
            if not within_cutoff(r, steric_cutoff):
                continue

            r0 = rr + rl
            overlap = r0 - r
            if overlap > clash_overlap_hard:
                d = overlap - clash_overlap_hard
                pen = clash_k * (d * d)
                out.append(
                    ClashPair(
                        ra_serial=int(getattr(ra, "serial", -1)),
                        ra_resname=str(getattr(ra, "resname", "?")),
                        ra_chain=str(getattr(ra, "chain", "?")),
                        ra_resseq=int(getattr(ra, "resseq", -1)),
                        ra_name=str(getattr(ra, "name", "?")),
                        ra_element=str(getattr(ra, "element", "?")),
                        la_idx=int(getattr(la, "idx", -1)),
                        la_element=str(getattr(la, "element", "?")),
                        r=float(r),
                        r0=float(r0),
                        overlap=float(overlap),
                        penalty=float(pen),
                    )
                )

    out.sort(key=lambda x: x.penalty, reverse=True)
    return out[: max(0, top_n)]
