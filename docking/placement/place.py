from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

from docking.optimize.pose import Pose
from docking.placement.hotspots import detect_receptor_hotspots, Hotspot
from docking.placement.ligand_anchors import detect_ligand_anchors, Anchor


def generate_initial_poses(
    receptor_atoms,
    ligand,
    pocket_center: Tuple[float, float, float],
    pocket_radius: float,
    max_poses: int = 300,
    rotations_per_match: int = 24,
    include_lipo: bool = False,
):
    """
    Glide-like initial placement (lightweight):
      - detect receptor hotspots (HBA/HBD/PI, optionally LIPO)
      - detect ligand anchors (HBD/HBA/PI)
      - overlay anchor centroid on hotspot
      - generate several random rotations
      - enforce centroid-in-pocket constraint
      - return a list of Pose objects (Pose.score left as None; caller can fast-score)

    The goal is to seed GA/MC with 'reasonable' starting poses inside the pocket.
    """

    hs = detect_receptor_hotspots(receptor_atoms)
    if not include_lipo:
        hs = [h for h in hs if h.kind in ("HBA", "HBD", "PI")]

    anchors = detect_ligand_anchors(ligand.rdkit_mol)

    poses: List[Pose] = []

    pcx, pcy, pcz = pocket_center
    pr2 = float(pocket_radius) * float(pocket_radius)

    def compatible(a: Anchor, h: Hotspot) -> bool:
        if a.kind == "HBD" and h.kind == "HBA":
            return True
        if a.kind == "HBA" and h.kind == "HBD":
            return True
        if a.kind == "PI" and h.kind == "PI":
            return True
        return False

    for a in anchors:
        for h in hs:
            if not compatible(a, h):
                continue

            base = Pose(ligand.atoms)
            # translate so anchor centroid overlays hotspot
            base.tx += (h.x - a.cx)
            base.ty += (h.y - a.cy)
            base.tz += (h.z - a.cz)

            for _ in range(int(rotations_per_match)):
                p = base.copy()
                p.rx = random.uniform(-math.pi, math.pi)
                p.ry = random.uniform(-math.pi, math.pi)
                p.rz = random.uniform(-math.pi, math.pi)

                cx, cy, cz = p.transformed_centroid()
                dx = cx - pcx
                dy = cy - pcy
                dz = cz - pcz
                if (dx*dx + dy*dy + dz*dz) > pr2:
                    continue

                poses.append(p)
                if len(poses) >= max_poses * 3:
                    break
            if len(poses) >= max_poses * 3:
                break
        if len(poses) >= max_poses * 3:
            break

    # If too few poses (no anchors), fall back to a few random-in-pocket
    if len(poses) < max(10, max_poses // 10):
        for _ in range(max(10, max_poses // 10)):
            p = Pose(ligand.atoms)
            # random translation near pocket center
            p.tx += random.uniform(-pocket_radius, pocket_radius)
            p.ty += random.uniform(-pocket_radius, pocket_radius)
            p.tz += random.uniform(-pocket_radius, pocket_radius)
            p.rx = random.uniform(-math.pi, math.pi)
            p.ry = random.uniform(-math.pi, math.pi)
            p.rz = random.uniform(-math.pi, math.pi)
            poses.append(p)

    # cap
    return poses[:max_poses]
