from __future__ import annotations
import math
import copy
import numpy as np


class Pose:
    """
    Rigid ligand pose:
    - translation (x,y,z)  [Å] : global position of ligand reference frame origin
    - rotation (Euler angles, radians) about the reference frame origin

    IMPORTANT (for optimizers):
    - We store reference coordinates CENTERED at the ligand centroid.
      This makes centroid constraints stable and makes it natural to initialize
      tx,ty,tz as a point inside the pocket sphere.
    """

    def __init__(self, atoms):
        self.ref_atoms = list(atoms)

        # Cache reference coordinates and center them at centroid
        xyz = np.array([[a.x, a.y, a.z] for a in self.ref_atoms], dtype=np.float32)
        centroid = xyz.mean(axis=0).astype(np.float32)

        # Reference coordinates centered at origin
        self.ref_xyz = (xyz - centroid).astype(np.float32)

        # The centroid of centered coordinates is ~0; keep explicitly for speed
        self.ref_centroid = np.zeros(3, dtype=np.float32)

        # Translation initialized to the ORIGINAL centroid, so the pose reproduces
        # the input ligand geometry when rotations are 0.
        self.tx = float(centroid[0])
        self.ty = float(centroid[1])
        self.tz = float(centroid[2])

        # Rotation (Euler) and score
        self.rx = 0.0
        self.ry = 0.0
        self.rz = 0.0
        self.score = None

    def copy(self):
        return copy.deepcopy(self)

    def rotation_matrix(self):
        cx = math.cos(self.rx)
        sx = math.sin(self.rx)
        cy = math.cos(self.ry)
        sy = math.sin(self.ry)
        cz = math.cos(self.rz)
        sz = math.sin(self.rz)

        # Rz * Ry * Rx
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
        return Rz @ Ry @ Rx

    def transformed_centroid(self):
        # Since ref coordinates are centered, centroid is translation
        return (float(self.tx), float(self.ty), float(self.tz))

    def transformed_xyz(self) -> np.ndarray:
        R = self.rotation_matrix()
        L = (R @ self.ref_xyz.T).T
        L[:, 0] += self.tx
        L[:, 1] += self.ty
        L[:, 2] += self.tz
        return L.astype(np.float32)

    def transformed_atoms(self):
        """
        Slow/debug helper: materialize Atom-like objects.
        """
        L_xyz = self.transformed_xyz()
        atoms = []
        for a, v2 in zip(self.ref_atoms, L_xyz):
            atoms.append(
                a.__class__(
                    idx=a.idx,
                    element=a.element,
                    x=float(v2[0]),
                    y=float(v2[1]),
                    z=float(v2[2]),
                    charge=a.charge,
                )
            )
        return atoms
