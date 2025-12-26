from __future__ import annotations
import math
import copy
import numpy as np


class Pose:
    """
    Rigid ligand pose:
    - translation (x,y,z)
    - rotation (Euler angles, radians)

    Performance notes:
    - ref_xyz is cached numpy array (N,3), so we can produce transformed_xyz() cheaply
    - transformed_atoms() is kept for debug/output compatibility (creates Atom objects)
    """

    def __init__(self, atoms):
        self.ref_atoms = list(atoms)
        self.tx = 0.0
        self.ty = 0.0
        self.tz = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.rz = 0.0
        self.score = None

        # cache reference coordinates
        self.ref_xyz = np.array([[a.x, a.y, a.z] for a in self.ref_atoms], dtype=np.float32)
        self.ref_centroid = self.ref_xyz.mean(axis=0).astype(np.float32)

    def copy(self):
        return copy.deepcopy(self)

    def rotation_matrix(self):
        cx = math.cos(self.rx)
        sx = math.sin(self.rx)
        cy = math.cos(self.ry)
        sy = math.sin(self.ry)
        cz = math.cos(self.rz)
        sz = math.sin(self.rz)

        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx, cx]], dtype=float)
        Ry = np.array([[cy, 0, sy],
                       [0, 1, 0],
                       [-sy, 0, cy]], dtype=float)
        Rz = np.array([[cz, -sz, 0],
                       [sz, cz, 0],
                       [0, 0, 1]], dtype=float)

        return Rz @ Ry @ Rx

    def transformed_centroid(self):
        R = self.rotation_matrix()
        c = R @ self.ref_centroid.astype(float) + np.array([self.tx, self.ty, self.tz], dtype=float)
        return (float(c[0]), float(c[1]), float(c[2]))

    def transformed_xyz(self) -> np.ndarray:
        """
        Return transformed ligand coordinates as float32 array (N,3).
        This is the preferred path for Numba/NumPy scoring.
        """
        R = self.rotation_matrix().astype(np.float32)
        t = np.array([self.tx, self.ty, self.tz], dtype=np.float32)
        return (self.ref_xyz @ R.T) + t

    def transformed_atoms(self):
        """
        Compatibility method: create new Atom objects at transformed positions.
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
