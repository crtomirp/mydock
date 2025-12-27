from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def _require_numba():
    if njit is None:
        raise ImportError("Numba is not available. Install with: pip install numba")


if njit is not None:
    @njit(fastmath=True)
    def _chemplp_pairwise(
        R_xyz, R_q, R_vdw, R_hba, R_hbd, R_lipo,
        L_xyz, L_q, L_vdw, L_hba, L_hbd, L_lipo,
        steric_cutoff2,
        w_steric, steric_width_in, steric_width_out, steric_slope_in, steric_slope_out,
        w_ele,
        w_hbond, hbond_r0, hbond_width_out, hbond_strength,
        w_lipo, lipo_cutoff2, lipo_sat,
        w_clash, clash_hard_overlap, clash_penalty
    ):
        nR = R_xyz.shape[0]
        nL = L_xyz.shape[0]

        steric = 0.0
        ele = 0.0
        hbond = 0.0
        lipo = 0.0
        clashes = 0.0

        for i in range(nR):
            Rx = R_xyz[i, 0]
            Ry = R_xyz[i, 1]
            Rz = R_xyz[i, 2]
            qi = R_q[i]
            rvdw = R_vdw[i]
            ri_hba = R_hba[i]
            ri_hbd = R_hbd[i]
            ri_lipo = R_lipo[i]

            for j in range(nL):
                dx = Rx - L_xyz[j, 0]
                dy = Ry - L_xyz[j, 1]
                dz = Rz - L_xyz[j, 2]
                r2 = dx*dx + dy*dy + dz*dz
                if r2 > steric_cutoff2:
                    continue

                r = math.sqrt(r2) + 1e-6
                lvdw = L_vdw[j]
                r0 = rvdw + lvdw
                overlap = r0 - r

                # steric piecewise (simple)
                if overlap > 0.0:
                    # inside
                    if overlap < steric_width_in:
                        steric += (steric_slope_in * overlap)
                    else:
                        steric += (steric_slope_in * steric_width_in)
                else:
                    # outside (attractive/relaxation)
                    out = -overlap
                    if out < steric_width_out:
                        steric += (steric_slope_out * out)

                # electrostatics (cheap Coulomb)
                ele += qi * L_q[j] / r

                # hbond (distance-only proxy)
                if w_hbond != 0.0:
                    lj_hba = L_hba[j]
                    lj_hbd = L_hbd[j]
                    # acceptor-donor complementarity proxy
                    if (ri_hba == 1 and lj_hbd == 1) or (ri_hbd == 1 and lj_hba == 1):
                        # piecewise: favorable when r <= r0+width_out
                        if r <= hbond_r0:
                            hbond += 0.0
                        else:
                            dr = r - hbond_r0
                            if dr < hbond_width_out:
                                hbond += (hbond_strength * (1.0 - dr / hbond_width_out))

                # lipophilic contact count with saturation
                if w_lipo != 0.0 and ri_lipo == 1 and L_lipo[j] == 1:
                    if r2 <= lipo_cutoff2:
                        lipo += 1.0

                # clash hard-cap explosion
                if overlap > clash_hard_overlap:
                    clashes += clash_penalty * (overlap - clash_hard_overlap) * (overlap - clash_hard_overlap)

        # saturation
        if lipo_sat > 1e-9:
            lipo = min(lipo / lipo_sat, 1.0) * lipo_sat

        return steric, ele, hbond, lipo, clashes


    @njit(fastmath=True)
    def _pi_stacking(R_pi_c, R_pi_n, L_pi_c, L_pi_n,
                     w_pi,
                     dmin, dmax,
                     cos_parallel_min, cos_t_min):
        # returns negative (favorable) contribution
        if w_pi == 0.0:
            return 0.0
        e = 0.0
        nR = R_pi_c.shape[0]
        nL = L_pi_c.shape[0]
        for i in range(nR):
            cx = R_pi_c[i, 0]
            cy = R_pi_c[i, 1]
            cz = R_pi_c[i, 2]
            nx = R_pi_n[i, 0]
            ny = R_pi_n[i, 1]
            nz = R_pi_n[i, 2]
            for j in range(nL):
                dx = cx - L_pi_c[j, 0]
                dy = cy - L_pi_c[j, 1]
                dz = cz - L_pi_c[j, 2]
                d = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9
                if d < dmin or d > dmax:
                    continue
                mx = L_pi_n[j, 0]
                my = L_pi_n[j, 1]
                mz = L_pi_n[j, 2]
                cosang = abs(nx*mx + ny*my + nz*mz)

                ok = 0
                # parallel stacking
                if cosang >= cos_parallel_min:
                    ok = 1
                # T-shaped
                if cosang <= cos_t_min:
                    ok = 1

                if ok == 1:
                    # smooth distance weight peaked near ~4.5 Å
                    # gaussian-like but cheap: parabola around mid
                    mid = 0.5*(dmin+dmax)
                    span = 0.5*(dmax-dmin)
                    t = (d - mid)/span
                    w = 1.0 - t*t
                    if w > 0.0:
                        e += (-w_pi * w)
        return e


def chemplp_numba_score(
    R_xyz, R_q, R_vdw, R_hba, R_hbd, R_lipo,
    L_xyz, L_q, L_vdw, L_hba, L_hbd, L_lipo,
    params: dict,
    R_pi_c=None, R_pi_n=None, L_pi_c=None, L_pi_n=None
):
    """
    Numba-accelerated ChemPLP-like scoring.
    Returns a dict breakdown: steric, ele, hbond, lipo, clashes, pi, total.

    Notes:
    - HBond term here is distance-only proxy (fast). Full directional HBond stays in python engine.
    - Pi stacking uses ring centroids/normals if provided.
    """
    _require_numba()

    steric_cutoff = float(params.get("steric_cutoff", params.get("steric_cutoff", 8.0)))
    steric_cutoff2 = steric_cutoff * steric_cutoff

    w_steric = float(params.get("w_steric", 1.0))
    steric_slope_in = float(params.get("steric_slope_in", 5.0))
    steric_slope_out = float(params.get("steric_slope_out", -1.0))
    steric_width_in = float(params.get("steric_width_in", 0.6))
    steric_width_out = float(params.get("steric_width_out", 1.5))

    w_ele = float(params.get("w_ele", 1.0))

    w_hbond = float(params.get("w_hbond", 1.0))
    hbond_r0 = float(params.get("hbond_r0", 2.9))
    hbond_width_out = float(params.get("hbond_width_out", 0.8))
    hbond_strength = float(params.get("hbond_strength", -1.0))

    w_lipo = float(params.get("w_lipo", 1.0))
    lipo_cutoff = float(params.get("lipo_cutoff", 4.5))
    lipo_cutoff2 = lipo_cutoff * lipo_cutoff
    lipo_sat = float(params.get("lipo_sat", 1.0))

    w_clash = float(params.get("w_clash", 1.0))
    clash_hard_overlap = float(params.get("clash_hard_overlap", 0.5))
    clash_penalty = float(params.get("clash_penalty", 60.0))

    steric, ele, hbond, lipo, clashes = _chemplp_pairwise(
        R_xyz, R_q, R_vdw, R_hba, R_hbd, R_lipo,
        L_xyz, L_q, L_vdw, L_hba, L_hbd, L_lipo,
        steric_cutoff2,
        w_steric, steric_width_in, steric_width_out, steric_slope_in, steric_slope_out,
        w_ele,
        w_hbond, hbond_r0, hbond_width_out, hbond_strength,
        w_lipo, lipo_cutoff2, lipo_sat,
        w_clash, clash_hard_overlap, clash_penalty
    )

    pi = 0.0
    w_pi = float(params.get("w_pi", 0.0))
    if w_pi != 0.0 and R_pi_c is not None and L_pi_c is not None and R_pi_c.shape[0] and L_pi_c.shape[0]:
        pi = _pi_stacking(
            R_pi_c.astype(np.float32), R_pi_n.astype(np.float32),
            L_pi_c.astype(np.float32), L_pi_n.astype(np.float32),
            w_pi,
            3.5, 6.0,
            0.7, 0.3
        )

    total = (
        w_steric * steric
        + w_ele * ele
        + w_hbond * hbond
        + w_lipo * lipo
        + w_clash * clashes
        + pi
    )

    return {
        "steric": float(w_steric * steric),
        "ele": float(w_ele * ele),
        "hbond": float(w_hbond * hbond),
        "lipo": float(w_lipo * lipo),
        "clashes": float(w_clash * clashes),
        "pi": float(pi),
        "total": float(total),
    }
