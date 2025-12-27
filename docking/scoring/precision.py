from __future__ import annotations

"""
Precision presets inspired by Glide-style modes.

HTVS: faster, rougher filtering
SP:   default balanced mode
XP:   stricter / more discriminating (heavier penalties)
"""

PRECISION_PRESETS = {
    "HTVS": {
        "steric_cutoff": 6.0,
        "w_steric": 0.7,
        "w_ele": 0.4,
        "w_hbond": 0.6,
        "w_lipo": 0.6,
        "w_clash": 1.0,
        "w_pi": 0.0,
        "clash_hard_overlap": 0.6,
        "clash_penalty": 25.0,
        "hbond_r0": 2.9,
        "hbond_width_out": 0.6,
        "lipo_cutoff": 4.2,
        "lipo_sat": 0.8,
    },
    "SP": {
        "steric_cutoff": 8.0,
        "w_steric": 1.0,
        "w_ele": 0.9,
        "w_hbond": 1.0,
        "w_lipo": 1.0,
        "w_clash": 1.0,
        "w_pi": 0.6,
        "clash_hard_overlap": 0.5,
        "clash_penalty": 60.0,
        "hbond_r0": 2.9,
        "hbond_width_out": 0.8,
        "lipo_cutoff": 4.5,
        "lipo_sat": 1.0,
    },
    "XP": {
        "steric_cutoff": 10.0,
        "w_steric": 1.3,
        "w_ele": 1.0,
        "w_hbond": 1.4,
        "w_lipo": 1.1,
        "w_clash": 1.6,
        "w_pi": 1.0,
        "clash_hard_overlap": 0.35,
        "clash_penalty": 180.0,
        "hbond_r0": 2.9,
        "hbond_width_out": 1.0,
        "lipo_cutoff": 4.8,
        "lipo_sat": 1.2,
        # XP-like extra strictness knobs (used by python engine; numba engine uses approximations)
        "directional_hbond": True,
        "hbond_angle_min_deg": 120.0,
    },
}


def merge_precision(params: dict | None, precision: str = "SP") -> dict:
    p = dict(params or {})
    prec = (precision or "SP").upper()
    if prec not in PRECISION_PRESETS:
        raise ValueError(f"Unknown precision '{precision}'. Use HTVS|SP|XP.")
    merged = dict(PRECISION_PRESETS[prec])
    merged.update(p)  # user overrides take precedence
    merged["precision"] = prec
    return merged
