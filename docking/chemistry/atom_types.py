"""docking.chemistry.atom_types

Tiny helpers for element normalization.
"""

from __future__ import annotations


def normalize_element(element: str) -> str:
    e = (element or "").strip()
    if not e:
        return "X"
    if len(e) == 1:
        return e.upper()
    return e[0].upper() + e[1:].lower()
