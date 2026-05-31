"""Physics unit normalization and SI conversion table."""

from __future__ import annotations

import re

UNIT_TO_SI: dict[str, tuple[float, str]] = {
    "cm": (1e-2, "m"),
    "cm2": (1e-4, "m2"),
    "cm^2": (1e-4, "m2"),
    "cm²": (1e-4, "m2"),
    "mm": (1e-3, "m"),
    "mm2": (1e-6, "m2"),
    "mm^2": (1e-6, "m2"),
    "mm²": (1e-6, "m2"),
    "km": (1e3, "m"),
    "m2": (1.0, "m2"),
    "m^2": (1.0, "m2"),
    "m²": (1.0, "m2"),
    "microc": (1e-6, "C"),
    "muc": (1e-6, "C"),
    "μc": (1e-6, "C"),
    "µc": (1e-6, "C"),
    "uc": (1e-6, "C"),
    "pc": (1e-12, "C"),
    "nc": (1e-9, "C"),
    "mc": (1e-3, "C"),
    "c": (1.0, "C"),
    "microf": (1e-6, "F"),
    "muf": (1e-6, "F"),
    "μf": (1e-6, "F"),
    "µf": (1e-6, "F"),
    "uf": (1e-6, "F"),
    "pf": (1e-12, "F"),
    "nf": (1e-9, "F"),
    "mf": (1e-3, "F"),
    "f": (1.0, "F"),
    "kohm": (1e3, "Ohm"),
    "kω": (1e3, "Ohm"),
    "ma": (1e-3, "A"),
    "microa": (1e-6, "A"),
    "mua": (1e-6, "A"),
    "μa": (1e-6, "A"),
    "µa": (1e-6, "A"),
    "ua": (1e-6, "A"),
    "a": (1.0, "A"),
    "v": (1.0, "V"),
    "mv": (1e-3, "V"),
    "kv": (1e3, "V"),
    "microh": (1e-6, "H"),
    "muh": (1e-6, "H"),
    "μh": (1e-6, "H"),
    "µh": (1e-6, "H"),
    "uh": (1e-6, "H"),
    "mh": (1e-3, "H"),
    "h": (1.0, "H"),
    "hz": (1.0, "Hz"),
    "khz": (1e3, "Hz"),
    "Mhz": (1e6, "Hz"),
    "rad/s": (1.0, "rad/s"),
    "ohm": (1.0, "Ohm"),
    "ω": (1.0, "Ohm"),
    "mohm": (1e-3, "Ohm"),
    "Mohm": (1e6, "Ohm"),
    "j": (1.0, "J"),
    "mj": (1e-3, "J"),
    "μj": (1e-6, "J"),
    "µj": (1e-6, "J"),
    "microj": (1e-6, "J"),
    "uj": (1e-6, "J"),
    "wb": (1.0, "Wb"),
    "mwb": (1e-3, "Wb"),
    "microwb": (1e-6, "Wb"),
    "muwb": (1e-6, "Wb"),
    "μwb": (1e-6, "Wb"),
    "µwb": (1e-6, "Wb"),
    "uwb": (1e-6, "Wb"),
    "nwb": (1e-9, "Wb"),
    "t": (1.0, "T"),
    "n": (1.0, "N"),
    "mn": (1e-3, "N"),
    "n/c": (1.0, "N/C"),
    "v/m": (1.0, "V/m"),
    "w": (1.0, "W"),
    "s": (1.0, "s"),
    "ms": (1e-3, "s"),
    "micros": (1e-6, "s"),
    "mus": (1e-6, "s"),
    "μs": (1e-6, "s"),
    "µs": (1e-6, "s"),
    "us": (1e-6, "s"),
    "ml": (1e-6, "m3"),
}
UNIT_PATTERN = "|".join(
    re.escape(unit)
    for unit in sorted(UNIT_TO_SI, key=len, reverse=True)
)


def _canonical_unit_key(unit: str) -> str:
    """Normalize unit spelling while preserving uppercase mega prefix."""
    raw = str(unit or "").strip()
    normalized = raw.replace("Ω", "Ohm").replace("μ", "u").replace("µ", "u")
    normalized = normalized.replace("^2", "2").replace("²", "2")
    normalized = re.sub(r"\s+", "", normalized)
    if len(normalized) > 1 and normalized[0] == "M" and normalized[1].isalpha():
        return "M" + normalized[1:].lower()
    return normalized.lower()
