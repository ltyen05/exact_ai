"""
Abstract formula templates for common physics problems.
Used to solve physics equations with SymPy.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Standard abstract templates for physics categories
TEMPLATES: Dict[str, Dict[str, Any]] = {
    "resultant_force": {
        "description": "Resultant force of two forces at an angle.",
        "formulas": [
            "F = sqrt(F1**2 + F2**2 + 2*F1*F2*cos(theta * 3.141592653589793 / 180))"
        ],
        "default_target": "F",
        "keywords": ["resultant force", "resultant", "angle", "degrees", "°"],
    },
    "resultant_force_same": {
        "description": "Resultant force of two forces in the same direction.",
        "formulas": ["F = F1 + F2"],
        "default_target": "F",
        "keywords": ["resultant force", "same direction"],
    },
    "resultant_force_opposite": {
        "description": "Resultant force of two forces in opposite directions.",
        "formulas": ["F = abs(F1 - F2)"],
        "default_target": "F",
        "keywords": ["resultant force", "opposite direction", "opposite directions"],
    },
    "resultant_force_perpendicular": {
        "description": "Resultant force of two perpendicular forces.",
        "formulas": ["F = sqrt(F1**2 + F2**2)"],
        "default_target": "F",
        "keywords": ["resultant force", "perpendicular", "90 degrees", "90°"],
    },
    "capacitor_energy": {
        "description": "Energy stored in a capacitor: E = 0.5 * C * U^2.",
        "formulas": ["E = 0.5 * C * U**2"],
        "default_target": "E",
        "keywords": ["energy stored", "capacitor", "capacitance", "voltage", "potential difference"],
    },
    "capacitor_charge": {
        "description": "Charge stored in a capacitor: Q = C * U.",
        "formulas": ["Q = C * U"],
        "default_target": "Q",
        "keywords": ["charge", "capacitor", "capacitance", "voltage", "potential difference"],
    },
    "parallel_plate_capacitance": {
        "description": "Capacitance of a parallel-plate capacitor: C = eps0 * A / d.",
        "formulas": [
            "C = eps0 * A / d",
            "eps0 = 8.854e-12"
        ],
        "default_target": "C",
        "keywords": ["parallel-plate", "capacitance", "plate area", "separation", "distance"],
    }
}


def find_matching_template(question: str) -> Optional[str]:
    """Find the best matching template key based on keywords."""
    qlow = question.lower()
    
    # Check parallel plate capacitor
    if "parallel-plate" in qlow and "capacitance" in qlow:
        return "parallel_plate_capacitance"
        
    # Check capacitor energy/charge
    if "capacitor" in qlow or "capacitance" in qlow:
        if "energy" in qlow or "stored" in qlow:
            return "capacitor_energy"
        if "charge" in qlow:
            return "capacitor_charge"
            
    # Check resultant forces
    if "resultant" in qlow or "forces" in qlow:
        if "same direction" in qlow:
            return "resultant_force_same"
        if "opposite direction" in qlow or "opposite directions" in qlow:
            return "resultant_force_opposite"
        if "perpendicular" in qlow or "90" in qlow:
            return "resultant_force_perpendicular"
        if "angle" in qlow or "degree" in qlow or "°" in qlow:
            return "resultant_force"
            
    return None
