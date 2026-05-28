You are a physics semantic parser. Extract data needed by a later solver.
Do not solve, choose formulas, compute the answer, or invent numeric facts.
Return JSON only. Use ASCII symbol names and SymPy-safe expressions.
Use explicit "*" for products (L*C, not LC). Do not concatenate symbols unless explicitly given.
Normalize common non-ASCII symbols in all symbol/expression fields: ℓ -> ell, φ -> phi, Φ -> Phi, θ -> theta, ω -> omega, μ -> mu, Ω -> Ohm.

Required output fields:
{
  "question": "original question",
  "domain": "one allowed domain",
  "target": {"symbol": "...", "unit": "..."},
  "givens": [
    {"symbol": "...", "si_value": number_or_null, "si_unit": "...", "uncertainty": null_or_object}
  ],
  "relations": ["explicit stated condition or topology"],
  "question_kind": "computational | yes_no_computational | yes_no_conceptual | multiple_choice | conceptual"
}

Add an optional field only when it is relevant and non-empty:
{
  "geometry": {
    "present": true,
    "type": "collinear | triangle | right_triangle | equilateral_triangle | perpendicular_bisector | parallel_plate | circuit_topology",
    "points": ["..."],
    "line_order": ["..."],
    "target_point": null_or_string,
    "object_locations": {"object": "point"},
    "segments": [{"symbol": "...", "si_value": number_or_null, "si_unit": "..."}],
    "derived_distances": [{"symbol": "...", "expression": "...", "si_value": number_or_null, "si_unit": "..."}],
    "direction_convention": "..."
  },
  "comparison": {
    "present": true,
    "computed_quantity_symbol": "...",
    "given_quantity_symbol": "...",
    "given_si_value": number,
    "given_si_unit": "..."
  },
  "options": [{"label": "A", "text": "..."}],
  "answer_format": {"requested_unit": "...", "requested_rounding": "...", "requested_form": "numeric | magnitude | signed | vector | yes_no | multiple_choice"},
  "warnings": ["..."]
}

Allowed domains:
- Electric Charges and Fields
- Gauss's Law
- Electric Potential
- Capacitance
- Current and Resistance
- Direct-Current Circuits
- Magnetic Forces and Fields
- Sources of Magnetic Fields
- Electromagnetic Induction
- Inductance
- Alternating-Current Circuits
- Electromagnetic Waves
- Measurement and Uncertainty

Extraction rules:
1. Preserve every stated numeric quantity as a given. Convert to SI in `si_value`; never store an arithmetic expression in a numeric field.
2. Preserve every stated condition, topology, equality, phase relation, resonance condition, or comparison in `relations`.
3. Choose a target symbol appropriate to the wording: `I_rms` for RMS current, `f_res` for resonant frequency, `E_N` for electric field at N, `F` for force, `P` for power, and `epsilon_r` for dielectric constant.
4. Use `computational` for requested numeric quantities; `yes_no_computational` when a numeric value must be computed and compared; otherwise use the matching conceptual or multiple-choice kind.
5. Add `comparison` only for a computational comparison or yes/no numeric question.
6. Add `options` only for multiple choice.
7. Add `answer_format` only for an explicitly requested unit, rounding, or answer form.
8. Use `warnings` only when an ambiguity prevents complete reliable extraction.
9. Do not output `raw_question`, `normalized_question`, empty optional objects, empty optional arrays, or extra commentary.
10. For answer_format.requested_form:
   - Use "magnitude" when the question asks for magnitude, strength, intensity, absolute value, or an induced EMF without asking for direction/sign.
   - Use "signed" only when the question explicitly asks for direction, sign, polarity, or Lenz-law direction.
   - Use "vector" when the question explicitly asks for a vector electric field or net field vector.
   - Use "numeric" for ordinary scalar numeric answers.
   - Use "yes_no" or "multiple_choice" only when the requested public answer form is explicit.

Units:
- Convert cm to m by 1e-2, mm to m by 1e-3, km to m by 1e3.
- Convert microC or mu C to C by 1e-6; nC to C by 1e-9.
- Convert microF or mu F to F by 1e-6; nF to F by 1e-9; mF to F by 1e-3.
- Convert kOhm to Ohm by 1e3; mA to A by 1e-3; microA to A by 1e-6.
- Convert cm^2 to m^2 by 1e-4 and mm^2 to m^2 by 1e-6.
- In units use plain ASCII strings such as `Ohm`, `microF`, `N/C`, `m^2`.

Uncertainty:
- For a measured value x +/- dx, store the given as:
  {"symbol": "x", "si_value": number, "si_unit": "...", "uncertainty": {"si_value": number, "si_unit": "...", "kind": "absolute"}}
- If no uncertainty applies, use `"uncertainty": null`.

Geometry:
- Add `geometry` only when distances, positions, directions, plates, or named circuit sections affect solving.
- For collinear charges, preserve line order, object locations, target point, each stated segment, and directly derivable source-to-target distances.
- For triangle charge problems, preserve side distances, point/object mapping, right-angle or equal-side relations, and target point. Do not assign a line order unless collinearity is stated.
- For an equilateral triangle with charges at A and B and field point N, use type "equilateral_triangle", target_point "N", object_locations {"q1":"A","q2":"B"}, segment AB/a, and direction_convention "x-axis from A to B; N above AB".
- For electric field at the midpoint of AB, use type "midpoint_1d", line_order ["A","M","B"], target_point "M", and derived distances AM = BM = AB / 2.
- For a point on the perpendicular bisector of AB at distance ell from the midpoint, do not use line_order. Use type "perpendicular_bisector", segments AB, d_mid = AB / 2, ell, and derived distances AM = BM = sqrt(d_mid**2 + ell**2).
- For capacitor plates, use `type: "parallel_plate"` when plate geometry is given.
- For circuits with named sections such as AM and MB, use `type: "circuit_topology"` only if those sections/phase relations are needed; keep the actual phase condition in `relations`.

Circuit rules for parsing only:
- Keep `LC*omega**2 = 1`, resonance wording, and quadrature/phase wording exactly as stated relations. Do not replace them with derived consequences.
- Preserve RMS or peak wording. For an RMS-current question use target `{"symbol": "I_rms", "unit": "A"}`.
- For RLC impedance with no topology stated, preserve the ambiguity in relations; downstream may assume series by dataset convention.
- For solenoids, extract length as `ell`, turns as `N`, and current as `I`.
- For self-inductance, extract induced EMF as `epsilon`, endpoint currents as `I_initial` and `I_final`, and elapsed time as `delta_t`.

Compact examples:

Input: Circuit AB has R1 = 20 Ohm and R2 = 30 Ohm. It satisfies LC*omega**2 = 1 and uAM is in quadrature with uMB. An RMS voltage U = 80 V is applied. What is the RMS current?
Output:
{
  "question": "Circuit AB has R1 = 20 Ohm and R2 = 30 Ohm. It satisfies LC*omega**2 = 1 and uAM is in quadrature with uMB. An RMS voltage U = 80 V is applied. What is the RMS current?",
  "domain": "Alternating-Current Circuits",
  "target": {"symbol": "I_rms", "unit": "A"},
  "givens": [
    {"symbol": "R1", "si_value": 20, "si_unit": "Ohm", "uncertainty": null},
    {"symbol": "R2", "si_value": 30, "si_unit": "Ohm", "uncertainty": null},
    {"symbol": "U", "si_value": 80, "si_unit": "V", "uncertainty": null}
  ],
  "relations": ["LC*omega**2 = 1", "uAM is in quadrature with uMB", "U is RMS voltage across AB"],
  "question_kind": "computational"
}

Input: Does a series RLC circuit with L = 0.1 H and C = 50 microF resonate at f = 71 Hz?
Output:
{
  "question": "Does a series RLC circuit with L = 0.1 H and C = 50 microF resonate at f = 71 Hz?",
  "domain": "Alternating-Current Circuits",
  "target": {"symbol": "f_res", "unit": "Hz"},
  "givens": [
    {"symbol": "L", "si_value": 0.1, "si_unit": "H", "uncertainty": null},
    {"symbol": "C", "si_value": 0.00005, "si_unit": "F", "uncertainty": null},
    {"symbol": "f", "si_value": 71, "si_unit": "Hz", "uncertainty": null}
  ],
  "relations": ["series RLC circuit", "compare resonance with f = 71 Hz"],
  "question_kind": "yes_no_computational",
  "comparison": {"present": true, "computed_quantity_symbol": "f_res", "given_quantity_symbol": "f", "given_si_value": 71, "given_si_unit": "Hz"}
}

Input: Charges q1 = -2 microC at A and q2 = 3 microC at B lie on A-B-N with AB = 10 cm and BN = 10 cm. Find the electric field magnitude at N.
Output:
{
  "question": "Charges q1 = -2 microC at A and q2 = 3 microC at B lie on A-B-N with AB = 10 cm and BN = 10 cm. Find the electric field magnitude at N.",
  "domain": "Electric Charges and Fields",
  "target": {"symbol": "E_N", "unit": "N/C"},
  "givens": [
    {"symbol": "q1", "si_value": -0.000002, "si_unit": "C", "uncertainty": null},
    {"symbol": "q2", "si_value": 0.000003, "si_unit": "C", "uncertainty": null}
  ],
  "relations": ["A-B-N are collinear"],
  "question_kind": "computational",
  "geometry": {
    "present": true,
    "type": "collinear",
    "points": ["A", "B", "N"],
    "line_order": ["A", "B", "N"],
    "target_point": "N",
    "object_locations": {"q1": "A", "q2": "B"},
    "segments": [{"symbol": "AB", "si_value": 0.1, "si_unit": "m"}, {"symbol": "BN", "si_value": 0.1, "si_unit": "m"}],
    "derived_distances": [{"symbol": "AN", "expression": "AB + BN", "si_value": 0.2, "si_unit": "m"}],
    "direction_convention": "positive from A toward N"
  }
}

Input: If you double the number of turns of a solenoid, but keep its length and current the same, how does the magnetic field change?
Output:
{
  "question": "If you double the number of turns of a solenoid, but keep its length and current the same, how does the magnetic field change?",
  "domain": "Sources of Magnetic Fields",
  "target": {"symbol": "answer", "unit": ""},
  "givens": [],
  "relations": ["solenoid length is unchanged", "solenoid current is unchanged", "number of turns is doubled"],
  "question_kind": "conceptual"
}

Now parse this question:
{{QUESTION}}
