from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import api
from agents.logic import LogicAgent
from agents.llm.openrouter_provider import OpenRouterClient
from agents.physics.Explain import ExplainAgent
from agents.physics.Parsing import ParsingAgent
from agents.physics.Solution.llm_provider import LLMSolutionProvider
from agents.physics.Solution.rag_provider import RAGSolutionProvider
from agents.physics.domain.context import build_calculation_input
from agents.physics.domain.units import UNIT_TO_SI
from agents.workflows.orchestrator import ExactGraph, FormatterNode, WorkflowExecutionError
from agents.workflows.physics import PhysicsWorkflow
from agents.workflows.tracing import _clean, _process_llm_inputs, _process_step_outputs
from eval.p1_evaluator import answer_match, evaluate_prediction, summarize_p1
from tools.calculator import solve_with_sympy_trace

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StaticClassifier:
    def __init__(self, route: str) -> None:
        self.route = route

    def run(self, question: str) -> dict[str, object]:
        del question
        return {"Type": self.route}


class RecordingLLM:
    enabled = True
    provider = "test"
    model = "test-model"

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        stage: str = "llm.chat",
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "stage": stage,
            }
        )
        return self.responses[stage]


class RaisingLLM(RecordingLLM):
    def __init__(self) -> None:
        super().__init__({})

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        stage: str = "llm.chat",
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
                "stage": stage,
            }
        )
        raise RuntimeError("simulated provider 400")


def solution_llm(payload: dict[str, Any]) -> RecordingLLM:
    return RecordingLLM({"physics.solution": json.dumps(payload)})


class CalculatorTests(unittest.TestCase):
    def test_supported_functions_and_intermediates(self) -> None:
        magnitude = solve_with_sympy_trace(
            {"k": 9e9, "q": -2e-6, "r": 0.3},
            ["E = Abs(k * q / r**2)"],
            "E",
        )
        multi = solve_with_sympy_trace(
            {"U": 80, "R1": 25, "R2": 40},
            ["R_total = R1 + R2", "I = U / R_total"],
            "I",
        )
        self.assertAlmostEqual(magnitude.value, 200000.0)
        self.assertAlmostEqual(multi.value, 80 / 65)
        self.assertIn("R_total = 65.0", multi.trace)

    def test_diff_equations_can_reference_prior_symbolic_expression(self) -> None:
        result = solve_with_sympy_trace(
            {"a": 2.0},
            [
                "E_total = h / (a**2 + h**2)**(3/2)",
                "dE_dh = diff(E_total, h)",
                "dE_dh = 0",
            ],
            "h",
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.value, math.sqrt(2.0))


class ParsingTests(unittest.TestCase):
    def test_prompt_is_compact_and_drops_empty_optional_sections(self) -> None:
        llm = RecordingLLM(
            {
                "physics.parsing": json.dumps(
                    {
                        "raw_question": "discard this",
                        "question": "Find RMS current.",
                        "domain": "Alternating-Current Circuits",
                        "target": {"symbol": "I_rms", "unit": "A"},
                        "givens": [{"symbol": "U", "si_value": 80, "si_unit": "V", "uncertainty": None}],
                        "relations": ["U is RMS"],
                        "question_kind": "computational",
                        "geometry": {"present": False},
                        "comparison": {"present": False},
                        "options": [],
                        "warnings": [],
                    }
                )
            }
        )
        agent = ParsingAgent(llm_provider=llm)
        output = agent.run("Find RMS current.")

        self.assertLessEqual(len(agent.prompt_template), 10000)
        self.assertEqual(llm.calls[0]["stage"], "physics.parsing")
        self.assertNotIn("raw_question", output)
        self.assertNotIn("geometry", output)
        self.assertNotIn("comparison", output)
        self.assertNotIn("options", output)

    def test_prompts_match_current_physics_contract(self) -> None:
        parser_prompt = (PROJECT_ROOT / "prompts" / "semantic_parser_type2.md").read_text(encoding="utf-8")
        solution_prompt = (PROJECT_ROOT / "prompts" / "solution_type2.md").read_text(encoding="utf-8")

        self.assertIn("mN", parser_prompt)
        self.assertIn("I_max", parser_prompt)
        self.assertIn("Q_source", parser_prompt)
        self.assertIn("diff", solution_prompt)
        self.assertIn("Preserve target consistency", solution_prompt)
        self.assertIn("Never use Coulomb constant `k`", solution_prompt)
        self.assertNotIn('"units": {}', solution_prompt)

    def test_physics_solver_imports_state_not_orchestrator(self) -> None:
        solver_files = [
            PROJECT_ROOT / "agents" / "physics" / "solver" / "answer_builder.py",
            PROJECT_ROOT / "agents" / "physics" / "solver" / "direct_answer.py",
            PROJECT_ROOT / "agents" / "physics" / "solver" / "repair.py",
        ]
        for path in solver_files:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIn("agents.workflows.state", source)
                self.assertNotIn("agents.workflows.orchestrator import WorkflowExecutionError", source)

    def test_parser_preserves_relevant_geometry_and_comparison(self) -> None:
        geometry = {"present": True, "type": "collinear", "line_order": ["A", "N"]}
        comparison = {"present": True, "given_quantity_symbol": "f", "given_si_value": 71}
        llm = RecordingLLM(
            {
                "physics.parsing": json.dumps(
                    {
                        "question": "Question",
                        "domain": "Alternating-Current Circuits",
                        "target": {"symbol": "f_res", "unit": "Hz"},
                        "givens": [],
                        "relations": [],
                        "question_kind": "yes_no_computational",
                        "geometry": geometry,
                        "comparison": comparison,
                    }
                )
            }
        )
        output = ParsingAgent(llm_provider=llm).run("Question")
        self.assertEqual(output["geometry"], geometry)
        self.assertEqual(output["comparison"], comparison)

    def test_parser_keeps_explicit_multiplication_relation(self) -> None:
        llm = RecordingLLM(
            {
                "physics.parsing": json.dumps(
                    {
                        "question": "Circuit AB satisfies LCω² = 1. Find omega.",
                        "domain": "Alternating-Current Circuits",
                        "target": {"symbol": "omega", "unit": "rad/s"},
                        "givens": [],
                        "relations": ["L*C*omega**2 = 1"],
                        "question_kind": "computational",
                    }
                )
            }
        )
        output = ParsingAgent(llm_provider=llm).run("Circuit AB satisfies LCω² = 1. Find omega.")
        self.assertIn("L*C*omega**2 = 1", output["relations"])

    def test_parser_corrects_prefixed_units_from_original_question(self) -> None:
        cases = [
            (
                "A capacitor has capacitance C = 25 μF and is connected to a voltage U = 120 V.",
                0.025,
                25e-6,
                120.0,
            ),
            (
                "A capacitor has capacitance C = 40 μF and is connected to a voltage U = 50 V.",
                0.04,
                40e-6,
                50.0,
            ),
        ]
        for question, bad_capacitance, expected_capacitance, expected_voltage in cases:
            with self.subTest(question=question):
                llm = RecordingLLM(
                    {
                        "physics.parsing": json.dumps(
                            {
                                "question": question,
                                "domain": "Capacitance",
                                "target": {"symbol": "W", "unit": "J"},
                                "givens": [
                                    {"symbol": "C", "si_value": bad_capacitance, "si_unit": "F", "uncertainty": None},
                                    {"symbol": "U", "si_value": expected_voltage, "si_unit": "V", "uncertainty": None},
                                ],
                                "relations": ["capacitor energy formula"],
                                "question_kind": "computational",
                            }
                        )
                    }
                )
                output = ParsingAgent(llm_provider=llm).run(question)

                self.assertAlmostEqual(output["givens"][0]["si_value"], expected_capacitance)
                self.assertEqual(output["givens"][0]["si_unit"], "F")
                self.assertEqual(output["givens"][1]["si_value"], expected_voltage)

    def test_parser_normalizes_maximum_current_alias(self) -> None:
        llm = RecordingLLM(
            {
                "physics.parsing": json.dumps(
                    {
                        "question": "An inductor has L = 0.25 H and maximum current 2sqrt2 A.",
                        "domain": "Inductance",
                        "target": {"symbol": "W_max", "unit": "J"},
                        "givens": [
                            {"symbol": "L", "si_value": 0.25, "si_unit": "H", "uncertainty": None},
                            {"symbol": "maximum_current", "si_value": str(2 * math.sqrt(2)), "si_unit": "A", "uncertainty": None},
                        ],
                        "relations": [],
                        "question_kind": "computational",
                    }
                )
            }
        )
        output = ParsingAgent(llm_provider=llm).run("An inductor has L = 0.25 H and maximum current 2sqrt2 A.")
        calculation = build_calculation_input(output)

        self.assertAlmostEqual(calculation["quantities"]["I_max"], 2 * math.sqrt(2))

    def test_domain_build_calculation_input_without_parser_agent(self) -> None:
        parsed = {
            "quantities": {"lambda": 0.012},
            "givens": [
                {"symbol": "U", "si_value": 12.0, "uncertainty": {"si_value": 0.2}},
                {"symbol": "I", "value": 0.5},
            ],
            "geometry": {
                "segments": [{"symbol": "AB", "si_value": 0.08}],
                "derived_distances": [{"symbol": "AM", "si_value": 0.04}],
            },
            "comparison": {
                "given_quantity_symbol": "56.3",
                "given_si_value": 56.3,
                "given_si_unit": "Hz",
            },
            "target": {"symbol": "R", "unit": "Ohm"},
        }

        calculation = build_calculation_input(parsed)

        self.assertEqual(calculation["target"], "R")
        self.assertEqual(calculation["unit"], "Ohm")
        quantities = calculation["quantities"]
        self.assertEqual(quantities["lambda_"], 0.012)
        self.assertEqual(quantities["flux_linkage"], 0.012)
        self.assertEqual(quantities["U"], 12.0)
        self.assertEqual(quantities["V"], 12.0)
        self.assertEqual(quantities["I"], 0.5)
        self.assertEqual(quantities["delta_U"], 0.2)
        self.assertEqual(quantities["AB"], 0.08)
        self.assertEqual(quantities["AM"], 0.04)
        self.assertEqual(quantities["f"], 56.3)
        self.assertEqual(quantities["frequency"], 56.3)

    def test_domain_unit_conversion_table_keeps_required_prefixed_units(self) -> None:
        cases = [
            ("microf", 1e-6, "F"),
            ("mh", 1e-3, "H"),
            ("cm²", 1e-4, "m2"),
            ("kohm", 1e3, "Ohm"),
        ]

        for unit, expected_scale, expected_si_unit in cases:
            with self.subTest(unit=unit):
                scale, si_unit = UNIT_TO_SI[unit]
                self.assertEqual(si_unit, expected_si_unit)
                self.assertTrue(math.isclose(scale, expected_scale, rel_tol=0, abs_tol=1e-18))

    def test_validation_imports_domain_context_not_parser(self) -> None:
        source = (PROJECT_ROOT / "agents" / "physics" / "validation.py").read_text(encoding="utf-8")

        self.assertIn("from agents.physics.domain.context import build_calculation_input", source)
        self.assertNotIn("from agents.physics.Parsing.Parsing_Agent import build_calculation_input", source)

    def test_formula_lib_imports_quantity_alias_groups_from_domain_not_parser(self) -> None:
        source = (PROJECT_ROOT / "agents" / "physics" / "formulas" / "legacy.py").read_text(encoding="utf-8")

        self.assertIn("from agents.physics.domain.symbols import QUANTITY_ALIAS_GROUPS", source)
        self.assertNotIn("from agents.physics.Parsing.Parsing_Agent import QUANTITY_ALIAS_GROUPS", source)

    def test_parser_normalizes_resonance_yes_no_question(self) -> None:
        question = "An AC circuit consists of R=10 Ω, L=0.05 H, C=100 μF. When the frequency is 225 Hz, does resonance occur?"
        llm = RecordingLLM(
            {
                "physics.parsing": json.dumps(
                    {
                        "question": question,
                        "domain": "Alternating-Current Circuits",
                        "target": {"symbol": "answer", "unit": ""},
                        "givens": [
                            {"symbol": "R", "si_value": 10, "si_unit": "Ohm", "uncertainty": None},
                            {"symbol": "L", "si_value": 0.05, "si_unit": "H", "uncertainty": None},
                            {"symbol": "C", "si_value": 0.1, "si_unit": "F", "uncertainty": None},
                        ],
                        "relations": ["AC circuit"],
                        "question_kind": "computational",
                    }
                )
            }
        )
        output = ParsingAgent(llm_provider=llm).run(question)

        self.assertEqual(output["question_kind"], "yes_no_computational")
        self.assertEqual(output["target"], {"symbol": "f_res", "unit": "Hz"})
        self.assertEqual(output["comparison"]["given_quantity_symbol"], "f")
        self.assertEqual(output["comparison"]["given_si_value"], 225)
        self.assertAlmostEqual(output["givens"][2]["si_value"], 100e-6)
        self.assertEqual(output["givens"][3]["symbol"], "f")

    def test_parser_normalizes_literal_resonance_frequency_comparison(self) -> None:
        question = "A series RLC circuit has R=75 Ω, L=0.2 H, C=40 μF. Is 56.3 Hz the resonant frequency?"
        llm = RecordingLLM(
            {
                "physics.parsing": json.dumps(
                    {
                        "question": question,
                        "domain": "Alternating-Current Circuits",
                        "target": {"symbol": "f_res", "unit": "Hz"},
                        "givens": [
                            {"symbol": "R", "si_value": 75, "si_unit": "Ohm", "uncertainty": None},
                            {"symbol": "L", "si_value": 0.2, "si_unit": "H", "uncertainty": None},
                            {"symbol": "C", "si_value": 0.04, "si_unit": "F", "uncertainty": None},
                        ],
                        "relations": ["series RLC circuit", "compare resonant frequency with 56.3 Hz"],
                        "question_kind": "yes_no_computational",
                        "comparison": {
                            "present": True,
                            "computed_quantity_symbol": "f_res",
                            "given_quantity_symbol": "56.3",
                            "given_si_value": 56.3,
                            "given_si_unit": "Hz",
                        },
                    }
                )
            }
        )
        output = ParsingAgent(llm_provider=llm).run(question)

        self.assertEqual(output["comparison"]["given_quantity_symbol"], "f")
        self.assertEqual(output["comparison"]["given_si_value"], 56.3)
        self.assertEqual(output["givens"][3], {"symbol": "f", "si_value": 56.3, "si_unit": "Hz", "uncertainty": None})
        self.assertAlmostEqual(output["givens"][2]["si_value"], 40e-6)

    def test_parser_marks_formula_only_resonant_angular_frequency_conceptual(self) -> None:
        question = "What is the resonant angular frequency of an LC circuit?"
        llm = RecordingLLM(
            {
                "physics.parsing": json.dumps(
                    {
                        "question": question,
                        "domain": "Alternating-Current Circuits",
                        "target": {"symbol": "omega_res", "unit": "rad/s"},
                        "givens": [],
                        "relations": ["LC circuit"],
                        "question_kind": "computational",
                    }
                )
            }
        )
        output = ParsingAgent(llm_provider=llm).run(question)

        self.assertEqual(output["question_kind"], "conceptual")
        self.assertEqual(output["target"], {"symbol": "answer", "unit": ""})
        self.assertEqual(output["answer_format"]["requested_form"], "conceptual")

    def test_parser_template_is_loaded_once_at_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "parser.md"
            prompt_path.write_text("first {{QUESTION}}", encoding="utf-8")
            llm = RecordingLLM(
                {
                    "physics.parsing": json.dumps(
                        {
                            "domain": "Capacitance",
                            "target": {"symbol": "C", "unit": "F"},
                            "givens": [],
                            "relations": [],
                            "question_kind": "computational",
                        }
                    )
                }
            )
            agent = ParsingAgent(prompt_path=str(prompt_path), llm_provider=llm)
            prompt_path.write_text("second {{QUESTION}}", encoding="utf-8")
            agent.run("question")
        sent_prompt = llm.calls[0]["messages"][0]["content"]
        self.assertIn("first question", sent_prompt)
        self.assertNotIn("second", sent_prompt)

    def test_parser_repairs_non_json_conceptual_response_and_normalizes_symbols(self) -> None:
        llm = RecordingLLM(
            {
                "physics.parsing": "The field doubles because B is proportional to N.",
                "physics.parsing.retry": "still not json",
                "physics.parsing.repair": json.dumps(
                    {
                        "question": "If you double the number of turns of a solenoid, but keep its length and current the same, how does the magnetic field change?",
                        "domain": "Sources of Magnetic Fields",
                        "target": {"symbol": "B", "unit": ""},
                        "givens": [{"symbol": "ℓ", "si_value": 0.2, "si_unit": "m", "uncertainty": None}],
                        "relations": ["solenoid length ℓ is unchanged", "current is unchanged", "number of turns is doubled"],
                        "question_kind": "conceptual",
                    }
                ),
            }
        )
        output = ParsingAgent(llm_provider=llm).run(
            "If you double the number of turns of a solenoid, but keep its length and current the same, how does the magnetic field change?"
        )
        self.assertEqual([call["stage"] for call in llm.calls], ["physics.parsing", "physics.parsing.retry", "physics.parsing.repair"])
        self.assertEqual(output["question_kind"], "conceptual")
        self.assertEqual(output["givens"][0]["symbol"], "ell")
        self.assertIn("ell", output["relations"][0])

    def test_parser_repairs_perpendicular_bisector_geometry(self) -> None:
        llm = RecordingLLM(
            {
                "physics.parsing": json.dumps(
                    {
                        "question": "Two charges at A and B. Find field at M on perpendicular bisector.",
                        "domain": "Electric Charges and Fields",
                        "target": {"symbol": "E_M", "unit": "N/C"},
                        "givens": [
                            {"symbol": "q1", "si_value": 5e-7, "si_unit": "C", "uncertainty": None},
                            {"symbol": "q2", "si_value": -5e-7, "si_unit": "C", "uncertainty": None},
                            {"symbol": "d_AB", "si_value": 0.06, "si_unit": "m", "uncertainty": None},
                            {"symbol": "ℓ", "si_value": 0.04, "si_unit": "m", "uncertainty": None},
                        ],
                        "relations": [
                            "M is on the perpendicular bisector of AB",
                            "distance from midpoint to M is ℓ",
                        ],
                        "question_kind": "computational",
                        "geometry": {
                            "present": True,
                            "type": "right_triangle",
                            "points": ["A", "B", "M"],
                            "line_order": ["A", "B", "M"],
                            "target_point": "M",
                            "object_locations": {"q1": "A", "q2": "B"},
                            "segments": [
                                {"symbol": "AB", "si_value": 0.06, "si_unit": "m"},
                                {"symbol": "BM", "si_value": 0.04, "si_unit": "m"},
                            ],
                            "derived_distances": [
                                {"symbol": "AM", "expression": "AB / 2", "si_value": 0.03, "si_unit": "m"}
                            ],
                        },
                    }
                )
            }
        )
        output = ParsingAgent(llm_provider=llm).run(
            "Two charges at A and B. Find field at M on perpendicular bisector."
        )
        geometry = output["geometry"]
        self.assertEqual(geometry["type"], "perpendicular_bisector")
        self.assertNotIn("line_order", geometry)
        self.assertEqual(geometry["segments"][1]["symbol"], "d_mid")
        self.assertAlmostEqual(geometry["segments"][1]["si_value"], 0.03)
        self.assertEqual([item["symbol"] for item in geometry["derived_distances"]], ["AM", "BM"])
        self.assertAlmostEqual(geometry["derived_distances"][0]["si_value"], 0.05)
        self.assertEqual(output["givens"][3]["symbol"], "ell")

    def test_parser_raises_when_repair_is_not_json(self) -> None:
        llm = RecordingLLM(
            {
                "physics.parsing": "not json",
                "physics.parsing.retry": "still not json",
                "physics.parsing.repair": "still not json",
                "physics.parsing.repair2": "still not json",
            }
        )
        with self.assertRaisesRegex(ValueError, "Physics parser response must be a JSON object"):
            ParsingAgent(llm_provider=llm).run("Conceptual question")

    def test_parser_retries_compact_json_after_truncated_response(self) -> None:
        question = "The current through the solenoid is 2 A and the number of turns per meter is 1500. Calculate the magnetic field inside."
        llm = RecordingLLM(
            {
                "physics.parsing": '{"question": "The current through the solenoid is 2 A", "domain": ',
                "physics.parsing.retry": json.dumps(
                    {
                        "question": question,
                        "domain": "Sources of Magnetic Fields",
                        "target": {"symbol": "B", "unit": "T"},
                        "givens": [
                            {"symbol": "I", "si_value": 2, "si_unit": "A", "uncertainty": None},
                            {"symbol": "n", "si_value": 1500, "si_unit": "1/m", "uncertainty": None},
                        ],
                        "relations": ["long solenoid"],
                        "question_kind": "computational",
                    }
                ),
            }
        )
        output = ParsingAgent(llm_provider=llm).run(question)

        self.assertEqual([call["stage"] for call in llm.calls], ["physics.parsing", "physics.parsing.retry"])
        self.assertEqual(output["target"]["symbol"], "B")

    def test_parser_uses_heuristic_fallback_for_solenoid_when_all_json_attempts_fail(self) -> None:
        question = "The current through the solenoid is 2 A, the number of turns per meter is 1500. Calculate the magnetic field inside."
        llm = RecordingLLM(
            {
                "physics.parsing": "not json",
                "physics.parsing.retry": "still not json",
                "physics.parsing.repair": "still not json",
                "physics.parsing.repair2": "still not json",
            }
        )
        output = ParsingAgent(llm_provider=llm).run(question)

        self.assertEqual(output["target"]["symbol"], "B")
        givens = {item["symbol"]: item["si_value"] for item in output["givens"]}
        self.assertEqual(givens["I"], 2.0)
        self.assertEqual(givens["n"], 1500.0)

class PhysicsWorkflowTests(unittest.TestCase):
    def test_solution_validator_treats_formula_identifiers_as_symbols(self) -> None:
        LLMSolutionProvider._validate_equation("E1x = E1 * (AC - AB / 2) / AC")

    def test_solution_validator_accepts_complex_functions(self) -> None:
        LLMSolutionProvider._validate_solution(
            {
                "mode": "computational",
                "answer_type": "numeric",
                "sympy_spec": {
                    "target_symbol": "P",
                    "target_unit": "W",
                    "equations": ["P = Im(conjugate(Z_total))"],
                    "known_values": {"Z_total": 5},
                },
                "solution_steps": [],
            }
        )

    def test_solution_validator_rejects_unresolved_helper_symbols(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unresolved symbols: .*r1.*x1"):
            LLMSolutionProvider._validate_solution(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "E_M",
                        "target_unit": "N/C",
                        "equations": [
                            "E1x = -k * q1 * x1 / r1**3",
                            "E_M = Abs(E1x)",
                        ],
                        "known_values": {"k": 9e9, "q1": 5e-7},
                    },
                    "solution_steps": [],
                }
            )

    def test_solution_validator_rejects_missing_vector_spec_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "vector_spec component symbols"):
            LLMSolutionProvider._validate_solution(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "E_net_magnitude",
                        "target_unit": "N/C",
                        "equations": ["E_net_magnitude = Abs(E1x)"],
                        "known_values": {"E1x": 10},
                    },
                    "solution_steps": [],
                    "vector_spec": {
                        "component_symbols": ["E_net_x"],
                        "magnitude_symbol": "E_net_magnitude",
                    },
                }
            )

    def test_solution_validator_rejects_degree_literals_in_trig(self) -> None:
        with self.assertRaisesRegex(ValueError, "degree literal"):
            LLMSolutionProvider._validate_solution(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "F_resultant",
                        "target_unit": "N",
                        "equations": ["F_resultant = sqrt(F1**2 + F2**2 + 2 * F1 * F2 * cos(60))"],
                        "known_values": {"F1": 5, "F2": 5},
                    },
                    "solution_steps": [],
                }
            )

    def test_solution_validator_rejects_raw_coordinate_force_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw Cx/Cy"):
            LLMSolutionProvider._validate_solution(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "F_net",
                        "target_unit": "N",
                        "equations": ["Fx = F * Cx", "Fy = F * Cy", "F_net = sqrt(Fx**2 + Fy**2)"],
                        "known_values": {"F": 10, "Cx": 3, "Cy": 4},
                    },
                    "solution_steps": [],
                }
            )

    def test_solution_validator_rejects_coulomb_force_missing_test_charge(self) -> None:
        with self.assertRaisesRegex(ValueError, "qi\\*q0"):
            LLMSolutionProvider._validate_solution(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "F_net",
                        "target_unit": "N",
                        "equations": ["F1 = k * Abs(q1) / r1**2", "F_net = F1"],
                        "known_values": {"k": 9e9, "q1": 4e-6, "q0": -2e-6, "r1": 0.04},
                    },
                    "solution_steps": [],
                }
            )

    def test_solution_provider_normalizes_lambda_and_charge_aliases(self) -> None:
        llm = solution_llm(
            {
                "mode": "computational",
                "answer_type": "numeric",
                "sympy_spec": {
                    "target_symbol": "E",
                    "target_unit": "N/C",
                    "equations": ["E = k * lambda * L / (r * sqrt(r**2 + L**2))", "F = k * charge_q2 / r**2"],
                    "known_values": {"charge_q2": 2e-9, "lambda": 3e-6},
                },
                "solution_steps": [],
            }
        )
        provider = LLMSolutionProvider(llm)
        parsed = {
            "question": "Find E.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "E", "unit": "N/C"},
            "givens": [{"symbol": "q2", "si_value": 2e-9}, {"symbol": "r", "si_value": 0.2}, {"symbol": "L", "si_value": 0.1}],
            "question_kind": "computational",
        }
        output = provider.get_solution("Find E.", parsed)
        self.assertIn("lambda_", " ".join(output["sympy_spec"]["equations"]))
        self.assertIn("q2", output["sympy_spec"]["known_values"])

    def test_zero_field_solution_respects_requested_distance_symbol(self) -> None:
        parsed = {
            "question": "Find distance BM where field is zero.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "BM", "unit": "m"},
            "givens": [
                {"symbol": "q1", "si_value": 9e-8},
                {"symbol": "q2", "si_value": -16e-8},
                {"symbol": "AB", "si_value": 0.12},
            ],
            "relations": ["electric field is zero"],
            "question_kind": "computational",
        }
        output = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
        self.assertEqual(output["sympy_spec"]["target_symbol"], "BM")

    def test_solution_provider_repairs_invalid_dependency_closure_once(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "E_M",
                            "target_unit": "N/C",
                            "equations": ["E1x = -k * q1 * x1 / r1**3", "E_M = Abs(E1x)"],
                            "known_values": {"k": 9e9, "q1": 5e-7, "d_AB": 0.06, "AM": 0.05},
                        },
                        "solution_steps": [],
                    }
                ),
                "physics.solution.retry": "still not json",
                "physics.solution.repair": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "E_M",
                            "target_unit": "N/C",
                            "equations": [
                                "x1 = d_AB / 2",
                                "r1 = AM",
                                "E1x = -k * q1 * x1 / r1**3",
                                "E_M = Abs(E1x)",
                            ],
                            "known_values": {"k": 9e9, "q1": 5e-7, "d_AB": 0.06, "AM": 0.05},
                        },
                        "solution_steps": ["Define helper coordinates and distances before computing components."],
                    }
                ),
            }
        )
        provider = LLMSolutionProvider(llm)
        output = provider.get_solution("Find E.", {"question": "Find E.", "target": {"symbol": "E_M", "unit": "N/C"}})
        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution", "physics.solution.retry", "physics.solution.repair"])
        self.assertIn("r1 = AM", output["sympy_spec"]["equations"])

    def test_solution_provider_retries_after_unresolved_symbols(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "R2",
                            "target_unit": "Ohm",
                            "equations": ["Z = sqrt(R2**2 + (2 * pi * f * L - 1 / (2 * pi * f * C))**2)"],
                            "known_values": {"Z": 100},
                        },
                        "solution_steps": [],
                    }
                ),
                "physics.solution.retry": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "R2",
                            "target_unit": "Ohm",
                            "equations": ["R2 = sqrt(Z**2 - X_net**2)"],
                            "known_values": {"Z": 100, "X_net": 60},
                        },
                        "solution_steps": ["Use only parsed impedance and net reactance values."],
                    }
                ),
            }
        )
        parsed = {
            "question": "Find R2 from total impedance Z = 100 Ohm and net reactance X_net = 60 Ohm.",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "R2", "unit": "Ohm"},
            "givens": [{"symbol": "Z", "si_value": 100}, {"symbol": "X_net", "si_value": 60}],
            "question_kind": "computational",
        }
        provider = LLMSolutionProvider(llm)
        output = provider.get_solution(parsed["question"], parsed)

        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution", "physics.solution.retry"])
        self.assertEqual(output["sympy_spec"]["equations"], ["R2 = sqrt(Z**2 - X_net**2)"])

    def test_solution_provider_repairs_non_json_initial_response(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": "Use Coulomb's law and compute the field.",
                "physics.solution.retry": "still not json",
                "physics.solution.repair": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "E",
                            "target_unit": "N/C",
                            "equations": ["E = Abs(k * q / r**2)"],
                            "known_values": {"k": 9e9, "q": 2e-6, "r": 0.3},
                        },
                        "solution_steps": ["Use Coulomb's law for a point charge field."],
                    }
                ),
            }
        )
        provider = LLMSolutionProvider(llm)
        output = provider.get_solution("Find E.", {"question": "Find E.", "target": {"symbol": "E", "unit": "N/C"}})
        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution", "physics.solution.retry", "physics.solution.repair"])
        self.assertEqual(output["sympy_spec"]["target_symbol"], "E")
        self.assertTrue(provider.last_prompt_diagnostics["used_repair"])

    def test_solution_provider_retries_general_solution_on_non_json(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": "not json",
                "physics.solution.retry": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "formula_ids": ["ac.resonance.frequency"],
                        "sympy_spec": {
                            "target_symbol": "f_res",
                            "target_unit": "Hz",
                            "equations": ["f_res = 1 / (2 * pi * sqrt(L * C))"],
                            "known_values": {"L": 0.1, "C": 50e-6},
                        },
                        "solution_steps": ["Use the LC resonance frequency relation."],
                    }
                ),
            }
        )
        parsed = {
            "question": "An LC circuit resonates with L = 0.1 H and C = 50 microF. Find the resonant frequency.",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "f_res", "unit": "Hz"},
            "givens": [
                {"symbol": "L", "si_value": 0.1},
                {"symbol": "C", "si_value": 50e-6},
            ],
            "relations": ["resonance"],
            "question_kind": "computational",
        }
        provider = LLMSolutionProvider(llm)
        output = provider._request_solution(provider._build_prompt(parsed), parsed)
        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution", "physics.solution.retry"])
        self.assertEqual(output["formula_ids"], ["ac.resonance.frequency"])

    def test_solution_provider_uses_deterministic_fallback_on_provider_exception(self) -> None:
        parsed = {
            "question": "An inductor has inductance 0.25 H. When current reaches maximum value 2sqrt2 A, what is maximum magnetic field energy?",
            "domain": "Inductance",
            "target": {"symbol": "W_max", "unit": "J"},
            "givens": [
                {"symbol": "L", "si_value": 0.25},
            ],
            "question_kind": "computational",
        }
        llm = RaisingLLM()
        provider = LLMSolutionProvider(llm)
        output = provider.get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": output, "errors": []})

        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution"])
        self.assertEqual(output["formula_ids"], ["inductance.magnetic_energy"])
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 1.0)

    def test_solution_provider_retries_compact_json_after_truncated_response(self) -> None:
        cases = [
            (
                {
                    "question": "A capacitor has C = 100 microF and U = 30 V. Calculate stored energy.",
                    "domain": "Capacitance",
                    "target": {"symbol": "W", "unit": "J"},
                    "givens": [{"symbol": "C", "si_value": 100e-6}, {"symbol": "U", "si_value": 30}],
                    "question_kind": "computational",
                },
                "W",
                "J",
                ["W = C * U**2 / 2"],
                {"C": 100e-6, "U": 30.0},
            ),
            (
                {
                    "question": "A capacitor has C = 80 microF and voltage V = 60 V. Find charge Q.",
                    "domain": "Capacitance",
                    "target": {"symbol": "Q", "unit": "C"},
                    "givens": [{"symbol": "C", "si_value": 80e-6}, {"symbol": "V", "si_value": 60}],
                    "question_kind": "computational",
                },
                "Q",
                "C",
                ["Q = C * U"],
                {"C": 80e-6, "U": 60.0},
            ),
            (
                {
                    "question": "Charge Q = 0.0048 C at voltage U = 60 V. Calculate capacitance.",
                    "domain": "Capacitance",
                    "target": {"symbol": "C", "unit": "F"},
                    "givens": [{"symbol": "Q", "si_value": 0.0048}, {"symbol": "U", "si_value": 60}],
                    "question_kind": "computational",
                },
                "C",
                "F",
                ["C = Q / U"],
                {"Q": 0.0048, "U": 60.0},
            ),
            (
                {
                    "question": "A capacitor stores charge Q = 0.012 C with capacitance C = 150 microF. Find V.",
                    "domain": "Capacitance",
                    "target": {"symbol": "V", "unit": "V"},
                    "givens": [{"symbol": "Q", "si_value": 0.012}, {"symbol": "C", "si_value": 150e-6}],
                    "question_kind": "computational",
                },
                "V",
                "V",
                ["V = Q / C"],
                {"Q": 0.012, "C": 150e-6},
            ),
        ]
        truncated_response = (
            '{\n  "mode": "computational",\n  "answer_type": "numeric",\n'
            '  "sympy_spec": {\n    "target_symbol": "W",\n'
            '    "target_unit": "J",\n    "equations": ["W = C * U**2 / 2"],\n'
            '    "known_values": {\n      "C": 0.0'
        )

        for parsed, target, unit, equations, known_values in cases:
            with self.subTest(target=target):
                llm = RecordingLLM(
                    {
                        "physics.solution": truncated_response,
                        "physics.solution.retry": json.dumps(
                            {
                                "mode": "computational",
                                "answer_type": "numeric",
                                "sympy_spec": {
                                    "target_symbol": target,
                                    "target_unit": unit,
                                    "equations": equations,
                                    "known_values": known_values,
                                },
                                "solution_steps": ["Use the selected capacitor relation."],
                            }
                        ),
                    }
                )
                provider = LLMSolutionProvider(llm)
                output = provider._request_solution(provider._build_prompt(parsed), parsed)
                spec = output["sympy_spec"]

                self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution", "physics.solution.retry"])
                self.assertFalse(provider.last_prompt_diagnostics["used_repair"])
                self.assertEqual(spec["target_symbol"], target)
                self.assertEqual(spec["target_unit"], unit)
                self.assertEqual(spec["equations"], equations)
                for symbol, value in known_values.items():
                    self.assertEqual(spec["known_values"][symbol], value)

    def test_solution_provider_repairs_when_retry_is_not_json(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": '{\n  "mode": "computational",\n  "sympy_spec": {',
                "physics.solution.retry": "still not json",
                "physics.solution.repair": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "Y",
                            "target_unit": "m",
                            "equations": ["Y = x"],
                            "known_values": {"x": 2},
                        },
                        "solution_steps": ["Use the parsed direct relation."],
                    }
                ),
            }
        )
        provider = LLMSolutionProvider(llm)
        parsed = {
            "question": "Find an unsupported numeric result.",
            "domain": "Unknown",
            "target": {"symbol": "Y", "unit": "m"},
            "givens": [{"symbol": "x", "si_value": 2}],
            "question_kind": "computational",
        }

        output = provider._request_solution(provider._build_prompt(parsed), parsed)
        self.assertEqual(output["sympy_spec"]["target_symbol"], "Y")
        self.assertEqual(
            [call["stage"] for call in llm.calls],
            ["physics.solution", "physics.solution.retry", "physics.solution.repair"],
        )

    def test_solution_prompt_template_is_compact(self) -> None:
        provider = LLMSolutionProvider(RecordingLLM({}))
        self.assertLessEqual(len(provider.prompt_template), 8000)

    def test_dynamic_solution_prompt_stays_under_size_caps(self) -> None:
        parsed = {
            "question": "Find the magnitude of the electric field at M on the perpendicular bisector.",
            "domain": "Electric Charges and Fields",
            "question_kind": "computational",
            "answer_format": {"requested_form": "magnitude"},
            "geometry": {"present": True, "type": "perpendicular_bisector"},
            "givens": [
                {"symbol": "q1", "si_value": 5e-7},
                {"symbol": "q2", "si_value": -5e-7},
                {"symbol": "d_AB", "si_value": 0.06},
                {"symbol": "ell", "si_value": 0.04},
            ],
            "target": {"symbol": "E_M", "unit": "N/C"},
        }
        provider = LLMSolutionProvider(RecordingLLM({}))
        prompt = provider._build_prompt(parsed)
        diagnostics = provider.last_prompt_diagnostics

        self.assertLessEqual(len(prompt), 12000)
        self.assertEqual(diagnostics["prompt_chars"], len(prompt))
        self.assertEqual(diagnostics["rag_chars"], 0)
        self.assertEqual(diagnostics["selected_rule_pack"], ["prompt"])
        self.assertIn("Domain guidance:", prompt)
        self.assertIn("Perpendicular bisector", prompt)
        self.assertNotIn("{{RULE_PACKS}}", prompt)
        self.assertNotIn("{{RAG_HINTS}}", prompt)

    def test_solution_provider_does_not_repair_valid_response(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "E",
                            "target_unit": "N/C",
                            "equations": ["E = Abs(k * q / r**2)"],
                            "known_values": {"k": 9e9, "q": 2e-6, "r": 0.3},
                        },
                        "solution_steps": ["Use Coulomb's law."],
                    }
                )
            }
        )
        provider = LLMSolutionProvider(llm)
        provider.get_solution("Find E.", {"question": "Find E.", "target": {"symbol": "E", "unit": "N/C"}})

        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution"])
        self.assertFalse(provider.last_prompt_diagnostics["used_repair"])

    def test_solution_provider_normalizes_supported_schema_aliases(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "formula",
                        "answer_type": "number",
                        "sympy_spec": {
                            "target_symbol": "E",
                            "target_unit": "J",
                            "equations": ["E = C * U**2 / 2"],
                            "known_values": {},
                        },
                        "solution_steps": ["Apply the capacitor energy relation."],
                    }
                )
            }
        )
        parsed = {
            "question": "Calculate stored energy.",
            "domain": "Capacitance",
            "target": {"symbol": "E", "unit": "J"},
            "givens": [
                {"symbol": "C", "si_value": 0.0001},
                {"symbol": "U", "si_value": 30},
            ],
            "question_kind": "computational",
        }
        provider = LLMSolutionProvider(llm)
        output = provider._request_solution(provider._build_prompt(parsed), parsed)

        self.assertEqual(output["mode"], "computational")
        self.assertEqual(output["answer_type"], "numeric")
        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution"])

    def test_solution_provider_merges_parsed_known_values_before_validation(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "I_rms",
                            "target_unit": "A",
                            "equations": ["R_total = R1 + R2", "I_rms = U / R_total"],
                            "known_values": {"U": 80},
                        },
                        "solution_steps": ["Use the equivalent resistance."],
                    }
                )
            }
        )
        parsed = {
            "question": "Circuit AB has R1 = 20 Ohm and R2 = 30 Ohm. An RMS voltage U = 80 V is applied. What is the RMS current?",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "I_rms", "unit": "A"},
            "givens": [
                {"symbol": "R1", "si_value": 20},
                {"symbol": "R2", "si_value": 30},
                {"symbol": "U", "si_value": 80},
            ],
            "question_kind": "computational",
        }
        output = LLMSolutionProvider(llm).get_solution(parsed["question"], parsed)
        self.assertEqual(output["sympy_spec"]["known_values"]["R2"], 30)

    def test_solution_provider_cleans_known_value_symbol_punctuation(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "Z",
                            "target_unit": "Ohm",
                            "equations": ["Z = R"],
                            "known_values": {"R.": 20},
                        },
                        "solution_steps": ["Use the resistance as impedance."],
                    }
                )
            }
        )
        parsed = {
            "question": "At resonance, impedance equals resistance R = 20 Ohm.",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "Z", "unit": "Ohm"},
            "givens": [{"symbol": "R", "si_value": 20}],
            "question_kind": "computational",
        }
        output = LLMSolutionProvider(llm).get_solution(parsed["question"], parsed)
        self.assertEqual(output["sympy_spec"]["known_values"], {"R": 20})

    def test_solution_provider_rewrites_expression_target_symbol(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "Abs(ZL)",
                            "target_unit": "Ohm",
                            "equations": ["ZL = omega * L"],
                            "known_values": {},
                        },
                        "solution_steps": ["Compute the magnitude of inductive reactance."],
                    }
                )
            }
        )
        parsed = {
            "question": "Calculate the magnitude of inductive reactance.",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "Z", "unit": "Ohm"},
            "givens": [{"symbol": "omega", "si_value": 100}, {"symbol": "L", "si_value": 0.2}],
            "question_kind": "computational",
        }
        output = LLMSolutionProvider(llm).get_solution(parsed["question"], parsed)
        spec = output["sympy_spec"]
        self.assertEqual(spec["target_symbol"], "Z")
        self.assertIn("Z = Abs(ZL)", spec["equations"])
        self.assertEqual(spec["known_values"]["omega"], 100)

    def test_solution_provider_retries_ac_formula_after_invalid_llm_spec(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "I_rms",
                            "target_unit": "A",
                            "equations": ["I_rms = U / (R2 + missing_helper)"],
                            "known_values": {"U": 80},
                        },
                        "solution_steps": [],
                    }
                ),
                "physics.solution.retry": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "formula_ids": ["ac.rms_current_resistive_equivalent"],
                        "sympy_spec": {
                            "target_symbol": "I_rms",
                            "target_unit": "A",
                            "equations": ["R_total = R1 + R2", "I_rms = U / R_total"],
                            "known_values": {"U": 80, "R1": 20, "R2": 30},
                        },
                        "solution_steps": ["Use the equivalent series resistance and RMS Ohm law."],
                    }
                ),
            }
        )
        parsed = {
            "question": "Circuit AB has R1 = 20 Ohm and R2 = 30 Ohm. An RMS voltage U = 80 V is applied. What is the RMS current?",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "I_rms", "unit": "A"},
            "givens": [
                {"symbol": "R1", "si_value": 20},
                {"symbol": "R2", "si_value": 30},
                {"symbol": "U", "si_value": 80},
            ],
            "question_kind": "computational",
        }
        output = LLMSolutionProvider(llm).get_solution(parsed["question"], parsed)
        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution", "physics.solution.retry"])
        self.assertEqual(output["formula_ids"], ["ac.rms_current_resistive_equivalent"])

    def test_equilateral_electric_field_vector_uses_signed_components(self) -> None:
        parsed = {
            "question": "Determine the net electric field vector at N for an equilateral triangle ABN.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "E_net_magnitude", "unit": "N/C"},
            "givens": [
                {"symbol": "q1", "si_value": 4e-10},
                {"symbol": "q2", "si_value": -4e-10},
                {"symbol": "a", "si_value": 0.02},
            ],
            "question_kind": "computational",
            "answer_format": {"requested_form": "vector"},
            "geometry": {"present": True, "type": "equilateral_triangle"},
        }
        solution = LLMSolutionProvider(
            solution_llm(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "formula_ids": ["electrostatics.point_charge_field_vector", "electrostatics.net_field_vector_cartesian"],
                    "sympy_spec": {
                        "target_symbol": "E_net_magnitude",
                        "target_unit": "N/C",
                        "equations": [
                            "Ax = 0",
                            "Ay = 0",
                            "Bx = a",
                            "By = 0",
                            "Nx = a / 2",
                            "Ny = a * sqrt(3) / 2",
                            "dx1 = Nx - Ax",
                            "dy1 = Ny - Ay",
                            "r1 = sqrt(dx1**2 + dy1**2)",
                            "dx2 = Nx - Bx",
                            "dy2 = Ny - By",
                            "r2 = sqrt(dx2**2 + dy2**2)",
                            "E1x = k * q1 * dx1 / r1**3",
                            "E1y = k * q1 * dy1 / r1**3",
                            "E2x = k * q2 * dx2 / r2**3",
                            "E2y = k * q2 * dy2 / r2**3",
                            "Ex_net = E1x + E2x",
                            "Ey_net = E1y + E2y",
                            "E_net_magnitude = sqrt(Ex_net**2 + Ey_net**2)",
                        ],
                        "known_values": {"q1": 4e-10, "q2": -4e-10, "a": 0.02, "k": 9e9},
                    },
                    "solution_steps": ["Use signed vector components for both charges."],
                    "vector_spec": {
                        "component_symbols": ["Ex_net", "Ey_net"],
                        "magnitude_symbol": "E_net_magnitude",
                        "direction": "parallel to AB, from A to B when Ex_net is positive and Ey_net is zero",
                    },
                }
            )
        ).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        vector = computed["verified_output"]["vector_result"]

        self.assertIn("electrostatics.point_charge_field_vector", solution["formula_ids"])
        self.assertIn("E1x = k * q1 * dx1 / r1**3", solution["sympy_spec"]["equations"])
        self.assertAlmostEqual(vector["components"][0], 9000, places=6)
        self.assertAlmostEqual(vector["components"][1], 0, places=6)
        self.assertAlmostEqual(vector["magnitude"], 9000, places=6)
        self.assertEqual(computed["result"]["answer"], "9000 N/C")
        self.assertEqual(vector["direction"], "parallel to AB, from A to B")

    def test_midpoint_opposite_charges_add_instead_of_canceling(self) -> None:
        parsed = {
            "question": "Find the electric field magnitude at the midpoint between opposite charges.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "E_net_magnitude", "unit": "N/C"},
            "givens": [
                {"symbol": "q1", "si_value": 4e-10},
                {"symbol": "q2", "si_value": -4e-10},
                {"symbol": "AB", "si_value": 0.02},
            ],
            "question_kind": "computational",
            "geometry": {"present": True, "type": "midpoint_1d"},
        }
        solution = LLMSolutionProvider(
            solution_llm(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "formula_ids": ["electrostatics.point_charge_field_vector", "electrostatics.net_field_vector_cartesian"],
                    "sympy_spec": {
                        "target_symbol": "E_net_magnitude",
                        "target_unit": "N/C",
                        "equations": [
                            "Ax = 0",
                            "Bx = AB",
                            "Mx = AB / 2",
                            "dx1 = Mx - Ax",
                            "dx2 = Mx - Bx",
                            "E1x = k * q1 * dx1 / Abs(dx1)**3",
                            "E2x = k * q2 * dx2 / Abs(dx2)**3",
                            "E_net_x = E1x + E2x",
                            "E_net_magnitude = Abs(E_net_x)",
                        ],
                        "known_values": {"q1": 4e-10, "q2": -4e-10, "AB": 0.02, "k": 9e9},
                    },
                    "solution_steps": ["Use signed 1D field components at the midpoint."],
                    "vector_spec": {
                        "component_symbols": ["E_net_x"],
                        "magnitude_symbol": "E_net_magnitude",
                        "direction": "along AB according to the sign of E_net_x",
                    },
                }
            )
        ).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        vector = computed["verified_output"]["vector_result"]

        self.assertAlmostEqual(vector["components"][0], 72000, places=6)
        self.assertAlmostEqual(vector["magnitude"], 72000, places=6)
        self.assertNotEqual(computed["result"]["answer"], "0")

    def test_series_rlc_impedance_formula_coverage(self) -> None:
        parsed = {
            "question": "An RLC circuit has R = 20 Ohm, L = 0.5 H, C = 100 microF, f = 50 Hz. Calculate total impedance Z.",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "Z", "unit": "Ohm"},
            "givens": [
                {"symbol": "R", "si_value": 20},
                {"symbol": "L", "si_value": 0.5},
                {"symbol": "C", "si_value": 100e-6},
                {"symbol": "f", "si_value": 50},
            ],
            "relations": ["RLC circuit"],
            "question_kind": "computational",
        }
        solution = LLMSolutionProvider(
            solution_llm(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "formula_ids": ["rlc.series.impedance"],
                    "sympy_spec": {
                        "target_symbol": "Z",
                        "target_unit": "Ohm",
                        "equations": [
                            "X_L = 2 * pi * f * L",
                            "X_C = 1 / (2 * pi * f * C)",
                            "Z = sqrt(R**2 + (X_L - X_C)**2)",
                        ],
                        "known_values": {"R": 20, "L": 0.5, "C": 100e-6, "f": 50},
                    },
                    "solution_steps": ["Use the series RLC impedance formula."],
                    "assumptions": {"circuit_type": "series_assumed"},
                }
            )
        ).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertEqual(solution["assumptions"]["circuit_type"], "series_assumed")
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 126.84, places=2)

    def test_resonant_power_from_voltage_and_resistance(self) -> None:
        parsed = {
            "question": "The RMS voltage is 180 V, and the resistance R = 90 Ohm. What is the power of the circuit at resonance?",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "P", "unit": "W"},
            "givens": [
                {"symbol": "U", "si_value": 180},
                {"symbol": "R", "si_value": 90},
            ],
            "relations": ["circuit is at resonance", "U is RMS voltage"],
            "question_kind": "computational",
            "answer_format": {"requested_form": "numeric"},
        }
        solution = LLMSolutionProvider(
            solution_llm(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "formula_ids": ["ac.resonance.power_from_voltage_resistance"],
                    "sympy_spec": {
                        "target_symbol": "P",
                        "target_unit": "W",
                        "equations": ["P = U**2 / R"],
                        "known_values": {"U": 180, "R": 90},
                    },
                    "solution_steps": ["At resonance the impedance is purely resistive."],
                }
            )
        ).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertEqual(solution["formula_ids"], ["ac.resonance.power_from_voltage_resistance"])
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 360)

    def test_phase_balanced_two_section_circuit_power_prompt_guidance(self) -> None:
        parsed = {
            "question": (
                "Circuit AB consists of segment AM, which has resistor R1 = 10 Ohm in series with capacitor C, "
                "and segment MB, which has resistor R2 = 90 Ohm in series with inductor L. "
                "Given that LComega² = 1 and uAM is 90° out of phase with uMB. "
                "An RMS voltage U = 100 V is applied across AB. What is the total power consumed by circuit AB?"
            ),
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "P", "unit": "W"},
            "givens": [
                {"symbol": "R1", "si_value": 10},
                {"symbol": "R2", "si_value": 90},
                {"symbol": "U", "si_value": 100},
            ],
            "relations": ["LC*omega**2 = 1", "uAM is in quadrature with uMB", "U is RMS voltage across AB"],
            "question_kind": "computational",
        }
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "P",
                            "target_unit": "W",
                            "equations": ["R_total = R1 + R2", "P = U**2 / R_total"],
                            "known_values": {"U": 100, "R1": 10, "R2": 90},
                        },
                        "solution_steps": ["Use the phase-balance rule for the two-section circuit."],
                    }
                )
            }
        )
        provider = LLMSolutionProvider(llm)
        solution = provider.get_solution(parsed["question"], parsed)
        equations = solution["sympy_spec"]["equations"]
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution"])
        self.assertEqual(solution["formula_ids"], ["ac.two_section_phase_balanced_power"])
        self.assertEqual(equations, ["R_total = R1 + R2", "P = U**2 / R_total"])
        self.assertNotIn("j", " ".join(equations))
        self.assertNotIn("Z_total", " ".join(equations))
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 100)

    def test_resonance_inductance_from_frequency_and_capacitance(self) -> None:
        parsed = {
            "question": "An electrical circuit needs to resonate at f=50 Hz. The capacitor C=200 microF. What inductor L should be chosen?",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "L", "unit": "H"},
            "givens": [
                {"symbol": "f", "si_value": 50},
                {"symbol": "C", "si_value": 200e-6},
            ],
            "relations": ["circuit needs to resonate"],
            "question_kind": "computational",
        }
        solution = LLMSolutionProvider(
            solution_llm(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "formula_ids": ["ac.resonance.inductance_from_frequency_capacitance"],
                    "sympy_spec": {
                        "target_symbol": "L",
                        "target_unit": "H",
                        "equations": ["L = 1 / (4 * pi**2 * f**2 * C)"],
                        "known_values": {"f": 50, "C": 200e-6},
                    },
                    "solution_steps": ["Rearrange the resonance frequency relation for L."],
                }
            )
        ).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertEqual(solution["formula_ids"], ["ac.resonance.inductance_from_frequency_capacitance"])
        self.assertAlmostEqual(
            computed["verified_output"]["final_answer"]["value"],
            1 / (4 * math.pi**2 * 50**2 * 200e-6),
        )

    def test_frequency_aliases_reject_extra_known_values(self) -> None:
        parsed = {
            "question": "Compute the angular resonance frequency.",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "omega", "unit": "rad/s"},
            "givens": [
                {"symbol": "L", "si_value": 0.5},
                {"symbol": "C", "si_value": 2e-6},
                {"symbol": "f_res", "si_value": 50},
            ],
            "question_kind": "computational",
        }
        solution = {
            "mode": "computational",
            "answer_type": "numeric",
            "sympy_spec": {
                "target_symbol": "omega",
                "target_unit": "rad/s",
                "equations": ["omega = 1 / sqrt(L * C)"],
                "known_values": {"L": 0.5, "C": 2e-6, "f": 50},
            },
            "solution_steps": [],
        }
        with self.assertRaisesRegex(WorkflowExecutionError, "untrusted numeric value f"):
            PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

    def test_frequency_scaled_resistor_voltage_at_new_resonance_does_not_need_r(self) -> None:
        parsed = {
            "question": "Given a series circuit with XL = 40 Ohm, XC = 160 Ohm, and U = 100 V. If the frequency is doubled, what is the RMS voltage across R?",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "U_R", "unit": "V"},
            "givens": [
                {"symbol": "XL", "si_value": 40},
                {"symbol": "XC", "si_value": 160},
                {"symbol": "U", "si_value": 100},
            ],
            "relations": ["series circuit", "frequency is doubled"],
            "question_kind": "computational",
        }
        solution = LLMSolutionProvider(
            solution_llm(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "formula_ids": ["ac.resonance.resistor_voltage_equals_source_voltage"],
                    "sympy_spec": {
                        "target_symbol": "U_R",
                        "target_unit": "V",
                        "equations": [
                            "frequency_ratio = 2",
                            "XL_new = frequency_ratio * XL",
                            "XC_new = XC / frequency_ratio",
                            "U_R = U",
                        ],
                        "known_values": {"XL": 40, "XC": 160, "U": 100},
                    },
                    "solution_steps": ["After doubling frequency, the reactances match and the resistor gets the source voltage."],
                    "assumptions": {"circuit_type": "series_assumed"},
                }
            )
        ).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertEqual(solution["formula_ids"], ["ac.resonance.resistor_voltage_equals_source_voltage"])
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 100)

    def test_resonance_frequency_shift_derives_inductive_reactance_from_duplicate_givens(self) -> None:
        parsed = {
            "question": "Given an RLC series circuit with a resistance R=30Ohm. At its resonant frequency f=50Hz, the current flowing through the circuit is I=2A. If the frequency is then changed to f=100Hz, the current becomes I=1.6A. Calculate the inductive reactance (ZL).",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "ZL", "unit": "Ohm"},
            "givens": [
                {"symbol": "R", "si_value": 30},
                {"symbol": "f", "si_value": 50},
                {"symbol": "I", "si_value": 2},
                {"symbol": "f", "si_value": 100},
                {"symbol": "I", "si_value": 1.6},
            ],
            "relations": ["RLC series circuit", "at resonant frequency", "frequency changed"],
            "question_kind": "computational",
        }
        solution = LLMSolutionProvider(
            solution_llm(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "formula_ids": ["ac.resonance_shift.inductive_reactance"],
                    "sympy_spec": {
                        "target_symbol": "ZL",
                        "target_unit": "Ohm",
                        "equations": [
                            "frequency_ratio = f_2 / f",
                            "U = I * R",
                            "Z_new = U / I_2",
                            "X_net_new = sqrt(Z_new**2 - R**2)",
                            "X_res = X_net_new / Abs(frequency_ratio - 1 / frequency_ratio)",
                            "ZL = frequency_ratio * X_res",
                        ],
                        "known_values": {"R": 30, "f": 50, "f_2": 100, "I": 2, "I_2": 1.6},
                    },
                    "solution_steps": ["Use resonance current to get source voltage, then changed-frequency impedance."],
                    "assumptions": {"circuit_type": "series_assumed"},
                }
            )
        ).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertEqual(solution["formula_ids"], ["ac.resonance_shift.inductive_reactance"])
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 30)

    def test_solenoid_magnetic_field_formula_coverage(self) -> None:
        parsed = {
            "question": "A solenoid is 1 m long, has 2000 turns, and current 3 A. Calculate magnetic field inside.",
            "domain": "Sources of Magnetic Fields",
            "target": {"symbol": "B", "unit": "T"},
            "givens": [
                {"symbol": "ell", "si_value": 1},
                {"symbol": "N", "si_value": 2000},
                {"symbol": "I", "si_value": 3},
            ],
            "question_kind": "computational",
        }
        solution = LLMSolutionProvider(
            solution_llm(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "formula_ids": ["magnetism.solenoid_magnetic_field"],
                    "sympy_spec": {
                        "target_symbol": "B",
                        "target_unit": "T",
                        "equations": ["B = mu_0 * N * I / ell"],
                        "known_values": {"ell": 1, "N": 2000, "I": 3},
                    },
                    "solution_steps": ["Use the long-solenoid magnetic-field formula."],
                }
            )
        ).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 0.00754, places=5)

    def test_self_inductance_accepts_verified_derived_current_rate(self) -> None:
        parsed = {
            "question": "Given induced electromotive force 0.3 V, current decreases uniformly from 2 A to 0 A in 0.05 s. Calculate self-inductance.",
            "domain": "Inductance",
            "target": {"symbol": "L_self", "unit": "H"},
            "givens": [
                {"symbol": "epsilon", "si_value": 0.3},
                {"symbol": "I_initial", "si_value": 2},
                {"symbol": "I_final", "si_value": 0},
                {"symbol": "delta_t", "si_value": 0.05},
            ],
            "question_kind": "computational",
        }
        solution = LLMSolutionProvider(
            solution_llm(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "formula_ids": ["inductance.self_inductance_from_emf_current_change"],
                    "sympy_spec": {
                        "target_symbol": "L_self",
                        "target_unit": "H",
                        "equations": [
                            "delta_I = I_final - I_initial",
                            "L_self = Abs(epsilon) * delta_t / Abs(delta_I)",
                        ],
                        "known_values": {"epsilon": 0.3, "I_initial": 2, "I_final": 0, "delta_t": 0.05},
                    },
                    "solution_steps": ["Use the magnitude form of the self-inductance EMF relation."],
                }
            )
        ).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 0.0075)
        self.assertTrue(computed["verified_output"]["verification_result"]["derived_values_allowed"])

    def test_self_inductance_known_derived_value_is_verified_by_equation(self) -> None:
        parsed = {
            "question": "Calculate self-inductance.",
            "domain": "Inductance",
            "target": {"symbol": "L_self", "unit": "H"},
            "givens": [
                {"symbol": "epsilon", "si_value": 0.3},
                {"symbol": "I_initial", "si_value": 2},
                {"symbol": "I_final", "si_value": 0},
                {"symbol": "delta_t", "si_value": 0.05},
            ],
        }
        solution = {
            "mode": "computational",
            "answer_type": "numeric",
            "sympy_spec": {
                "target_symbol": "L_self",
                "target_unit": "H",
                "equations": ["delta_I = I_final - I_initial", "rate_I = Abs(delta_I) / delta_t", "L_self = Abs(epsilon) / rate_I"],
                "known_values": {"epsilon": 0.3, "I_initial": 2, "I_final": 0, "delta_t": 0.05, "rate_I": 40},
            },
            "solution_steps": [],
        }
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 0.0075)

    def test_zero_field_point_uses_square_root_relation_not_cube_root(self) -> None:
        parsed = {
            "question": "Find point M between same-sign charges where the electric field is zero.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "BM", "unit": "m"},
            "givens": [
                {"symbol": "q1", "si_value": 4e-6},
                {"symbol": "q2", "si_value": 9e-6},
                {"symbol": "AB", "si_value": 0.1},
            ],
            "question_kind": "computational",
            "geometry": {"present": True, "type": "collinear"},
        }
        solution = LLMSolutionProvider(
            solution_llm(
                {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "formula_ids": ["electrostatics.zero_field_point_two_charges_1d"],
                    "sympy_spec": {
                        "target_symbol": "BM",
                        "target_unit": "m",
                        "equations": ["BM = AB * sqrt(Abs(q2)) / (sqrt(Abs(q1)) + sqrt(Abs(q2)))"],
                        "known_values": {"q1": 4e-6, "q2": 9e-6, "AB": 0.1},
                    },
                    "solution_steps": ["Use the square-root distance ratio for equal field magnitudes."],
                }
            )
        ).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        equation_text = " ".join(solution["sympy_spec"]["equations"])
        self.assertIn("sqrt(Abs(q2))", equation_text)
        self.assertNotIn("**(1/3)", equation_text)
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 0.06)

    def test_end_to_end_numeric_contract_includes_non_empty_evidence(self) -> None:
        llm = RecordingLLM(
            {
                "physics.parsing": json.dumps(
                    {
                        "question": "Calculate stored energy.",
                        "domain": "Capacitance",
                        "target": {"symbol": "E", "unit": "J"},
                        "givens": [
                            {"symbol": "C", "si_value": 0.0001, "si_unit": "F", "uncertainty": None},
                            {"symbol": "U", "si_value": 30, "si_unit": "V", "uncertainty": None},
                        ],
                        "relations": [],
                        "question_kind": "computational",
                    }
                ),
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "E",
                            "target_unit": "J",
                            "equations": ["E = C * U**2 / 2"],
                            "known_values": {"C": 0.0001, "U": 30},
                        },
                        "solution_steps": ["Apply stored capacitor energy."],
                    }
                ),
                "physics.explanation": json.dumps(
                    {
                        "answer": {"symbol": "E", "value": 0.045, "unit": "J"},
                        "explanation": "Using the verified capacitor-energy equation gives E = 0.045 J.",
                    }
                ),
            }
        )
        graph = ExactGraph(llm=llm, classifier=StaticClassifier("physics"))
        output = graph.predict({"question": "Calculate stored energy."})
        self.assertEqual(set(output), {"answer", "explanation", "cot", "premises"})
        self.assertEqual(output["answer"], "0.045 J")
        self.assertIsInstance(output["cot"], list)
        self.assertGreaterEqual(len(output["cot"]), 3)
        self.assertEqual(output["premises"], ["E = C * U**2 / 2"])

    def test_explain_agent_builds_reasoning_list_from_solution_and_sympy_result(self) -> None:
        agent = ExplainAgent()
        parsed = {
            "question": "Calculate stored energy.",
            "target": {"symbol": "E", "unit": "J"},
        }
        solution_output = {
            "mode": "computational",
            "answer_type": "numeric",
            "sympy_spec": {
                "target_symbol": "E",
                "target_unit": "J",
                "equations": ["E = C * U**2 / 2"],
                "known_values": {"C": 0.0001, "U": 30},
            },
            "solution_steps": [],
        }
        verified = agent._build_verified_output(parsed, solution_output)
        cot = agent.build_cot(parsed, solution_output, verified)

        self.assertIsInstance(cot, list)
        self.assertGreaterEqual(len(cot), 3)
        self.assertTrue(any("E = C * U**2 / 2" in step for step in cot))
        self.assertTrue(any(step.startswith("Therefore, E = 0.045") for step in cot))

    def test_geometry_and_constants_feed_computation(self) -> None:
        output = PhysicsWorkflow().compute_sympy(
            {
                "parsed_question": {
                    "givens": [{"symbol": "q1", "si_value": -2e-6}],
                    "geometry": {"derived_distances": [{"symbol": "AN", "si_value": 0.3}]},
                    "target": {"symbol": "E_N", "unit": "N/C"},
                },
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "E_N",
                        "target_unit": "N/C",
                        "equations": ["E_N = Abs(k * q1 / AN**2)"],
                        "known_values": {"q1": -2e-6, "AN": 0.3, "k": 9e9},
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        self.assertEqual(output["result"]["answer"], "200000")

    def test_perpendicular_bisector_geometry_computes_with_defined_helpers(self) -> None:
        output = PhysicsWorkflow().compute_sympy(
            {
                "parsed_question": {
                    "question": "What is the magnitude of the electric field intensity at M?",
                    "givens": [
                        {"symbol": "q1", "si_value": 5e-7},
                        {"symbol": "q2", "si_value": -5e-7},
                        {"symbol": "d_AB", "si_value": 0.06},
                        {"symbol": "ell", "si_value": 0.04},
                    ],
                    "target": {"symbol": "E_M", "unit": "N/C"},
                    "answer_format": {"requested_form": "magnitude"},
                },
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "E_M",
                        "target_unit": "N/C",
                        "equations": [
                            "xA = -d_AB / 2",
                            "yA = 0",
                            "xB = d_AB / 2",
                            "yB = 0",
                            "xM = 0",
                            "yM = ell",
                            "AM = sqrt((xM - xA)**2 + (yM - yA)**2)",
                            "BM = sqrt((xM - xB)**2 + (yM - yB)**2)",
                            "E1x = k * q1 * (xM - xA) / AM**3",
                            "E1y = k * q1 * (yM - yA) / AM**3",
                            "E2x = k * q2 * (xM - xB) / BM**3",
                            "E2y = k * q2 * (yM - yB) / BM**3",
                            "E_Mx = E1x + E2x",
                            "E_My = E1y + E2y",
                            "E_M = sqrt(E_Mx**2 + E_My**2)",
                        ],
                        "known_values": {"q1": 5e-7, "q2": -5e-7, "d_AB": 0.06, "ell": 0.04, "k": 9e9},
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        self.assertEqual(output["result"]["answer"], "2160000")

    def test_computational_yes_no_does_not_append_unit(self) -> None:
        workflow = PhysicsWorkflow()
        computed = workflow.compute_sympy(
            {
                "parsed_question": {
                    "givens": [
                        {"symbol": "L", "si_value": 0.1},
                        {"symbol": "C", "si_value": 0.00005},
                        {"symbol": "f", "si_value": 71},
                    ],
                    "target": {"symbol": "f_res", "unit": "Hz"},
                },
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "yes_no",
                    "sympy_spec": {
                        "target_symbol": "f_res",
                        "target_unit": "Hz",
                        "equations": ["f_res = 1 / (2 * pi * sqrt(L * C))"],
                        "known_values": {"L": 0.1, "C": 0.00005, "f": 71},
                    },
                    "decision_spec": {
                        "expected_symbol": "f",
                        "answer_if_true": "Yes",
                        "answer_if_false": "No",
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        formatted = FormatterNode()({"result": computed["result"]})["output"]
        self.assertEqual(formatted["answer"], "Yes")
        self.assertNotIn("fol", formatted)
        self.assertIsInstance(formatted["cot"], list)
        self.assertGreaterEqual(len(formatted["cot"]), 3)
        self.assertTrue(any(step.startswith("Step 1") for step in formatted["cot"]))
        self.assertIn("premises", formatted)

    def test_ac_resonance_yes_no_computes_no_without_frequency_alias(self) -> None:
        parsed = {
            "question": "An AC circuit consists of R=10 Ohm, L=0.05 H, C=100 microF. When the frequency is 225 Hz, does resonance occur?",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "f_res", "unit": "Hz"},
            "givens": [
                {"symbol": "R", "si_value": 10, "si_unit": "Ohm"},
                {"symbol": "L", "si_value": 0.05, "si_unit": "H"},
                {"symbol": "C", "si_value": 100e-6, "si_unit": "F"},
                {"symbol": "f", "si_value": 225, "si_unit": "Hz"},
            ],
            "relations": ["compare resonance with f = 225 Hz"],
            "question_kind": "yes_no_computational",
            "comparison": {
                "present": True,
                "computed_quantity_symbol": "f_res",
                "given_quantity_symbol": "f",
                "given_si_value": 225,
                "given_si_unit": "Hz",
            },
        }
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "yes_no",
                        "formula_ids": ["ac.resonance.frequency"],
                        "sympy_spec": {
                            "target_symbol": "f_res",
                            "target_unit": "Hz",
                            "equations": ["f_res = 1 / (2 * pi * sqrt(L * C))"],
                            "known_values": {"R": 10, "L": 0.05, "C": 100e-6, "f": 225},
                        },
                        "decision_spec": {
                            "computed_symbol": "f_res",
                            "expected_symbol": "f",
                            "operator": "approximately_equal",
                            "tolerance_policy": "significant_figures",
                            "answer_if_true": "Yes",
                            "answer_if_false": "No",
                        },
                        "solution_steps": ["Compute the LC resonance frequency and compare to the given frequency."],
                    }
                )
            }
        )
        solution = LLMSolutionProvider(llm).get_solution(parsed["question"], parsed)
        known_values = solution["sympy_spec"]["known_values"]
        self.assertIn("f", known_values)
        self.assertNotIn("f_res", known_values)
        self.assertNotIn("f0", known_values)

        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertEqual(computed["result"]["answer"], "No")
        decision = computed["verified_output"]["decision_result"]
        self.assertAlmostEqual(decision["computed_value"], 71.176254, places=5)
        self.assertEqual(decision["expected_value"], 225)

    def test_ac_resonance_yes_no_uses_f_symbol_for_literal_frequency(self) -> None:
        parsed = {
            "question": "A series RLC circuit has R=75 Ohm, L=0.2 H, C=40 microF. Is 56.3 Hz the resonant frequency?",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "f_res", "unit": "Hz"},
            "givens": [
                {"symbol": "R", "si_value": 75, "si_unit": "Ohm"},
                {"symbol": "L", "si_value": 0.2, "si_unit": "H"},
                {"symbol": "C", "si_value": 40e-6, "si_unit": "F"},
                {"symbol": "f", "si_value": 56.3, "si_unit": "Hz"},
            ],
            "relations": ["series RLC circuit", "compare resonant frequency with 56.3 Hz"],
            "question_kind": "yes_no_computational",
            "comparison": {
                "present": True,
                "computed_quantity_symbol": "f_res",
                "given_quantity_symbol": "f",
                "given_si_value": 56.3,
                "given_si_unit": "Hz",
            },
        }
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "yes_no",
                        "formula_ids": ["ac.resonance.frequency"],
                        "sympy_spec": {
                            "target_symbol": "f_res",
                            "target_unit": "Hz",
                            "equations": ["f_res = 1 / (2 * pi * sqrt(L * C))"],
                            "known_values": {"R": 75, "L": 0.2, "C": 40e-6, "f": 56.3},
                        },
                        "decision_spec": {
                            "computed_symbol": "f_res",
                            "expected_symbol": "f",
                            "operator": "approximately_equal",
                            "tolerance_policy": "significant_figures",
                            "answer_if_true": "Yes",
                            "answer_if_false": "No",
                        },
                        "solution_steps": ["Compute the LC resonance frequency and compare to the given literal frequency."],
                    }
                )
            }
        )
        solution = LLMSolutionProvider(llm).get_solution(parsed["question"], parsed)
        known_values = solution["sympy_spec"]["known_values"]

        self.assertEqual(solution["decision_spec"]["expected_symbol"], "f")
        self.assertIn("f", known_values)
        self.assertNotIn("56_3", known_values)

    def test_ac_resonance_literal_frequency_comparison_is_normalized_to_f(self) -> None:
        parsed = {
            "question": "A series RLC circuit has R=75 Ohm, L=0.2 H, C=40 microF. Is 56.3 Hz the resonant frequency?",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "f_res", "unit": "Hz"},
            "givens": [
                {"symbol": "R", "si_value": 75, "si_unit": "Ohm"},
                {"symbol": "L", "si_value": 0.2, "si_unit": "H"},
                {"symbol": "C", "si_value": 40e-6, "si_unit": "F"},
            ],
            "relations": ["series RLC circuit", "compare resonant frequency with 56.3 Hz"],
            "question_kind": "yes_no_computational",
            "comparison": {
                "present": True,
                "computed_quantity_symbol": "f_res",
                "given_quantity_symbol": "56.3",
                "given_si_value": 56.3,
                "given_si_unit": "Hz",
            },
        }
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
        known_values = solution["sympy_spec"]["known_values"]
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertEqual(solution["decision_spec"]["expected_symbol"], "f")
        self.assertIn("f", known_values)
        self.assertNotIn("56_3", known_values)
        self.assertEqual(computed["result"]["answer"], "Yes")

    def test_formula_lib_accepts_parser_omega_l_symbol_for_reactance(self) -> None:
        parsed = {
            "question": "Calculate inductive reactance for L = 0.2 H and omega_L = 100 rad/s.",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "ZL", "unit": "Ohm"},
            "givens": [
                {"symbol": "omega_L", "si_value": 100, "si_unit": "rad/s"},
                {"symbol": "L", "si_value": 0.2, "si_unit": "H"},
            ],
            "question_kind": "computational",
        }
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertEqual(solution["sympy_spec"]["equations"], ["ZL = omega_L * L"])
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 20)

    def test_formula_only_lc_angular_frequency_uses_direct_mode(self) -> None:
        parsed = {
            "question": "What is the resonant angular frequency of an LC circuit?",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "omega", "unit": "rad/s"},
            "givens": [],
            "question_kind": "computational",
        }
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertEqual(solution["mode"], "direct")
        self.assertIn("omega_0 = 1/sqrt(L*C)", computed["result"]["answer"])

    def test_faraday_computation_includes_expected_cot_steps(self) -> None:
        output = PhysicsWorkflow().compute_sympy(
            {
                "parsed_question": {
                    "givens": [
                        {"symbol": "N", "si_value": 10},
                        {"symbol": "phi_initial", "si_value": 0.1},
                        {"symbol": "phi_final", "si_value": 0.4},
                        {"symbol": "t", "si_value": 2},
                    ],
                    "target": {"symbol": "E_ind", "unit": "V"},
                },
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "E_ind",
                        "target_unit": "V",
                        "equations": ["E_ind = -N * (phi_final - phi_initial) / t"],
                        "known_values": {"N": 10, "phi_initial": 0.1, "phi_final": 0.4, "t": 2},
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        self.assertEqual(
            output["result"]["cot"][:2],
            [
                "Step1: Use Faraday's law of electromagnetic induction to relate the induced electromotive force (EMF) to the rate of change of magnetic flux.",
                "Step 2:The induced EMF is given by the equation E_ind = -N * (phi_final - phi_initial) / t, where N is the number of turns, phi_final is the final magnetic flux, phi_initial is the initial magnetic flux, and t is the time interval.",
            ],
        )

    def test_magnitude_emf_public_answer_uses_absolute_value(self) -> None:
        computed = PhysicsWorkflow().compute_sympy(
            {
                "parsed_question": {
                    "question": "Calculate the induced electromotive force.",
                    "givens": [
                        {"symbol": "N", "si_value": 2000},
                        {"symbol": "phi_per_turn", "si_value": 2e-6},
                        {"symbol": "t", "si_value": 0.01},
                    ],
                    "target": {"symbol": "E_ind", "unit": "V"},
                    "answer_format": {"requested_form": "magnitude"},
                },
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "E_ind",
                        "target_unit": "V",
                        "equations": ["E_ind = -N * phi_per_turn / t"],
                        "known_values": {"N": 2000, "phi_per_turn": 2e-6, "t": 0.01},
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        self.assertEqual(computed["result"]["answer"], "0.4")
        self.assertEqual(computed["verified_output"]["final_answer"]["value"], 0.4)
        self.assertEqual(computed["verified_output"]["sympy_result"]["value"], -0.4)

    def test_signed_emf_preserves_negative_value(self) -> None:
        computed = PhysicsWorkflow().compute_sympy(
            {
                "parsed_question": {
                    "question": "Calculate the signed induced electromotive force direction.",
                    "givens": [
                        {"symbol": "N", "si_value": 2000},
                        {"symbol": "phi_per_turn", "si_value": 2e-6},
                        {"symbol": "t", "si_value": 0.01},
                    ],
                    "target": {"symbol": "E_ind", "unit": "V"},
                    "answer_format": {"requested_form": "signed"},
                },
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "E_ind",
                        "target_unit": "V",
                        "equations": ["E_ind = -N * phi_per_turn / t"],
                        "known_values": {"N": 2000, "phi_per_turn": 2e-6, "t": 0.01},
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        self.assertEqual(computed["result"]["answer"], "-0.4")

    def test_direct_conceptual_solution_does_not_require_sympy(self) -> None:
        workflow = PhysicsWorkflow()
        computed = workflow.compute_sympy(
            {
                "solution_output": {
                    "mode": "direct",
                    "answer_type": "conceptual",
                    "direct_answer": {
                        "answer": "The net field is zero by symmetry.",
                        "rationale_steps": ["Equal same-sign charges cancel at the midpoint."],
                    },
                },
                "errors": [],
            }
        )
        self.assertEqual(computed["result"]["answer"], "The net field is zero by symmetry.")
        self.assertEqual(computed["verified_output"]["mode"], "direct")

    def test_direct_yes_no_solution_does_not_require_sympy(self) -> None:
        workflow = PhysicsWorkflow()
        computed = workflow.compute_sympy(
            {
                "solution_output": {
                    "mode": "direct",
                    "answer_type": "yes_no",
                    "direct_answer": {
                        "answer": "Yes",
                        "rationale_steps": ["The statement follows from the stated definition."],
                    },
                },
                "errors": [],
            }
        )
        self.assertEqual(computed["result"]["answer"], "Yes")
        self.assertFalse(computed["result"]["append_unit"])

    def test_direct_multiple_choice_returns_concrete_answer_without_options(self) -> None:
        computed = PhysicsWorkflow().compute_sympy(
            {
                "parsed_question": {"question_kind": "multiple_choice"},
                "solution_output": {
                    "mode": "direct",
                    "answer_type": "multiple_choice",
                    "direct_answer": {
                        "answer": "Electromagnets",
                        "selected_option": "A",
                        "rationale_steps": ["Solenoids are used to create electromagnets."],
                    },
                },
                "errors": [],
            }
        )
        self.assertEqual(computed["result"]["answer"], "Electromagnets")

    def test_direct_multiple_choice_includes_option_text_when_available(self) -> None:
        computed = PhysicsWorkflow().compute_sympy(
            {
                "parsed_question": {
                    "question_kind": "multiple_choice",
                    "options": [{"label": "A", "text": "Electromagnets"}],
                },
                "solution_output": {
                    "mode": "direct",
                    "answer_type": "multiple_choice",
                    "direct_answer": {
                        "answer": "Electromagnets",
                        "selected_option": "A",
                        "rationale_steps": ["Solenoids are used to create electromagnets."],
                    },
                },
                "errors": [],
            }
        )
        self.assertEqual(computed["result"]["answer"], "A. Electromagnets")

    def test_nonnumeric_physical_constant_known_value_uses_builtin_constant(self) -> None:
        computed = PhysicsWorkflow().compute_sympy(
            {
                "parsed_question": {
                    "givens": [
                        {"symbol": "N", "si_value": 1500},
                        {"symbol": "I", "si_value": 2},
                        {"symbol": "ell", "si_value": 1},
                    ],
                    "target": {"symbol": "B", "unit": "T"},
                },
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "B",
                        "target_unit": "T",
                        "equations": ["B = mu_0 * N * I / ell"],
                        "known_values": {"mu_0": "4*pi*10**-7", "N": 1500, "I": 2, "ell": 1},
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 1.25663706212e-6 * 1500 * 2)

    def test_invalid_solution_specification_raises_workflow_error(self) -> None:
        with self.assertRaisesRegex(WorkflowExecutionError, "No valid physics solution specification"):
            PhysicsWorkflow().compute_sympy({"solution_output": {}, "errors": []})

    def test_unresolved_sympy_target_raises_workflow_error(self) -> None:
        with self.assertRaisesRegex(WorkflowExecutionError, "equations could not resolve the target"):
            PhysicsWorkflow().compute_sympy(
                {
                    "parsed_question": {"target": {"symbol": "x", "unit": "m"}},
                    "solution_output": {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "x",
                            "target_unit": "m",
                            "equations": ["y = 1"],
                            "known_values": {},
                        },
                        "solution_steps": [],
                    },
                    "errors": [],
                }
            )

    def test_uncomputable_sympy_spec_repairs_and_recomputes(self) -> None:
        class RepairingSolutionAgent:
            def repair(
                self,
                question: str,
                semantic_output: dict[str, Any],
                invalid_solution: dict[str, Any],
                validation_error: str,
            ) -> dict[str, Any]:
                del question, semantic_output, invalid_solution
                self.validation_error = validation_error
                return {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "x",
                        "target_unit": "m",
                        "equations": ["x = 2"],
                        "known_values": {},
                    },
                    "solution_steps": ["Repair the underdetermined equation system."],
                }

        repair_agent = RepairingSolutionAgent()
        workflow = PhysicsWorkflow()
        workflow.llm = object()
        workflow.solution_agent = repair_agent
        output = workflow.compute_sympy(
            {
                "question": "Find x.",
                "parsed_question": {"target": {"symbol": "x", "unit": "m"}},
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "x",
                        "target_unit": "m",
                        "equations": ["x = y", "y = x"],
                        "known_values": {},
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        self.assertEqual(output["result"]["answer"], "2")
        self.assertIn("Undefined symbols before SymPy", repair_agent.validation_error)

    def test_sympy_target_resolution_failure_repairs_and_recomputes(self) -> None:
        class RepairingSolutionAgent:
            def repair(
                self,
                question: str,
                semantic_output: dict[str, Any],
                invalid_solution: dict[str, Any],
                validation_error: str,
            ) -> dict[str, Any]:
                del question, semantic_output, invalid_solution
                self.validation_error = validation_error
                return {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "x",
                        "target_unit": "m",
                        "equations": ["x = 3"],
                        "known_values": {},
                    },
                    "solution_steps": ["Repair unresolved target."],
                }

        repair_agent = RepairingSolutionAgent()
        workflow = PhysicsWorkflow()
        workflow.llm = object()
        workflow.solution_agent = repair_agent
        output = workflow.compute_sympy(
            {
                "question": "Find x.",
                "parsed_question": {"target": {"symbol": "x", "unit": "m"}},
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "x",
                        "target_unit": "m",
                        "equations": ["y = 1"],
                        "known_values": {},
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        self.assertEqual(output["result"]["answer"], "3")
        self.assertIn("equations could not resolve the target", repair_agent.validation_error)

    def test_rag_documents_are_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            kb_path = Path(directory) / "kb.json"
            kb_path.write_text(json.dumps([{"question": "first", "cot": "a"}]), encoding="utf-8")
            provider = RAGSolutionProvider(RecordingLLM({}), kb_path=kb_path, top_k=1)
            self.assertEqual(provider.retrieve("first")[0]["question"], "first")
            kb_path.write_text(json.dumps([{"question": "second", "cot": "b"}]), encoding="utf-8")
            self.assertEqual(provider.retrieve("first")[0]["question"], "first")

    def test_rag_solution_provider_uses_prompt_path_when_no_deterministic_shortcut_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            kb_path = Path(directory) / "kb.json"
            kb_path.write_text(json.dumps([{"question": "electric field", "cot": "Use Coulomb's law."}]), encoding="utf-8")
            llm = RecordingLLM(
                {
                    "physics.solution": json.dumps(
                        {
                            "mode": "computational",
                            "answer_type": "numeric",
                            "sympy_spec": {
                                "target_symbol": "E",
                                "target_unit": "N/C",
                                "equations": ["E = Abs(k * q / r**2)"],
                                "known_values": {"k": 9e9, "q": 2e-6, "r": 0.3},
                            },
                            "solution_steps": ["Use Coulomb's law for a point charge."],
                        }
                    )
                }
            )
            provider = RAGSolutionProvider(llm, kb_path=kb_path)
            parsed = {
                "question": "Find the electric field magnitude.",
                "domain": "Electric Charges and Fields",
                "target": {"symbol": "E", "unit": "N/C"},
                "givens": [{"symbol": "q", "si_value": 2e-6}, {"symbol": "r", "si_value": 0.3}],
                "question_kind": "computational",
            }
            output = provider.get_solution(parsed["question"], parsed)

        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution"])
        self.assertEqual(output["sympy_spec"]["target_symbol"], "E")
        self.assertEqual(provider.last_prompt_diagnostics["selected_rule_pack"], ["prompt"])

    def test_rag_examples_are_compact_hints_with_capped_prompt(self) -> None:
        parsed = {
            "question": "Find the magnitude of the electric field at M on the perpendicular bisector.",
            "domain": "Electric Charges and Fields",
            "question_kind": "computational",
            "answer_format": {"requested_form": "magnitude"},
            "geometry": {"present": True, "type": "perpendicular_bisector"},
            "givens": [],
            "target": {"symbol": "E_M", "unit": "N/C"},
        }
        long_cot = (
            "Step 1: Define coordinates for A, B, and M. "
            "Step 2: Use E1x = k * q1 * (xM - xA) / AM**3 and sum components. "
            + "extra context " * 300
        )
        docs = [
            {
                "question": "electric field at perpendicular bisector",
                "answer": "2.16e6",
                "unit": "N/C",
                "cot": long_cot,
            },
            {
                "question": "point charge magnitude field",
                "answer": "E",
                "unit": "N/C",
                "cot": "Step 1: Use E = Abs(k * q / r**2). Step 2: Report magnitude.",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            kb_path = Path(directory) / "kb.json"
            kb_path.write_text(json.dumps(docs), encoding="utf-8")
            provider = RAGSolutionProvider(RecordingLLM({}), kb_path=kb_path)
            retrieved = provider.retrieve("electric field at perpendicular bisector")
            compact = [provider._compact_example(item) for item in retrieved]
            prompt = provider._build_prompt(parsed, compact)

        self.assertEqual(compact, [provider._compact_example(item) for item in retrieved])
        self.assertNotIn("cot", compact[0])
        self.assertLessEqual(len(compact[0]["strategy"]), 2)
        self.assertLessEqual(provider.last_prompt_diagnostics["rag_chars"], 1200)
        self.assertLessEqual(len(prompt), 14000)
        self.assertNotIn("extra context extra context extra context", prompt)

    def test_compute_sympy_prefers_absolute_error_target_when_question_asks_it(self) -> None:
        workflow = PhysicsWorkflow()
        output = workflow.compute_sympy(
            {
                "question": "When measuring voltage with a voltmeter, what is the absolute error of the power?",
                "parsed_question": {
                    "question": "When measuring voltage with a voltmeter, the result is 6.3 +/- 0.1 V. If this is used to calculate power with a current of 0.6 +/- 0.02 A, what is the absolute error of the power?",
                    "target": {"symbol": "P", "unit": "W"},
                    "givens": [
                        {"symbol": "V", "si_value": 6.3, "si_unit": "V", "uncertainty": {"si_value": 0.1}},
                        {"symbol": "I", "si_value": 0.6, "si_unit": "A", "uncertainty": {"si_value": 0.02}},
                    ],
                },
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "P",
                        "target_unit": "W",
                        "equations": ["P = V * I", "delta_P = V * delta_I + I * delta_V"],
                        "known_values": {"V": 6.3, "I": 0.6, "delta_V": 0.1, "delta_I": 0.02},
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        self.assertEqual(output["verified_output"]["final_answer"]["symbol"], "delta_P")
        self.assertEqual(output["result"]["answer"], "0.186")

    def test_compute_sympy_returns_both_absolute_and_relative_error_when_requested(self) -> None:
        workflow = PhysicsWorkflow()
        output = workflow.compute_sympy(
            {
                "question": "Calculate both absolute and relative error.",
                "parsed_question": {
                    "question": "The true value is 30.0 cm, the measured result is 29.7 cm. Calculate the absolute error and the relative error.",
                    "target": {"symbol": "relative_error", "unit": ""},
                    "givens": [
                        {"symbol": "true_value", "si_value": 30.0, "si_unit": "cm", "uncertainty": None},
                        {"symbol": "measured_result", "si_value": 29.7, "si_unit": "cm", "uncertainty": None},
                    ],
                },
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "relative_error",
                        "target_unit": "",
                        "equations": [
                            "absolute_error = Abs(true_value - measured_result)",
                            "relative_error = absolute_error / true_value",
                        ],
                        "known_values": {"true_value": 30.0, "measured_result": 29.7},
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        self.assertEqual(output["result"]["answer"], "absolute_error = 0.3; relative_error = 0.01")

    def test_deterministic_parallel_resistor_total_current_uses_equivalent_resistance(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        solution = deterministic_solution(
            {
                "question": "Resistor D1 is 10 Ohm and D2 is 5 Ohm in parallel with 10 V source. Calculate total current.",
                "domain": "Electric Circuits",
                "target": {"symbol": "I_total", "unit": "A"},
                "givens": [
                    {"symbol": "R1", "si_value": 10, "si_unit": "Ohm"},
                    {"symbol": "R2", "si_value": 5, "si_unit": "Ohm"},
                    {"symbol": "U", "si_value": 10, "si_unit": "V"},
                ],
                "relations": [],
                "question_kind": "computational",
            }
        )
        self.assertIsNotNone(solution)
        equations = solution["sympy_spec"]["equations"]
        self.assertIn("R_eq = 1 / (1 / R1 + 1 / R2)", equations)
        self.assertIn("I_total = U / R_eq", equations)

    def test_compute_sympy_returns_mean_and_mae_when_question_requests_both(self) -> None:
        workflow = PhysicsWorkflow()
        output = workflow.compute_sympy(
            {
                "question": "Three temperature measurements: 36.6C; 36.8C; 36.7C. Calculate the mean and the mean absolute error.",
                "parsed_question": {
                    "question": "Three temperature measurements: 36.6C; 36.8C; 36.7C. Calculate the mean and the mean absolute error.",
                    "target": {"symbol": "mean", "unit": "degC"},
                    "givens": [
                        {"symbol": "x1", "si_value": 36.6, "si_unit": "degC", "uncertainty": None},
                        {"symbol": "x2", "si_value": 36.8, "si_unit": "degC", "uncertainty": None},
                        {"symbol": "x3", "si_value": 36.7, "si_unit": "degC", "uncertainty": None},
                    ],
                },
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "mean",
                        "target_unit": "degC",
                        "equations": [
                            "mean = (x1 + x2 + x3) / 3",
                            "mean_absolute_error = (Abs(x1 - mean) + Abs(x2 - mean) + Abs(x3 - mean)) / 3",
                        ],
                        "known_values": {"x1": 36.6, "x2": 36.8, "x3": 36.7},
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        self.assertEqual(output["result"]["answer"], "mean = 36.7; mean_absolute_error = 0.0666667")

    def test_compute_sympy_ignores_known_target_value_when_target_is_defined_by_equation(self) -> None:
        output = PhysicsWorkflow().compute_sympy(
            {
                "parsed_question": {
                    "question": "A capacitor with C = 4 microF is charged to U = 6 V. Calculate the electric field energy.",
                    "target": {"symbol": "U", "unit": "J"},
                    "givens": [
                        {"symbol": "C", "si_value": 4e-6},
                        {"symbol": "U", "si_value": 6.0},
                    ],
                },
                "solution_output": {
                    "mode": "computational",
                    "answer_type": "numeric",
                    "sympy_spec": {
                        "target_symbol": "U",
                        "target_unit": "J",
                        "equations": ["U = C * V**2 / 2"],
                        "known_values": {"C": 4e-6, "V": 6.0, "U": 6.0},
                    },
                    "solution_steps": [],
                },
                "errors": [],
            }
        )
        self.assertEqual(output["result"]["answer"], "7.2e-05")

    def test_deterministic_midpoint_identical_charges_cancels_to_zero(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Two identical point charges q = 2.0 x 10^-9 C are at A and B, 6.0 cm apart. Find field magnitude at midpoint M.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "E_M", "unit": "N/C"},
            "givens": [
                {"symbol": "q", "si_value": 2e-9},
                {"symbol": "AB", "si_value": 0.06},
            ],
            "question_kind": "computational",
            "geometry": {"present": True, "type": "midpoint_1d"},
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertEqual(computed["result"]["answer"], "0")

    def test_deterministic_solves_unknown_charge_from_zero_field_and_sum(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Two point charges q1 and q2 are placed at A and B, with AB = 2 cm. Given q1 + q2 = 7 x 10^-8 C. At point M, 6 cm from q1 and 8 cm from q2, net electric field strength is E = 0. Find q2.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "q2", "unit": "C"},
            "givens": [{"symbol": "AB", "si_value": 0.02}],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertEqual(computed["result"]["answer"], "1.6e-07")

    def test_deterministic_solves_q1_from_zero_field_and_sum(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Two point charges q1 and q2 are placed at A and B, with AB = 2 cm. Given q1 + q2 = 7 x 10^-8 C. At point M, 6 cm from q1 and 8 cm from q2, net electric field strength is E = 0. Find q1.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "q1", "unit": "C"},
            "givens": [{"symbol": "AB", "si_value": 0.02}],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertEqual(computed["result"]["answer"], "-9e-08")

    def test_deterministic_parallel_plate_capacitor_charge(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "A parallel plate capacitor has area S = 400 cm^2, plate separation d = 2 mm, dielectric constant epsilon = 1.5, and voltage U = 100 V. Calculate charge.",
            "domain": "Capacitance",
            "target": {"symbol": "Q", "unit": "C"},
            "givens": [
                {"symbol": "S", "si_value": 0.04},
                {"symbol": "d", "si_value": 0.002},
                {"symbol": "epsilon", "si_value": 1.5},
                {"symbol": "U", "si_value": 100},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertEqual(computed["result"]["answer"], "2.65626e-08")

    def test_deterministic_charge_sharing_two_identical_capacitors(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "A 2 microF capacitor is charged to 12 V, then disconnected, and its charge is equally shared among two identical capacitors. Calculate the total remaining energy.",
            "domain": "Capacitance",
            "target": {"symbol": "E_total", "unit": "J"},
            "givens": [
                {"symbol": "C", "si_value": 2e-6},
                {"symbol": "V", "si_value": 12.0},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertEqual(computed["result"]["answer"], "7.2e-05")

    def test_deterministic_inductance_from_magnetic_energy(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "An inductor has magnetic field energy of 0.2 mJ when current is 0.2 A. Calculate inductance.",
            "domain": "Inductance",
            "target": {"symbol": "L", "unit": "H"},
            "givens": [
                {"symbol": "W_B", "si_value": 0.2e-3},
                {"symbol": "I", "si_value": 0.2},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertEqual(computed["result"]["answer"], "0.01")

    def test_deterministic_inductor_energy_after_current_halved_keeps_requested_mj(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "An inductor has a magnetic field energy of 2.0 mJ. If the current is halved, what is the remaining energy?",
            "domain": "Inductance",
            "target": {"symbol": "W_new", "unit": "mJ"},
            "givens": [{"symbol": "W", "si_value": 2e-3, "si_unit": "J"}],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 0.5)

    def test_deterministic_source_charge_from_force_on_test_charge(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "A charge q = 10^-7 C is placed in the electric field of a point charge Q, experiencing a force F = 3 mN. The two charges are separated by a distance of 30 cm in vacuum. Calculate Q.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "Q", "unit": "C"},
            "givens": [
                {"symbol": "q", "si_value": 1e-7, "si_unit": "C"},
                {"symbol": "F", "si_value": 3e-3, "si_unit": "N"},
                {"symbol": "r", "si_value": 0.3, "si_unit": "m"},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        self.assertIn("electrostatics.source_charge_from_force_on_test_charge", solution["formula_ids"])
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 3e-7)

    def test_deterministic_self_induced_emf_does_not_use_coulomb_constant(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "A solenoid has L = 0.2 H. The current decreases uniformly from 2 A to 0 in 0.02 s. Calculate the induced electromotive force.",
            "domain": "Inductance",
            "target": {"symbol": "epsilon", "unit": "V"},
            "givens": [
                {"symbol": "L", "si_value": 0.2, "si_unit": "H"},
                {"symbol": "I_initial", "si_value": 2.0, "si_unit": "A"},
                {"symbol": "I_final", "si_value": 0.0, "si_unit": "A"},
                {"symbol": "delta_t", "si_value": 0.02, "si_unit": "s"},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        self.assertNotIn("k", " ".join(solution["sympy_spec"]["equations"]))
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 20.0)

    def test_deterministic_max_magnetic_energy_from_l_and_imax(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "An inductor has inductance 0.25 H. When current reaches maximum value 2sqrt2 A, what is maximum magnetic field energy?",
            "domain": "Inductance",
            "target": {"symbol": "W_max", "unit": "J"},
            "givens": [
                {"symbol": "L", "si_value": 0.25},
                {"symbol": "I_max", "si_value": 2 * math.sqrt(2)},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertEqual(computed["result"]["answer"], "1")

    def test_deterministic_point_charge_in_dielectric_computes_signed_charge(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "A point charge q is in a medium with dielectric constant 2.5. At point M, 0.4 m away, electric field has magnitude 9 x 10^5 V/m and points toward q.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "q", "unit": "C"},
            "givens": [
                {"symbol": "epsilon_r", "si_value": 2.5},
                {"symbol": "r", "si_value": 0.4},
                {"symbol": "E", "si_value": 9e5},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertEqual(computed["result"]["answer"], "-4e-05")

    def test_deterministic_two_section_phase_power_solves_r2(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "The circuit AB consists of R1 = 50 Ohm and segment MB with R2 and L, with LComega^2 = 1. uAM and uMB are in quadrature. With U = 100 V, total consumed power is 100 W. Find R2.",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "R2", "unit": "Ohm"},
            "givens": [
                {"symbol": "R1", "si_value": 50},
                {"symbol": "U", "si_value": 100},
                {"symbol": "P", "si_value": 100},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertEqual(computed["result"]["answer"], "50")

    def test_deterministic_field_from_force_on_charge(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "A charge q = 3 microC experiences an electric force of magnitude 0.48 N. Determine the electric field magnitude at its position.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "E", "unit": "N/C"},
            "givens": [
                {"symbol": "q", "si_value": 3e-6, "si_unit": "C"},
                {"symbol": "F", "si_value": 0.48, "si_unit": "N"},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        self.assertIn("electrostatics.field_from_force_on_charge", solution["formula_ids"])
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertEqual(computed["result"]["answer"], "160000")

    def test_deterministic_isosceles_ac_equals_bc_field(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Two identical charges +6 microC are at A and B, 10 cm apart. Point C satisfies AC = BC = 8 cm. Find the electric field magnitude at C.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "E_C", "unit": "N/C"},
            "givens": [
                {"symbol": "q", "si_value": 6e-6, "si_unit": "C"},
                {"symbol": "AB", "si_value": 0.1, "si_unit": "m"},
                {"symbol": "AC", "si_value": 0.08, "si_unit": "m"},
                {"symbol": "BC", "si_value": 0.08, "si_unit": "m"},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        self.assertIn("electrostatics.triangle_distance_geometry", solution["formula_ids"])
        self.assertIn("electrostatics.point_charge_field_vector", solution["formula_ids"])
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertEqual(computed["result"]["answer"], "1.3173e+07 N/C")

    def test_vector_engine_perpendicular_bisector_adds_components(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Two electric charges q1 = +2e-8 C and q2 = +2e-8 C are 8 cm apart. Calculate the electric field at point M on the perpendicular bisector, 6 cm from AB.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "E_net_magnitude", "unit": "N/C"},
            "givens": [
                {"symbol": "q1", "si_value": 2e-8},
                {"symbol": "q2", "si_value": 2e-8},
                {"symbol": "AB", "si_value": 0.08},
                {"symbol": "ell", "si_value": 0.06},
            ],
            "relations": ["M lies on the perpendicular bisector of AB"],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        self.assertIn("E_net_x = E1x + E2x", solution["sympy_spec"]["equations"])
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        vector = computed["verified_output"]["vector_result"]

        self.assertAlmostEqual(vector["components"][0], 0, places=7)
        self.assertAlmostEqual(vector["magnitude"], 57600, delta=5)
        self.assertNotEqual(computed["result"]["answer"], "144000 N/C")

    def test_vector_engine_triangle_distances_for_force(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Two electric charges q1 = +6e-8 C and q2 = +1e-8 C are at A and B, AB = 15 cm. A charge q3 = -1e-8 C is at C with AC = 14 cm and BC = 6 cm. Calculate the net electric force on q3.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "F_net_magnitude", "unit": "N"},
            "givens": [
                {"symbol": "q1", "si_value": 6e-8},
                {"symbol": "q2", "si_value": 1e-8},
                {"symbol": "q3", "si_value": -1e-8},
                {"symbol": "AB", "si_value": 0.15},
                {"symbol": "AC", "si_value": 0.14},
                {"symbol": "BC", "si_value": 0.06},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        equations = solution["sympy_spec"]["equations"]
        self.assertIn("Cx = (AC**2 + AB**2 - BC**2) / (2 * AB)", equations)
        self.assertIn("F_net_x = q3 * E_net_x", equations)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        ax, ay = 0.0, 0.0
        bx, by = 0.15, 0.0
        cx = (0.14**2 + 0.15**2 - 0.06**2) / (2 * 0.15)
        cy = math.sqrt(0.14**2 - cx**2)
        e1x = 9e9 * 6e-8 * (cx - ax) / 0.14**3
        e1y = 9e9 * 6e-8 * (cy - ay) / 0.14**3
        e2x = 9e9 * 1e-8 * (cx - bx) / 0.06**3
        e2y = 9e9 * 1e-8 * (cy - by) / 0.06**3
        expected = math.sqrt((-1e-8 * (e1x + e2x)) ** 2 + (-1e-8 * (e1y + e2y)) ** 2)

        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"]["magnitude"], expected)
        self.assertNotAlmostEqual(computed["verified_output"]["final_answer"]["value"]["magnitude"], 0.0005255102040816325)

    def test_vector_engine_right_angle_vertex_force(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Three identical charges q = +5e-7 C are placed at the three vertices of an isosceles right triangle with legs of 15 cm. Calculate the net force acting on the charge at the right angle.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "F_net_magnitude", "unit": "N"},
            "givens": [
                {"symbol": "q", "si_value": 5e-7},
                {"symbol": "a", "si_value": 0.15},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        expected = math.sqrt(2) * 9e9 * (5e-7) ** 2 / 0.15**2
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"]["magnitude"], expected)
        self.assertNotEqual(computed["result"]["answer"], "0.1 N")

    def test_zero_field_same_sign_respects_requested_distance(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        base = {
            "question": "At A and B, q1 = +4e-6 C and q2 = +1e-6 C are 12 cm apart. Find point M where the electric field is zero.",
            "domain": "Electric Charges and Fields",
            "givens": [
                {"symbol": "q1", "si_value": 4e-6},
                {"symbol": "q2", "si_value": 1e-6},
                {"symbol": "AB", "si_value": 0.12},
            ],
            "relations": ["electric field is zero"],
            "question_kind": "computational",
        }
        parsed_am = {**base, "target": {"symbol": "AM", "unit": "m"}}
        parsed_bm = {**base, "target": {"symbol": "BM", "unit": "m"}}
        solution_am = deterministic_solution(parsed_am)
        solution_bm = deterministic_solution(parsed_bm)
        self.assertIsNotNone(solution_am)
        self.assertIsNotNone(solution_bm)
        computed_am = PhysicsWorkflow().compute_sympy({"parsed_question": parsed_am, "solution_output": solution_am, "errors": []})
        computed_bm = PhysicsWorkflow().compute_sympy({"parsed_question": parsed_bm, "solution_output": solution_bm, "errors": []})

        self.assertAlmostEqual(computed_am["verified_output"]["final_answer"]["value"], 0.08)
        self.assertAlmostEqual(computed_bm["verified_output"]["final_answer"]["value"], 0.04)

    def test_zero_field_same_sign_negative_charges_uses_magnitudes(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        base = {
            "question": "At A and B, q1 = -4e-6 C and q2 = -1e-6 C are 12 cm apart. Find point M where the electric field is zero.",
            "domain": "Electric Charges and Fields",
            "givens": [
                {"symbol": "q1", "si_value": -4e-6},
                {"symbol": "q2", "si_value": -1e-6},
                {"symbol": "AB", "si_value": 0.12},
            ],
            "relations": ["electric field is zero"],
            "question_kind": "computational",
        }
        for target, expected in (("AM", 0.08), ("BM", 0.04)):
            with self.subTest(target=target):
                parsed = {**base, "target": {"symbol": target, "unit": "m"}}
                solution = deterministic_solution(parsed)
                self.assertIsNotNone(solution)
                computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
                self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], expected)

    def test_conceptual_lc_electric_energy_maximum_uses_direct_answer(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "In an LC circuit, when does electric field energy reach maximum?",
            "domain": "Alternating-Current Circuits",
            "target": {"symbol": "answer", "unit": ""},
            "givens": [],
            "question_kind": "conceptual",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertIn("capacitor charge is maximum", computed["result"]["answer"])
        self.assertIn("current is zero", computed["result"]["answer"])

    def test_perpendicular_bisector_max_field_height_template(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Two equal charges are 10 cm apart. On the perpendicular bisector, find h where the electric field is maximum.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "h", "unit": "m"},
            "givens": [{"symbol": "AB", "si_value": 0.10, "si_unit": "m"}],
            "relations": ["perpendicular bisector"],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 0.10 / (2 * math.sqrt(2)))

    def test_self_induction_validator_rejects_coulomb_constant_contamination(self) -> None:
        with self.assertRaises(WorkflowExecutionError):
            PhysicsWorkflow().compute_sympy(
                {
                    "parsed_question": {
                        "question": "Calculate the induced electromotive force of a self-inductor.",
                        "target": {"symbol": "epsilon", "unit": "V"},
                        "givens": [
                            {"symbol": "L", "si_value": 0.2},
                            {"symbol": "I_initial", "si_value": 2.0},
                            {"symbol": "I_final", "si_value": 0.0},
                            {"symbol": "delta_t", "si_value": 0.02},
                        ],
                    },
                    "solution_output": {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "epsilon",
                            "target_unit": "V",
                            "equations": ["delta_I = I_final - I_initial", "epsilon = -k * L * delta_I / delta_t"],
                            "known_values": {"L": 0.2, "I_initial": 2.0, "I_final": 0.0, "delta_t": 0.02, "k": 9e9},
                        },
                        "solution_steps": [],
                    },
                    "errors": [],
                }
            )

    def test_measurement_maximum_possible_current_uses_uncertainty(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "When measuring the current through a resistor, a value of 0.25 A was obtained with an uncertainty of +/-0.01 A. What is the maximum possible current?",
            "domain": "Measurement and Uncertainty",
            "target": {"symbol": "I_max", "unit": "A"},
            "givens": [
                {"symbol": "I", "si_value": 0.25, "si_unit": "A", "uncertainty": {"si_value": 0.01, "si_unit": "A", "kind": "absolute"}},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 0.26)

    def test_measurement_percentage_relative_uncertainty_uses_delta_alias(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "The length is measured as 60.0 +/- 0.3 cm. Calculate the percentage relative uncertainty.",
            "domain": "Measurement and Uncertainty",
            "target": {"symbol": "error_percentage", "unit": "%"},
            "givens": [
                {"symbol": "L", "si_value": 0.6, "si_unit": "m", "uncertainty": {"si_value": 0.003, "si_unit": "m", "kind": "absolute"}},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 0.5)

    def test_least_count_percentage_error_uses_half_least_count(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "A length is measured as 10.0 cm with least count 0.1 cm. Calculate the percentage relative error.",
            "domain": "Measurement and Uncertainty",
            "target": {"symbol": "error_percentage", "unit": "%"},
            "givens": [
                {"symbol": "measured_value", "si_value": 0.1, "si_unit": "m"},
                {"symbol": "least_count", "si_value": 0.001, "si_unit": "m"},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 0.5)

    def test_capacitance_from_energy_voltage_in_requested_microfarads(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "A capacitor stores 3.6 mJ of electrical energy when the voltage across it is 120 V. Calculate its capacitance C (microF).",
            "domain": "Capacitance",
            "target": {"symbol": "C", "unit": "microF"},
            "givens": [],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 0.5)

    def test_isolated_capacitor_energy_decreases_when_permittivity_increases(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "A parallel-plate capacitor is fully charged and then disconnected from its power source. Subsequently, it is placed in an environment where the permittivity increases by a factor of 3. Calculate the new energy stored if the initial energy was 1 microJ.",
            "domain": "Capacitance",
            "target": {"symbol": "U_new", "unit": "J"},
            "givens": [],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 1e-6 / 3)

    def test_parallel_plate_capacitance_corrects_area_units_from_text(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Calculate the capacitance of a parallel-plate air capacitor with a plate area of 34.0 cm^2 and a plate separation of 0.50 mm.",
            "domain": "Capacitance",
            "target": {"symbol": "C", "unit": "F"},
            "givens": [
                {"symbol": "A", "si_value": 3400, "si_unit": "m2"},
                {"symbol": "d", "si_value": 0.0005, "si_unit": "m"},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 8.8541878128e-12 * 0.0034 / 0.0005)

    def test_parallel_plate_breakdown_charge_uses_epsilon0_area(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "A parallel-plate capacitor has two circular plates, each with a radius of 45 cm, separated by 2.1 mm. Calculate the maximum charge without dielectric breakdown of air (Emax = 3x10^5 V/m).",
            "domain": "Capacitance",
            "target": {"symbol": "Q_max", "unit": "C"},
            "givens": [
                {"symbol": "r", "si_value": 0.45, "si_unit": "m"},
                {"symbol": "d", "si_value": 0.0021, "si_unit": "m"},
                {"symbol": "E_max", "si_value": 3e5, "si_unit": "V/m"},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        expected = 8.8541878128e-12 * 3e5 * math.pi * 0.45**2
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], expected)

    def test_dielectric_constant_from_parallel_plate_geometry_uses_nf(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "A capacitor has a capacitance of 9.2 nF, its plates are square with an area of 10.5 cm^2, and the distance between the two plates is 1.1x10^-5 m. Calculate the dielectric constant.",
            "domain": "Capacitance",
            "target": {"symbol": "epsilon_r", "unit": ""},
            "givens": [
                {"symbol": "C", "si_value": 9.2e-6, "si_unit": "F"},
                {"symbol": "A", "si_value": 0.00105, "si_unit": "m2"},
                {"symbol": "d", "si_value": 1.1e-5, "si_unit": "m"},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        expected = 9.2e-9 * 1.1e-5 / (8.8541878128e-12 * 0.00105)
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], expected)

    def test_parallel_capacitor_voltage_uses_constraint_to_choose_branch(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Two capacitors C1 = 0.4 microF and C2 = 0.6 microF are connected in parallel to a source with voltage U < 60 V. One capacitor has charge Q = 3e-5 C. Calculate U.",
            "domain": "Capacitance",
            "target": {"symbol": "U", "unit": "V"},
            "givens": [
                {"symbol": "C1", "si_value": 0.4e-6},
                {"symbol": "C2", "si_value": 0.6e-6},
                {"symbol": "Q", "si_value": 3e-5},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 50)

    def test_dielectric_point_charge_can_use_text_extracted_epsilon_without_untrusted_known(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "A point charge q is in a medium with dielectric constant 2.5. At point M, 0.4 m away, electric field has magnitude 9 x 10^5 V/m and points toward q.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "q", "unit": "C"},
            "givens": [],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        self.assertIn("epsilon_r_eff = 2.5", solution["sympy_spec"]["equations"])
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], -4e-5)

    def test_magnetic_flux_from_uniform_field_and_area(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Given the cross-sectional area of a solenoid is 2 cm^2, and B = 2x10^-3 T. Calculate the magnetic flux through this cross-section.",
            "domain": "Sources of Magnetic Fields",
            "target": {"symbol": "Phi", "unit": "Wb"},
            "givens": [
                {"symbol": "A", "si_value": 0.0002, "si_unit": "m2"},
                {"symbol": "B", "si_value": 2e-3, "si_unit": "T"},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 4e-7)

    def test_resultant_force_angle_in_degrees_is_converted_to_radians(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Given F1 = 9 N, F2 = 12 N and the angle between them is theta = 135 degrees. Find the resultant force.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "F_resultant", "unit": "N"},
            "givens": [
                {"symbol": "F1", "si_value": 9},
                {"symbol": "F2", "si_value": 12},
                {"symbol": "theta", "si_value": 135},
            ],
            "question_kind": "computational",
        }
        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        expected = math.sqrt(9**2 + 12**2 + 2 * 9 * 12 * math.cos(math.radians(135)))
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], expected)

    def _assert_deterministic_numeric_answer(self, parsed: dict[str, Any], expected: float, rel_tol: float = 1e-3) -> dict[str, Any]:
        from agents.physics.Solution.formula_lib import deterministic_solution

        solution = deterministic_solution(parsed)
        self.assertIsNotNone(solution, parsed["question"])
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        value = computed["verified_output"]["final_answer"]["value"]
        actual = value["magnitude"] if isinstance(value, dict) else value
        tolerance = max(abs(expected) * rel_tol, 1e-6)
        self.assertAlmostEqual(actual, expected, delta=tolerance, msg=parsed["question"])
        return computed

    def test_acceptance_coulomb_vector_force_regressions(self) -> None:
        cases = [
            (
                {
                    "question": "Two charges q1 = +4 microC and q2 = +4 microC are 10 cm apart. A charge q0 = -2 microC lies between them, 4 cm from q1 and 6 cm from q2. Find the magnitude of the net force on q0.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 4e-6},
                        {"symbol": "q2", "si_value": 4e-6},
                        {"symbol": "q0", "si_value": -2e-6},
                        {"symbol": "AB", "si_value": 0.10},
                        {"symbol": "AM", "si_value": 0.04},
                        {"symbol": "BM", "si_value": 0.06},
                    ],
                    "relations": ["q0 lies between q1 at A and q2 at B"],
                    "question_kind": "computational",
                    "answer_format": {"requested_form": "magnitude"},
                    "geometry": {"present": True, "type": "collinear", "line_order": ["A", "M", "B"]},
                },
                25.0,
            ),
            (
                {
                    "question": "Charges q1 = +3 microC and q2 = -3 microC are at A and B, 8 cm apart. Point M is on the extension beyond A with AM = 4 cm and BM = 12 cm. A charge q0 = +1 microC is placed at M. Find the net force on q0.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 3e-6},
                        {"symbol": "q2", "si_value": -3e-6},
                        {"symbol": "q0", "si_value": 1e-6},
                        {"symbol": "AB", "si_value": 0.08},
                        {"symbol": "AM", "si_value": 0.04},
                        {"symbol": "BM", "si_value": 0.12},
                    ],
                    "relations": ["M is on the extension beyond A"],
                    "question_kind": "computational",
                    "geometry": {"present": True, "type": "collinear", "line_order": ["M", "A", "B"]},
                },
                15.0,
            ),
            (
                {
                    "question": "Two charges q1 = +1 microC and q2 = -4 microC are 10 cm apart. A charge q0 = +2 microC is placed at the midpoint. Calculate the magnitude of the net force on q0.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 1e-6},
                        {"symbol": "q2", "si_value": -4e-6},
                        {"symbol": "q0", "si_value": 2e-6},
                        {"symbol": "AB", "si_value": 0.10},
                    ],
                    "question_kind": "computational",
                    "answer_format": {"requested_form": "magnitude"},
                    "geometry": {"present": True, "type": "midpoint_1d"},
                },
                36.0,
            ),
            (
                {
                    "question": "Points M, A, B are collinear in that order, with AM = 3 cm and AB = 5 cm. Charges q1 = +2 microC at A, q2 = -3 microC at B, and q0 = +1 microC at M. Calculate the net force on q0.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 2e-6},
                        {"symbol": "q2", "si_value": -3e-6},
                        {"symbol": "q0", "si_value": 1e-6},
                        {"symbol": "AM", "si_value": 0.03},
                        {"symbol": "AB", "si_value": 0.05},
                    ],
                    "relations": ["Points M, A, B are collinear in that order"],
                    "question_kind": "computational",
                    "geometry": {"present": True, "type": "collinear", "line_order": ["M", "A", "B"]},
                },
                15.78125,
            ),
            (
                {
                    "question": "Two equal positive charges q1 = q2 = +3 microC are fixed at A and B, 12 cm apart. A charge q0 = +1 microC is placed at the midpoint. Determine the net force on q0.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 3e-6},
                        {"symbol": "q2", "si_value": 3e-6},
                        {"symbol": "q0", "si_value": 1e-6},
                        {"symbol": "AB", "si_value": 0.12},
                    ],
                    "question_kind": "computational",
                    "geometry": {"present": True, "type": "midpoint_1d"},
                },
                0.0,
            ),
            (
                {
                    "question": "Charges q1 = +2 microC and q2 = -2 microC are placed at A and B, 8 cm apart. A charge q0 = +1 microC is placed at the midpoint of AB. Find the net force on q0.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 2e-6},
                        {"symbol": "q2", "si_value": -2e-6},
                        {"symbol": "q0", "si_value": 1e-6},
                        {"symbol": "AB", "si_value": 0.08},
                    ],
                    "question_kind": "computational",
                    "geometry": {"present": True, "type": "midpoint_1d"},
                },
                22.5,
            ),
            (
                {
                    "question": "Two charges q1 = +5 microC and q2 = +5 microC are 10 cm apart. A test charge q0 = +1 microC is at C with AC = 6 cm and BC = 8 cm. Calculate the net force on q0.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 5e-6},
                        {"symbol": "q2", "si_value": 5e-6},
                        {"symbol": "q0", "si_value": 1e-6},
                        {"symbol": "AB", "si_value": 0.10},
                        {"symbol": "AC", "si_value": 0.06},
                        {"symbol": "BC", "si_value": 0.08},
                    ],
                    "question_kind": "computational",
                    "geometry": {"present": True, "type": "triangle_by_sides"},
                },
                14.343,
            ),
            (
                {
                    "question": "Two identical charges q1 = q2 = +2 microC are 6 cm apart. A charge q0 = +1 microC is on the perpendicular bisector, 4 cm from AB. Find the net force on q0.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 2e-6},
                        {"symbol": "q2", "si_value": 2e-6},
                        {"symbol": "q0", "si_value": 1e-6},
                        {"symbol": "AB", "si_value": 0.06},
                        {"symbol": "ell", "si_value": 0.04},
                    ],
                    "relations": ["q0 lies on the perpendicular bisector of AB"],
                    "question_kind": "computational",
                },
                11.52,
            ),
            (
                {
                    "question": "Two charges q1 = +2 microC and q2 = -2 microC are 6 cm apart. A test charge q0 = +1 microC is on the perpendicular bisector of AB, 4 cm from AB. Find the magnitude of the net force on q0.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 2e-6},
                        {"symbol": "q2", "si_value": -2e-6},
                        {"symbol": "q0", "si_value": 1e-6},
                        {"symbol": "AB", "si_value": 0.06},
                        {"symbol": "ell", "si_value": 0.04},
                    ],
                    "relations": ["q0 lies on the perpendicular bisector of AB"],
                    "question_kind": "computational",
                    "answer_format": {"requested_form": "magnitude"},
                },
                8.64,
            ),
            (
                {
                    "question": "Three identical charges q = +1 microC are placed at the vertices of an isosceles right triangle with equal legs 12 cm. Find the net force on the charge at the right-angle vertex.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [{"symbol": "q", "si_value": 1e-6}, {"symbol": "a", "si_value": 0.12}],
                    "question_kind": "computational",
                },
                0.883883,
            ),
            (
                {
                    "question": "Three identical charges q = +2 microC are placed at the vertices of an equilateral triangle with side 15 cm. Calculate the net force on one charge.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [{"symbol": "q", "si_value": 2e-6}, {"symbol": "side", "si_value": 0.15}],
                    "question_kind": "computational",
                },
                2.77128,
            ),
            (
                {
                    "question": "Charges q1 = q2 = +1 microC and q3 = -1 microC are placed at the vertices of an equilateral triangle of side 10 cm. Find the net force on q3.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 1e-6},
                        {"symbol": "q2", "si_value": 1e-6},
                        {"symbol": "q3", "si_value": -1e-6},
                        {"symbol": "side", "si_value": 0.10},
                    ],
                    "question_kind": "computational",
                },
                1.55885,
            ),
            (
                {
                    "question": "Two equal charges q1 = q2 = +2 microC are at two vertices of an equilateral triangle of side 5 cm. A charge q0 = +1 microC is at the third vertex. Find the magnitude of the net force on q0.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 2e-6},
                        {"symbol": "q2", "si_value": 2e-6},
                        {"symbol": "q0", "si_value": 1e-6},
                        {"symbol": "side", "si_value": 0.05},
                    ],
                    "question_kind": "computational",
                    "answer_format": {"requested_form": "magnitude"},
                },
                12.4708,
            ),
            (
                {
                    "question": "A charge is acted on by two equal electric forces of 5 N. The resultant force is also 5 N. Find the angle between the two forces.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "theta", "unit": "degrees"},
                    "givens": [
                        {"symbol": "F1", "si_value": 5},
                        {"symbol": "F2", "si_value": 5},
                        {"symbol": "F_resultant", "si_value": 5},
                    ],
                    "question_kind": "computational",
                },
                120.0,
            ),
        ]

        for parsed, expected in cases:
            with self.subTest(question=parsed["question"]):
                computed = self._assert_deterministic_numeric_answer(parsed, expected)
                self.assertNotEqual(computed["result"]["answer"], "Unknown")

    def test_acceptance_existing_correct_cases_do_not_regress(self) -> None:
        cases = [
            (
                {
                    "question": "Two charges q1 = -3 microC and q2 = +2 microC are separated by 9 cm. Calculate the magnitude of the force between them and state whether it is attractive or repulsive.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F", "unit": "N"},
                    "givens": [{"symbol": "q1", "si_value": -3e-6}, {"symbol": "q2", "si_value": 2e-6}, {"symbol": "r", "si_value": 0.09}],
                    "question_kind": "computational",
                },
                6.66667,
                "attractive",
            ),
            (
                {
                    "question": "Three charges lie on a straight line: q1 = +2 microC, q0 = +1 microC, and q2 = +6 microC. The distances are q1q0 = 3 cm and q0q2 = 6 cm. Find the net force on q0.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net_magnitude", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 2e-6},
                        {"symbol": "q0", "si_value": 1e-6},
                        {"symbol": "q2", "si_value": 6e-6},
                        {"symbol": "AM", "si_value": 0.03},
                        {"symbol": "BM", "si_value": 0.06},
                    ],
                    "relations": ["q0 lies between q1 and q2 on a straight line"],
                    "question_kind": "computational",
                    "geometry": {"present": True, "type": "collinear", "line_order": ["A", "M", "B"]},
                },
                5.0,
                "",
            ),
            (
                {
                    "question": "Two charges q1 = +2x10^-8 C and q2 = +5x10^-8 C are separated by 4 cm. Calculate the repulsive force between them.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F", "unit": "N"},
                    "givens": [{"symbol": "q1", "si_value": 2e-8}, {"symbol": "q2", "si_value": 5e-8}, {"symbol": "r", "si_value": 0.04}],
                    "question_kind": "computational",
                },
                0.005625,
                "repulsive",
            ),
            (
                {
                    "question": "Two identical point charges q are 12 cm apart in air and repel each other with a force of 3.6 N. Find q.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "q", "unit": "C"},
                    "givens": [{"symbol": "F", "si_value": 3.6}, {"symbol": "r", "si_value": 0.12}],
                    "question_kind": "computational",
                },
                2.40166e-6,
                "",
            ),
            (
                {
                    "question": "Two point charges q1 = +3 microC and q2 = -4 microC are placed 6 cm apart in air. Calculate the magnitude of the electric force between them.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F", "unit": "N"},
                    "givens": [{"symbol": "q1", "si_value": 3e-6}, {"symbol": "q2", "si_value": -4e-6}, {"symbol": "r", "si_value": 0.06}],
                    "question_kind": "computational",
                    "answer_format": {"requested_form": "magnitude"},
                },
                30.0,
                "",
            ),
            (
                {
                    "question": "Two equal electric forces, each 12 N, act at an angle of 120 degrees to each other. Calculate the resultant force.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_resultant", "unit": "N"},
                    "givens": [{"symbol": "F1", "si_value": 12}, {"symbol": "F2", "si_value": 12}, {"symbol": "theta", "si_value": 120}],
                    "question_kind": "computational",
                },
                12.0,
                "",
            ),
            (
                {
                    "question": "Two electric forces have magnitudes 6 N and 10 N and form an angle of 60 degrees. Find the resultant force.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_resultant", "unit": "N"},
                    "givens": [{"symbol": "F1", "si_value": 6}, {"symbol": "F2", "si_value": 10}, {"symbol": "theta", "si_value": 60}],
                    "question_kind": "computational",
                },
                14.0,
                "",
            ),
            (
                {
                    "question": "Two electric forces of 8 N and 15 N act perpendicular to each other. Calculate the magnitude of the resultant force.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_resultant", "unit": "N"},
                    "givens": [{"symbol": "F1", "si_value": 8}, {"symbol": "F2", "si_value": 15}],
                    "question_kind": "computational",
                    "answer_format": {"requested_form": "magnitude"},
                },
                17.0,
                "",
            ),
            (
                {
                    "question": "Two collinear electric forces of 18 N and 7 N act in opposite directions on a charge. Find the magnitude of the resultant force.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_resultant", "unit": "N"},
                    "givens": [{"symbol": "F1", "si_value": 18}, {"symbol": "F2", "si_value": 7}],
                    "question_kind": "computational",
                    "answer_format": {"requested_form": "magnitude"},
                },
                11.0,
                "",
            ),
            (
                {
                    "question": "Two electric forces act in the same direction with magnitudes 6 N and 9 N. Calculate the resultant force.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_resultant", "unit": "N"},
                    "givens": [{"symbol": "F1", "si_value": 6}, {"symbol": "F2", "si_value": 9}],
                    "question_kind": "computational",
                },
                15.0,
                "",
            ),
            (
                {
                    "question": "A series RLC circuit with C = 20 microF resonates at f = 100 Hz. Calculate the inductance.",
                    "domain": "Alternating-Current Circuits",
                    "target": {"symbol": "L", "unit": "H"},
                    "givens": [{"symbol": "C", "si_value": 20e-6}, {"symbol": "f", "si_value": 100}],
                    "relations": ["series RLC resonance"],
                    "question_kind": "computational",
                },
                0.126651,
                "",
            ),
        ]

        for parsed, expected, answer_fragment in cases:
            with self.subTest(question=parsed["question"]):
                computed = self._assert_deterministic_numeric_answer(parsed, expected)
                if answer_fragment:
                    self.assertIn(answer_fragment, computed["result"]["answer"])

    def test_acceptance_uncompleted_run_rule_based_regressions(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        cases: list[tuple[dict[str, Any], Any]] = [
            (
                {
                    "question": "An ammeter has a least count of 0.05 A and reads 2.5 A. Taking the absolute error equal to the least count, calculate the relative error.",
                    "domain": "Measurement and Uncertainty",
                    "target": {"symbol": "relative_error", "unit": ""},
                    "givens": [
                        {"symbol": "least_count", "si_value": 0.05, "si_unit": "A"},
                        {"symbol": "measured_value", "si_value": 2.5, "si_unit": "A"},
                    ],
                    "question_kind": "computational",
                },
                0.02,
            ),
            (
                {
                    "question": "Power is calculated by P = UI. Given U = 15 +/- 0.6 V and I = 2.0 +/- 0.05 A, calculate the percentage relative error of P.",
                    "domain": "Measurement and Uncertainty",
                    "target": {"symbol": "percentage_relative_error", "unit": "%"},
                    "givens": [
                        {"symbol": "U", "si_value": 15.0, "si_unit": "V", "uncertainty": {"si_value": 0.6}},
                        {"symbol": "I", "si_value": 2.0, "si_unit": "A", "uncertainty": {"si_value": 0.05}},
                    ],
                    "question_kind": "computational",
                },
                6.5,
            ),
            (
                {
                    "question": "An isolated charged air capacitor initially has voltage U0 = 500 V. A dielectric with relative permittivity epsilon_r = 5 is inserted fully between the plates. Find the new voltage.",
                    "domain": "Capacitance",
                    "target": {"symbol": "U_new", "unit": "V"},
                    "givens": [
                        {"symbol": "U0", "si_value": 500.0, "si_unit": "V"},
                        {"symbol": "epsilon_r", "si_value": 5.0},
                    ],
                    "question_kind": "computational",
                },
                100.0,
            ),
            (
                {
                    "question": "A 30 pF capacitor remains connected to a 90 V battery. A dielectric with relative permittivity epsilon_r = 4 is inserted fully. Find the new charge on the capacitor.",
                    "domain": "Capacitance",
                    "target": {"symbol": "Q_new", "unit": "C"},
                    "givens": [
                        {"symbol": "C", "si_value": 30e-12, "si_unit": "F"},
                        {"symbol": "U", "si_value": 90.0, "si_unit": "V"},
                        {"symbol": "epsilon_r", "si_value": 4.0},
                    ],
                    "question_kind": "computational",
                },
                1.08e-8,
            ),
            (
                {
                    "question": "In an ideal LC circuit, the total energy is 80 mJ. At one instant, the electric energy equals the magnetic energy. Find each energy.",
                    "domain": "Alternating-Current Circuits",
                    "target": {"symbol": "each_energy", "unit": "mJ"},
                    "givens": [{"symbol": "E_total", "si_value": 80e-3, "si_unit": "J"}],
                    "question_kind": "computational",
                },
                {"electric_energy": 40.0, "magnetic_energy": 40.0},
            ),
            (
                {
                    "question": "A coil of inductance L = 0.04 H stores magnetic energy W = 0.08 J. Find the current through the coil.",
                    "domain": "Inductance",
                    "target": {"symbol": "I", "unit": "A"},
                    "givens": [
                        {"symbol": "L", "si_value": 0.04, "si_unit": "H"},
                        {"symbol": "W", "si_value": 0.08, "si_unit": "J"},
                    ],
                    "question_kind": "computational",
                },
                2.0,
            ),
            (
                {
                    "question": "Two identical positive point charges Q1 = Q2 = +3 microC are 6 cm apart. Determine the net electric field at the midpoint.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "E_net_magnitude", "unit": "N/C"},
                    "givens": [
                        {"symbol": "Q1", "si_value": 3e-6, "si_unit": "C"},
                        {"symbol": "Q2", "si_value": 3e-6, "si_unit": "C"},
                        {"symbol": "AB", "si_value": 0.06, "si_unit": "m"},
                    ],
                    "question_kind": "computational",
                    "geometry": {"present": True, "type": "midpoint_1d"},
                },
                0.0,
            ),
            (
                {
                    "question": "Four identical charges are placed at the four vertices of a square. A test charge is placed at the center of the square. What is the net electric force on the test charge?",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F_net", "unit": "N"},
                    "givens": [],
                    "question_kind": "computational",
                    "geometry": {"present": True, "type": "square_center"},
                },
                0.0,
            ),
            (
                {
                    "question": "Two identical point charges q are 12 cm apart in air and repel each other with a force of 3.6 N. Find q.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "q", "unit": "C"},
                    "givens": [
                        {"symbol": "F", "si_value": 3.6, "si_unit": "N"},
                        {"symbol": "r", "si_value": 0.12, "si_unit": "m"},
                    ],
                    "question_kind": "computational",
                },
                2.40166e-6,
            ),
            (
                {
                    "question": "A series AC circuit has resistance R = 40 Ohm and net reactance |XL - XC| = 30 Ohm. Calculate the power factor.",
                    "domain": "Alternating-Current Circuits",
                    "target": {"symbol": "power_factor", "unit": ""},
                    "givens": [
                        {"symbol": "R", "si_value": 40.0, "si_unit": "Ohm"},
                        {"symbol": "X_net", "si_value": 30.0, "si_unit": "Ohm"},
                    ],
                    "question_kind": "computational",
                },
                0.8,
            ),
        ]

        for parsed, expected in cases:
            with self.subTest(question=parsed["question"]):
                solution = deterministic_solution(parsed)
                self.assertIsNotNone(solution)
                computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
                value = computed["verified_output"]["final_answer"]["value"]
                self.assertNotIn("status", computed["result"])
                if isinstance(expected, dict):
                    self.assertIsInstance(value, dict)
                    for key, expected_value in expected.items():
                        self.assertAlmostEqual(value[key], expected_value, delta=max(abs(expected_value) * 1e-6, 1e-9))
                    continue
                actual = value["magnitude"] if isinstance(value, dict) and "magnitude" in value else value
                self.assertAlmostEqual(actual, expected, delta=max(abs(expected) * 1e-3, 1e-9))

    def test_solution_provider_falls_back_to_deterministic_coulomb_when_json_repair_fails(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution

        parsed = {
            "question": "Two charges q1 = +4 microC and q2 = +4 microC are 10 cm apart. A charge q0 = -2 microC lies between them, 4 cm from q1 and 6 cm from q2. Find the magnitude of the net force on q0.",
            "domain": "Electric Charges and Fields",
            "target": {"symbol": "F_net_magnitude", "unit": "N"},
            "givens": [
                {"symbol": "q1", "si_value": 4e-6},
                {"symbol": "q2", "si_value": 4e-6},
                {"symbol": "q0", "si_value": -2e-6},
                {"symbol": "AB", "si_value": 0.10},
                {"symbol": "AM", "si_value": 0.04},
                {"symbol": "BM", "si_value": 0.06},
            ],
            "relations": ["q0 lies between q1 and q2"],
            "question_kind": "computational",
            "answer_format": {"requested_form": "magnitude"},
            "geometry": {"present": True, "type": "collinear", "line_order": ["A", "M", "B"]},
        }
        llm = RecordingLLM({"physics.solution": "not json", "physics.solution.retry": "still not json", "physics.solution.repair": "still not json"})
        provider = LLMSolutionProvider(llm)

        deterministic = deterministic_solution(parsed)
        output = provider._request_solution(provider._build_prompt(parsed), parsed, deterministic)

        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution", "physics.solution.retry", "physics.solution.repair"])
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": output, "errors": []})
        value = computed["verified_output"]["final_answer"]["value"]
        self.assertAlmostEqual(value["magnitude"], 25.0)


class FormattingRegressionTests(unittest.TestCase):
    def test_format_number_keeps_tiny_nonzero_values(self) -> None:
        from agents.formatting import format_number

        self.assertEqual(format_number(9.34128e-10), "9.34128e-10")


class SolutionProviderRegressionTests(unittest.TestCase):
    def test_formula_registry_and_legacy_import_support_core_domains(self) -> None:
        from agents.physics.Solution.formula_lib import deterministic_solution as legacy_deterministic_solution
        from agents.physics.formulas.registry import deterministic_solution as registry_deterministic_solution

        cases = [
            (
                {
                    "question": "A capacitor has capacitance C = 4 microF and voltage U = 50 V. Calculate energy.",
                    "domain": "Capacitance",
                    "target": {"symbol": "W", "unit": "J"},
                    "givens": [{"symbol": "C", "si_value": 4e-6}, {"symbol": "U", "si_value": 50}],
                },
                "capacitance.stored_energy",
                "W",
            ),
            (
                {
                    "question": "Two charges q1 and q2 separated by r. Calculate force.",
                    "domain": "Electric Charges and Fields",
                    "target": {"symbol": "F", "unit": "N"},
                    "givens": [
                        {"symbol": "q1", "si_value": 2e-6},
                        {"symbol": "q2", "si_value": 3e-6},
                        {"symbol": "r", "si_value": 0.1},
                    ],
                },
                "electrostatics.coulomb_two_charge_force",
                "F",
            ),
            (
                {
                    "question": "A series RLC circuit has L and C. Find resonant frequency.",
                    "domain": "Alternating-Current Circuits",
                    "target": {"symbol": "f_res", "unit": "Hz"},
                    "givens": [{"symbol": "L", "si_value": 0.2}, {"symbol": "C", "si_value": 40e-6}],
                },
                "ac.resonance.frequency",
                "f_res",
            ),
            (
                {
                    "question": "The true value is 30.0 cm, the measured result is 29.7 cm. Calculate the absolute error and relative error.",
                    "domain": "Measurements",
                    "target": {"symbol": "percentage_relative_error", "unit": "%"},
                    "givens": [{"symbol": "true_value", "si_value": 30.0}, {"symbol": "measured_result", "si_value": 29.7}],
                },
                "measurement.absolute_and_relative_error",
                "percentage_relative_error",
            ),
            (
                {
                    "question": "A solenoid has N = 1000 turns, current I = 2 A and length ell = 0.5 m. Calculate magnetic field.",
                    "domain": "Sources of Magnetic Fields",
                    "target": {"symbol": "B", "unit": "T"},
                    "givens": [
                        {"symbol": "N", "si_value": 1000},
                        {"symbol": "I", "si_value": 2},
                        {"symbol": "ell", "si_value": 0.5},
                    ],
                },
                "magnetism.solenoid_magnetic_field",
                "B",
            ),
        ]

        for parsed, expected_formula_id, expected_target in cases:
            with self.subTest(formula_id=expected_formula_id):
                legacy_solution = legacy_deterministic_solution(parsed)
                registry_solution = registry_deterministic_solution(parsed)
                self.assertIsNotNone(legacy_solution)
                self.assertIsNotNone(registry_solution)
                self.assertEqual(registry_solution, legacy_solution)
                self.assertIn(expected_formula_id, registry_solution["formula_ids"])
                self.assertEqual(registry_solution["sympy_spec"]["target_symbol"], expected_target)

    def test_solution_provider_retargets_undefined_mean_and_mae_target(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "mean_and_mae",
                            "target_unit": "degC",
                            "equations": [
                                "mean = (x1 + x2 + x3) / 3",
                                "mean_absolute_error = (Abs(x1 - mean) + Abs(x2 - mean) + Abs(x3 - mean)) / 3",
                            ],
                            "known_values": {"x1": 36.6, "x2": 36.8, "x3": 36.7},
                        },
                        "solution_steps": [],
                    }
                )
            }
        )
        provider = LLMSolutionProvider(llm)
        output = provider.get_solution(
            "Three temperature measurements. Calculate mean and mean absolute error.",
            {
                "question": "Three temperature measurements: 36.6C; 36.8C; 36.7C. Calculate the mean and the mean absolute error.",
                "target": {"symbol": "mean_and_mae", "unit": "degC"},
                "givens": [
                    {"symbol": "x1", "si_value": 36.6},
                    {"symbol": "x2", "si_value": 36.8},
                    {"symbol": "x3", "si_value": 36.7},
                ],
                "question_kind": "computational",
            },
        )
        self.assertEqual(output["sympy_spec"]["target_symbol"], "mean_absolute_error")

    def test_solution_provider_normalizes_expression_without_equals_to_equation(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "q1",
                            "target_unit": "C",
                            "equations": ["q1 + q2 - 7e-08", "q1 / r1**2 + q2 / r2**2 = 0"],
                            "known_values": {"r1": 0.06, "r2": 0.08},
                        },
                        "solution_steps": [],
                    }
                )
            }
        )
        provider = LLMSolutionProvider(llm)
        output = provider.get_solution(
            "Find q1.",
            {
                "question": "Given q1 + q2 = 7x10^-8 and E=0 at M with distances 6 cm and 8 cm, find q1.",
                "target": {"symbol": "q1", "unit": "C"},
                "givens": [{"symbol": "r1", "si_value": 0.06}, {"symbol": "r2", "si_value": 0.08}],
                "question_kind": "computational",
            },
        )
        self.assertIn("q1 + q2 - 7e-08 = 0", output["sympy_spec"]["equations"])

    def test_solution_provider_accepts_acos_equations(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": json.dumps(
                    {
                        "mode": "computational",
                        "answer_type": "numeric",
                        "sympy_spec": {
                            "target_symbol": "theta",
                            "target_unit": "rad",
                            "equations": [
                                "cos(theta) = (F1**2 + F2**2 - F_resultant**2) / (2 * F1 * F2)",
                                "theta = acos((F1**2 + F2**2 - F_resultant**2) / (2 * F1 * F2))",
                            ],
                            "known_values": {"F1": 3.0, "F2": 4.0, "F_resultant": 5.0},
                        },
                        "solution_steps": [],
                    }
                )
            }
        )
        provider = LLMSolutionProvider(llm)
        output = provider.get_solution(
            "Find angle theta.",
            {
                "question": "Given F1=3 N, F2=4 N and resultant force 5 N, find theta.",
                "target": {"symbol": "theta", "unit": "rad"},
                "givens": [
                    {"symbol": "F1", "si_value": 3.0},
                    {"symbol": "F2", "si_value": 4.0},
                    {"symbol": "F_resultant", "si_value": 5.0},
                ],
                "question_kind": "computational",
            },
        )
        self.assertIn("acos(", " ".join(output["sympy_spec"]["equations"]))


class LogicWorkflowTests(unittest.TestCase):
    def test_logic_xai_pipeline_returns_public_evidence(self) -> None:
        llm = RecordingLLM(
            {
                "logic.xai.classify": "YesNo",
                "logic.xai.plan": "Check whether P holds for A, then apply the implication from P to Q.",
                "logic.xai.execute": (
                    "Step 1: Premise 2 states that P holds for A.\n"
                    "Step 2: Premise 1 states that P implies Q.\n"
                    "Final answer: Yes\n"
                    "idx: [1, 2]\n"
                    "explanation: Premise 2 states that P holds for A. Premise 1 states that P implies Q, so Q(A) follows."
                ),
            }
        )
        graph = ExactGraph(llm=llm, classifier=StaticClassifier("logic"), use_rag=False)
        output = graph.predict(
            {"question": "Is Q(A) true?", "premises": ["P implies Q.", "P holds for A."]}
        )
        self.assertEqual(set(output), {"answer", "explanation", "idx", "cot", "premises"})
        self.assertEqual(output["answer"], "Yes")
        self.assertEqual(output["idx"], [1, 2])
        self.assertTrue(output["cot"])
        self.assertEqual([call["stage"] for call in llm.calls], ["logic.xai.classify", "logic.xai.plan", "logic.xai.execute"])
        classify_prompt = llm.calls[0]["messages"][1]["content"]
        plan_prompt = llm.calls[1]["messages"][1]["content"]
        solver_prompt = llm.calls[2]["messages"][1]["content"]
        self.assertIn("Question: Is Q(A) true?", classify_prompt)
        self.assertIn("P implies Q.", plan_prompt)
        self.assertIn("Reasoning plan to follow", solver_prompt)

    def test_logic_without_llm_returns_unknown(self) -> None:
        graph = ExactGraph(llm=None, classifier=StaticClassifier("logic"), use_rag=False)
        output = graph.predict(
            {
                "question": "Is prepared true?",
                "premises": ["All diligent are prepared."],
            }
        )
        self.assertEqual(output["answer"], "Unknown")
        self.assertNotIn("fol", output)
        self.assertTrue(any("requires a configured LLM" in step for step in output["cot"]))

    def test_logic_agent_uses_symbcot_provider(self) -> None:
        llm = RecordingLLM(
            {
                "logic.xai.classify": "YesNo",
                "logic.xai.plan": "Check the explicit negation.",
                "logic.xai.execute": (
                    "Final answer: No\n"
                    "idx: [1]\n"
                    "explanation: Premise 1 explicitly blocks the queried condition."
                ),
            }
        )
        agent = LogicAgent(llm=llm, use_rag=False)
        output = agent.solve(
            question="Is prepared true?",
            premises_nl=["The student is not prepared."],
        )
        self.assertEqual(output["answer"], "No")
        self.assertEqual(output["idx"], [1])
        self.assertEqual([call["stage"] for call in llm.calls], ["logic.xai.classify", "logic.xai.plan", "logic.xai.execute"])

    def test_logic_xai_pipeline_does_not_inject_rag_examples(self) -> None:
        llm = RecordingLLM(
            {
                "logic.xai.classify": "YesNo",
                "logic.xai.plan": "Use the rule in premise 1.",
                "logic.xai.execute": (
                    "Final answer: Yes\n"
                    "idx: [1]\n"
                    "explanation: Premise 1 supports the answer."
                ),
            }
        )
        graph = ExactGraph(llm=llm, classifier=StaticClassifier("logic"))
        output = graph.predict(
            {
                "question": "Is careful true?",
                "premises": ["All coders are careful."],
            }
        )

        prompts = "\n".join(call["messages"][1]["content"] for call in llm.calls)
        self.assertNotIn("few-shot", prompts.lower())
        self.assertIn("All coders are careful.", prompts)
        self.assertEqual(output["answer"], "Yes")
        self.assertNotIn("rag_used", output)
        self.assertNotIn("confidence", output)


class P1EvaluatorTests(unittest.TestCase):
    def test_logic_answer_aliases_follow_zip_scoring(self) -> None:
        self.assertTrue(answer_match("true", "Yes"))
        self.assertTrue(answer_match("cannot be determined", "Unknown"))
        self.assertTrue(answer_match("a", "A"))
        result = evaluate_prediction("false", "No")
        self.assertTrue(result["correct"])
        self.assertEqual(result["method"], "normalized_answer_exact")

    def test_physics_numeric_scoring_uses_zip_tolerance_and_units(self) -> None:
        self.assertTrue(answer_match("100.05 V", "100", "v", "volt"))
        self.assertFalse(answer_match("101 V", "100", "v", "volt"))
        result = evaluate_prediction("1000 mV", "1", "v")
        self.assertTrue(result["correct"])
        self.assertEqual(result["method"], "numeric")

    def test_summary_exposes_zip_p1_and_existing_accuracy_key(self) -> None:
        summary = summarize_p1(
            [
                {"task": "logic", "correct": True},
                {"task": "logic", "correct": False},
                {"task": "physics", "correct": True},
            ]
        )
        self.assertEqual(summary["p1"], 2 / 3)
        self.assertEqual(summary["p1_accuracy"], 2 / 3)
        self.assertEqual(summary["by_task"]["logic"]["p1"], 0.5)


class ApiContractTests(unittest.TestCase):
    def test_predict_reports_workflow_failures_as_500(self) -> None:
        graph = ExactGraph(llm=None, classifier=StaticClassifier("physics"))
        api.app.dependency_overrides[api.get_graph] = lambda: graph
        try:
            response = TestClient(api.app).post("/predict", json={"question": "Calculate energy."})
        finally:
            api.app.dependency_overrides.clear()
        self.assertEqual(response.status_code, 500)
        self.assertIn("Physics ParsingAgent requires a configured LLM.", response.json()["detail"])

    def test_predict_accepts_direct_conceptual_physics(self) -> None:
        llm = RecordingLLM(
            {
                "physics.parsing": json.dumps(
                    {
                        "question": "Why is the midpoint field zero?",
                        "domain": "Electric Charges and Fields",
                        "target": {"symbol": "answer", "unit": ""},
                        "givens": [],
                        "relations": [],
                        "question_kind": "conceptual",
                    }
                ),
                "physics.solution": json.dumps(
                    {
                        "mode": "direct",
                        "answer_type": "conceptual",
                        "direct_answer": {
                            "answer": "The net field is zero by symmetry.",
                            "rationale_steps": ["The equal fields have opposite directions."],
                        },
                    }
                ),
                "physics.explanation": json.dumps(
                    {
                        "answer": {
                            "symbol": "answer",
                            "value": "The net field is zero by symmetry.",
                            "unit": "",
                        },
                        "explanation": "The equal fields have opposite directions, so the net field is zero.",
                    }
                ),
            }
        )
        graph = ExactGraph(llm=llm, classifier=StaticClassifier("physics"))
        api.app.dependency_overrides[api.get_graph] = lambda: graph
        try:
            response = TestClient(api.app).post("/predict", json={"question": "Why is the midpoint field zero?"})
        finally:
            api.app.dependency_overrides.clear()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "The net field is zero by symmetry.")

    def test_predict_accepts_conceptual_physics_after_parser_repair(self) -> None:
        llm = RecordingLLM(
            {
                "physics.parsing": "Doubling the turns doubles the magnetic field.",
                "physics.parsing.retry": "still not json",
                "physics.parsing.repair": json.dumps(
                    {
                        "question": "If you double the number of turns of a solenoid, but keep its length and current the same, how does the magnetic field change?",
                        "domain": "Sources of Magnetic Fields",
                        "target": {"symbol": "answer", "unit": ""},
                        "givens": [],
                        "relations": ["length is unchanged", "current is unchanged", "number of turns is doubled"],
                        "question_kind": "conceptual",
                    }
                ),
                "physics.solution": json.dumps(
                    {
                        "mode": "direct",
                        "answer_type": "conceptual",
                        "direct_answer": {
                            "answer": "The magnetic field doubles.",
                            "rationale_steps": ["For a solenoid, B is proportional to the number of turns per unit length."],
                        },
                    }
                ),
                "physics.explanation": json.dumps(
                    {
                        "answer": {"symbol": "answer", "value": "The magnetic field doubles.", "unit": ""},
                        "explanation": "The magnetic field doubles because the turns per unit length doubles.",
                    }
                ),
            }
        )
        graph = ExactGraph(llm=llm, classifier=StaticClassifier("physics"))
        api.app.dependency_overrides[api.get_graph] = lambda: graph
        try:
            response = TestClient(api.app).post(
                "/predict",
                json={
                    "question": "If you double the number of turns of a solenoid, but keep its length and current the same, how does the magnetic field change?"
                },
            )
        finally:
            api.app.dependency_overrides.clear()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "The magnetic field doubles.")

    def test_predict_accepts_direct_yes_no_physics(self) -> None:
        llm = RecordingLLM(
            {
                "physics.parsing": json.dumps(
                    {
                        "question": "Is the net field zero at the midpoint?",
                        "domain": "Electric Charges and Fields",
                        "target": {"symbol": "answer", "unit": ""},
                        "givens": [],
                        "relations": [],
                        "question_kind": "conceptual",
                    }
                ),
                "physics.solution": json.dumps(
                    {
                        "mode": "direct",
                        "answer_type": "yes_no",
                        "direct_answer": {
                            "answer": "Yes",
                            "rationale_steps": ["The equal fields cancel at the midpoint."],
                        },
                    }
                ),
                "physics.explanation": json.dumps(
                    {
                        "answer": {"symbol": "answer", "value": "Yes", "unit": ""},
                        "explanation": "Yes. The equal fields cancel at the midpoint.",
                    }
                ),
            }
        )
        graph = ExactGraph(llm=llm, classifier=StaticClassifier("physics"))
        api.app.dependency_overrides[api.get_graph] = lambda: graph
        try:
            response = TestClient(api.app).post("/predict", json={"question": "Is the net field zero at the midpoint?"})
        finally:
            api.app.dependency_overrides.clear()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Yes")

    def test_predict_rejects_removed_advanced_fields(self) -> None:
        response = TestClient(api.app).post(
            "/predict",
            json={"question": "Calculate energy.", "parsed_question": {}},
        )
        self.assertEqual(response.status_code, 422)

    def test_operational_endpoints_do_not_publish_tracing_configuration(self) -> None:
        client = TestClient(api.app)
        for path in ("/health", "/info"):
            payload = client.get(path).json()
            self.assertNotIn("tracing", payload)
            self.assertNotIn("metadata", payload)


class TracingFilterTests(unittest.TestCase):
    def test_step_trace_processor_strips_disallowed_diagnostics(self) -> None:
        processed = _process_step_outputs(
            {
                "output": {
                    "answer": "Yes",
                    "metadata": {"raw_type": "2"},
                    "confidence": 0.8,
                    "tracing": {"provider": "langsmith"},
                }
            }
        )
        serialized = json.dumps(processed)
        for blocked in ("metadata", "raw_type", "confidence", "tracing"):
            self.assertNotIn(blocked, serialized)

    def test_llm_trace_inputs_store_size_not_prompt(self) -> None:
        client = OpenRouterClient(model="test", api_key_env="")
        inputs = _process_llm_inputs(
            {
                "self": client,
                "stage": "physics.parsing",
                "attempt": 1,
                "payload": {
                    "messages": [{"role": "user", "content": "secret prompt"}],
                    "response_format": {"type": "json_object"},
                },
            }
        )
        self.assertEqual(inputs["request_chars"], len("secret prompt"))
        self.assertNotIn("secret prompt", json.dumps(inputs))
        self.assertEqual(_clean({"source": "hidden", "answer": "ok"}), {"answer": "ok"})


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self.payload = payload

    def json(self) -> dict[str, object]:
        return self.payload


class OpenRouterTests(unittest.TestCase):
    def _client(self) -> OpenRouterClient:
        client = OpenRouterClient(model="qwen/qwen-2.5-7b-instruct", api_key_env="")
        client.require_api_key = False
        return client

    def test_json_request_routes_by_parameters_then_retries_prompt_only(self) -> None:
        client = self._client()
        payloads: list[dict[str, object]] = []

        def post(url: str, headers: dict[str, str], json: dict[str, object], timeout: float) -> FakeResponse:
            del url, headers, timeout
            payloads.append(dict(json))
            if len(payloads) == 1:
                return FakeResponse(400, {"error": {"message": "Provider returned error"}})
            return FakeResponse(200, {"choices": [{"message": {"content": "{}"}}]})

        client._session.post = post
        self.assertEqual(
            client.chat(
                [{"role": "user", "content": "json"}],
                response_format={"type": "json_object"},
                stage="physics.parsing",
            ),
            "{}",
        )
        self.assertEqual(payloads[0]["provider"], {"require_parameters": True})
        self.assertIn("response_format", payloads[0])
        self.assertNotIn("provider", payloads[1])
        self.assertNotIn("response_format", payloads[1])

    def test_invalid_model_does_not_retry(self) -> None:
        client = self._client()
        calls: list[int] = []

        def post(url: str, headers: dict[str, str], json: dict[str, object], timeout: float) -> FakeResponse:
            del url, headers, json, timeout
            calls.append(1)
            return FakeResponse(400, {"error": {"message": "not a valid model ID"}})

        client._session.post = post
        with self.assertRaises(RuntimeError):
            client.chat([{"role": "user", "content": "json"}], response_format={"type": "json_object"})
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
