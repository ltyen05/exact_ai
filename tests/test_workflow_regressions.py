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
from agents.llm.openrouter_provider import OpenRouterClient
from agents.physics.Parsing import ParsingAgent
from agents.physics.Solution.llm_provider import LLMSolutionProvider
from agents.physics.Solution.rag_provider import RAGSolutionProvider
from agents.workflows.orchestrator import ExactGraph, FormatterNode, WorkflowExecutionError
from agents.workflows.physics import PhysicsWorkflow
from agents.workflows.tracing import _clean, _process_llm_inputs, _process_step_outputs
from tools.calculator import solve_with_sympy_trace


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
        self.assertEqual([call["stage"] for call in llm.calls], ["physics.parsing", "physics.parsing.repair"])
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
                "physics.parsing.repair": "still not json",
                "physics.parsing.repair2": "still not json",
            }
        )
        with self.assertRaisesRegex(ValueError, "Physics parser response must be a JSON object"):
            ParsingAgent(llm_provider=llm).run("Conceptual question")

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
        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution", "physics.solution.repair"])
        self.assertIn("r1 = AM", output["sympy_spec"]["equations"])

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

    def test_solution_provider_falls_back_to_deterministic_on_non_json(self) -> None:
        llm = RecordingLLM(
            {
                "physics.solution": "not json",
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
        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution"])
        self.assertEqual(output["formula_ids"], ["ac.resonance.frequency"])

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
        self.assertIn("electric", diagnostics["selected_rule_pack"])
        self.assertIn("perpendicular_bisector", diagnostics["selected_rule_pack"])
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

    def test_solution_provider_falls_back_to_deterministic_ac_formula_after_invalid_llm_spec(self) -> None:
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
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
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
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
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
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
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
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertEqual(solution["formula_ids"], ["ac.resonance.power_from_voltage_resistance"])
        self.assertAlmostEqual(computed["verified_output"]["final_answer"]["value"], 360)

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
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})

        self.assertEqual(solution["formula_ids"], ["ac.resonance.inductance_from_frequency_capacitance"])
        self.assertAlmostEqual(
            computed["verified_output"]["final_answer"]["value"],
            1 / (4 * math.pi**2 * 50**2 * 200e-6),
        )

    def test_frequency_aliases_accept_extra_known_values(self) -> None:
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
        computed = PhysicsWorkflow().compute_sympy({"parsed_question": parsed, "solution_output": solution, "errors": []})
        self.assertAlmostEqual(
            computed["verified_output"]["final_answer"]["value"],
            1 / math.sqrt(0.5 * 2e-6),
        )

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
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
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
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
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
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
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
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
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
        solution = LLMSolutionProvider(RecordingLLM({})).get_solution(parsed["question"], parsed)
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
        self.assertEqual(output["premises"], ["E = C * U**2 / 2"])

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
        self.assertNotIn("cot", formatted)

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

    def test_rag_documents_are_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            kb_path = Path(directory) / "kb.json"
            kb_path.write_text(json.dumps([{"question": "first", "cot": "a"}]), encoding="utf-8")
            provider = RAGSolutionProvider(RecordingLLM({}), kb_path=kb_path, top_k=1)
            self.assertEqual(provider.retrieve("first")[0]["question"], "first")
            kb_path.write_text(json.dumps([{"question": "second", "cot": "b"}]), encoding="utf-8")
            self.assertEqual(provider.retrieve("first")[0]["question"], "first")

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


class LogicWorkflowTests(unittest.TestCase):
    def test_logic_prompt_builds_horn_kb_and_public_evidence(self) -> None:
        llm = RecordingLLM(
            {
                "logic.formalize": json.dumps(
                    {
                        "facts": [{"pred": "p", "args": ["a"], "truth": True, "premise_id": 2}],
                        "rules": [
                            {
                                "if": [{"pred": "p", "args": ["x"], "truth": True}],
                                "then": {"pred": "q", "args": ["x"], "truth": True},
                                "premise_id": 1,
                            }
                        ],
                        "query": {"pred": "q", "args": ["a"], "truth": True},
                        "choices": {},
                    }
                ),
                "logic.explanation": "Because P(a) holds and P implies Q, Q(a) follows.",
            }
        )
        graph = ExactGraph(llm=llm, classifier=StaticClassifier("logic"))
        output = graph.predict(
            {"question": "Is Q(A) true?", "premises": ["P implies Q.", "P holds for A."]}
        )
        self.assertEqual(set(output), {"answer", "explanation", "fol", "cot", "premises"})
        self.assertEqual(output["answer"], "Yes")
        self.assertEqual(output["fol"], "q(a)")
        self.assertTrue(output["cot"])
        formalize_prompt = llm.calls[0]["messages"][0]["content"]
        self.assertIn("P implies Q.", formalize_prompt)


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
