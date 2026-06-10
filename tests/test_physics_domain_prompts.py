from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agents.physics.Solution.llm_provider import DOMAIN_PROMPT_FILES, LLMSolutionProvider
from agents.physics.Solution.rag_provider import RAGSolutionProvider
from tools.calculator import solve_with_sympy_trace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAIN_PROMPT_DIR = PROJECT_ROOT / "prompts" / "physics_solution_domains"


class DummyLLM:
    enabled = True
    provider = "test"
    model = "test-model"


class RecordingLLM(DummyLLM):
    def __init__(self, responses: dict[str, Any]) -> None:
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
        payload = self.responses[stage]
        if isinstance(payload, list):
            payload = payload.pop(0)
        return json.dumps(payload)


class PhysicsDomainPromptTests(unittest.TestCase):
    def test_all_configured_domain_prompts_exist_and_have_few_shots(self) -> None:
        unique_files = sorted(set(DOMAIN_PROMPT_FILES.values()))
        self.assertGreaterEqual(len(unique_files), 13)
        for file_name in unique_files:
            with self.subTest(file_name=file_name):
                path = DOMAIN_PROMPT_DIR / file_name
                self.assertTrue(path.exists(), f"Missing domain prompt: {file_name}")
                text = path.read_text(encoding="utf-8")
                self.assertIn("{{RAG_HINTS}}", text)
                self.assertIn("{{DETERMINISTIC_HINTS}}", text)
                self.assertIn("{{PARSED_QUESTION}}", text)
                self.assertGreaterEqual(text.count("Example "), 2)

    def test_solution_provider_selects_prompt_from_parsed_domain(self) -> None:
        provider = LLMSolutionProvider(DummyLLM())
        cases = {
            "Electric Charges and Fields": "electric_charges_and_fields.md",
            "Gauss's Law": "gausss_law.md",
            "Electric Potential": "electric_potential.md",
            "Capacitance": "capacitance.md",
            "Current and Resistance": "current_and_resistance.md",
            "Direct-Current Circuits": "direct_current_circuits.md",
            "Magnetic Forces and Fields": "magnetic_forces_and_fields.md",
            "Sources of Magnetic Fields": "sources_of_magnetic_fields.md",
            "Electromagnetic Induction": "electromagnetic_induction.md",
            "Inductance": "inductance.md",
            "Alternating-Current Circuits": "alternating_current_circuits.md",
            "Electromagnetic Waves": "electromagnetic_waves.md",
            "Measurement and Uncertainty": "measurement_and_uncertainty.md",
        }
        for domain, expected_file in cases.items():
            with self.subTest(domain=domain):
                prompt = provider._build_prompt(
                    {
                        "domain": domain,
                        "target": {"symbol": "x", "unit": ""},
                        "givens": [],
                        "relations": [],
                        "question_kind": "computational",
                    }
                )
                self.assertIn("Output JSON:", prompt)
                self.assertEqual(provider.last_prompt_diagnostics["selected_rule_pack"], ["domain", expected_file])
                self.assertTrue(provider.last_prompt_diagnostics["selected_prompt"].endswith(expected_file))

    def test_solution_provider_falls_back_for_unknown_domain(self) -> None:
        provider = LLMSolutionProvider(DummyLLM())
        provider._build_prompt({"domain": "Unknown Domain", "target": {"symbol": "x", "unit": ""}})
        self.assertEqual(provider.last_prompt_diagnostics["selected_rule_pack"], ["fallback", "solution_type2.md"])

    def test_get_solution_runs_convert_to_sympy_before_returning(self) -> None:
        solution_draft = {
            "mode": "computational",
            "answer_type": "numeric",
            "sympy_spec": {
                "target_symbol": "W_C",
                "target_unit": "J",
                "equations": ["W_C = 1/2 C U^2"],
                "known_values": {"C": "100 uF", "U": "30 V"},
            },
        }
        converted_solution = {
            "mode": "computational",
            "answer_type": "numeric",
            "sympy_spec": {
                "target_symbol": "W_C",
                "target_unit": "J",
                "equations": ["W_C = C*U**2/2"],
                "known_values": {"C": 0.0001, "U": 30},
            },
            "solution_steps": ["Use parser-normalized values."],
        }
        llm = RecordingLLM(
            {
                "physics.solution": solution_draft,
                "physics.solution.convert_to_sympy": converted_solution,
            }
        )
        provider = LLMSolutionProvider(llm)
        parsed = {
            "domain": "Capacitance",
            "target": {"symbol": "W_C", "unit": "J"},
            "givens": [
                {"symbol": "C", "si_value": 0.0001, "si_unit": "F"},
                {"symbol": "U", "si_value": 30, "si_unit": "V"},
            ],
            "question_kind": "computational",
        }

        result = provider.get_solution("Find energy.", parsed)

        self.assertEqual(result, converted_solution)
        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution", "physics.solution.convert_to_sympy"])
        self.assertTrue(provider.last_prompt_diagnostics["used_convert_to_sympy"])
        self.assertFalse(provider.last_prompt_diagnostics["used_solution_parser_verifier"])
        self.assertTrue(provider.last_prompt_diagnostics["convert_prompt"].endswith("convert_to_sympy.md"))

    def test_solution_provider_verifies_only_when_solution_disagrees_with_parser(self) -> None:
        solution_draft = {
            "mode": "computational",
            "answer_type": "numeric",
            "sympy_spec": {
                "target_symbol": "W_R",
                "target_unit": "J",
                "equations": ["W_R = C*U**2/2"],
                "known_values": {"C": 0.0001, "U": 30},
            },
        }
        verified_solution = {
            "mode": "computational",
            "answer_type": "numeric",
            "sympy_spec": {
                "target_symbol": "W_C",
                "target_unit": "J",
                "equations": ["W_C = C*U**2/2"],
                "known_values": {"C": 0.0001, "U": 30},
            },
            "solution_steps": ["Repair target mismatch with parsed_question."],
        }
        llm = RecordingLLM(
            {
                "physics.solution": solution_draft,
                "physics.solution.convert_to_sympy": [solution_draft, verified_solution],
                "physics.solution.verify_against_parser": verified_solution,
            }
        )
        provider = LLMSolutionProvider(llm)
        parsed = {
            "domain": "Capacitance",
            "target": {"symbol": "W_C", "unit": "J"},
            "givens": [
                {"symbol": "C", "si_value": 0.0001, "si_unit": "F"},
                {"symbol": "U", "si_value": 30, "si_unit": "V"},
            ],
            "question_kind": "computational",
        }

        result = provider.get_solution("Find capacitor energy.", parsed)

        self.assertEqual(result, verified_solution)
        self.assertEqual(
            [call["stage"] for call in llm.calls],
            [
                "physics.solution",
                "physics.solution.convert_to_sympy",
                "physics.solution.verify_against_parser",
                "physics.solution.convert_to_sympy",
            ],
        )
        self.assertTrue(provider.last_prompt_diagnostics["used_solution_parser_verifier"])
        self.assertIn("target_symbol", " ".join(provider.last_prompt_diagnostics["solution_parser_mismatches"]))

    def test_rag_solution_provider_also_runs_convert_to_sympy(self) -> None:
        solution_draft = {
            "mode": "computational",
            "answer_type": "numeric",
            "sympy_spec": {
                "target_symbol": "I",
                "target_unit": "A",
                "equations": ["I = U/R"],
                "known_values": {"U": 10, "R": 5},
            },
        }
        converted_solution = {
            "mode": "computational",
            "answer_type": "numeric",
            "sympy_spec": {
                "target_symbol": "I",
                "target_unit": "A",
                "equations": ["I = U/R"],
                "known_values": {"U": 10, "R": 5},
            },
            "solution_steps": ["Use Ohm's law."],
        }
        llm = RecordingLLM(
            {
                "physics.solution": solution_draft,
                "physics.solution.convert_to_sympy": converted_solution,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            kb_path = Path(directory) / "kb.json"
            kb_path.write_text(json.dumps([{"question": "Find current through resistor.", "answer": "2 A"}]), encoding="utf-8")
            provider = RAGSolutionProvider(llm, kb_path=kb_path, top_k=1)
            parsed = {
                "domain": "Direct-Current Circuits",
                "target": {"symbol": "I", "unit": "A"},
                "givens": [{"symbol": "U", "si_value": 10}, {"symbol": "R", "si_value": 5}],
                "question_kind": "computational",
            }
            result = provider.get_solution("Find current through resistor.", parsed)

        self.assertEqual(result, converted_solution)
        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution", "physics.solution.convert_to_sympy"])

    def test_direct_solution_skips_convert_to_sympy(self) -> None:
        direct_solution = {
            "mode": "direct",
            "answer_type": "conceptual",
            "direct_answer": {"answer": "Electric field is a vector field.", "selected_option": None, "rationale_steps": []},
        }
        llm = RecordingLLM({"physics.solution": direct_solution})
        provider = LLMSolutionProvider(llm)
        parsed = {"domain": "Electric Charges and Fields", "target": {"symbol": "answer", "unit": ""}}

        result = provider.get_solution("What is an electric field?", parsed)

        self.assertEqual(result, direct_solution)
        self.assertEqual([call["stage"] for call in llm.calls], ["physics.solution"])
        self.assertEqual(provider.last_prompt_diagnostics["skipped_convert_to_sympy"], "direct_mode")

    def test_sympy_executor_supports_converter_allowed_functions(self) -> None:
        result = solve_with_sympy_trace(
            {"Z_total": 5, "x": 0.5},
            ["P = Im(conjugate(Z_total))", "angle = asin(x)"],
            "angle",
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.value, 0.5235987755982989)


if __name__ == "__main__":
    unittest.main()
