from __future__ import annotations

import unittest
from pathlib import Path

from agents.physics.Solution.llm_provider import DOMAIN_PROMPT_FILES, LLMSolutionProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAIN_PROMPT_DIR = PROJECT_ROOT / "prompts" / "physics_solution_domains"


class DummyLLM:
    enabled = True
    provider = "test"
    model = "test-model"


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


if __name__ == "__main__":
    unittest.main()
