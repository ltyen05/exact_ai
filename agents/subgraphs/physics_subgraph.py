"""
Physics problem solver subgraph.

Pipeline:
1. ExtractQuantities - LLM extracts quantities, facts, objectives
2. ClassifyProblem - LLM classifies problem type (mechanics, electricity, etc)
3. SelectAbstract - Select problem abstract/template or use RAG fallback
4. Compute - SymPy calculation from formulas and facts
5. Explain - LLM generates detailed explanation
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph

from agents.llm.base import LLMClientBase
from agents.models.state import AgentOutput, WorkflowState
from agents.physics_agent import PhysicsRetriever
from agents.physics.templates import TEMPLATES, find_matching_template
from tools.calculator import solve_by_formula, solve_with_sympy
from agents.formatter import parse_float_like, format_number


class ExtractQuantitiesNode:
    """Extract quantities, facts, and objectives from physics problem."""

    def __init__(self, llm: LLMClientBase):
        self.llm = llm

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Extract structured data from problem statement."""
        if not self.llm.enabled:
            state.metadata["extract_quantities"] = {"source": "fallback", "data": {}}
            return state

        system = (
            "You are a physics problem analyzer. Extract key quantities, given facts, formulas, and target objective. "
            "Return JSON only:\n"
            "{\n"
            "  \"quantities\": [{\"symbol\": \"symbol_name\", \"value\": float_or_string, \"unit\": \"unit_name\"}],\n"
            "  \"objective\": \"target_symbol\",\n"
            "  \"objective_unit\": \"expected_unit\",\n"
            "  \"formulas\": [\"suggested_equation_1\", \"suggested_equation_2\"]\n"
            "}\n"
            "Be precise. Do not output anything else than JSON."
        )

        try:
            text = self.llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": state.question},
                ],
                temperature=0,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            data = json.loads(text)
            state.metadata["quantities"] = data.get("quantities", [])
            state.metadata["objective"] = data.get("objective", "")
            state.metadata["objective_unit"] = data.get("objective_unit", "")
            state.metadata["formulas"] = data.get("formulas", [])
            state.metadata["extract_quantities"] = {"source": "llm", "data": data}
        except Exception as e:
            state.add_error(f"ExtractQuantities error: {str(e)}")
            state.metadata["extract_quantities"] = {"source": "error", "error": str(e)}

        return state


class ClassifyProblemNode:
    """Classify the type of physics problem."""

    def __init__(self, llm: LLMClientBase):
        self.llm = llm

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Classify problem type."""
        if not self.llm.enabled:
            state.metadata["problem_type"] = "unknown"
            return state

        system = (
            "Classify this physics problem into ONE category. "
            "Return JSON: {type: 'mechanics'|'electricity'|'thermodynamics'|'waves'|'optics'|'modern'|'other', "
            "confidence: 0.0-1.0, reason: 'brief explanation'}. "
            "Base on problem content and keywords."
        )

        try:
            text = self.llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": state.question},
                ],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(text)
            state.metadata["problem_type"] = data.get("type", "unknown")
            state.metadata["problem_confidence"] = data.get("confidence", 0.0)
        except Exception as e:
            state.add_error(f"ClassifyProblem error: {str(e)}")
            state.metadata["problem_type"] = "unknown"

        return state


class SelectAbstractNode:
    """Select problem abstract if available, otherwise use RAG retrieval."""

    def __init__(self, kb_path: Optional[str] = None, use_rag: bool = True, use_abstract_templates: bool = True):
        self.retriever = PhysicsRetriever(kb_path) if kb_path and use_rag else None
        self.use_rag = use_rag
        self.use_abstract_templates = use_abstract_templates

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Try to find similar problem abstract or use RAG."""
        # 1. Try abstract formula templates first if enabled
        if self.use_abstract_templates:
            template_key = find_matching_template(state.question)
            if template_key:
                template = TEMPLATES[template_key]
                state.metadata["abstract_source"] = "template"
                state.metadata["abstract_template_key"] = template_key
                state.metadata["abstract_data"] = {
                    "formulas": template["formulas"],
                    "target": template["default_target"],
                    "approach": "Abstract Formula Template",
                    "unit": state.metadata.get("objective_unit") or ""
                }
                return state

        # 2. Fallback to RAG retrieval if enabled
        if self.retriever and self.use_rag:
            example, similarity = self.retriever.best(state.question)
            if example and similarity >= 0.75:
                state.metadata["abstract_source"] = "rag"
                state.metadata["abstract_example_id"] = example.get("id")
                state.metadata["abstract_similarity"] = similarity
                state.metadata["abstract_data"] = {
                    "cot": example.get("cot", []),
                    "unit": example.get("unit", ""),
                    "approach": "Retrieval-Augmented Generation (RAG)",
                }
                return state

        # No good match found
        state.metadata["abstract_source"] = "none"
        state.metadata["abstract_data"] = {"approach": "Direct computation"}
        return state


class ComputeNode:
    """Use SymPy to compute solution from formulas and facts."""

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Attempt symbolic computation using SymPy."""
        # Gather quantities
        quantities_dict: Dict[str, float] = {}
        raw_quantities = state.metadata.get("quantities", [])
        for q in raw_quantities:
            sym = q.get("symbol")
            val = q.get("value")
            if sym and val is not None:
                parsed_val = parse_float_like(val)
                if parsed_val is not None:
                    quantities_dict[sym] = parsed_val

        # Gather formulas and target
        formulas = []
        target = state.metadata.get("objective") or "F"
        
        abstract_data = state.metadata.get("abstract_data", {})
        
        # If template matched, use template equations
        if state.metadata.get("abstract_source") == "template":
            formulas = abstract_data.get("formulas", [])
            target = abstract_data.get("target") or target
        # Otherwise if LLM extracted formulas, use those
        elif "formulas" in state.metadata and state.metadata["formulas"]:
            formulas = state.metadata["formulas"]

        # Run SymPy Solver
        if quantities_dict and formulas:
            ans_val = solve_with_sympy(quantities_dict, formulas, target)
            if ans_val is not None:
                state.metadata["compute_source"] = "sympy"
                state.metadata["compute_result"] = {
                    "answer": format_number(ans_val),
                    "unit": abstract_data.get("unit") or state.metadata.get("objective_unit") or "",
                    "confidence": 0.95,
                    "cot": [
                        f"Target objective: {target}",
                        f"Constructed equations: {formulas}",
                        f"Extracted values: {quantities_dict}",
                        f"Symbolic calculation result: {ans_val}"
                    ],
                }
                state.metadata["formula_matched"] = True
                return state

        # Fallback to regex-based formula solver
        result = solve_by_formula(state.question)
        if result:
            state.metadata["compute_source"] = "formula_regex"
            state.metadata["compute_result"] = {
                "answer": result.answer,
                "unit": result.unit,
                "confidence": result.confidence,
                "cot": result.cot,
            }
            state.metadata["formula_matched"] = True
        else:
            state.metadata["compute_source"] = "none"
            state.metadata["formula_matched"] = False

        return state


class ExplainNode:
    """Generate detailed explanation using LLM with a dedicated prompt."""

    def __init__(self, llm: LLMClientBase):
        self.llm = llm

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Generate physics explanation."""
        if not self.llm.enabled:
            state.metadata["explanation_source"] = "unavailable"
            return state

        quantities = state.metadata.get("quantities", [])
        objective = state.metadata.get("objective", "")
        problem_type = state.metadata.get("problem_type", "")
        compute_result = state.metadata.get("compute_result", {})

        context = f"""
Question: {state.question}

Extracted Variables:
{json.dumps(quantities, indent=2)}

Target Variable: {objective}
Problem Category: {problem_type}

Calculation Output:
- Numeric Answer: {compute_result.get('answer', 'Not computed')}
- Unit: {compute_result.get('unit', '')}
- Steps: {compute_result.get('cot', [])}
"""

        # Separate educational prompt for physics agent
        system = (
            "You are a helpful physics tutor. Provide a clear, educational, step-by-step physics explanation. "
            "Start by listing the given quantities, identify the physical laws or formulas, "
            "show the substitution steps, and interpret the final result. "
            "Keep the explanation clean, mathematically sound, and concise. "
            "Explain in Vietnamese if the input query contains Vietnamese terms, otherwise explain in English."
        )

        try:
            explanation = self.llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": context},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            state.metadata["explanation"] = explanation
            state.metadata["explanation_source"] = "llm"
        except Exception as e:
            state.add_error(f"ExplainNode error: {str(e)}")
            state.metadata["explanation"] = ""
            state.metadata["explanation_source"] = "error"

        return state


def build_physics_subgraph(
    llm: LLMClientBase,
    kb_path: Optional[str] = None,
    use_abstract_templates: bool = True
) -> Any:
    """
    Build physics problem solving subgraph.
    
    Nodes:
    1. extract - Extract problem data
    2. classify - Classify problem type
    3. select_abstract - Find similar examples or templates
    4. compute - SymPy calculation
    5. explain - LLM explanation
    """
    graph = StateGraph(WorkflowState)

    # Create nodes
    extract_node = ExtractQuantitiesNode(llm)
    classify_node = ClassifyProblemNode(llm)
    select_node = SelectAbstractNode(kb_path=kb_path, use_rag=True, use_abstract_templates=use_abstract_templates)
    compute_node = ComputeNode()
    explain_node = ExplainNode(llm)

    # Add nodes to graph
    graph.add_node("extract", extract_node)
    graph.add_node("classify", classify_node)
    graph.add_node("select_abstract", select_node)
    graph.add_node("compute", compute_node)
    graph.add_node("explain", explain_node)

    # Set entry point
    graph.set_entry_point("extract")

    # Add edges (sequential pipeline)
    graph.add_edge("extract", "classify")
    graph.add_edge("classify", "select_abstract")
    graph.add_edge("select_abstract", "compute")
    graph.add_edge("compute", "explain")

    # Compile
    return graph.compile()
