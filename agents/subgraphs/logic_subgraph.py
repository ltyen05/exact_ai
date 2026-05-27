"""
Logic (Educational) problem solver subgraph.

Pipeline:
1. ExtractLogic - Extract premises, rules, and query
2. ConvertToFOL - Convert natural language to First Order Logic
3. VerifyWithZ3 - Use Z3 to verify logical implications
4. ExplainLogic - LLM generates educational explanation
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph

from agents.llm.base import LLMClientBase
from agents.models.state import AgentOutput, WorkflowState
from agents.logic_agent import LogicNLParserAgent, Z3ReasonerAgent
from tools.z3_logic import HornKB


class ExtractLogicNode:
    """Extract premises, rules, and query from problem."""

    def __init__(self, llm: LLMClientBase):
        self.llm = llm

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Extract structured logic data."""
        # Use existing premises if available
        state.metadata["premises_nl"] = state.premises_nl
        state.metadata["premises_fol"] = state.premises_fol
        state.metadata["query"] = state.question

        # Try LLM extraction if needed
        if self.llm.enabled and not state.premises_nl:
            system = (
                "Extract logical premises and query from text. "
                "Return JSON: {premises:[...], query: '...'}"
            )
            try:
                text = self.llm.chat(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": state.question},
                    ],
                    temperature=0,
                    max_tokens=600,
                    response_format={"type": "json_object"},
                )
                data = json.loads(text)
                state.metadata["extracted_premises"] = data.get("premises", [])
            except Exception as e:
                state.add_error(f"ExtractLogic error: {str(e)}")

        return state


class ConvertToFOLNode:
    """Convert natural language to First Order Logic."""

    def __init__(self, llm: LLMClientBase):
        self.llm = llm
        self.parser = LogicNLParserAgent(llm)

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Convert premises and query to FOL."""
        try:
            # Build knowledge base from premises
            kb, parsed = self.parser.build_kb(
                premises_nl=state.premises_nl,
                question=state.question,
                premises_fol=state.premises_fol,
            )

            # Store knowledge base in state
            state.parsed_kb = kb

            state.metadata["kb_facts"] = len(kb.facts) if kb else 0
            state.metadata["kb_rules"] = len(kb.rules) if kb else 0

            # Store FOL representation
            if parsed:
                state.metadata["fol_conversion"] = {
                    "method": "llm",
                    "facts": parsed.get("facts", []),
                    "rules": parsed.get("rules", []),
                    "query": parsed.get("query"),
                }
            else:
                state.metadata["fol_conversion"] = {
                    "method": "heuristic",
                    "facts_count": len(kb.facts) if kb else 0,
                    "rules_count": len(kb.rules) if kb else 0,
                }

        except Exception as e:
            state.add_error(f"ConvertToFOL error: {str(e)}")
            state.metadata["fol_conversion"] = {"error": str(e)}

        return state


class VerifyWithZ3Node:
    """Use Z3 solver to verify logical implications."""

    def __init__(self):
        self.reasoner = Z3ReasonerAgent()

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Verify query using Z3."""
        try:
            # Get knowledge base from state
            kb = state.parsed_kb
            if not kb:
                state.add_error("No knowledge base for Z3 verification")
                return state

            # Get FOL conversion info for reasoning
            parsed = state.metadata.get("fol_conversion", {})

            # Find best choice (for multiple choice questions)
            best_label, best_atom, confidence = self.reasoner.best_choice(
                kb, state.question, parsed if parsed.get("method") == "llm" else None
            )

            state.metadata["z3_result"] = {
                "answer": best_label,
                "atom": str(best_atom) if best_atom else None,
                "confidence": confidence,
            }

            state.agent_output = AgentOutput(
                answer=best_label,
                reasoning="Logical deduction using Z3 constraint solver",
                confidence=confidence,
                metadata={
                    "atom": str(best_atom),
                    "kb_facts": len(kb.facts),
                    "kb_rules": len(kb.rules),
                    "method": "z3_solver",
                },
                agent_name="LogicAgent",
            )

        except Exception as e:
            state.add_error(f"VerifyWithZ3 error: {str(e)}")

        return state


class ExplainLogicNode:
    """Generate educational explanation of logical reasoning."""

    def __init__(self, llm: LLMClientBase):
        self.llm = llm

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Generate logical explanation."""
        if not self.llm.enabled:
            state.metadata["explanation_source"] = "unavailable"
            return state

        # Build context for explanation
        z3_result = state.metadata.get("z3_result", {})
        fol_conversion = state.metadata.get("fol_conversion", {})
        premises = state.premises_nl

        context = f"""
Question: {state.question}

Premises:
{json.dumps(premises, ensure_ascii=False, indent=2)}

First Order Logic Conversion:
{json.dumps(fol_conversion, ensure_ascii=False, indent=2)}

Z3 Verification Result:
- Answer: {z3_result.get('answer', 'Unknown')}
- Confidence: {z3_result.get('confidence', 0)}
- Entailed Atom: {z3_result.get('atom')}

Please provide:
1. Summary of premises
2. Key logical rules extracted
3. Step-by-step reasoning
4. Conclusion and answer justification
"""

        system = (
            "Explain the logical reasoning process clearly and educationally. "
            "Break down the premises, show how rules are applied, and justify the conclusion. "
            "Use simple language suitable for educational context."
        )

        try:
            explanation = self.llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": context},
                ],
                temperature=0.3,
                max_tokens=1200,
            )

            # Update agent output with explanation
            if state.agent_output:
                state.agent_output.reasoning = explanation
            state.metadata["explanation"] = explanation
            state.metadata["explanation_source"] = "llm"

        except Exception as e:
            state.add_error(f"ExplainLogic error: {str(e)}")
            state.metadata["explanation_source"] = "error"

        return state


def build_logic_subgraph(llm: LLMClientBase) -> Any:
    """
    Build logic problem solving subgraph.
    
    Nodes:
    1. extract_logic - Extract premises and query
    2. convert_fol - Convert to First Order Logic
    3. verify_z3 - Verify with Z3 solver
    4. explain_logic - Generate explanation
    
    Args:
        llm: LLM client
    
    Returns:
        Compiled LangGraph
    """
    graph = StateGraph(WorkflowState)

    # Create nodes
    extract_node = ExtractLogicNode(llm)
    convert_node = ConvertToFOLNode(llm)
    verify_node = VerifyWithZ3Node()
    explain_node = ExplainLogicNode(llm)

    # Add nodes
    graph.add_node("extract_logic", extract_node)
    graph.add_node("convert_fol", convert_node)
    graph.add_node("verify_z3", verify_node)
    graph.add_node("explain_logic", explain_node)

    # Set entry point
    graph.set_entry_point("extract_logic")

    # Add edges (sequential pipeline)
    graph.add_edge("extract_logic", "convert_fol")
    graph.add_edge("convert_fol", "verify_z3")
    graph.add_edge("verify_z3", "explain_logic")

    # Compile
    return graph.compile()
