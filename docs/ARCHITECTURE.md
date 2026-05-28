# EXACT 2026 Workflow Architecture

## Public Contract

`POST /predict` accepts only:

```json
{
  "question": "required non-empty text",
  "premises": ["optional natural-language premises for logic questions"]
}
```

The response always contains `answer` and `explanation`. It includes `fol`,
`cot`, or `premises` only when a workflow produced non-empty evidence.

## Graph Flow

```text
START
  -> classify_route
     -> physics_subgraph: parse_question -> select_solution -> compute_sympy -> explain_answer
     -> logic_subgraph: extract_logic -> convert_to_fol -> verify_z3 -> explain_logic
  -> format_output
  -> END
```

`WorkflowState` contains only business data required by the route:

```python
class WorkflowState(TypedDict, total=False):
    question: str
    premises: list[str]
    route: str
    parsed_question: dict
    solution_output: dict
    verified_output: dict
    logic_spec: dict
    result: dict
    errors: list[str]
    output: dict
```

Routing chooses `logic` or `physics`; it does not publish diagnostic scores.

## Physics Route

- `ParsingAgent` uses a compact prompt and emits required fields plus only relevant optional sections.
- `SolutionAgent` produces one deterministic solution specification; RAG documents and prompt templates are cached.
- `compute_sympy` validates extracted numeric input before solving.
- `ExplainAgent` writes the final explanation from the verified result.

## Logic Route

- The formalizer reads `prompts/logic_to_fol.txt` and returns compact `facts`, `rules`, `query`, and `choices`.
- The workflow builds `HornKB`, performs verification, and exposes proof evidence only when available.
- The explanation prompt is read from `prompts/logic_explain.txt`.

## Tracing

Workflow functions use `@langsmith.traceable` processors. The trace view stores
compact business artifacts and LLM attempt metrics, not full prompts or routing
diagnostics. `llm.http_attempt` spans capture stage, provider/model, JSON mode,
attempt number, request/response character counts, duration, and status.
