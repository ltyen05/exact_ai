# EXACT 2026 Workflow Architecture

## Public Contract

`POST /predict` accepts only:

```json
{
  "query_id": "T1_0001",
  "type": "type1",
  "query": "required non-empty text",
  "premises": ["0-indexed natural-language premises; [] for type2"],
  "options": ["choice answers, or [] for free-form/type2"]
}
```

The endpoint always returns a JSON list with one result object:

```json
[
  {
    "query_id": "T1_0001",
    "answer": "Yes",
    "unit": "",
    "explanation": "Non-empty explanation.",
    "premises_used": [0, 1],
    "reasoning": {"type": "fol", "steps": ["..."]}
  }
]
```

For `type1`, `unit` is empty and `premises_used` contains 0-based premise
indices. For `type2`, `answer` contains the value only, `unit` contains the
ASCII unit, and `premises_used` is `[]`.

## Graph Flow

```text
START
  -> classify_route
     -> type2/physics_subgraph: parse_question -> select_solution -> compute_sympy -> explain_answer
     -> type1/logic_subgraph: extract_logic -> classify -> plan -> execute -> extract -> explain_logic
  -> format_output
  -> END
```

`WorkflowState` contains only business data required by the route:

```python
class WorkflowState(TypedDict, total=False):
    query_id: str
    query_type: str
    question: str
    premises: list[str]
    options: list[str]
    route: str
    parsed_question: dict
    solution_output: dict
    verified_output: dict
    logic_spec: dict
    result: dict
    errors: list[str]
    output: dict
```

Competition routing uses the explicit `type` field: `type1` routes to logic and
`type2` routes to physics. The legacy internal `question/premises` path still
falls back to the trained router for local debugging.

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

Workflow functions can use `@langsmith.traceable` processors. Tracing is off by
default for submission runs and is enabled only with `EXACT_ENABLE_LANGSMITH=true`.
The trace view stores
compact business artifacts and LLM attempt metrics, not full prompts or routing
diagnostics. `llm.http_attempt` spans capture stage, provider/model, JSON mode,
attempt number, request/response character counts, duration, and status.
