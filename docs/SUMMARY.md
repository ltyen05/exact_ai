# EXACT 2026 Current Summary

## Runtime

- Entrypoint: `api.py`
- Orchestration: `agents/workflows/orchestrator.py`
- Domain flows: `agents/workflows/physics.py` and `agents/workflows/logic.py`
- Observability: `agents/workflows/tracing.py`

## Contract

Input:

```json
{"question": "...", "premises": ["..."]}
```

Output:

```json
{
  "answer": "...",
  "explanation": "...",
  "fol": "optional",
  "cot": ["optional"],
  "premises": ["optional"]
}
```

Only `answer` and `explanation` are mandatory. Advanced intermediate physics
objects and formal-logic request fields are not public inputs.

## Performance Changes

- Physics parsing prompt is compact and emits optional sections only when applicable.
- Parser, solution, and explanation templates are cached per configured agent.
- Physics RAG loads and normalizes its document corpus once per provider.
- LLM traces report request attempt timing and size without storing prompts.

## Validation

Regression tests cover the public API contract, compact parser behavior,
physics and logic execution, RAG caching, trace filtering, and OpenRouter retry
behavior.
