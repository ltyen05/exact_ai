# Compact Workflow Refactoring

## Intent

This revision minimizes request/response data and reduces physics parsing
latency while retaining the routed LangGraph design.

## Breaking API Change

`/predict` now accepts only `question` and optional natural-language
`premises`. Inputs that previously injected routing hints, FOL, parsed physics
objects, or solution specifications are rejected.

Responses contain `answer` and `explanation`; structured evidence is present
only when non-empty.

## Main Changes

- Workflow state no longer carries routing or tracing diagnostics.
- Router returns only the selected domain route.
- Physics parser uses a prompt below 10 KB and code-enforced compact output.
- Logic formalization and explanation are now loaded from prompt files.
- Prompt templates and the physics retrieval corpus are cached.
- LangSmith uses filtered `traceable` spans; LLM HTTP attempts expose latency
  and retry information without prompt content.

## Migration

Old request:

```json
{"type": "logic", "question": "...", "premises-NL": ["..."]}
```

New request:

```json
{"question": "...", "premises": ["..."]}
```

Clients should not rely on empty evidence arrays being returned.
