# Routing And Domain Workflows

## Router

The TF-IDF classifier is an internal routing step. It returns one route:
`logic` or `physics`. Its model labels and probability diagnostics are not
part of workflow state or any public response.

## Physics

Physics requests follow four steps:

```text
compact semantic parsing -> solution specification -> verified computation -> explanation
```

The compact parser preserves necessary givens, conditions, geometry,
comparisons and answer-format instructions while omitting unused sections.

## Logic

Logic requests accept natural-language premises. The formalizer converts them
to compact Horn-rule JSON from `prompts/logic_to_fol.txt`; the verifier derives
the answer and optional proof evidence.

## Observability

Use LangSmith step traces for route timing and `llm.http_attempt` spans for
LLM latency or retry diagnosis. The workflow intentionally excludes internal
diagnostic data and full prompts from traced input/output.
