from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from eval.p1_evaluator import evaluate_prediction, summarize_p1


ROOT = Path(__file__).resolve().parent
DEFAULT_LOGIC_FILE = ROOT / "data" / "Logic_Based_Educational_Queries.json"
DEFAULT_PHYSICS_FILE = ROOT / "data" / "Physics_Problems_Text_Only_removeQA.json"
DEFAULT_COMPETITION_FILE = ROOT / "data" / "EXACT_2026_test_payload.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "demo_p1_output.json"
TRUE_VALUES = {"1", "true", "yes", "on"}


def configure_tracing() -> None:
    if os.getenv("EXACT_ENABLE_LANGSMITH", "").strip().lower() not in TRUE_VALUES:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


@dataclass(frozen=True)
class DemoSample:
    sample_id: str
    task: str
    question: str
    expected_answer: str
    expected_unit: str = ""
    premises: tuple[str, ...] = ()
    source_file: str = ""
    record_index: int = -1
    question_index: int | None = None

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": self.question}
        if self.premises:
            payload["premises"] = list(self.premises)
        return payload


class DatasetRouteClassifier:
    """Route demo records by the file they came from."""

    def __init__(self, route_by_question: dict[str, str]) -> None:
        self.route_by_question = route_by_question

    def run(self, question: str) -> dict[str, str]:
        return {"Type": self.route_by_question.get(question, "logic")}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_items(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in _as_list(value))


def build_logic_samples(records: list[dict[str, Any]], source_file: Path) -> list[DemoSample]:
    samples: list[DemoSample] = []
    for record_index, record in enumerate(records):
        premises = _string_items(record.get("premises-NL") or record.get("premises"))
        questions = _as_list(record.get("questions"))
        answers = _as_list(record.get("answers"))
        for question_index, question in enumerate(questions):
            if question_index >= len(answers):
                continue
            samples.append(
                DemoSample(
                    sample_id=f"logic-{record_index}-{question_index}",
                    task="logic",
                    question=str(question),
                    expected_answer=str(answers[question_index]),
                    premises=premises,
                    source_file=str(source_file),
                    record_index=record_index,
                    question_index=question_index,
                )
            )
    return samples


def build_physics_samples(records: list[dict[str, Any]], source_file: Path) -> list[DemoSample]:
    samples: list[DemoSample] = []
    for record_index, record in enumerate(records):
        question = record.get("question")
        answer = record.get("answer")
        if question is None or answer is None:
            continue
        samples.append(
            DemoSample(
                sample_id=str(record.get("id") or f"physics-{record_index}"),
                task="physics",
                question=str(question),
                expected_answer=str(answer),
                expected_unit=str(record.get("unit") or ""),
                source_file=str(source_file),
                record_index=record_index,
            )
        )
    return samples


def load_samples(args: argparse.Namespace) -> tuple[list[DemoSample], list[DemoSample]]:
    logic_file = Path(args.logic_file)
    physics_file = Path(args.physics_file)
    logic_records = [record for record in _as_list(load_json(logic_file)) if isinstance(record, dict)]
    physics_records = [record for record in _as_list(load_json(physics_file)) if isinstance(record, dict)]
    return build_logic_samples(logic_records, logic_file), build_physics_samples(physics_records, physics_file)


def select_samples(
    logic_samples: list[DemoSample],
    physics_samples: list[DemoSample],
    *,
    logic_limit: int,
    physics_limit: int,
    seed: int,
    shuffle: bool,
) -> list[DemoSample]:
    logic_pool = list(logic_samples)
    physics_pool = list(physics_samples)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(logic_pool)
        rng.shuffle(physics_pool)
    return [*logic_pool[: max(0, logic_limit)], *physics_pool[: max(0, physics_limit)]]


def make_predictor(
    samples: list[DemoSample],
    *,
    api_key_env: str,
    physics_kb_path: Path,
) -> Callable[[DemoSample], dict[str, Any]]:
    from agents.llm import OpenRouterClient
    from agents.workflows import ExactGraph

    llm = OpenRouterClient(api_key_env=api_key_env)
    if not llm.enabled:
        raise SystemExit(f"OpenRouter is not configured. Set {api_key_env} in .env before running the demo.")

    route_by_question = {sample.question: sample.task for sample in samples}
    graph = ExactGraph(
        llm=llm,
        physics_kb_path=str(physics_kb_path),
        classifier=DatasetRouteClassifier(route_by_question),
    )

    def predict(sample: DemoSample) -> dict[str, Any]:
        return graph.predict(sample.payload())

    return predict


def make_graph(args: argparse.Namespace) -> Any:
    from agents.llm import OpenRouterClient, VLLMClient
    from agents.workflows import ExactGraph

    provider = str(args.provider or "openrouter").strip().lower()
    if provider == "vllm":
        llm = VLLMClient()
    else:
        llm = OpenRouterClient(api_key_env=args.api_key_env)
    if not llm.enabled:
        raise SystemExit(f"{llm.provider} is not configured.")
    return ExactGraph(
        llm=llm,
        physics_kb_path=str(args.physics_file),
        logic_kb_path=str(args.logic_file),
    )


def run_competition_payload_demo(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    configure_tracing()
    payload_file = Path(args.competition_file)
    records = [
        record
        for record in _as_list(load_json(payload_file))
        if isinstance(record, dict) and str(record.get("query") or "").strip()
    ]
    limit = max(0, int(args.competition_samples))
    selected = records[:limit] if limit else records
    if args.dry_run:
        return {
            "meta": {
                "dry_run": True,
                "competition_file": str(payload_file),
                "available_payloads": len(records),
                "selected_payloads": len(selected),
            },
            "payloads": selected,
        }

    graph = make_graph(args)
    results: list[dict[str, Any]] = []
    for index, payload in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {payload.get('type')} {payload.get('query_id')}")
        started = time.perf_counter()
        output: dict[str, Any] = {}
        error = ""
        try:
            output = graph.predict(payload)
        except Exception as exc:
            error = str(exc)
        results.append(
            {
                "payload": payload,
                "output": output,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": error,
            }
        )
    return {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "competition_file": str(payload_file),
            "provider": getattr(graph.llm, "provider", "unknown"),
            "available_payloads": len(records),
            "selected_payloads": len(selected),
            "contract": "EXACT 2026 unified input/output schema",
        },
        "results": results,
    }


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    configure_tracing()
    if args.competition_file:
        return run_competition_payload_demo(args)
    logic_pool, physics_pool = load_samples(args)
    samples = select_samples(
        logic_pool,
        physics_pool,
        logic_limit=args.logic_samples,
        physics_limit=args.physics_samples,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    if args.dry_run:
        return {
            "meta": {
                "dry_run": True,
                "logic_file": str(args.logic_file),
                "physics_file": str(args.physics_file),
                "logic_samples_available": len(logic_pool),
                "physics_samples_available": len(physics_pool),
                "selected_samples": len(samples),
            },
            "samples": [asdict(sample) for sample in samples],
        }

    predictor = make_predictor(
        samples,
        api_key_env=args.api_key_env,
        physics_kb_path=Path(args.physics_file),
    )

    results: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {sample.task} {sample.sample_id}")
        started = time.perf_counter()
        output: dict[str, Any] = {}
        error = ""
        try:
            output = predictor(sample)
        except Exception as exc:
            error = str(exc)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        predicted_answer = str(output.get("answer") or "")
        p1_evaluation = evaluate_prediction(
            predicted_answer,
            sample.expected_answer,
            sample.expected_unit,
            predicted_unit=str(output.get("unit") or ""),
            rtol=args.rtol,
            atol=args.atol,
        )
        if error:
            p1_evaluation = {
                "correct": False,
                "method": "prediction_error",
                "reason": f"prediction error: {error}",
            }
        results.append(
            {
                **asdict(sample),
                "payload": sample.payload(),
                "expected": {"answer": sample.expected_answer, "unit": sample.expected_unit},
                "predicted_answer": predicted_answer,
                "output": output,
                "correct": bool(p1_evaluation["correct"]),
                "match_reason": str(p1_evaluation["reason"]),
                "p1_evaluation": p1_evaluation,
                "latency_ms": latency_ms,
                "error": error,
            }
        )

    return {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "logic_file": str(args.logic_file),
            "physics_file": str(args.physics_file),
            "routing": "dataset_source",
            "p1_definition": "Correctness of Answers",
            "logic_samples_requested": args.logic_samples,
            "physics_samples_requested": args.physics_samples,
            "logic_samples_available": len(logic_pool),
            "physics_samples_available": len(physics_pool),
        },
        "metrics": summarize_p1(results),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a P1 correctness demo over the two EXACT data files.")
    parser.add_argument("--logic-file", type=Path, default=DEFAULT_LOGIC_FILE)
    parser.add_argument("--physics-file", type=Path, default=DEFAULT_PHYSICS_FILE)
    parser.add_argument("--competition-file", type=Path, default=None, help="Run the unified EXACT 2026 payload file instead of gold-labeled demo data.")
    parser.add_argument("--competition-samples", type=int, default=5, help="Number of competition payloads to run; 0 means all.")
    parser.add_argument("--logic-samples", type=int, default=2)
    parser.add_argument("--physics-samples", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Randomly sample records before truncating.")
    parser.add_argument("--dry-run", action="store_true", help="Only write selected samples; do not call the model.")
    parser.add_argument("--provider", choices=["openrouter", "vllm"], default="openrouter", help="LLM provider for local demo runs.")
    parser.add_argument("--api-key-env", default="OR_TOKEN", help="Environment variable used by OpenRouter.")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for numeric P1 matching.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for numeric P1 matching.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_demo(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        print(
            "P1 accuracy: "
            f"{metrics['p1_accuracy']:.3f} "
            f"({metrics['correct']}/{metrics['total']})"
        )
        for task, task_metrics in metrics["by_task"].items():
            print(
                f"  {task}: {task_metrics['p1_accuracy']:.3f} "
                f"({task_metrics['correct']}/{task_metrics['total']})"
            )
    else:
        selected = len(report.get("samples", [])) or len(report.get("payloads", [])) or len(report.get("results", []))
        print(f"Selected {selected} samples.")
    print(f"Output written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
