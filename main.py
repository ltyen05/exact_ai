from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from agents.pipeline import ExactPipeline
from eval.p1_evaluator import eval_logic, eval_physics, summarize

ROOT = Path(__file__).resolve().parent
DEFAULT_PHYSICS = ROOT / "data" / "Physics_Problems_Text_Only_removeQA.json"
DEFAULT_LOGIC = ROOT / "data" / "Logic_Based_Educational_Queries.json"


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def cmd_infer(args: argparse.Namespace) -> None:
    pipe = ExactPipeline(physics_kb_path=args.physics_kb)
    records = load_json_or_jsonl(args.input)
    outputs = []
    for rec in records:
        outputs.extend(pipe.predict_record(rec))
    out_text = "\n".join(json.dumps(o, ensure_ascii=False) for o in outputs)
    if args.output:
        Path(args.output).write_text(out_text + "\n", encoding="utf-8")
    else:
        print(out_text)


def cmd_eval(args: argparse.Namespace) -> None:
    pipe = ExactPipeline(physics_kb_path=args.physics)
    results = []
    if args.logic:
        results.append(eval_logic(args.logic, pipe, max_records=args.max_records, use_fol=not args.no_fol))
    if args.physics:
        results.append(eval_physics(args.physics, pipe, max_records=args.max_records))
    summary = summarize(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.details:
        Path(args.details).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_demo(args: argparse.Namespace) -> None:
    pipe = ExactPipeline(physics_kb_path=str(DEFAULT_PHYSICS))
    samples = [
        {
            "type": "physics",
            "question": "Two electric forces, each with a magnitude of 5 N, act at an angle of 60° to each other. What is the resultant force?",
        },
        {
            "type": "logic",
            "premises-NL": [
                "If a student completes all required courses, they are eligible for graduation.",
                "If a student is eligible for graduation and maintains a GPA above 3.5, they graduate with honors.",
                "If a student graduates with honors and completes a thesis, they receive academic distinction.",
                "John has completed all required courses.",
                "John maintains a GPA of 3.8.",
                "John has completed a thesis.",
            ],
            "question": "Does John receive academic distinction, according to the premises?",
        },
    ]
    for s in samples:
        print(json.dumps(pipe.predict(s), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EXACT 2026 multi-agent inference + P1 evaluation")
    sub = p.add_subparsers(required=True)

    infer = sub.add_parser("infer")
    infer.add_argument("--input", required=True, help="JSON/JSONL file with unified test payloads")
    infer.add_argument("--output", default="", help="Optional JSONL output path")
    infer.add_argument("--physics-kb", default=str(DEFAULT_PHYSICS), help="Training physics KB for retrieval fallback")
    infer.set_defaults(func=cmd_infer)

    ev = sub.add_parser("eval")
    ev.add_argument("--logic", default=str(DEFAULT_LOGIC))
    ev.add_argument("--physics", default=str(DEFAULT_PHYSICS))
    ev.add_argument("--max-records", type=int, default=None)
    ev.add_argument("--no-fol", action="store_true", help="Disable training FOL annotations during local logic evaluation")
    ev.add_argument("--details", default="", help="Write detailed per-sample predictions to this JSON file")
    ev.set_defaults(func=cmd_eval)

    demo = sub.add_parser("demo")
    demo.set_defaults(func=cmd_demo)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
