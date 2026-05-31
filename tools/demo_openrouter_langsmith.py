"""Demo EXACT physics workflow with OpenRouter and LangSmith tracing.

Run from the repository root:
    python -B tools/demo_openrouter_langsmith.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_PHYSICS_KB = ROOT / "data" / "Physics_Problems_Text_Only_removeQA.json"
DEFAULT_QUESTION = "A capacitor has capacitance C = 25 microF and voltage U = 120 V. Find the energy stored."


def _load_environment() -> None:
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGSMITH_PROJECT", "exact-2026-demo")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", os.environ["LANGSMITH_TRACING"])
    os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ["LANGSMITH_ENDPOINT"])
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"])


def _mask(value: str | None) -> str:
    if not value:
        return "NOT SET"
    if len(value) <= 12:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


def _env_status() -> dict[str, Any]:
    return {
        "OR_TOKEN": _mask(os.getenv("OR_TOKEN")),
        "OPENROUTER_MODEL": os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct"),
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING", "false"),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT", "NOT SET"),
        "LANGSMITH_API_KEY": _mask(os.getenv("LANGSMITH_API_KEY")),
    }


class PhysicsOnlyClassifier:
    """Route the demo question directly to the physics workflow."""

    def run(self, question: str) -> dict[str, str]:
        del question
        return {"Type": "physics"}


def main() -> int:
    _load_environment()

    from agents.llm import OpenRouterClient
    from agents.workflows import ExactGraph

    parser = argparse.ArgumentParser(description="Run one OpenRouter + LangSmith EXACT demo.")
    parser.add_argument("-q", "--question", default=DEFAULT_QUESTION, help="Question to run through the workflow.")
    parser.add_argument("--auto-route", action="store_true", help="Use the local classifier instead of forcing physics.")
    parser.add_argument("--dry-run", action="store_true", help="Only print masked configuration; do not call OpenRouter.")
    args = parser.parse_args()

    llm = OpenRouterClient(
        api_key_env="OR_TOKEN",
        http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
        app_title=os.getenv("OPENROUTER_APP_TITLE", "EXACT 2026 Demo"),
    )

    print("Configuration:")
    print(json.dumps(_env_status(), indent=2, ensure_ascii=False))
    print(f"OpenRouter enabled: {llm.enabled}")

    if args.dry_run:
        return 0
    if not llm.enabled:
        raise SystemExit("OpenRouter is not configured. Check OR_TOKEN and OPENROUTER_MODEL in .env.")

    classifier = None if args.auto_route else PhysicsOnlyClassifier()
    graph = ExactGraph(llm=llm, physics_kb_path=str(DEFAULT_PHYSICS_KB), classifier=classifier)
    output = graph.predict(
        {"question": args.question},
        config={
            "run_name": "demo_openrouter_langsmith",
            "tags": ["demo", "openrouter", "langsmith"],
        },
    )

    print("\nQuestion:")
    print(args.question)
    print("\nOutput:")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
