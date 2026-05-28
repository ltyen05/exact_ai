"""JSON extraction utilities for LLM responses."""

import json
import re
from typing import Any, Dict, Optional


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON content."""
    # Handle ```json ... ``` or ``` ... ```
    stripped = re.sub(
        r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
        r"\1",
        text,
        flags=re.S,
    )
    return stripped.strip()


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] (common LLM JSON error)."""
    # ,} or ,]
    return re.sub(r",\s*([}\]])", r"\1", text)


def _remove_js_comments(text: str) -> str:
    """Remove single-line // comments outside of strings."""
    # Simple heuristic: remove lines that are just comments, or trailing comments
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        # Skip pure comment lines
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        # Remove trailing comments (naive: only outside quotes)
        # Only strip if // is not inside a quoted string
        in_string = False
        escape_next = False
        cut_index = None
        for i, ch in enumerate(line):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
            elif ch == "/" and not in_string and i + 1 < len(line) and line[i + 1] == "/":
                cut_index = i
                break
        if cut_index is not None:
            line = line[:cut_index].rstrip()
        cleaned.append(line)
    return "\n".join(cleaned)


def _clean_control_chars(text: str) -> str:
    """Remove BOM and control characters that break JSON parsing."""
    # Remove BOM
    text = text.lstrip("\ufeff")
    # Remove control chars except \n \r \t
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def _try_parse(text: str) -> Optional[Dict[str, Any]]:
    """Try parsing text as JSON, return dict or None."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extractor for local LLM outputs.

    Handles common LLM output issues:
    - Markdown code fences (```json ... ```)
    - Trailing commas in objects/arrays
    - JavaScript-style // comments
    - BOM and control characters
    - Extra text before/after the JSON object
    - Multiple JSON blocks (returns the first valid one)
    """
    if not text:
        return None
    text = _clean_control_chars(text).strip()

    # 1. Try direct parse
    result = _try_parse(text)
    if result is not None:
        return result

    # 2. Strip markdown fences and retry
    stripped = _strip_markdown_fences(text)
    if stripped != text:
        result = _try_parse(stripped)
        if result is not None:
            return result

    # 3. Try with trailing commas removed
    no_commas = _remove_trailing_commas(stripped)
    if no_commas != stripped:
        result = _try_parse(no_commas)
        if result is not None:
            return result

    # 4. Try with JS comments removed
    no_comments = _remove_js_comments(no_commas)
    if no_comments != no_commas:
        result = _try_parse(no_comments)
        if result is not None:
            return result

    # 5. Extract first JSON object by brace matching
    depth = 0
    start = None
    in_string = False
    escape_next = False
    for i, ch in enumerate(no_comments):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = no_comments[start:i + 1]
                result = _try_parse(candidate)
                if result is not None:
                    return result
                # Try with trailing commas removed on this candidate
                result = _try_parse(_remove_trailing_commas(candidate))
                if result is not None:
                    return result
                # Reset and keep searching for next top-level object
                start = None

    # 6. Last resort: find outermost { ... } with simple regex
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        result = _try_parse(candidate)
        if result is not None:
            return result
        result = _try_parse(_remove_trailing_commas(candidate))
        if result is not None:
            return result

    return None
