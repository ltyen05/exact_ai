from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agents.formatter import normalize_answer, parse_float_like
from agents.pipeline import ExactPipeline


def unit_norm(u: Any) -> str:
    u = str(u or "").strip().lower()
    u = u.replace("µ", "μ")
    aliases = {"degree": "độ", "degrees": "độ", "ohms": "ω", "ohm": "ω"}
    return aliases.get(u, u)


def answer_match(pred: Any, gold: Any, pred_unit: Any = "", gold_unit: Any = "", numeric_tol: float = 1e-2) -> bool:
    pg = normalize_answer(pred)
    gg = normalize_answer(gold)
    if pg == gg:
        # for physics also check unit when gold unit exists
        return not gold_unit or unit_norm(pred_unit) == unit_norm(gold_unit)
    pv, gv = parse_float_like(pred), parse_float_like(gold)
    if pv is not None and gv is not None:
        tol = max(numeric_tol, abs(gv) * 1e-3)
        return abs(pv - gv) <= tol and (not gold_unit or unit_norm(pred_unit) == unit_norm(gold_unit))
    return False


def eval_logic(path: str, pipe: ExactPipeline, max_records: Optional[int] = None, use_fol: bool = True) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    total = correct = 0
    rows = []
    for ridx, rec in enumerate(data[:max_records] if max_records else data):
        for qidx, q in enumerate(rec.get("questions", [])):
            payload = {
                "type": "logic",
                "question": q,
                "premises-NL": rec.get("premises-NL", []),
            }
            if use_fol:
                payload["premises-FOL"] = rec.get("premises-FOL", [])
            pred = pipe.predict(payload)
            gold = rec.get("answers", [None])[qidx]
            ok = answer_match(pred.get("answer"), gold)
            total += 1
            correct += int(ok)
            rows.append({"record": ridx, "question_index": qidx, "pred": pred.get("answer"), "gold": gold, "ok": ok})
    return {"task": "logic", "total": total, "correct": correct, "p1": correct / total if total else 0.0, "rows": rows}


def eval_physics(path: str, pipe: ExactPipeline, max_records: Optional[int] = None) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data = [r for r in data if not str(r.get("id", "")).startswith("QA")]
    total = correct = 0
    rows = []
    for ridx, rec in enumerate(data[:max_records] if max_records else data):
        payload = {"type": "physics", "id": rec.get("id"), "question": rec.get("question", "")}
        pred = pipe.predict(payload)
        ok = answer_match(pred.get("answer"), rec.get("answer"), pred.get("unit"), rec.get("unit"))
        total += 1
        correct += int(ok)
        rows.append({"record": ridx, "id": rec.get("id"), "pred": pred.get("answer"), "pred_unit": pred.get("unit"), "gold": rec.get("answer"), "gold_unit": rec.get("unit"), "ok": ok})
    return {"task": "physics", "total": total, "correct": correct, "p1": correct / total if total else 0.0, "rows": rows}


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = sum(r["total"] for r in results)
    correct = sum(r["correct"] for r in results)
    return {"total": total, "correct": correct, "p1": correct / total if total else 0.0, "by_task": [{k: v for k, v in r.items() if k != "rows"} for r in results]}
