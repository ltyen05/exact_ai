from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agents.formatter import extract_json, standard_response
from agents.llm_client import VLLMClient
from tools.z3_logic import Atom, HornKB, Rule, parse_fol_to_kb, pred_name


def split_choices(question: str) -> Dict[str, str]:
    choices: Dict[str, str] = {}
    matches = list(re.finditer(r"(?:^|\n)\s*([A-D])\.\s*", question))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
        choices[m.group(1)] = question[start:end].strip()
    return choices


def question_stem(question: str) -> str:
    return re.split(r"\n\s*A\.\s*", question)[0].strip()


def tokenize(s: str) -> set:
    stop = {"a", "an", "the", "is", "are", "be", "to", "for", "of", "with", "and", "or", "if", "then", "all", "every", "does", "do", "did", "can", "could", "should", "must", "based", "premises", "according", "about", "on", "in", "that", "it", "he", "she", "they", "student", "students", "project", "projects", "system", "systems", "faculty", "member", "members"}
    return {w for w in re.findall(r"[a-zA-Z0-9_]+", s.lower()) if w not in stop and len(w) > 1}


def atom_words(atom: Atom) -> set:
    return tokenize(atom.pred.replace("_", " ") + " " + " ".join(atom.args))


class LogicNLParserAgent:
    """Convert natural language premises/questions to a compact rule JSON.

    The preferred path is a local <=8B LLM served by vLLM. A small regex fallback
    is included so the project can run without a model for smoke tests.
    """

    def __init__(self, llm: Optional[VLLMClient] = None):
        self.llm = llm or VLLMClient()

    def parse_with_llm(self, premises_nl: List[str], question: str) -> Optional[Dict[str, Any]]:
        if not self.llm.enabled:
            return None
        system = (
            "You convert educational rules into a finite Horn-logic JSON for Z3. "
            "Return JSON only. Predicates must be snake_case. Use concrete constants when named entities appear. "
            "Schema: {facts:[{pred,args,truth,source}], rules:[{if:[{pred,args,truth}], then:{pred,args,truth}, source}], "
            "query:{pred,args,truth}|null, choices:{A:{pred,args,truth},...}|{}}. "
            "Use args ['entity'] for generic universal claims if no named entity is present."
        )
        user = json.dumps({"premises": premises_nl, "question": question}, ensure_ascii=False)
        try:
            text = self.llm.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ], temperature=0, max_tokens=1800, response_format={"type": "json_object"})
            return extract_json(text)
        except Exception:
            return None

    def fallback_parse_premises(self, premises_nl: List[str]) -> HornKB:
        kb = HornKB()
        for i, p in enumerate(premises_nl, 1):
            src = f"Premise {i}: {p}"
            s = p.strip().rstrip(".")
            # If A then B.
            m = re.match(r"if\s+(.+?),?\s+then\s+(.+)$", s, flags=re.I)
            if not m:
                m = re.match(r"(.+?)\s+implies\s+(.+)$", s, flags=re.I)
            if m:
                ant = pred_name(m.group(1))
                cons = pred_name(m.group(2))
                kb.add_rule(Rule([Atom(ant, ("entity",), True)], Atom(cons, ("entity",), True), src))
                continue
            # All X are Y / Every X is Y.
            m = re.match(r"(?:all|every)\s+(.+?)\s+(?:are|is)\s+(.+)$", s, flags=re.I)
            if m:
                ant = pred_name(m.group(1))
                cons = pred_name(m.group(2))
                kb.add_rule(Rule([Atom(ant, ("entity",), True)], Atom(cons, ("entity",), True), src))
                kb.add_fact(Atom(ant, ("entity",), True), src)
                continue
            # Named fact: Sophia has completed ... / John maintains ...
            m = re.match(r"(?:professor\s+|dr\.\s+)?([A-Z][A-Za-z0-9_]+)\s+(.+)$", s)
            if m:
                name = pred_name(m.group(1))
                phrase = pred_name(m.group(2))
                phrase = re.sub(r"^(has|have|is|are|maintains|completed|received)_", "", phrase)
                kb.add_fact(Atom(phrase, (name,), True), src)
        kb.closure()
        return kb

    def build_kb(self, premises_nl: List[str], question: str, premises_fol: Optional[List[str]] = None) -> Tuple[HornKB, Optional[Dict[str, Any]]]:
        parsed = self.parse_with_llm(premises_nl, question)
        if parsed:
            kb = HornKB()
            for f in parsed.get("facts", []):
                kb.add_fact(Atom(pred_name(f.get("pred", "predicate")), tuple(map(pred_name, f.get("args", ["entity"]))), bool(f.get("truth", True))), f.get("source", "LLM fact"))
            for r in parsed.get("rules", []):
                ants = [Atom(pred_name(a.get("pred", "predicate")), tuple(map(pred_name, a.get("args", ["entity"]))), bool(a.get("truth", True))) for a in r.get("if", [])]
                th = r.get("then", {})
                if ants and th:
                    kb.add_rule(Rule(ants, Atom(pred_name(th.get("pred", "predicate")), tuple(map(pred_name, th.get("args", ["entity"]))), bool(th.get("truth", True))), r.get("source", "LLM rule")))
            kb.closure()
            return kb, parsed
        if premises_fol:
            return parse_fol_to_kb(premises_fol), None
        return self.fallback_parse_premises(premises_nl), None


def json_atom_to_atom(obj: Any) -> Optional[Atom]:
    if not isinstance(obj, dict) or not obj.get("pred"):
        return None
    return Atom(pred_name(obj.get("pred")), tuple(map(pred_name, obj.get("args", ["entity"]))), bool(obj.get("truth", True)))


class Z3ReasonerAgent:
    def answer_atom(self, kb: HornKB, atom: Atom) -> str:
        ent = kb.entails(atom)
        if ent is True:
            return "Yes"
        if ent is False:
            return "No"
        return "Unknown"

    def best_choice(self, kb: HornKB, question: str, parsed: Optional[Dict[str, Any]]) -> Tuple[str, Optional[Atom], float]:
        choices = split_choices(question)
        if not choices:
            return "Unknown", None, 0.25
        # Preferred: LLM provided formal choices.
        if parsed and isinstance(parsed.get("choices"), dict):
            for label, obj in parsed["choices"].items():
                atom = json_atom_to_atom(obj)
                if atom and kb.entails(atom) is True:
                    return label, atom, 0.85
        # Fallback: pick option whose text overlaps derived atoms the most.
        kb.closure()
        positive_atoms = [a for a in kb.facts if a.truth]
        best = ("Unknown", None, 0.0)
        for label, text in choices.items():
            tw = tokenize(text)
            score = 0.0
            best_atom = None
            for atom in positive_atoms:
                aw = atom_words(atom)
                if not aw:
                    continue
                overlap = len(tw & aw) / max(1, len(tw | aw))
                if overlap > score:
                    score, best_atom = overlap, atom
            # penalize option words that indicate need/cannot/not when no negation exists
            if re.search(r"\b(needs?|cannot|not|insufficient|lacks?)\b", text, flags=re.I) and best_atom and best_atom.truth:
                score *= 0.65
            if score > best[2]:
                best = (label, best_atom, score)
        if best[2] >= 0.12:
            return best[0], best[1], min(0.65, 0.35 + best[2])
        return "Unknown", None, 0.25

    def yes_no_unknown(self, kb: HornKB, question: str, parsed: Optional[Dict[str, Any]]) -> Tuple[str, Optional[Atom], float]:
        if parsed:
            atom = json_atom_to_atom(parsed.get("query"))
            if atom:
                return self.answer_atom(kb, atom), atom, 0.85
        # Fallback: match question words to known positive/negative facts.
        qw = tokenize(question)
        kb.closure()
        best_atom, best_score = None, 0.0
        for atom in kb.facts:
            aw = atom_words(atom)
            score = len(qw & aw) / max(1, len(qw | aw))
            if score > best_score:
                best_score, best_atom = score, atom
        if best_atom and best_score >= 0.10:
            return ("Yes" if best_atom.truth else "No"), best_atom, min(0.7, 0.35 + best_score)
        return "Unknown", None, 0.25


class ExplanationAgent:
    def __init__(self, llm: Optional[VLLMClient] = None):
        self.llm = llm or VLLMClient()

    def explain(self, question: str, answer: str, atom: Optional[Atom], kb: HornKB, premises_nl: List[str], cot: List[str]) -> str:
        if self.llm.enabled:
            try:
                prompt = {
                    "question": question,
                    "answer": answer,
                    "entailed_atom": atom.label() if atom else None,
                    "tool_trace": cot,
                    "premises": premises_nl,
                }
                return self.llm.chat([
                    {"role": "system", "content": "Write a concise, faithful explanation for an educational QA answer. Mention the symbolic tool trace but do not invent premises."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ], temperature=0, max_tokens=700)
            except Exception:
                pass
        if atom:
            chain = kb.trace.get(atom, [])[-6:]
            if chain:
                return f"Z3/forward-chaining derives {atom.label()} from the cited premises, so the answer is {answer}. Trace: " + " | ".join(chain)
        if cot:
            return f"The symbolic checker could not prove a stronger conclusion, so the answer is {answer}. Evidence: " + " | ".join(cot[:5])
        return f"The available premises do not provide enough support for a stronger conclusion, so the answer is {answer}."


class LogicAgent:
    def __init__(self, llm: Optional[VLLMClient] = None):
        self.parser = LogicNLParserAgent(llm)
        self.reasoner = Z3ReasonerAgent()
        self.explainer = ExplanationAgent(llm)

    def solve(self, question: str, premises_nl: List[str], premises_fol: Optional[List[str]] = None) -> Dict[str, Any]:
        kb, parsed = self.parser.build_kb(premises_nl, question, premises_fol)
        choices = split_choices(question)
        if choices:
            answer, atom, conf = self.reasoner.best_choice(kb, question, parsed)
        else:
            answer, atom, conf = self.reasoner.yes_no_unknown(kb, question, parsed)
        cot = []
        kb.closure()
        if atom:
            cot = kb.trace.get(atom, [f"Z3 checked entailment for {atom.label()}."])
        else:
            cot = ["Parsed premises into a Horn-rule knowledge base.", "Ran forward chaining and Z3 entailment checks."]
        explanation = self.explainer.explain(question, answer, atom, kb, premises_nl, cot)
        fol_evidence = atom.label() if atom else None
        return standard_response(answer, explanation, fol=fol_evidence, cot=cot, confidence=round(conf, 3))
