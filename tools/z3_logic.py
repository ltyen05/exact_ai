from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import z3  # type: ignore
except Exception:  # pragma: no cover
    z3 = None


@dataclass(frozen=True)
class Atom:
    pred: str
    args: Tuple[str, ...] = ("entity",)
    truth: bool = True

    def pos(self) -> "Atom":
        return Atom(self.pred, self.args, True)

    def neg(self) -> "Atom":
        return Atom(self.pred, self.args, False)

    def key(self) -> Tuple[str, Tuple[str, ...]]:
        return (self.pred, self.args)

    def label(self) -> str:
        args = ",".join(self.args)
        return f"{'¬' if not self.truth else ''}{self.pred}({args})"


@dataclass
class Rule:
    antecedents: List[Atom]
    consequent: Atom
    source: str = ""


class HornKB:
    """Finite Horn knowledge base with Z3 entailment verification.

    It uses forward chaining to materialize facts, and a Z3 SAT check to verify
    whether a candidate atom is entailed by facts + implications.
    """

    def __init__(self) -> None:
        self.facts: Set[Atom] = set()
        self.rules: List[Rule] = []
        self.trace: Dict[Atom, List[str]] = {}

    def add_fact(self, atom: Atom, source: str = "") -> None:
        self.facts.add(atom)
        if source:
            self.trace.setdefault(atom, []).append(source)

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def closure(self, max_iter: int = 100) -> Set[Atom]:
        known = set(self.facts)
        for _ in range(max_iter):
            changed = False
            for rule in self.rules:
                if all(a in known for a in rule.antecedents) and rule.consequent not in known:
                    known.add(rule.consequent)
                    chain = []
                    for a in rule.antecedents:
                        chain.extend(self.trace.get(a, [a.label()]))
                    chain.append(rule.source or f"{[a.label() for a in rule.antecedents]} -> {rule.consequent.label()}")
                    self.trace[rule.consequent] = chain
                    changed = True
            if not changed:
                break
        self.facts = known
        return known

    def _z3_var(self, atom: Atom, cache: Dict[Tuple[str, Tuple[str, ...]], object]):
        assert z3 is not None
        k = atom.key()
        if k not in cache:
            cache[k] = z3.Bool(atom.label().replace("¬", "not_"))
        return cache[k]

    def entails(self, atom: Atom) -> Optional[bool]:
        """Return True/False if Z3 can prove atom/negated atom, else None."""
        self.closure()
        if atom in self.facts:
            return True
        if atom.neg() in self.facts:
            return False
        if z3 is None:
            return None
        cache: Dict[Tuple[str, Tuple[str, ...]], object] = {}
        solver = z3.Solver()
        for f in self.facts:
            v = self._z3_var(f.pos(), cache)
            solver.add(v if f.truth else z3.Not(v))
        for r in self.rules:
            ants = []
            for a in r.antecedents:
                v = self._z3_var(a.pos(), cache)
                ants.append(v if a.truth else z3.Not(v))
            cv = self._z3_var(r.consequent.pos(), cache)
            cons = cv if r.consequent.truth else z3.Not(cv)
            solver.add(z3.Implies(z3.And(*ants), cons))
        # prove atom by contradiction
        s1 = z3.Solver()
        s1.add(solver.assertions())
        av = self._z3_var(atom.pos(), cache)
        s1.add(z3.Not(av) if atom.truth else av)
        if s1.check() == z3.unsat:
            return True
        # prove negation by contradiction
        s2 = z3.Solver()
        s2.add(solver.assertions())
        s2.add(av if atom.truth else z3.Not(av))
        if s2.check() == z3.unsat:
            return False
        return None


def pred_name(text: str) -> str:
    text = text.strip()
    text = re.sub(r"([a-z])([A-Z])", r"\1_\2", text)
    text = text.replace("-", "_")
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "predicate"


def _split_top_level(s: str, sep: str) -> List[str]:
    out, cur, depth = [], [], 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth == 0 and s.startswith(sep, i):
            out.append(''.join(cur).strip())
            cur = []
            i += len(sep)
            continue
        cur.append(ch)
        i += 1
    out.append(''.join(cur).strip())
    return [x for x in out if x]


def _strip_outer_parens(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        ok = True
        for i, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(text) - 1:
                    ok = False
                    break
        if ok:
            text = text[1:-1].strip()
        else:
            break
    return text


def parse_atom(s: str, var_bind: Optional[Dict[str, str]] = None) -> Optional[Atom]:
    var_bind = var_bind or {}
    s = s.strip()
    if not s:
        return None
    truth = True
    s = _strip_outer_parens(s)
    if s.startswith(("¬", "~")):
        truth = False
        s = s[1:].strip()
    if s.startswith("Not(") and s.endswith(")"):
        truth = False
        s = s[4:-1].strip()
    m = re.match(r"([A-Za-z_][\w]*)\s*\((.*)\)$", s)
    if not m:
        return None
    pred = pred_name(m.group(1))
    args_raw = [a.strip() for a in _split_top_level(m.group(2), ",")]
    args = tuple(var_bind.get(a, pred_name(a)) for a in args_raw if a)
    if not args:
        args = ("entity",)
    return Atom(pred, args, truth)


def _normalize_fol(s: str) -> str:
    s = s.replace("�forall", "ForAll")
    s = s.replace("∀", "ForAll ").replace("∃", "Exists ")
    s = s.replace("→", "->").replace("⇒", "->")
    s = s.replace("∧", "&").replace("∨", "|")
    s = s.replace("¬", "~")
    s = re.sub(r"ForAll\s*([a-zA-Z]\w*)\s*\(", r"ForAll(\1, ", s)
    s = re.sub(r"Exists\s*([a-zA-Z]\w*)\s*\(", r"Exists(\1, ", s)
    return s.strip()


def parse_fol_to_kb(fol_list: Sequence[str], domain_hint: Optional[Iterable[str]] = None) -> HornKB:
    kb = HornKB()
    domain: Set[str] = set(domain_hint or [])
    # first pass: constants
    for raw in fol_list:
        for name in re.findall(r"\(([A-Z][A-Za-z0-9_]*)\)", raw):
            domain.add(pred_name(name))
        for name in re.findall(r",\s*([A-Z][A-Za-z0-9_]*)\)", raw):
            domain.add(pred_name(name))
    if not domain:
        domain.add("entity")

    for i, raw in enumerate(fol_list, 1):
        s = _normalize_fol(str(raw))
        source = f"Premise {i}: {raw}"
        # ForAll(x, body)
        m = re.match(r"ForAll\s*\(\s*(\w+)\s*,\s*(.*)\)\s*$", s)
        if m:
            var, body = m.group(1), m.group(2).strip()
            parts = _split_top_level(body, "->") if "->" in body else []
            if len(parts) >= 2:
                left, right = parts[0], parts[1]
                left = _strip_outer_parens(left.strip())
                right = _strip_outer_parens(right.strip())
                ant_parts = _split_top_level(left, "&")
                for const in domain:
                    bind = {var: const}
                    ants = [parse_atom(p, bind) for p in ant_parts]
                    cons = parse_atom(right, bind)
                    if cons and all(ants):
                        kb.add_rule(Rule([a for a in ants if a], cons, source))
            else:
                # universal fact P(x): instantiate over known domain
                for const in domain:
                    atom = parse_atom(body, {var: const})
                    if atom:
                        kb.add_fact(atom, source)
            continue
        # Exists(x, body) -> create witness
        m = re.match(r"Exists\s*\(\s*(\w+)\s*,\s*(.*)\)\s*$", s)
        if m:
            var, body = m.group(1), _strip_outer_parens(m.group(2).strip())
            witness = f"some_{i}"
            parts = _split_top_level(body, "&")
            for p in parts:
                atom = parse_atom(p, {var: witness})
                if atom:
                    kb.add_fact(atom, source)
            continue
        # implication without explicit forall
        if "->" in s:
            parts = _split_top_level(_strip_outer_parens(s), "->")
            if len(parts) >= 2:
                left, right = parts[0], parts[1]
                ants = [parse_atom(_strip_outer_parens(p.strip())) for p in _split_top_level(_strip_outer_parens(left.strip()), "&")]
                cons = parse_atom(_strip_outer_parens(right.strip()))
                if cons and all(ants):
                    kb.add_rule(Rule([a for a in ants if a], cons, source))
            continue
        # fact or conjunction of facts
        for p in _split_top_level(_strip_outer_parens(s.strip()), "&"):
            atom = parse_atom(p)
            if atom:
                kb.add_fact(atom, source)
    kb.closure()
    return kb
