"""SymPy execution service for validated physics solve contexts."""

from __future__ import annotations

from typing import Any

from tools.calculator import solve_with_sympy_trace

from .validator import ValidatedSolveContext


class SympyExecutor:
    """Execute validated equation systems without formatting public answers."""

    def solve(self, context: ValidatedSolveContext) -> Any:
        return solve_with_sympy_trace(context.quantities, context.equations, context.target)
