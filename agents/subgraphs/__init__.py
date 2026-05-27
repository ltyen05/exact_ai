"""Subgraphs package for specialized problem solving."""

from agents.subgraphs.physics_subgraph import build_physics_subgraph
from agents.subgraphs.logic_subgraph import build_logic_subgraph

__all__ = ["build_physics_subgraph", "build_logic_subgraph"]
