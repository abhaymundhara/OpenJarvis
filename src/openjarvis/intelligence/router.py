"""Backward-compat shim — canonical location is learning.router."""

from openjarvis.learning.router import (  # noqa: F401
    DefaultQueryAnalyzer,
    HeuristicRouter,
    build_routing_context,
)

__all__ = ["DefaultQueryAnalyzer", "HeuristicRouter", "build_routing_context"]
