"""
diff.py
=======
GraphDiff — the unit of V2V graph synchronisation (F3).

A GraphDiff captures everything that changed in one agent's local
SemanticGraph since its last broadcast.  It is serialised to JSON,
broadcast over the V2V bus, and merged into every peer's local graph.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Tuple

from av_simulation.graph.graph import Node, Edge


@dataclass
class GraphDiff:
    """
    Represents the delta between two versions of a SemanticGraph.

    Fields
    ------
    added_nodes    : Nodes that are new since the last broadcast.
    updated_nodes  : Nodes whose attrs changed since the last broadcast.
    removed_nodes  : IDs of nodes that were deleted.
    added_edges    : Edges that are new since the last broadcast.
    removed_edges  : (from_id, to_id, label) tuples of deleted edges.
    sender_version : The graph's _version counter at time of diff creation.
                     Recipients can use this to detect missing diffs.
    """
    added_nodes:    List[Node]
    updated_nodes:  List[Node]
    removed_nodes:  List[str]                    # node_ids
    added_edges:    List[Edge]
    removed_edges:  List[Tuple[str, str, str]]   # (from_id, to_id, label)
    sender_version: int = 0

    # ── Inspection ─────────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        """True if nothing changed — no broadcast needed."""
        return not any([
            self.added_nodes,
            self.updated_nodes,
            self.removed_nodes,
            self.added_edges,
            self.removed_edges,
        ])

    def node_count(self) -> int:
        return len(self.added_nodes) + len(self.updated_nodes)

    def __repr__(self) -> str:
        return (
            f"GraphDiff("
            f"+nodes={len(self.added_nodes)}, "
            f"~nodes={len(self.updated_nodes)}, "
            f"-nodes={len(self.removed_nodes)}, "
            f"+edges={len(self.added_edges)}, "
            f"-edges={len(self.removed_edges)}, "
            f"v={self.sender_version})"
        )

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "added_nodes":    [n.to_dict() for n in self.added_nodes],
            "updated_nodes":  [n.to_dict() for n in self.updated_nodes],
            "removed_nodes":  self.removed_nodes,
            "added_edges":    [e.to_dict() for e in self.added_edges],
            "removed_edges":  [list(t) for t in self.removed_edges],
            "sender_version": self.sender_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "GraphDiff":
        return cls(
            added_nodes    = [Node.from_dict(n) for n in d.get("added_nodes",   [])],
            updated_nodes  = [Node.from_dict(n) for n in d.get("updated_nodes", [])],
            removed_nodes  = d.get("removed_nodes", []),
            added_edges    = [Edge.from_dict(e) for e in d.get("added_edges",   [])],
            removed_edges  = [tuple(t) for t in d.get("removed_edges",  [])],
            sender_version = d.get("sender_version", 0),
        )

    @classmethod
    def from_json(cls, s: str) -> "GraphDiff":
        return cls.from_dict(json.loads(s))