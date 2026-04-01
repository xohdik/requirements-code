"""
graph.py
========
Shared Semantic Graph  G = (V, E, l, A)

The central data structure for VLA-MAC. Every layer — Perception,
Coordination, Decision, Execution, V2V Sync — communicates exclusively
by reading from and writing to this graph.

Node types : Obstacle, Vehicle, Lane, Intent, Message,
             CoordPlan, Trajectory, ExecutionStatus
Edge labels: blocks, has_intent, sends, assigned_trajectory, executed
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Data primitives ────────────────────────────────────────────────────────────

@dataclass
class Node:
    node_id:   str
    node_type: str
    attrs:     Dict[str, Any]
    source:    str          # agent_id that wrote this node
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "node_id":   self.node_id,
            "node_type": self.node_type,
            "attrs":     self.attrs,
            "source":    self.source,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            node_id   = d["node_id"],
            node_type = d["node_type"],
            attrs     = d["attrs"],
            source    = d["source"],
            timestamp = d["timestamp"],
        )


@dataclass
class Edge:
    from_id:   str
    to_id:     str
    label:     str
    attrs:     Dict[str, Any]
    source:    str
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "from_id":   self.from_id,
            "to_id":     self.to_id,
            "label":     self.label,
            "attrs":     self.attrs,
            "source":    self.source,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(
            from_id   = d["from_id"],
            to_id     = d["to_id"],
            label     = d["label"],
            attrs     = d["attrs"],
            source    = d["source"],
            timestamp = d["timestamp"],
        )


# ── SemanticGraph ──────────────────────────────────────────────────────────────

class SemanticGraph:
    """
    Versioned, mergeable Shared Semantic Graph.

    Each agent maintains its own local copy. Changes are propagated
    fleet-wide via GraphDiff objects broadcast over the V2V bus
    (see coordination/fleet_coordinator.py :: V2VBus.broadcast_graph_diff).

    Conflict resolution: timestamp-based — the newer write always wins.
    This matches F3 of the thesis requirements.
    """

    def __init__(self) -> None:
        self.nodes:    Dict[str, Node] = {}
        self.edges:    List[Edge]      = []
        self._version: int             = 0

    # ── Write API ──────────────────────────────────────────────────────────────

    def add_node(
        self,
        node_id:   str,
        node_type: str,
        attrs:     Dict[str, Any],
        source:    str,
        timestamp: Optional[float] = None,
    ) -> Node:
        """
        Insert or update a node. If a node with the same ID already exists
        the newer timestamp wins (timestamp-based conflict resolution, F3).
        """
        ts   = timestamp if timestamp is not None else time.time()
        node = Node(node_id, node_type, attrs, source, ts)

        existing = self.nodes.get(node_id)
        if existing is None or ts >= existing.timestamp:
            self.nodes[node_id] = node
            self._version += 1

        return self.nodes[node_id]

    def add_edge(
        self,
        from_id:   str,
        to_id:     str,
        label:     str,
        attrs:     Dict[str, Any],
        source:    str,
        timestamp: Optional[float] = None,
    ) -> Edge:
        """
        Insert or replace an edge (from, to, label). Duplicates are removed
        before inserting so the graph always holds the latest version.
        """
        ts   = timestamp if timestamp is not None else time.time()
        edge = Edge(from_id, to_id, label, attrs, source, ts)

        self.edges = [
            e for e in self.edges
            if not (e.from_id == from_id
                    and e.to_id == to_id
                    and e.label == label)
        ]
        self.edges.append(edge)
        self._version += 1
        return edge

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all edges that reference it."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.edges = [
                e for e in self.edges
                if e.from_id != node_id and e.to_id != node_id
            ]
            self._version += 1

    def update_node_attrs(
        self,
        node_id: str,
        updates: Dict[str, Any],
        source:  str,
    ) -> Optional[Node]:
        """Merge *updates* into an existing node's attrs dict."""
        node = self.nodes.get(node_id)
        if node is None:
            return None
        node.attrs.update(updates)
        node.timestamp = time.time()
        node.source    = source
        self._version += 1
        return node

    # ── Read API ───────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def query(
        self,
        node_type: Optional[str]           = None,
        filters:   Optional[Dict[str, Any]] = None,
    ) -> List[Node]:
        """
        Return nodes matching node_type and/or attribute filters.

        Example
        -------
        # Find all Obstacle nodes with confidence > 0.7
        obstacles = graph.query("Obstacle")
        high_conf = [n for n in obstacles if n.attrs.get("confidence", 0) > 0.7]
        """
        results = list(self.nodes.values())
        if node_type:
            results = [n for n in results if n.node_type == node_type]
        if filters:
            results = [
                n for n in results
                if all(n.attrs.get(k) == v for k, v in filters.items())
            ]
        return results

    def get_edges(
        self,
        from_id: Optional[str] = None,
        to_id:   Optional[str] = None,
        label:   Optional[str] = None,
    ) -> List[Edge]:
        edges = self.edges
        if from_id is not None:
            edges = [e for e in edges if e.from_id == from_id]
        if to_id is not None:
            edges = [e for e in edges if e.to_id == to_id]
        if label is not None:
            edges = [e for e in edges if e.label == label]
        return edges

    def get_leader(self) -> Optional[Node]:
        """Convenience: return the Vehicle node where is_leader == True."""
        leaders = self.query("Vehicle", {"is_leader": True})
        return leaders[0] if leaders else None

    def get_obstacles(self) -> List[Node]:
        """Return all Obstacle nodes sorted by confidence descending."""
        obs = self.query("Obstacle")
        return sorted(obs, key=lambda n: n.attrs.get("confidence", 0), reverse=True)

    def get_trajectory(self, agent_id: str) -> Optional[Node]:
        """Return the Trajectory node assigned to *agent_id* (if any)."""
        vehicle_id = f"vehicle_{agent_id}"
        traj_edges = self.get_edges(from_id=vehicle_id, label="assigned_trajectory")
        if not traj_edges:
            return None
        return self.get_node(traj_edges[0].to_id)

    # ── Diff / Merge (F3 — V2V graph synchronisation) ─────────────────────────

    def snapshot(self) -> "SemanticGraph":
        """Return a deep copy for diff comparison. Call before mutating."""
        return copy.deepcopy(self)

    def diff(self, prev: "SemanticGraph") -> "GraphDiff":
        """
        Compute what changed between *prev* and *self*.

        Returns a GraphDiff that can be serialised to JSON and broadcast
        over the V2V bus.
        """
        from av_simulation.graph.diff import GraphDiff  # local import — avoids circular

        added_nodes   = [n for nid, n in self.nodes.items() if nid not in prev.nodes]
        updated_nodes = [
            n for nid, n in self.nodes.items()
            if nid in prev.nodes and n.attrs != prev.nodes[nid].attrs
        ]
        removed_nodes = [nid for nid in prev.nodes if nid not in self.nodes]

        prev_edge_keys = {(e.from_id, e.to_id, e.label) for e in prev.edges}
        curr_edge_keys = {(e.from_id, e.to_id, e.label) for e in self.edges}
        added_edges    = [
            e for e in self.edges
            if (e.from_id, e.to_id, e.label) not in prev_edge_keys
        ]
        removed_edges  = [k for k in prev_edge_keys if k not in curr_edge_keys]

        return GraphDiff(
            added_nodes    = added_nodes,
            updated_nodes  = updated_nodes,
            removed_nodes  = removed_nodes,
            added_edges    = added_edges,
            removed_edges  = removed_edges,
            sender_version = self._version,
        )

    def merge_diff(self, diff: "GraphDiff") -> None:
        """
        Apply a remote diff into the local graph.
        Timestamp-based conflict resolution: newer timestamp always wins (F3).
        """
        for node in diff.added_nodes + diff.updated_nodes:
            existing = self.nodes.get(node.node_id)
            if existing is None or node.timestamp >= existing.timestamp:
                self.nodes[node.node_id] = node
                self._version += 1

        for node_id in diff.removed_nodes:
            self.remove_node(node_id)

        for edge in diff.added_edges:
            self.add_edge(
                edge.from_id, edge.to_id, edge.label,
                edge.attrs, edge.source, edge.timestamp,
            )

        for from_id, to_id, label in diff.removed_edges:
            self.edges = [
                e for e in self.edges
                if not (e.from_id == from_id
                        and e.to_id == to_id
                        and e.label == label)
            ]

    def has_changed(self, version_at_last_broadcast: int) -> bool:
        """True if the graph changed since the given version number."""
        return self._version > version_at_last_broadcast

    # ── Serialisation ──────────────────────────────────────────────────────────

    def serialize_to_text(self, max_nodes: int = 30) -> str:
        """
        Compact text representation for LLM prompts (F5, F7).
        Truncates to *max_nodes* most-recently-written nodes.
        """
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.timestamp,
            reverse=True,
        )[:max_nodes]

        lines = []
        for n in sorted_nodes:
            lines.append(f"[{n.node_type}] {n.node_id}: {json.dumps(n.attrs)}")
        for e in self.edges:
            lines.append(f"  EDGE {e.from_id} --{e.label}--> {e.to_id}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "nodes":   [n.to_dict() for n in self.nodes.values()],
            "edges":   [e.to_dict() for e in self.edges],
            "version": self._version,
        }, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "SemanticGraph":
        data = json.loads(s)
        g    = cls()
        for nd in data["nodes"]:
            n = Node.from_dict(nd)
            g.nodes[n.node_id] = n
        for ed in data["edges"]:
            g.edges.append(Edge.from_dict(ed))
        g._version = data.get("version", 0)
        return g

    # ── Helpers ────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        type_counts: Dict[str, int] = {}
        for n in self.nodes.values():
            type_counts[n.node_type] = type_counts.get(n.node_type, 0) + 1
        parts = ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items()))
        return f"SemanticGraph(v{self._version} | {parts} | edges={len(self.edges)})"

    def __repr__(self) -> str:
        return self.summary()