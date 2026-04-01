"""
decision/repository.py
======================
StrategyRepository — graph-pattern templates for fleet coordination (F6).

Thesis requirement F6:
  A repository of parameterised strategies (Stop-and-Wait Queue,
  Spatially-Ordered Bypass, Distributed Lane Swap) must be available.
  Each strategy is a template (graph pattern) with placeholders for
  parameters (passing order, lateral offset, etc.).

Phase 2 change vs. original StrategyRepository in fleet_coordinator.py:
  - Each strategy now includes a `graph_template` dict describing the
    CoordPlan node structure and Trajectory placeholders.
  - query_by_name() returns the template directly (used by StrategySelector).
  - Original condition-based query() still works for backward compat.
  - Templates use placeholder tokens (__N__, __WPS__, __SPD__) that
    StrategySelector.instantiate() fills with real values.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional


# ── Template definitions ───────────────────────────────────────────────────────
#
# Each strategy template describes:
#   "coord_plan_attrs"  — attributes for the CoordPlan graph node
#   "trajectory_schema" — per-vehicle Trajectory node schema (placeholders)
#   "edges"             — edges to create between nodes
#
# Placeholder tokens (replaced at instantiation time):
#   __AGENT_ID__        — replaced with each agent's ID
#   __N__               — replaced with number of agents
#   __WPS__             — replaced with computed waypoints list
#   __SPD__             — replaced with target speed float
#   __SLOT__            — replaced with agent's position index in vehicle_order
#   __MERGE_SIDE__      — replaced with "left" or "right"
#   __SPACING__         — replaced with inter-vehicle spacing (metres)

STRATEGY_TEMPLATES: Dict[str, dict] = {

    "SpatiallyOrderedBypass": {
        "description": (
            "Vehicles bypass the obstacle one at a time in spatial order "
            "(furthest-ahead first). Each vehicle follows an individual "
            "lateral-offset waypoint trajectory."
        ),
        "default_params": {
            "merge_side": "left",
            "spacing":    6.0,
        },
        "coord_plan_attrs": {
            "strategy_name": "SpatiallyOrderedBypass",
            "merge_side":    "__MERGE_SIDE__",
            "spacing":       "__SPACING__",
            "num_vehicles":  "__N__",
        },
        "trajectory_schema": {
            "waypoints":     "__WPS__",
            "speed_profile": "__SPD__",
            "agent_id":      "__AGENT_ID__",
            "slot_index":    "__SLOT__",
        },
        "edges": [
            # CoordPlan -> Trajectory (one per vehicle)
            {"from": "coord_plan", "to": "traj___AGENT_ID__",
             "label": "contains_trajectory"},
            # Vehicle -> Trajectory
            {"from": "vehicle___AGENT_ID__", "to": "traj___AGENT_ID__",
             "label": "assigned_trajectory"},
        ],
        "condition": "obstacle_width > lane_width",
    },

    "StopAndWaitQueue": {
        "description": (
            "All vehicles stop at safe distance behind the obstacle. "
            "Fleet waits until obstacle is cleared or reassessment triggers "
            "a strategy change."
        ),
        "default_params": {
            "stop_distance": 15.0,
        },
        "coord_plan_attrs": {
            "strategy_name":   "StopAndWaitQueue",
            "stop_distance":   "__STOP_DIST__",
            "num_vehicles":    "__N__",
            "action":          "stop",
        },
        "trajectory_schema": {
            "waypoints":     "__WPS__",   # short decel-to-stop path
            "speed_profile": 0.0,
            "agent_id":      "__AGENT_ID__",
            "slot_index":    "__SLOT__",
        },
        "edges": [
            {"from": "coord_plan", "to": "traj___AGENT_ID__",
             "label": "contains_trajectory"},
            {"from": "vehicle___AGENT_ID__", "to": "traj___AGENT_ID__",
             "label": "assigned_trajectory"},
        ],
        "condition": "obstacle_count > 1",
    },

    "DistributedLaneSwap": {
        "description": (
            "Vehicles perform a coordinated lane swap to pass a partially "
            "blocking obstacle. Odd-indexed vehicles offset right, "
            "even-indexed offset left."
        ),
        "default_params": {
            "merge_side": "left",
            "spacing":    4.0,
        },
        "coord_plan_attrs": {
            "strategy_name": "DistributedLaneSwap",
            "merge_side":    "__MERGE_SIDE__",
            "spacing":       "__SPACING__",
            "num_vehicles":  "__N__",
        },
        "trajectory_schema": {
            "waypoints":     "__WPS__",
            "speed_profile": "__SPD__",
            "agent_id":      "__AGENT_ID__",
            "slot_index":    "__SLOT__",
        },
        "edges": [
            {"from": "coord_plan", "to": "traj___AGENT_ID__",
             "label": "contains_trajectory"},
            {"from": "vehicle___AGENT_ID__", "to": "traj___AGENT_ID__",
             "label": "assigned_trajectory"},
        ],
        "condition": "obstacle_width <= lane_width * 0.5",
    },

    "MaintainFormation": {
        "description": (
            "No confirmed obstacle — keep platoon formation and continue."
        ),
        "default_params": {},
        "coord_plan_attrs": {
            "strategy_name": "MaintainFormation",
            "num_vehicles":  "__N__",
            "action":        "continue",
        },
        "trajectory_schema": {
            "waypoints":     "__WPS__",
            "speed_profile": "__SPD__",
            "agent_id":      "__AGENT_ID__",
            "slot_index":    "__SLOT__",
        },
        "edges": [
            {"from": "vehicle___AGENT_ID__", "to": "traj___AGENT_ID__",
             "label": "assigned_trajectory"},
        ],
        "condition": "no_obstacle",
    },
}

# Map LLM intent action names -> canonical strategy name
ACTION_TO_STRATEGY: Dict[str, str] = {
    "SpatiallyOrderedBypass":  "SpatiallyOrderedBypass",
    "StopAndWaitQueue":        "StopAndWaitQueue",
    "DistributedLaneSwap":     "DistributedLaneSwap",
    "MaintainFormation":       "MaintainFormation",
    # Aliases (in case LLM uses shorthand)
    "bypass":                  "SpatiallyOrderedBypass",
    "stop":                    "StopAndWaitQueue",
    "swap":                    "DistributedLaneSwap",
    "maintain":                "MaintainFormation",
    "time_space_reservation":  "SpatiallyOrderedBypass",   # legacy compat
    "single_lane_bypass":      "SpatiallyOrderedBypass",
    "full_stop_reassess":      "StopAndWaitQueue",
    "yield_and_stop":          "StopAndWaitQueue",
}


# ── StrategyRepository ─────────────────────────────────────────────────────────

class StrategyRepository:
    """
    SQLite-backed strategy repository with graph template support (F6).

    The original condition-based query() method is retained for backward
    compatibility with FleetCoordinator.execute_strategy_pipeline().

    New in Phase 2:
      - query_by_name(name)  → returns the full strategy dict including
                               graph_template, used by StrategySelector.
      - list_strategies()    → returns all available strategy names.
    """

    def __init__(self) -> None:
        self.conn   = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self) -> None:
        self.cursor.execute(
            """CREATE TABLE strategies (
                id             INTEGER PRIMARY KEY,
                name           TEXT UNIQUE,
                condition      TEXT,
                parameters     TEXT,
                graph_template TEXT,
                description    TEXT
            )"""
        )
        for name, tmpl in STRATEGY_TEMPLATES.items():
            self.cursor.execute(
                "INSERT INTO strategies "
                "(name, condition, parameters, graph_template, description) "
                "VALUES (?,?,?,?,?)",
                (
                    name,
                    tmpl["condition"],
                    json.dumps(tmpl["default_params"]),
                    json.dumps(tmpl),          # full template stored as JSON
                    tmpl["description"],
                ),
            )
        self.conn.commit()
        print(
            f"[STAGE 5] Strategy repository ready "
            f"({len(STRATEGY_TEMPLATES)} strategies with graph templates)."
        )

    # ── Original condition-based query (backward compat) ──────────────────────

    def query(self, condition_key: str) -> dict:
        """
        Original interface: maps a condition keyword to a strategy.
        Still used by FleetCoordinator.execute_strategy_pipeline().
        """
        cmap = {
            "lane_blockage":    "obstacle_width > lane_width",
            "partial_blockage": "obstacle_width <= lane_width * 0.5",
            "multi_obstacle":   "obstacle_count > 1",
            "emergency":        "emergency_vehicle_nearby",
            "no_obstacle":      "no_obstacle",
        }
        cond = cmap.get(condition_key, condition_key)
        self.cursor.execute(
            "SELECT name, parameters FROM strategies WHERE condition=?",
            (cond,),
        )
        row = self.cursor.fetchone()
        if row:
            name, pj = row
            print(f"[STAGE 5] Query '{condition_key}' -> {name}")
            return {"name": name, "params": json.loads(pj)}
        return {"name": "SpatiallyOrderedBypass",
                "params": STRATEGY_TEMPLATES["SpatiallyOrderedBypass"]["default_params"]}

    # ── New: name-based lookup with full template (F6) ─────────────────────────

    def query_by_name(self, name: str) -> Optional[dict]:
        """
        Return the full strategy dict (including graph_template) for *name*.
        Resolves aliases via ACTION_TO_STRATEGY.

        Returns None if not found.
        """
        canonical = ACTION_TO_STRATEGY.get(name, name)
        self.cursor.execute(
            "SELECT graph_template FROM strategies WHERE name=?",
            (canonical,),
        )
        row = self.cursor.fetchone()
        if row:
            tmpl = json.loads(row[0])
            print(f"[STAGE 5] Template retrieved: {canonical}")
            return tmpl
        # Try case-insensitive fallback
        self.cursor.execute(
            "SELECT name, graph_template FROM strategies "
            "WHERE LOWER(name) = LOWER(?)",
            (name,),
        )
        row = self.cursor.fetchone()
        if row:
            tmpl = json.loads(row[1])
            print(f"[STAGE 5] Template retrieved (fuzzy): {row[0]}")
            return tmpl
        print(f"[STAGE 5] Strategy '{name}' not found — using SpatiallyOrderedBypass")
        return STRATEGY_TEMPLATES.get("SpatiallyOrderedBypass")

    def list_strategies(self) -> List[str]:
        """Return all strategy names."""
        self.cursor.execute("SELECT name FROM strategies")
        return [r[0] for r in self.cursor.fetchall()]

    def get_description(self, name: str) -> str:
        canonical = ACTION_TO_STRATEGY.get(name, name)
        self.cursor.execute(
            "SELECT description FROM strategies WHERE name=?",
            (canonical,),
        )
        row = self.cursor.fetchone()
        return row[0] if row else "Unknown strategy."