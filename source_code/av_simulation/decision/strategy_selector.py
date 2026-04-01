"""
decision/strategy_selector.py
==============================
StrategySelector — LLM reads fleet graph, selects strategy, writes
CoordPlan subgraph including Trajectory nodes for each vehicle (F7).

Thesis requirement F7:
  The leader's LLM must select the most appropriate strategy from the
  repository based on the current graph state and instantiate it (fill
  parameters).  The instantiated plan must be written as a CoordPlan
  subgraph.

How it works
------------
1. serialize fleet graph → text (SemanticGraph.serialize_to_text)
2. LLMReasoner.generate_intent() → intent_node (action, priority, params)
3. StrategyRepository.query_by_name(action) → graph_template
4. instantiate(template, fleet_state) → CoordPlan node + Trajectory nodes
5. Write CoordPlan subgraph to graph
6. Return plan_id so FleetCoordinator can broadcast the diff

Only runs on the LEADER agent (conditional activation, NF4).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from av_simulation.graph.graph import SemanticGraph
from av_simulation.decision.repository import StrategyRepository
from av_simulation.config import simulation_config as cfg

# LLMReasoner imported lazily inside select_and_instantiate to avoid
# the metadrive import chain at module load time.
_LLMReasoner = None

def _get_llm_reasoner_cls():
    global _LLMReasoner
    if _LLMReasoner is None:
        try:
            from av_simulation.coordination.llm_reasoner import LLMReasoner
            _LLMReasoner = LLMReasoner
        except ImportError:
            pass
    return _LLMReasoner


class StrategySelector:
    """
    Orchestrates: graph → LLM intent → strategy template → CoordPlan subgraph.

    Parameters
    ----------
    repository : StrategyRepository
    reasoner   : LLMReasoner
    """

    def __init__(
        self,
        repository: StrategyRepository,
        reasoner=None,   # LLMReasoner instance or None (lazy-created)
    ) -> None:
        self.repo     = repository
        self._reasoner = reasoner  # may be None; created on first call

    @property
    def reasoner(self):
        if self._reasoner is None:
            LLMReasonerCls = _get_llm_reasoner_cls()
            if LLMReasonerCls is not None:
                self._reasoner = LLMReasonerCls(quantize=True)
        return self._reasoner

    @reasoner.setter
    def reasoner(self, value):
        self._reasoner = value

    # ── Main entry point ──────────────────────────────────────────────────────

    def select_and_instantiate(
        self,
        graph:       SemanticGraph,
        fleet_state: dict,
        agent_id:    str = "leader",
    ) -> str:
        """
        Read the fleet graph, select a strategy, write a CoordPlan subgraph.

        Parameters
        ----------
        graph        : the leader's local SemanticGraph (will be mutated)
        fleet_state  : {
              "agent_ids":        List[str],
              "agents_positions": Dict[str, list],   # agent_id -> [x, y]
              "leader_id":        str,
              "obstacle_position": list or None,
          }
        agent_id     : leader's agent_id (used for graph node sourcing)

        Returns
        -------
        plan_id : str — the node_id of the CoordPlan node written to graph.
                  Pass this to FleetCoordinator so it knows which plan to use.
        """
        # 1. Serialise graph for LLM (cap at 30 nodes to stay within token budget)
        graph_subtext = graph.serialize_to_text(max_nodes=30)

        # 2. LLM generates intent (writes Intent + Message nodes to graph)
        intent_node, message_node = self.reasoner.generate_intent(
            graph_subtext,
            graph    = graph,
            agent_id = agent_id,
        )

        action = intent_node.get("action", "SpatiallyOrderedBypass")
        params = intent_node.get("params", {})

        # 3. Fetch graph template from repository
        template = self.repo.query_by_name(action)
        if template is None:
            template = self.repo.query_by_name("SpatiallyOrderedBypass")

        # 4. Merge LLM params into template defaults
        merged_params = dict(template.get("default_params", {}))
        merged_params.update(params)

        # 5. Instantiate: CoordPlan node + per-vehicle Trajectory nodes
        plan_id = self._instantiate(
            template      = template,
            merged_params = merged_params,
            graph         = graph,
            fleet_state   = fleet_state,
            agent_id      = agent_id,
            action        = action,
        )

        print(
            f"[F7] StrategySelector: selected '{action}' | "
            f"plan_id='{plan_id}' | "
            f"vehicles={len(fleet_state.get('agent_ids', []))}"
        )
        return plan_id

    # ── Instantiation ──────────────────────────────────────────────────────────

    def _instantiate(
        self,
        template:      dict,
        merged_params: dict,
        graph:         SemanticGraph,
        fleet_state:   dict,
        agent_id:      str,
        action:        str,
    ) -> str:
        """
        Fill template placeholders with real values and write the CoordPlan
        subgraph (CoordPlan node + Trajectory nodes + edges) to *graph*.
        """
        ts        = time.time()
        plan_id   = f"coordplan_{int(ts * 1000)}"
        agent_ids = fleet_state.get("agent_ids", [])
        positions = fleet_state.get("agents_positions", {})
        obs_pos   = fleet_state.get("obstacle_position") or cfg.OBSTACLE_POSITION

        # Sort agents by longitudinal position (furthest ahead first)
        vehicle_order = sorted(
            [a for a in agent_ids if a in positions],
            key=lambda a: positions[a][0],
            reverse=True,
        )
        n = len(vehicle_order)

        merge_side = merged_params.get("merge_side", "left")
        spacing    = merged_params.get("spacing", cfg.PLATOON_SPACING)
        speed      = cfg.LEADER_SPEED

        # ── CoordPlan node ─────────────────────────────────────────────────────
        coord_attrs = self._fill_placeholders(
            template.get("coord_plan_attrs", {}),
            {
                "__N__":           n,
                "__MERGE_SIDE__":  merge_side,
                "__SPACING__":     spacing,
                "__STOP_DIST__":   merged_params.get("stop_distance", 15.0),
            },
        )
        coord_attrs["timestamp"]   = ts
        coord_attrs["strategy_name"] = action
        coord_attrs["parameters"]  = merged_params

        graph.add_node(plan_id, "CoordPlan", coord_attrs, source=agent_id, timestamp=ts)

        # ── Per-vehicle Trajectory nodes ───────────────────────────────────────
        lateral_sign = 1.0 if merge_side == "left" else -1.0
        obs_np = np.array(obs_pos) if obs_pos else np.array([cfg.OBSTACLE_LONGITUDE, 0.0])

        for slot, vid in enumerate(vehicle_order):
            pos = np.array(positions[vid][:2])

            if action == "StopAndWaitQueue":
                wps = self._compute_stop_waypoints(
                    pos, obs_np, merged_params.get("stop_distance", 15.0)
                )
                spd_profile = [0.0] * len(wps)
            else:
                lateral_offset = lateral_sign * (4.0 if slot % 2 == 0 else 3.0)
                wps = self._compute_bypass_waypoints(
                    pos, obs_np, lateral_offset, speed, slot, spacing
                )
                spd_profile = [speed] * len(wps)

            traj_id   = f"traj_{vid[:8]}_{int(ts)}"
            traj_attrs = self._fill_placeholders(
                template.get("trajectory_schema", {}),
                {
                    "__AGENT_ID__": vid,
                    "__WPS__":      [list(w) for w in wps],
                    "__SPD__":      spd_profile,
                    "__SLOT__":     slot,
                },
            )
            traj_attrs["agent_id"]   = vid
            traj_attrs["slot_index"] = slot
            traj_attrs["timestamp"]  = ts

            graph.add_node(traj_id, "Trajectory", traj_attrs,
                           source=agent_id, timestamp=ts)

            # CoordPlan --contains_trajectory--> Trajectory
            graph.add_edge(plan_id, traj_id, "contains_trajectory",
                           {}, agent_id, ts)

            # Vehicle --assigned_trajectory--> Trajectory
            vehicle_node_id = f"vehicle_{vid}"
            graph.add_edge(vehicle_node_id, traj_id, "assigned_trajectory",
                           {"slot": slot}, agent_id, ts)

            print(
                f"[F7]   {vid[:8]} slot={slot} "
                f"wps={len(wps)} "
                f"action={action}"
            )

        # ── Leader -> CoordPlan edge ────────────────────────────────────────────
        leader_node_id = f"vehicle_{agent_id}"
        graph.add_edge(leader_node_id, plan_id, "has_plan",
                       {}, agent_id, ts)

        return plan_id

    # ── Waypoint computation ───────────────────────────────────────────────────

    def _compute_bypass_waypoints(
        self,
        start_pos:      np.ndarray,
        obstacle_pos:   np.ndarray,
        lateral_offset: float,
        speed:          float,
        slot:           int,
        spacing:        float,
    ) -> List[Tuple[float, float, float]]:
        """
        Generate a smooth bypass path as a list of (x, y, speed) waypoints.

        The path has three phases:
          1. Approach  — decelerate slightly, shift laterally
          2. Bypass    — hold lateral offset past the obstacle
          3. Return    — merge back to centre lane
        """
        obs_x, obs_y = float(obstacle_pos[0]), float(obstacle_pos[1])
        sx, sy       = float(start_pos[0]),    float(start_pos[1])

        # Time-gap for this slot
        time_gap = slot * max(0.4, spacing / max(speed, 1.0))

        # Pre-obstacle: shift starts 30 m before obstacle
        p1_x = obs_x - 30.0
        p1_y = sy + lateral_offset * 0.3   # partial lateral shift

        # At obstacle
        p2_x = obs_x + 5.0
        p2_y = lateral_offset              # full lateral offset

        # Clear of obstacle
        p3_x = obs_x + 20.0
        p3_y = lateral_offset

        # Return to centre lane
        p4_x = obs_x + 45.0
        p4_y = 0.0

        wps = []
        for x, y in [(p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y), (p4_x, p4_y)]:
            # Only include waypoints ahead of the current position
            if x > sx:
                wps.append((round(x, 2), round(y, 2), round(speed, 2)))

        if not wps:
            # Agent is already past obstacle — just push forward
            wps = [(round(sx + 20.0, 2), 0.0, round(speed, 2))]

        return wps

    def _compute_stop_waypoints(
        self,
        start_pos:     np.ndarray,
        obstacle_pos:  np.ndarray,
        stop_distance: float,
    ) -> List[Tuple[float, float, float]]:
        """
        Generate a deceleration path ending at *stop_distance* before obstacle.
        """
        obs_x = float(obstacle_pos[0])
        sx    = float(start_pos[0])
        sy    = float(start_pos[1])

        stop_x = obs_x - stop_distance
        if stop_x <= sx:
            # Already close — hold position
            return [(round(sx, 2), round(sy, 2), 0.0)]

        mid_x = sx + (stop_x - sx) * 0.6
        return [
            (round(mid_x, 2),  round(sy, 2), round(cfg.LEADER_SPEED * 0.5, 2)),
            (round(stop_x, 2), round(sy, 2), 0.0),
        ]

    # ── Template placeholder fill ──────────────────────────────────────────────

    @staticmethod
    def _fill_placeholders(template_dict: dict, values: dict) -> dict:
        """
        Deep-copy *template_dict*, replacing string placeholder tokens with
        values from *values*.

        Non-string values (lists, ints, floats) replace the entire field.
        String tokens that don't match a key are left unchanged.
        """
        result = {}
        for k, v in template_dict.items():
            if isinstance(v, str) and v in values:
                result[k] = values[v]       # replace entire string token
            elif isinstance(v, str):
                # Replace tokens embedded in a longer string
                filled = v
                for token, replacement in values.items():
                    filled = filled.replace(token, str(replacement))
                result[k] = filled
            else:
                result[k] = v
        return result