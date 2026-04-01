"""
fleet_coordinator.py
====================
Phase 1 changes (marked  # [P1]):
  - V2VBus.broadcast_graph_diff()  : sends GraphDiff objects over V2V (F3)
  - V2VBus.broadcast()            : handles GRAPH_DIFF type in subscribers
  - SimpleRaft.update_fitness()   : writes Vehicle node to agent's graph (F4)
  - SimpleRaft.elect_leader()     : writes is_leader attr to Vehicle nodes (F4)

Everything else is unchanged from the original file.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from av_simulation.config import simulation_config as cfg
# WaypointPlanner is imported lazily inside LLMStrategyAdapter.__init__
# to avoid pulling in metadrive at module load time.

# [P2] StrategySelector — imported lazily inside execute_strategy_pipeline
# to avoid circular imports at module load time.
_StrategySelector = None

def _get_strategy_selector():
    global _StrategySelector
    if _StrategySelector is None:
        try:
            from av_simulation.decision.strategy_selector import StrategySelector
            _StrategySelector = StrategySelector
        except ImportError:
            pass
    return _StrategySelector


# ── V2V Bus ───────────────────────────────────────────────────────────────────

class V2VBus:
    """Vehicle-to-Vehicle communication bus for inter-agent messaging."""

    def __init__(self) -> None:
        self.messages:    List[dict]          = []
        self.subscribers: Dict[str, callable] = {}

    def broadcast(self, sender_id: str, msg_type: str, payload: dict) -> dict:
        """Publish a message to all subscribers except the sender."""
        msg = {
            "sender":    sender_id,
            "type":      msg_type,
            "payload":   payload,
            "timestamp": time.time(),
        }
        self.messages.append(msg)

        # Suppress noisy per-step graph diff prints; keep other stage logs
        if msg_type != "GRAPH_DIFF":
            print(f"[STAGE 2] V2V Broadcast {sender_id[:8]}: {msg_type}")

        for aid, cb in self.subscribers.items():
            if aid != sender_id:
                cb(msg)
        return msg

    # [P1] ── Graph diff broadcast (F3) ────────────────────────────────────────

    def broadcast_graph_diff(self, sender_id: str, diff) -> None:
        """
        Serialise *diff* (a GraphDiff) and broadcast it as a GRAPH_DIFF
        message.  Recipients must call graph.merge_diff(diff) in their
        V2V callback.

        Parameters
        ----------
        sender_id : str       — agent_id of the broadcasting agent
        diff      : GraphDiff — produced by SemanticGraph.diff(prev_snapshot)
        """
        if diff.is_empty():
            return  # nothing changed — save bandwidth

        payload = diff.to_dict()   # serialise to plain dict (JSON-safe)
        print(
            f"[GRAPH-SYNC] {sender_id[:8]} broadcasting diff: "
            f"+{len(diff.added_nodes)} nodes, "
            f"~{len(diff.updated_nodes)} updates, "
            f"+{len(diff.added_edges)} edges"
        )
        self.broadcast(sender_id, "GRAPH_DIFF", payload)

    # ── Standard helpers ───────────────────────────────────────────────────────

    def subscribe(self, agent_id: str, callback) -> None:
        self.subscribers[agent_id] = callback

    def get_recent_messages(
        self,
        message_type:     Optional[str]   = None,
        since_timestamp:  Optional[float] = None,
    ) -> List[dict]:
        msgs = self.messages
        if message_type:
            msgs = [m for m in msgs if m["type"] == message_type]
        if since_timestamp:
            msgs = [m for m in msgs if m["timestamp"] > since_timestamp]
        return msgs


# ── Strategy Repository ───────────────────────────────────────────────────────

class StrategyRepository:

    def __init__(self) -> None:
        self.conn   = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self) -> None:
        self.cursor.execute(
            """CREATE TABLE strategies (
                id INTEGER PRIMARY KEY,
                condition TEXT,
                strategy_name TEXT,
                parameters TEXT
            )"""
        )
        rows = [
            (
                "obstacle_width > lane_width",
                "time_space_reservation",
                json.dumps({"merge_side": "adaptive", "spacing": 6.0}),
            ),
            (
                "obstacle_width <= lane_width * 0.5",
                "single_lane_bypass",
                json.dumps({"merge_side": "left", "spacing": 4.0}),
            ),
            (
                "obstacle_count > 1",
                "full_stop_reassess",
                json.dumps({"stop_distance": 15.0}),
            ),
            (
                "emergency_vehicle_nearby",
                "yield_and_stop",
                json.dumps({"yield_distance": 30.0}),
            ),
        ]
        self.cursor.executemany(
            "INSERT INTO strategies (condition, strategy_name, parameters) "
            "VALUES (?,?,?)",
            rows,
        )
        self.conn.commit()
        print("[STAGE 5] Strategy repository ready.")

    def query(self, condition_key: str) -> dict:
        cmap = {
            "lane_blockage":    "obstacle_width > lane_width",
            "partial_blockage": "obstacle_width <= lane_width * 0.5",
            "multi_obstacle":   "obstacle_count > 1",
            "emergency":        "emergency_vehicle_nearby",
        }
        cond = cmap.get(condition_key, condition_key)
        self.cursor.execute(
            "SELECT strategy_name, parameters FROM strategies WHERE condition=?",
            (cond,),
        )
        row = self.cursor.fetchone()
        if row:
            name, pj = row
            print(f"[STAGE 5] Query '{condition_key}' -> {name}")
            return {"name": name, "params": json.loads(pj)}
        return {"name": "maintain_formation", "params": {}}


# ── LLM Strategy Adapter ──────────────────────────────────────────────────────

class LLMStrategyAdapter:
    """Adapts LLM/VLM outputs into executable fleet plans with waypoints."""

    def __init__(self) -> None:
        from av_simulation.utils.sim_context import WaypointPlanner
        self.waypoint_planner = WaypointPlanner(resolution=2.0, lookahead=70.0)

    def adapt_strategy(
        self,
        strategy:        dict,
        agent_positions: Dict[str, list],
        obstacle_info:   dict,
        leader_id:       Optional[str],
    ) -> dict:
        strategy_name = strategy.get("name", "maintain_formation")
        params        = strategy.get("params", {})

        sorted_agents = sorted(
            agent_positions.keys(),
            key=lambda a: agent_positions[a][0],
            reverse=True,
        )

        obs_pos = obstacle_info.get("position") if obstacle_info else None

        adapted_plan: dict = {
            "vehicle_order": [],
            "time_slots":    [],
            "merge_side":    params.get("merge_side", "left"),
            "spacing":       params.get("spacing", 6.0),
            "strategy_name": strategy_name,
            "waypoints":     {},
        }

        def _gap(aid: str, n: int, speed: float = 8.0) -> float:
            dist = (
                np.linalg.norm(
                    np.array(agent_positions[aid]) - np.array(obs_pos)
                )
                if obs_pos else 999.0
            )
            return min(0.4, max(1.0, dist / speed) / max(n, 1))

        if strategy_name in ("time_space_reservation", "single_lane_bypass"):
            n = len(sorted_agents)
            for i, aid in enumerate(sorted_agents):
                adapted_plan["vehicle_order"].append(aid)
                adapted_plan["time_slots"].append(i * _gap(aid, n))

            avg_y = np.mean([agent_positions[a][1] for a in sorted_agents])
            adapted_plan["merge_side"] = "left" if avg_y >= 0 else "right"

        elif strategy_name == "full_stop_reassess":
            adapted_plan["vehicle_order"] = sorted_agents
            adapted_plan["time_slots"]    = [0] * len(sorted_agents)
            adapted_plan["action"]        = "stop"

        lateral_sign = 1.0 if adapted_plan["merge_side"] == "left" else -1.0

        obs_np = (
            np.array(obs_pos)
            if obs_pos else np.array([cfg.OBSTACLE_LONGITUDE, 0.0])
        )

        for i, aid in enumerate(adapted_plan["vehicle_order"]):
            lateral_offset = lateral_sign * (4.0 if i % 2 == 0 else 3.0)
            wps = self.waypoint_planner.plan(
                start_pos      = np.array(agent_positions[aid][:2]),
                obstacle_pos   = obs_np,
                lateral_target = lateral_offset,
                target_speed   = cfg.LEADER_SPEED,
                slot_index     = i,
                headway        = cfg.PLATOON_SPACING,
            )
            adapted_plan["waypoints"][aid] = wps
            print(
                f"[NEW-6] {aid[:8]} waypoints: {len(wps)} pts  "
                f"lateral={lateral_offset:+.1f}m  slot={i}"
            )

        print(
            f"[STAGE 6] Plan: {strategy_name}  "
            f"order={[a[:8] for a in adapted_plan['vehicle_order']]}  "
            f"merge={adapted_plan['merge_side']}"
        )
        return adapted_plan


# ── Simple Raft ───────────────────────────────────────────────────────────────

class SimpleRaft:
    """Minimal Raft consensus implementation for leader election."""

    def __init__(self, agent_ids=None) -> None:
        self.agent_ids:         set              = set(agent_ids) if agent_ids else set()
        self.fitness:           Dict[str, float] = {aid: 0.0   for aid in self.agent_ids}
        self.leader_id:         Optional[str]    = None
        self.is_leader:         Dict[str, bool]  = {aid: False for aid in self.agent_ids}
        self.local_assessments: Dict[str, dict]  = {}

    def register_agent(self, aid: str) -> None:
        if aid not in self.agent_ids:
            self.agent_ids.add(aid)
            self.fitness[aid]           = 0.0
            self.is_leader[aid]         = False
            self.local_assessments[aid] = None

    def update_fitness(
        self,
        aid:   str,
        vis:   float,
        conf:  float,
        res:   float,
        graph=None,           # [P1] optional SemanticGraph to write into
        position: list = None,
    ) -> None:
        """
        Record weighted fitness: phi_i = 0.4*vis + 0.3*conf + 0.3*res  (F4).

        [P1] If *graph* is provided, also writes/updates the Vehicle node
        so that fitness values are visible fleet-wide after the next V2V diff
        broadcast.
        """
        self.register_agent(aid)
        fitness_score = 0.4 * vis + 0.3 * conf + 0.3 * res
        self.fitness[aid] = fitness_score

        # [P1] Write Vehicle node into the agent's local graph (F4)
        if graph is not None:
            node_id = f"vehicle_{aid}"
            attrs   = {
                "id":        aid,
                "fitness":   round(fitness_score, 4),
                "vis":       round(vis,  4),
                "conf":      round(conf, 4),
                "res":       round(res,  4),
                "is_leader": self.is_leader.get(aid, False),
            }
            if position is not None:
                attrs["position"] = position
            graph.add_node(node_id, "Vehicle", attrs, source=aid)

    def submit_local_assessment(
        self, aid: str, can_stop: bool, can_maneuver: bool
    ) -> None:
        self.register_agent(aid)
        self.local_assessments[aid] = {
            "can_stop_safely": can_stop,
            "can_maneuver":    can_maneuver,
        }

    def check_group_stop(self) -> bool:
        if not self.local_assessments:
            return False
        return all(
            a["can_stop_safely"]
            for a in self.local_assessments.values()
            if a
        )

    def elect_leader(self, graph=None) -> Optional[str]:
        """
        Promote the highest-fitness agent to leader (F4).

        [P1] If *graph* is provided, updates is_leader attribute on all
        Vehicle nodes so the election result propagates fleet-wide via the
        next graph diff broadcast.
        """
        if not self.agent_ids:
            return None

        self.leader_id = max(self.fitness, key=self.fitness.get)
        for aid in self.agent_ids:
            self.is_leader[aid] = (aid == self.leader_id)

        print(
            f"[STAGE 4] Leader: {self.leader_id[:8]} "
            f"(fitness {self.fitness[self.leader_id]:.2f})"
        )

        # [P1] Write leader result into every Vehicle node (F4)
        if graph is not None:
            for aid in self.agent_ids:
                node_id = f"vehicle_{aid}"
                existing = graph.get_node(node_id)
                if existing is not None:
                    graph.update_node_attrs(
                        node_id,
                        {"is_leader": (aid == self.leader_id)},
                        source=aid,
                    )

        return self.leader_id


# ── Formation State ───────────────────────────────────────────────────────────

class FormationState:
    PLATOON      = "platoon"
    ZIPPER_MERGE = "zipper_merge"
    REFORMING    = "reforming"


# ── Fleet Coordinator ─────────────────────────────────────────────────────────

class FleetCoordinator:

    def __init__(
        self,
        v2v_bus:       V2VBus,
        strategy_repo: StrategyRepository,
        llm_adapter:   LLMStrategyAdapter,
    ) -> None:
        self.state                = FormationState.PLATOON
        self.obstacle_detected    = False
        self.obstacle_cleared     = False
        self.obstacle_position    = None
        self.merge_assignments:   Dict[str, float] = {}
        self.step_in_state        = 0
        self.adapted_plan         = None
        self.execution_start_time = None
        self.reformation_notified: set = set()

        self.v2v_bus              = v2v_bus
        self.strategy_repo        = strategy_repo
        self.llm_adapter          = llm_adapter

        self.merge_unlock_index: int  = 0
        self._pipeline_running: bool  = False

        self.started_avoidance: Dict[str, float] = {}
        self.cleared_obstacle:  Dict[str, float] = {}
        self._bypass_complete:  bool              = False  # permanent — never resets

        # [P2] StrategySelector — wired in lazily on first pipeline run
        self._strategy_selector = None

    def update_state(
        self,
        agents_positions: Dict[str, list],
        obstacle_pos:     Optional[list],
        current_step:     int,
    ) -> bool:
        if obstacle_pos is None:
            obstacle_pos = cfg.OBSTACLE_POSITION
        if obstacle_pos is None or not agents_positions:
            return False

        self.obstacle_position  = obstacle_pos
        self.step_in_state     += 1

        past_ct = sum(
            1 for p in agents_positions.values()
            if p[0] > obstacle_pos[0] + 15
        )

        if self.state == FormationState.PLATOON:
            # [FIX] Permanent guard — once bypass is done, never re-trigger
            if self._bypass_complete:
                return False
            frontmost_pos = max(agents_positions.values(), key=lambda p: p[0])

            leader_dist   = np.linalg.norm(
                np.array(frontmost_pos[:2]) - np.array(obstacle_pos[:2])
            )
            if leader_dist < 120 and not self.obstacle_detected:
                print(f"[STATE] PLATOON -> ZIPPER_MERGE (leader_dist={leader_dist:.1f}m)")
                self.state              = FormationState.ZIPPER_MERGE
                self.obstacle_detected  = True
                self.step_in_state      = 0
                self.execution_start_time = current_step
                self.merge_unlock_index   = 0
                return True

        elif self.state == FormationState.ZIPPER_MERGE:
            if past_ct >= len(agents_positions) * 0.75:
                print(f"[STATE] ZIPPER_MERGE -> REFORMING ({past_ct} past)")
                self.state            = FormationState.REFORMING
                self.obstacle_cleared = True
                self._bypass_complete = True  # permanent flag
                self.step_in_state    = 0
                self.reformation_notified.clear()

        elif self.state == FormationState.REFORMING:
            if (
                len(self.reformation_notified) >= len(agents_positions)
                or self.step_in_state > 150
            ):
                print("[STAGE 8] REFORMING -> PLATOON complete")
                self.state              = FormationState.PLATOON
                self.obstacle_detected  = False
                self.obstacle_cleared   = False
                self.step_in_state      = 0
                self.merge_assignments  = {}
                self.adapted_plan       = None
                self.merge_unlock_index = 0
        return False

    def notify_predecessor_started(self, agent_id: str, t: float):
        if agent_id not in getattr(self, "vehicle_order", []):
            return
        idx = self.vehicle_order.index(agent_id)
        if idx + 1 >= len(self.vehicle_order):
            return
        succ_id = self.vehicle_order[idx + 1]
        self.v2v_bus.broadcast(
            agent_id,
            "AVOIDANCE_STARTED",
            {
                "successor": succ_id,
                "timestamp": t,
                "pos_x": self.agents_positions.get(agent_id, [0, 0])[0],
                "pos_y": self.agents_positions.get(agent_id, [0, 0])[1],
            },
        )
        print(f"[V2V] {agent_id[:8]} -> {succ_id[:8]} : AVOIDANCE_STARTED")

    def notify_predecessor_cleared(self, agent_id: str, t: float):
        if agent_id not in getattr(self, "vehicle_order", []):
            return
        idx = self.vehicle_order.index(agent_id)
        if idx + 1 >= len(self.vehicle_order):
            return
        succ_id = self.vehicle_order[idx + 1]
        self.v2v_bus.broadcast(
            agent_id,
            "AVOIDANCE_COMPLETE",
            {"successor": succ_id, "timestamp": t},
        )
        print(f"[V2V] {agent_id[:8]} -> {succ_id[:8]} : AVOIDANCE_COMPLETE")

    def trigger_immediate_bypass(
        self,
        agent_ids:        List[str],
        agents_positions: Dict[str, list],
        raft:             "SimpleRaft",
        current_step:     int,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if self.obstacle_detected:
            if self.adapted_plan is None and not self._pipeline_running:
                raft.elect_leader()
                self.execute_strategy_pipeline(agent_ids, agents_positions, raft)
            return self.get_formation_directives(
                agent_ids, agents_positions, raft.leader_id,
                current_step=current_step,
            )

        print("\n[STEER-4] Immediate bypass triggered.")
        self.state              = FormationState.ZIPPER_MERGE
        self.obstacle_detected  = True
        self.obstacle_position  = cfg.OBSTACLE_POSITION
        self.step_in_state      = 0
        self.execution_start_time = current_step
        self.merge_unlock_index   = 0
        raft.elect_leader()
        self.execute_strategy_pipeline(agent_ids, agents_positions, raft)

        directives = self.get_formation_directives(
            agent_ids, agents_positions, raft.leader_id,
            current_step=current_step,
        )
        print(
            f"[STEER-4] Directives populated for "
            f"{len(directives)} agents: {[a[:8] for a in directives]}"
        )
        return directives

    def execute_strategy_pipeline(
        self,
        agent_ids:        List[str],
        agents_positions: Dict[str, list],
        raft:             "SimpleRaft",
        graph=None,       # [P2] leader's SemanticGraph — if provided, use StrategySelector
    ) -> dict:
        """
        Run stages 5–7: query strategy, build adapted plan.

        [P2] If *graph* is provided AND StrategySelector is available,
        the full LLM-driven F7 pipeline runs:
          LLM reads graph → selects strategy → writes CoordPlan subgraph.

        Falls back to the original rule-based adapter if graph is None
        or StrategySelector is unavailable.
        """
        if self._pipeline_running:
            print("[PIPELINE] Re-entry blocked — returning existing plan.")
            return self.adapted_plan
        self._pipeline_running = True
        try:
            # [P2] Attempt LLM-driven path (F7)
            if graph is not None:
                StrategySelectorCls = _get_strategy_selector()
                if StrategySelectorCls is not None:
                    # Lazily build the selector (holds LLMReasoner singleton)
                    if self._strategy_selector is None:
                        from av_simulation.coordination.llm_reasoner import LLMReasoner
                        from av_simulation.decision.repository import StrategyRepository as SR2
                        reasoner  = LLMReasoner(quantize=True)
                        repo2     = SR2()
                        self._strategy_selector = StrategySelectorCls(repo2, reasoner)
                        print("[P2] StrategySelector initialised with Phi-3 LLMReasoner")

                    fleet_state = {
                        "agent_ids":         agent_ids,
                        "agents_positions":  agents_positions,
                        "leader_id":         raft.leader_id,
                        "obstacle_position": self.obstacle_position,
                    }
                    plan_id = self._strategy_selector.select_and_instantiate(
                        graph       = graph,
                        fleet_state = fleet_state,
                        agent_id    = raft.leader_id or agent_ids[0],
                    )

                    # Extract adapted_plan-compatible dict from CoordPlan node
                    plan_node = graph.get_node(plan_id)
                    if plan_node is not None:
                        strategy_name = plan_node.attrs.get("strategy_name",
                                                            "SpatiallyOrderedBypass")
                        merge_side    = plan_node.attrs.get("merge_side", "left")

                        # Collect waypoints from Trajectory nodes
                        waypoints = {}
                        vehicle_order = []
                        traj_edges = graph.get_edges(from_id=plan_id,
                                                     label="contains_trajectory")
                        for e in traj_edges:
                            traj_node = graph.get_node(e.to_id)
                            if traj_node:
                                vid = traj_node.attrs.get("agent_id", "")
                                if vid:
                                    vehicle_order.append(vid)
                                    waypoints[vid] = traj_node.attrs.get("waypoints", [])

                        # Sort by slot_index
                        vehicle_order.sort(
                            key=lambda v: next(
                                (graph.get_node(e.to_id).attrs.get("slot_index", 99)
                                 for e in traj_edges
                                 if graph.get_node(e.to_id) and
                                 graph.get_node(e.to_id).attrs.get("agent_id") == v),
                                99,
                            )
                        )

                        self.adapted_plan = {
                            "strategy_name": strategy_name,
                            "vehicle_order": vehicle_order,
                            "merge_side":    merge_side,
                            "spacing":       plan_node.attrs.get("spacing",
                                                                  cfg.PLATOON_SPACING),
                            "time_slots":    [i * 0.4
                                              for i in range(len(vehicle_order))],
                            "waypoints":     waypoints,
                            "plan_id":       plan_id,
                        }

                        # Populate merge_assignments
                        lateral_sign = 1.0 if merge_side == "left" else -1.0
                        for i, vid in enumerate(vehicle_order):
                            self.merge_assignments[vid] = (
                                lateral_sign * (4.0 if i % 2 == 0 else 3.0)
                            )

                        print(
                            f"[P2] CoordPlan '{plan_id}' written to graph — "
                            f"strategy={strategy_name}, "
                            f"vehicles={vehicle_order}"
                        )
                        return self.adapted_plan

            # ── Fallback: original rule-based adapter ──────────────────────────
            print("[PIPELINE] Using rule-based adapter (no graph or selector unavailable)")
            strategy = self.strategy_repo.query("lane_blockage")
            self.adapted_plan = self.llm_adapter.adapt_strategy(
                strategy,
                agents_positions,
                {"position": self.obstacle_position},
                raft.leader_id,
            )
            merge_side    = self.adapted_plan.get("merge_side", "left")
            vehicle_order = self.adapted_plan.get("vehicle_order", [])
            for i, aid in enumerate(vehicle_order):
                self.merge_assignments[aid] = (
                    4.0 if i % 2 == 0 else -4.0
                    if merge_side != "right"
                    else -4.0 if i % 2 == 0 else 4.0
                )
        finally:
            self._pipeline_running = False
        return self.adapted_plan

    def refresh_directives(
        self,
        agent_ids:        List[str],
        agents_positions: Dict[str, list],
        leader_id:        Optional[str],
        current_step:     int = 0,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if not self.obstacle_detected or self.adapted_plan is None:
            return {}
        return self.get_formation_directives(
            agent_ids, agents_positions, leader_id,
            current_step=current_step,
        )

    def notify_reformation_complete(self, agent_id: str) -> None:
        self.reformation_notified.add(agent_id)
        self.v2v_bus.broadcast(
            agent_id,
            "REFORMATION_COMPLETE",
            {"agent": agent_id[:8]},
        )

    def get_formation_directives(
        self,
        agent_ids:        List[str],
        agents_positions: Dict[str, list],
        leader_id:        Optional[str],
        leader_velocity:  Optional[float] = None,
        current_step:     int = 0,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        directives: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if not agents_positions:
            return directives

        if self.adapted_plan and self.adapted_plan.get("vehicle_order"):
            sorted_agents = [
                a for a in self.adapted_plan["vehicle_order"]
                if a in agents_positions
            ]
        else:
            sorted_agents = sorted(
                [a for a in agent_ids if a in agents_positions],
                key=lambda a: agents_positions[a][0],
                reverse=True,
            )
        if not sorted_agents:
            return directives

        target_speed = leader_velocity if leader_velocity else cfg.LEADER_SPEED

        if self.state == FormationState.PLATOON:
            for i, aid in enumerate(sorted_agents):
                cp = agents_positions.get(aid, [0, 0])
                if i == 0:
                    ref_pos = np.array([cp[0] + 30, 0.0])
                    ref_vel = np.array([target_speed, 0.0])
                else:
                    pp      = agents_positions.get(sorted_agents[i - 1], [0, 0])
                    ref_pos = np.array([pp[0] - cfg.PLATOON_SPACING, 0.0])
                    ref_vel = np.array([target_speed, 0.0])
                directives[aid] = (ref_pos, ref_vel)

        elif self.state == FormationState.ZIPPER_MERGE:
            waypoints  = (self.adapted_plan or {}).get("waypoints", {})
            time_slots = (self.adapted_plan or {}).get("time_slots", [])
            _STEPS_PER_SEC     = 10
            _LATERAL_THRESHOLD = 0.3
            obs_x = self.obstacle_position[0] if self.obstacle_position else 1e9

            def _time_gate_open(i: int) -> bool:
                if not time_slots or i >= len(time_slots):
                    return True
                elapsed = current_step - (self.execution_start_time or 0)
                return elapsed >= int(time_slots[i] * _STEPS_PER_SEC)

            active_flags: Dict[str, bool] = {}
            for i, aid in enumerate(sorted_agents):
                cp = agents_positions.get(aid, [0.0, 0.0])
                if i == 0:
                    active = True
                else:
                    pred_id = sorted_agents[i - 1]
                    active  = (
                        pred_id in self.started_avoidance
                        and any(
                            m["type"] == "AVOIDANCE_STARTED"
                            and m["payload"].get("successor") == aid
                            for m in self.v2v_bus.get_recent_messages(
                                since_timestamp=self.execution_start_time
                            )
                        )
                    )
                active_flags[aid] = active

                if not active:
                    prev_cp    = agents_positions.get(sorted_agents[i - 1], [0.0, 0.0])
                    hold_x     = prev_cp[0] - cfg.PLATOON_SPACING
                    gap        = max(hold_x - cp[0], 0.0)
                    hold_speed = cfg.LEADER_SPEED * min(1.0, gap / cfg.PLATOON_SPACING)
                    directives[aid] = (
                        np.array([hold_x, 0.0]),
                        np.array([hold_speed, 0.0]),
                    )

            for i, aid in enumerate(sorted_agents):
                if not active_flags.get(aid, True):
                    continue

                cp = agents_positions.get(aid, [0.0, 0.0])

                # [FIX] Unlock all agents simultaneously once plan is active.
                # Sequential gating was blocking all but the leader from moving.
                # Time-gate still applies to stagger the actual merge timing.
                if not _time_gate_open(i):
                    if i > 0:
                        prev_cp = agents_positions.get(sorted_agents[i - 1], [0.0, 0.0])
                        hold_x  = prev_cp[0] - cfg.PLATOON_SPACING
                    else:
                        hold_x = cp[0] + 20.0
                    gap        = max(hold_x - cp[0], 0.0)
                    hold_speed = cfg.LEADER_SPEED * min(1.0, gap / cfg.PLATOON_SPACING)
                    directives[aid] = (
                        np.array([hold_x, 0.0]),
                        np.array([hold_speed, 0.0]),
                    )
                    continue

                if aid in waypoints and waypoints[aid]:
                    cx         = cp[0]
                    future_wps = [(x, y, spd) for x, y, spd in waypoints[aid] if x > cx]
                    if future_wps:
                        target_wp = next(
                            (wp for wp in future_wps
                             if abs(wp[1] - cp[1]) >= _LATERAL_THRESHOLD),
                            None,
                        )
                        if target_wp is None:
                            target_wp = future_wps[len(future_wps) // 2]
                        tx, ty, tspd = target_wp
                        dx     = max(abs(tx - cp[0]), 1.0)
                        vy_ref = float(np.clip(
                            (ty - cp[1]) / (dx / max(tspd, 1.0)), -3.0, 3.0
                        ))
                        directives[aid] = (
                            np.array([tx, ty]),
                            np.array([tspd, vy_ref]),
                        )
                        continue

                directives[aid] = (
                    np.array([max(obs_x + 30.0, cp[0] + 15.0), 0.0]),
                    np.array([cfg.LEADER_SPEED, 0.0]),
                )

        elif self.state == FormationState.REFORMING:
            for i, aid in enumerate(sorted_agents):
                cp    = agents_positions.get(aid, [0, 0])
                blend = min(1.0, self.step_in_state / 50.0)
                ty    = cp[1] * (1 - blend)
                if abs(ty) < 0.5 and aid not in self.reformation_notified:
                    self.notify_reformation_complete(aid)
                vy_ref = float(np.clip(-cp[1] * 0.1, -2.0, 2.0))
                directives[aid] = (
                    np.array([cp[0] + 20, ty]),
                    np.array([target_speed, vy_ref]),
                )

        return directives