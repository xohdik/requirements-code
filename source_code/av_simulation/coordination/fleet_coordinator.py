
from __future__ import annotations

import json # Standard library JSON module: serializes/deserializes strategy parameters stored in SQLite.
import sqlite3 # SQLite3 database module: provides lightweight disk-based or in-memory database for strategy storage.
import time
from typing import Dict, List, Optional, Tuple # Type hints: Dict for key-value mappings, List for sequences, Optional for values that may be None, Tuple for fixed-length sequences.

import numpy as np

# WaypointPlanner lives in sim_context.  Imported here so LLMStrategyAdapter
# can reference it directly.  If this import ever creates a circular dependency,
# move it inside LLMStrategyAdapter.__init__ as a local import instead.
from av_simulation.utils.sim_context import WaypointPlanner
# Imports WaypointPlanner for generating smooth trajectories around obstacles.
# The comment notes potential circular dependency and suggests a mitigation strategy.

from av_simulation.config import simulation_config as cfg # Imports configuration module as 'cfg' for accessing simulation constants (OBSTACLE_LONGITUDE, PLATOON_SPACING, LEADER_SPEED, etc.).

class V2VBus:
    """Vehicle-to-Vehicle communication bus for inter-agent messaging."""
    def __init__(self) -> None:
        self.messages:    List[dict]          = [] # List of all broadcast messages. Each message is a dict with sender, type, payload, and timestamp. Acts as an append-only log for debugging and retrospective analysis.

        self.subscribers: Dict[str, callable] = {} # Dictionary mapping agent_id to callback function. When a message is broadcast, each subscriber's callback is invoked (except the sender).


    def broadcast(self, sender_id: str, msg_type: str, payload: dict) -> dict:
        """Publish a message to all subscribers except the sender."""
        msg = {
            "sender":    sender_id,  # Agent ID of the message originator
            "type":      msg_type, # Message type (e.g., "OBSTACLE_DETECTED", "AVOIDANCE_STARTED")
            "payload":   payload, # Message-specific data (position, confidence, semantic description, etc.)
            "timestamp": time.time(),
        }
        self.messages.append(msg)   # Store message in history log

        print(f"[STAGE 2] V2V Broadcast {sender_id[:8]}: {msg_type}") # Logs broadcast with truncated agent ID (first 8 chars) for readability.

        for aid, cb in self.subscribers.items():
            if aid != sender_id: # Don't send message back to sender
                cb(msg) # Invoke callback with the message
        return msg

    def subscribe(self, agent_id: str, callback) -> None:
        """Register *callback* to receive all future broadcasts."""
        self.subscribers[agent_id] = callback # Stores callback function keyed by agent ID. When messages arrive, this callback will be invoked.

    def get_recent_messages(
        self,
        message_type: Optional[str] = None,
        since_timestamp: Optional[float] = None,
    ) -> List[dict]:
        """Return messages optionally filtered by type and/or timestamp."""
        msgs = self.messages
        if message_type:
            msgs = [m for m in msgs if m['type'] == message_type] # Filters messages by type (e.g., only "OBSTACLE_DETECTED").

        if since_timestamp:
            msgs = [m for m in msgs if m['timestamp'] > since_timestamp]
        return msgs # Filters messages to those after a specific timestamp. Used to only consider messages from the current event window.

class StrategyRepository:

    def __init__(self) -> None:
        self.conn   = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self) -> None:
        self.cursor.execute(
            '''CREATE TABLE strategies (
                id INTEGER PRIMARY KEY,
                condition TEXT,
                strategy_name TEXT,
                parameters TEXT
            )'''
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
            'INSERT INTO strategies (condition, strategy_name, parameters) '
            'VALUES (?,?,?)',
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
            'SELECT strategy_name, parameters FROM strategies WHERE condition=?',
            (cond,),
        )
        row = self.cursor.fetchone()
        if row:
            name, pj = row
            print(f"[STAGE 5] Query '{condition_key}' -> {name}")
            return {"name": name, "params": json.loads(pj)}
        return {"name": "maintain_formation", "params": {}}

class LLMStrategyAdapter:
    """Adapts LLM/VLM outputs into executable fleet plans with waypoints."""
    def __init__(self) -> None:
        self.waypoint_planner = WaypointPlanner(resolution=2.0, lookahead=70.0)
        # Creates WaypointPlanner with 2.0 metre resolution and 70 metre lookahead. Resolution: distance between waypoints along the path. Lookahead: how far ahead to plan the trajectory.

    def adapt_strategy(
        self,
        strategy:      dict,
        agent_positions: Dict[str, list],
        obstacle_info:   dict,
        leader_id:       Optional[str],
    ) -> dict:
        """Convert a retrieved strategy into an adapted plan with waypoints."""
        strategy_name = strategy.get("name", "maintain_formation")
        params        = strategy.get("params", {})
        # Extracts strategy name and parameters with fallbacks.

        sorted_agents = sorted(
            agent_positions.keys(),
            key=lambda a: agent_positions[a][0],
            reverse=True,
        )
        # Sorts agents by longitudinal position (x-coordinate) descending. reverse=True means leader (largest x) is first in list.

        obs_pos = obstacle_info.get("position") if obstacle_info else None # Extracts obstacle position if available else obs_pos is set to None.

        adapted_plan: dict = {
            "vehicle_order": [], # List of agent IDs in order from leader to tail
            "time_slots":    [], # Time offsets for each agent to start merging
            "merge_side":    params.get("merge_side", "left"),
            "spacing":       params.get("spacing", 6.0), # Target spacing between vehicles
            "strategy_name": strategy_name, # Name of the selected strategy
            "waypoints":     {}, # Dict mapping agent_id to list of waypoints (x, y, speed)
        }

        def _gap(aid: str, n: int, speed: float = 8.0) -> float:
            dist = (
                np.linalg.norm(
                    np.array(agent_positions[aid]) - np.array(obs_pos)
                )
                if obs_pos else 999.0
            ) # Distance from agent to obstacle.

            return min(0.4, max(1.0, dist / speed) / max(n, 1)) # Gap calculation: smaller when agent is closer to obstacle. Clamped between 0.4 and a value proportional to distance/speed.

        if strategy_name in ("time_space_reservation", "single_lane_bypass"):
            n = len(sorted_agents)
            for i, aid in enumerate(sorted_agents):
                adapted_plan["vehicle_order"].append(aid)
                adapted_plan["time_slots"].append(i * _gap(aid, n))# Time slots increase with index, scaled by agent's gap factor. Agents further from obstacle get larger time gaps.

            avg_y = np.mean([agent_positions[a][1] for a in sorted_agents])
            adapted_plan["merge_side"] = "left" if avg_y >= 0 else "right"
            # Automatically determines merge side based on average lateral position. If fleet is left of center, merge right; if right, merge left.

        elif strategy_name == "full_stop_reassess":
            adapted_plan["vehicle_order"] = sorted_agents
            adapted_plan["time_slots"]    = [0] * len(sorted_agents)
            adapted_plan["action"]        = "stop" # All agents stop simultaneously (no staggering).

        lateral_sign = 1.0 if adapted_plan["merge_side"] == "left" else -1.0 # Determines sign for lateral offset: +1 for left, -1 for right.

        obs_np = (
            np.array(obs_pos)
            if obs_pos else np.array([cfg.OBSTACLE_LONGITUDE, 0.0]) # Converts obstacle position to numpy array. Uses default if not provided.
        )

        for i, aid in enumerate(adapted_plan["vehicle_order"]):
            lateral_offset = lateral_sign * (4.0 if i % 2 == 0 else 3.0)
           # Alternating lateral offsets for even/odd agents in the plan. Even-indexed (leader, third, etc.) 
           # offset 4.0m; odd-indexed offset 3.0m. Creates staggered merge pattern to avoid collisions.

            wps = self.waypoint_planner.plan(
                start_pos      = np.array(agent_positions[aid][:2]),
                obstacle_pos   = obs_np,
                lateral_target = lateral_offset,
                target_speed   = cfg.LEADER_SPEED,
                slot_index     = i, # Position in merge order (0=leader)
                headway        = cfg.PLATOON_SPACING, # Staggering distance between vehicles
            )
            # Generates smooth waypoints for this agent to merge around obstacle.

            adapted_plan["waypoints"][aid] = wps
            print(
                f"[NEW-6] {aid[:8]} waypoints: {len(wps)} pts  "
                f"lateral={lateral_offset:+.1f}m  slot={i}" # Logs waypoint generation for debugging.
            )

        print(
            f"[STAGE 6] Plan: {strategy_name}  "
            f"order={[a[:8] for a in adapted_plan['vehicle_order']]}  "
            f"merge={adapted_plan['merge_side']}" # Logs the final adapted plan summary.
        )
        return adapted_plan

class SimpleRaft:
    """Minimal Raft consensus implementation for leader election and group decisions."""
    def __init__(self, agent_ids=None) -> None:
        self.agent_ids:         set              = set(agent_ids) if agent_ids else set() # Set of all registered agent IDs.
        self.fitness:           Dict[str, float] = {aid: 0.0   for aid in self.agent_ids}  # Fitness score for each agent. Higher fitness = more likely to be elected leader.

        self.leader_id:         Optional[str]    = None # Currently elected leader agent ID.

        self.is_leader:         Dict[str, bool]  = {aid: False for aid in self.agent_ids} # Boolean flag indicating whether each agent is the leader.

        self.local_assessments: Dict[str, dict]  = {} #  Stores each agent's self-assessment of stopping capability and maneuverability

    def register_agent(self, aid: str) -> None:
        """Add *aid* to the consensus group if not already present."""
        if aid not in self.agent_ids:
            self.agent_ids.add(aid)
            self.fitness[aid]           = 0.0
            self.is_leader[aid]         = False
            self.local_assessments[aid] = None
            # Initializes tracking data structures for the new agent.

    def update_fitness(
        self, aid: str, vis: float, conf: float, res: float
    ) -> None:
        """Record weighted fitness: 0.4·vis + 0.3·conf + 0.3·res."""
        self.register_agent(aid)
        self.fitness[aid] = 0.4 * vis + 0.3 * conf + 0.3 * res
        # Weighted combination:
        # - visibility (40%): how well the agent sees the obstacle
        # - confidence (30%): VLM confidence in detection
        # - resource (30%): agent's maneuverability resources (position, lane centering)

    def submit_local_assessment(
        self, aid: str, can_stop: bool, can_maneuver: bool
    ) -> None:
        """Store each agent's self-assessed manoeuvrability."""
        self.register_agent(aid)
        self.local_assessments[aid] = {
            "can_stop_safely": can_stop,
            "can_maneuver":    can_maneuver,
        } # Stores whether this agent believes it can stop safely and/or maneuver around obstacle.

    def check_group_stop(self) -> bool:
        """Return True if every agent reports it can stop safely."""
        if not self.local_assessments:
            return False
        return all(
            a['can_stop_safely']
            for a in self.local_assessments.values()
            if a
        )
        # Checks if all agents (with non-None assessments) can stop safely.
        # Used to determine if fleet should initiate emergency stop.

    def elect_leader(self) -> Optional[str]:
        """Promote the highest-fitness agent to leader and return its id."""
        if not self.agent_ids:
            return None
        self.leader_id = max(self.fitness, key=self.fitness.get) # Selects agent with maximum fitness score.

        for aid in self.agent_ids:
            self.is_leader[aid] = (aid == self.leader_id) # Updates leader flags.
        print(
            f"[STAGE 4] Leader: {self.leader_id[:8]} "
            f"(fitness {self.fitness[self.leader_id]:.2f})"
        )
        return self.leader_id

class FormationState:
    """String constants for the FleetCoordinator FSM states."""
    PLATOON      = "platoon"
    ZIPPER_MERGE = "zipper_merge"
    REFORMING    = "reforming"

class FleetCoordinator:

    def __init__(
        self,
        v2v_bus:       V2VBus,
        strategy_repo: StrategyRepository,
        llm_adapter:   LLMStrategyAdapter,
    ) -> None:
        self.state                = FormationState.PLATOON # Current state of the fleet (PLATOON, ZIPPER_MERGE, or REFORMING).

        self.obstacle_detected    = False # Flag indicating whether an obstacle has been detected by any agent.
        self.obstacle_cleared     = False # Flag indicating whether the obstacle has been passed by the entire fleet.
        self.obstacle_position    = None  # [x, y] coordinates of the detected obstacle.
        self.merge_assignments:   Dict[str, float] = {} # Maps agent_id to assigned lateral target position during merge.
        self.step_in_state        = 0 # Number of simulation steps spent in the current state.
        self.adapted_plan         = None # The currently active adapted plan (from LLMStrategyAdapter).
        self.execution_start_time = None # Simulation step when the current merge/avoidance began.
        self.reformation_notified: set = set() # Set of agents that have completed reformation (returned to center lane).

        self.v2v_bus              = v2v_bus
        self.strategy_repo        = strategy_repo
        self.llm_adapter          = llm_adapter # Dependencies injected via constructor.
        
        self.merge_unlock_index: int  = 0 # Index into vehicle_order indicating which agents are cleared to merge.
        # Increments as each agent completes its merge, ensuring sequential execution.
        
        self._pipeline_running: bool  = False # If it is set to True no strategy would be adapted and vehicle crash into barrier.Flag to prevent multiple concurrent pipeline executions in the same simulation step. 

        self.started_avoidance: Dict[str, float]   = {} # Timestamp when each agent began obstacle avoidance maneuver.
        self.cleared_obstacle:  Dict[str, float]   = {}  # Timestamp when each agent cleared the obstacle.

    def update_state(
        self,
        agents_positions: Dict[str, list],
        obstacle_pos:     Optional[list],
        current_step:     int,
    ) -> bool:
        """Update fleet state machine based on current positions and step."""
        # Returns True if the full pipeline should be triggered.
        
        if obstacle_pos is None:
            obstacle_pos = cfg.OBSTACLE_POSITION  # Uses module-level config if obstacle_pos not provided.

        if obstacle_pos is None or not agents_positions:
            return False        # No obstacle or no agents → no state change.

        self.obstacle_position  = obstacle_pos
        self.step_in_state     += 1

        dists   = [
            np.linalg.norm(np.array(p) - np.array(obstacle_pos))
            for p in agents_positions.values()
        ] # Computes distances from each agent to obstacle.

        past_ct = sum(
            1 for p in agents_positions.values()
            if p[0] > obstacle_pos[0] + 15
        )  # Counts agents that have passed the obstacle (x > obstacle.x + 15m).

        if self.state == FormationState.PLATOON:
            
            frontmost_pos = max(agents_positions.values(), key=lambda p: p[0]) # Finds the agent with the largest x-coordinate (furthest ahead).

            leader_dist   = np.linalg.norm(
                np.array(frontmost_pos[:2]) - np.array(obstacle_pos[:2])
            ) # Distance from leader to obstacle.

            if leader_dist < 120 and not self.obstacle_detected:
                print(f"[STATE] PLATOON -> ZIPPER_MERGE (leader_dist={leader_dist:.1f}m)")
                self.state              = FormationState.ZIPPER_MERGE
                self.obstacle_detected  = True
                self.step_in_state      = 0
                self.execution_start_time = current_step
                self.merge_unlock_index   = 0
                return True
             # Transitions to zipper merge state when leader is within 120m of obstacle. Returns True to trigger full pipeline execution.

        elif self.state == FormationState.ZIPPER_MERGE:
            if past_ct >= len(agents_positions) * 0.75:
                print(f"[STATE] ZIPPER_MERGE -> REFORMING ({past_ct} past)")
                self.state            = FormationState.REFORMING
                self.obstacle_cleared = True
                self.step_in_state    = 0
                self.reformation_notified.clear()
                # Transitions to reforming when 75% of agents have passed the obstacle.

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
                 # Returns to platoon when all agents have reformed or 150 steps elapsed.
        return False
    
        ### ── New: explicit predecessor → successor signaling ────────────────────────
    def notify_predecessor_started(self, agent_id: str, t: float):
        """Leader or any vehicle that just started avoidance notifies its successor."""
        if agent_id not in self.vehicle_order:
            return
        idx = self.vehicle_order.index(agent_id)
        if idx + 1 >= len(self.vehicle_order):
            return  # tail has no successor
        succ_id = self.vehicle_order[idx + 1]
        self.v2v_bus.broadcast(
            agent_id,
            "AVOIDANCE_STARTED",
            {
                "successor": succ_id,
                "timestamp": t,
                "pos_x": self.agents_positions.get(agent_id, [0,0])[0],
                "pos_y": self.agents_positions.get(agent_id, [0,0])[1]
            }
        )# Broadcasts to successor that this agent has started its avoidance maneuver.
        print(f"[V2V] {agent_id[:8]} → {succ_id[:8]} : AVOIDANCE_STARTED")

    ## ------- only executes after each vehicle has bypassed the obstacle

    def notify_predecessor_cleared(self, agent_id: str, t: float):
        """Leader or any vehicle that just started avoidance notifies its successor."""
        # Part of the predecessor-successor signaling mechanism to coordinate sequential merging.
        if agent_id not in self.vehicle_order:
            return
        idx = self.vehicle_order.index(agent_id)
        if idx + 1 >= len(self.vehicle_order):
            return # tail has no successor
        succ_id = self.vehicle_order[idx + 1]
        self.v2v_bus.broadcast(
            agent_id,
            "AVOIDANCE_COMPLETE",
            {"successor": succ_id, "timestamp": t}
        )
        # Broadcasts to successor that obstacle clearance is complete.
        print(f"[V2V] {agent_id[:8]} → {succ_id[:8]} : AVOIDANCE_COMPLETE")


    def trigger_immediate_bypass(
        self,
        agent_ids:        List[str],
        agents_positions: Dict[str, list],
        raft:             SimpleRaft,
        current_step:     int,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]: 
        """Immediate pipeline trigger when obstacle is detected (bypasses state machine)."""
        # Called directly from VLAPolicy when an agent detects an obstacle.
        
        if self.obstacle_detected:

            if self.adapted_plan is None and not self._pipeline_running:
                raft.elect_leader()
                self.execute_strategy_pipeline(agent_ids, agents_positions, raft)
            # Always return fresh directives so callers can write them in.
            return self.get_formation_directives(
                agent_ids, agents_positions, raft.leader_id,
                current_step=current_step,
            )
            # If obstacle already detected, just refresh directives.

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
        raft:             SimpleRaft,
    ) -> dict:
        """Run stages 5–6: query strategy and build adapted plan."""
        if self._pipeline_running:
            print("[PIPELINE] Re-entry blocked — returning existing plan.")
            return self.adapted_plan
        self._pipeline_running = True
        try:
            strategy = self.strategy_repo.query("lane_blockage") # Queries strategy for lane blockage scenario.
            self.adapted_plan = self.llm_adapter.adapt_strategy(
                strategy,
                agents_positions,
                {"position": self.obstacle_position},
                raft.leader_id,
            ) # Adapts strategy to create waypoint-based plan.

            merge_side    = self.adapted_plan.get("merge_side", "left")
            vehicle_order = self.adapted_plan.get("vehicle_order", [])
            for i, aid in enumerate(vehicle_order):
                self.merge_assignments[aid] = (
                    4.0 if i % 2 == 0 else -4.0
                    if merge_side != "right"
                    else -4.0 if i % 2 == 0 else 4.0
                )# Assigns lateral targets based on merge side and parity.
                # For left merge: even index → +4m, odd index → -4m.
                # For right merge: even index → -4m, odd index → +4m.
                # Creates alternating pattern to avoid collisions.
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
        """Refresh formation directives (re-query from current plan)."""
        if not self.obstacle_detected or self.adapted_plan is None:
            return {}
        return self.get_formation_directives(
            agent_ids, agents_positions, leader_id,
            current_step=current_step,
        )
      # Delegates to get_formation_directives with current state.

    def notify_reformation_complete(self, agent_id: str) -> None:
        """Record that *agent_id* has returned to centre lane."""
        self.reformation_notified.add(agent_id)
        self.v2v_bus.broadcast(
            agent_id,
            "REFORMATION_COMPLETE",
            {"agent": agent_id[:8]},
        )# Broadcasts reformation completion to all agents.

    def get_formation_directives(
        self,
        agent_ids:        List[str],
        agents_positions: Dict[str, list],
        leader_id:        Optional[str],
        leader_velocity:  Optional[float] = None,
        current_step:     int = 0,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate current-step directives for all agents based on fleet state."""
        # Returns dict mapping agent_id → (ref_pos, ref_vel) for tracking.

        directives: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if not agents_positions:
            return directives
        # Determine order of agents (leader first, tail last)
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
        # If adapted plan exists, use its vehicle_order. Otherwise, sort by x-coordinate descending.
        if not sorted_agents:
            return directives

        target_speed = leader_velocity if leader_velocity else cfg.LEADER_SPEED
        # Use leader's current speed if available, otherwise config default.

        # ── PLATOON
        if self.state == FormationState.PLATOON:
            for i, aid in enumerate(sorted_agents):
                cp = agents_positions.get(aid, [0, 0])
                if i == 0:
                    ref_pos = np.array([cp[0] + 30, 0.0])
                    ref_vel = np.array([target_speed, 0.0]) # Leader: target 30m ahead, same lane.
                else:
                    pp      = agents_positions.get(sorted_agents[i - 1], [0, 0])
                    ref_pos = np.array([pp[0] - cfg.PLATOON_SPACING, 0.0])
                    ref_vel = np.array([target_speed, 0.0])
                    # Followers: maintain spacing behind predecessor.
                directives[aid] = (ref_pos, ref_vel)

        # ── ZIPPER_MERGE
        elif self.state == FormationState.ZIPPER_MERGE:
            waypoints  = (self.adapted_plan or {}).get("waypoints", {})
            # Retrieve waypoints and time slots from adapted plan.

            time_slots = (self.adapted_plan or {}).get("time_slots", [])
            ## MPC runs at 10 Hz (0.1s per step)..
            _STEPS_PER_SEC  = 10
            _LATERAL_THRESHOLD = 0.3 # Minimum lateral deviation to consider "in merge".

            obs_x = self.obstacle_position[0] if self.obstacle_position else 1e9
            # Obstacle x-coordinate (large default if not set).

            def _time_gate_open(i: int) -> bool:
                """Return True when enough sim-steps have elapsed for slot i."""
                if not time_slots or i >= len(time_slots):
                    return True
                elapsed = current_step - (self.execution_start_time or 0)
                return elapsed >= int(time_slots[i] * _STEPS_PER_SEC) # Compares elapsed steps to required steps from time slot.

            active_flags: Dict[str, bool] = {}
            for i, aid in enumerate(sorted_agents):
                cp = agents_positions.get(aid, [0.0, 0.0])  

                if i == 0:  # leader is always active
                    active = True
                else:
                    # Check whether predecessor has sent AVOIDANCE_STARTED
                    pred_id = sorted_agents[i - 1]
                    active = (
                        pred_id in self.started_avoidance and
                        any(
                            m['type'] == "AVOIDANCE_STARTED" and
                            m['payload'].get("successor") == aid
                            for m in self.v2v_bus.get_recent_messages(
                                since_timestamp=self.execution_start_time
                            )
                        )
                    ) # Active if predecessor has started avoidance and broadcast to this agent.

                active_flags[aid] = active

                if not active:
                    # Hold position behind predecessor until the V2V signal arrives
                    prev_cp = agents_positions.get(sorted_agents[i - 1], [0.0, 0.0])
                    hold_x  = prev_cp[0] - cfg.PLATOON_SPACING
                    gap     = max(hold_x - cp[0], 0.0)  # cp always defined here now
                    hold_speed = cfg.LEADER_SPEED * min(1.0, gap / cfg.PLATOON_SPACING) # Slows down proportionally to how far behind the hold position.
                    directives[aid] = (
                        np.array([hold_x, 0.0]),
                        np.array([hold_speed, 0.0]),
                    )# Assigns hold directive (no merge yet).

            # ── Pass 2: space-gate + time-gate → waypoint following / fallback.
            # Only processes agents not already handled by the V2V hold above.
            for i, aid in enumerate(sorted_agents):
                if not active_flags.get(aid, True):
                    # Already given a hold directive in Pass 1 — skip.
                    continue

                cp = agents_positions.get(aid, [0.0, 0.0])  

                if i > self.merge_unlock_index or not _time_gate_open(i):
                    # Not yet cleared to merge (spatially or temporally).
                    if i > 0:
                        prev_cp = agents_positions.get(
                            sorted_agents[i - 1], [0.0, 0.0]
                        )
                        hold_x = prev_cp[0] - cfg.PLATOON_SPACING
                    else:
                        hold_x = cp[0] + 20.0

                    gap        = max(hold_x - cp[0], 0.0)
                    hold_speed = cfg.LEADER_SPEED * min(1.0, gap / cfg.PLATOON_SPACING)
                    directives[aid] = (
                        np.array([hold_x, 0.0]),
                        np.array([hold_speed, 0.0]),
                    )
                    continue

                # Vehicle is cleared — follow its merge waypoints.
                if aid in waypoints and waypoints[aid]:
                    cx         = cp[0]
                    future_wps = [
                        (x, y, spd) for x, y, spd in waypoints[aid] if x > cx
                    ]
                    if future_wps:
                        target_wp = next(
                            (wp for wp in future_wps
                             if abs(wp[1] - cp[1]) >= _LATERAL_THRESHOLD),
                            None,
                        )
                        # Find first waypoint where lateral change exceeds threshold.
                        # This ensures we track the active merging segment.
                        if target_wp is None:
                            target_wp = future_wps[len(future_wps) // 2]
                            # Fallback to midpoint if no lateral change detected.

                        tx, ty, tspd = target_wp
                        dx     = max(abs(tx - cp[0]), 1.0)
                        vy_ref = float(np.clip(
                            (ty - cp[1]) / (dx / max(tspd, 1.0)), -3.0, 3.0
                        ))
                         # Computes required lateral velocity to reach target.
                        # Scales by travel time to target.
                        directives[aid] = (
                            np.array([tx, ty]),
                            np.array([tspd, vy_ref]),
                        )
                        continue

                # Fallback: no usable waypoints — push past obstacle.
                directives[aid] = (
                    np.array([max(obs_x + 30.0, cp[0] + 15.0), 0.0]),
                    np.array([cfg.LEADER_SPEED, 0.0]),
                )# Simple fallback: target 30m past obstacle or 15m ahead.
        # ── REFORMING
        elif self.state == FormationState.REFORMING:
            for i, aid in enumerate(sorted_agents):
                cp    = agents_positions.get(aid, [0, 0])
                blend = min(1.0, self.step_in_state / 50.0)
                # Blend factor: increases from 0 to 1 over 50 steps.
                ty    = cp[1] * (1 - blend)
                # Gradually returns lateral position to center lane (y=0).
                if abs(ty) < 0.5 and aid not in self.reformation_notified:
                    self.notify_reformation_complete(aid)
                    # Notifies when within 0.5m of center lane.
                vy_ref = float(np.clip(-cp[1] * 0.1, -2.0, 2.0))
                directives[aid] = (
                    np.array([cp[0] + 20, ty]),
                    np.array([target_speed, vy_ref]),
                )
                # Target: 20m ahead, gradually centering.
        return directives
