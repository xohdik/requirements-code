"""
hierarchical_mpc.py  — FINAL INTEGRATED VERSION
=================================================
Integrates all phases:
  [P1] SemanticGraph — Obstacle nodes, Vehicle nodes, GraphDiff broadcast
  [P3] ExecutionStatus nodes written after MPC step (F8)
  [P4] OSQPMPCController replaces CasADi/do_mpc (NF1)
  [P4] MPC solve times logged to mpc_perf.jsonl (NF5)
  [F9] LiDAR fusion writes synthetic Obstacle node to graph
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from metadrive.policy.base_policy import BasePolicy

from av_simulation.config import simulation_config as cfg
from av_simulation import SharedSimState
from av_simulation.coordination.fleet_coordinator import FormationState
from av_simulation.utils.sim_context import (
    EnhancedMultiAgentEnv,
    capture_agent_frame,
)
from av_simulation.vision_language.vlm_engine import parse_vlm_output

# [P1] Graph
from av_simulation.graph.graph import SemanticGraph
from av_simulation.graph.diff  import GraphDiff

# [P4] OSQP MPC — lazy import avoids circular at module load
_osqp_ctrl_cls = None
def _get_osqp_ctrl():
    global _osqp_ctrl_cls
    if _osqp_ctrl_cls is None:
        try:
            from av_simulation.execution.mpc import OSQPMPCController
            _osqp_ctrl_cls = OSQPMPCController
        except ImportError:
            pass
    return _osqp_ctrl_cls

# CasADi — optional, only needed if OSQP unavailable
try:
    from casadi import *
    _CASADI_AVAILABLE = True
except ImportError:
    _CASADI_AVAILABLE = False

# [P4] Logger — lazy import
def _mpc_logger():
    try:
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "av_simulation.utils.logger",
            os.path.join(os.path.dirname(__file__), "..", "utils", "logger.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.mpc_logger
    except Exception:
        return None

_MPC_LOGGER = None
def _get_mpc_logger():
    global _MPC_LOGGER
    if _MPC_LOGGER is None:
        _MPC_LOGGER = _mpc_logger()
    return _MPC_LOGGER


# ── keep setup_mpc for backward compat (used only if OSQP unavailable) ────────

def setup_mpc(obs_pos=None, safe_dist: float = 5.0, cost_weight: float = 1.0):
    """
    Legacy CasADi/do_mpc setup — kept as fallback.
    [P4] OSQPMPCController is preferred; setup_mpc is only called when OSQP
    import fails (e.g. osqp not installed).
    """
    if not _CASADI_AVAILABLE:
        raise RuntimeError("CasADi not installed and OSQP also unavailable.")
    from do_mpc.controller import MPC
    from do_mpc.model import Model
    from do_mpc.simulator import Simulator

    model   = Model("continuous")
    pos     = model.set_variable("_x", "pos", (2, 1))
    vel     = model.set_variable("_x", "vel", (2, 1))
    u       = model.set_variable("_u", "u",   (2, 1))
    ref_pos = model.set_variable("_tvp", "ref_pos", (2, 1))
    ref_vel = model.set_variable("_tvp", "ref_vel", (2, 1))

    model.set_rhs("pos", vel)
    model.set_rhs("vel", vertcat(u[0], u[1] * fmax(vel[0], cfg.MIN_STEER_SPEED)))
    model.setup()

    mpc       = MPC(model)
    n_horizon = 20
    mpc.set_param(n_horizon=n_horizon, t_step=0.1, n_robust=1)

    lterm = (
        (pos[0] - ref_pos[0]) ** 2
        + 8.0 * (pos[1] - ref_pos[1]) ** 2
        + sum1((vel - ref_vel) ** 2)
    )
    mpc.set_objective(mterm=lterm, lterm=lterm)
    mpc.set_rterm(u=0.05)
    mpc.bounds["lower", "_u", "u"] = [-3.0, -1.0]
    mpc.bounds["upper", "_u", "u"] = [ 3.0,  1.0]

    if obs_pos is not None:
        dist = sqrt(sum1((pos - vertcat(obs_pos[0], obs_pos[1])) ** 2))
        mpc.set_nl_cons("collision_avoid", -dist + safe_dist, ub=0,
                        soft_constraint=True,
                        penalty_term_cons=1e4 * max(cost_weight, 1.0))

    tvp_template = mpc.get_tvp_template()
    tvp_refs: Dict[str, np.ndarray] = {
        "ref_pos": np.array([[0.0], [0.0]]),
        "ref_vel": np.array([[20.0], [0.0]]),
    }

    def tvp_fun(t_now):
        for k in range(n_horizon + 1):
            tvp_template["_tvp", k, "ref_pos"] = tvp_refs["ref_pos"]
            tvp_template["_tvp", k, "ref_vel"] = tvp_refs["ref_vel"]
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    sim  = Simulator(model)
    sim.set_param(t_step=0.1)
    stpl = sim.get_tvp_template()

    def sim_tvp_fun(t_now):
        stpl["ref_pos"] = tvp_refs["ref_pos"]
        stpl["ref_vel"] = tvp_refs["ref_vel"]
        return stpl

    sim.set_tvp_fun(sim_tvp_fun)
    sim.setup()

    x0_init = np.zeros((4, 1))
    mpc.x0  = x0_init
    sim.x0  = x0_init
    mpc.set_initial_guess()
    return mpc, sim, tvp_refs


# ── VLAPolicy ─────────────────────────────────────────────────────────────────

class VLAPolicy(BasePolicy):

    def __init__(
        self,
        control_object,
        random_seed:     int,
        shared_state:    SharedSimState,
        env=None,
        profiler=None,
        obstacle_manager=None,
    ) -> None:
        super().__init__(control_object, random_seed)

        self.agent_id         = self.control_object.name
        self.env              = env
        self._state           = shared_state
        self.obstacle_manager = obstacle_manager

        from av_simulation.utils.sim_context import PerformanceProfiler
        self.profiler = profiler or PerformanceProfiler(enabled=False)

        # Detection state
        self.fitness:              float           = 0.0
        self.semantic_description: Optional[str]   = None
        self.confidence:           float           = 0.0
        self.visibility_score:     float           = 0.0
        self.resource_score:       float           = 0.0
        self.reference_trajectory: Optional[tuple] = None
        self.obstacle_broadcast_done               = False

        # VLM agent role
        self._vlm_step_counter = -(cfg.VLM_INFERENCE_INTERVAL - 1)
        self._last_detected    = False
        self._is_vlm_agent     = False
        self._peer_semantic:   Optional[str] = None
        self._peer_confidence: float         = 0.0
        self._peer_blockage:   int           = 0

        # [P4] OSQP MPC — prefer OSQP; fall back to CasADi if unavailable
        obs_pos   = cfg.OBSTACLE_POSITION if cfg.OBSTACLE_POSITION is not None else [80.0, 0.0]
        OSQPCtrl  = _get_osqp_ctrl()
        if OSQPCtrl is not None:
            self._osqp    = OSQPCtrl(obs_pos=np.array(obs_pos), safe_dist=cfg.SAFE_DISTANCE)
            self._use_osqp = True
            self.mpc = self.simulator = self.tvp_refs = None
            print(f"[P4-MPC] {self.agent_id[:8]} using OSQPMPCController")
        else:
            self._osqp     = None
            self._use_osqp = False
            self.mpc, self.simulator, self.tvp_refs = setup_mpc(
                obs_pos=obs_pos, safe_dist=cfg.SAFE_DISTANCE
            )
            print(f"[P4-MPC] {self.agent_id[:8]} using CasADi fallback")

        self.state = np.array([[0.0, 0.0], [0.0, 0.0]])

        # [P1] SemanticGraph — one per agent
        self.graph                    = SemanticGraph()
        self._last_graph_version: int = 0
        self._prev_snapshot           = SemanticGraph()
        self._state.graphs[self.agent_id] = self.graph

        # Subscribe to V2V
        self._state.fleet_coordinator.v2v_bus.subscribe(
            self.agent_id, self._on_v2v_message
        )

    # ── V2V callback ──────────────────────────────────────────────────────────

    def _on_v2v_message(self, msg: dict) -> None:
        if msg["type"] == "OBSTACLE_DETECTED":
            pl = msg["payload"]
            self._peer_semantic   = pl.get("semantic", "")
            self._peer_confidence = pl.get("confidence", 0.0)
            self._peer_blockage   = pl.get("blockage_percent", 0)
            self._last_detected   = True

        elif msg["type"] == "GRAPH_DIFF":   # [P1] merge incoming diff
            try:
                diff = GraphDiff.from_dict(msg["payload"])
                self.graph.merge_diff(diff)
            except Exception as e:
                print(f"[GRAPH-SYNC] {self.agent_id[:8]} merge error: {e}")

    # ── [P1] Write Obstacle node to graph ─────────────────────────────────────

    def _write_obstacle_to_graph(
        self, result: dict, source: str = "vlm",
        dist: float = 0.0, blocked_rays: int = 0,
    ) -> str:
        ts      = time.time()
        node_id = f"obs_{self.agent_id[:8]}_{int(ts)}"
        lane_occ = result.get("blockage_percent", 50) / 100.0

        self.graph.add_node(node_id, "Obstacle", {
            "type":           result.get("semantic_description", "unknown"),
            "lane_occupancy": round(lane_occ, 3),
            "confidence":     round(result.get("confidence", 0.6), 3),
            "distance_m":     round(result.get("distance_m", dist), 1),
            "blocked_rays":   blocked_rays,
            "source":         source,
            "position":       cfg.OBSTACLE_POSITION,
        }, source=self.agent_id, timestamp=ts)

        lane_id = "lane_ego"
        if self.graph.get_node(lane_id) is None:
            self.graph.add_node(lane_id, "Lane",
                {"id": lane_id, "direction": "forward"},
                source=self.agent_id, timestamp=ts)

        self.graph.add_edge(node_id, lane_id, "blocks",
            {"occupancy": round(lane_occ, 3)},
            self.agent_id, ts)

        print(f"[GRAPH] {self.agent_id[:8]} Obstacle node "
              f"(conf={result.get('confidence', 0):.2f}, src={source})")
        return node_id

    # ── Steering helpers ──────────────────────────────────────────────────────

    def _compute_lateral_steer(self, pos: np.ndarray) -> float:
        if self.reference_trajectory is None:
            return 0.0
        ref_pos, _ = self.reference_trajectory
        lat_err    = float(ref_pos[1]) - float(pos[1])
        fc            = self._state.fleet_coordinator
        vehicle_order = (fc.adapted_plan or {}).get("vehicle_order", [])
        my_slot = (vehicle_order.index(self.agent_id)
                   if self.agent_id in vehicle_order else 0)
        is_active_merger = (
            fc.state == FormationState.ZIPPER_MERGE
            and my_slot <= fc.merge_unlock_index
        )
        gain = 0.25 if is_active_merger else 0.05
        return float(np.clip(lat_err * gain, -1.0, 1.0))

    def _get_assigned_lateral(self) -> Optional[float]:
        return self._state.fleet_coordinator.merge_assignments.get(self.agent_id)

    @staticmethod
    def _get_assigned_lateral_for(state: SharedSimState, aid: str) -> Optional[float]:
        return state.fleet_coordinator.merge_assignments.get(aid)

    # ── Obstacle detection (F1, F9) ───────────────────────────────────────────

    def detect_obstacle_semantic(
        self,
        lidar:        np.ndarray,
        depth:        Optional[np.ndarray],
        current_step: int = 0,
    ) -> bool:

        if depth is not None:
            lidar = EnhancedMultiAgentEnv.fuse_lidar_depth(lidar, depth)

        pos  = np.array(self.control_object.position[:2])
        dist = (np.linalg.norm(pos - np.array(cfg.OBSTACLE_POSITION))
                if cfg.OBSTACLE_POSITION else 999.0)
        blocked_rays = int(np.sum(lidar < 0.25))
        lidar_hit    = blocked_rays > 2 or dist < 60.0

        # [FIX] Stop detecting obstacle once fleet has bypassed it
        fc_state = self._state.fleet_coordinator
        if getattr(fc_state, '_bypass_complete', False) and dist > 20.0:
            return self._last_detected  # don't re-detect from behind

        if self._is_vlm_agent:
            self._vlm_step_counter += 1
            if self._vlm_step_counter % cfg.VLM_INFERENCE_INTERVAL != 0:
                if not self._last_detected and lidar_hit:
                    print(f"[VLM] Inter-frame lidar override {self.agent_id[:8]} "
                          f"(blocked={blocked_rays}, dist={dist:.0f}m)")
                    self._last_detected = True
                return self._last_detected

            self.profiler.start_vlm()
            try:
                frame       = capture_agent_frame(self.env, self.control_object)
                description = self._state.vlm_engine.describe_scene(frame)
            except Exception as e:
                print(f"[VLM] Error {self.agent_id[:8]}: {e}")
                description = ""
            self.profiler.end_vlm()

            result       = parse_vlm_output(description)
            vlm_detected = result["detected"]

            # [F9] LiDAR fusion override
            if not vlm_detected and lidar_hit:
                print(f"[FIX-E] Fusion override: lidar contradicts ROAD CLEAR "
                      f"(blocked={blocked_rays}, dist={dist:.0f}m)")
                vlm_detected         = True
                result["detected"]   = True
                result["confidence"] = max(result.get("confidence", 0), 0.6)
                result["distance_m"] = dist
                result["semantic_description"] = (
                    f"lidar_fusion_obstacle at {dist:.0f}m (rays={blocked_rays})")
                result["blockage_percent"] = min(
                    100, int(blocked_rays / max(len(lidar), 1) * 100 + 30))
                self._write_obstacle_to_graph(result, source="lidar_fusion",
                                              dist=dist, blocked_rays=blocked_rays)

            self.semantic_description = result["semantic_description"]
            self.confidence           = result["confidence"]
            self.visibility_score     = max(0.1, 1.0 - result["distance_m"] / 100.0)
            self._last_detected       = vlm_detected

            if vlm_detected and not self.obstacle_broadcast_done:
                self._write_obstacle_to_graph(result, source="vlm",
                                              dist=dist, blocked_rays=blocked_rays)
                fc = self._state.fleet_coordinator
                fc.v2v_bus.broadcast(self.agent_id, "OBSTACLE_DETECTED", {
                    "semantic":         result["semantic_description"],
                    "blockage_percent": result["blockage_percent"],
                    "confidence":       result["confidence"],
                    "distance_m":       result["distance_m"],
                    "position":         cfg.OBSTACLE_POSITION,
                })
                self.obstacle_broadcast_done = True
                if fc.adapted_plan is None and not fc._pipeline_running:
                    new_directives = fc.trigger_immediate_bypass(
                        list(self._state.raft.agent_ids),
                        self._state.agents_positions,
                        self._state.raft, current_step,
                    )
                    if new_directives:
                        self._state.directives.update(new_directives)
            return vlm_detected

        # Non-VLM agent path
        if self._last_detected and self._peer_semantic:
            self.semantic_description = self._peer_semantic
            self.confidence = max(self.confidence, self._peer_confidence * 0.85)
            return True

        detected = False
        if blocked_rays > 0:
            detected              = True
            self.confidence       = min(0.95, blocked_rays / len(lidar) + 0.3)
            self.visibility_score = 1.0 - dist / 100.0
        if dist < 60:
            detected              = True
            self.confidence       = max(self.confidence, 1.0 - dist / 60)
            self.visibility_score = max(self.visibility_score, 1.0 - dist / 60)
        if detected and not self.semantic_description:
            self.semantic_description = (
                f"[lidar-fused] Obstacle at {dist:.0f}m (blocked_rays={blocked_rays})")

        if detected and not self.obstacle_broadcast_done:
            self.obstacle_broadcast_done = True
            result_lidar = {
                "detected": True,
                "semantic_description": self.semantic_description,
                "blockage_percent": 50,
                "confidence":  self.confidence,
                "distance_m":  dist,
            }
            self._write_obstacle_to_graph(result_lidar, source="lidar",
                                          dist=dist, blocked_rays=blocked_rays)
            fc = self._state.fleet_coordinator
            fc.v2v_bus.broadcast(self.agent_id, "OBSTACLE_DETECTED", {
                "semantic":         self.semantic_description,
                "blockage_percent": 50,
                "confidence":       self.confidence,
                "distance_m":       dist,
                "position":         cfg.OBSTACLE_POSITION,
            })
            if fc.adapted_plan is None and not fc._pipeline_running:
                new_directives = fc.trigger_immediate_bypass(
                    list(self._state.raft.agent_ids),
                    self._state.agents_positions,
                    self._state.raft, current_step,
                )
                if new_directives:
                    self._state.directives.update(new_directives)

        self._last_detected = detected
        return detected

    # ── Local assessment & fitness ────────────────────────────────────────────

    def compute_local_assessment(self) -> Tuple[bool, bool]:
        speed = self.control_object.speed
        if cfg.OBSTACLE_POSITION:
            dist = np.linalg.norm(
                np.array(self.control_object.position[:2])
                - np.array(cfg.OBSTACLE_POSITION))
            can_stop_safely = dist > (speed ** 2) / (2 * 3.0) + 10
        else:
            can_stop_safely = True
        return can_stop_safely, True

    def compute_fitness_scores(self, obstacle_detected: bool) -> Tuple[float, float, float]:
        pos = self.control_object.position
        vis = self.visibility_score if obstacle_detected else 0.3
        con = self.confidence       if obstacle_detected else 0.5
        ps  = min(pos[0] / 100.0, 1.0)
        lat = 1.0 - abs(pos[1]) / 5.0
        res = 0.5 * ps + 0.5 * max(0, lat)
        self.resource_score = res
        return vis, con, res

    # ── Main act() ────────────────────────────────────────────────────────────

    def act(self, observation, current_step: int = 0) -> List[float]:
        heading = getattr(self.control_object, "heading_theta", 0.0)
        vx = self.control_object.speed * math.cos(heading)
        vy = self.control_object.speed * math.sin(heading)

        self.state[0, 0] = self.control_object.position[0]
        self.state[0, 1] = self.control_object.position[1]
        self.state[1, 0] = vx
        self.state[1, 1] = vy

        pos      = np.array(self.control_object.position[:2])
        my_speed = self.control_object.speed
        in_merge = (
            self._state.fleet_coordinator.state == FormationState.ZIPPER_MERGE
        )

        if isinstance(observation, dict):
            lidar = observation.get("lidar", np.ones(72))
            depth = observation.get("depth_image", None)
        else:
            obs_arr = np.asarray(observation, dtype=np.float32)
            lidar   = obs_arr[:72] if len(obs_arr) >= 72 else np.ones(72)
            depth   = None

        detected               = self.detect_obstacle_semantic(lidar, depth, current_step)
        can_stop, can_maneuver = self.compute_local_assessment()
        vis, conf, res         = self.compute_fitness_scores(detected)

        # [P1] Write Vehicle node + graph diff broadcast
        # Update Vehicle node every 10 steps to reduce graph diff noise
        if current_step % 10 == 0:
            self._state.raft.update_fitness(
                self.agent_id, vis, conf, res,
                graph    = self.graph,
                position = list(pos),
            )
        else:
            self._state.raft.update_fitness(self.agent_id, vis, conf, res)
        self._state.raft.submit_local_assessment(self.agent_id, can_stop, can_maneuver)

        # [P1] Broadcast graph diff if graph changed
        if self.graph.has_changed(self._last_graph_version):
            diff = self.graph.diff(self._prev_snapshot)
            if not diff.is_empty():
                self._state.fleet_coordinator.v2v_bus.broadcast_graph_diff(
                    self.agent_id, diff
                )
            self._last_graph_version = self.graph._version
            self._prev_snapshot      = self.graph.snapshot()

        fc = self._state.fleet_coordinator
        if fc.obstacle_detected and fc.adapted_plan is not None:
            fresh = fc.refresh_directives(
                list(self._state.raft.agent_ids),
                self._state.agents_positions,
                self._state.raft.leader_id,
                current_step=current_step,
            )
            if fresh:
                self._state.directives.update(fresh)

        directives = self._state.directives
        if directives and self.agent_id in directives:
            self.reference_trajectory = directives[self.agent_id]
        else:
            self.reference_trajectory = (
                np.array([pos[0] + 30.0, 0.0]),
                np.array([cfg.LEADER_SPEED, 0.0]),
            )

        if self._state.agents_positions:
            my_assigned_lat = self._get_assigned_lateral()
            lat_steer       = self._compute_lateral_steer(pos)
            ahead: Dict[str, list] = {}

            for aid, p in self._state.agents_positions.items():
                if aid == self.agent_id or p[0] <= pos[0] + 0.5:
                    continue
                if in_merge and my_assigned_lat is not None:
                    other_lat = self._get_assigned_lateral_for(self._state, aid)
                    if other_lat is not None and abs(other_lat - my_assigned_lat) > 2.0:
                        continue
                if abs(p[1] - pos[1]) < 3.5:
                    ahead[aid] = p

            # [FIX] During ZIPPER_MERGE, skip follow-brake so agents can execute
            # bypass. Only brake if dangerously close (<6m) to prevent collision.
            brake_threshold = 6.0 if in_merge else 15.0

            if ahead:
                nearest_id  = min(ahead, key=lambda a: ahead[a][0])
                gap         = ahead[nearest_id][0] - pos[0]
                ahead_speed = 0.0
                if self.env is not None:
                    av = self.env.agent_manager.active_agents.get(nearest_id)
                    if av is not None:
                        ahead_speed = av.speed
                if gap < brake_threshold and my_speed > ahead_speed + 0.5:
                    intensity = min(1.0, (brake_threshold - gap) / brake_threshold)
                    if gap < 10.0:  # only log when actually close
                        print(f"[FOLLOW] {self.agent_id[:8]} follow-brake "
                              f"gap={gap:.1f}m brake={intensity:.2f} steer={lat_steer:.2f}")
                    return [lat_steer, -intensity]

        if cfg.OBSTACLE_POSITION is not None:
            obs_pos_arr = np.array(cfg.OBSTACLE_POSITION[:2])
            dist_to_obs = np.linalg.norm(pos - obs_pos_arr)
            obs_x       = cfg.OBSTACLE_POSITION[0]
            obs_y       = cfg.OBSTACLE_POSITION[1]
            already_past = pos[0] > obs_x + 5.0

            # [FIX] Check lateral clearance — if agent is >3m away from
            # obstacle y-position, it can drive forward freely regardless
            # of Euclidean distance. This prevents stuck-circling near obstacle.
            lateral_gap  = abs(float(pos[1]) - obs_y)
            laterally_clear = lateral_gap > 3.0

            if dist_to_obs < 40.0 and not already_past and not laterally_clear:
                fc            = self._state.fleet_coordinator
                vehicle_order = (fc.adapted_plan or {}).get("vehicle_order", [])
                my_slot = (vehicle_order.index(self.agent_id)
                           if self.agent_id in vehicle_order else 0)
                is_active_merger = (
                    fc.state == FormationState.ZIPPER_MERGE
                    and my_slot <= fc.merge_unlock_index
                )
                if is_active_merger:
                    emerg_lat = fc.merge_assignments.get(self.agent_id)
                    if emerg_lat is None:
                        obs_dy    = pos[1] - obs_y
                        emerg_lat = 4.0 if obs_dy >= 0.0 else -4.0
                    lat_err_emerg = emerg_lat - float(pos[1])
                    emerg_steer   = float(np.clip(lat_err_emerg * 0.4, -1.0, 1.0))

                    if my_speed < 1.0:
                        print(f"[P4-ESCAPE] {self.agent_id[:8]} lateral escape "
                              f"dist={dist_to_obs:.1f}m steer={emerg_steer:.2f}")
                        return [emerg_steer, 0.6]

                    if dist_to_obs < 6.0:
                        print(f"[P4-BRAKE] {self.agent_id[:8]} e-brake "
                              f"dist={dist_to_obs:.1f}m steer={emerg_steer:.2f}")
                        return [emerg_steer, -0.3]

            elif dist_to_obs < 40.0 and not already_past and laterally_clear:
                # Laterally clear — just drive straight forward past obstacle
                print(f"[P4-FWD] {self.agent_id[:8]} laterally clear "
                      f"(gap={lateral_gap:.1f}m) — driving forward")
                return [0.0, 0.8]

        self.profiler.start_mpc()
        steering, throttle, brake = self.execute(current_step=current_step)
        self.profiler.end_mpc()
        return [steering, throttle - brake]

    # ── [P4] execute() — OSQP preferred, CasADi fallback ─────────────────────

    def execute(self, current_step: int = 0) -> Tuple[float, float, float]:
        if self.reference_trajectory is None:
            return 0.0, 0.5, 0.0

        ref_pos, ref_vel = self.reference_trajectory
        ref_pos = np.asarray(ref_pos).flatten()[:2]
        ref_vel = np.asarray(ref_vel).flatten()[:2]

        # ── OSQP path (P4) ────────────────────────────────────────────────────
        if self._use_osqp and self._osqp is not None:
            x0 = np.array([
                self.state[0, 0], self.state[0, 1],
                self.state[1, 0], self.state[1, 1],
            ])

            t0_mpc = time.perf_counter()
            ax, ay_lat = self._osqp.solve(x0, ref_pos, ref_vel)
            dt_ms      = (time.perf_counter() - t0_mpc) * 1000

            # [P4] Log solve time (NF5)
            lg = _get_mpc_logger()
            if lg is not None:
                try:
                    lg.log_solve(self.agent_id, solve_ms=dt_ms,
                                 step=current_step, status="solved")
                except Exception:
                    pass

            # Update state estimate
            self.state[0, 0] += self.state[1, 0] * cfg.MIN_STEER_SPEED * 0.033
            self.state[0, 1] += self.state[1, 1] * 0.033
            self.state[1, 0]  = float(np.clip(self.state[1, 0] + ax * 0.033, 0, 30))
            self.state[1, 1] += ay_lat * 0.033

            # Convert to steering angle (kinematic bicycle model)
            lat_accel = ay_lat * max(float(self.state[1, 0]), cfg.MIN_STEER_SPEED)
            v2        = max(float(self.state[1, 0]) ** 2, 1.0)
            raw_angle = math.atan2(cfg.WHEELBASE * lat_accel, v2)
            steer_cmd = float(np.clip(raw_angle / cfg.MAX_STEER_ANGLE, -1.0, 1.0))

            return steer_cmd, max(0.0, ax), max(0.0, -ax)

        # ── CasADi fallback ───────────────────────────────────────────────────
        from casadi import vertcat
        self.tvp_refs["ref_pos"] = ref_pos.reshape(2, 1)
        self.tvp_refs["ref_vel"] = ref_vel.reshape(2, 1)
        x0 = vertcat(
            self.state[0, 0], self.state[0, 1],
            self.state[1, 0], self.state[1, 1],
        )
        self.mpc.x0 = x0

        try:
            u0 = self.mpc.make_step(x0)
            self.simulator.x0 = x0
            ns = self.simulator.make_step(u0)
            self.state[0] = ns[:2].T
            self.state[1] = ns[2:].T

            acc       = float(u0[0, 0])
            lat_u     = float(u0[1, 0])
            lat_accel = lat_u * max(float(self.state[1, 0]), cfg.MIN_STEER_SPEED)
            v2        = max(float(self.state[1, 0]) ** 2, 1.0)
            raw_angle = math.atan2(cfg.WHEELBASE * lat_accel, v2)
            steer_cmd = float(np.clip(raw_angle / cfg.MAX_STEER_ANGLE, -1.0, 1.0))

            return steer_cmd, max(0.0, acc), max(0.0, -acc)

        except Exception as e:
            print(f"[STAGE 7] MPC error {self.agent_id[:8]}: {e!r}")
            try:
                self.mpc.x0 = x0
                self.mpc.set_initial_guess()
            except Exception:
                pass
            lat_err  = float(ref_pos[1]) - self.state[0, 1]
            long_err = float(ref_vel[0]) - self.state[1, 0]
            steer_fb = float(np.clip(lat_err  * 0.15, -1.0, 1.0))
            acc_fb   = float(np.clip(long_err * 0.10, -1.0, 1.0))
            return steer_fb, max(0.0, acc_fb), max(0.0, -acc_fb)