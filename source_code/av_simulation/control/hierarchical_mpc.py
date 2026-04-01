"""
hierarchical_mpc.py
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from casadi import *
from do_mpc.controller import MPC
from do_mpc.model import Model
from do_mpc.simulator import Simulator

from metadrive.policy.base_policy import BasePolicy

from av_simulation.config import simulation_config as cfg
from av_simulation import SharedSimState
from av_simulation.coordination.fleet_coordinator import FormationState
from av_simulation.utils.sim_context import (
    EnhancedMultiAgentEnv,
    capture_agent_frame,
)
from av_simulation.vision_language.vlm_engine import parse_vlm_output

def setup_mpc(
    obs_pos=None,
    safe_dist: float = 5.0,
    cost_weight: float = 1.0,
):

    model   = Model('continuous')
    pos     = model.set_variable('_x', 'pos', (2, 1))
    vel     = model.set_variable('_x', 'vel', (2, 1))
    u       = model.set_variable('_u', 'u',   (2, 1))
    ref_pos = model.set_variable('_tvp', 'ref_pos', (2, 1))
    ref_vel = model.set_variable('_tvp', 'ref_vel', (2, 1))

    model.set_rhs('pos', vel)
    model.set_rhs('vel', vertcat(u[0], u[1] * fmax(vel[0], cfg.MIN_STEER_SPEED)))
    model.setup()

    mpc       = MPC(model)
    n_horizon = 20
    mpc.set_param(n_horizon=n_horizon, t_step=0.1, n_robust=1)

    # [STEER-7] 8× lateral weight to actively pursue merge targets
    lterm = (
        (pos[0] - ref_pos[0]) ** 2
        + 8.0 * (pos[1] - ref_pos[1]) ** 2
        + sum1((vel - ref_vel) ** 2)
    )
    mpc.set_objective(mterm=lterm, lterm=lterm)
    mpc.set_rterm(u=0.05)
    mpc.bounds['lower', '_u', 'u'] = [-3.0, -1.0]
    mpc.bounds['upper', '_u', 'u'] = [ 3.0,  1.0]

    if obs_pos is not None:
        dist = sqrt(sum1((pos - vertcat(obs_pos[0], obs_pos[1])) ** 2))
        mpc.set_nl_cons(
            'collision_avoid',
            -dist + safe_dist,
            ub=0,
            soft_constraint=True,
            penalty_term_cons=1e4 * max(cost_weight, 1.0),
        )

    tvp_template = mpc.get_tvp_template()
    tvp_refs: Dict[str, np.ndarray] = {
        'ref_pos': np.array([[0.0], [0.0]]),
        'ref_vel': np.array([[20.0], [0.0]]),
    }

    def tvp_fun(t_now):
        for k in range(n_horizon + 1):
            tvp_template['_tvp', k, 'ref_pos'] = tvp_refs['ref_pos']
            tvp_template['_tvp', k, 'ref_vel'] = tvp_refs['ref_vel']
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    sim  = Simulator(model)
    sim.set_param(t_step=0.1)
    stpl = sim.get_tvp_template()

    def sim_tvp_fun(t_now):
        stpl['ref_pos'] = tvp_refs['ref_pos']
        stpl['ref_vel'] = tvp_refs['ref_vel']
        return stpl

    sim.set_tvp_fun(sim_tvp_fun)
    sim.setup()

    # [STEER-5] Provide initial guess before first make_step()
    x0_init = np.zeros((4, 1))
    mpc.x0  = x0_init
    sim.x0  = x0_init
    mpc.set_initial_guess()

    return mpc, sim, tvp_refs

class VLAPolicy(BasePolicy):

    def __init__(
        self,
        control_object,
        random_seed: int,
        shared_state: SharedSimState,
        env=None,
        profiler=None,
        obstacle_manager=None,
    ) -> None:
        super().__init__(control_object, random_seed)

        self.agent_id         = self.control_object.name
        self.env              = env
        self._state           = shared_state
        self.obstacle_manager = obstacle_manager

        # Import here to avoid a circular at module load time
        from av_simulation.utils.sim_context import PerformanceProfiler
        self.profiler = profiler or PerformanceProfiler(enabled=False)

        # Detection state
        self.fitness:              float           = 0.0
        self.semantic_description: Optional[str]   = None
        self.confidence:           float           = 0.0
        self.visibility_score:     float           = 0.0
        self.resource_score:       float           = 0.0
        self.reference_trajectory: Optional[tuple] = None
        # V2V state
        self.obstacle_broadcast_done = False
        # VLM agent role
        self._vlm_step_counter = -(cfg.VLM_INFERENCE_INTERVAL - 1)
        self._last_detected    = False
        self._is_vlm_agent     = False
        self._peer_semantic:    Optional[str]   = None
        self._peer_confidence:  float           = 0.0
        self._peer_blockage:    int             = 0

        # MPC initialisation
        obs_pos = cfg.OBSTACLE_POSITION if cfg.OBSTACLE_POSITION is not None else [80.0, 0.0]
        self.mpc, self.simulator, self.tvp_refs = setup_mpc(
            obs_pos=obs_pos, safe_dist=cfg.SAFE_DISTANCE
        )
        self.state = np.array([[0.0, 0.0], [0.0, 0.0]])

        # Subscribe to V2V bus
        self._state.fleet_coordinator.v2v_bus.subscribe(
            self.agent_id, self._on_v2v_message
        )

    def _on_v2v_message(self, msg: dict) -> None:
        if msg['type'] == 'OBSTACLE_DETECTED':
            pl = msg['payload']
            self._peer_semantic   = pl.get('semantic', '')
            self._peer_confidence = pl.get('confidence', 0.0)
            self._peer_blockage   = pl.get('blockage_percent', 0)
            self._last_detected   = True

    def _compute_lateral_steer(self, pos: np.ndarray) -> float:
    
        if self.reference_trajectory is None:
            return 0.0
        ref_pos, _ = self.reference_trajectory
        lat_err    = float(ref_pos[1]) - float(pos[1])

        fc           = self._state.fleet_coordinator
        vehicle_order = (fc.adapted_plan or {}).get("vehicle_order", [])
        my_slot = (
            vehicle_order.index(self.agent_id)
            if self.agent_id in vehicle_order else 0
        )
        is_active_merger = (
            fc.state == FormationState.ZIPPER_MERGE
            and my_slot <= fc.merge_unlock_index
        )
        # Active mergers: full proportional gain.
        # Holders / platoon-cruising: very small gain suppresses drift steer.
        gain = 0.25 if is_active_merger else 0.05
        return float(np.clip(lat_err * gain, -1.0, 1.0))

    def _get_assigned_lateral(self) -> Optional[float]:
        return self._state.fleet_coordinator.merge_assignments.get(self.agent_id)

    @staticmethod
    def _get_assigned_lateral_for(
        state: SharedSimState, aid: str
    ) -> Optional[float]:
        return state.fleet_coordinator.merge_assignments.get(aid)

    def detect_obstacle_semantic(
        self,
        lidar:         np.ndarray,
        depth:         Optional[np.ndarray],
        current_step:  int = 0,
    ) -> bool:
    
        if depth is not None:
            lidar = EnhancedMultiAgentEnv.fuse_lidar_depth(lidar, depth)

        pos  = np.array(self.control_object.position[:2])
        dist = (
            np.linalg.norm(pos - np.array(cfg.OBSTACLE_POSITION))
            if cfg.OBSTACLE_POSITION else 999.0
        )
        blocked_rays = int(np.sum(lidar < 0.25))
        lidar_hit    = blocked_rays > 2 or dist < 60.0

        if self._is_vlm_agent:
            self._vlm_step_counter += 1
            if self._vlm_step_counter % cfg.VLM_INFERENCE_INTERVAL != 0:
            
                if not self._last_detected and lidar_hit:
                    print(
                        f"[VLM] Inter-frame lidar override for "
                        f"{self.agent_id[:8]} "
                        f"(blocked={blocked_rays}, dist={dist:.0f}m)"
                    )
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

            if not vlm_detected and lidar_hit:
                print(
                    f"[VLM] Override: lidar contradicts 'ROAD CLEAR' "
                    f"(blocked={blocked_rays}, dist={dist:.0f}m) — "
                    f"treating as detected"
                )
                vlm_detected         = True
                result["confidence"] = max(result["confidence"], 0.6)
                result["distance_m"] = dist

            self.semantic_description = result["semantic_description"]
            self.confidence           = result["confidence"]
            self.visibility_score     = max(0.1, 1.0 - result["distance_m"] / 100.0)
            self._last_detected       = vlm_detected

            if vlm_detected and not self.obstacle_broadcast_done:
                fc = self._state.fleet_coordinator
                fc.v2v_bus.broadcast(
                    self.agent_id,
                    "OBSTACLE_DETECTED",
                    {
                        "semantic":         result["semantic_description"],
                        "blockage_percent": result["blockage_percent"],
                        "confidence":       result["confidence"],
                        "distance_m":       result["distance_m"],
                        "position":         cfg.OBSTACLE_POSITION,
                    },
                )
                self.obstacle_broadcast_done = True
               
                if fc.adapted_plan is None and not fc._pipeline_running:
                    new_directives = fc.trigger_immediate_bypass(
                        list(self._state.raft.agent_ids),
                        self._state.agents_positions,
                        self._state.raft,
                        current_step,
                    )
                    if new_directives:
                        self._state.directives.update(new_directives)
            return vlm_detected

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
                f"[lidar-fused] Obstacle at {dist:.0f}m "
                f"(blocked_rays={blocked_rays})"
            )
        if detected and not self.obstacle_broadcast_done:
            self.obstacle_broadcast_done = True
            fc = self._state.fleet_coordinator
            fc.v2v_bus.broadcast(
                self.agent_id,
                "OBSTACLE_DETECTED",
                {
                    "semantic":         self.semantic_description,
                    "blockage_percent": 50,
                    "confidence":       self.confidence,
                    "distance_m":       dist,
                    "position":         cfg.OBSTACLE_POSITION,
                },
            )

            if fc.adapted_plan is None and not fc._pipeline_running:
                new_directives = fc.trigger_immediate_bypass(
                    list(self._state.raft.agent_ids),
                    self._state.agents_positions,
                    self._state.raft,
                    current_step,
                )
                if new_directives:
                    self._state.directives.update(new_directives)

        self._last_detected = detected
        return detected

    def compute_local_assessment(self) -> Tuple[bool, bool]:
        """Return (can_stop_safely, can_maneuver) for this vehicle."""
        speed = self.control_object.speed
        if cfg.OBSTACLE_POSITION:
            dist = np.linalg.norm(
                np.array(self.control_object.position[:2])
                - np.array(cfg.OBSTACLE_POSITION)
            )
            can_stop_safely = dist > (speed ** 2) / (2 * 3.0) + 10
        else:
            can_stop_safely = True
        return can_stop_safely, True

    def compute_fitness_scores(
        self, obstacle_detected: bool
    ) -> Tuple[float, float, float]:
        """Return (visibility, confidence, resource) scores for Raft."""
        pos = self.control_object.position
        vis = self.visibility_score if obstacle_detected else 0.3
        con = self.confidence       if obstacle_detected else 0.5
        ps  = min(pos[0] / 100.0, 1.0)
        lat = 1.0 - abs(pos[1]) / 5.0
        res = 0.5 * ps + 0.5 * max(0, lat)
        self.resource_score = res
        return vis, con, res

    def act(self, observation, current_step: int = 0) -> List[float]:
      
        heading = getattr(self.control_object, 'heading_theta', 0.0)
        vx = self.control_object.speed * math.cos(heading)
        vy = self.control_object.speed * math.sin(heading)

        self.state[0, 0] = self.control_object.position[0]
        self.state[0, 1] = self.control_object.position[1]
        self.state[1, 0] = vx
        self.state[1, 1] = vy   # [STEER-1]

        pos      = np.array(self.control_object.position[:2])
        my_speed = self.control_object.speed
        in_merge = (
            self._state.fleet_coordinator.state == FormationState.ZIPPER_MERGE
        )

        if isinstance(observation, dict):
            lidar = observation.get('lidar', np.ones(72))
            depth = observation.get('depth_image', None)
        else:
            obs_arr = np.asarray(observation, dtype=np.float32)
            lidar   = obs_arr[:72] if len(obs_arr) >= 72 else np.ones(72)
            depth   = None

        detected               = self.detect_obstacle_semantic(
            lidar, depth, current_step
        )
        can_stop, can_maneuver = self.compute_local_assessment()
        vis, conf, res         = self.compute_fitness_scores(detected)
        self._state.raft.update_fitness(self.agent_id, vis, conf, res)
        self._state.raft.submit_local_assessment(
            self.agent_id, can_stop, can_maneuver
        )

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
            
            lat_steer = self._compute_lateral_steer(pos)
            ahead: Dict[str, list] = {}

            for aid, p in self._state.agents_positions.items():
                if aid == self.agent_id or p[0] <= pos[0] + 0.5:
                    continue
                if in_merge and my_assigned_lat is not None:
                    other_lat = self._get_assigned_lateral_for(
                        self._state, aid
                    )
                    if other_lat is not None and abs(other_lat - my_assigned_lat) > 2.0:
                        continue
                if abs(p[1] - pos[1]) < 3.5:
                    ahead[aid] = p

            if ahead:
                nearest_id  = min(ahead, key=lambda a: ahead[a][0])
                gap         = ahead[nearest_id][0] - pos[0]
                ahead_speed = 0.0
                if self.env is not None:
                    av = self.env.agent_manager.active_agents.get(nearest_id)
                    if av is not None:
                        ahead_speed = av.speed
                if gap < 15.0 and my_speed > ahead_speed + 0.5:
                    intensity = min(1.0, (15.0 - gap) / 10.0)
                    print(
                        f"[FOLLOW] {self.agent_id[:8]} follow-brake "
                        f"gap={gap:.1f}m brake={intensity:.2f} "
                        f"steer={lat_steer:.2f}"
                    )
                    # [FIX-B] Return steer + brake, not [0, -brake]
                    return [lat_steer, -intensity]

        if cfg.OBSTACLE_POSITION is not None:
            dist_to_obs = np.linalg.norm(pos - np.array(cfg.OBSTACLE_POSITION))
            if dist_to_obs < 40.0:
                
                fc           = self._state.fleet_coordinator
                vehicle_order = (fc.adapted_plan or {}).get("vehicle_order", [])
                my_slot = (
                    vehicle_order.index(self.agent_id)
                    if self.agent_id in vehicle_order else 0
                )
                is_active_merger = (
                    fc.state == FormationState.ZIPPER_MERGE
                    and my_slot <= fc.merge_unlock_index
                )
                if is_active_merger:
                    emerg_lat = fc.merge_assignments.get(self.agent_id)
                    if emerg_lat is None:
                        # Steer left (positive) by default — away from centre
                        obs_dy = pos[1] - cfg.OBSTACLE_POSITION[1]
                        emerg_lat = 4.0 if obs_dy >= 0.0 else -4.0
                    lat_err_emerg = emerg_lat - float(pos[1])
                    emerg_steer   = float(np.clip(lat_err_emerg * 0.4, -1.0, 1.0))

                    # [FIX-D] Speed guard: stopped vehicles escape laterally
                    if my_speed < 0.5:
                        print(
                            f"[P4-ESCAPE] {self.agent_id[:8]} lateral escape "
                            f"dist={dist_to_obs:.1f}m steer={emerg_steer:.2f}"
                        )
                        return [emerg_steer, 0.3]

                    stop_dist = (my_speed ** 2) / (2 * 4.0)
                    if stop_dist > dist_to_obs - 5.0:
                        print(
                            f"[P4-BRAKE] {self.agent_id[:8]} e-brake "
                            f"dist={dist_to_obs:.1f}m steer={emerg_steer:.2f}"
                        )
                        return [emerg_steer, -1.0]

        self.profiler.start_mpc()
        steering, throttle, brake = self.execute()
        self.profiler.end_mpc()
        return [steering, throttle - brake]

    def execute(self) -> Tuple[float, float, float]:

        if self.reference_trajectory is None:
            return 0.0, 0.5, 0.0

        ref_pos, ref_vel = self.reference_trajectory
        self.tvp_refs['ref_pos'] = ref_pos.reshape(2, 1)
        self.tvp_refs['ref_vel'] = ref_vel.reshape(2, 1)
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

            # [STEER-6] Explicit 2-D indexing
            acc       = float(u0[0, 0])
            lat_u     = float(u0[1, 0])
            lat_accel = lat_u * max(float(self.state[1, 0]), cfg.MIN_STEER_SPEED)

            # [STEER-2] Kinematic bicycle model: δ = atan(L·a_y / v²)
            v2        = max(float(self.state[1, 0]) ** 2, 1.0)
            raw_angle = math.atan2(cfg.WHEELBASE * lat_accel, v2)
            steer_cmd = float(np.clip(raw_angle / cfg.MAX_STEER_ANGLE, -1.0, 1.0))

            return steer_cmd, max(0.0, acc), max(0.0, -acc)

        except Exception as e:
            # [STEER-8] Surface error, recover solver, proportional fallback
            print(f"[STAGE 7] MPC error {self.agent_id[:8]}: {e!r}")
            try:
                self.mpc.x0 = x0
                self.mpc.set_initial_guess()
            except Exception:
                pass
            ref_pos, ref_vel = self.reference_trajectory
            lat_err  = float(ref_pos[1]) - self.state[0, 1]
            long_err = float(ref_vel[0]) - self.state[1, 0]
            steer_fb = float(np.clip(lat_err  * 0.15, -1.0, 1.0))
            acc_fb   = float(np.clip(long_err * 0.10, -1.0, 1.0))
            return steer_fb, max(0.0, acc_fb), max(0.0, -acc_fb)
