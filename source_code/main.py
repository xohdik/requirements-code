"""
main.py
=======
Entry point for the VLA-MAC Fleet Coordination Simulation.

Phase 3 changes (marked  # [P3]):
  - run_simulation() now returns a metrics dict (for benchmark runner)
  - Termination bug fixed: mission_complete condition replaces crash-based done
  - execute_strategy_pipeline called with graph= argument (P2 LLM path)
  - ExecutionStatus nodes written to graph after each MPC step (F8)
  - --no-render flag added for headless benchmark runs
  - Metrics tracked: task_success, ttfm, fct, graph_diffs, collisions
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from metadrive.utils import setup_logger

from av_simulation import SharedSimState
from av_simulation.config import simulation_config as cfg
from av_simulation.config.simulation_config import parse_args, MAP_PRESETS

from av_simulation.coordination.fleet_coordinator import (
    V2VBus,
    StrategyRepository,
    LLMStrategyAdapter,
    SimpleRaft,
    FleetCoordinator,
)
from av_simulation.control.hierarchical_mpc import VLAPolicy

from av_simulation.utils.sim_context import (
    EnhancedMultiAgentEnv,
    MultiObstacleManager,
    PerformanceProfiler,
    RealWorldDataLoader,
    WaypointPlanner,
    assign_vlm_agent,
    attach_front_camera,
    detach_all_front_cameras,
    get_agent_spawn_lane,
)
from av_simulation.vision_language.vlm_engine import VLMEngine


# =============================================================================
# run_simulation
# =============================================================================

def run_simulation(args=None) -> dict:
    """
    Initialise all subsystems and run the main simulation loop.

    Returns
    -------
    dict with keys:
        task_success  : bool   — True if all agents cleared obstacle
        ttfm          : float  — time-to-first-move (steps)
        fct           : float  — fleet clearance time (steps)
        graph_diffs   : int    — total GraphDiff broadcasts
        llm_tokens    : int    — estimated LLM tokens used (placeholder)
        collisions    : int    — total collision events
        steps_run     : int    — actual steps executed
    """

    if args is None:
        args = parse_args()

    cfg.SIMULATION_STEPS = args.steps
    NUM_AGENTS_RUN       = args.num_agents
    profiler             = PerformanceProfiler(enabled=args.profile)

    # [P3] --no-render support for headless benchmark runs
    use_render = not getattr(args, "no_render", False)

    setup_logger(True)
    rw_extra_config = RealWorldDataLoader.get_env_config(args)

    # [P4-NF5] Enable auditability logging
    try:
        import importlib.util as _ilu, os as _os
        _spec = _ilu.spec_from_file_location(
            "av_simulation.utils.logger",
            _os.path.join(_os.path.dirname(__file__),
                          "av_simulation", "utils", "logger.py"),
        )
        _lmod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_lmod)
        _lmod.enable_all_loggers()
    except Exception as _e:
        print(f"[NF5] Logger init failed (non-fatal): {_e}")

    print("=" * 60)
    print(f"Scenario : {args.env}  ({MAP_PRESETS[args.env]['description']})")
    print(f"Agents   : {NUM_AGENTS_RUN}")
    print(f"Obstacles: {args.num_obstacles}")
    print(f"Profile  : {args.profile}")
    print(f"Render   : {use_render}")
    print("=" * 60)

    # ── Subsystem construction ─────────────────────────────────────────────
    vlm_engine       = VLMEngine()
    obstacle_manager = MultiObstacleManager(
        num_obstacles = args.num_obstacles,
        seed          = getattr(args, "seed", 42),
    )

    v2v_bus           = V2VBus()
    strategy_repo     = StrategyRepository()
    llm_adapter       = LLMStrategyAdapter()
    raft              = SimpleRaft()
    fleet_coordinator = FleetCoordinator(v2v_bus, strategy_repo, llm_adapter)

    shared = SharedSimState(
        raft              = raft,
        fleet_coordinator = fleet_coordinator,
        agents_positions  = {},
        vlm_engine        = vlm_engine,
        directives        = {},
        graphs            = {},     # [P1] per-agent SemanticGraph dict
    )

    print("=" * 60)

    # ── MetaDrive environment config ───────────────────────────────────────
    agent_configs: dict = {}
    for i in range(NUM_AGENTS_RUN):
        agent_configs[f"agent{i}"] = {
            "spawn_lane_index": cfg.SPAWN_LANE_INDEX,
            "spawn_longitude":  10.0 + i * cfg.PLATOON_SPACING,
            "spawn_lateral":    0.0,
        }

    map_str             = MAP_PRESETS[args.env]["map"]
    traffic_density_val = args.traffic_density if args.reactive_traffic else 0.0

    env_config = {
        "use_render":         use_render,
        "num_agents":         NUM_AGENTS_RUN,
        "map":                map_str,
        "traffic_mode":       "Trigger",
        "horizon":            cfg.SIMULATION_STEPS,
        "agent_configs":      agent_configs,
        "show_logo": False,
        "vehicle_config": {
            "lidar": {"num_lasers": 72, "distance": 50, "num_others": 4},
            "show_lidar": False,
        },
        "accident_prob":      0.0,
        "traffic_density":    traffic_density_val,
        "crash_vehicle_done": False,   # [P3-BUG] override in EnhancedMultiAgentEnv too
        "crash_object_done":  False,
        "out_of_route_done":  False,
        "out_of_road_done":   False,   # agents must survive bypass manoeuvre
        "use_semantic":       False,
        "use_depth":          False,
    }
    env_config.update(rw_extra_config)

    env          = EnhancedMultiAgentEnv(env_config)
    reset_result = env.reset()
    o = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    # ── Optional top-down camera ───────────────────────────────────────────
    if use_render and getattr(args, "top_down", False):
        try:
            env.main_camera.camera.setPos(0, 0, 200)
            env.main_camera.camera.lookAt(0, 0, 0)
            print("[NEW-1] Top-down camera activated.")
        except Exception as e:
            print(f"[NEW-1] Top-down camera failed (non-fatal): {e}")

    # ── Spawn obstacles ────────────────────────────────────────────────────
    obstacle_lane = get_agent_spawn_lane(env)
    if obstacle_lane is None:
        raise RuntimeError("Could not find agent spawn lane.")
    obstacle_manager.spawn_all(env, obstacle_lane)

    print("=" * 60)
    print("PIPELINE STAGES ACTIVE:")
    print("  1. Semantic Detection — VLM + lidar cross-check  [FIX-E]")
    print("  2. V2V Broadcast + immediate fleet trigger  [STEER-4]")
    print("  3. Local Assessment + cost tracking")
    print("  4. Raft Leader Election  (runs every step)  [FIX-A]")
    print("  5. Strategy Selection (SQLite + graph templates)  [P2]")
    print("  6. Waypoint-Adapted Strategy")
    print("  7. MPC — steer preserved in all return paths  [FIX-B/C/D]")
    print("  8. ExecutionStatus node written after MPC  [P3-F8]")
    print("  9. Reformation + ASCII map")
    print("=" * 60)

    # ── Instantiate per-agent policies ─────────────────────────────────────
    agent_policies: Dict[str, VLAPolicy] = {}

    for agent_id, vehicle in env.agent_manager.active_agents.items():
        raft.register_agent(agent_id)
        policy = VLAPolicy(
            vehicle,
            random_seed      = 0,
            shared_state     = shared,
            env              = env,
            profiler         = profiler,
            obstacle_manager = obstacle_manager,
        )
        agent_policies[agent_id] = policy
        attach_front_camera(env, vehicle)

    shared.agents_positions = {
        aid: [v.position[0], v.position[1]]
        for aid, v in env.agent_manager.active_agents.items()
    }

    for aid in raft.agent_ids:
        raft.update_fitness(aid, 0.5, 0.5, 0.5,
                            graph    = shared.graphs.get(aid),
                            position = shared.agents_positions.get(aid))
    raft.elect_leader(graph=shared.graphs.get(raft.leader_id))
    assign_vlm_agent(agent_policies, shared.agents_positions)

    shared.directives = fleet_coordinator.get_formation_directives(
        list(raft.agent_ids),
        shared.agents_positions,
        raft.leader_id,
        current_step=0,
    )

    print(
        f"[DIAG] {len(agent_policies)} agents, {cfg.SIMULATION_STEPS} steps, "
        f"map='{map_str}', obstacles={len(obstacle_manager.obstacle_positions)}"
    )

    active_count      = len(agent_policies)
    pipeline_executed = False
    wp_planner        = WaypointPlanner()

    # [P3] Metrics tracking
    metrics = {
        "task_success": False,
        "ttfm":         None,   # step when first agent started moving toward obstacle
        "fct":          None,   # step when last agent cleared obstacle
        "graph_diffs":  0,
        "llm_tokens":   0,
        "collisions":   0,
        "steps_run":    0,
    }
    first_move_logged  = False
    agents_cleared: set = set()

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    for step in range(cfg.SIMULATION_STEPS):
        profiler.start_step()

        active_agents_this_step = []
        new_positions: Dict[str, list] = {}

        for aid in list(o.keys()):
            if aid in env.agent_manager.active_agents:
                veh = env.agent_manager.active_agents[aid]
                new_positions[aid] = [veh.position[0], veh.position[1]]
                active_agents_this_step.append(aid)

                if aid not in agent_policies:
                    raft.register_agent(aid)
                    policy = VLAPolicy(
                        veh,
                        random_seed      = 0,
                        shared_state     = shared,
                        env              = env,
                        profiler         = profiler,
                        obstacle_manager = obstacle_manager,
                    )
                    agent_policies[aid] = policy
                    attach_front_camera(env, veh)
                    assign_vlm_agent(agent_policies, shared.agents_positions)

        shared.agents_positions.update(new_positions)

        # ── Fleet state machine tick ───────────────────────────────────────
        trigger_pipeline = fleet_coordinator.update_state(
            shared.agents_positions, cfg.OBSTACLE_POSITION, step
        )

        if trigger_pipeline and not pipeline_executed:
            print("\n" + "=" * 60)
            print("EXECUTING FULL PIPELINE  [ALL STAGES]")
            print("=" * 60)
            raft.elect_leader(graph=shared.graphs.get(raft.leader_id))
            assign_vlm_agent(agent_policies, shared.agents_positions)

            if not raft.check_group_stop():
                # [P3] Pass leader graph for P2 LLM-driven strategy selection
                leader_graph = shared.graphs.get(raft.leader_id)
                fleet_coordinator.execute_strategy_pipeline(
                    list(raft.agent_ids),
                    shared.agents_positions,
                    raft,
                    graph=leader_graph,   # [P2] enables LLM path
                )

            pipeline_executed = True
            print("=" * 60 + "\n")

            # [P3] Track time-to-first-move
            if metrics["ttfm"] is None:
                metrics["ttfm"] = step
                first_move_logged = True

        # [FIX] pipeline_executed locks permanently once the pipeline fires.
        # Previously reset to False on PLATOON re-entry, causing re-execution.
        if fleet_coordinator.obstacle_detected or fleet_coordinator.obstacle_cleared:
            pipeline_executed = True

        if step % 30 == 0:
            assign_vlm_agent(agent_policies, shared.agents_positions)

        # ── Formation directives ───────────────────────────────────────────
        if shared.agents_positions:
            leader_vel = None
            if raft.leader_id and raft.leader_id in env.agent_manager.active_agents:
                leader_vel = env.agent_manager.active_agents[raft.leader_id].speed

            shared.directives = fleet_coordinator.get_formation_directives(
                list(raft.agent_ids),
                shared.agents_positions,
                raft.leader_id,
                leader_vel,
                current_step=step,
            )

        # ── ASCII mini-map ─────────────────────────────────────────────────
        if step % 60 == 0 and shared.agents_positions and use_render:
            ascii_map = wp_planner.render_ascii_map(
                shared.agents_positions, obstacle_manager.obstacle_positions
            )
            if ascii_map:
                print(ascii_map)

        # ── Agent actions ──────────────────────────────────────────────────
        a = {
            k: agent_policies[k].act(o[k], current_step=step)
            for k in o
            if k in agent_policies and k in active_agents_this_step
        }

        # [P3-F8] Write ExecutionStatus nodes to each agent's graph
        _write_execution_status(
            agent_policies, shared, active_agents_this_step, step
        )

        # ── Step environment ───────────────────────────────────────────────
        step_result = env.step(a)
        if len(step_result) == 5:
            o, r, terminated, truncated, i = step_result
            d = {
                k: terminated.get(k, False) or truncated.get(k, False)
                for k in terminated
            }
        else:
            o, r, d, i = step_result

        # [P3] Count graph diffs — running total from full message log
        metrics["graph_diffs"] = sum(
            1 for m in fleet_coordinator.v2v_bus.messages
            if isinstance(m, dict) and m.get("type") == "GRAPH_DIFF"
        )

        # ── Collision logging ──────────────────────────────────────────────
        for aid, info in (i.items() if isinstance(i, dict) else []):
            if isinstance(info, dict):
                if info.get("crash_vehicle", False):
                    print(f"[COLLISION] {aid[:8]} hit vehicle!")
                    metrics["collisions"] += 1
                if info.get("crash_object", False):
                    print(f"[COLLISION] {aid[:8]} hit object!")
                    metrics["collisions"] += 1

        if use_render:
            env.render()
        profiler.end_step()

        # ── [P3-BUG] Mission complete check (replaces crash-based done) ────
        # Count agents that have cleared the obstacle (>50m past it)
        if cfg.OBSTACLE_POSITION:
            obs_x = cfg.OBSTACLE_POSITION[0]
            for aid, pos in shared.agents_positions.items():
                if pos[0] > obs_x + 50.0:
                    agents_cleared.add(aid)

            if len(agents_cleared) >= active_count and active_count > 0:
                if metrics["fct"] is None:
                    metrics["fct"] = step
                print(f"[P3] Mission complete — all {active_count} agents "
                      f"cleared obstacle at step {step}")
                metrics["task_success"] = True
                break

        # [P3-BUG] Only break on done if it's a genuine arrival, not a crash
        # (EnhancedMultiAgentEnv.override handles this — this is a safety net)
        if d:
            genuine_done = {
                k: v for k, v in d.items()
                if isinstance(i.get(k), dict) and i[k].get("arrive_dest", False)
            }
            if len(genuine_done) == active_count and active_count > 0:
                print(f"[DIAG] All agents arrived at destination at step {step}")
                metrics["task_success"] = True
                break
            # Do NOT break on crash-based done

        # ── Periodic diagnostics ───────────────────────────────────────────
        if step % 60 == 0:
            fleet_cost = obstacle_manager.compute_fleet_cost(shared.agents_positions)
            print(
                f"[DIAG] Step {step}/{cfg.SIMULATION_STEPS} | "
                f"State: {fleet_coordinator.state} | "
                f"Active: {len(active_agents_this_step)} | "
                f"Leader: {raft.leader_id[:8] if raft.leader_id else 'None'} | "
                f"FleetCost: {fleet_cost:.3f} | "
                f"Cleared: {len(agents_cleared)}/{active_count}"
            )

        if use_render:
            time.sleep(0.033)

    metrics["steps_run"] = step + 1

    # If fct was never set but task succeeded implicitly
    if metrics["task_success"] and metrics["fct"] is None:
        metrics["fct"] = step

    print(f"[DIAG] Simulation complete — {step + 1} steps")
    print(f"[P3]  Metrics: {metrics}")
    profiler.summary()
    detach_all_front_cameras()
    env.close()

    return metrics


# =============================================================================
# [P3-F8] ExecutionStatus node writer
# =============================================================================

def _write_execution_status(
    agent_policies:         Dict[str, "VLAPolicy"],
    shared:                 "SharedSimState",
    active_agents_this_step: list,
    step:                   int,
) -> None:
    """
    Write an ExecutionStatus node to each agent's graph after the MPC step.
    Satisfies F8: tracking_error, completed, safety_warning.
    """
    import numpy as np, time as _time

    if cfg.OBSTACLE_POSITION is None:
        return

    obs_pos = np.array(cfg.OBSTACLE_POSITION[:2])

    for aid in active_agents_this_step:
        policy = agent_policies.get(aid)
        if policy is None:
            continue

        graph = shared.graphs.get(aid)
        if graph is None:
            continue

        pos = np.array(shared.agents_positions.get(aid, [0, 0])[:2])

        # Compute tracking error vs reference trajectory
        tracking_error = 0.0
        if policy.reference_trajectory is not None:
            ref_pos, _ = policy.reference_trajectory
            tracking_error = float(np.linalg.norm(pos - ref_pos[:2]))

        dist_to_obs = float(np.linalg.norm(pos - obs_pos))
        completed   = dist_to_obs > 50.0      # >50m past obstacle = cleared
        safety_warn = tracking_error > 8.0    # large deviation = warning

        node_id = f"execstatus_{aid[:8]}_{step}"
        graph.add_node(
            node_id   = node_id,
            node_type = "ExecutionStatus",
            attrs     = {
                "tracking_error":  round(tracking_error, 3),
                "completed":       completed,
                "safety_warning":  safety_warn,
                "step":            step,
                "dist_to_obstacle": round(dist_to_obs, 2),
            },
            source    = aid,
            timestamp = _time.time(),
        )

        # Vehicle --executed--> ExecutionStatus
        vehicle_node_id = f"vehicle_{aid}"
        if graph.get_node(vehicle_node_id) is not None:
            graph.add_edge(
                vehicle_node_id, node_id, "executed",
                {}, aid, _time.time(),
            )

        # Trigger replan if tracking error is dangerously high
        if safety_warn and shared.fleet_coordinator.obstacle_detected:
            print(
                f"[P3-F8] {aid[:8]} safety_warning: "
                f"tracking_error={tracking_error:.2f}m"
            )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    run_simulation()