"""
main.py
=======
Entry point for the VLA-MAC Fleet Coordination Simulation.

Wires together all modules:
  config        → constants + CLI (command line interfrace) args
  utils         → env, obstacle manager, cameras, helpers
  coordination  → V2V, Raft, strategy repo, fleet coordinator
  control       → MPC + VLAPolicy
  vision_language → VLMEngine

Run with:
    python main.py [--env straight] [--num_agents 4] [--steps 1800] ...
"""

from __future__ import annotations # Enables forward references in type hints (e.g., using a class name before it's defined).

import time # Standard library: provides sleep() for loop timing (0.033s ≈ 30 FPS) and time measurement.

from typing import Dict, Optional # Type hint imports: Dict for key-value mappings (e.g., agent_id → policy), Optional for values that may be None.

from metadrive.utils import setup_logger # MetaDrive utility: configures logging output format and verbosity level.

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------
from av_simulation import SharedSimState # Imports the SharedSimState dataclass from the package root __init__.py. This dataclass holds references to singletons (raft, fleet_coordinator, vlm_engine)
# and mutable state (agents_positions, directives) shared across all VLAPolicy instances.

from av_simulation.config import simulation_config as cfg # Imports the simulation_config module and aliases it as 'cfg' for brevity. Contains immutable constants (SIMULATION_STEPS, PLATOON_SPACING, etc.).
from av_simulation.config.simulation_config import parse_args, MAP_PRESETS # MAP_PRESETS: dictionary mapping scenario names to map strings and descriptions.

from av_simulation.coordination.fleet_coordinator import (
    V2VBus, # Vehicle-to-Vehicle communication bus: handles message passing between agents.
    StrategyRepository, # SQLite-backed repository for storing/retrieving formation strategies.
    LLMStrategyAdapter,  # Adapts LLM/VLM outputs into structured strategies for fleet coordination.
    SimpleRaft, # Minimal Raft consensus implementation: leader election, fitness tracking.
    FleetCoordinator,# High-level coordinator: state machine, obstacle detection, pipeline triggering.
)
from av_simulation.control.hierarchical_mpc import VLAPolicy # Imports the VLAPolicy class: per-agent controller combining VLM semantic input with MPC trajectory planning.

from av_simulation.utils.sim_context import (
    EnhancedMultiAgentEnv,
    MultiObstacleManager,# Manages spawning, positioning, and cost computation for static/dynamic obstacles.

    PerformanceProfiler, # Step-level performance timing: start_step(), end_step(), summary().
    RealWorldDataLoader,# Loads real-world data (e.g., Waymo) to override environment config.

    WaypointPlanner,# Generates ASCII maps and waypoint-based navigation helpers.
    assign_vlm_agent,# Selects which agent is responsible for VLM queries based on position/leadership.

    attach_front_camera,# Adds a front-facing camera to a vehicle for VLM image capture.
    detach_all_front_cameras,# Cleans up all attached cameras at simulation end.
    get_agent_spawn_lane, # Retrieves the lane object where agents should spawn.
)
from av_simulation.vision_language.vlm_engine import VLMEngine # Imports the VLMEngine class: handles communication with Ollama/LLaVA, image processing, and output parsing.

# =============================================================================
# run_simulation
# =============================================================================

def run_simulation(args=None) -> None:
    """Initialise all subsystems and run the main simulation loop."""

    if args is None:
        args = parse_args() # If no args object was passed (default None), parse command-line arguments.
    # Allows the function to be called programmatically with custom args for testing

    # Overrides the module-level constant SIMULATION_STEPS with the CLI-provided value.
    # Allows simulation length to be set without modifying source code.
    cfg.SIMULATION_STEPS = args.steps
    NUM_AGENTS_RUN       = args.num_agents  # Local variable storing the number of agents to spawn; used in agent config generation.
    profiler             = PerformanceProfiler(enabled=args.profile) # Initializes the performance profiler. If --profile flag is set, it collects step timing metrics.

    setup_logger(True) # Configures MetaDrive's logger with default settings. True argument likely enables stdout logging.
    rw_extra_config = RealWorldDataLoader.get_env_config(args)  # Attempts to load real-world data configuration (e.g., Waymo trajectories) if specified via CLI. Returns a dict of environment overrides, or empty dict if no real-world data is requested.

    print("=" * 60)
    print(f"Scenario : {args.env}  ({MAP_PRESETS[args.env]['description']})") # Prints the scenario name (e.g., 'straight', 'intersection') and its human-readable description from MAP_PRESETS.
    print(f"Agents   : {NUM_AGENTS_RUN}") # Prints the number of autonomous agents in the simulation.
    print(f"Obstacles: {args.num_obstacles}") # Prints the number of static obstacles to spawn.
    print(f"Profile  : {args.profile}")# Indicates whether performance profiling is enabled.
    print("=" * 60)

    # ── Subsystem construction ───────────────────────────────────────────
    vlm_engine        = VLMEngine()# Creates the VLM engine instance. Likely initializes connection to Ollama server or loads a local model.
    obstacle_manager  = MultiObstacleManager(num_obstacles=args.num_obstacles, seed=42)# Creates obstacle manager with specified count and fixed random seed (42) for reproducible obstacle placement.

    v2v_bus           = V2VBus() # Creates the V2V communication bus for message passing between agents
    strategy_repo     = StrategyRepository() # Creates the strategy repository (likely SQLite-backed) for storing formation strategies.
    llm_adapter       = LLMStrategyAdapter()# Creates adapter that converts LLM/VLM outputs into structured fleet strategies.

    raft              = SimpleRaft() # Creates the Raft consensus instance for leader election and agent fitness tracking.
    fleet_coordinator = FleetCoordinator(v2v_bus, strategy_repo, llm_adapter) # Creates the fleet coordinator, passing the communication bus, strategy repo, and LLM adapter as dependencies.

    # SharedSimState threads these singletons into every VLAPolicy
    # without requiring module-level globals.
    shared = SharedSimState(
        raft              = raft,
        fleet_coordinator = fleet_coordinator,# For accessing fleet state and directives
        agents_positions  = {},      # Mutable dict: agent_id → [x, y]; updated each step
        vlm_engine        = vlm_engine,
        directives        = {},      # Mutable dict: formation directives for each agent. Updated each step
    )

    print("=" * 60)

    # ── MetaDrive environment config ─────────────────────────────────────
    agent_configs: dict = {} # Initializes an empty dictionary to hold per-agent spawn configurations.

    for i in range(NUM_AGENTS_RUN):
        agent_configs[f"agent{i}"] = {
            "spawn_lane_index": cfg.SPAWN_LANE_INDEX,
            "spawn_longitude":  10.0 + i * cfg.PLATOON_SPACING,# Staggered spawn positions along the lane
            "spawn_lateral":    0.0,# Centered in the lane
        }

    map_str             = MAP_PRESETS[args.env]["map"]# Retrieves the MetaDrive map string (e.g., "SS" for straight, "C" for circular) from the preset
    traffic_density_val = args.traffic_density if args.reactive_traffic else 0.0

    env_config = {
        "use_render":         True,# Enables Panda3D rendering window
        "num_agents":         NUM_AGENTS_RUN,
        "map":                map_str,# Map identifier (e.g., "SS", "C", "X")
        "traffic_mode":       "Trigger",# Traffic mode: "Trigger" likely spawns vehicles on trigger events
        "horizon":            cfg.SIMULATION_STEPS,# Max simulation steps before episode ends
        "agent_configs":      agent_configs,
        "vehicle_config": {
            "lidar": {"num_lasers": 72, "distance": 50, "num_others": 4},# LIDAR sensor configuration
            "show_lidar": False,# Disables LIDAR point cloud visualization for performance
        },
        "accident_prob":      0.0,# No random accidents
        "traffic_density":    traffic_density_val,# Background traffic density (if enabled)
        "crash_vehicle_done": False,# Agent does not terminate on vehicle collision
        "crash_object_done":  False,# Agent does not terminate on object collision
        "out_of_route_done":  False,# Agent does not terminate on out-of-route
        "use_semantic":       False,# Disables semantic segmentation output (saves memory)
        "use_depth":          False, # Disables depth map output (saves memory)
    }
    env_config.update(rw_extra_config)

    env          = EnhancedMultiAgentEnv(env_config) # Creates the enhanced multi-agent environment wrapper around MetaDrive's environment.
    reset_result = env.reset()# Resets the environment to initial state. Returns observation dict with agent_id → observation.
    o = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    # ── Optional top-down camera ─────────────────────────────────────────
    if args.top_down:
        try:
            env.main_camera.camera.setPos(0, 0, 200) # Positions camera high above the scene
            env.main_camera.camera.lookAt(0, 0, 0)# Points camera downward to center
            print("[NEW-1] Top-down camera activated.")
        except Exception as e:
            print(f"[NEW-1] Top-down camera failed (non-fatal): {e}")

    # ── Spawn obstacles ──────────────────────────────────────────────────
    obstacle_lane = get_agent_spawn_lane(env)# Retrieves the lane object where agents spawned; obstacles are placed relative to this lane.
    if obstacle_lane is None:
        raise RuntimeError("Could not find agent spawn lane.") # Ensures obstacle placement has a reference lane; raises error if not found.
    obstacle_manager.spawn_all(env, obstacle_lane)
    ## Spawns all obstacles in the environment, positioned relative to the spawn lane. cfg.OBSTACLE_POSITION is now set by spawn_all via the module reference. The spawn_all method updates the module-level constant OBSTACLE_POSITION (side effect) so other components (like fleet_coordinator) can reference obstacle locations.

    print("=" * 60)
    print("PIPELINE STAGES ACTIVE:")
    print("  1. Semantic Detection — VLM + lidar cross-check  [FIX-E]")
    print("  2. V2V Broadcast + immediate fleet trigger  [STEER-4]")
    print("  3. Local Assessment + cost tracking")
    print("  4. Raft Leader Election  (runs every step)  [FIX-A]")
    print("  5. Strategy Selection (SQLite)")
    print("  6. Waypoint-Adapted Strategy")
    print("  7. MPC — steer preserved in all return paths  [FIX-B/C/D]")
    print("  8. Reformation + ASCII map")
    print("=" * 60)

    # ── Instantiate per-agent policies ───────────────────────────────────
    agent_policies: Dict[str, VLAPolicy] = {} # Dictionary mapping agent_id to its VLAPolicy instance.

    for agent_id, vehicle in env.agent_manager.active_agents.items():
        raft.register_agent(agent_id) # Registers each agent with the Raft consensus instance for leader election
        policy = VLAPolicy(
            vehicle,
            random_seed      = 0, # Fixed seed for reproducibility (deterministic behavior)
            shared_state     = shared, # Shared state dataclass with singletons and mutable data
            env              = env,
            profiler         = profiler,
            obstacle_manager = obstacle_manager,
        )
        agent_policies[agent_id] = policy
        attach_front_camera(env, vehicle) # Attaches a front-facing camera to the vehicle for VLM image capture.

    shared.agents_positions = {
        aid: [v.position[0], v.position[1]]
        for aid, v in env.agent_manager.active_agents.items()  # Initializes the shared positions dict with current agent positions (x, y coordinates)
    }

    for aid in raft.agent_ids:
        raft.update_fitness(aid, 0.5, 0.5, 0.5)# Sets initial fitness values (likely metrics like response_time, accuracy, efficiency) to 0.5 for all agents. These values may be updated during simulation based on performance.
    raft.elect_leader() # Triggers leader election based on fitness values. Leader is used for strategy coordination.
    assign_vlm_agent(agent_policies, shared.agents_positions) # Selects which agent(s) are responsible for issuing VLM queries. Typically assigns to the leader or the agent closest to obstacles.

    shared.directives = fleet_coordinator.get_formation_directives(
        list(raft.agent_ids),
        shared.agents_positions,
        raft.leader_id,
        current_step=0,  # Initial formation directives for all agents at step 0.
    )

    print(
        f"[DIAG] {len(agent_policies)} agents, {cfg.SIMULATION_STEPS} steps, "
        f"map='{map_str}', obstacles={len(obstacle_manager.obstacle_positions)}" # Diagnostic output summarizing simulation configuration.
    )

    active_count      = len(agent_policies)  # Tracks the number of active agents (may decrease if agents are removed).
    pipeline_executed = False # Flag to ensure the full pipeline executes only once when triggered.

    wp_planner        = WaypointPlanner() # Creates waypoint planner instance for ASCII map generation and navigation.

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    for step in range(cfg.SIMULATION_STEPS):
        profiler.start_step() # Records start time for this step's performance measurement.

        active_agents_this_step = [] # List to track which agents are active in this iteration.
        new_positions: Dict[str, list] = {} # Temporary dict to collect updated positions before merging into shared state.

        for aid in list(o.keys()):
            if aid in env.agent_manager.active_agents: # Handle case where an agent appears mid-simulation (e.g., spawned later)
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
                    assign_vlm_agent(agent_policies, shared.agents_positions) # Iterates through observation keys to collect active agents and their positions. Dynamically creates policies for newly detected agents.

        # Update shared mutable position dict in-place
        shared.agents_positions.update(new_positions) # Merges new positions into the shared dict without replacing unchanged entries.

        # ── Fleet state machine tick ─────────────────────────────────────
        trigger_pipeline = fleet_coordinator.update_state(
            shared.agents_positions, cfg.OBSTACLE_POSITION, step
        ) 
        # Updates the fleet coordinator's state machine with current positions and obstacle location. Returns True if the full pipeline should be triggered (e.g., obstacle detected).

        if trigger_pipeline and not pipeline_executed:
            print("\n" + "=" * 60)
            print("EXECUTING FULL PIPELINE  [ALL STAGES]")
            print("=" * 60)
            raft.elect_leader() # Re-elects leader to ensure current fitness values are used.

            assign_vlm_agent(agent_policies, shared.agents_positions)
            if not raft.check_group_stop():
                fleet_coordinator.execute_strategy_pipeline(
                    list(raft.agent_ids), shared.agents_positions, raft
                )
                # If group stop is not active, executes the full strategy pipeline: queries VLM, selects strategy from repo, adapts with LLM, and broadcasts directives.

            pipeline_executed = True  # Prevents re-execution until the next trigger condition resets this flag.
            print("=" * 60 + "\n")

        if fleet_coordinator.obstacle_detected:
            pipeline_executed = True  # If obstacle detection flag is set, marks pipeline as executed (may prevent duplicate triggers or indicate ongoing response).

        if step % 30 == 0:
            assign_vlm_agent(agent_policies, shared.agents_positions) # Periodically reassigns VLM responsibility every 30 steps to account for position changes.

        # ── Formation directives ─────────────────────────────────────────
        if shared.agents_positions:
            leader_vel = None
            if raft.leader_id and raft.leader_id in env.agent_manager.active_agents:
                leader_vel = env.agent_manager.active_agents[raft.leader_id].speed
            # Retrieves current leader's speed for formation adaptation.

            # Update in-place so VLAPolicy instances always see current values
            shared.directives = fleet_coordinator.get_formation_directives(
                list(raft.agent_ids),
                shared.agents_positions,
                raft.leader_id,
                leader_vel,
                current_step=step,
            )
            # Computes formation directives (target positions, speeds) based on current fleet state.
        # Directives are stored in shared state for policies to access during act().

        # ── ASCII mini-map ────────────────────────────────────────────────
        if step % 60 == 0 and shared.agents_positions:
            ascii_map = wp_planner.render_ascii_map(
                shared.agents_positions, obstacle_manager.obstacle_positions
            )
            if ascii_map:
                print(ascii_map) # Every 60 steps, prints an ASCII representation of agent and obstacle positions.
        # Helps visualize fleet formation without full 3D rendering.

        a = {
            k: agent_policies[k].act(o[k], current_step=step)
            for k in o
            if k in agent_policies and k in active_agents_this_step
        }
        # For each agent with an observation and active policy, calls act() to compute control action.
        # Returns action dict (typically throttle, brake, steering) for each agent.

        # ── Step environment ──────────────────────────────────────────────
        step_result = env.step(a) # Applies actions to environment, advances simulation by one timestep.
        if len(step_result) == 5:
            # MetaDrive 5-value return: observation, reward, terminated, truncated, info
            o, r, terminated, truncated, i = step_result
            d = {
                k: terminated.get(k, False) or truncated.get(k, False)
                for k in terminated
            }
        else:
             # Legacy 4-value return: observation, reward, done, info
            o, r, d, i = step_result
            # Handles both return signature formats for MetaDrive compatibility. d is a dict mapping agent_id to done flag (terminated or truncated).

        # ── Collision logging ─────────────────────────────────────────────
        for aid, info in (i.items() if isinstance(i, dict) else []):
            if isinstance(info, dict):
                if info.get('crash_vehicle', False):
                    print(f"[COLLISION] {aid[:8]} hit vehicle!")
                if info.get('crash_object', False):
                    print(f"[COLLISION] {aid[:8]} hit object!") # Iterates through info dict (if present) to log collision events. Prints first 8 characters of agent ID for brevity.


        env.render() # Updates the Panda3D rendering window.
        profiler.end_step() # Records step duration and updates statistics.

        # ── Termination check ─────────────────────────────────────────────
        if d:
            if sum(1 for v in d.values() if v) == active_count and active_count > 0:
                print(f"[DIAG] All agents done at step {step}")
                break
            # If all agents are marked done (terminated/truncated), exits simulation loop.

        # ── Periodic diagnostics ──────────────────────────────────────────
        if step % 60 == 0:
            fleet_cost = obstacle_manager.compute_fleet_cost(shared.agents_positions)
            # Computes a fleet-level cost metric (e.g., distance to obstacles, formation error).
            print(
                f"[DIAG] Step {step}/{cfg.SIMULATION_STEPS} | "
                f"State: {fleet_coordinator.state} | "
                f"Active: {len(active_agents_this_step)} | "
                f"Leader: {raft.leader_id[:8] if raft.leader_id else 'None'} | "
                f"FleetCost: {fleet_cost:.3f}"
            )
            # Every 60 steps, prints diagnostic information:
            # - Current step / total steps
            # - Fleet coordinator state (e.g., 'NORMAL', 'EVASIVE')
            # - Number of active agents
            # - Leader ID (truncated)
            # - Current fleet cost

        time.sleep(0.033) # Sleeps for ~33ms to maintain approximately 30 FPS (1/30 ≈ 0.0333). Prevents simulation from running too fast for real-time observation.

    print(f"[DIAG] Simulation complete — {step + 1} steps") # Prints completion message with actual number of steps executed (step is last index, +1 for count).
    profiler.summary() # Prints performance summary: average step time, min/max, total runtime.
    detach_all_front_cameras() # Cleans up all attached front cameras to free resources.
    env.close() # Closes the MetaDrive environment and Panda3D rendering window.



# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    run_simulation()
