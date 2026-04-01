# Re-export every constant that is set once at startup and never mutated.
#
# *** OBSTACLE_POSITION is intentionally NOT re-exported here. ***
# It starts as None and is mutated at runtime by MultiObstacleManager.spawn_all().
# Importing it as a value would freeze it to None at import time.
# Always access it through the module reference:
#
#     from av_simulation.config import simulation_config as cfg
#     ...
#     if cfg.OBSTACLE_POSITION is not None: ...

from av_simulation.config.simulation_config import (
    MAP_PRESETS,
    SIMULATION_STEPS,
    OBSTACLE_LONGITUDE,
    PLATOON_SPACING,
    SAFE_DISTANCE,
    NUM_AGENTS,
    LEADER_SPEED,
    VLM_INFERENCE_INTERVAL,
    VLM_AGENT_ONLY_LEADER,
    SPAWN_LANE_INDEX,
    FRONT_CAM_W,
    FRONT_CAM_H,
    FRONT_CAM_FOV,
    FRONT_CAM_OFFSET,
    WHEELBASE,
    MAX_STEER_ANGLE,
    MIN_STEER_SPEED,
    parse_args,
)

# Expose the module itself so callers can reach the live OBSTACLE_POSITION
from av_simulation.config import simulation_config  # noqa: F401

__all__ = [
    # module reference — use for OBSTACLE_POSITION
    "simulation_config",
    # immutable constants
    "MAP_PRESETS",
    "SIMULATION_STEPS",
    "OBSTACLE_LONGITUDE",
    "PLATOON_SPACING",
    "SAFE_DISTANCE",
    "NUM_AGENTS",
    "LEADER_SPEED",
    "VLM_INFERENCE_INTERVAL",
    "VLM_AGENT_ONLY_LEADER",
    "SPAWN_LANE_INDEX",
    "FRONT_CAM_W",
    "FRONT_CAM_H",
    "FRONT_CAM_FOV",
    "FRONT_CAM_OFFSET",
    "WHEELBASE",
    "MAX_STEER_ANGLE",
    "MIN_STEER_SPEED",
    "parse_args",
]
