"""
simulation_config.py
====================
All simulation-wide constants and the CLI argument parser.

OBSTACLE_POSITION is the one constant that is mutated at runtime: it is set
by MultiObstacleManager.spawn_all() once the physics world is ready.
Other modules should import this *module* (not the symbol) so they always see
the live value:

    from av_simulation.config import simulation_config as cfg
    ...
    if cfg.OBSTACLE_POSITION is not None:
        ...
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Map presets
# ---------------------------------------------------------------------------

MAP_PRESETS: Dict[str, dict] = {
    "straight":     {"map": "SSSSSS",  "description": "6-segment straight highway"},
    "roundabout":   {"map": "O",        "description": "Single roundabout"},
    "intersection": {"map": "X",        "description": "4-way intersection"},
    "toll":         {"map": "SSTSS",    "description": "Straight with a tollgate section"},
    "mixed":        {"map": "SCSXS",    "description": "Curve + intersection mix"},
}
 # "SSSSSS": MetaDrive shorthand for 6 consecutive straight road segments.
    # Used for testing platoon formation and high-speed following.

   # "roundabout":   {"map": "O",        "description": "Single roundabout"},
    # "O": MetaDrive shorthand for a roundabout intersection.

# ---------------------------------------------------------------------------
# Simulation timing & agent counts
# ---------------------------------------------------------------------------

SIMULATION_STEPS: int  = 1800
NUM_AGENTS:        int  = 4

# ---------------------------------------------------------------------------
# Obstacle / platoon geometry (metres)
# ---------------------------------------------------------------------------

OBSTACLE_LONGITUDE: float = 80.0 # Longitudinal position where static obstacles spawn, measured from agent spawn point. 80 metres ahead provides enough reaction distance for the fleet

PLATOON_SPACING:    float = 10.0 # Initial spacing between consecutive agents in the platoon (bumper-to-bumper distance). 10 metres ≈ 2-3 car lengths at typical highway speeds.

SAFE_DISTANCE:      float = 10.0 # Minimum safe following distance (m). Used in MPC cost functions and collision avoidance. If distance to lead vehicle < SAFE_DISTANCE, penalty increases.
LEADER_SPEED:       float = 15.0

# ---------------------------------------------------------------------------
# VLM cadence
# ---------------------------------------------------------------------------

VLM_INFERENCE_INTERVAL: int  = 15
VLM_AGENT_ONLY_LEADER:  bool = True

# ---------------------------------------------------------------------------
# Lane / obstacle spawn
# ---------------------------------------------------------------------------

SPAWN_LANE_INDEX = (">>", ">>>", 0)
# MetaDrive-specific lane identifier format. Structure is typically:
#   - First element: road identifier (e.g., ">>" for straight road)
#   - Second element: lane identifier (e.g., ">>>" for specific lane)
#   - Third element: lane index (0 for rightmost, increments left)
# This tuple determines which lane agents initially spawn on.

# Mutable at runtime — set by MultiObstacleManager.spawn_all()
OBSTACLE_POSITION: Optional[list] = None
# Holds the [x, y] coordinates of the static obstacle after spawning. Initialized to None; set to actual position when MultiObstacleManager.spawn_all() executes

# ---------------------------------------------------------------------------
# Front camera
# ---------------------------------------------------------------------------

FRONT_CAM_W:      int   = 640
FRONT_CAM_H:      int   = 480
# Camera image height in pixels. 640x480 (4:3 aspect ratio) is common for vision models.

FRONT_CAM_FOV:    int   = 60 # Field of view in degrees. 60° provides a reasonable balance between peripheral awareness and central detail

FRONT_CAM_OFFSET: tuple = (1.0, 0.0, 1.8)
# Camera attachment offset relative to vehicle origin:
#   - x: 1.0 metres forward (front bumper/windshield area)
#   - y: 0.0 metres lateral (centered)
#   - z: 1.8 metres height (approximate eye level for a sedan/SUV)

# ---------------------------------------------------------------------------
# Vehicle kinematics
# ---------------------------------------------------------------------------

WHEELBASE:       float = 2.7 # Distance between front and rear axles (metres). Typical for mid-size sedan. Used in kinematic bicycle model for trajectory prediction and MPC.

MAX_STEER_ANGLE: float = 0.6 # Maximum steering angle in radians. 0.6 rad ≈ 34.4 degrees. Upper bound for MPC control output; prevents unrealistic steering inputs.
MIN_STEER_SPEED: float = 3.0
# Minimum vehicle speed (m/s) below which steering angle is limited.
# At very low speeds (≈10.8 km/h), steering is constrained to prevent unrealistic maneuvers.


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the VLA-MAC simulation."""
    p = argparse.ArgumentParser(description="VLA-MAC Fleet Coordination Simulation")
    p.add_argument("--env",              default="straight",
                   choices=list(MAP_PRESETS.keys()),
                   help="Road layout preset")  # --env: selects scenario from MAP_PRESETS keys (straight, roundabout, intersection, toll, mixed)
    
    p.add_argument("--top_down",         action="store_true",
                   help="Enable top-down camera view")
    
    p.add_argument("--num_agents",       type=int,   default=4,
                   help="Number of AV agents in the fleet") # --num_agents: overrides NUM_AGENTS constant with custom value.
    
    p.add_argument("--reactive_traffic", action="store_true",
                   help="Enable reactive NPC traffic") # --reactive_traffic: flag to enable background NPC vehicles.
    # NPCs react to AVs, creating more challenging scenarios.

    p.add_argument("--traffic_density",  type=float, default=0.15,
                   help="NPC traffic density (0.0–1.0)") # controls number of NPC vehicles when reactive_traffic enabled. 0.15 = 15% density (sparse traffic).
    
    p.add_argument("--waymo",            action="store_true",
                   help="Use Waymo real-world dataset (requires metadrive ScenarioEnv)")
    p.add_argument("--nuscenes",         action="store_true",
                   help="Use nuScenes dataset stub")
     # Replaces procedural map generation with real-world datasets.

    p.add_argument("--profile",          action="store_true",
                   help="Enable wall-time profiling")
    p.add_argument("--steps",            type=int,   default=1800,
                   help="Maximum simulation steps") # --steps: overrides SIMULATION_STEPS constant.
    p.add_argument("--num_obstacles",    type=int,   default=1,
                   help="Number of static obstacles to spawn")  # --num_obstacles: number of static obstacles (e.g., construction barrels, debris).
    return p.parse_args()
