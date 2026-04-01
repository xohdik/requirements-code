# VLA-MAC Fleet Coordination Simulation

Multi-agent autonomous vehicle simulation with MPC control, V2V coordination,
Raft leader election, and LLaVA vision-language obstacle detection.

---

## Directory structure

```
simulation_modules/
├── setup.py                              # pip-installable package
├── main.py                               # Entry point  →  python main.py
└── av_simulation/
    ├── __init__.py                       # SharedSimState dataclass
    ├── config/
    │   ├── __init__.py                   # Re-exports all immutable constants
    │   └── simulation_config.py          # Constants & CLI parser
    ├── coordination/
    │   ├── __init__.py
    │   └── fleet_coordinator.py          # V2VBus, SimpleRaft, FleetCoordinator
    ├── control/
    │   ├── __init__.py
    │   └── hierarchical_mpc.py           # setup_mpc(), VLAPolicy
    ├── utils/
    │   ├── __init__.py
    │   └── sim_context.py                # Profiler, EnhancedEnv, cameras,
    │                                     # MultiObstacleManager, WaypointPlanner
    └── vision_language/
        ├── __init__.py
        └── vlm_engine.py                 # VLMEngine (Ollama/LLaVA), parse_vlm_output
```

---

## Prerequisites

| Dependency | Install |
|---|---|
| Python ≥ 3.9 | — |
| MetaDrive | `pip install metadrive-simulator` |
| CasADi | `pip install casadi` |
| do-mpc | `pip install do-mpc` |
| Panda3D | `pip install panda3d` |
| Pillow | `pip install Pillow` |
| NumPy | `pip install numpy` |
| Requests | `pip install requests` |
| Ollama + LLaVA | see below |

### Install Ollama and pull LLaVA

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama serve            # keep running in a separate terminal
ollama pull llava       # ~4 GB download
```

---

## Installation

```bash
cd simulation_modules
pip install -e .        # installs av_simulation as an editable package
```

> **Why `-e`?**  The editable install means Python resolves `av_simulation.*`
> imports from the local source tree, so edits to any module take effect
> immediately without reinstalling.

---

## Running

```bash
# Make sure you are inside simulation_modules/
cd simulation_modules

# Ollama must be running in another terminal:
#   ollama serve

# Basic run (4 agents, straight highway, 1 obstacle)
python main.py

# All CLI options
python main.py \
    --env         straight      \   # straight | roundabout | intersection | toll | mixed
    --num_agents  4             \   # number of AV agents
    --steps       1800          \   # max simulation steps
    --num_obstacles 1           \   # number of road obstacles
    --traffic_density 0.15      \   # NPC density (0.0–1.0)
    --reactive_traffic          \   # enable NPC traffic
    --top_down                  \   # bird's-eye camera
    --profile                   \   # print step/VLM/MPC timing summary
    --waymo                     \   # use Waymo dataset (requires ScenarioEnv)
    --nuscenes                      # nuScenes stub
```

---

## Common errors and fixes

### `RuntimeError: Cannot reach Ollama at http://localhost:11434`
Ollama is not running.  Start it with:
```bash
ollama serve
```

### `RuntimeError: Model 'llava' not found`
The model has not been pulled yet:
```bash
ollama pull llava
```

### `ModuleNotFoundError: No module named 'av_simulation'`
The package has not been installed into the current Python environment:
```bash
cd simulation_modules
pip install -e .
```

### `ModuleNotFoundError: No module named 'metadrive'`
```bash
pip install metadrive-simulator
```

### `ModuleNotFoundError: No module named 'casadi'` / `do_mpc`
```bash
pip install casadi do-mpc
```

### MPC solver warnings (`Infeasible problem` / `NaN in solution`)
These are non-fatal. VLAPolicy catches solver errors and falls back to a
proportional controller automatically ([STEER-8]).  They typically appear
during sharp merge manoeuvres and resolve on the next step.

---

## Key design note — `OBSTACLE_POSITION`

`OBSTACLE_POSITION` is `None` at import time and is set once at runtime by
`MultiObstacleManager.spawn_all()`.  Always access it through the module
reference, **not** as a re-imported constant:

```python
# ✓ correct — sees the live value
from av_simulation.config import simulation_config as cfg
if cfg.OBSTACLE_POSITION is not None:
    ...

# ✗ wrong — captures None at import time, never updates
from av_simulation.config import OBSTACLE_POSITION
```

---

## Modifying the simulation

| Goal | File to edit |
|---|---|
| Change map, speeds, camera FOV | `config/simulation_config.py` |
| Swap LLM model or Ollama host | `vision_language/vlm_engine.py` (top constants) |
| Add a new bypass strategy | `coordination/fleet_coordinator.py` → `StrategyRepository._init_db` |
| Tune MPC horizon / weights | `control/hierarchical_mpc.py` → `setup_mpc` |
| Add a new obstacle type | `utils/sim_context.py` → `MultiObstacleManager.OBSTACLE_TYPES` |
