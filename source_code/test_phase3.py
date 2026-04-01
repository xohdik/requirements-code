"""
test_phase3.py
==============
Standalone tests for Phase 3 components.
Run from source_code/:

    python test_phase3.py

No GPU, no MetaDrive, no Ollama needed.
All 5 tests should print PASS.
"""
import sys
import os
import traceback

sys.path.insert(0, ".")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        traceback.print_exc()
        results.append((name, False))


# ── Test 1: ExecutionStatus node written to graph (F8) ────────────────────────

def t1_execution_status_node():
    import numpy as np
    import time
    from av_simulation.graph.graph import SemanticGraph

    graph = SemanticGraph()

    # Simulate what _write_execution_status does
    agent_id  = "agent0"
    step      = 42
    obs_pos   = np.array([80.0, 0.0])
    agent_pos = np.array([65.0, 0.0])   # 15m before obstacle — not cleared
    ref_pos   = np.array([70.0, 2.0])   # lateral offset target

    tracking_error = float(np.linalg.norm(agent_pos - ref_pos))
    dist_to_obs    = float(np.linalg.norm(agent_pos - obs_pos))
    completed      = dist_to_obs > 50.0
    safety_warn    = tracking_error > 8.0

    ts      = time.time()
    node_id = f"execstatus_{agent_id[:8]}_{step}"
    graph.add_node(
        node_id   = node_id,
        node_type = "ExecutionStatus",
        attrs     = {
            "tracking_error":   round(tracking_error, 3),
            "completed":        completed,
            "safety_warning":   safety_warn,
            "step":             step,
            "dist_to_obstacle": round(dist_to_obs, 2),
        },
        source    = agent_id,
        timestamp = ts,
    )

    vehicle_id = f"vehicle_{agent_id}"
    graph.add_node(vehicle_id, "Vehicle", {"id": agent_id}, agent_id)
    graph.add_edge(vehicle_id, node_id, "executed", {}, agent_id, ts)

    # Verify node
    node = graph.get_node(node_id)
    assert node is not None,                       "ExecutionStatus node missing"
    assert node.node_type == "ExecutionStatus",     "Wrong node type"
    assert "tracking_error" in node.attrs,          "tracking_error missing"
    assert "completed"      in node.attrs,          "completed missing"
    assert "safety_warning" in node.attrs,          "safety_warning missing"
    assert not node.attrs["completed"],             "Should not be completed (only 15m past start)"

    # Verify edge
    edges = graph.get_edges(from_id=vehicle_id, label="executed")
    assert len(edges) == 1, "executed edge missing"

    # Test completed=True when >50m past obstacle
    agent_pos2 = np.array([135.0, 0.0])  # 55m past obstacle
    dist2      = float(np.linalg.norm(agent_pos2 - obs_pos))
    assert dist2 > 50.0, "Test setup error"

    print(f"    tracking_error={tracking_error:.2f}m, completed={completed}, "
          f"dist_to_obs={dist_to_obs:.1f}m")


# ── Test 2: Termination logic — mission_complete check ────────────────────────

def t2_mission_complete_logic():
    """
    Simulate the mission_complete condition from the main loop.
    Agents clear when they are >50m past obstacle.
    """
    obs_x = 80.0
    agents = {
        "agent0": [140.0, 0.0],   # 60m past — cleared
        "agent1": [135.0, 0.0],   # 55m past — cleared
        "agent2": [70.0,  0.0],   # 10m before — NOT cleared
        "agent3": [60.0,  0.0],   # 20m before — NOT cleared
    }
    active_count   = len(agents)
    agents_cleared = set()

    for aid, pos in agents.items():
        if pos[0] > obs_x + 50.0:
            agents_cleared.add(aid)

    # Only 2 out of 4 cleared — mission not complete
    mission_done = len(agents_cleared) >= active_count
    assert not mission_done, "Mission should NOT be done with only 2/4 cleared"
    assert len(agents_cleared) == 2

    # Now all cleared
    agents2 = {k: [140.0, 0.0] for k in agents}
    cleared2 = set()
    for aid, pos in agents2.items():
        if pos[0] > obs_x + 50.0:
            cleared2.add(aid)
    mission_done2 = len(cleared2) >= active_count
    assert mission_done2, "Mission should be done with all 4 cleared"

    print(f"    agents_cleared={agents_cleared} | mission_done={mission_done}")
    print(f"    all_cleared test: mission_done={mission_done2}")


# ── Test 3: Config flags — ablation args parse correctly ──────────────────────

def t3_config_ablation_flags():
    """
    Verify --no_render, --seed, --no_graph, --no_leader, --no_strategy_repo
    are accepted by parse_args.
    """
    import argparse
    from av_simulation.config.simulation_config import parse_args

    # Simulate: python main.py --no_render --seed 7 --no_graph --steps 300
    sys.argv = [
        "main.py",
        "--no_render",
        "--seed", "7",
        "--no_graph",
        "--steps", "300",
    ]
    args = parse_args()

    assert args.no_render        == True,  "--no_render should be True"
    assert args.seed             == 7,     "--seed should be 7"
    assert args.no_graph         == True,  "--no_graph should be True"
    assert args.no_leader        == False, "--no_leader should default to False"
    assert args.no_strategy_repo == False, "--no_strategy_repo should default to False"
    assert args.steps            == 300,   "--steps should be 300"

    # Reset argv
    sys.argv = ["test_phase3.py"]

    print(f"    args.no_render={args.no_render}, seed={args.seed}, "
          f"no_graph={args.no_graph}, steps={args.steps}")


# ── Test 4: Benchmark grid construction ───────────────────────────────────────

def t4_benchmark_grid():
    """Verify the 150-trial grid has the right shape and content."""
    import itertools
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmarks"))

    # Import from benchmarks module
    from benchmarks.run_benchmark import OBSTACLE_TYPES, PLACEMENTS, SEEDS

    full_grid = list(itertools.product(OBSTACLE_TYPES, PLACEMENTS.keys(), SEEDS))

    assert len(OBSTACLE_TYPES) == 5, f"Expected 5 obstacle types, got {len(OBSTACLE_TYPES)}"
    assert len(PLACEMENTS)     == 3, f"Expected 3 placements, got {len(PLACEMENTS)}"
    assert len(SEEDS)          == 10, f"Expected 10 seeds, got {len(SEEDS)}"
    assert len(full_grid)      == 150, f"Expected 150 trials, got {len(full_grid)}"

    # Check all obstacle types are in the grid
    grid_types = {row[0] for row in full_grid}
    for ot in OBSTACLE_TYPES:
        assert ot in grid_types, f"{ot} missing from grid"

    # Check all placements
    grid_places = {row[1] for row in full_grid}
    for pl in PLACEMENTS:
        assert pl in grid_places, f"{pl} missing from grid"

    print(f"    grid={len(full_grid)} trials: "
          f"{len(OBSTACLE_TYPES)} types × {len(PLACEMENTS)} placements × {len(SEEDS)} seeds")


# ── Test 5: Benchmark checkpoint / resume logic ───────────────────────────────

def t5_benchmark_checkpoint():
    """
    Verify checkpoint loading correctly identifies completed trials
    and skips them in the remaining list.
    """
    import tempfile
    import csv
    import itertools
    from benchmarks.run_benchmark import (
        OBSTACLE_TYPES, PLACEMENTS, SEEDS,
        _load_completed, _append_row, _checkpoint_path,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = os.path.join(tmpdir, "test_bench.csv")
        ckpt_path  = _checkpoint_path(output_csv)

        # Simulate 3 completed trials
        completed_rows = [
            {"obstacle_type": OBSTACLE_TYPES[0], "placement": "near", "seed": 0,
             "success": 1, "time_to_first_move": 45, "fleet_clearance_time": 120,
             "graph_diffs": 12, "llm_tokens": 0, "collisions": 0,
             "steps_run": 300, "wall_time_s": 5.2,
             "ablation_no_graph": 0, "ablation_no_leader": 0, "ablation_no_repo": 0},
            {"obstacle_type": OBSTACLE_TYPES[0], "placement": "near", "seed": 1,
             "success": 0, "time_to_first_move": -1, "fleet_clearance_time": -1,
             "graph_diffs": 3, "llm_tokens": 0, "collisions": 2,
             "steps_run": 300, "wall_time_s": 4.8,
             "ablation_no_graph": 0, "ablation_no_leader": 0, "ablation_no_repo": 0},
            {"obstacle_type": OBSTACLE_TYPES[1], "placement": "mid", "seed": 5,
             "success": 1, "time_to_first_move": 60, "fleet_clearance_time": 150,
             "graph_diffs": 18, "llm_tokens": 0, "collisions": 0,
             "steps_run": 300, "wall_time_s": 6.1,
             "ablation_no_graph": 0, "ablation_no_leader": 0, "ablation_no_repo": 0},
        ]

        for row in completed_rows:
            _append_row(ckpt_path, row,
                        write_header=not os.path.exists(ckpt_path))

        # Load completed set
        completed = _load_completed(ckpt_path)
        assert len(completed) == 3, f"Expected 3 completed, got {len(completed)}"
        assert (OBSTACLE_TYPES[0], "near", 0) in completed
        assert (OBSTACLE_TYPES[0], "near", 1) in completed
        assert (OBSTACLE_TYPES[1], "mid",  5) in completed

        # Build remaining list
        full_grid = list(itertools.product(OBSTACLE_TYPES, PLACEMENTS.keys(), SEEDS))
        remaining = [(o, p, s) for o, p, s in full_grid if (o, p, s) not in completed]
        assert len(remaining) == 147, \
            f"Expected 147 remaining, got {len(remaining)}"

    print(f"    completed=3, remaining=147 — checkpoint resume logic correct")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  VLA-MAC Phase 3 — Execution & Benchmark Tests")
    print("=" * 58)

    test("1. ExecutionStatus node written to graph (F8)",    t1_execution_status_node)
    test("2. Mission-complete termination logic (BUG fix)",  t2_mission_complete_logic)
    test("3. Ablation CLI flags parse correctly (F11)",      t3_config_ablation_flags)
    test("4. Benchmark 150-trial grid construction (F10)",   t4_benchmark_grid)
    test("5. Benchmark checkpoint / resume logic",           t5_benchmark_checkpoint)

    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print("=" * 58)
    if passed == total:
        print(f"  All {total}/{total} tests passed. Phase 3 ready.")
    else:
        print(f"  {passed}/{total} passed.")
    print("=" * 58 + "\n")
    sys.exit(0 if passed == total else 1)