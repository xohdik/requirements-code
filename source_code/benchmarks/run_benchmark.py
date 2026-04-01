"""
benchmarks/run_benchmark.py
============================
Automated benchmark suite — 150 trials (F10).

Grid: 5 obstacle types × 3 placements × 10 seeds

Metrics logged per trial:
  - task_success       : bool   — all agents cleared obstacle without deadlock
  - time_to_first_move : int    — steps until fleet started obstacle response
  - fleet_clearance_time : int  — steps until last agent cleared obstacle
  - graph_diffs        : int    — total GraphDiff broadcasts (comm overhead)
  - llm_tokens         : int    — LLM tokens used (estimated)
  - collisions         : int    — total collision events
  - steps_run          : int    — actual steps executed before termination

Run from source_code/:
    python benchmarks/run_benchmark.py [--trials 150] [--output results/benchmark.csv]

Run a quick smoke test (5 trials):
    python benchmarks/run_benchmark.py --trials 5 --steps 300

Ablation example (disable graph):
    python benchmarks/run_benchmark.py --trials 30 --no_graph
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
import traceback
from typing import Dict, List, Optional

# Ensure source_code/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from av_simulation.config.simulation_config import parse_args

# ── Benchmark configuration ────────────────────────────────────────────────────

OBSTACLE_TYPES = [
    "fallen_tree",
    "debris",
    "road_closure",
    "stationary_vehicle",
    "construction_barrier",
]

# Placement maps to obstacle longitude offset from default (cfg.OBSTACLE_LONGITUDE)
PLACEMENTS = {
    "near": -20.0,    # 20m closer than default spawn
    "mid":   0.0,     # default position
    "far":  +20.0,    # 20m further than default spawn
}

SEEDS = list(range(10))   # seeds 0–9

DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results"
)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _checkpoint_path(output_csv: str) -> str:
    return output_csv.replace(".csv", ".checkpoint.csv")


def _load_completed(ckpt_path: str) -> set:
    """Return set of (obs_type, placement, seed) tuples already completed."""
    completed = set()
    if not os.path.exists(ckpt_path):
        return completed
    with open(ckpt_path, newline="") as f:
        for row in csv.DictReader(f):
            completed.add((row["obstacle_type"], row["placement"], int(row["seed"])))
    return completed


def _append_row(path: str, row: dict, write_header: bool) -> None:
    """Append a single row to CSV, writing header if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Trial runner ───────────────────────────────────────────────────────────────

def run_trial(
    obs_type:    str,
    placement:   str,
    seed:        int,
    steps:       int,
    ablation:    dict,
) -> dict:
    """
    Run one simulation trial and return the metrics dict.

    Modifies cfg.OBSTACLE_LONGITUDE temporarily to simulate placement.
    """
    from av_simulation.config import simulation_config as cfg
    from main import run_simulation

    # Build args namespace for this trial
    sim_args = argparse.Namespace(
        env              = "straight",
        top_down         = False,
        num_agents       = 4,
        reactive_traffic = False,
        traffic_density  = 0.0,
        waymo            = False,
        nuscenes         = False,
        profile          = False,
        steps            = steps,
        num_obstacles    = 1,
        no_render        = True,          # always headless in benchmark
        seed             = seed,
        no_graph         = ablation.get("no_graph",         False),
        no_leader        = ablation.get("no_leader",        False),
        no_strategy_repo = ablation.get("no_strategy_repo", False),
    )

    # Adjust obstacle placement
    original_lon = cfg.OBSTACLE_LONGITUDE
    cfg.OBSTACLE_LONGITUDE = original_lon + PLACEMENTS.get(placement, 0.0)

    # Patch obstacle type into MultiObstacleManager for this trial
    _patch_obstacle_type(obs_type)

    try:
        t0      = time.time()
        metrics = run_simulation(args=sim_args)
        elapsed = time.time() - t0
    except Exception as e:
        print(f"[BENCHMARK] Trial FAILED ({obs_type}/{placement}/seed={seed}): {e}")
        traceback.print_exc()
        metrics = {
            "task_success": False,
            "ttfm":         None,
            "fct":          None,
            "graph_diffs":  0,
            "llm_tokens":   0,
            "collisions":   0,
            "steps_run":    0,
        }
        elapsed = 0.0
    finally:
        # Restore
        cfg.OBSTACLE_LONGITUDE = original_lon
        _restore_obstacle_type()

    result = {
        "obstacle_type":        obs_type,
        "placement":            placement,
        "seed":                 seed,
        "success":              int(metrics.get("task_success", False)),
        "time_to_first_move":   metrics.get("ttfm")    if metrics.get("ttfm")    is not None else -1,
        "fleet_clearance_time": metrics.get("fct")     if metrics.get("fct")     is not None else -1,
        "graph_diffs":          metrics.get("graph_diffs",  0),
        "llm_tokens":           metrics.get("llm_tokens",   0),
        "collisions":           metrics.get("collisions",   0),
        "steps_run":            metrics.get("steps_run",    0),
        "wall_time_s":          round(elapsed, 2),
        "ablation_no_graph":    int(ablation.get("no_graph",         False)),
        "ablation_no_leader":   int(ablation.get("no_leader",        False)),
        "ablation_no_repo":     int(ablation.get("no_strategy_repo", False)),
    }
    return result


# ── Obstacle type patching ─────────────────────────────────────────────────────

_ORIGINAL_OBSTACLE_TYPES: Optional[list] = None

OBSTACLE_TYPE_MAP = {
    "fallen_tree":           {"name": "fallen_tree",    "half": (2.0, 0.5, 0.7), "color": (0.4, 0.2, 0.1, 1.0)},
    "debris":                {"name": "debris",         "half": (1.6, 1.5, 0.5), "color": (1.0, 0.6, 0.0, 1.0)},
    "road_closure":          {"name": "road_closure",   "half": (0.3, 2.0, 1.0), "color": (1.0, 0.0, 0.0, 1.0)},
    "stationary_vehicle":    {"name": "stationary_veh", "half": (2.2, 1.0, 0.8), "color": (0.2, 0.2, 0.8, 1.0)},
    "construction_barrier":  {"name": "construction",   "half": (0.6, 1.6, 0.7), "color": (1.0, 1.0, 0.0, 1.0)},
}


def _patch_obstacle_type(obs_type: str) -> None:
    global _ORIGINAL_OBSTACLE_TYPES
    from av_simulation.utils.sim_context import MultiObstacleManager
    _ORIGINAL_OBSTACLE_TYPES = MultiObstacleManager.OBSTACLE_TYPES[:]
    otype = OBSTACLE_TYPE_MAP.get(obs_type, _ORIGINAL_OBSTACLE_TYPES[0])
    MultiObstacleManager.OBSTACLE_TYPES = [otype]


def _restore_obstacle_type() -> None:
    global _ORIGINAL_OBSTACLE_TYPES
    if _ORIGINAL_OBSTACLE_TYPES is not None:
        from av_simulation.utils.sim_context import MultiObstacleManager
        MultiObstacleManager.OBSTACLE_TYPES = _ORIGINAL_OBSTACLE_TYPES
        _ORIGINAL_OBSTACLE_TYPES = None


# ── Summary stats ──────────────────────────────────────────────────────────────

def _print_summary(rows: List[dict]) -> None:
    if not rows:
        print("[BENCHMARK] No results to summarise.")
        return

    total       = len(rows)
    successes   = sum(r["success"] for r in rows)
    tsr         = successes / total
    ttfm_vals   = [r["time_to_first_move"]   for r in rows if r["time_to_first_move"]   >= 0]
    fct_vals    = [r["fleet_clearance_time"] for r in rows if r["fleet_clearance_time"]  >= 0]
    diff_vals   = [r["graph_diffs"] for r in rows]
    coll_vals   = [r["collisions"]  for r in rows]

    def _mean(lst): return sum(lst) / len(lst) if lst else float("nan")
    def _pct(lst, p):
        if not lst: return float("nan")
        s = sorted(lst); i = int(len(s) * p / 100)
        return s[min(i, len(s)-1)]

    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Trials         : {total}")
    print(f"  Task Success   : {successes}/{total} ({tsr:.1%})")
    print(f"  TTFM (mean)    : {_mean(ttfm_vals):.1f} steps")
    print(f"  FCT  (mean)    : {_mean(fct_vals):.1f}  steps")
    print(f"  FCT  (p95)     : {_pct(fct_vals, 95):.1f}  steps")
    print(f"  Graph diffs/ep : {_mean(diff_vals):.1f}")
    print(f"  Collisions/ep  : {_mean(coll_vals):.2f}")
    print(f"  NF3 target     : >90% TSR — {'MET' if tsr >= 0.9 else 'NOT MET'}")
    print("=" * 60 + "\n")


# ── Main entry point ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="VLA-MAC Benchmark Runner (F10)")
    p.add_argument("--trials",   type=int, default=150,
                   help="Total trials (default 150 = 5 types × 3 placements × 10 seeds)")
    p.add_argument("--steps",    type=int, default=1800,
                   help="Steps per trial")
    p.add_argument("--output",   default=os.path.join(DEFAULT_RESULTS_DIR, "benchmark.csv"),
                   help="Output CSV path")
    p.add_argument("--no_graph",         action="store_true")
    p.add_argument("--no_leader",        action="store_true")
    p.add_argument("--no_strategy_repo", action="store_true")
    args = p.parse_args()

    ablation = {
        "no_graph":         args.no_graph,
        "no_leader":        args.no_leader,
        "no_strategy_repo": args.no_strategy_repo,
    }

    # Build full trial grid (truncated to --trials)
    full_grid = list(itertools.product(OBSTACLE_TYPES, PLACEMENTS.keys(), SEEDS))
    trial_grid = full_grid[:args.trials]

    output_csv = args.output
    ckpt_path  = _checkpoint_path(output_csv)
    completed  = _load_completed(ckpt_path)

    remaining = [(o, p, s) for o, p, s in trial_grid if (o, p, s) not in completed]
    print(f"\n[BENCHMARK] {len(trial_grid)} total trials | "
          f"{len(completed)} already done | "
          f"{len(remaining)} to run")
    print(f"[BENCHMARK] Output: {output_csv}")
    if any(ablation.values()):
        print(f"[BENCHMARK] Ablation: {ablation}")
    print()

    all_rows   = []
    header_written = os.path.exists(output_csv)

    for idx, (obs_type, placement, seed) in enumerate(remaining, 1):
        print(
            f"[BENCHMARK] Trial {idx}/{len(remaining)}: "
            f"{obs_type} / {placement} / seed={seed}"
        )
        row = run_trial(obs_type, placement, seed, args.steps, ablation)
        all_rows.append(row)

        # Write to both checkpoint and final output after each trial
        _append_row(ckpt_path,  row, write_header=not os.path.exists(ckpt_path))
        _append_row(output_csv, row, write_header=not header_written)
        header_written = True

        print(
            f"  success={row['success']} | "
            f"ttfm={row['time_to_first_move']} | "
            f"fct={row['fleet_clearance_time']} | "
            f"diffs={row['graph_diffs']} | "
            f"wall={row['wall_time_s']}s"
        )

    _print_summary(all_rows)
    print(f"[BENCHMARK] Results saved to: {output_csv}")


if __name__ == "__main__":
    main()