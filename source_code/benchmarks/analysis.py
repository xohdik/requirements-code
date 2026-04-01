"""
benchmarks/analysis.py
=======================
Post-benchmark analysis and ablation comparison (F11).

Reads benchmark CSV(s) and produces:
  1. Task Success Rate (TSR) — overall and per obstacle type
  2. Ablation comparison table — full system vs disabled components
  3. Fleet Clearance Time (FCT) distributions
  4. Communication overhead (graph diffs per episode)
  5. NF1 MPC compliance check (from mpc_perf.jsonl if available)

Run from source_code/:
    python benchmarks/analysis.py --csv results/benchmark.csv

Or compare two runs:
    python benchmarks/analysis.py --csv results/full.csv results/no_graph.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── CSV reader ─────────────────────────────────────────────────────────────────

def load_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        print(f"[ANALYSIS] File not found: {path}")
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # Cast numeric fields
    int_fields   = {"seed", "success", "time_to_first_move",
                    "fleet_clearance_time", "graph_diffs", "llm_tokens",
                    "collisions", "steps_run",
                    "ablation_no_graph", "ablation_no_leader", "ablation_no_repo"}
    float_fields = {"wall_time_s"}
    for row in rows:
        for f in int_fields:
            if f in row and row[f] not in ("", None):
                try:
                    row[f] = int(row[f])
                except ValueError:
                    row[f] = 0
        for f in float_fields:
            if f in row and row[f] not in ("", None):
                try:
                    row[f] = float(row[f])
                except ValueError:
                    row[f] = 0.0
    return rows


# ── Stats helpers ──────────────────────────────────────────────────────────────

def _mean(lst):
    return sum(lst) / len(lst) if lst else float("nan")

def _pct(lst, p):
    if not lst:
        return float("nan")
    s = sorted(lst)
    i = int(len(s) * p / 100)
    return s[min(i, len(s) - 1)]

def _fmt(v, decimals=1):
    if isinstance(v, float) and v != v:  # nan
        return "  N/A"
    return f"{v:.{decimals}f}"


# ── Per-run statistics ─────────────────────────────────────────────────────────

def compute_stats(rows: List[dict], label: str = "") -> dict:
    if not rows:
        return {}

    total     = len(rows)
    successes = sum(r.get("success", 0) for r in rows)
    tsr       = successes / total

    ttfm_vals = [r["time_to_first_move"]   for r in rows if r.get("time_to_first_move",   -1) >= 0]
    fct_vals  = [r["fleet_clearance_time"] for r in rows if r.get("fleet_clearance_time", -1) >= 0]
    diff_vals = [r.get("graph_diffs", 0)   for r in rows]
    coll_vals = [r.get("collisions",  0)   for r in rows]
    wall_vals = [r.get("wall_time_s", 0.0) for r in rows]

    # Per obstacle-type TSR
    by_type: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        by_type[r.get("obstacle_type", "unknown")].append(r.get("success", 0))
    type_tsr = {t: sum(v)/len(v) for t, v in by_type.items()}

    # Per placement TSR
    by_place: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        by_place[r.get("placement", "?")].append(r.get("success", 0))
    place_tsr = {p: sum(v)/len(v) for p, v in by_place.items()}

    return {
        "label":       label,
        "total":       total,
        "successes":   successes,
        "tsr":         tsr,
        "ttfm_mean":   _mean(ttfm_vals),
        "ttfm_p95":    _pct(ttfm_vals, 95),
        "fct_mean":    _mean(fct_vals),
        "fct_p95":     _pct(fct_vals, 95),
        "diffs_mean":  _mean(diff_vals),
        "coll_mean":   _mean(coll_vals),
        "wall_mean":   _mean(wall_vals),
        "type_tsr":    type_tsr,
        "place_tsr":   place_tsr,
        "nf3_ok":      tsr >= 0.90,
    }


# ── Ablation split ─────────────────────────────────────────────────────────────

def split_ablation(rows: List[dict]) -> Dict[str, List[dict]]:
    """
    Split rows into ablation groups based on ablation_* columns.
    Returns dict: condition_label -> rows
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        ng = r.get("ablation_no_graph",  0)
        nl = r.get("ablation_no_leader", 0)
        nr = r.get("ablation_no_repo",   0)

        if not ng and not nl and not nr:
            label = "Full system"
        elif ng and not nl and not nr:
            label = "No SemanticGraph"
        elif not ng and nl and not nr:
            label = "No leader election"
        elif not ng and not nl and nr:
            label = "No strategy repo"
        elif ng and nl:
            label = "Baseline (no graph + no leader)"
        else:
            label = f"Ablation(g={ng},l={nl},r={nr})"
        groups[label].append(r)
    return dict(groups)


# ── Print functions ────────────────────────────────────────────────────────────

def print_summary(stats: dict) -> None:
    label = stats.get("label", "")
    print(f"\n{'='*60}")
    print(f"  {label or 'Summary'}")
    print(f"{'='*60}")
    print(f"  Trials          : {stats['total']}")
    print(f"  Task Success    : {stats['successes']}/{stats['total']} "
          f"({stats['tsr']:.1%})   "
          f"NF3={'MET ✓' if stats['nf3_ok'] else 'NOT MET ✗'}")
    print(f"  TTFM mean       : {_fmt(stats['ttfm_mean'])} steps")
    print(f"  TTFM p95        : {_fmt(stats['ttfm_p95'])} steps")
    print(f"  FCT  mean       : {_fmt(stats['fct_mean'])} steps")
    print(f"  FCT  p95        : {_fmt(stats['fct_p95'])} steps")
    print(f"  Graph diffs/ep  : {_fmt(stats['diffs_mean'])}")
    print(f"  Collisions/ep   : {_fmt(stats['coll_mean'], 2)}")
    print(f"  Wall time/ep    : {_fmt(stats['wall_mean'])}s")

    print(f"\n  TSR by obstacle type:")
    for ot, tsr in sorted(stats["type_tsr"].items()):
        bar = "█" * int(tsr * 20) + "░" * (20 - int(tsr * 20))
        print(f"    {ot:<26} {bar} {tsr:.1%}")

    print(f"\n  TSR by placement:")
    for pl, tsr in sorted(stats["place_tsr"].items()):
        bar = "█" * int(tsr * 20) + "░" * (20 - int(tsr * 20))
        print(f"    {pl:<10} {bar} {tsr:.1%}")


def print_ablation_table(all_stats: List[dict]) -> None:
    if len(all_stats) < 2:
        return

    print(f"\n{'='*72}")
    print(f"  ABLATION COMPARISON")
    print(f"{'='*72}")
    header = f"  {'Condition':<30} {'TSR':>6} {'FCT':>7} {'Diffs':>7} {'Coll':>6}"
    print(header)
    print(f"  {'-'*68}")

    # Sort: full system first, then by TSR descending
    def sort_key(s):
        return (0 if s["label"] == "Full system" else 1, -s["tsr"])

    for s in sorted(all_stats, key=sort_key):
        nf3 = "✓" if s["nf3_ok"] else "✗"
        print(
            f"  {s['label']:<30} "
            f"{s['tsr']:>5.1%}{nf3} "
            f"{_fmt(s['fct_mean']):>7} "
            f"{_fmt(s['diffs_mean']):>7} "
            f"{_fmt(s['coll_mean'], 2):>6}"
        )

    # Compute delta vs full system
    full = next((s for s in all_stats if s["label"] == "Full system"), None)
    if full:
        print(f"\n  Delta vs Full System:")
        for s in all_stats:
            if s["label"] == "Full system":
                continue
            delta_tsr = s["tsr"] - full["tsr"]
            sign      = "+" if delta_tsr >= 0 else ""
            print(f"    {s['label']:<30} TSR {sign}{delta_tsr:.1%}")


# ── NF1 MPC check ─────────────────────────────────────────────────────────────

def check_nf1(log_dir: str) -> None:
    path = os.path.join(log_dir, "mpc_perf.jsonl")
    if not os.path.exists(path):
        print("\n[NF1] mpc_perf.jsonl not found — run simulation with logging enabled")
        return

    solve_times = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if "solve_ms" in rec:
                    solve_times.append(rec["solve_ms"])
            except json.JSONDecodeError:
                continue

    if not solve_times:
        print("\n[NF1] No MPC solve records found.")
        return

    import statistics
    mean_ms = statistics.mean(solve_times)
    max_ms  = max(solve_times)
    p95_ms  = _pct(solve_times, 95)
    p99_ms  = _pct(solve_times, 99)

    print(f"\n{'='*60}")
    print(f"  NF1 — MPC REAL-TIME PERFORMANCE ({len(solve_times)} solves)")
    print(f"{'='*60}")
    print(f"  Mean solve time : {mean_ms:.2f} ms")
    print(f"  Max  solve time : {max_ms:.2f} ms  "
          f"({'< 33ms ✓' if max_ms < 33 else '≥ 33ms ✗ DEADLINE MISSED'})")
    print(f"  p95  solve time : {p95_ms:.2f} ms  "
          f"({'< 20ms ✓' if p95_ms < 20 else '≥ 20ms target missed'})")
    print(f"  p99  solve time : {p99_ms:.2f} ms")
    print(f"  NF1 Hard deadline (<33ms): {'MET ✓' if max_ms < 33 else 'NOT MET ✗'}")
    print(f"  NF1 Target       (<20ms): {'MET ✓' if p95_ms < 20 else 'NOT MET ✗'}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="VLA-MAC Benchmark Analysis (F11)")
    p.add_argument("--csv",    nargs="+", required=True,
                   help="One or more benchmark CSV files")
    p.add_argument("--log_dir", default=os.path.join(
        os.path.dirname(__file__), "..", "logs"),
                   help="Directory containing log files (for NF1 check)")
    p.add_argument("--no_nf1", action="store_true",
                   help="Skip NF1 MPC performance check")
    args = p.parse_args()

    all_rows = []
    for path in args.csv:
        rows = load_csv(path)
        label = os.path.splitext(os.path.basename(path))[0]
        for r in rows:
            r.setdefault("_source", label)
        all_rows.extend(rows)
        print(f"[ANALYSIS] Loaded {len(rows)} rows from {path}")

    if not all_rows:
        print("[ANALYSIS] No data to analyse.")
        return

    # ── Overall summary ────────────────────────────────────────────────────────
    stats = compute_stats(all_rows, label="Overall Results")
    print_summary(stats)

    # ── Ablation breakdown (if present in single file) ─────────────────────────
    groups = split_ablation(all_rows)
    if len(groups) > 1:
        all_group_stats = [
            compute_stats(rows, label=label)
            for label, rows in groups.items()
        ]
        print_ablation_table(all_group_stats)

    # ── Multi-file comparison ──────────────────────────────────────────────────
    if len(args.csv) > 1:
        file_stats = []
        for path in args.csv:
            rows = load_csv(path)
            if rows:
                label = os.path.splitext(os.path.basename(path))[0]
                file_stats.append(compute_stats(rows, label=label))
        if file_stats:
            print_ablation_table(file_stats)

    # ── NF1 check ─────────────────────────────────────────────────────────────
    if not args.no_nf1:
        check_nf1(args.log_dir)

    print()


if __name__ == "__main__":
    main()