"""
test_phase4.py
==============
Standalone tests for Phase 4 components.
Run from source_code/:

    python test_phase4.py

No GPU, no MetaDrive, no Ollama needed.
OSQP must be installed: pip install osqp
All 5 tests should print PASS.
"""
import sys
import os
import traceback
import importlib.util as _ilu

# Load logger directly — bypasses utils/__init__.py which pulls metadrive
def _load_logger():
    spec = _ilu.spec_from_file_location(
        "av_simulation.utils.logger",
        os.path.join(os.path.dirname(__file__), "av_simulation", "utils", "logger.py"),
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

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


# ── Test 1: OSQP MPC solve returns valid control output ───────────────────────

def t1_osqp_mpc_solve():
    import numpy as np
    from av_simulation.execution.mpc import OSQPMPCController, _OSQP_AVAILABLE

    ctrl = OSQPMPCController(obs_pos=np.array([80.0, 0.0]), safe_dist=5.0)

    # State: at x=50, y=0, vx=15, vy=0  (heading straight)
    x0      = np.array([50.0, 0.0, 15.0, 0.0])
    ref_pos = np.array([80.0, 4.0])   # bypass target: lateral offset 4m
    ref_vel = np.array([15.0, 0.5])

    ax, ay_lat = ctrl.solve(x0, ref_pos, ref_vel)

    # Control must be within bounds
    from av_simulation.execution.mpc import U_MIN, U_MAX
    assert U_MIN[0] <= ax     <= U_MAX[0], f"ax={ax} out of bounds"
    assert U_MIN[1] <= ay_lat <= U_MAX[1], f"ay_lat={ay_lat} out of bounds"

    # Should prefer positive ay_lat (steering left toward ref_pos y=4.0)
    assert ay_lat >= -1.0, f"ay_lat={ay_lat} unexpectedly negative"

    # Solve multiple times — warm-starting should work
    times = []
    import time
    for _ in range(20):
        t0     = time.perf_counter()
        ax, ay = ctrl.solve(x0, ref_pos, ref_vel)
        dt_ms  = (time.perf_counter() - t0) * 1000
        times.append(dt_ms)
    import numpy as np2
    mean_ms = np2.mean(times)
    max_ms  = np2.max(times)

    print(f"    OSQP available={_OSQP_AVAILABLE} | "
          f"mean={mean_ms:.1f}ms max={max_ms:.1f}ms "
          f"NF1={'OK' if max_ms < 33 else 'WARN'}")

    # Performance summary
    perf = ctrl.performance_summary()
    assert "mean_ms" in perf
    assert "nf1_ok"  in perf


# ── Test 2: OSQP fallback when osqp not available ────────────────────────────

def t2_osqp_proportional_fallback():
    import numpy as np
    from av_simulation.execution import mpc as mpc_mod

    # Temporarily disable OSQP
    orig = mpc_mod._OSQP_AVAILABLE
    mpc_mod._OSQP_AVAILABLE = False

    ctrl = mpc_mod.OSQPMPCController()
    x0      = np.array([50.0, 0.0, 15.0, 0.0])
    ref_pos = np.array([80.0, 4.0])
    ref_vel = np.array([15.0, 0.0])

    ax, ay = ctrl.solve(x0, ref_pos, ref_vel)

    mpc_mod._OSQP_AVAILABLE = orig   # restore

    from av_simulation.execution.mpc import U_MIN, U_MAX
    assert U_MIN[0] <= ax  <= U_MAX[0]
    assert U_MIN[1] <= ay  <= U_MAX[1]
    print(f"    Fallback: ax={ax:.3f}, ay={ay:.3f}")


# ── Test 3: Auditability logger (NF5) ────────────────────────────────────────

def t3_audit_logger():
    import tempfile, json, os
    logger_mod = _load_logger()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Point logger to temp dir
        orig_dir = logger_mod.LOG_DIR
        logger_mod.LOG_DIR = tmpdir

        g = logger_mod.GraphOpLogger(enabled=True)
        l = logger_mod.LLMAuditLogger(enabled=True)
        m = logger_mod.MPCPerfLogger(enabled=True)

        # Graph write
        g.log_write("obs_001", "Obstacle",
                    {"confidence": 0.9, "lane_occupancy": 1.0},
                    source="agent0", step=42)
        g.log_edge("vehicle_agent0", "obs_001", "blocks", "agent0", step=42)
        g.log_merge("agent1", added_nodes=2, updated_nodes=0,
                    added_edges=1, agent_id="agent0", step=43)

        # LLM audit
        l.log_intent(
            graph_subtext  = "Obstacle: tree blocks lane",
            intent_node    = {"action": "SpatiallyOrderedBypass", "priority": 1, "params": {}},
            message_node   = {"content": "Initiating bypass"},
            agent_id       = "agent0",
            step           = 42,
            model_used     = "rule_based",
            latency_ms     = 0.5,
        )
        l.log_strategy_selection(
            strategy_name  = "SpatiallyOrderedBypass",
            plan_id        = "coordplan_12345",
            agent_id       = "agent0",
            step           = 42,
            graph_snapshot = "Obstacle=1, Vehicle=4",
        )

        # MPC perf
        m.log_solve("agent0", solve_ms=12.3, step=42, status="solved")
        m.log_solve("agent0", solve_ms=18.7, step=43, status="solved")

        # Flush
        g.flush()
        l.flush()
        m.flush()

        # Verify files written
        graph_path = os.path.join(tmpdir, "graph_ops.jsonl")
        llm_path   = os.path.join(tmpdir, "llm_audit.jsonl")
        mpc_path   = os.path.join(tmpdir, "mpc_perf.jsonl")

        assert os.path.exists(graph_path), "graph_ops.jsonl not created"
        assert os.path.exists(llm_path),   "llm_audit.jsonl not created"
        assert os.path.exists(mpc_path),   "mpc_perf.jsonl not created"

        # Parse and verify
        with open(graph_path) as f:
            graph_records = [json.loads(l) for l in f if l.strip()]
        assert len(graph_records) == 3
        assert graph_records[0]["op"]        == "write_node"
        assert graph_records[0]["node_type"] == "Obstacle"
        assert graph_records[1]["op"]        == "write_edge"
        assert graph_records[2]["op"]        == "merge_diff"

        with open(llm_path) as f:
            llm_records = [json.loads(l) for l in f if l.strip()]
        assert len(llm_records) == 2
        assert llm_records[0]["intent_action"]  == "SpatiallyOrderedBypass"
        assert llm_records[1]["strategy_name"]  == "SpatiallyOrderedBypass"

        with open(mpc_path) as f:
            mpc_records = [json.loads(l) for l in f if l.strip()]
        assert len(mpc_records) == 2
        assert mpc_records[0]["solve_ms"] == 12.3
        assert mpc_records[0]["nf1_ok"]   == True
        assert mpc_records[1]["nf1_ok"]   == True

        logger_mod.LOG_DIR = orig_dir

    print(f"    graph_records={len(graph_records)}, llm_records={len(llm_records)}, "
          f"mpc_records={len(mpc_records)} — all correct")


# ── Test 4: Analysis script stats computation ─────────────────────────────────

def t4_analysis_stats():
    from benchmarks.analysis import compute_stats, split_ablation

    # Build synthetic rows
    rows = []
    for i in range(100):
        rows.append({
            "obstacle_type":        ["fallen_tree", "debris"][i % 2],
            "placement":            ["near", "mid", "far"][i % 3],
            "seed":                 i % 10,
            "success":              1 if i % 4 != 0 else 0,   # 75% TSR
            "time_to_first_move":   40 + i % 20,
            "fleet_clearance_time": 120 + i % 60,
            "graph_diffs":          10 + i % 5,
            "llm_tokens":           0,
            "collisions":           0,
            "steps_run":            300,
            "wall_time_s":          5.0,
            "ablation_no_graph":    0,
            "ablation_no_leader":   0,
            "ablation_no_repo":     0,
        })
    # Add some ablation rows
    for i in range(20):
        rows.append({
            "obstacle_type":        "fallen_tree",
            "placement":            "mid",
            "seed":                 i,
            "success":              1 if i % 3 != 0 else 0,   # ~67% TSR (worse)
            "time_to_first_move":   60,
            "fleet_clearance_time": 200,
            "graph_diffs":          0,   # no graph = no diffs
            "llm_tokens":           0,
            "collisions":           1,
            "steps_run":            300,
            "wall_time_s":          4.0,
            "ablation_no_graph":    1,
            "ablation_no_leader":   0,
            "ablation_no_repo":     0,
        })

    stats = compute_stats(rows, label="Test run")
    assert 0.0 <= stats["tsr"] <= 1.0
    assert stats["total"] == 120
    assert "type_tsr"  in stats
    assert "place_tsr" in stats
    assert "nf3_ok"    in stats

    groups = split_ablation(rows)
    assert "Full system"     in groups
    assert "No SemanticGraph" in groups
    assert len(groups["Full system"])      == 100
    assert len(groups["No SemanticGraph"]) == 20

    full_tsr = compute_stats(groups["Full system"],      "Full")["tsr"]
    no_g_tsr = compute_stats(groups["No SemanticGraph"], "No graph")["tsr"]

    print(f"    full TSR={full_tsr:.1%}, no-graph TSR={no_g_tsr:.1%}")
    print(f"    ablation groups={list(groups.keys())}")


# ── Test 5: End-to-end — logger + analysis round-trip ────────────────────────

def t5_logger_to_analysis():
    """Write MPC logs then check NF1 via analysis.check_nf1."""
    import tempfile, json, os
    logger_mod = _load_logger()
    from benchmarks.analysis import check_nf1
    import io
    from contextlib import redirect_stdout

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_dir       = logger_mod.LOG_DIR
        logger_mod.LOG_DIR = tmpdir

        m = logger_mod.MPCPerfLogger(enabled=True)

        # Log 100 solves, all < 20 ms
        for i in range(100):
            m.log_solve("agent0", solve_ms=8.0 + (i % 5), step=i)
        m.flush()

        logger_mod.LOG_DIR = orig_dir

        # Capture and parse check_nf1 output
        buf = io.StringIO()
        with redirect_stdout(buf):
            check_nf1(tmpdir)
        output = buf.getvalue()

        assert "MET" in output, "NF1 check should be MET"
        assert "100" in output, "Should show 100 solves"

    print(f"    NF1 round-trip: logged 100 solves, analysis confirmed MET")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  VLA-MAC Phase 4 — Optimisation & Analysis Tests")
    print("=" * 60)

    test("1. OSQP MPC solve + warm-start performance (NF1)",   t1_osqp_mpc_solve)
    test("2. OSQP proportional fallback when unavailable",     t2_osqp_proportional_fallback)
    test("3. Auditability logger JSONL output (NF5)",          t3_audit_logger)
    test("4. Analysis stats + ablation split (F11)",           t4_analysis_stats)
    test("5. Logger → analysis NF1 round-trip",                t5_logger_to_analysis)

    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print("=" * 60)
    if passed == total:
        print(f"  All {total}/{total} tests passed. Phase 4 complete.")
        print(f"  ALL PHASES DONE — VLA-MAC implementation ready for simulation.")
    else:
        print(f"  {passed}/{total} passed.")
    print("=" * 60 + "\n")
    sys.exit(0 if passed == total else 1)