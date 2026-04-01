"""
test_phase2.py
==============
Standalone tests for all Phase 2 components.
Run from source_code/ directory:

    cd source_code
    python test_phase2.py

No GPU, no MetaDrive, no Ollama needed.
Phi-3 is mocked — tests validate logic, not model weights.
All 5 tests should print PASS.
"""
import sys
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


# ── Test 1: LLMReasoner rule-based fallback ────────────────────────────────────

def t1_llm_reasoner_fallback():
    from av_simulation.coordination.llm_reasoner import LLMReasoner, _rule_based_intent

    # Direct rule-based (no model needed)
    intent, msg = _rule_based_intent("obstacle detected tree blocking lane")
    assert intent["action"] == "SpatiallyOrderedBypass"
    assert isinstance(intent["priority"], int)
    assert "content" in msg

    intent2, msg2 = _rule_based_intent("multiple obstacles unknown")
    assert intent2["action"] == "StopAndWaitQueue"

    intent3, msg3 = _rule_based_intent("partial debris swap")
    assert intent3["action"] == "DistributedLaneSwap"

    intent4, msg4 = _rule_based_intent("road clear no obstacle")
    assert intent4["action"] == "MaintainFormation"

    # LLMReasoner with no model available — should use fallback
    reasoner = LLMReasoner(eager_load=False)
    reasoner._load_error = "mocked: no GPU in test"   # force fallback
    intent5, msg5 = reasoner.generate_intent(
        "obstacle: tree blocks lane",
        agent_id="agent0",
    )
    assert intent5["action"] in {
        "SpatiallyOrderedBypass", "StopAndWaitQueue",
        "DistributedLaneSwap", "MaintainFormation"
    }
    print(f"    fallback intent={intent5['action']}")


# ── Test 2: LLMReasoner response parser ───────────────────────────────────────

def t2_llm_response_parser():
    from av_simulation.coordination.llm_reasoner import LLMReasoner

    reasoner = LLMReasoner(eager_load=False)

    # Well-formed response
    raw = ('INTENT: SpatiallyOrderedBypass|1|{"merge_side": "left", "spacing": 6.0}\n'
           'MESSAGE: Obstacle confirmed — executing spatially ordered bypass.')
    intent, msg = reasoner._parse_response(raw)
    assert intent  is not None,         "Should parse well-formed response"
    assert intent["action"]   == "SpatiallyOrderedBypass"
    assert intent["priority"] == 1
    assert intent["params"]["merge_side"] == "left"
    assert "Obstacle" in msg["content"]

    # Unknown action — should default to SpatiallyOrderedBypass
    raw2 = "INTENT: WeirdActionXYZ|2|{}\nMESSAGE: doing something"
    intent2, _ = reasoner._parse_response(raw2)
    assert intent2 is not None
    assert intent2["action"] == "SpatiallyOrderedBypass"

    # Completely malformed — should return None
    bad = reasoner._parse_response("This is not valid output at all")
    assert bad == (None, None)

    print(f"    parsed: action={intent['action']} priority={intent['priority']}")


# ── Test 3: StrategyRepository graph templates ────────────────────────────────

def t3_strategy_repository():
    from av_simulation.decision.repository import StrategyRepository, ACTION_TO_STRATEGY

    repo = StrategyRepository()

    # list_strategies returns all 4
    names = repo.list_strategies()
    assert "SpatiallyOrderedBypass" in names
    assert "StopAndWaitQueue"       in names
    assert "DistributedLaneSwap"    in names
    assert "MaintainFormation"      in names
    assert len(names) == 4,  f"Expected 4 strategies, got {len(names)}"

    # query_by_name returns full template with graph_template keys
    tmpl = repo.query_by_name("SpatiallyOrderedBypass")
    assert tmpl is not None
    assert "coord_plan_attrs"  in tmpl
    assert "trajectory_schema" in tmpl
    assert "edges"             in tmpl
    assert "default_params"    in tmpl

    # Alias resolution
    tmpl2 = repo.query_by_name("bypass")
    assert tmpl2 is not None
    assert tmpl2["coord_plan_attrs"]["strategy_name"] == "SpatiallyOrderedBypass"

    # Legacy condition-based query still works
    legacy = repo.query("lane_blockage")
    assert legacy["name"] == "SpatiallyOrderedBypass"

    legacy2 = repo.query("multi_obstacle")
    assert legacy2["name"] == "StopAndWaitQueue"

    print(f"    strategies: {names}")


# ── Test 4: StrategySelector instantiation ────────────────────────────────────

def t4_strategy_selector():
    from av_simulation.decision.repository       import StrategyRepository
    from av_simulation.decision.strategy_selector import StrategySelector
    from av_simulation.coordination.llm_reasoner  import LLMReasoner
    from av_simulation.graph.graph                import SemanticGraph

    # Build with mocked reasoner (no GPU)
    repo     = StrategyRepository()
    reasoner = LLMReasoner(eager_load=False)
    reasoner._load_error = "mocked: no GPU"  # force rule-based fallback

    selector = StrategySelector(repository=repo, reasoner=reasoner)

    # Build a test graph with obstacle + vehicles
    graph = SemanticGraph()
    graph.add_node("obs_001", "Obstacle",
                   {"confidence": 0.9, "lane_occupancy": 1.0,
                    "distance_m": 45.0, "type": "fallen_tree"},
                   source="agent0")
    graph.add_node("lane_ego", "Lane", {"direction": "forward"}, source="agent0")
    graph.add_edge("obs_001", "lane_ego", "blocks", {}, "agent0")

    agents = ["agent0", "agent1", "agent2", "agent3"]
    positions = {
        "agent0": [70.0, 0.0],
        "agent1": [60.0, 0.0],
        "agent2": [50.0, 0.0],
        "agent3": [40.0, 0.0],
    }
    for aid, pos in positions.items():
        graph.add_node(f"vehicle_{aid}", "Vehicle",
                       {"id": aid, "fitness": 0.5, "is_leader": aid == "agent0",
                        "position": pos},
                       source=aid)

    fleet_state = {
        "agent_ids":         agents,
        "agents_positions":  positions,
        "leader_id":         "agent0",
        "obstacle_position": [80.0, 0.0],
    }

    plan_id = selector.select_and_instantiate(graph, fleet_state, agent_id="agent0")

    # CoordPlan node must exist
    coord_node = graph.get_node(plan_id)
    assert coord_node is not None, f"CoordPlan node '{plan_id}' missing"
    assert coord_node.node_type == "CoordPlan"
    assert "strategy_name" in coord_node.attrs

    # Trajectory nodes must exist for each vehicle
    traj_edges = graph.get_edges(from_id=plan_id, label="contains_trajectory")
    assert len(traj_edges) == len(agents), (
        f"Expected {len(agents)} trajectory edges, got {len(traj_edges)}"
    )

    # Each vehicle must have assigned_trajectory edge
    for aid in agents:
        assign_edges = graph.get_edges(
            from_id=f"vehicle_{aid}", label="assigned_trajectory"
        )
        assert len(assign_edges) >= 1, \
            f"vehicle_{aid} missing assigned_trajectory edge"
        traj_node = graph.get_node(assign_edges[0].to_id)
        assert traj_node is not None
        assert "waypoints"  in traj_node.attrs
        assert len(traj_node.attrs["waypoints"]) > 0, \
            f"vehicle_{aid} has empty waypoints"

    # Intent + Message nodes must be in graph (from LLMReasoner)
    intents  = graph.query("Intent")
    messages = graph.query("Message")
    assert len(intents)  >= 1, "Intent node missing from graph"
    assert len(messages) >= 1, "Message node missing from graph"

    print(f"    plan_id={plan_id}")
    print(f"    strategy={coord_node.attrs['strategy_name']}")
    print(f"    traj_count={len(traj_edges)}")
    print(f"    graph={graph.summary()}")


# ── Test 5: FleetCoordinator execute_strategy_pipeline with graph ─────────────

def t5_fleet_coordinator_pipeline():
    """
    Test that execute_strategy_pipeline works with the P2 StrategySelector
    path (graph provided) and falls back cleanly without graph.
    Uses a mock adapter to avoid importing metadrive via LLMStrategyAdapter.
    """
    from av_simulation.graph.graph import SemanticGraph
    from av_simulation.coordination.fleet_coordinator import (
        V2VBus, FleetCoordinator, SimpleRaft, FormationState,
    )
    from av_simulation.decision.repository import StrategyRepository
    from av_simulation.coordination.llm_reasoner import LLMReasoner
    from av_simulation.decision.strategy_selector import StrategySelector

    # Minimal mock adapter — avoids importing WaypointPlanner/metadrive
    class MockAdapter:
        def adapt_strategy(self, strategy, agent_positions, obstacle_info, leader_id):
            sorted_agents = sorted(agent_positions.keys(),
                                   key=lambda a: agent_positions[a][0], reverse=True)
            return {
                "strategy_name": strategy.get("name", "SpatiallyOrderedBypass"),
                "vehicle_order": sorted_agents,
                "merge_side":    "left",
                "spacing":       6.0,
                "time_slots":    [i * 0.4 for i in range(len(sorted_agents))],
                "waypoints":     {a: [[80.0, 4.0, 8.0]] for a in sorted_agents},
            }

    bus     = V2VBus()
    repo    = StrategyRepository()
    adapter = MockAdapter()
    fc      = FleetCoordinator(bus, repo, adapter)

    fc.state             = FormationState.ZIPPER_MERGE
    fc.obstacle_detected = True
    fc.obstacle_position = [80.0, 0.0]

    agents    = ["agent0", "agent1", "agent2"]
    positions = {"agent0": [70.0, 0.0], "agent1": [60.0, 0.0], "agent2": [50.0, 0.0]}
    raft      = SimpleRaft()
    for aid in agents:
        raft.register_agent(aid)
        raft.update_fitness(aid, 0.7, 0.6, 0.5)
    raft.elect_leader()

    # Build a minimal graph
    graph = SemanticGraph()
    for aid, pos in positions.items():
        graph.add_node(f"vehicle_{aid}", "Vehicle",
                       {"id": aid, "fitness": 0.5,
                        "is_leader": aid == raft.leader_id,
                        "position": pos},
                       source=aid)

    # Wire P2 selector manually (avoids GPU model load)
    reasoner = LLMReasoner(eager_load=False)
    reasoner._load_error = "mocked"
    selector = StrategySelector(repository=repo, reasoner=reasoner)
    fc._strategy_selector = selector

    # Run pipeline WITH graph (P2 path)
    plan = fc.execute_strategy_pipeline(agents, positions, raft, graph=graph)
    assert plan is not None, "execute_strategy_pipeline returned None"
    assert "vehicle_order" in plan
    assert len(plan["vehicle_order"]) == len(agents)
    assert "waypoints" in plan

    for aid in agents:
        assert aid in fc.merge_assignments, f"{aid} missing from merge_assignments"

    # Run pipeline WITHOUT graph (original fallback path)
    fc2 = FleetCoordinator(V2VBus(), repo, adapter)
    fc2.state             = FormationState.ZIPPER_MERGE
    fc2.obstacle_detected = True
    fc2.obstacle_position = [80.0, 0.0]
    plan2 = fc2.execute_strategy_pipeline(agents, positions, raft, graph=None)
    assert plan2 is not None, "Fallback pipeline returned None"

    print(f"    P2 path:  strategy={plan.get('strategy_name')}, "
          f"order={[v[:8] for v in plan['vehicle_order']]}")
    print(f"    Fallback: strategy={plan2.get('strategy_name')}")

    fc.state             = FormationState.ZIPPER_MERGE
    fc.obstacle_detected = True
    fc.obstacle_position = [80.0, 0.0]

    agents    = ["agent0", "agent1", "agent2"]
    positions = {"agent0": [70.0, 0.0], "agent1": [60.0, 0.0], "agent2": [50.0, 0.0]}
    raft      = SimpleRaft()
    for aid in agents:
        raft.register_agent(aid)
        raft.update_fitness(aid, 0.7, 0.6, 0.5)
    raft.elect_leader()

    # Build a minimal graph
    graph = SemanticGraph()
    for aid, pos in positions.items():
        graph.add_node(f"vehicle_{aid}", "Vehicle",
                       {"id": aid, "fitness": 0.5, "is_leader": aid == raft.leader_id,
                        "position": pos},
                       source=aid)

    # Run pipeline WITH graph (P2 path)
    plan = fc.execute_strategy_pipeline(agents, positions, raft, graph=graph)
    assert plan is not None, "execute_strategy_pipeline returned None"
    assert "vehicle_order" in plan
    assert len(plan["vehicle_order"]) == len(agents)
    assert "waypoints" in plan

    # merge_assignments should be populated
    for aid in agents:
        assert aid in fc.merge_assignments, \
            f"{aid} missing from merge_assignments"

    # Run pipeline WITHOUT graph (fallback path)
    fc2 = FleetCoordinator(V2VBus(), repo, adapter)
    fc2.state             = FormationState.ZIPPER_MERGE
    fc2.obstacle_detected = True
    fc2.obstacle_position = [80.0, 0.0]
    plan2 = fc2.execute_strategy_pipeline(agents, positions, raft, graph=None)
    assert plan2 is not None, "Fallback pipeline returned None"

    print(f"    P2 path: strategy={plan.get('strategy_name')}, "
          f"order={[v[:8] for v in plan['vehicle_order']]}")
    print(f"    Fallback: strategy={plan2.get('strategy_name')}")


# ── Run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  VLA-MAC Phase 2 — LLM Reasoning Tests")
    print("=" * 58)

    test("1. LLMReasoner rule-based fallback (no GPU needed)",   t1_llm_reasoner_fallback)
    test("2. LLMReasoner response parser",                       t2_llm_response_parser)
    test("3. StrategyRepository graph templates (F6)",           t3_strategy_repository)
    test("4. StrategySelector instantiates CoordPlan graph (F7)",t4_strategy_selector)
    test("5. FleetCoordinator P2 pipeline integration",          t5_fleet_coordinator_pipeline)

    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print("=" * 58)
    if passed == total:
        print(f"  All {total}/{total} tests passed. Phase 2 ready.")
    else:
        print(f"  {passed}/{total} passed — fix failures before running sim.")
    print("=" * 58 + "\n")
    sys.exit(0 if passed == total else 1)