"""
test_phase1.py
==============
Standalone test for all Phase 1 components.
Run from source_code/ directory:

    cd source_code
    python test_phase1.py

No GPU, no MetaDrive, no Ollama needed — pure Python.
All 5 tests should print PASS.
"""
import sys
import time
import traceback

sys.path.insert(0, ".")   # ensure source_code/ is on the path

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


# ── Test 1: SemanticGraph basic CRUD ──────────────────────────────────────────

def t1_graph_crud():
    from av_simulation.graph.graph import SemanticGraph

    g = SemanticGraph()

    # add_node
    n = g.add_node("obs_001", "Obstacle",
                   {"confidence": 0.9, "lane_occupancy": 1.0},
                   source="agent0", timestamp=1000.0)
    assert n.node_id == "obs_001"
    assert g._version == 1

    # timestamp conflict: older write must not overwrite
    g.add_node("obs_001", "Obstacle",
               {"confidence": 0.1},
               source="agent1", timestamp=500.0)  # older
    assert g.nodes["obs_001"].attrs["confidence"] == 0.9, \
        "Older timestamp should not overwrite newer node"

    # newer write wins
    g.add_node("obs_001", "Obstacle",
               {"confidence": 0.99},
               source="agent1", timestamp=2000.0)
    assert g.nodes["obs_001"].attrs["confidence"] == 0.99, \
        "Newer timestamp should overwrite older node"

    # query
    obs = g.query("Obstacle")
    assert len(obs) == 1

    # add_edge
    g.add_node("lane_ego", "Lane", {"direction": "forward"}, "agent0")
    g.add_edge("obs_001", "lane_ego", "blocks", {}, "agent0")
    edges = g.get_edges(from_id="obs_001", label="blocks")
    assert len(edges) == 1

    # remove_node (also removes edges)
    g.remove_node("obs_001")
    assert "obs_001" not in g.nodes
    assert len(g.get_edges(from_id="obs_001")) == 0

    # serialize roundtrip
    g.add_node("v0", "Vehicle", {"id": "agent0", "fitness": 0.7}, "agent0")
    js = g.to_json()
    g2 = SemanticGraph.from_json(js)
    assert "v0" in g2.nodes
    assert g2.nodes["v0"].attrs["fitness"] == 0.7


# ── Test 2: GraphDiff serialisation ───────────────────────────────────────────

def t2_diff_serialise():
    from av_simulation.graph.graph import SemanticGraph

    g1 = SemanticGraph()
    g1.add_node("n1", "Vehicle", {"fitness": 0.5}, "a0")

    snap = g1.snapshot()

    g1.add_node("n2", "Obstacle", {"confidence": 0.8}, "a1")
    g1.add_edge("n2", "n1", "blocks", {}, "a1")

    diff = g1.diff(snap)
    assert not diff.is_empty()
    assert len(diff.added_nodes) == 1
    assert diff.added_nodes[0].node_id == "n2"
    assert len(diff.added_edges) == 1

    # Round-trip through JSON
    d2 = diff.from_json(diff.to_json())
    assert d2.added_nodes[0].node_id == "n2"
    assert d2.added_edges[0].label == "blocks"


# ── Test 3: merge_diff (V2V sync) ─────────────────────────────────────────────

def t3_merge_diff():
    from av_simulation.graph.graph import SemanticGraph

    # Agent A's graph
    gA = SemanticGraph()
    snap = gA.snapshot()
    gA.add_node("obs_A", "Obstacle", {"confidence": 0.9, "source": "A"},
                "agentA", timestamp=1000.0)

    diff_A = gA.diff(snap)

    # Agent B merges A's diff
    gB = SemanticGraph()
    gB.merge_diff(diff_A)
    assert "obs_A" in gB.nodes
    assert gB.nodes["obs_A"].attrs["confidence"] == 0.9

    # Conflict: B has older version of same node — A's should win
    gB.add_node("obs_A", "Obstacle", {"confidence": 0.1, "source": "B"},
                "agentB", timestamp=500.0)
    assert gB.nodes["obs_A"].attrs["confidence"] == 0.9, \
        "Older B timestamp should not beat A's newer timestamp"

    # C has newer conflicting node — should win
    gC = SemanticGraph()
    gC.add_node("obs_A", "Obstacle", {"confidence": 0.5, "source": "C"},
                "agentC", timestamp=9999.0)
    snap_C = gC.snapshot()
    gC.add_node("obs_extra", "Vehicle", {}, "agentC")
    diff_C = gC.diff(snap_C)

    gA.merge_diff(diff_C)
    assert "obs_extra" in gA.nodes

    print(f"    gA={gA.summary()}, gB={gB.summary()}")


# ── Test 4: V2VBus graph diff broadcast ───────────────────────────────────────

def t4_v2v_graph_diff():
    from av_simulation.graph.graph import SemanticGraph
    from av_simulation.graph.diff  import GraphDiff
    from av_simulation.coordination.fleet_coordinator import V2VBus

    bus    = V2VBus()
    graphA = SemanticGraph()
    graphB = SemanticGraph()

    received = []

    # B subscribes — merges incoming diffs
    def on_msg_B(msg):
        if msg["type"] == "GRAPH_DIFF":
            diff = GraphDiff.from_dict(msg["payload"])
            graphB.merge_diff(diff)
            received.append(diff)

    bus.subscribe("agentA", lambda m: None)
    bus.subscribe("agentB", on_msg_B)

    # A writes an obstacle node then broadcasts diff
    snap = graphA.snapshot()
    graphA.add_node("obs_X", "Obstacle",
                    {"confidence": 0.88, "lane_occupancy": 0.75},
                    "agentA")
    diff = graphA.diff(snap)
    bus.broadcast_graph_diff("agentA", diff)

    assert len(received) == 1, "B should have received 1 diff"
    assert "obs_X" in graphB.nodes, "B's graph should contain obs_X after merge"
    assert graphB.nodes["obs_X"].attrs["confidence"] == 0.88

    # Empty diff should NOT be broadcast
    bus.broadcast_graph_diff("agentA", graphA.diff(graphA.snapshot()))
    assert len(received) == 1, "Empty diff should not be broadcast"

    print(f"    graphB after merge: {graphB.summary()}")


# ── Test 5: SimpleRaft graph-backed leader election ───────────────────────────

def t5_raft_graph_leader():
    from av_simulation.graph.graph import SemanticGraph
    from av_simulation.coordination.fleet_coordinator import SimpleRaft

    g     = SemanticGraph()
    raft  = SimpleRaft()

    agents = ["agent0", "agent1", "agent2", "agent3"]
    for aid in agents:
        raft.register_agent(aid)

    # Update fitness — Vehicle nodes written to graph
    raft.update_fitness("agent0", vis=0.9, conf=0.8, res=0.7, graph=g,
                        position=[80.0, 0.0])
    raft.update_fitness("agent1", vis=0.6, conf=0.5, res=0.5, graph=g,
                        position=[70.0, 0.0])
    raft.update_fitness("agent2", vis=0.7, conf=0.6, res=0.6, graph=g,
                        position=[60.0, 0.0])
    raft.update_fitness("agent3", vis=0.5, conf=0.4, res=0.4, graph=g,
                        position=[50.0, 0.0])

    # Vehicle nodes must be in graph
    for aid in agents:
        node = g.get_node(f"vehicle_{aid}")
        assert node is not None, f"Vehicle node for {aid} missing from graph"
        assert "fitness" in node.attrs

    # agent0 should have highest fitness: 0.4*0.9 + 0.3*0.8 + 0.3*0.7 = 0.81
    expected_fitness = round(0.4 * 0.9 + 0.3 * 0.8 + 0.3 * 0.7, 4)
    assert g.nodes["vehicle_agent0"].attrs["fitness"] == expected_fitness, \
        f"Expected {expected_fitness}, got {g.nodes['vehicle_agent0'].attrs['fitness']}"

    # Elect leader — writes is_leader to graph
    leader = raft.elect_leader(graph=g)
    assert leader == "agent0", f"Expected agent0 to be leader, got {leader}"

    leader_node = g.get_leader()
    assert leader_node is not None, "get_leader() should return a node"
    assert leader_node.node_id == "vehicle_agent0"

    # Other agents must not be marked as leader
    for aid in ["agent1", "agent2", "agent3"]:
        node = g.get_node(f"vehicle_{aid}")
        assert not node.attrs.get("is_leader", False), \
            f"{aid} should not be marked as leader"

    print(f"    Graph: {g.summary()}")
    print(f"    Leader node: {leader_node.node_id} "
          f"(fitness={leader_node.attrs['fitness']})")


# ── Run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  VLA-MAC Phase 1 — Component Tests")
    print("=" * 56)

    test("1. SemanticGraph CRUD + timestamp conflict resolution", t1_graph_crud)
    test("2. GraphDiff serialisation roundtrip (JSON)",           t2_diff_serialise)
    test("3. merge_diff — V2V graph synchronisation (F3)",        t3_merge_diff)
    test("4. V2VBus.broadcast_graph_diff integration (F3)",       t4_v2v_graph_diff)
    test("5. SimpleRaft graph-backed leader election (F4)",       t5_raft_graph_leader)

    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print("=" * 56)
    if passed == total:
        print(f"  All {total}/{total} tests passed. Phase 1 ready.")
    else:
        print(f"  {passed}/{total} passed — fix failures before running sim.")
    print("=" * 56 + "\n")
    sys.exit(0 if passed == total else 1)