"""
av_simulation — VLA-MAC Fleet Coordination Simulation package.

SharedSimState is the single mutable container that replaces the monolithic
module-level globals (vlm_engine, raft, fleet_coordinator, directives_global,
agents_positions_global).  One instance is created in main.py and threaded
through every component that previously referenced those globals directly.

Phase 1 addition: `graphs` — per-agent SemanticGraph instances,
keyed by agent_id.  Each VLAPolicy writes to its own graph and reads
merged state (via V2V diffs) from the same dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SharedSimState:
    """Mutable simulation-wide state shared across agents and coordinators.

    Attributes
    ----------
    raft               : SimpleRaft consensus / leader-election object.
    fleet_coordinator  : FleetCoordinator state-machine object.
    agents_positions   : Dict[agent_id -> [x, y]] updated each step in-place.
    vlm_engine         : VLMEngine (Ollama/LLaVA) singleton; None until init.
    directives         : Latest formation directives from FleetCoordinator.
    graphs             : Dict[agent_id -> SemanticGraph] — one local graph per
                         agent, kept in sync via V2V GraphDiff broadcasts (F2-F3).
                         Populated lazily when each VLAPolicy is first created.
    """
    raft:              Any                  = None
    fleet_coordinator: Any                  = None
    agents_positions:  Dict[str, list]      = field(default_factory=dict)
    vlm_engine:        Any                  = None
    directives:        Dict[str, Any]       = field(default_factory=dict)
    graphs:            Dict[str, Any]       = field(default_factory=dict)
    # ^ Any = SemanticGraph; typed as Any to avoid import cycle at package level