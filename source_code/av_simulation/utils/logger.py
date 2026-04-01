"""
utils/logger.py
===============
Auditability logger for VLA-MAC (NF5).

Thesis requirement NF5:
  All LLM outputs (intent, strategy selection) must be logged with their
  corresponding graph state for auditability.

Every graph read/write and every LLM output is logged as a single JSON
object per line (JSONL format) to logs/graph_ops.jsonl and
logs/llm_audit.jsonl respectively.

Usage
-----
from av_simulation.utils.logger import graph_logger, llm_logger

# Log a graph write
graph_logger.log_write(node_id, node_type, attrs, source, step)

# Log an LLM call
llm_logger.log_intent(graph_subtext, intent_node, message_node, agent_id, step)

Thread-safe: uses a threading.Lock per file.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional


# ── Log directory ──────────────────────────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


# ── Base JSONL logger ──────────────────────────────────────────────────────────

class JSONLLogger:
    """
    Thread-safe append-only JSONL logger.
    Each log() call writes one JSON object per line.
    """

    def __init__(self, filename: str, enabled: bool = True) -> None:
        self.enabled  = enabled
        self._path    = os.path.join(LOG_DIR, filename)
        self._lock    = threading.Lock()
        self._buffer  = []
        self._buf_size = 10   # flush every N records
        if enabled:
            _ensure_log_dir()

    def log(self, record: dict) -> None:
        if not self.enabled:
            return
        record["_ts"] = round(time.time(), 4)
        line = json.dumps(record, default=str) + "\n"
        with self._lock:
            self._buffer.append(line)
            if len(self._buffer) >= self._buf_size:
                self._flush_locked()

    def flush(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        try:
            with open(self._path, "a") as f:
                f.writelines(self._buffer)
            self._buffer.clear()
        except Exception as e:
            print(f"[LOGGER] Failed to write {self._path}: {e}")

    def __del__(self) -> None:
        try:
            self.flush()
        except Exception:
            pass


# ── Graph operation logger ─────────────────────────────────────────────────────

class GraphOpLogger(JSONLLogger):
    """
    Logs every SemanticGraph read and write with timestamp and step.
    Output: logs/graph_ops.jsonl
    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__("graph_ops.jsonl", enabled=enabled)

    def log_write(
        self,
        node_id:   str,
        node_type: str,
        attrs:     Dict[str, Any],
        source:    str,
        step:      int = -1,
    ) -> None:
        """Log a graph node write."""
        # Truncate long attrs for readability
        safe_attrs = {
            k: (str(v)[:80] if isinstance(v, (list, dict)) else v)
            for k, v in attrs.items()
        }
        self.log({
            "op":        "write_node",
            "node_id":   node_id,
            "node_type": node_type,
            "attrs":     safe_attrs,
            "source":    source,
            "step":      step,
        })

    def log_edge(
        self,
        from_id:  str,
        to_id:    str,
        label:    str,
        source:   str,
        step:     int = -1,
    ) -> None:
        """Log a graph edge write."""
        self.log({
            "op":      "write_edge",
            "from_id": from_id,
            "to_id":   to_id,
            "label":   label,
            "source":  source,
            "step":    step,
        })

    def log_merge(
        self,
        sender_id:    str,
        added_nodes:  int,
        updated_nodes: int,
        added_edges:  int,
        agent_id:     str,
        step:         int = -1,
    ) -> None:
        """Log a graph diff merge (V2V sync)."""
        self.log({
            "op":            "merge_diff",
            "from_agent":    sender_id,
            "into_agent":    agent_id,
            "added_nodes":   added_nodes,
            "updated_nodes": updated_nodes,
            "added_edges":   added_edges,
            "step":          step,
        })

    def log_read(
        self,
        node_id:  str,
        agent_id: str,
        purpose:  str,
        step:     int = -1,
    ) -> None:
        """Log a graph node read (for auditability)."""
        self.log({
            "op":       "read_node",
            "node_id":  node_id,
            "agent_id": agent_id,
            "purpose":  purpose,
            "step":     step,
        })


# ── LLM audit logger ──────────────────────────────────────────────────────────

class LLMAuditLogger(JSONLLogger):
    """
    Logs all LLM inputs + outputs paired with graph state snapshot (NF5).
    Output: logs/llm_audit.jsonl
    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__("llm_audit.jsonl", enabled=enabled)

    def log_intent(
        self,
        graph_subtext: str,
        intent_node:   dict,
        message_node:  dict,
        agent_id:      str,
        step:          int = -1,
        model_used:    str = "rule_based",
        latency_ms:    float = 0.0,
    ) -> None:
        """Log an LLMReasoner.generate_intent() call (F5 + NF5)."""
        self.log({
            "event":         "generate_intent",
            "agent_id":      agent_id,
            "step":          step,
            "model":         model_used,
            "latency_ms":    round(latency_ms, 2),
            "graph_subtext": graph_subtext[:500],   # truncated to keep logs manageable
            "intent_action": intent_node.get("action",   ""),
            "intent_priority": intent_node.get("priority", 0),
            "intent_params": intent_node.get("params",   {}),
            "message":       message_node.get("content", ""),
        })

    def log_strategy_selection(
        self,
        strategy_name:  str,
        plan_id:        str,
        agent_id:       str,
        step:           int = -1,
        graph_snapshot: str = "",
    ) -> None:
        """Log a StrategySelector.select_and_instantiate() call (F7 + NF5)."""
        self.log({
            "event":          "strategy_selected",
            "agent_id":       agent_id,
            "step":           step,
            "strategy_name":  strategy_name,
            "plan_id":        plan_id,
            "graph_snapshot": graph_snapshot[:300],
        })

    def log_execution_status(
        self,
        agent_id:        str,
        step:            int,
        tracking_error:  float,
        completed:       bool,
        safety_warning:  bool,
    ) -> None:
        """Log an ExecutionStatus write (F8 + NF5)."""
        self.log({
            "event":           "execution_status",
            "agent_id":        agent_id,
            "step":            step,
            "tracking_error":  round(tracking_error, 3),
            "completed":       completed,
            "safety_warning":  safety_warning,
        })


# ── MPC performance logger ─────────────────────────────────────────────────────

class MPCPerfLogger(JSONLLogger):
    """
    Logs MPC solve times for NF1 compliance monitoring.
    Output: logs/mpc_perf.jsonl
    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__("mpc_perf.jsonl", enabled=enabled)

    def log_solve(
        self,
        agent_id:   str,
        solve_ms:   float,
        step:       int,
        status:     str = "solved",
    ) -> None:
        self.log({
            "event":     "mpc_solve",
            "agent_id":  agent_id,
            "step":      step,
            "solve_ms":  round(solve_ms, 3),
            "status":    status,
            "nf1_ok":    solve_ms < 33.0,
        })


# ── Module-level singletons (lazy-init) ───────────────────────────────────────
#
# Import these in any module that needs to log:
#
#   from av_simulation.utils.logger import graph_logger, llm_logger, mpc_logger
#
# They are disabled by default and enabled by calling enable() in main.py.

class _LazyLogger:
    """Proxy that creates the real logger on first use."""

    def __init__(self, cls, **kwargs):
        self._cls    = cls
        self._kwargs = kwargs
        self._inner  = None

    def _get(self):
        if self._inner is None:
            self._inner = self._cls(**self._kwargs)
        return self._inner

    def enable(self):
        self._inner = self._cls(enabled=True, **{
            k: v for k, v in self._kwargs.items() if k != "enabled"
        })

    def __getattr__(self, name):
        return getattr(self._get(), name)


graph_logger = _LazyLogger(GraphOpLogger,   enabled=False)
llm_logger   = _LazyLogger(LLMAuditLogger,  enabled=False)
mpc_logger   = _LazyLogger(MPCPerfLogger,   enabled=False)


def enable_all_loggers() -> None:
    """Call once in main.py to activate all audit logs."""
    graph_logger.enable()
    llm_logger.enable()
    mpc_logger.enable()
    print(f"[NF5] Auditability logging enabled → {os.path.abspath(LOG_DIR)}/")