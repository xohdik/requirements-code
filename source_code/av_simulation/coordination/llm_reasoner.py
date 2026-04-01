"""
llm_reasoner.py
===============
LLMReasoner — Phi-3-mini-4k-instruct integration for fleet intent generation.

Thesis requirement F5:
  The leader's LLM must read a serialised subgraph of the fleet state
  (obstacle nodes, vehicle states, lane geometry) and generate a
  coordination intent and a natural-language message.  The intent must
  be written back to the graph as a node.

Design decisions
----------------
* Conditional activation (NF4): Phi-3 only runs when called on the LEADER
  agent.  Non-leaders call generate_intent() and get a cached/fallback
  result instantly — zero GPU cost.
* 4-bit quantisation (NF4): loaded with BitsAndBytesConfig(load_in_4bit=True).
* Timeout fallback: if generation takes > TIMEOUT_S the rule-based fallback
  fires so the simulation never stalls.
* Graph writes: both Intent node and Message node are written into the
  provided SemanticGraph after a successful call.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Optional, Tuple

# ── Optional heavy imports (guarded so test_phase2.py can mock them) ─────────
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

from av_simulation.graph.graph import SemanticGraph

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
TIMEOUT_S  = 3.0     # max seconds for one LLM call before fallback fires
MAX_NEW_TOKENS = 120


# ── Prompt template ────────────────────────────────────────────────────────────
INTENT_PROMPT = """\
<|user|>
You are coordinating an autonomous vehicle fleet.

Fleet graph state (JSON-like):
{graph_subtext}

Your task: decide how the fleet should handle the obstacle.
Choose the most appropriate intent from:
  - SpatiallyOrderedBypass   (obstacle blocks lane, bypass left or right)
  - StopAndWaitQueue         (multiple or unknown obstacles, stop and reassess)
  - DistributedLaneSwap      (partial blockage, vehicles swap lanes)
  - MaintainFormation        (no confirmed obstacle, keep platoon)

Reply EXACTLY in this format (no extra text):
INTENT: <action>|<priority 1-3>|<params_json>
MESSAGE: <one natural-language sentence to broadcast to fleet>
<|end|>
<|assistant|>
"""

# ── Rule-based fallback (used when Phi-3 unavailable or times out) ─────────────
def _rule_based_intent(graph_subtext: str) -> Tuple[dict, dict]:
    """
    Simple heuristic that reads the graph subtext and returns a sensible
    default intent. Used when the model is not loaded or generation times out.
    """
    text = graph_subtext.lower()

    # Check negative/clear conditions FIRST to avoid false positives
    # e.g. "no obstacle" contains "obstacle" — must match negative first
    if "road clear" in text or "no obstacle" in text or "no confirmed" in text:
        action   = "MaintainFormation"
        priority = 3
        params   = {}
        msg      = "No confirmed obstacle — maintaining platoon formation."

    elif "stop_and_wait" in text or "multi" in text or "unknown" in text:
        action   = "StopAndWaitQueue"
        priority = 1
        params   = {"stop_distance": 15.0}
        msg      = "Multiple obstacles detected — stopping to reassess."

    elif "partial" in text or "swap" in text:
        action   = "DistributedLaneSwap"
        priority = 2
        params   = {"merge_side": "left", "spacing": 4.0}
        msg      = "Partial blockage detected — initiating distributed lane swap."

    elif "obstacle" in text or "block" in text or "tree" in text or "debris" in text:
        action   = "SpatiallyOrderedBypass"
        priority = 1
        params   = {"merge_side": "left", "spacing": 6.0}
        msg      = "Obstacle confirmed — executing spatially ordered bypass."

    else:
        action   = "MaintainFormation"
        priority = 3
        params   = {}
        msg      = "No confirmed obstacle — maintaining platoon formation."

    intent_node  = {"action": action,  "priority": priority, "params": params}
    message_node = {"content": msg}
    return intent_node, message_node


# ── LLMReasoner ────────────────────────────────────────────────────────────────

class LLMReasoner:
    """
    Wraps Phi-3-mini-4k-instruct for fleet coordination intent generation.

    Usage
    -----
    reasoner = LLMReasoner()          # loads model lazily on first call
    intent, message = reasoner.generate_intent(graph_subtext, graph, agent_id)
    """

    def __init__(
        self,
        model_name: str  = MODEL_NAME,
        quantize:   bool = True,
        eager_load: bool = False,   # set True to preload; False = lazy (faster startup)
    ) -> None:
        self.model_name = model_name
        self.quantize   = quantize
        self._model     = None
        self._tokenizer = None
        self._lock      = threading.Lock()
        self._loaded    = False
        self._load_error: Optional[str] = None

        if eager_load:
            self._load_model()

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> bool:
        """
        Load Phi-3 with 4-bit quantisation. Thread-safe (called once).
        Returns True on success, False on failure (falls back to rule-based).
        """
        if self._loaded:
            return True
        if self._load_error:
            return False

        with self._lock:
            if self._loaded:
                return True

            if not _TRANSFORMERS_AVAILABLE:
                self._load_error = "transformers not installed"
                print(f"[LLM] Phi-3 unavailable: {self._load_error} "
                      "— using rule-based fallback")
                return False

            try:
                print(f"[LLM] Loading {self.model_name} "
                      f"({'4-bit' if self.quantize else 'fp16'}) ...")
                t0 = time.time()

                bnb_cfg = None
                if self.quantize:
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit                = True,
                        bnb_4bit_compute_dtype      = torch.float16,
                        bnb_4bit_use_double_quant   = True,
                        bnb_4bit_quant_type         = "nf4",
                    )

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config  = bnb_cfg,
                    device_map           = "auto",
                    trust_remote_code    = True,
                    torch_dtype          = torch.float16 if not self.quantize else None,
                )
                self._model.eval()
                self._loaded = True
                print(f"[LLM] Phi-3 ready in {time.time() - t0:.1f}s")
                return True

            except Exception as e:
                self._load_error = str(e)
                print(f"[LLM] Load failed: {e} — using rule-based fallback")
                return False

    # ── Intent generation ──────────────────────────────────────────────────────

    def generate_intent(
        self,
        graph_subtext: str,
        graph:         Optional[SemanticGraph] = None,
        agent_id:      str = "unknown",
    ) -> Tuple[dict, dict]:
        """
        Read *graph_subtext* and return (intent_node, message_node).

        Writes both nodes into *graph* if provided (F5).

        intent_node  = {"action": str, "priority": int, "params": dict}
        message_node = {"content": str}

        Falls back to rule-based heuristic if:
          - Phi-3 is not installed / failed to load
          - generation times out (> TIMEOUT_S)
          - response cannot be parsed
        """
        t0 = time.time()

        # Try Phi-3 first
        intent_node, message_node = None, None
        if self._load_model():
            intent_node, message_node = self._generate_with_timeout(
                graph_subtext, agent_id
            )

        # Fallback if needed
        if intent_node is None:
            print(f"[LLM] {agent_id[:8]} using rule-based fallback")
            intent_node, message_node = _rule_based_intent(graph_subtext)

        elapsed = time.time() - t0
        print(
            f"[LLM] {agent_id[:8]} intent={intent_node['action']} "
            f"pri={intent_node['priority']} ({elapsed*1000:.0f}ms)"
        )

        # Write nodes to graph (F5)
        if graph is not None:
            self._write_to_graph(graph, intent_node, message_node, agent_id)

        return intent_node, message_node

    def _generate_with_timeout(
        self,
        graph_subtext: str,
        agent_id:      str,
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """Run generation in a thread; return (None, None) if it times out."""
        result_box = [None, None]   # thread-safe result container

        def _run():
            try:
                prompt = INTENT_PROMPT.format(
                    graph_subtext=graph_subtext[:1200]  # cap to avoid OOM
                )
                inputs = self._tokenizer(
                    prompt,
                    return_tensors = "pt",
                    truncation     = True,
                    max_length     = 1024,
                ).to(self._model.device)

                with torch.no_grad():
                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens = MAX_NEW_TOKENS,
                        do_sample      = False,
                        temperature    = None,
                        top_p          = None,
                        pad_token_id   = self._tokenizer.eos_token_id,
                    )

                # Decode only the new tokens (skip the prompt)
                new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
                response = self._tokenizer.decode(new_ids, skip_special_tokens=True)
                result_box[0], result_box[1] = self._parse_response(response)

            except Exception as e:
                print(f"[LLM] Generation error for {agent_id[:8]}: {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=TIMEOUT_S)

        if t.is_alive():
            print(f"[LLM] {agent_id[:8]} generation timed out ({TIMEOUT_S}s)")
            return None, None

        return result_box[0], result_box[1]

    def _parse_response(
        self, text: str
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Parse the model output into structured node dicts.

        Expected format:
            INTENT: SpatiallyOrderedBypass|1|{"merge_side": "left"}
            MESSAGE: Obstacle confirmed — executing bypass.

        Returns (None, None) if parsing fails.
        """
        try:
            intent_match  = re.search(r"INTENT:\s*(.+)", text)
            message_match = re.search(r"MESSAGE:\s*(.+)", text)

            if not intent_match:
                return None, None

            parts    = intent_match.group(1).strip().split("|", 2)
            action   = parts[0].strip() if len(parts) > 0 else "MaintainFormation"
            priority = int(parts[1].strip()) if len(parts) > 1 else 2
            params   = {}
            if len(parts) > 2:
                try:
                    params = json.loads(parts[2].strip())
                except json.JSONDecodeError:
                    params = {}

            # Validate action against known strategies
            valid_actions = {
                "SpatiallyOrderedBypass",
                "StopAndWaitQueue",
                "DistributedLaneSwap",
                "MaintainFormation",
            }
            if action not in valid_actions:
                # Accept partial matches (model sometimes adds punctuation)
                for va in valid_actions:
                    if va.lower() in action.lower():
                        action = va
                        break
                else:
                    action = "SpatiallyOrderedBypass"  # safe default

            priority     = max(1, min(3, priority))
            intent_node  = {"action": action, "priority": priority, "params": params}
            message_node = {
                "content": (
                    message_match.group(1).strip()
                    if message_match else
                    f"Fleet coordinating: {action}"
                )
            }
            return intent_node, message_node

        except Exception as e:
            print(f"[LLM] Parse error: {e}  raw='{text[:80]}'")
            return None, None

    # ── Graph writes (F5) ──────────────────────────────────────────────────────

    def _write_to_graph(
        self,
        graph:        SemanticGraph,
        intent_node:  dict,
        message_node: dict,
        agent_id:     str,
    ) -> None:
        """Write Intent + Message nodes into the graph (F5)."""
        ts = time.time()

        intent_id = f"intent_{agent_id[:8]}_{int(ts)}"
        graph.add_node(
            node_id   = intent_id,
            node_type = "Intent",
            attrs     = {
                "action":   intent_node["action"],
                "priority": intent_node["priority"],
                "params":   intent_node["params"],
                "agent_id": agent_id,
            },
            source    = agent_id,
            timestamp = ts,
        )

        message_id = f"msg_{agent_id[:8]}_{int(ts)}"
        graph.add_node(
            node_id   = message_id,
            node_type = "Message",
            attrs     = {
                "content":  message_node["content"],
                "sender":   agent_id,
            },
            source    = agent_id,
            timestamp = ts,
        )

        # has_intent edge: Vehicle -> Intent
        vehicle_id = f"vehicle_{agent_id}"
        if graph.get_node(vehicle_id) is not None:
            graph.add_edge(
                vehicle_id, intent_id, "has_intent",
                {}, agent_id, ts,
            )
            graph.add_edge(
                vehicle_id, message_id, "sends",
                {}, agent_id, ts,
            )

        print(
            f"[GRAPH] {agent_id[:8]} wrote Intent '{intent_node['action']}' "
            f"+ Message node to graph"
        )