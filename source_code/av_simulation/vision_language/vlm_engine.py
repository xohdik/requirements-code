"""
vlm_engine.py
=============
Ollama / LLaVA integration for forward-camera scene understanding.

Public surface:
    VLMEngine      — singleton that wraps the Ollama REST API.
    parse_vlm_output(description) -> dict
                   — extract structured detection fields from raw LLaVA text.
"""

from __future__ import annotations

import base64
import io
import re
from typing import TYPE_CHECKING

import requests
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Ollama connection settings
# ---------------------------------------------------------------------------

OLLAMA_MODEL:   str = "llava"
OLLAMA_HOST:    str = "http://localhost:11434"
OLLAMA_TIMEOUT: int = 30


# ---------------------------------------------------------------------------
# VLMEngine
# ---------------------------------------------------------------------------

class VLMEngine:
    """Thin wrapper around the Ollama /api/generate endpoint for LLaVA.

    Raises RuntimeError on construction if:
      * Ollama is not reachable at OLLAMA_HOST, or
      * the configured model is not available locally.
    """

    SCENE_PROMPT = (
        "You are a forward-facing camera sensor on an autonomous vehicle.\n\n"
        "CRITICAL CONTEXT: The road surface is DARK GREY or BLACK. Any obstacle "
        "will appear as a shape that interrupts the road — look for EDGES, "
        "SILHOUETTES, COLOUR CHANGES, or any rectangular / box-like form sitting "
        "on the road, even if it blends with the background.\n\n"
        "Step 1 — Scan the road surface from bottom-centre to the horizon. "
        "Is there ANY discontinuity, edge, raised object, or colour patch?\n"
        "Step 2 — If yes, describe it. If no, confirm the road is clear.\n\n"
        "If an obstacle IS present, reply EXACTLY:\n"
        "OBSTACLE DETECTED: <type>. "
        "Blocks approximately <N>% of the lane. "
        "Estimated distance: <D> metres.\n\n"
        "If the road is genuinely clear, reply EXACTLY: ROAD CLEAR\n\n"
        "Be brief. One sentence max beyond the template."
    )

    def __init__(self) -> None:
        self._available = True
        self._verify_server()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verify_server(self) -> None:
        """Confirm Ollama is running and the required model is pulled."""
        try:
            resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(
                f"\n[VLM] WARNING: Cannot reach Ollama at {OLLAMA_HOST}.\n"
                "  VLM disabled — lidar-only fallback active.\n"
                "  To enable VLM: run 'ollama serve' in another terminal."
            )
            self._available = False
            return
        available = [m["name"] for m in resp.json().get("models", [])]
        if not any(
            m == OLLAMA_MODEL or m.startswith(OLLAMA_MODEL + ":")
            for m in available
        ):
            print(
                f"\n[VLM] WARNING: Model '{OLLAMA_MODEL}' not found.\n"
                f"  Pull it with: ollama pull {OLLAMA_MODEL}\n"
                f"  Available: {available}\n"
                "  VLM disabled — lidar-only fallback active."
            )
            self._available = False
            return
        self._available = True
        print(f"[VLM] Ollama OK — model '{OLLAMA_MODEL}'")

    def _pil_to_b64(self, img: PILImage.Image) -> str:
        """Encode a PIL image as a base-64 JPEG string for the Ollama payload."""
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe_scene(self, img: PILImage.Image) -> str:
        """Run LLaVA inference on *img* and return the raw text response."""
        if not self._available:
            return ""   # lidar fallback handles detection
        payload = {
            "model":   OLLAMA_MODEL,
            "prompt":  self.SCENE_PROMPT,
            "images":  [self._pil_to_b64(img)],
            "stream":  False,
            "options": {"temperature": 0.1, "num_predict": 80},
        }
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            ans = resp.json().get("response", "").strip()
            print(f"[VLM] Raw: \"{ans}\"")
            return ans
        except requests.exceptions.Timeout:
            print("[VLM] Timeout — lidar fallback.")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"[VLM] Request failed: {e} — lidar fallback.")
            return ""


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def parse_vlm_output(description: str) -> dict:
    """Parse a raw LLaVA response into a structured detection dictionary.

    Returns:
        {
            "detected":              bool,
            "semantic_description":  str,
            "blockage_percent":      int   (0–100),
            "distance_m":            float (metres),
            "confidence":            float (0.0–1.0),
        }
    """
    dl = description.lower()

    clear_phrases = [
        "road clear", "no obstacle", "nothing blocking",
        "clear road", "no object", "road is clear",
    ]
    road_is_clear = any(p in dl for p in clear_phrases)

    obstacle_kws = [
        "obstacle", "block", "object", "barrier", "vehicle",
        "debris", "box", "wall", "cone", "truck", "car", "crate",
    ]
    detected = (not road_is_clear) and any(k in dl for k in obstacle_kws)

    # --- blockage percentage ---
    pct_m = re.search(r'(\d{1,3})\s*(?:%|percent)', dl)
    if pct_m:
        blockage = min(100, int(pct_m.group(1)))
    elif any(w in dl for w in ["full", "complete", "entire", "whole"]):
        blockage = 90
    elif any(w in dl for w in ["half", "partial", "most"]):
        blockage = 55
    elif detected:
        blockage = 50
    else:
        blockage = 0

    # --- distance ---
    dist_m   = re.search(r'(\d+\.?\d*)\s*(?:m\b|metre|meter)', dl)
    distance = float(dist_m.group(1)) if dist_m else 40.0

    # --- confidence ---
    wc   = len(description.split())
    conf = min(0.95, 0.35 + wc / 80.0 + (0.1 if pct_m else 0.0))
    if not detected:
        conf = max(0.1, conf * 0.4)

    return {
        "detected":             detected,
        "semantic_description": description,
        "blockage_percent":     blockage,
        "distance_m":           distance,
        "confidence":           conf,
    }