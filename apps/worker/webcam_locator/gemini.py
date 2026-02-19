"""
gemini.py – Gemini Vision candidate proposer.

Uses percentage-based coordinates for position + gap descriptions for type info.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_client = None
_MODEL = "gemini-2.0-flash"


def _get_client():
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    if not api_key:
        log.warning("GEMINI_API_KEY not set")
        return None

    from google import genai
    _client = genai.Client(api_key=api_key)
    log.info("Gemini client initialized")
    return _client


def _parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()

    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    end = start
    for i, ch in enumerate(text[start:], start=start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    json_str = text[start:end]
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        log.warning("Failed to parse Gemini JSON: %s", json_str[:200])
        return None


def gemini_propose(
    image_bgr: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> Optional[dict]:
    """Ask Gemini for a webcam bbox proposal."""
    client = _get_client()
    if client is None:
        return None

    try:
        from google.genai import types
    except ImportError:
        log.warning("google-genai not installed")
        return None

    _, jpg_buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_bytes = jpg_buf.tobytes()

    prompt = f"""Analyze this {frame_w}x{frame_h} streaming screenshot. Return ONLY JSON.

Find the WEBCAM OVERLAY rectangle (real person from a camera feed, NOT game characters).

RULES:
- Return the TIGHTEST bounding box around the webcam overlay only
- Do NOT include gameplay pixels
- x=0 is LEFT edge, y=0 is TOP edge
- Return coordinates as PERCENTAGES of frame dimensions (0-100)
- "corner_overlay": webcam touches 2 frame edges (within 2%)
- "side_box": webcam is floating near an edge but has gaps to frame edges
- "full_cam": webcam covers >70% of frame (no gameplay visible)
- "none": no webcam found
- Webcams are typically 15-30% of frame width and 15-30% of frame height
- Prefer "side_box" over "corner_overlay" if there are visible gaps to edges

JSON format:
{{"found": true, "type": "side_box|corner_overlay|full_cam|none", "x_pct": 73.5, "y_pct": 14.0, "w_pct": 25.0, "h_pct": 21.0, "confidence": 0.92}}

{{"found": false, "type": "none", "confidence": 0.0}}"""

    try:
        response = client.models.generate_content(
            model=_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(temperature=0),
        )

        result = _parse_json(response.text)
        if result is None:
            return None

        if not result.get("found", False):
            return None

        x_pct = float(result.get("x_pct", 0))
        y_pct = float(result.get("y_pct", 0))
        w_pct = float(result.get("w_pct", 0))
        h_pct = float(result.get("h_pct", 0))

        x = int(x_pct / 100.0 * frame_w)
        y = int(y_pct / 100.0 * frame_h)
        w = int(w_pct / 100.0 * frame_w)
        h = int(h_pct / 100.0 * frame_h)

        x = max(0, min(x, frame_w - 30))
        y = max(0, min(y, frame_h - 30))
        w = max(30, min(w, frame_w - x))
        h = max(30, min(h, frame_h - y))

        conf = float(result.get("confidence", 0.7))
        gem_type = result.get("type", "unknown")

        log.debug(
            "Gemini: type=%s pct=(%.1f,%.1f,%.1f,%.1f) -> (%d,%d) %dx%d conf=%.2f",
            gem_type, x_pct, y_pct, w_pct, h_pct, x, y, w, h, conf,
        )

        return {
            "x": x, "y": y, "width": w, "height": h,
            "source": "gemini",
            "gemini_type": gem_type,
            "gemini_confidence": conf,
        }

    except Exception as exc:
        log.warning("Gemini API error: %s", exc)
        return None
