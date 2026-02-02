"""
Gemini Vision for webcam detection.

Uses Google's Gemini model to analyze video frames and detect
the webcam overlay rectangle, which is more reliable than face detection.

Supports multiple layout types:
- corner_overlay: Traditional small webcam in a corner
- top_band: Wide webcam band near top of frame
- bottom_band: Wide webcam band near bottom of frame
- center_box: Large centered webcam block
- full_cam: Entire frame is webcam (no gameplay)
- none: No webcam detected
"""

import os
import re
import json
from typing import Optional, Dict, Any, Tuple
from config import config

# Lazy import to avoid loading if not configured
_client = None


def get_gemini_client():
    """Get or create the Gemini client instance."""
    global _client
    
    if not config.GEMINI_API_KEY:
        return None
    
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
        print("✅ Gemini Vision initialized")
    
    return _client


def list_available_models():
    """List models available for your API key (for debugging)."""
    client = get_gemini_client()
    if not client:
        return []
    
    try:
        models = []
        for m in client.models.list():
            if hasattr(m, 'supported_actions') and m.supported_actions:
                if "generateContent" in m.supported_actions:
                    models.append(m.name)
        return models
    except Exception as e:
        print(f"⚠️ Could not list models: {e}")
        return []


def _parse_gemini_json(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parse JSON from Gemini response.
    
    Handles:
    - Code fences (```json ... ```)
    - Trailing commas
    - Extra whitespace
    - Nested objects
    
    Returns:
        Parsed dict or None if parsing fails
    """
    text = response_text.strip()
    
    # Remove code fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        # Remove closing fence
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()
    
    # Try to find JSON object (handle nested braces)
    # Find the first { and match to its closing }
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # Count braces to find matching close
    brace_count = 0
    end_idx = start_idx
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    
    json_str = text[start_idx:end_idx]
    
    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  ⚠️ JSON parse error: {e}")
        print(f"  ⚠️ Attempted to parse: {json_str[:200]}")
        return None


def _clamp_bbox(x: int, y: int, w: int, h: int, video_width: int, video_height: int) -> Tuple[int, int, int, int]:
    """Clamp bbox to video bounds, ensuring valid dimensions."""
    x = max(0, min(x, video_width - 50))
    y = max(0, min(y, video_height - 50))
    w = max(50, min(w, video_width - x))
    h = max(50, min(h, video_height - y))
    return x, y, w, h


def _to_python_int(val: Any) -> int:
    """Convert numpy/other types to Python int."""
    if hasattr(val, 'item'):  # numpy scalar
        return int(val.item())
    return int(val)


def _to_python_float(val: Any) -> float:
    """Convert numpy/other types to Python float."""
    if hasattr(val, 'item'):  # numpy scalar
        return float(val.item())
    return float(val)


def _is_true_corner_overlay(
    bbox: dict,
    frame_width: int,
    frame_height: int,
    corner: str,
    edge_threshold_ratio: float = 0.02,
) -> bool:
    """
    Check if bbox is truly attached to its corner edges (within 2% of frame dimension).
    
    For mid-right/mid-left overlays that Gemini incorrectly calls "corner_overlay",
    this function returns False so we don't snap them to edges.
    
    Args:
        bbox: Dict with x, y, width, height
        frame_width: Video frame width
        frame_height: Video frame height
        corner: Expected corner ('top-left', 'top-right', etc.)
        edge_threshold_ratio: Max gap ratio to consider "touching" (0.02 = 2%)
        
    Returns:
        True if bbox is actually touching corner edges, False otherwise
    """
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    
    # Calculate thresholds in pixels (2% of dimension)
    threshold_x = int(frame_width * edge_threshold_ratio)
    threshold_y = int(frame_height * edge_threshold_ratio)
    
    # Calculate gaps to each edge
    gap_left = x
    gap_right = frame_width - (x + w)
    gap_top = y
    gap_bottom = frame_height - (y + h)
    
    print(f"  🔍 True corner check: corner={corner}")
    print(f"     Gaps: L={gap_left}, R={gap_right}, T={gap_top}, B={gap_bottom}")
    print(f"     Thresholds: x={threshold_x}px ({edge_threshold_ratio*100:.0f}%), y={threshold_y}px")
    
    if corner == 'top-right':
        touches_right = gap_right <= threshold_x
        touches_top = gap_top <= threshold_y
        print(f"     top-right: touches_right={touches_right}, touches_top={touches_top}")
        return touches_right and touches_top
    
    elif corner == 'top-left':
        touches_left = gap_left <= threshold_x
        touches_top = gap_top <= threshold_y
        return touches_left and touches_top
    
    elif corner == 'bottom-right':
        touches_right = gap_right <= threshold_x
        touches_bottom = gap_bottom <= threshold_y
        return touches_right and touches_bottom
    
    elif corner == 'bottom-left':
        touches_left = gap_left <= threshold_x
        touches_bottom = gap_bottom <= threshold_y
        return touches_left and touches_bottom
    
    return False


def detect_webcam_with_gemini(frame_path: str, video_width: int, video_height: int) -> Optional[Dict[str, Any]]:
    """
    Use Gemini to detect the webcam overlay in a video frame.
    
    Returns a structured result with:
    - found: bool - Whether a webcam was detected
    - type: str - Layout type (corner_overlay, top_band, bottom_band, center_box, full_cam, none)
    - x, y, width, height: int - Bounding box coordinates
    - confidence: float - Detection confidence (0.0 to 1.0)
    - corner: str - Corner position (for corner_overlay type)
    - reason: str - Explanation of the detection
    
    Args:
        frame_path: Path to a frame image from the video
        video_width: Width of the source video
        video_height: Height of the source video
    
    Returns:
        Dict with detection result, or None if API error
    """
    client = get_gemini_client()
    if client is None:
        print("⚠️ Gemini not configured, skipping vision detection")
        return None
    
    try:
        from google.genai import types
        
        # Read the image file
        with open(frame_path, 'rb') as f:
            image_bytes = f.read()
        
        prompt = f"""You are a webcam overlay detector. Analyze this streaming screenshot and return ONLY JSON.

IMAGE: {video_width}×{video_height} pixels. x=0 is LEFT, y=0 is TOP.

=== TASK ===
Find the WEBCAM OVERLAY rectangle (real human person from camera feed) and return its TIGHTEST axis-aligned bounding box.

=== CRITICAL RULES ===
1. Return ONLY the webcam overlay rectangle, EXCLUDING all gameplay
2. Return the TIGHTEST bbox that fully contains the webcam - do NOT over-expand
3. If uncertain, return {{"found": false}} rather than guessing
4. Include only keys: found, type, corner, x, y, width, height, confidence, reason

=== WHAT IS A WEBCAM ===
- Contains a HUMAN FACE or BODY (streamer)
- May have background visible (room, chair, etc.)
- May have a visible border/frame OR be borderless

=== WHAT IS NOT A WEBCAM ===
- Minimap/radar (game UI showing terrain/map)
- HUD elements (health, ammo, abilities)
- Chat overlay (text only)
- Sponsor logos

=== TYPE CLASSIFICATION (strict thresholds) ===

"corner_overlay": Webcam TOUCHES 2 frame edges
- At least one edge within 2% of frame boundary ({int(video_width * 0.02)}px horizontal, {int(video_height * 0.02)}px vertical)
- Example: x < {int(video_width * 0.02)} means touches left edge

"side_box": Webcam is FLOATING (gaps to all edges)
- All edges are > 2% away from frame boundary
- Use this if you see clear space between webcam and frame edges
- VERY IMPORTANT: Return EXACT rectangle edges, not a larger guess

"top_band": Wide bar at top (width >= 55%, height 18-60%, near y=0)
"bottom_band": Wide bar at bottom (width >= 55%, height 18-60%, near bottom)
"center_box": Large centered webcam
"full_cam": Webcam covers >70% of frame
"none": No webcam found

=== PREFERENCE: side_box over corner_overlay ===
If webcam appears near a corner BUT has visible gaps to edges → use "side_box"
Only use "corner_overlay" if edges genuinely touch/nearly touch the frame boundary.

=== MEASUREMENT ===
Return the FULL webcam rectangle (not just face):
- x, y = top-left corner of webcam overlay
- width, height = full dimensions including any border

For floating/side_box webcams:
- Find the exact pixel boundaries of the overlay rectangle
- Do NOT include gameplay pixels
- The webcam usually has sharp edges - find them precisely

=== CONFIDENCE ===
0.9-1.0: Clear webcam, sharp boundary
0.7-0.9: Webcam visible but boundary slightly uncertain
0.5-0.7: Possible webcam
<0.5: Do not return as found

=== RESPONSE FORMAT (JSON ONLY) ===
{{"found": true, "type": "side_box", "corner": "top-right", "x": 850, "y": 100, "width": 350, "height": 250, "confidence": 0.92, "reason": "Floating webcam on right, gaps to all edges"}}

{{"found": true, "type": "corner_overlay", "corner": "bottom-right", "x": {video_width - 400}, "y": {video_height - 280}, "width": 400, "height": 280, "confidence": 0.95, "reason": "Webcam touching right and bottom edges"}}

{{"found": false, "type": "none", "confidence": 0.0, "reason": "Only gameplay visible, no human"}}"""

        # Try models in order of preference (most capable first)
        models_to_try = [
            "gemini-2.5-pro",         # Most capable, try first
            "gemini-2.5-flash",       # Fast and capable
            "gemini-2.5-flash-lite",  # Lightweight fallback
            "gemini-2.0-flash-lite",  # Legacy fast fallback
            "gemini-1.5-flash",       # Legacy fallback
        ]
        
        print(f"  🤖 Gemini models to try: {models_to_try}")
        
        last_error = None
        for model_name in models_to_try:
            try:
                print(f"  🤖 Trying model: {model_name}")
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=prompt),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0,
                    ),
                )
                
                # If we got here, the model worked
                response_text = response.text.strip()
                
                # Log raw response for debugging
                print(f"  📝 Gemini raw response: {response_text[:500]}")
                
                # Parse JSON with robust parser
                result = _parse_gemini_json(response_text)
                
                if result is None:
                    print(f"⚠️ Could not parse Gemini response")
                    return None
                
                print(f"  📊 Parsed JSON: {result}")
                
                # Validate and normalize result
                found = result.get('found', False)
                webcam_type = result.get('type', 'none')
                confidence = _to_python_float(result.get('confidence', 0.0))
                reason = result.get('reason', '')
                
                if not found or webcam_type == 'none':
                    # No webcam detected
                    print(f"ℹ️ Gemini: No webcam found. Reason: {reason}")
                    return {
                        'no_webcam_confirmed': True,
                        'type': 'none',
                        'confidence': confidence,
                        'reason': reason
                    }
                
                # Extract and validate bbox
                x = _to_python_int(result.get('x', 0))
                y = _to_python_int(result.get('y', 0))
                width = _to_python_int(result.get('width', 0))
                height = _to_python_int(result.get('height', 0))
                corner = result.get('corner', None)
                
                # Clamp to video bounds
                x, y, width, height = _clamp_bbox(x, y, width, height, video_width, video_height)
                
                print(f"  📍 Webcam detected: type={webcam_type}, corner={corner}")
                print(f"  📐 Bbox: x={x}, y={y}, w={width}, h={height}, conf={confidence:.2f}")
                
                # =================================================================
                # TRUE CORNER CHECK: Only snap if bbox is truly edge-attached
                # =================================================================
                # For mid-right/mid-left overlays, we should NOT snap to edges
                # because the webcam genuinely doesn't touch the edges.
                # =================================================================
                effective_type = webcam_type  # May be changed if not true corner
                
                if webcam_type == 'corner_overlay' and corner:
                    # Check if bbox actually touches expected corner edges (2% threshold)
                    is_true_corner = _is_true_corner_overlay(
                        {'x': x, 'y': y, 'width': width, 'height': height},
                        video_width, video_height, corner
                    )
                    
                    if is_true_corner:
                        print(f"  ✅ TRUE CORNER detected - enabling edge snapping")
                        # Only snap for true corner overlays
                        if corner == 'top-right':
                            gap_right = video_width - (x + width)
                            gap_top = y
                            if 0 < gap_right <= max(24, int(video_width * 0.03)):
                                print(f"  ✅ top-right: snapping x to right edge (small gap={gap_right}px)")
                                x = video_width - width
                            if 0 < gap_top <= max(24, int(video_height * 0.03)):
                                print(f"  ✅ top-right: snapping y to top edge (small gap={gap_top}px)")
                                y = 0
                        elif corner == 'top-left':
                            gap_left = x
                            gap_top = y
                            if 0 < gap_left <= max(24, int(video_width * 0.03)):
                                x = 0
                            if 0 < gap_top <= max(24, int(video_height * 0.03)):
                                y = 0
                        elif corner == 'bottom-right':
                            gap_right = video_width - (x + width)
                            gap_bottom = video_height - (y + height)
                            if 0 < gap_right <= max(24, int(video_width * 0.03)):
                                x = video_width - width
                            if 0 < gap_bottom <= max(24, int(video_height * 0.03)):
                                y = video_height - height
                        elif corner == 'bottom-left':
                            gap_left = x
                            gap_bottom = video_height - (y + height)
                            if 0 < gap_left <= max(24, int(video_width * 0.03)):
                                x = 0
                            if 0 < gap_bottom <= max(24, int(video_height * 0.03)):
                                y = video_height - height
                    else:
                        # NOT a true corner - treat as side_box (mid-right, etc.)
                        print(f"  ⚠️ NOT a true corner overlay (bbox doesn't touch edges)")
                        print(f"     Gemini said '{webcam_type}' + '{corner}' but edges don't touch")
                        print(f"     → Treating as 'side_box' - NO SNAPPING applied")
                        effective_type = 'side_box'  # Override type for downstream processing
                
                # Final validation
                if width < 50 or height < 50:
                    print(f"⚠️ Gemini webcam too small: {width}x{height}")
                    return None
                
                print(f"✅ Gemini SUCCESS using {model_name}: type={webcam_type}, confidence={confidence:.2f}")
                print(f"   effective_type={effective_type}, corner={corner}, bbox=({x},{y},{width}x{height})")
                
                return {
                    'found': True,
                    'type': webcam_type,
                    'effective_type': effective_type,  # May differ from 'type' for mid-right overlays
                    'corner': corner,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'confidence': confidence,
                    'reason': reason,
                }
                
            except Exception as model_error:
                last_error = model_error
                error_str = str(model_error)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    print(f"  ⚠️ {model_name}: Rate limited, trying next model...")
                    continue
                elif "404" in error_str or "not found" in error_str.lower():
                    print(f"  ⚠️ {model_name}: Model not available, trying next...")
                    continue
                else:
                    # Different error, might be worth reporting
                    print(f"  ⚠️ {model_name} error: {error_str[:100]}")
                    continue
        
        # All models failed
        print(f"⚠️ All Gemini models failed. Last error: {last_error}")
        print("💡 Check your Google AI Studio project quota at https://ai.google.dev/")
        return None
        
    except Exception as e:
        print(f"⚠️ Gemini vision error: {e}")
        return None


def extract_frame_for_analysis(video_path: str, output_path: str, timestamp: float = 2.0) -> bool:
    """
    Extract a single FULL frame from video for analysis.
    
    Args:
        video_path: Path to source video
        output_path: Where to save the frame
        timestamp: When in the video to extract (seconds)
    
    Returns:
        True if extraction succeeded
    """
    import subprocess
    
    try:
        # Extract FULL frame at original resolution
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(timestamp),
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '1',  # Highest quality JPEG
            '-vf', 'scale=-1:-1',  # Keep original size, no scaling
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(output_path):
            # Log the extracted frame dimensions
            import cv2
            img = cv2.imread(output_path)
            if img is not None:
                h, w = img.shape[:2]
                file_size = os.path.getsize(output_path) / 1024
                print(f"  📷 Extracted frame: {w}x{h} pixels, {file_size:.1f}KB")
            return True
        else:
            stderr = result.stderr.decode() if result.stderr else "unknown error"
            print(f"  ⚠️ FFmpeg failed: {stderr[:200]}")
            return False
        
    except Exception as e:
        print(f"⚠️ Frame extraction failed: {e}")
        return False


def save_debug_frame_with_box(frame_path: str, output_path: str, x: int, y: int, w: int, h: int, 
                               label: str = "WEBCAM", color: tuple = (0, 255, 0)):
    """Save a debug image showing the detected webcam box."""
    try:
        import cv2
        img = cv2.imread(frame_path)
        if img is not None:
            # Draw rectangle where webcam was detected
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, f"{label}: {x},{y} {w}x{h}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imwrite(output_path, img)
            print(f"  🖼️ Debug frame saved: {output_path}")
    except Exception as e:
        print(f"  ⚠️ Could not save debug frame: {e}")


def save_debug_frame_multi_box(frame_path: str, output_path: str, boxes: list):
    """
    Save a debug image with multiple boxes drawn.
    
    Args:
        frame_path: Source frame path
        output_path: Output path for debug image
        boxes: List of dicts with keys: x, y, w, h, label, color (BGR tuple)
    """
    try:
        import cv2
        img = cv2.imread(frame_path)
        if img is not None:
            for box in boxes:
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                label = box.get('label', '')
                color = box.get('color', (0, 255, 0))
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                if label:
                    cv2.putText(img, label, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imwrite(output_path, img)
            print(f"  🖼️ Debug frame saved: {output_path}")
    except Exception as e:
        print(f"  ⚠️ Could not save debug frame: {e}")


def get_fullcam_anchor_bbox_with_gemini(
    frame_path: str,
    video_width: int,
    video_height: int
) -> Optional[Dict]:
    """
    Use Gemini to identify the MAIN STREAMER's bounding box for FULL_CAM mode.
    
    This is used as an anchor when face detection picks the wrong person
    (e.g., a background person instead of the streamer).
    
    Args:
        frame_path: Path to a frame image from the video
        video_width: Width of the source video
        video_height: Height of the source video
    
    Returns:
        Dict with {x, y, w, h, confidence} or None if not detected
    """
    client = get_gemini_client()
    if client is None:
        print("⚠️ Gemini not configured, skipping anchor detection")
        return None
    
    try:
        from google.genai import types
        
        # Read the image file
        with open(frame_path, 'rb') as f:
            image_bytes = f.read()
        
        prompt = f"""TASK: Find the MAIN STREAMER's face/head bounding box in this livestream frame.

IMAGE SIZE: {video_width} × {video_height} pixels
COORDINATES: x=0 is LEFT edge, y=0 is TOP edge

CRITICAL RULES - FOLLOW EXACTLY:
1. The MAIN STREAMER is the person SEATED at a desk/setup
2. The MAIN STREAMER is usually in the LOWER-LEFT or CENTER-LEFT area
3. The MAIN STREAMER often wears headphones/headset and may have glasses
4. IGNORE any person STANDING in the background (they are NOT the streamer)
5. IGNORE any person on the far RIGHT side of the frame (likely background)

WHAT TO RETURN:
- x, y: TOP-LEFT corner of face/head bounding box
- w, h: Width and height of face/head region (include some padding)
- confidence: How sure you are this is the main streamer (0.0 to 1.0)

TYPICAL STREAMER POSITION:
- X position: 15% to 50% from left edge ({int(video_width * 0.15)} to {int(video_width * 0.50)})
- Y position: 40% to 80% from top ({int(video_height * 0.40)} to {int(video_height * 0.80)})
- Face size: Usually 150-400 pixels wide

RESPOND WITH ONLY JSON (no markdown, no explanation):
{{"x": <int>, "y": <int>, "w": <int>, "h": <int>, "confidence": <float>}}

If you cannot find the main streamer or are unsure, respond:
{{"error": "reason"}}"""

        # Try models in order of preference (most capable first)
        models_to_try = [
            "gemini-2.5-pro",         # Most capable, try first
            "gemini-2.5-flash",       # Fast and capable
            "gemini-2.5-flash-lite",  # Lightweight fallback
            "gemini-2.0-flash-lite",  # Legacy fast fallback
            "gemini-1.5-flash",       # Legacy fallback
        ]
        
        print(f"  🤖 Gemini anchor models to try: {models_to_try}")
        
        last_error = None
        for model_name in models_to_try:
            try:
                print(f"  🤖 Gemini anchor detection using: {model_name}")
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=prompt),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0,
                    ),
                )
                
                response_text = response.text.strip()
                print(f"  📝 Gemini anchor response: {response_text[:300]}")
                
                # Parse JSON with robust parser
                result = _parse_gemini_json(response_text)
                
                if result is None:
                    print(f"  ⚠️ Could not parse Gemini anchor response")
                    return None
                
                if 'error' in result:
                    print(f"  ⚠️ Gemini could not find streamer: {result['error']}")
                    return None
                
                x = _to_python_int(result.get('x', 0))
                y = _to_python_int(result.get('y', 0))
                w = _to_python_int(result.get('w', 0))
                h = _to_python_int(result.get('h', 0))
                confidence = _to_python_float(result.get('confidence', 0.5))
                
                # Validate bounds
                if w > 50 and h > 50 and x >= 0 and y >= 0:
                    # Clamp to frame
                    x = max(0, min(x, video_width - w))
                    y = max(0, min(y, video_height - h))
                    w = min(w, video_width - x)
                    h = min(h, video_height - y)
                    
                    # Calculate ratios for logging
                    center_x = x + w // 2
                    center_y = y + h // 2
                    x_ratio = center_x / video_width
                    y_ratio = center_y / video_height
                    
                    print(f"  ✅ Gemini anchor bbox: ({x},{y}) {w}x{h} conf={confidence:.2f}")
                    print(f"     Center: ({center_x},{center_y}) x_ratio={x_ratio:.2f} y_ratio={y_ratio:.2f}")
                    
                    return {
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'confidence': confidence,
                        'center_x': center_x,
                        'center_y': center_y,
                    }
                else:
                    print(f"  ⚠️ Invalid bbox from Gemini: ({x},{y}) {w}x{h}")
                
                return None
                
            except Exception as model_error:
                last_error = model_error
                error_str = str(model_error)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    print(f"  ⚠️ {model_name}: Rate limited, trying next...")
                    continue
                elif "404" in error_str or "not found" in error_str.lower():
                    print(f"  ⚠️ {model_name}: Not available, trying next...")
                    continue
                else:
                    print(f"  ⚠️ {model_name} error: {error_str[:100]}")
                    continue
        
        print(f"⚠️ All Gemini models failed for anchor detection. Last error: {last_error}")
        return None
        
    except Exception as e:
        print(f"⚠️ Gemini anchor detection error: {e}")
        return None
