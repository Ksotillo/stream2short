"""
Gemini Vision for webcam detection.

Uses Google's Gemini model to analyze video frames and detect
the webcam overlay rectangle, which is more reliable than face detection.
"""

import os
import re
import json
from typing import Optional, Dict
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


def detect_webcam_with_gemini(frame_path: str, video_width: int, video_height: int) -> Optional[Dict[str, int]]:
    """
    Use Gemini to detect the webcam overlay in a video frame.
    
    Args:
        frame_path: Path to a frame image from the video
        video_width: Width of the source video
        video_height: Height of the source video
    
    Returns:
        Dict with x, y, width, height of webcam region, or None if not detected
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
        
        prompt = f"""TASK: Find the WEBCAM OVERLAY RECTANGLE in this gaming stream screenshot.

IMAGE SIZE: {video_width} × {video_height} pixels
COORDINATES: x=0 is LEFT edge, y=0 is TOP edge

IMPORTANT: I need the RECTANGLE BOUNDARY of the webcam overlay, NOT the face position!

WHAT TO LOOK FOR:
A webcam overlay is a rectangular video feed showing a REAL PERSON overlaid on the game.
It usually has a VISIBLE BOUNDARY where it meets the game - this is what I need you to measure.

HOW TO MEASURE:
1. Find where the webcam overlay STARTS (left edge of the webcam rectangle)
2. Find where the webcam overlay ENDS (right edge of the webcam rectangle)  
3. Measure the FULL rectangle including any border/frame

COMMON POSITIONS:
- TOP-RIGHT: The webcam rectangle's RIGHT edge touches or nearly touches x={video_width}
- TOP-LEFT: The webcam rectangle's LEFT edge touches or nearly touches x=0
- Webcams are usually 300-500 pixels wide and 200-350 pixels tall

MEASUREMENT RULES:
- x = LEFT edge of the webcam rectangle (not the face, the RECTANGLE)
- y = TOP edge of the webcam rectangle
- width = FULL width of webcam rectangle (right_edge - left_edge)
- height = FULL height of webcam rectangle (bottom_edge - top_edge)

EXAMPLE for {video_width}x{video_height}:
If webcam is in TOP-RIGHT, touching the right edge, and is 400px wide:
- Right edge of webcam = {video_width}
- Left edge of webcam = {video_width} - 400 = {video_width - 400}
- Answer: {{"found": true, "corner": "top-right", "x": {video_width - 400}, "y": 0, "width": 400, "height": 250}}

DOUBLE-CHECK YOUR ANSWER:
- If corner is "top-right": x + width should ≈ {video_width}
- If corner is "top-left": x should ≈ 0
- If corner is "bottom-right": x + width should ≈ {video_width} AND y + height should ≈ {video_height}

RESPOND WITH ONLY JSON:
{{"found": true, "corner": "<corner>", "x": <number>, "y": <number>, "width": <number>, "height": <number>}}
OR
{{"found": false, "reason": "<what you see>"}}"""

        # Try models in order of preference (free tier)
        models_to_try = [
            "gemini-2.0-flash-lite",  # Free tier, fast
            "gemini-1.5-flash",       # Free tier fallback
        ]
        
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
                
                # Try to extract JSON from response
                json_match = re.search(r'\{[^}]+\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                    print(f"  📊 Parsed JSON: {result}")
                    
                    if result.get('found'):
                        x = int(result.get('x', 0))
                        y = int(result.get('y', 0))
                        width = int(result.get('width', 0))
                        height = int(result.get('height', 0))
                        corner = result.get('corner', 'unknown')
                        
                        print(f"  📍 Webcam at {corner}: x={x}, y={y}, w={width}, h={height}")
                        
                        # Sanity check: verify x+width or y+height match edges for corner positions
                        # Only adjust if SIGNIFICANTLY off (>20% of webcam size)
                        adjustment_threshold = max(width, height) * 0.2
                        
                        if corner == 'top-right':
                            expected_right_edge = video_width
                            actual_right_edge = x + width
                            gap = expected_right_edge - actual_right_edge
                            if gap > adjustment_threshold:
                                # Webcam should touch right edge but doesn't
                                print(f"  ⚠️ top-right webcam ends at x={actual_right_edge}, but video width={video_width} (gap={gap}px)")
                                print(f"     Adjusting x from {x} to {video_width - width}")
                                x = video_width - width
                        elif corner == 'top-left':
                            if x > adjustment_threshold:
                                print(f"  ⚠️ top-left webcam starts at x={x}, adjusting to 0")
                                x = 0
                        elif corner == 'bottom-right':
                            expected_right_edge = video_width
                            actual_right_edge = x + width
                            gap = expected_right_edge - actual_right_edge
                            if gap > adjustment_threshold:
                                print(f"  ⚠️ bottom-right webcam ends at x={actual_right_edge}, adjusting")
                                x = video_width - width
                            expected_bottom = video_height
                            actual_bottom = y + height
                            if expected_bottom - actual_bottom > adjustment_threshold:
                                y = video_height - height
                        elif corner == 'bottom-left':
                            if x > adjustment_threshold:
                                x = 0
                            if video_height - (y + height) > adjustment_threshold:
                                y = video_height - height
                        
                        # Validate bounds
                        if width > 50 and height > 50 and x >= 0 and y >= 0:
                            if x + width <= video_width and y + height <= video_height:
                                print(f"✅ Gemini detected webcam: corner={corner}, x={x}, y={y}, w={width}, h={height}")
                                return {
                                    'x': x,
                                    'y': y,
                                    'width': width,
                                    'height': height,
                                    'corner': corner
                                }
                            else:
                                # Try to fix bounds if they're slightly off
                                print(f"  ⚠️ Bounds exceed video ({video_width}x{video_height}), adjusting...")
                                x = max(0, min(x, video_width - 100))
                                y = max(0, min(y, video_height - 100))
                                width = min(width, video_width - x)
                                height = min(height, video_height - y)
                                if width > 50 and height > 50:
                                    print(f"✅ Adjusted webcam: x={x}, y={y}, w={width}, h={height}")
                                    return {
                                        'x': x,
                                        'y': y,
                                        'width': width,
                                        'height': height,
                                        'corner': corner
                                    }
                                print(f"  ❌ Could not fix bounds")
                        else:
                            print(f"⚠️ Gemini webcam too small: x={x}, y={y}, w={width}, h={height}")
                    else:
                        reason = result.get('reason', 'unknown')
                        print(f"ℹ️ Gemini: No webcam found. Reason: {reason}")
                        # Return special flag so we know Gemini explicitly said "no webcam"
                        # (vs an API error where we should fall back to OpenCV)
                        return {'no_webcam_confirmed': True, 'reason': reason}
                else:
                    print(f"⚠️ Could not parse Gemini response: {response_text[:500]}")
                
                return None  # Model worked but no webcam found
                
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


def save_debug_frame_with_box(frame_path: str, output_path: str, x: int, y: int, w: int, h: int):
    """Save a debug image showing the detected webcam box."""
    try:
        import cv2
        img = cv2.imread(frame_path)
        if img is not None:
            # Draw rectangle where webcam was detected
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, f"WEBCAM: {x},{y} {w}x{h}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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

        # Try models in order of preference
        models_to_try = [
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
        ]
        
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
                
                # Parse JSON
                json_match = re.search(r'\{[^}]+\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    if 'error' in result:
                        print(f"  ⚠️ Gemini could not find streamer: {result['error']}")
                        return None
                    
                    x = int(result.get('x', 0))
                    y = int(result.get('y', 0))
                    w = int(result.get('w', 0))
                    h = int(result.get('h', 0))
                    confidence = float(result.get('confidence', 0.5))
                    
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
