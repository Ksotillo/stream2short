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
        
        prompt = f"""TASK: Find the STREAMER'S WEBCAM OVERLAY in this gaming stream screenshot.

IMAGE DIMENSIONS: {video_width} pixels wide × {video_height} pixels tall
COORDINATE SYSTEM: x=0 is LEFT edge, y=0 is TOP edge

STEP 1 - SCAN THE IMAGE:
Look at ALL four corners for a small rectangular area showing a REAL HUMAN PERSON:
- TOP-LEFT corner (x near 0, y near 0)
- TOP-RIGHT corner (x near {video_width}, y near 0)  
- BOTTOM-LEFT corner (x near 0, y near {video_height})
- BOTTOM-RIGHT corner (x near {video_width}, y near {video_height})

STEP 2 - IDENTIFY THE WEBCAM:
A webcam overlay shows:
✓ A REAL HUMAN face/body (not a game character)
✓ Real-world lighting (room lights, not game lighting)
✓ Often has a border, frame, or different background
✓ Typical size: 200-500px wide, 150-400px tall

NOT a webcam:
✗ Game UI (health bars, maps, inventory)
✗ Game characters or NPCs
✗ Game environments (walls, floors, ceilings)

STEP 3 - MEASURE PRECISELY:
If you find a webcam, give the EXACT pixel coordinates of its bounding box:
- x = distance from LEFT edge of image to LEFT edge of webcam
- y = distance from TOP edge of image to TOP edge of webcam
- width = webcam rectangle width in pixels
- height = webcam rectangle height in pixels

EXAMPLE: If webcam is in TOP-RIGHT corner of a 1920x1080 video:
- A 400x300 webcam starting at x=1520, y=0 would be: {{"found": true, "corner": "top-right", "x": 1520, "y": 0, "width": 400, "height": 300}}

RESPOND WITH ONLY JSON:
{{"found": true, "corner": "<which corner>", "x": <number>, "y": <number>, "width": <number>, "height": <number>}}
OR
{{"found": false, "reason": "<what you see instead of a webcam>"}}"""

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
    Extract a single frame from video for analysis.
    
    Args:
        video_path: Path to source video
        output_path: Where to save the frame
        timestamp: When in the video to extract (seconds)
    
    Returns:
        True if extraction succeeded
    """
    import subprocess
    
    try:
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(timestamp),
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '2',  # High quality JPEG
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0 and os.path.exists(output_path)
        
    except Exception as e:
        print(f"⚠️ Frame extraction failed: {e}")
        return False
