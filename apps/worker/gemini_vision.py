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
        # Read the image file
        with open(frame_path, 'rb') as f:
            image_bytes = f.read()
        
        prompt = f"""Analyze this video game stream screenshot. The image is {video_width}x{video_height} pixels.

Look for a WEBCAM OVERLAY showing the streamer's face/upper body. This is typically:
- A rectangular area in a corner (usually top-left or top-right)
- Shows a person (the streamer) 
- Has different lighting/background than the game
- May have a border or frame around it

If you find a webcam overlay, respond with ONLY a JSON object like this:
{{"found": true, "x": 0, "y": 0, "width": 320, "height": 180}}

Where x,y is the top-left corner position and width,height are the dimensions.

If there is NO webcam overlay visible, respond with:
{{"found": false}}

IMPORTANT: Respond with ONLY the JSON, no other text."""

        # Upload image and generate content
        from google.genai import types
        
        response = client.models.generate_content(
            model="gemini-1.5-flash-8b",  # Free tier model
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=prompt),
                    ]
                )
            ]
        )
        
        # Parse response
        response_text = response.text.strip()
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            
            if result.get('found'):
                x = int(result.get('x', 0))
                y = int(result.get('y', 0))
                width = int(result.get('width', 0))
                height = int(result.get('height', 0))
                
                # Validate bounds
                if width > 50 and height > 50 and x >= 0 and y >= 0:
                    if x + width <= video_width and y + height <= video_height:
                        print(f"✅ Gemini detected webcam: x={x}, y={y}, w={width}, h={height}")
                        return {
                            'x': x,
                            'y': y,
                            'width': width,
                            'height': height
                        }
                    else:
                        print(f"⚠️ Gemini webcam bounds exceed video dimensions, ignoring")
                else:
                    print(f"⚠️ Gemini webcam too small or invalid position")
            else:
                print("ℹ️ Gemini: No webcam overlay detected in frame")
        else:
            print(f"⚠️ Could not parse Gemini response: {response_text[:200]}")
        
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
