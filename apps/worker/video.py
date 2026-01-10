"""Video processing using FFmpeg for Stream2Short Worker."""

import subprocess
import os
from pathlib import Path


class VideoProcessingError(Exception):
    """Video processing error."""
    pass


def render_vertical_video(
    input_path: str,
    output_path: str,
    subtitle_path: str | None = None,
    width: int = 1080,
    height: int = 1920,
) -> str:
    """
    Render a vertical (9:16) video with optional burned-in subtitles.
    
    Uses center-crop to fill the vertical frame.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        subtitle_path: Optional path to SRT or ASS subtitle file
        width: Output width (default 1080)
        height: Output height (default 1920)
        
    Returns:
        Path to output video
        
    Raises:
        VideoProcessingError: If FFmpeg fails
    """
    print(f"🎬 Rendering vertical video: {input_path} -> {output_path}")
    
    # Build filter chain
    filters = []
    
    # Scale to fill height, then crop to target dimensions
    # This ensures the video fills the vertical frame without letterboxing
    filters.append(f"scale={width}:{height}:force_original_aspect_ratio=increase")
    filters.append(f"crop={width}:{height}")
    
    # Add subtitles if provided
    if subtitle_path and os.path.exists(subtitle_path):
        # Escape path for FFmpeg filter
        escaped_path = subtitle_path.replace(":", "\\:").replace("'", "\\'")
        
        if subtitle_path.endswith(".ass"):
            # ASS subtitles with custom styling
            filters.append(f"ass='{escaped_path}'")
        else:
            # SRT subtitles with default styling
            # Force styling for better readability on vertical video
            filters.append(
                f"subtitles='{escaped_path}':force_style="
                "'FontName=Arial Black,FontSize=24,PrimaryColour=&H00FFFFFF,"
                "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,"
                "Alignment=2,MarginV=80'"
            )
    
    filter_complex = ",".join(filters)
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", input_path,
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    
    print(f"🔧 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print("✅ Video rendering complete")
        return output_path
        
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg failed: {e.stderr}"
        print(f"❌ {error_msg}")
        raise VideoProcessingError(error_msg)


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using FFprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def get_video_info(video_path: str) -> dict:
    """
    Get video information using FFprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dict with video info (width, height, duration, etc.)
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    
    import json
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    # Find video stream
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        {}
    )
    
    return {
        "width": video_stream.get("width", 0),
        "height": video_stream.get("height", 0),
        "duration": float(data.get("format", {}).get("duration", 0)),
        "codec": video_stream.get("codec_name", ""),
        "fps": eval(video_stream.get("r_frame_rate", "0/1")) if "/" in video_stream.get("r_frame_rate", "0") else 0,
    }

