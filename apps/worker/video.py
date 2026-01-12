"""Video processing using FFmpeg for Stream2Short Worker."""

import subprocess
import os
from pathlib import Path


class VideoProcessingError(Exception):
    """Video processing error."""
    pass


def escape_ffmpeg_path(path: str) -> str:
    """
    Escape a file path for use in FFmpeg's subtitle filter.
    
    FFmpeg's filtergraph has multiple levels of escaping:
    1. Backslashes need to be escaped
    2. Single quotes need to be escaped  
    3. Colons need to be escaped (they're option separators)
    """
    # Escape backslashes first, then other special chars
    escaped = path.replace("\\", "\\\\")
    escaped = escaped.replace(":", "\\:")
    escaped = escaped.replace("'", "'\\''")  # End quote, escaped quote, start quote
    return escaped


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
    
    # Check if subtitle file exists and has content
    use_subtitles = False
    if subtitle_path and os.path.exists(subtitle_path):
        file_size = os.path.getsize(subtitle_path)
        print(f"📝 Subtitle file size: {file_size} bytes")
        if file_size > 10:  # More than 10 bytes means it has actual content
            use_subtitles = True
        else:
            print("⚠️ Subtitle file is empty or too small, skipping subtitles")
    
    # Build filter chain
    filters = []
    
    # Scale to fill height, then crop to target dimensions
    # Use lanczos for sharper upscaling
    filters.append(f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos")
    filters.append(f"crop={width}:{height}")
    
    # Add subtitles if available and valid
    if use_subtitles:
        # For FFmpeg subtitles filter, we need to escape special characters
        escaped_path = escape_ffmpeg_path(subtitle_path)
        
        if subtitle_path.endswith(".ass"):
            filters.append(f"ass='{escaped_path}'")
        else:
            # Modern social media style subtitles
            # - Montserrat SemiBold (recommended font from creator research)
            # - Bottom center positioning with safe margins
            # - White text with black outline for readability
            filters.append(
                f"subtitles=filename='{escaped_path}'"
                ":force_style='"
                "FontName=Montserrat SemiBold,"  # Clean modern font
                "FontSize=18,"               # Readable size
                "PrimaryColour=&H00FFFFFF,"  # White text
                "OutlineColour=&H00000000,"  # Black outline
                "BackColour=&H80000000,"     # Semi-transparent shadow
                "Bold=1,"                    # Bold weight
                "BorderStyle=1,"             # Outline + shadow style
                "Outline=2,"                 # Clean outline
                "Shadow=1,"                  # Subtle drop shadow
                "Alignment=2,"               # Bottom center
                "MarginL=20,"                # Left margin
                "MarginR=20,"                # Right margin
                "MarginV=50"                 # Bottom margin
                "'"
            )
    
    filter_complex = ",".join(filters)
    
    # Build FFmpeg command with maximum quality settings
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", input_path,
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-preset", "veryslow",  # Maximum quality encoding
        "-crf", "16",           # Very high quality
        "-profile:v", "high",   # High profile for better compression
        "-level", "4.2",        # Modern compatibility level
        "-pix_fmt", "yuv420p",  # Maximum compatibility
        "-c:a", "aac",
        "-b:a", "256k",         # High quality audio
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
        # If subtitles failed, try without them
        if use_subtitles and "Unable to open" in e.stderr:
            print("⚠️ Subtitle rendering failed, trying without subtitles...")
            return render_vertical_video(
                input_path=input_path,
                output_path=output_path,
                subtitle_path=None,  # Disable subtitles
                width=width,
                height=height,
            )
        
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

