"""Video processing using FFmpeg for Stream2Short Worker."""

import subprocess
import os
import json
from pathlib import Path
from typing import Optional

from webcam_detect import detect_webcam_region, WebcamRegion, get_video_dimensions


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
            # ASS filter respects embedded styling (large text, zoom animation)
            filters.append(f"ass='{escaped_path}'")
        else:
            # SRT fallback - Large TikTok/Reels style subtitles
            # - Montserrat ExtraBold for impact
            # - Higher positioning (not too close to bottom)
            # - White text with thick black outline for readability
            filters.append(
                f"subtitles=filename='{escaped_path}'"
                ":force_style='"
                "FontName=Montserrat ExtraBold,"  # Bold modern font
                "FontSize=75,"               # Large TikTok style
                "PrimaryColour=&H00FFFFFF,"  # White text
                "OutlineColour=&H00000000,"  # Black outline
                "Bold=1,"                    # Bold weight
                "BorderStyle=1,"             # Outline style
                "Outline=4,"                 # Thick outline
                "Shadow=0,"                  # No shadow
                "Alignment=2,"               # Bottom center
                "MarginL=50,"                # Left margin
                "MarginR=50,"                # Right margin
                "MarginV=280"                # Higher up from bottom
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


def render_split_layout_video(
    input_path: str,
    output_path: str,
    webcam_region: WebcamRegion,
    subtitle_path: str | None = None,
    width: int = 1080,
    height: int = 1920,
    max_upscale: float = 2.0,  # Maximum upscale to avoid pixelation
    min_webcam_ratio: float = 0.15,  # Minimum 15% for visibility
    max_webcam_ratio: float = 0.35,  # Maximum 35% to leave room for game
) -> str:
    """
    Render a split-layout vertical video with webcam on top and main content below.
    
    Layout:
    ┌──────────────────┐
    │    WEBCAM        │  adaptive height
    │   (face crop)    │
    ├──────────────────┤
    │                  │
    │  MAIN CONTENT    │  remaining height
    │   (gameplay)     │
    │                  │
    └──────────────────┘
    
    The webcam height is calculated adaptively based on the source webcam size
    to avoid excessive upscaling (pixelation) or downscaling (wasted space).
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        webcam_region: Detected webcam region
        subtitle_path: Optional path to subtitle file
        width: Output width (default 1080)
        height: Output height (default 1920)
        max_upscale: Maximum upscale factor to avoid pixelation (default 2.0)
        min_webcam_ratio: Minimum webcam section ratio (default 0.15 = 15%)
        max_webcam_ratio: Maximum webcam section ratio (default 0.35 = 35%)
        
    Returns:
        Path to output video
        
    Raises:
        VideoProcessingError: If FFmpeg fails
    """
    print(f"🎬 Rendering split-layout video: {input_path} -> {output_path}")
    print(f"   Webcam region: {webcam_region}")
    
    # EXACT FIT MODE: Preserve the webcam's exact aspect ratio - NO CROPPING
    # The webcam section height is calculated from the webcam's actual aspect ratio
    
    # Get source dimensions
    src_width, src_height = get_video_dimensions(input_path)
    print(f"   Source video: {src_width}x{src_height}")
    print(f"   Source webcam: {webcam_region.width}x{webcam_region.height}")
    
    # Calculate webcam height that PRESERVES EXACT ASPECT RATIO when scaled to output width
    # Formula: new_height = original_height * (new_width / original_width)
    webcam_aspect_ratio = webcam_region.height / webcam_region.width
    webcam_h = int(width * webcam_aspect_ratio)
    
    # Clamp to reasonable bounds (15% - 40% of output height)
    min_h = int(height * min_webcam_ratio)
    max_h = int(height * max_webcam_ratio)
    
    original_webcam_h = webcam_h
    if webcam_h < min_h:
        print(f"   ⚠️ Webcam would be too small ({webcam_h}px), using minimum {min_h}px")
        webcam_h = min_h
    elif webcam_h > max_h:
        print(f"   ⚠️ Webcam would be too large ({webcam_h}px), using maximum {max_h}px")
        webcam_h = max_h
    
    content_h = height - webcam_h
    
    # Calculate scale factor
    scale_factor = width / webcam_region.width
    
    print(f"   📐 EXACT FIT: {webcam_region.width}x{webcam_region.height} → {width}x{webcam_h}")
    print(f"   Scale factor: {scale_factor:.2f}x")
    print(f"   Layout: webcam={webcam_h}px ({webcam_h*100//height}%), content={content_h}px ({content_h*100//height}%)")
    
    # Check if subtitle file exists and has content
    use_subtitles = False
    if subtitle_path and os.path.exists(subtitle_path):
        file_size = os.path.getsize(subtitle_path)
        if file_size > 10:
            use_subtitles = True
    
    # Build complex filter graph
    # 
    # Input [0:v] splits into:
    #   1. Webcam: crop exact region, scale to fit width (NO additional cropping!)
    #   2. Main content: scale to fill width, center crop for height
    # Then stack vertically
    
    # Webcam: crop the EXACT detected region, scale to EXACTLY fit width
    # Using explicit dimensions - this preserves the webcam's full content
    webcam_filter = (
        f"[0:v]crop={webcam_region.width}:{webcam_region.height}:{webcam_region.x}:{webcam_region.y},"
        f"scale={width}:{webcam_h}:flags=lanczos[webcam]"
    )
    
    # Main content: exclude the webcam corner, scale and center crop
    # We scale the full video to fit the content area, maintaining aspect ratio
    content_filter = (
        f"[0:v]scale={width}:{content_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{content_h}[content]"
    )
    
    # Stack vertically: webcam on top, content below
    stack_filter = "[webcam][content]vstack=inputs=2[stacked]"
    
    # Combine all filters
    filter_parts = [webcam_filter, content_filter, stack_filter]
    
    # Add subtitles if available (apply to final stacked output)
    if use_subtitles:
        escaped_path = escape_ffmpeg_path(subtitle_path)
        
        # Use ASS filter for .ass files (respects embedded styling)
        # Use subtitles filter with force_style for SRT files
        if subtitle_path.endswith(".ass"):
            # ASS filter uses the styling embedded in the ASS file
            subtitle_filter = f"[stacked]ass='{escaped_path}'[final]"
        else:
            # SRT fallback with inline styling
            subtitle_margin_v = int(content_h * 0.15)  # 15% from bottom (higher up)
            subtitle_filter = (
                f"[stacked]subtitles=filename='{escaped_path}'"
                ":force_style='"
                "FontName=Montserrat ExtraBold,"
                "FontSize=75,"
                "PrimaryColour=&H00FFFFFF,"
                "OutlineColour=&H00000000,"
                "Bold=1,"
                "BorderStyle=1,"
                "Outline=4,"
                "Shadow=0,"
                "Alignment=2,"
                "MarginL=50,"
                "MarginR=50,"
                f"MarginV={subtitle_margin_v}"
                "'[final]"
            )
        filter_parts.append(subtitle_filter)
        output_label = "[final]"
    else:
        output_label = "[stacked]"
    
    filter_complex = ";".join(filter_parts)
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", output_label,
        "-map", "0:a",  # Keep original audio
        "-c:v", "libx264",
        "-preset", "veryslow",
        "-crf", "16",
        "-profile:v", "high",
        "-level", "4.2",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "256k",
        "-movflags", "+faststart",
        output_path,
    ]
    
    print(f"🔧 Running FFmpeg with split layout...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print("✅ Split-layout video rendering complete")
        return output_path
        
    except subprocess.CalledProcessError as e:
        # If subtitles failed, try without them
        if use_subtitles and ("Unable to open" in e.stderr or "Error" in e.stderr):
            print("⚠️ Subtitle rendering failed, trying without subtitles...")
            return render_split_layout_video(
                input_path=input_path,
                output_path=output_path,
                webcam_region=webcam_region,
                subtitle_path=None,
                width=width,
                height=height,
            )
        
        error_msg = f"FFmpeg failed: {e.stderr}"
        print(f"❌ {error_msg}")
        raise VideoProcessingError(error_msg)


def render_video_auto(
    input_path: str,
    output_path: str,
    subtitle_path: str | None = None,
    width: int = 1080,
    height: int = 1920,
    enable_webcam_detection: bool = True,
    temp_dir: str = "/tmp",
) -> str:
    """
    Automatically render the best layout based on webcam detection.
    
    If a webcam/face is detected in a corner, uses split layout.
    Otherwise, falls back to simple center crop.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        subtitle_path: Optional path to subtitle file
        width: Output width (default 1080)
        height: Output height (default 1920)
        enable_webcam_detection: Whether to try detecting webcam (default True)
        temp_dir: Temporary directory for frame extraction
        
    Returns:
        Path to output video
    """
    webcam_region = None
    
    if enable_webcam_detection:
        try:
            webcam_region = detect_webcam_region(input_path, temp_dir=temp_dir)
        except Exception as e:
            print(f"⚠️ Webcam detection failed: {e}")
            webcam_region = None
    
    if webcam_region:
        # Use split layout with webcam on top
        print("🎥 Using split layout (webcam + content)")
        return render_split_layout_video(
            input_path=input_path,
            output_path=output_path,
            webcam_region=webcam_region,
            subtitle_path=subtitle_path,
            width=width,
            height=height,
        )
    else:
        # Fall back to simple center crop
        print("📹 Using simple center crop (no webcam detected)")
        return render_vertical_video(
            input_path=input_path,
            output_path=output_path,
            subtitle_path=subtitle_path,
            width=width,
            height=height,
        )

