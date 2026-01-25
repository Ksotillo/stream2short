"""Video processing using FFmpeg for Stream2Short Worker."""

import subprocess
import os
import json
from pathlib import Path
from typing import Optional

from webcam_detect import (
    detect_webcam_region,
    detect_layout_with_cache,
    WebcamRegion,
    LayoutInfo,
    FaceCenter,
    get_video_dimensions,
)
from face_tracking import (
    track_faces,
    track_faces_with_fallback,
    generate_ffmpeg_crop_expr,
    FaceTrack,
)


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
    
    # Helper to ensure even dimensions (required by some encoders)
    def even(n):
        return n - (n % 2)
    
    # Get source dimensions
    src_width, src_height = get_video_dimensions(input_path)
    print(f"   Source video: {src_width}x{src_height}")
    print(f"   Original webcam: {webcam_region.width}x{webcam_region.height} at ({webcam_region.x},{webcam_region.y})")
    
    # ==========================================================================
    # STEP 1: Expand detected bbox by 8% margin to avoid tight crops
    # ==========================================================================
    margin = 0.08
    pad_x = int(webcam_region.width * margin)
    pad_y = int(webcam_region.height * margin)
    
    cam_x = max(0, webcam_region.x - pad_x)
    cam_y = max(0, webcam_region.y - pad_y)
    cam_w = min(src_width - cam_x, webcam_region.width + 2 * pad_x)
    cam_h = min(src_height - cam_y, webcam_region.height + 2 * pad_y)
    
    # Make all crop dimensions EVEN to prevent implicit padding with yuv420p
    cam_x = even(cam_x)
    cam_y = even(cam_y)
    cam_w = even(cam_w)
    cam_h = even(cam_h)
    
    # Re-clamp to bounds after snapping to even
    cam_w = min(cam_w, even(src_width - cam_x))
    cam_h = min(cam_h, even(src_height - cam_y))
    
    print(f"   Expanded bbox (even): {cam_w}x{cam_h} at ({cam_x},{cam_y}) (+{margin*100:.0f}% margin)")
    
    # ==========================================================================
    # STEP 2: Calculate native webcam height (preserves exact aspect ratio)
    # Priority: 1) Never crop  2) Avoid bars  3) Keep webcam small if possible
    # ==========================================================================
    native_webcam_h = even(round(width * (cam_h / cam_w)))
    
    # Define bounds with new priority system
    min_webcam_h = even(int(height * min_webcam_ratio))      # 15% = 288px - minimum for visibility
    soft_max_webcam_h = even(int(height * max_webcam_ratio)) # 35% = 672px - prefer to stay under this
    hard_max_webcam_h = even(int(height * 0.60))             # 60% = 1152px - allow growth to avoid bars
    min_content_h = even(int(height * 0.40))                 # 40% = 768px - ensure game still visible
    
    print(f"   Native webcam height: {native_webcam_h}px")
    print(f"   Bounds: min={min_webcam_h}, soft_max={soft_max_webcam_h}, hard_max={hard_max_webcam_h}")
    print(f"   Min content height: {min_content_h}px")
    
    # Choose webcam_h based on priority rules
    clamped = False
    
    if native_webcam_h <= soft_max_webcam_h:
        # Native fits within soft max - use it (ideal case)
        webcam_h = max(min_webcam_h, native_webcam_h)  # Also enforce minimum
        print(f"   ✅ Native height {native_webcam_h}px <= soft_max {soft_max_webcam_h}px - BRANCH A (no bars)")
        
    elif native_webcam_h <= hard_max_webcam_h and (height - native_webcam_h) >= min_content_h:
        # Native exceeds soft max but:
        # - Still under hard max
        # - Leaves enough room for content
        # Use native to AVOID BARS
        webcam_h = native_webcam_h
        print(f"   ✅ Native height {native_webcam_h}px exceeds soft_max but fits hard_max - BRANCH A (no bars)")
        print(f"      Allowing larger webcam to avoid black bars")
        
    else:
        # Must clamp - webcam would take too much space
        clamped = True
        webcam_h = even(height - min_content_h)  # Give content its minimum, webcam gets the rest
        webcam_h = max(min_webcam_h, min(hard_max_webcam_h, webcam_h))  # Clamp to bounds
        print(f"   ⚠️ Native height {native_webcam_h}px too large - clamped to {webcam_h}px")
        print(f"      Using BRANCH B (dimmed background fill, no blur)")
    
    content_h = even(height - webcam_h)
    
    print(f"   Final layout: webcam={webcam_h}px ({webcam_h*100//height}%), content={content_h}px ({content_h*100//height}%)")
    
    # Check if subtitle file exists and has content
    use_subtitles = False
    if subtitle_path and os.path.exists(subtitle_path):
        file_size = os.path.getsize(subtitle_path)
        if file_size > 10:
            use_subtitles = True
    
    # ==========================================================================
    # STEP 3: Build filter_complex based on whether we're clamped or not
    # ==========================================================================
    
    if not clamped:
        # ---------------------------------------------------------------------
        # BRANCH A: Native height fits - simple direct scale, no padding needed
        # The webcam aspect ratio matches exactly, so just scale to fill width
        # Added setsar=1 to force square pixels and prevent pillarboxing
        # ---------------------------------------------------------------------
        webcam_filter = (
            f"[0:v]crop={cam_w}:{cam_h}:{cam_x}:{cam_y},"
            f"setsar=1,"
            f"scale={width}:-2:flags=lanczos,"
            f"setsar=1[webcam]"
        )
        
        content_filter = (
            f"[0:v]setsar=1,"
            f"scale={width}:{content_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{content_h},"
            f"setsar=1[content]"
        )
        
        stack_filter = "[webcam][content]vstack=inputs=2,setsar=1[stacked]"
        filter_parts = [webcam_filter, content_filter, stack_filter]
        
    else:
        # ---------------------------------------------------------------------
        # BRANCH B: Clamped - use dimmed background (NO BLUR) + foreground overlay
        # Background: fills area (may crop) + dim/desaturate
        # Foreground: fits inside (never crops) + centered
        # Added setsar=1 throughout to force square pixels
        # ---------------------------------------------------------------------
        split_filter = "[0:v]split=2[vw][vg]"
        
        webcam_filter = (
            # Crop webcam and set square pixels, then split
            f"[vw]crop={cam_w}:{cam_h}:{cam_x}:{cam_y},setsar=1,split=2[wbg][wfg];"
            # Background: scale to FILL (may crop edges) + dim & desaturate (NO BLUR)
            f"[wbg]scale={width}:{webcam_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{webcam_h},"
            f"eq=brightness=-0.25:saturation=0.5,setsar=1[bg];"
            # Foreground: scale to FIT (never crops) + pad to exact size
            f"[wfg]scale={width}:{webcam_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{webcam_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[fg];"
            # Overlay foreground centered on dimmed background
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[webcam]"
        )
        
        content_filter = (
            f"[vg]setsar=1,"
            f"scale={width}:{content_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{content_h},"
            f"setsar=1[content]"
        )
        
        stack_filter = "[webcam][content]vstack=inputs=2,setsar=1[stacked]"
        filter_parts = [split_filter, webcam_filter, content_filter, stack_filter]
    
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


def render_full_cam_video(
    input_path: str,
    output_path: str,
    face_center: Optional[FaceCenter] = None,
    subtitle_path: str | None = None,
    width: int = 1080,
    height: int = 1920,
    temp_dir: str = "/tmp",
    enable_face_tracking: bool = True,
) -> str:
    """
    Render a FULL_CAM layout video - entire clip is webcam/streamer face.
    
    Uses DYNAMIC face tracking to follow the speaker throughout the video.
    The crop smoothly follows the face position, creating a professional
    portrait-style video that keeps the subject centered.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        face_center: Optional FaceCenter for static fallback crop
        subtitle_path: Optional path to subtitle file
        width: Output width (default 1080)
        height: Output height (default 1920)
        temp_dir: Temporary directory for face tracking
        enable_face_tracking: Whether to use dynamic tracking (default True)
        
    Returns:
        Path to output video
        
    Raises:
        VideoProcessingError: If FFmpeg fails
    """
    print(f"🎬 Rendering FULL_CAM video with face tracking: {input_path} -> {output_path}")
    
    # Get source dimensions
    src_width, src_height = get_video_dimensions(input_path)
    print(f"   Source: {src_width}x{src_height}")
    
    # Check subtitle file
    use_subtitles = False
    if subtitle_path and os.path.exists(subtitle_path):
        file_size = os.path.getsize(subtitle_path)
        if file_size > 10:
            use_subtitles = True
    
    # ==========================================================================
    # Calculate crop region for 9:16 output
    # ==========================================================================
    target_ar = width / height  # 9:16 = 0.5625
    src_ar = src_width / src_height
    
    if src_ar > target_ar:
        # Source is wider - crop sides
        crop_h = src_height
        crop_w = int(src_height * target_ar)
    else:
        # Source is taller - crop top/bottom
        crop_w = src_width
        crop_h = int(src_width / target_ar)
    
    # Ensure even dimensions
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)
    
    # ==========================================================================
    # DYNAMIC FACE TRACKING (with Gemini anchor fallback for background faces)
    # ==========================================================================
    face_track: Optional[FaceTrack] = None
    use_dynamic_tracking = False
    
    if enable_face_tracking:
        print("   🔍 Running face tracking (with Gemini fallback if needed)...")
        
        # Extract initial_center from face_center (from layout detection) to seed tracking
        initial_center = None
        initial_size = None
        if face_center:
            initial_center = (face_center.center_x, face_center.center_y)
            initial_size = (face_center.width, face_center.height)
            print(f"   🎯 Using layout face_center as tracking seed: {initial_center}")
        
        try:
            # Use the new fallback-aware tracking function
            # This will try standard face tracking first, then Gemini anchor if
            # the detected faces appear to be background people
            # IMPORTANT: Pass initial_center to prevent wrong anchor lock
            face_track = track_faces_with_fallback(
                video_path=input_path,
                temp_dir=temp_dir,
                sample_interval=1.5,  # Sample every 1.5 seconds for better coverage
                ema_alpha=0.4,  # Smooth but responsive
                initial_center=initial_center,
                initial_size=initial_size,
            )
            
            if face_track and len(face_track.keyframes) > 1:
                use_dynamic_tracking = True
                print(f"   ✅ Face tracking: {len(face_track.keyframes)} keyframes")
            elif face_track and len(face_track.keyframes) == 1:
                print("   📍 Single face position detected, using static crop")
            else:
                print("   ⚠️ No faces tracked, falling back to left-biased static crop")
                
        except Exception as e:
            print(f"   ⚠️ Face tracking failed: {e}, using static crop")
    
    # ==========================================================================
    # Build crop filter (dynamic or static)
    # ==========================================================================
    if use_dynamic_tracking and face_track:
        # Generate dynamic FFmpeg expressions
        x_expr, y_expr = generate_ffmpeg_crop_expr(
            face_track=face_track,
            crop_width=crop_w,
            crop_height=crop_h,
            target_face_y_ratio=0.45,  # Face at 45% from top
        )
        
        # Check if expressions are dynamic (contain 't' for time-based interpolation)
        is_dynamic = 'if(lt(t,' in x_expr or 'if(lt(t,' in y_expr
        
        if is_dynamic:
            print(f"   🎬 TRUE DYNAMIC PANNING with {len(face_track.keyframes)} keyframes")
            print(f"   📐 Crop filter uses time-based interpolation (smooth panning)")
            
            # CRITICAL: Escape commas in expressions for FFmpeg filter graph parsing
            # FFmpeg uses commas to separate filters, so commas inside expressions must be escaped with backslash
            # When using subprocess.run with a list, no shell escaping needed - just FFmpeg escaping
            x_expr_escaped = x_expr.replace(',', r'\,')
            y_expr_escaped = y_expr.replace(',', r'\,')
            
            crop_filter = f"crop={crop_w}:{crop_h}:{x_expr_escaped}:{y_expr_escaped}"
            print(f"   📐 CROP FILTER: crop={crop_w}:{crop_h}:[{len(x_expr)}ch expr]:[{len(y_expr)}ch expr]")
        else:
            print(f"   📍 Static crop (low movement or single keyframe)")
            crop_filter = f"crop={crop_w}:{crop_h}:{x_expr}:{y_expr}"
            print(f"   📐 CROP FILTER: {crop_filter}")
        
    elif face_track and len(face_track.keyframes) == 1:
        # Single keyframe - static crop at detected position
        kf = face_track.keyframes[0]
        crop_x = max(0, min(kf.center_x - crop_w // 2, src_width - crop_w))
        crop_y = max(0, min(kf.center_y - int(crop_h * 0.45), src_height - crop_h))
        crop_filter = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
        print(f"   📍 Static crop at tracked position: ({crop_x},{crop_y})")
        print(f"   📐 CROP FILTER: {crop_filter}")
        
    elif face_center:
        # Fallback to provided face_center
        print(f"   👤 Using provided face center: ({face_center.center_x}, {face_center.center_y})")
        
        target_face_y = int(crop_h * 0.45)
        crop_y = face_center.center_y - target_face_y
        crop_x = face_center.center_x - crop_w // 2
        
        crop_x = max(0, min(crop_x, src_width - crop_w))
        crop_y = max(0, min(crop_y, src_height - crop_h))
        
        crop_filter = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
        print(f"   📐 Face-centered crop: {crop_w}x{crop_h} at ({crop_x},{crop_y})")
        print(f"   📐 CROP FILTER: {crop_filter}")
    else:
        # Center crop (no face detected)
        crop_x = (src_width - crop_w) // 2
        crop_y = (src_height - crop_h) // 2
        crop_filter = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
        print(f"   📐 Center crop (no face): {crop_w}x{crop_h} at ({crop_x},{crop_y})")
        print(f"   📐 CROP FILTER: {crop_filter}")
    
    # ==========================================================================
    # Build FFmpeg filter chain
    # ==========================================================================
    filters = []
    
    # Crop (dynamic or static)
    filters.append(crop_filter)
    
    # Scale to output size
    filters.append(f"scale={width}:{height}:flags=lanczos")
    
    # Force square pixels
    filters.append("setsar=1")
    
    # Add subtitles if available
    if use_subtitles:
        escaped_path = escape_ffmpeg_path(subtitle_path)
        
        if subtitle_path.endswith(".ass"):
            filters.append(f"ass='{escaped_path}'")
        else:
            filters.append(
                f"subtitles=filename='{escaped_path}'"
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
                "MarginV=280"
                "'"
            )
    
    filter_complex = ",".join(filters)
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", filter_complex,
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
    
    print(f"🔧 Running FULL_CAM render...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        tracking_mode = "dynamic" if use_dynamic_tracking else "static"
        print(f"✅ FULL_CAM video rendering complete ({tracking_mode} tracking)")
        return output_path
        
    except subprocess.CalledProcessError as e:
        if use_subtitles and "Unable to open" in e.stderr:
            print("⚠️ Subtitle rendering failed, trying without subtitles...")
            return render_full_cam_video(
                input_path=input_path,
                output_path=output_path,
                face_center=face_center,
                subtitle_path=None,
                width=width,
                height=height,
                temp_dir=temp_dir,
                enable_face_tracking=enable_face_tracking,
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
    Automatically render the best layout based on layout classification.
    
    Uses cached layout detection to avoid running detection twice (for
    with-subs and without-subs versions).
    
    Layouts:
    - FULL_CAM: Entire clip is webcam. Use face-centered full-frame crop.
    - SPLIT: Webcam + gameplay. Use split layout with webcam on top.
    - NO_WEBCAM: No webcam detected. Use simple center crop.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        subtitle_path: Optional path to subtitle file
        width: Output width (default 1080)
        height: Output height (default 1920)
        enable_webcam_detection: Whether to try detecting webcam (default True)
        temp_dir: Temporary directory for caching and frame extraction
        
    Returns:
        Path to output video
    """
    if not enable_webcam_detection:
        # Webcam detection disabled - use simple center crop
        print("📹 Webcam detection disabled, using simple center crop")
        return render_vertical_video(
            input_path=input_path,
            output_path=output_path,
            subtitle_path=subtitle_path,
            width=width,
            height=height,
        )
    
    # Use cached layout detection (runs detection only once per job)
    try:
        layout_info = detect_layout_with_cache(
            video_path=input_path,
            temp_dir=temp_dir,
        )
    except Exception as e:
        print(f"⚠️ Layout detection failed: {e}")
        # Fall back to simple center crop
        return render_vertical_video(
            input_path=input_path,
            output_path=output_path,
            subtitle_path=subtitle_path,
            width=width,
            height=height,
        )
    
    # ==========================================================================
    # Choose render function based on layout
    # ==========================================================================
    layout = layout_info.layout
    
    if layout == 'FULL_CAM':
        # Entire clip is webcam - use dynamic face tracking
        print(f"🎥 Using FULL_CAM layout with face tracking: {layout_info.reason}")
        return render_full_cam_video(
            input_path=input_path,
            output_path=output_path,
            face_center=layout_info.face_center,
            subtitle_path=subtitle_path,
            width=width,
            height=height,
            temp_dir=temp_dir,
            enable_face_tracking=True,  # Always enable face tracking for FULL_CAM
        )
    
    elif layout == 'SPLIT' and layout_info.webcam_region:
        # Webcam + gameplay split layout
        print(f"🎥 Using SPLIT layout: {layout_info.reason}")
        return render_split_layout_video(
            input_path=input_path,
            output_path=output_path,
            webcam_region=layout_info.webcam_region,
            subtitle_path=subtitle_path,
            width=width,
            height=height,
        )
    
    else:
        # NO_WEBCAM or fallback - simple center crop
        print(f"📹 Using simple center crop: {layout_info.reason}")
        return render_vertical_video(
            input_path=input_path,
            output_path=output_path,
            subtitle_path=subtitle_path,
            width=width,
            height=height,
        )

