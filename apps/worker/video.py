"""Video processing using FFmpeg for Stream2Short Worker."""

import subprocess
import os
import json
from pathlib import Path
from typing import Optional

from webcam_detect import (
    detect_webcam_region,
    detect_layout_with_cache,
    detect_layout_segments,
    WebcamRegion,
    LayoutInfo,
    LayoutSegment,
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
    effective_type: str = None,  # Webcam type for margin adjustment (side_box gets smaller margin)
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
    # STEP 1: Use YOLO bbox directly (no margin expansion).
    # YOLO is trained on the full webcam overlay rectangle, so its bbox
    # already covers the entire overlay accurately (Mean IoU 0.921).
    # Just snap to even dimensions for encoder compatibility.
    # ==========================================================================
    cam_x = even(max(0, webcam_region.x))
    cam_y = even(max(0, webcam_region.y))
    cam_w = even(webcam_region.width)
    cam_h = even(webcam_region.height)
    
    # Re-clamp to bounds after snapping to even
    cam_w = min(cam_w, even(src_width - cam_x))
    cam_h = min(cam_h, even(src_height - cam_y))
    
    print(f"   Crop bbox (even): {cam_w}x{cam_h} at ({cam_x},{cam_y})")
    
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


def render_top_band_video(
    input_path: str,
    output_path: str,
    webcam_region: WebcamRegion,
    subtitle_path: str | None = None,
    width: int = 1080,
    height: int = 1920,
) -> str:
    """
    Render a TOP_BAND layout video - source already has webcam band + gameplay stacked.
    
    For clips where the webcam is a wide band across the top (or bottom), instead of
    creating a new split overlay, we preserve the source layout and crop to vertical.
    
    Strategy:
    1. Scale source to fill output width (1080px) while maintaining aspect ratio
    2. Crop to 9:16, anchoring so the webcam band remains visible
    3. Optionally detect horizontal divider and center crop around it
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        webcam_region: Detected webcam region (used to determine anchor point)
        subtitle_path: Optional path to subtitle file
        width: Output width (default 1080)
        height: Output height (default 1920)
        
    Returns:
        Path to output video
        
    Raises:
        VideoProcessingError: If FFmpeg fails
    """
    print(f"🎬 Rendering TOP_BAND video: {input_path} -> {output_path}")
    print(f"   Webcam region: {webcam_region}")
    
    # Get source dimensions
    src_width, src_height = get_video_dimensions(input_path)
    print(f"   Source: {src_width}x{src_height}")
    
    # Check subtitle file
    use_subtitles = False
    if subtitle_path and os.path.exists(subtitle_path):
        file_size = os.path.getsize(subtitle_path)
        if file_size > 10:
            use_subtitles = True
    
    # Calculate scale factor to fill output width
    scale_factor = width / src_width
    scaled_height = int(src_height * scale_factor)
    
    # Calculate crop/pad offset to preserve webcam visibility
    # If webcam is at top (y near 0), anchor at top
    # If webcam is at bottom (y near src_height), anchor at bottom
    webcam_center_y = webcam_region.y + webcam_region.height / 2
    webcam_y_ratio = webcam_center_y / src_height
    
    # Build filter chain
    filters = []
    
    # Scale to fill output width (use -2 for height to maintain aspect ratio and ensure even)
    filters.append(f"scale={width}:-2:flags=lanczos")
    
    if scaled_height < height:
        # Source is SHORTER than target after scaling - need to PAD (letterbox)
        # Use pad filter to add black bars
        pad_total = height - scaled_height
        
        # Determine where to place the video content based on webcam position
        if webcam_y_ratio < 0.35:
            # Webcam near top - put video at top, pad bottom
            pad_y = 0
            print(f"   Source shorter ({scaled_height}px < {height}px), padding bottom (webcam at top)")
        elif webcam_y_ratio > 0.65:
            # Webcam near bottom - put video at bottom, pad top
            pad_y = pad_total
            print(f"   Source shorter ({scaled_height}px < {height}px), padding top (webcam at bottom)")
        else:
            # Webcam in middle - center the video
            pad_y = pad_total // 2
            print(f"   Source shorter ({scaled_height}px < {height}px), centering with padding")
        
        # pad=width:height:x:y - adds padding around video
        filters.append(f"pad={width}:{height}:0:{pad_y}:black")
    else:
        # Source is TALLER than or equal to target - CROP
        if scaled_height > height:
            max_crop_y = scaled_height - height
            
            if webcam_y_ratio < 0.35:
                # Webcam near top - anchor crop at top
                crop_y = 0
                print(f"   Webcam near top (y_ratio={webcam_y_ratio:.2f}), anchoring crop at top")
            elif webcam_y_ratio > 0.65:
                # Webcam near bottom - anchor crop at bottom
                crop_y = max_crop_y
                print(f"   Webcam near bottom (y_ratio={webcam_y_ratio:.2f}), anchoring crop at bottom")
            else:
                # Webcam in middle - center the crop
                crop_y = max_crop_y // 2
                print(f"   Webcam in middle (y_ratio={webcam_y_ratio:.2f}), centering crop")
            
            # crop=width:height:x:y - x=0 since we scaled to exact width
            filters.append(f"crop={width}:{height}:0:{crop_y}")
        else:
            # Exact height match after scaling - no crop/pad needed
            print(f"   Scaled height matches target exactly")
    
    # Add setsar=1 to ensure square pixels
    filters.append("setsar=1")
    
    # Add subtitles if available
    if use_subtitles:
        escaped_path = escape_ffmpeg_path(subtitle_path)
        if subtitle_path.endswith(".ass"):
            filters.append(f"ass='{escaped_path}'")
        else:
            filters.append(f"subtitles='{escaped_path}':force_style='FontSize=28,Bold=1,Alignment=2'")
    
    filter_complex = ",".join(filters)
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]
    
    print(f"   Filter: {filter_complex}")
    print(f"   Running FFmpeg...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ TOP_BAND render complete: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg failed: {e.stderr}"
        print(f"❌ {error_msg}")
        raise VideoProcessingError(error_msg)


def apply_subtitles_to_video(
    input_path: str,
    output_path: str,
    subtitle_path: str,
) -> str:
    """
    Apply subtitles to an already-rendered base video.
    
    This is an optimization - instead of re-running the full render pipeline
    twice (once without subs, once with), we render the base once and then
    apply subtitles in a quick second pass.
    
    Args:
        input_path: Path to base video (already rendered to 9:16)
        output_path: Path to output video with subtitles
        subtitle_path: Path to ASS or SRT subtitle file
        
    Returns:
        Path to output video
        
    Raises:
        VideoProcessingError: If FFmpeg fails
    """
    print(f"🎬 Applying subtitles: {input_path} -> {output_path}")
    
    if not os.path.exists(subtitle_path):
        print(f"⚠️ Subtitle file not found, copying base video")
        import shutil
        shutil.copy(input_path, output_path)
        return output_path
    
    file_size = os.path.getsize(subtitle_path)
    if file_size <= 10:
        print(f"⚠️ Subtitle file empty, copying base video")
        import shutil
        shutil.copy(input_path, output_path)
        return output_path
    
    escaped_path = escape_ffmpeg_path(subtitle_path)
    
    if subtitle_path.endswith(".ass"):
        sub_filter = f"ass='{escaped_path}'"
    else:
        sub_filter = f"subtitles='{escaped_path}':force_style='FontSize=28,Bold=1,Alignment=2'"
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vf", sub_filter,
        "-c:v", "libx264",
        "-preset", "fast",  # Fast since it's just adding subs
        "-crf", "23",
        "-c:a", "copy",  # Copy audio stream (no re-encode)
        "-movflags", "+faststart",
        output_path
    ]
    
    print(f"   Running subtitle overlay...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Subtitles applied: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg subtitle overlay failed: {e.stderr}"
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
    layout_info: LayoutInfo | None = None,
) -> str:
    """
    Automatically render the best layout based on layout classification.
    
    Uses cached layout detection to avoid running detection twice (for
    with-subs and without-subs versions).
    
    Layouts:
    - FULL_CAM: Entire clip is webcam. Use face-centered full-frame crop.
    - TOP_BAND: Source already stacked (webcam band + gameplay). Use smart crop.
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
        layout_info: Pre-computed layout (skips detection when provided)
        
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
    
    # Use pre-computed layout or run detection with cache
    if layout_info is None:
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
    
    elif layout == 'TOP_BAND' and layout_info.webcam_region:
        # Source is already stacked (webcam band + gameplay) - use smart vertical crop
        print(f"🎥 Using TOP_BAND layout: {layout_info.reason}")
        return render_top_band_video(
            input_path=input_path,
            output_path=output_path,
            webcam_region=layout_info.webcam_region,
            subtitle_path=subtitle_path,
            width=width,
            height=height,
        )
    
    elif layout == 'SPLIT' and layout_info.webcam_region:
        # Webcam + gameplay split layout
        print(f"🎥 Using SPLIT layout: {layout_info.reason}")
        # Get effective_type from webcam_region for margin adjustment
        eff_type = getattr(layout_info.webcam_region, 'effective_type', None)
        return render_split_layout_video(
            input_path=input_path,
            output_path=output_path,
            webcam_region=layout_info.webcam_region,
            subtitle_path=subtitle_path,
            width=width,
            height=height,
            effective_type=eff_type,
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


# =============================================================================
# TEMPORAL LAYOUT TRANSITIONS
# =============================================================================

def _ass_time_to_seconds(t: str) -> float:
    """Parse an ASS timestamp string (H:MM:SS.cs) to seconds."""
    h, m, s_cs = t.split(':')
    s, cs = s_cs.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100


def _seconds_to_ass_time(secs: float) -> str:
    """Format seconds as an ASS timestamp (H:MM:SS.cs)."""
    secs = max(0.0, secs)
    h = int(secs // 3600)
    secs -= h * 3600
    m = int(secs // 60)
    secs -= m * 60
    s = int(secs)
    cs = round((secs - s) * 100)
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def slice_ass_subtitles(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
) -> Optional[str]:
    """
    Slice an ASS subtitle file to the window [start_time, end_time] and
    shift all timestamps so the window starts at t=0.

    Returns output_path if any dialogue lines remain, None otherwise.
    """
    if not os.path.exists(input_path) or os.path.getsize(input_path) <= 10:
        return None

    try:
        with open(input_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  ⚠️ Could not read subtitle file: {e}")
        return None

    header_lines = []
    event_lines = []
    in_events = False

    for line in lines:
        stripped = line.rstrip('\n')
        if stripped.strip() == '[Events]':
            in_events = True
            header_lines.append(line)
            continue
        if not in_events:
            header_lines.append(line)
            continue

        # Inside [Events] section
        if not stripped.startswith('Dialogue:'):
            header_lines.append(line)
            continue

        # Parse Dialogue line: Dialogue: Layer,Start,End,Style,Name,mL,mR,mV,Effect,Text
        parts = stripped.split(',', 9)
        if len(parts) < 10:
            continue

        try:
            ev_start = _ass_time_to_seconds(parts[1])
            ev_end = _ass_time_to_seconds(parts[2])
        except (ValueError, IndexError):
            continue

        # Keep events that overlap with [start_time, end_time]
        if ev_end <= start_time or ev_start >= end_time:
            continue

        # Clamp and shift by -start_time
        new_start = max(0.0, ev_start - start_time)
        new_end = min(end_time - start_time, ev_end - start_time)

        parts[1] = _seconds_to_ass_time(new_start)
        parts[2] = _seconds_to_ass_time(new_end)
        event_lines.append(','.join(parts) + '\n')

    if not event_lines:
        return None

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(header_lines)
            f.writelines(event_lines)
        return output_path
    except Exception as e:
        print(f"  ⚠️ Could not write sliced subtitle: {e}")
        return None


def render_segment(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    layout_info: LayoutInfo,
    subtitle_path: Optional[str],
    temp_dir: str,
    width: int = 1080,
    height: int = 1920,
    seg_index: int = 0,
) -> str:
    """
    Render a time-trimmed segment of the source clip with a specific layout.

    Steps:
    1. Trim [start_time, end_time] from the source via stream-copy (fast).
    2. Optionally slice+shift the subtitle file to match the segment window.
    3. Render the trimmed clip with render_video_auto() using the pre-computed layout.

    Args:
        input_path: Full source video path
        output_path: Destination for the rendered segment
        start_time: Segment start in seconds (from source)
        end_time: Segment end in seconds (from source)
        layout_info: Pre-detected layout for this segment (skips re-detection)
        subtitle_path: ASS subtitle file for the full clip (will be sliced), or None
        temp_dir: Job temp dir for intermediates
        seg_index: Segment index (used to name temp files)

    Returns:
        output_path
    """
    duration = end_time - start_time
    print(f"\n  ✂️  Trimming segment {seg_index}: [{start_time:.2f}s – {end_time:.2f}s] ({duration:.2f}s)")

    # Step 1: Stream-copy trim (no re-encode — fast, preserves quality)
    trimmed_path = os.path.join(temp_dir, f"seg_{seg_index}_raw.mp4")
    trim_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", input_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        trimmed_path,
    ]
    try:
        subprocess.run(trim_cmd, capture_output=True, text=True, check=True)
        print(f"  ✅ Trim complete: {trimmed_path}")
    except subprocess.CalledProcessError as e:
        raise VideoProcessingError(f"Segment trim failed: {e.stderr}")

    # Step 2: Slice subtitle file to this segment window
    seg_subtitle_path = None
    if subtitle_path and os.path.exists(subtitle_path):
        sliced_sub = os.path.join(temp_dir, f"seg_{seg_index}_captions.ass")
        result = slice_ass_subtitles(subtitle_path, sliced_sub, start_time, end_time)
        if result:
            seg_subtitle_path = result
            print(f"  📝 Subtitle sliced for segment {seg_index}")

    # Step 3: Render with pre-computed layout (no re-detection)
    render_video_auto(
        input_path=trimmed_path,
        output_path=output_path,
        subtitle_path=seg_subtitle_path,
        width=width,
        height=height,
        temp_dir=temp_dir,
        layout_info=layout_info,
    )

    print(f"  ✅ Segment {seg_index} rendered: {output_path}")
    return output_path


def concat_video_segments(
    segment_paths: list[str],
    output_path: str,
    crossfade_duration: float = 0.15,
) -> str:
    """
    Concatenate rendered segment videos into a single output.

    When crossfade_duration > 0, uses FFmpeg xfade (video) and acrossfade
    (audio) filters for smooth transitions between layout changes.
    Falls back to a simple stream-copy concat when crossfade_duration == 0.

    Args:
        segment_paths: Ordered list of rendered segment paths
        output_path: Final concatenated video path
        crossfade_duration: Seconds of crossfade at each boundary (default 0.3)

    Returns:
        output_path
    """
    import shutil

    if len(segment_paths) == 1:
        shutil.copy(segment_paths[0], output_path)
        return output_path

    n = len(segment_paths)
    print(f"\n🔗 Concatenating {n} segments (crossfade={crossfade_duration}s)...")

    if crossfade_duration <= 0:
        # Simple stream-copy concat — fastest, no quality loss
        filelist_path = output_path + ".filelist.txt"
        with open(filelist_path, 'w') as f:
            for p in segment_paths:
                f.write(f"file '{p}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", filelist_path,
            "-c", "copy",
            output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Concat complete (stream copy): {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            raise VideoProcessingError(f"Segment concat failed: {e.stderr}")
        finally:
            try:
                os.remove(filelist_path)
            except Exception:
                pass

    # Crossfade concat — measure each segment's duration first
    durations = [get_video_duration(p) for p in segment_paths]

    # Build inputs
    inputs = []
    for p in segment_paths:
        inputs += ["-i", p]

    # Build filter_complex with chained xfade (video) and acrossfade (audio)
    filter_parts = []

    # --- VIDEO: chain xfade ---
    # Running cumulative offset tracks where the next transition starts in output time
    cumulative_offset = 0.0
    prev_video = "0:v"

    for i in range(1, n):
        cumulative_offset += durations[i - 1] - crossfade_duration
        out_label = f"v{i}" if i < n - 1 else "vout"
        filter_parts.append(
            f"[{prev_video}][{i}:v]xfade=transition=fade"
            f":duration={crossfade_duration}"
            f":offset={cumulative_offset:.4f}"
            f"[{out_label}]"
        )
        prev_video = out_label

    # --- AUDIO: chain acrossfade ---
    prev_audio = "0:a"
    for i in range(1, n):
        out_label = f"a{i}" if i < n - 1 else "aout"
        filter_parts.append(
            f"[{prev_audio}][{i}:a]acrossfade=d={crossfade_duration}"
            f":c1=tri:c2=tri"
            f"[{out_label}]"
        )
        prev_audio = out_label

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "256k",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Concat complete (xfade): {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        # If xfade fails (e.g., audio codec issues), fall back to stream-copy
        print(f"⚠️ xfade concat failed, falling back to stream-copy: {e.stderr[:300]}")
        return concat_video_segments(segment_paths, output_path, crossfade_duration=0)


def _anchor_segments_to_primary(
    segments: list[LayoutSegment],
    primary: LayoutInfo,
) -> list[LayoutSegment]:
    """
    Correct false NO_WEBCAM temporal segments by replacing them with the primary layout.

    Single-frame YOLO (used in temporal sampling) is less reliable than the
    multi-frame consensus used in primary detection. Small corner overlays
    (e.g. 7-8% of frame area) are frequently missed per-frame, producing
    spurious NO_WEBCAM results even when the overlay is clearly present.

    Strategy:
    1. If the primary detection found a webcam, replace any temporal NO_WEBCAM
       segment with the primary layout + webcam_region.
    2. Collapse consecutive segments that now share the same layout into one.

    This preserves genuine transitions (e.g. SPLIT → FULL_CAM) while
    eliminating false negatives.
    """
    if not segments or primary.layout == 'NO_WEBCAM':
        return segments

    # Step 1: replace NO_WEBCAM with primary layout
    anchored: list[LayoutSegment] = []
    replaced = 0
    for seg in segments:
        if seg.layout == 'NO_WEBCAM':
            anchored.append(LayoutSegment(
                start_time=seg.start_time,
                end_time=seg.end_time,
                layout=primary.layout,
                webcam_region=primary.webcam_region,
            ))
            replaced += 1
        else:
            anchored.append(seg)

    if replaced:
        print(f"  🔧 Anchoring: replaced {replaced} false NO_WEBCAM segment(s) → {primary.layout}")

    # Step 2: collapse consecutive segments that now share the same layout
    if not anchored:
        return anchored

    collapsed: list[LayoutSegment] = [anchored[0]]
    for seg in anchored[1:]:
        if seg.layout == collapsed[-1].layout:
            collapsed[-1] = LayoutSegment(
                start_time=collapsed[-1].start_time,
                end_time=seg.end_time,
                layout=collapsed[-1].layout,
                webcam_region=collapsed[-1].webcam_region,
            )
        else:
            collapsed.append(seg)

    if len(collapsed) < len(anchored):
        print(f"  🔧 Collapsed to {len(collapsed)} segment(s) after anchoring")

    return collapsed


def render_video_with_transitions(
    input_path: str,
    output_path: str,
    subtitle_path: Optional[str] = None,
    width: int = 1080,
    height: int = 1920,
    temp_dir: str = "/tmp",
) -> str:
    """
    Render a vertical video with automatic mid-clip layout transitions.

    Detects layout changes across the clip (e.g., SPLIT corner_overlay → FULL_CAM)
    and renders each segment with its optimal layout, then concatenates with a
    smooth crossfade. Falls back to single-layout render when no transitions are found.

    This is the primary render entry point — it replaces render_video_auto() in the
    pipeline for all non-segment-specific calls.

    Args:
        input_path: Source clip path
        output_path: Destination for the final vertical video
        subtitle_path: ASS subtitle file path (will be sliced per segment if transitions)
        width: Output width (default 1080)
        height: Output height (default 1920)
        temp_dir: Job temp directory

    Returns:
        output_path
    """
    print(f"\n{'='*60}")
    print(f"🎬 Layout-aware render: {input_path}")
    print(f"{'='*60}")

    # Detect layout including temporal segments (uses cache)
    try:
        layout_info = detect_layout_with_cache(
            video_path=input_path,
            temp_dir=temp_dir,
        )
    except Exception as e:
        print(f"⚠️ Layout detection failed: {e} — falling back to center crop")
        return render_vertical_video(
            input_path=input_path,
            output_path=output_path,
            subtitle_path=subtitle_path,
            width=width,
            height=height,
        )

    segments = layout_info.segments or []

    # Anchor: single-frame YOLO misses small overlays that multi-frame consensus finds.
    # Replace NO_WEBCAM temporal segments with the primary layout when primary found a webcam,
    # then collapse any consecutive same-layout segments that result from the correction.
    segments = _anchor_segments_to_primary(segments, layout_info)

    # Determine unique layouts across segments
    unique_layouts = list(dict.fromkeys(s.layout for s in segments))
    has_transitions = len(unique_layouts) > 1

    if not has_transitions:
        # Single layout — use fast standard render (no per-segment work)
        if segments:
            print(f"📹 Single layout throughout: {unique_layouts[0]}")
        return render_video_auto(
            input_path=input_path,
            output_path=output_path,
            subtitle_path=subtitle_path,
            width=width,
            height=height,
            temp_dir=temp_dir,
            layout_info=layout_info,
        )

    # Multi-layout: render each segment independently then concatenate
    print(f"🎬 Multi-layout detected: {' → '.join(s.layout for s in segments)}")

    segment_render_paths = []
    for i, seg in enumerate(segments):
        seg_out = os.path.join(temp_dir, f"seg_{i}_final.mp4")

        seg_layout_info = LayoutInfo(
            layout=seg.layout,
            webcam_region=seg.webcam_region,
            face_center=None,  # face tracking resolves this per-segment at render time
            reason=f"Temporal segment {i}: {seg.layout}",
            bbox_area_ratio=0.0,
            gemini_type=seg.webcam_region.gemini_type if seg.webcam_region else 'unknown',
            confidence=seg.webcam_region.gemini_confidence if seg.webcam_region else 0.0,
            segments=[],
        )

        render_segment(
            input_path=input_path,
            output_path=seg_out,
            start_time=seg.start_time,
            end_time=seg.end_time,
            layout_info=seg_layout_info,
            subtitle_path=subtitle_path,
            temp_dir=temp_dir,
            width=width,
            height=height,
            seg_index=i,
        )
        segment_render_paths.append(seg_out)

    return concat_video_segments(segment_render_paths, output_path, crossfade_duration=0.15)

