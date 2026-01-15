"""ASS subtitle writer with speaker-based styling for Stream2Short Worker.

Generates .ass (Advanced SubStation Alpha) subtitle files with different
styles for each speaker.

Features:
- Large, TikTok/Reels style text (75pt)
- Zoom-in animation effect on text appearance
- Multi-color speaker support (primary=white, others=varied colors)

Color palette (ASS BGR format):
- Primary speaker: White (&H00FFFFFF)
- Speaker 2: Cyan (&H00FFFF00) - cool/masculine
- Speaker 3: Pink (&H00FF77FF) - warm/feminine  
- Speaker 4: Yellow (&H0000FFFF)
- Speaker 5: Lime (&H0000FF00)
- Speaker 6: Orange (&H000099FF)
"""

from typing import Optional


# Animation settings
ZOOM_ANIMATION_DURATION_MS = 100  # Duration of zoom-in effect in milliseconds (fast & snappy)
ZOOM_START_SCALE = 70  # Start at 70% scale
ZOOM_END_SCALE = 100   # End at 100% scale

# Speaker color palette (ASS format: &H00BBGGRR)
SPEAKER_COLORS = {
    "primary": "&H00FFFFFF",    # White - main speaker
    "speaker_1": "&H00FFFF00",  # Cyan - secondary speaker
    "speaker_2": "&H00FF77FF",  # Pink - feminine accent
    "speaker_3": "&H0000FFFF",  # Yellow
    "speaker_4": "&H0000FF00",  # Lime green
    "speaker_5": "&H000099FF",  # Orange
    "speaker_6": "&H00FF0000",  # Blue
    "speaker_7": "&H00FF00FF",  # Magenta
}


def format_ass_time(seconds: float) -> str:
    """
    Format time in ASS format: H:MM:SS.cc (centiseconds).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def generate_ass_header(
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_name: str = "Montserrat ExtraBold",
    font_size: int = 75,  # Large TikTok/Reels style
    margin_v: int = 280,  # Higher up from bottom
    margin_l: int = 50,
    margin_r: int = 50,
    outline: int = 4,     # Thick outline for readability
    shadow: int = 0,      # No shadow, just outline
) -> str:
    """
    Generate ASS file header with script info and styles for multiple speakers.
    
    Styles:
    - Primary: White text for main speaker
    - Speaker1-7: Various colors for other speakers
    
    Args:
        play_res_x: Video width
        play_res_y: Video height
        font_name: Font family name
        font_size: Font size in points
        margin_v: Vertical margin from bottom
        margin_l: Left margin
        margin_r: Right margin
        outline: Outline thickness
        shadow: Shadow distance
        
    Returns:
        ASS header string
    """
    black = "&H00000000"  # Black for outline
    
    # Build styles for all speakers
    styles = []
    for style_name, color in SPEAKER_COLORS.items():
        # Convert style name to ASS style name (Primary, Speaker1, etc.)
        ass_style_name = style_name.replace("_", "").title()
        style_line = (
            f"Style: {ass_style_name},{font_name},{font_size},{color},{color},"
            f"{black},{black},1,0,0,0,100,100,0,0,1,{outline},{shadow},2,"
            f"{margin_l},{margin_r},{margin_v},1"
        )
        styles.append(style_line)
    
    header = f"""[Script Info]
Title: Stream2Short Subtitles
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{chr(10).join(styles)}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header


def get_speaker_style(speaker: str, is_primary: bool, speaker_index: int = 0) -> str:
    """
    Get the ASS style name for a speaker.
    
    Args:
        speaker: Speaker ID from diarization
        is_primary: True if this is the primary (most speaking) speaker
        speaker_index: Index of non-primary speakers (0-based)
        
    Returns:
        ASS style name
    """
    if is_primary:
        return "Primary"
    
    # Map speaker index to style name
    style_keys = [k for k in SPEAKER_COLORS.keys() if k != "primary"]
    if speaker_index < len(style_keys):
        return style_keys[speaker_index].replace("_", "").title()
    
    # Fallback to cycling through colors
    return style_keys[speaker_index % len(style_keys)].replace("_", "").title()


def generate_ass_dialogue(
    start: float,
    end: float,
    text: str,
    is_primary: bool = True,
    speaker: str = "",
    speaker_index: int = 0,
    layer: int = 0,
    enable_zoom_animation: bool = True,
) -> str:
    """
    Generate a single ASS dialogue line with optional zoom-in animation.
    
    Args:
        start: Start time in seconds
        end: End time in seconds
        text: Subtitle text
        is_primary: True for primary speaker (white)
        speaker: Speaker ID from diarization
        speaker_index: Index of this speaker (for color assignment)
        layer: Layer number (default 0)
        enable_zoom_animation: Add zoom-in effect on text appearance
        
    Returns:
        ASS dialogue line
    """
    style = get_speaker_style(speaker, is_primary, speaker_index)
    start_str = format_ass_time(start)
    end_str = format_ass_time(end)
    
    # Escape special ASS characters
    # Newlines use \N, literal backslash uses \\
    text_escaped = text.replace("\\", "\\\\").replace("\n", "\\N")
    
    # Add zoom-in animation effect
    # \fscx = font scale X, \fscy = font scale Y
    # \t(t1,t2,effect) = animate from t1 to t2 milliseconds
    if enable_zoom_animation:
        # Start smaller and zoom to full size over ZOOM_ANIMATION_DURATION_MS
        animation_tag = (
            f"{{\\fscx{ZOOM_START_SCALE}\\fscy{ZOOM_START_SCALE}"
            f"\\t(0,{ZOOM_ANIMATION_DURATION_MS},"
            f"\\fscx{ZOOM_END_SCALE}\\fscy{ZOOM_END_SCALE})}}"
        )
        text_with_effect = animation_tag + text_escaped
    else:
        text_with_effect = text_escaped
    
    return f"Dialogue: {layer},{start_str},{end_str},{style},,0,0,0,,{text_with_effect}"


def segments_to_ass(
    segments: list[dict],
    output_path: str,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_name: str = "Montserrat ExtraBold",
    font_size: int = 75,  # Large TikTok/Reels style
    margin_v: int = 280,  # Higher up from bottom
    margin_l: int = 50,
    margin_r: int = 50,
    enable_zoom_animation: bool = True,
) -> str:
    """
    Convert transcript segments to ASS subtitle file with multi-color speakers.
    
    Segments should have:
    - start: float (seconds)
    - end: float (seconds)  
    - text: str
    - is_primary: bool (optional, default True)
    - speaker: str (optional, speaker ID)
    
    Args:
        segments: List of segment dictionaries
        output_path: Path to write ASS file
        play_res_x: Video width
        play_res_y: Video height
        font_name: Font family
        font_size: Font size
        margin_v: Vertical margin
        margin_l: Left margin
        margin_r: Right margin
        enable_zoom_animation: Enable zoom-in effect on subtitles
        
    Returns:
        Path to output file
    """
    # Generate header
    ass_content = generate_ass_header(
        play_res_x=play_res_x,
        play_res_y=play_res_y,
        font_name=font_name,
        font_size=font_size,
        margin_v=margin_v,
        margin_l=margin_l,
        margin_r=margin_r,
    )
    
    # Build speaker index map (non-primary speakers get colors in order of appearance)
    speaker_indices = {}
    next_index = 0
    for seg in segments:
        speaker = seg.get('speaker', '')
        is_primary = seg.get('is_primary', True)
        if speaker and not is_primary and speaker not in speaker_indices:
            speaker_indices[speaker] = next_index
            next_index += 1
    
    # Generate dialogue lines
    for seg in segments:
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        text = seg.get('text', '').strip()
        is_primary = seg.get('is_primary', True)
        speaker = seg.get('speaker', '')
        speaker_index = speaker_indices.get(speaker, 0)
        
        if text:  # Skip empty segments
            dialogue = generate_ass_dialogue(
                start=start,
                end=end,
                text=text.upper(),  # ALL CAPS for social media style
                is_primary=is_primary,
                speaker=speaker,
                speaker_index=speaker_index,
                enable_zoom_animation=enable_zoom_animation,
            )
            ass_content += dialogue + "\n"
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    # Log speaker color assignments
    if speaker_indices:
        print(f"📝 Speaker colors: Primary=white, Others={list(speaker_indices.keys())}")
    
    print(f"📝 Generated ASS subtitles: {output_path} ({len(segments)} segments)")
    return output_path


def segments_to_ass_with_diarization(
    segments: list[dict],
    output_path: str,
    has_diarization: bool = False,
    **kwargs
) -> str:
    """
    Convert segments to ASS, with proper styling based on diarization availability.
    
    If diarization was performed, segments will have 'is_primary' set correctly.
    If not, all segments default to primary (white) styling.
    
    Args:
        segments: List of segment dictionaries
        output_path: Path to write ASS file
        has_diarization: Whether diarization was performed
        **kwargs: Additional arguments passed to segments_to_ass
        
    Returns:
        Path to output file
    """
    if not has_diarization:
        # Ensure all segments are marked as primary when no diarization
        for seg in segments:
            seg['is_primary'] = True
    
    return segments_to_ass(segments, output_path, **kwargs)

