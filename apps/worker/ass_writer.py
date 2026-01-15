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
# Colors are assigned based on gender detection when available
SPEAKER_COLORS = {
    "primary": "&H00FFFFFF",    # White - main speaker (any gender)
}

# Colors for masculine voices (cooler tones)
MASCULINE_COLORS = [
    "&H00FFFF00",  # Cyan
    "&H00FF9900",  # Light Blue  
    "&H0000FF00",  # Lime green
    "&H00FFCC00",  # Sky blue
]

# Colors for feminine voices (warmer tones)
FEMININE_COLORS = [
    "&H00FF77FF",  # Pink
    "&H00FF00FF",  # Magenta
    "&H009966FF",  # Coral/Salmon
    "&H00CC99FF",  # Light Pink
]

# Fallback colors (when gender is unknown)
NEUTRAL_COLORS = [
    "&H0000FFFF",  # Yellow
    "&H000099FF",  # Orange
    "&H00FF0000",  # Blue
    "&H0000FF99",  # Light green
]


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
    speaker_genders: dict = None,  # Dict of speaker_id -> gender
) -> str:
    """
    Generate ASS file header with script info and styles for multiple speakers.
    
    Colors are assigned based on gender:
    - Primary speaker: White
    - Masculine voices: Cyan, Blue, Green tones
    - Feminine voices: Pink, Magenta, Coral tones
    - Unknown: Yellow, Orange tones
    
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
        speaker_genders: Dict mapping speaker IDs to gender ("masculine"/"feminine"/"unknown")
        
    Returns:
        ASS header string
    """
    black = "&H00000000"  # Black for outline
    
    if speaker_genders is None:
        speaker_genders = {}
    
    # Build styles - Primary plus gender-based colors
    styles = []
    
    # Primary style (white)
    styles.append(
        f"Style: Primary,{font_name},{font_size},&H00FFFFFF,&H00FFFFFF,"
        f"{black},{black},1,0,0,0,100,100,0,0,1,{outline},{shadow},2,"
        f"{margin_l},{margin_r},{margin_v},1"
    )
    
    # Add styles for masculine speakers
    for i, color in enumerate(MASCULINE_COLORS):
        styles.append(
            f"Style: Masculine{i},{font_name},{font_size},{color},{color},"
            f"{black},{black},1,0,0,0,100,100,0,0,1,{outline},{shadow},2,"
            f"{margin_l},{margin_r},{margin_v},1"
        )
    
    # Add styles for feminine speakers
    for i, color in enumerate(FEMININE_COLORS):
        styles.append(
            f"Style: Feminine{i},{font_name},{font_size},{color},{color},"
            f"{black},{black},1,0,0,0,100,100,0,0,1,{outline},{shadow},2,"
            f"{margin_l},{margin_r},{margin_v},1"
        )
    
    # Add styles for unknown gender speakers
    for i, color in enumerate(NEUTRAL_COLORS):
        styles.append(
            f"Style: Neutral{i},{font_name},{font_size},{color},{color},"
            f"{black},{black},1,0,0,0,100,100,0,0,1,{outline},{shadow},2,"
            f"{margin_l},{margin_r},{margin_v},1"
        )
    
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


def get_speaker_style(
    speaker: str,
    is_primary: bool,
    speaker_index: int = 0,
    speaker_info: dict = None,  # Dict of speaker_id -> SpeakerInfo
) -> str:
    """
    Get the ASS style name for a speaker based on gender.
    
    Colors assigned by gender:
    - Primary speaker: White (regardless of gender)
    - Masculine voices: Cyan, Blue, Green tones
    - Feminine voices: Pink, Magenta, Coral tones
    - Unknown: Yellow, Orange tones
    
    Args:
        speaker: Speaker ID from diarization
        is_primary: True if this is the primary (most speaking) speaker
        speaker_index: Index of non-primary speakers (0-based)
        speaker_info: Dict mapping speaker IDs to SpeakerInfo (with gender)
        
    Returns:
        ASS style name
    """
    if is_primary:
        return "Primary"
    
    if speaker_info is None:
        speaker_info = {}
    
    # Get gender for this speaker
    info = speaker_info.get(speaker)
    
    # Handle both SpeakerInfo objects and dicts
    if info:
        gender = info.gender if hasattr(info, 'gender') else info.get('gender', 'unknown')
    else:
        gender = "unknown"
    
    # Use speaker_index to cycle through colors of the same gender type
    if gender == "feminine":
        style_index = speaker_index % len(FEMININE_COLORS)
        return f"Feminine{style_index}"
    elif gender == "masculine":
        style_index = speaker_index % len(MASCULINE_COLORS)
        return f"Masculine{style_index}"
    else:
        style_index = speaker_index % len(NEUTRAL_COLORS)
        return f"Neutral{style_index}"


def generate_ass_dialogue(
    start: float,
    end: float,
    text: str,
    is_primary: bool = True,
    speaker: str = "",
    speaker_index: int = 0,
    speaker_info: dict = None,
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
        speaker_info: Dict mapping speaker IDs to SpeakerInfo (for gender-based colors)
        layer: Layer number (default 0)
        enable_zoom_animation: Add zoom-in effect on text appearance
        
    Returns:
        ASS dialogue line
    """
    style = get_speaker_style(speaker, is_primary, speaker_index, speaker_info)
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
    speaker_info: dict = None,  # Dict of speaker_id -> SpeakerInfo (for gender-based colors)
) -> str:
    """
    Convert transcript segments to ASS subtitle file with multi-color speakers.
    
    Colors are assigned based on speaker gender:
    - Primary speaker: White (any gender)
    - Masculine voices: Cyan, Blue, Green tones
    - Feminine voices: Pink, Magenta, Coral tones
    - Unknown: Yellow, Orange tones
    
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
        speaker_info: Dict mapping speaker IDs to SpeakerInfo (gender, pitch, etc.)
        
    Returns:
        Path to output file
    """
    if speaker_info is None:
        speaker_info = {}
    
    # Generate header
    ass_content = generate_ass_header(
        play_res_x=play_res_x,
        play_res_y=play_res_y,
        font_name=font_name,
        font_size=font_size,
        margin_v=margin_v,
        margin_l=margin_l,
        margin_r=margin_r,
        speaker_genders={k: (v.gender if hasattr(v, 'gender') else v.get('gender', 'unknown')) for k, v in speaker_info.items()},
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
                speaker_info=speaker_info,
                enable_zoom_animation=enable_zoom_animation,
            )
            ass_content += dialogue + "\n"
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    # Log speaker color assignments with gender
    if speaker_indices:
        speaker_details = []
        for spk, idx in speaker_indices.items():
            info = speaker_info.get(spk)
            gender = info.gender if info and hasattr(info, 'gender') else (info.get('gender', '?') if info else '?')
            speaker_details.append(f"{spk}({gender})")
        print(f"📝 Speaker colors: Primary=white, Others={speaker_details}")
    
    print(f"📝 Generated ASS subtitles: {output_path} ({len(segments)} segments)")
    return output_path


def segments_to_ass_with_diarization(
    segments: list[dict],
    output_path: str,
    has_diarization: bool = False,
    speaker_info: dict = None,
    **kwargs
) -> str:
    """
    Convert segments to ASS, with proper styling based on diarization availability.
    
    If diarization was performed:
    - segments will have 'is_primary' set correctly
    - speaker_info provides gender for color assignment:
      - Masculine: Cyan, Blue, Green tones
      - Feminine: Pink, Magenta, Coral tones
      - Unknown: Yellow, Orange tones
    
    If not, all segments default to primary (white) styling.
    
    Args:
        segments: List of segment dictionaries
        output_path: Path to write ASS file
        has_diarization: Whether diarization was performed
        speaker_info: Dict mapping speaker IDs to SpeakerInfo (from DiarizationResult)
        **kwargs: Additional arguments passed to segments_to_ass
        
    Returns:
        Path to output file
    """
    if not has_diarization:
        # Ensure all segments are marked as primary when no diarization
        for seg in segments:
            seg['is_primary'] = True
    
    return segments_to_ass(segments, output_path, speaker_info=speaker_info, **kwargs)

