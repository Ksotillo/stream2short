"""ASS subtitle writer with speaker-based styling for Stream2Short Worker.

Generates .ass (Advanced SubStation Alpha) subtitle files with different
styles for primary and secondary speakers.

Colors:
- Primary speaker (white): &H00FFFFFF
- Other speakers (yellow): &H0000FFFF (BGR format)
"""

from typing import Optional


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
    font_name: str = "Montserrat SemiBold",
    font_size: int = 18,
    margin_v: int = 50,
    margin_l: int = 20,
    margin_r: int = 20,
    outline: int = 2,
    shadow: int = 1,
) -> str:
    """
    Generate ASS file header with script info and styles.
    
    Styles:
    - Primary: White text for main speaker
    - Other: Yellow text for secondary speakers
    
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
    # ASS color format is &HAABBGGRR (Alpha, Blue, Green, Red)
    # For subtitle text, we typically use &H00BBGGRR (no alpha)
    white = "&H00FFFFFF"      # Pure white
    yellow = "&H0000FFFF"     # Yellow (BGR: 00, FF, FF = Blue=0, Green=255, Red=255)
    black = "&H00000000"      # Black for outline
    shadow_color = "&H80000000"  # Semi-transparent black for shadow
    
    header = f"""[Script Info]
Title: Stream2Short Subtitles
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Primary,{font_name},{font_size},{white},{white},{black},{shadow_color},1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_l},{margin_r},{margin_v},1
Style: Other,{font_name},{font_size},{yellow},{yellow},{black},{shadow_color},1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_l},{margin_r},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header


def generate_ass_dialogue(
    start: float,
    end: float,
    text: str,
    is_primary: bool = True,
    layer: int = 0,
) -> str:
    """
    Generate a single ASS dialogue line.
    
    Args:
        start: Start time in seconds
        end: End time in seconds
        text: Subtitle text
        is_primary: True for primary speaker (white), False for other (yellow)
        layer: Layer number (default 0)
        
    Returns:
        ASS dialogue line
    """
    style = "Primary" if is_primary else "Other"
    start_str = format_ass_time(start)
    end_str = format_ass_time(end)
    
    # Escape special ASS characters
    # Newlines use \N, literal backslash uses \\
    text_escaped = text.replace("\\", "\\\\").replace("\n", "\\N")
    
    return f"Dialogue: {layer},{start_str},{end_str},{style},,0,0,0,,{text_escaped}"


def segments_to_ass(
    segments: list[dict],
    output_path: str,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_name: str = "Montserrat SemiBold",
    font_size: int = 18,
    margin_v: int = 50,
    margin_l: int = 20,
    margin_r: int = 20,
) -> str:
    """
    Convert transcript segments to ASS subtitle file.
    
    Segments should have:
    - start: float (seconds)
    - end: float (seconds)  
    - text: str
    - is_primary: bool (optional, default True)
    
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
    
    # Generate dialogue lines
    for seg in segments:
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        text = seg.get('text', '').strip()
        is_primary = seg.get('is_primary', True)
        
        if text:  # Skip empty segments
            dialogue = generate_ass_dialogue(
                start=start,
                end=end,
                text=text.upper(),  # ALL CAPS for social media style
                is_primary=is_primary,
            )
            ass_content += dialogue + "\n"
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    print(f"📝 Generated ASS subtitles: {output_path}")
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

