"""Transcription using faster-whisper for Stream2Short Worker."""

import os
from pathlib import Path
from typing import Optional
from faster_whisper import WhisperModel
from config import config

# Global model instance (loaded lazily)
_model: Optional[WhisperModel] = None


def get_model() -> WhisperModel:
    """Get or create the Whisper model instance."""
    global _model
    
    if _model is None:
        print(f"🔊 Loading Whisper model: {config.WHISPER_MODEL}")
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device="cpu",  # Use GPU if available: "cuda"
            compute_type="int8",  # Faster on CPU
        )
        print("✅ Whisper model loaded")
    
    return _model


def transcribe_video(video_path: str, output_srt_path: str) -> str:
    """
    Transcribe video audio to SRT subtitles.
    
    Args:
        video_path: Path to input video file
        output_srt_path: Path to output SRT file
        
    Returns:
        Path to generated SRT file
    """
    model = get_model()
    
    print(f"🎙️ Transcribing: {video_path}")
    
    # Transcribe with word-level timestamps
    segments, info = model.transcribe(
        video_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,  # Filter out silence
    )
    
    print(f"📝 Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    # Convert to SRT format
    segments_list = list(segments)
    print(f"📝 Found {len(segments_list)} speech segments")
    
    srt_content = segments_to_srt(segments_list)
    
    # Write SRT file
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    # Verify file was written
    file_size = os.path.getsize(output_srt_path)
    print(f"✅ Subtitles saved to {output_srt_path} ({file_size} bytes)")
    
    if file_size == 0:
        print("⚠️ Warning: No speech detected in video, subtitle file is empty")
    
    return output_srt_path


def segments_to_srt(segments: list) -> str:
    """
    Convert faster-whisper segments to SRT format.
    
    Args:
        segments: List of transcription segments
        
    Returns:
        SRT formatted string
    """
    srt_lines = []
    
    for i, segment in enumerate(segments, start=1):
        start_time = format_srt_time(segment.start)
        end_time = format_srt_time(segment.end)
        text = segment.text.strip()
        
        if text:
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between entries
    
    return "\n".join(srt_lines)


def format_srt_time(seconds: float) -> str:
    """
    Format seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        SRT formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_ass_subtitles(srt_path: str, output_ass_path: str) -> str:
    """
    Convert SRT to ASS format with custom styling for vertical video.
    
    Args:
        srt_path: Path to input SRT file
        output_ass_path: Path to output ASS file
        
    Returns:
        Path to generated ASS file
    """
    # ASS header with styling optimized for vertical video
    ass_header = """[Script Info]
Title: Stream2Short Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,2,2,40,40,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Parse SRT and convert to ASS events
    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()
    
    events = []
    entries = srt_content.strip().split("\n\n")
    
    for entry in entries:
        lines = entry.strip().split("\n")
        if len(lines) >= 3:
            # Parse timestamp line
            time_line = lines[1]
            if " --> " in time_line:
                start, end = time_line.split(" --> ")
                start_ass = srt_time_to_ass(start.strip())
                end_ass = srt_time_to_ass(end.strip())
                
                # Join text lines
                text = " ".join(lines[2:])
                # Escape special characters for ASS
                text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
                
                events.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}")
    
    # Write ASS file
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        f.write("\n".join(events))
    
    print(f"✅ ASS subtitles saved to {output_ass_path}")
    
    return output_ass_path


def srt_time_to_ass(srt_time: str) -> str:
    """
    Convert SRT timestamp (HH:MM:SS,mmm) to ASS format (H:MM:SS.cc).
    
    Args:
        srt_time: SRT formatted timestamp
        
    Returns:
        ASS formatted timestamp
    """
    # Replace comma with dot and trim milliseconds to centiseconds
    parts = srt_time.replace(",", ".").split(".")
    time_part = parts[0]
    centis = parts[1][:2] if len(parts) > 1 else "00"
    
    # Remove leading zero from hours if present
    h, m, s = time_part.split(":")
    h = str(int(h))
    
    return f"{h}:{m}:{s}.{centis}"

