"""Speaker diarization using pyannote.audio for Stream2Short Worker.

This module provides speaker diarization to identify who is speaking when,
allowing subtitles to be colored by speaker.

Requirements:
- HF_TOKEN environment variable with a Hugging Face read token
- User must accept model conditions for:
  - pyannote/segmentation-3.0
  - pyannote/speaker-diarization-3.1
"""

import subprocess
import os
from dataclasses import dataclass
from typing import Optional

from config import config


@dataclass
class SpeakerTurn:
    """A single speaker turn with timing."""
    start: float
    end: float
    speaker: str


@dataclass 
class DiarizationResult:
    """Result of speaker diarization."""
    turns: list[SpeakerTurn]
    primary_speaker: str
    speakers: list[str]


class DiarizationError(Exception):
    """Error during diarization."""
    pass


def check_diarization_available() -> tuple[bool, str]:
    """
    Check if diarization is available and properly configured.
    
    Returns:
        Tuple of (is_available, reason_if_not)
    """
    if not config.ENABLE_DIARIZATION:
        return False, "ENABLE_DIARIZATION is false"
    
    if not config.HF_TOKEN:
        return False, "HF_TOKEN is not set. Required for gated pyannote models."
    
    try:
        import torch
        import pyannote.audio
        return True, "OK"
    except ImportError as e:
        return False, f"Missing dependency: {e}"


def extract_audio_for_diarization(video_path: str, output_path: str) -> str:
    """
    Extract mono 16kHz WAV audio from video for diarization.
    
    Args:
        video_path: Path to input video
        output_path: Path to output WAV file
        
    Returns:
        Path to extracted audio file
        
    Raises:
        DiarizationError: If extraction fails
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # Mono
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise DiarizationError(f"Failed to extract audio: {e.stderr}")


def run_diarization(audio_path: str) -> list[SpeakerTurn]:
    """
    Run speaker diarization on audio file using pyannote.audio.
    
    Args:
        audio_path: Path to WAV audio file (16kHz mono recommended)
        
    Returns:
        List of SpeakerTurn objects
        
    Raises:
        DiarizationError: If diarization fails
    """
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise DiarizationError(f"pyannote.audio not installed: {e}")
    
    if not config.HF_TOKEN:
        raise DiarizationError(
            "HF_TOKEN not set. Required for gated pyannote models. "
            "Ensure you have accepted model conditions for "
            "pyannote/segmentation-3.0 and pyannote/speaker-diarization-3.1"
        )
    
    print(f"🎤 Loading diarization model: {config.DIARIZATION_MODEL}")
    
    try:
        pipeline = Pipeline.from_pretrained(
            config.DIARIZATION_MODEL,
            token=config.HF_TOKEN
        )
    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            raise DiarizationError(
                f"Failed to load diarization model (access denied). "
                f"Ensure HF_TOKEN is valid and you have accepted model conditions for "
                f"pyannote/segmentation-3.0 and pyannote/speaker-diarization-3.1 at huggingface.co. "
                f"Error: {e}"
            )
        raise DiarizationError(f"Failed to load diarization model: {e}")
    
    print(f"🎤 Running diarization on: {audio_path}")
    
    try:
        diarization = pipeline(audio_path)
    except Exception as e:
        raise DiarizationError(f"Diarization failed: {e}")
    
    # Convert pyannote output to our format
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(SpeakerTurn(
            start=turn.start,
            end=turn.end,
            speaker=speaker
        ))
    
    print(f"🎤 Found {len(turns)} speaker turns, {len(set(t.speaker for t in turns))} unique speakers")
    return turns


def determine_primary_speaker(
    turns: list[SpeakerTurn],
    strategy: str = "most_time"
) -> str:
    """
    Determine the primary speaker based on strategy.
    
    Args:
        turns: List of speaker turns
        strategy: "most_time" (default) or "first"
        
    Returns:
        Speaker ID of primary speaker
    """
    if not turns:
        return "SPEAKER_00"
    
    if strategy == "first":
        # Speaker who talks first
        return turns[0].speaker
    
    # Default: most_time - speaker with most total duration
    speaker_durations: dict[str, float] = {}
    for turn in turns:
        duration = turn.end - turn.start
        speaker_durations[turn.speaker] = speaker_durations.get(turn.speaker, 0) + duration
    
    primary = max(speaker_durations, key=speaker_durations.get)
    print(f"🎤 Primary speaker ({strategy}): {primary} ({speaker_durations[primary]:.1f}s)")
    return primary


def assign_speakers_to_segments(
    segments: list[dict],
    turns: list[SpeakerTurn],
    primary_speaker: str
) -> list[dict]:
    """
    Assign speaker labels to transcript segments based on overlap with diarization.
    
    Args:
        segments: List of transcript segments with 'start', 'end', 'text' keys
        turns: List of SpeakerTurn from diarization
        primary_speaker: ID of the primary speaker
        
    Returns:
        Segments with added 'speaker' and 'is_primary' keys
    """
    if not turns:
        # No diarization data - all segments are primary
        for seg in segments:
            seg['speaker'] = primary_speaker
            seg['is_primary'] = True
        return segments
    
    for seg in segments:
        seg_start = seg.get('start', 0)
        seg_end = seg.get('end', 0)
        
        # Find speaker with maximum overlap
        best_speaker = None
        best_overlap = 0
        
        for turn in turns:
            # Calculate overlap
            overlap_start = max(seg_start, turn.start)
            overlap_end = min(seg_end, turn.end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
        
        # Assign speaker (fallback to primary if no overlap)
        seg['speaker'] = best_speaker or primary_speaker
        seg['is_primary'] = seg['speaker'] == primary_speaker
    
    return segments


def merge_adjacent_segments(
    segments: list[dict],
    max_gap: float = 0.25
) -> list[dict]:
    """
    Merge consecutive segments with same speaker if gap is small.
    
    Args:
        segments: List of segments with 'start', 'end', 'text', 'speaker' keys
        max_gap: Maximum gap in seconds to merge (default 0.25s)
        
    Returns:
        Merged segments
    """
    if not segments:
        return segments
    
    merged = [segments[0].copy()]
    
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg.get('start', 0) - prev.get('end', 0)
        
        # Merge if same speaker and small gap
        if (seg.get('speaker') == prev.get('speaker') and 
            gap <= max_gap and gap >= 0):
            # Extend previous segment
            prev['end'] = seg.get('end', prev['end'])
            prev['text'] = prev.get('text', '') + ' ' + seg.get('text', '')
        else:
            merged.append(seg.copy())
    
    return merged


def diarize_video(
    video_path: str,
    temp_dir: str,
    channel_settings: Optional[dict] = None
) -> Optional[DiarizationResult]:
    """
    Full diarization pipeline for a video.
    
    Args:
        video_path: Path to video file
        temp_dir: Directory for temporary files
        channel_settings: Optional channel settings dict (may contain enable_diarization)
        
    Returns:
        DiarizationResult if successful, None if disabled or failed
    """
    # Check if diarization is enabled (channel setting overrides global)
    enable = config.ENABLE_DIARIZATION
    if channel_settings and 'enable_diarization' in channel_settings:
        enable = bool(channel_settings['enable_diarization'])
    
    if not enable:
        print("🎤 Diarization disabled (ENABLE_DIARIZATION=false)")
        return None
    
    # Check prerequisites
    available, reason = check_diarization_available()
    if not available:
        print(f"⚠️ Diarization not available: {reason}")
        return None
    
    try:
        # Extract audio
        audio_path = os.path.join(temp_dir, "diarization_audio.wav")
        print("🎤 Extracting audio for diarization...")
        extract_audio_for_diarization(video_path, audio_path)
        
        # Run diarization
        turns = run_diarization(audio_path)
        
        if not turns:
            print("⚠️ No speaker turns detected")
            return None
        
        # Determine primary speaker
        primary = determine_primary_speaker(turns, config.PRIMARY_SPEAKER_STRATEGY)
        speakers = list(set(t.speaker for t in turns))
        
        return DiarizationResult(
            turns=turns,
            primary_speaker=primary,
            speakers=speakers
        )
        
    except DiarizationError as e:
        print(f"⚠️ Diarization failed: {e}")
        print("   Continuing without speaker diarization...")
        return None
    except Exception as e:
        print(f"⚠️ Unexpected diarization error: {e}")
        print("   Continuing without speaker diarization...")
        return None

