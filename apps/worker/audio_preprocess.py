"""Audio preprocessing for Stream2Short Worker.

Extracts and normalizes audio from video for optimal transcription and diarization.

Features:
- Consistent 16kHz mono WAV output (required by Whisper and pyannote)
- Loudness normalization (EBU R128 loudnorm filter)
- Single extraction used for both transcription and diarization

Benefits:
- Improved speech recognition accuracy
- Better speaker separation in diarization
- Consistent audio levels across clips
"""

import subprocess
import os
from pathlib import Path
from typing import Optional

from config import config


class AudioPreprocessError(Exception):
    """Error during audio preprocessing."""
    pass


def extract_and_normalize_audio(
    video_path: str,
    output_path: str,
    sample_rate: int = 16000,
    channels: int = 1,
    enable_loudnorm: bool = True,
) -> str:
    """
    Extract audio from video with normalization for optimal ASR/diarization.
    
    Uses FFmpeg's loudnorm filter (EBU R128) for consistent loudness,
    and outputs 16kHz mono WAV as required by Whisper and pyannote.
    
    Args:
        video_path: Path to input video file
        output_path: Path to output WAV file
        sample_rate: Output sample rate (default 16000 for Whisper/pyannote)
        channels: Output channels (default 1 = mono)
        enable_loudnorm: Apply EBU R128 loudness normalization
        
    Returns:
        Path to extracted and normalized audio file
        
    Raises:
        AudioPreprocessError: If extraction fails
    """
    print(f"🔊 Extracting audio: {video_path}")
    
    # Build FFmpeg filter chain
    audio_filters = []
    
    # Loudness normalization (EBU R128)
    # Target: -16 LUFS (good for speech), LRA 11, True Peak -1.5dB
    if enable_loudnorm:
        audio_filters.append(
            "loudnorm=I=-16:LRA=11:TP=-1.5:print_format=summary"
        )
        print("   📊 Applying loudness normalization (EBU R128)")
    
    # High-pass filter to remove low-frequency noise (below 80Hz)
    # This helps both transcription and diarization
    audio_filters.append("highpass=f=80")
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", video_path,
        "-vn",  # No video
    ]
    
    # Add audio filters if any
    if audio_filters:
        cmd.extend(["-af", ",".join(audio_filters)])
    
    # Output format: 16-bit PCM WAV, 16kHz, mono
    cmd.extend([
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        output_path
    ])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check if output was created
        if not os.path.exists(output_path):
            raise AudioPreprocessError("Output file was not created")
        
        file_size = os.path.getsize(output_path)
        duration_estimate = file_size / (sample_rate * channels * 2)  # 16-bit = 2 bytes
        
        print(f"   ✅ Audio extracted: {output_path}")
        print(f"   📊 Size: {file_size / 1024:.1f}KB, Duration: ~{duration_estimate:.1f}s")
        
        # Parse loudnorm output if available
        if enable_loudnorm and "Output Integrated" in result.stderr:
            _parse_loudnorm_stats(result.stderr)
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        raise AudioPreprocessError(f"FFmpeg audio extraction failed: {error_msg}")
    except Exception as e:
        raise AudioPreprocessError(f"Audio extraction failed: {e}")


def _parse_loudnorm_stats(ffmpeg_output: str) -> None:
    """Parse and log loudnorm statistics from FFmpeg output."""
    try:
        lines = ffmpeg_output.split('\n')
        for line in lines:
            if 'Input Integrated' in line or 'Output Integrated' in line:
                print(f"   📊 {line.strip()}")
            elif 'Input LRA' in line or 'Output LRA' in line:
                print(f"   📊 {line.strip()}")
    except Exception:
        pass  # Non-critical, just skip logging


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_audio_stats(audio_path: str) -> dict:
    """
    Get audio statistics using FFmpeg.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dict with audio statistics (sample_rate, channels, duration, etc.)
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "stream=sample_rate,channels,duration,bit_rate",
            "-show_entries", "format=duration,size",
            "-of", "json",
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}


def preprocess_audio_for_pipeline(
    video_path: str,
    temp_dir: str,
) -> str:
    """
    Preprocess audio from video for the entire pipeline.
    
    This is the main entry point - extracts and normalizes audio once,
    to be used by both transcription (Whisper) and diarization (pyannote).
    
    Args:
        video_path: Path to input video file
        temp_dir: Temporary directory for output
        
    Returns:
        Path to preprocessed audio file
        
    Raises:
        AudioPreprocessError: If preprocessing fails
    """
    output_path = os.path.join(temp_dir, "preprocessed_audio.wav")
    
    return extract_and_normalize_audio(
        video_path=video_path,
        output_path=output_path,
        sample_rate=16000,  # Required by Whisper and pyannote
        channels=1,         # Mono for speech processing
        enable_loudnorm=config.ENABLE_AUDIO_NORMALIZATION,
    )

