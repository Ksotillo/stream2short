"""Speaker diarization using pyannote.audio for Stream2Short Worker.

This module provides speaker diarization to identify who is speaking when,
allowing subtitles to be colored by speaker.

Features:
- Speaker diarization (who speaks when)
- Speaker merging (same voice = same color even with gaps)
- Gender detection via pitch analysis (feminine/masculine colors)

Requirements:
- HF_TOKEN environment variable with a Hugging Face read token
- User must accept model conditions for:
  - pyannote/segmentation-3.0
  - pyannote/speaker-diarization-3.1
"""

import subprocess
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from config import config


# Gender detection thresholds (Hz)
# Female voices typically have higher fundamental frequency
FEMALE_PITCH_THRESHOLD = 165  # Hz - above this is likely female
MALE_PITCH_THRESHOLD = 165    # Hz - below this is likely male


@dataclass
class SpeakerTurn:
    """A single speaker turn with timing."""
    start: float
    end: float
    speaker: str


@dataclass
class SpeakerInfo:
    """Information about a detected speaker."""
    id: str
    gender: str = "unknown"  # "masculine", "feminine", or "unknown"
    avg_pitch: float = 0.0   # Average fundamental frequency in Hz
    total_duration: float = 0.0


@dataclass 
class DiarizationResult:
    """Result of speaker diarization."""
    turns: list[SpeakerTurn]
    primary_speaker: str
    speakers: list[str]
    speaker_info: dict[str, SpeakerInfo] = field(default_factory=dict)


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
        # pyannote.audio 3.x + huggingface-hub <1.0: use use_auth_token=
        pipeline = Pipeline.from_pretrained(
            config.DIARIZATION_MODEL,
            use_auth_token=config.HF_TOKEN,
        )
        print("🎤 Model loaded successfully")
            
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


def analyze_speaker_pitch(
    audio_path: str,
    turns: list[SpeakerTurn],
    speaker_id: str
) -> tuple[float, str]:
    """
    Analyze pitch (F0) for a speaker to estimate gender.
    
    Args:
        audio_path: Path to audio file
        turns: All speaker turns
        speaker_id: ID of speaker to analyze
        
    Returns:
        Tuple of (average_pitch_hz, gender_estimate)
    """
    try:
        import librosa
    except ImportError:
        print("⚠️ librosa not installed, skipping pitch analysis")
        return 0.0, "unknown"
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Collect pitch values from all segments of this speaker
        all_pitches = []
        
        for turn in turns:
            if turn.speaker != speaker_id:
                continue
            
            # Extract segment audio
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            segment = y[start_sample:end_sample]
            
            if len(segment) < sr * 0.1:  # Skip segments < 0.1s
                continue
            
            # Extract fundamental frequency using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment,
                fmin=50,   # Min frequency (Hz)
                fmax=400,  # Max frequency (Hz)
                sr=sr
            )
            
            # Get valid (non-NaN) pitch values
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                all_pitches.extend(valid_f0.tolist())
        
        if not all_pitches:
            return 0.0, "unknown"
        
        # Use median for robustness against outliers
        avg_pitch = float(np.median(all_pitches))
        
        # Classify gender based on pitch
        if avg_pitch > FEMALE_PITCH_THRESHOLD:
            gender = "feminine"
        else:
            gender = "masculine"
        
        return avg_pitch, gender
        
    except Exception as e:
        print(f"⚠️ Pitch analysis failed for {speaker_id}: {e}")
        return 0.0, "unknown"


def merge_similar_speakers(
    turns: list[SpeakerTurn],
    audio_path: str,
    similarity_threshold: float = 0.75
) -> tuple[list[SpeakerTurn], dict[str, str]]:
    """
    Merge speakers that have similar voice embeddings.
    
    This fixes the issue where the same person speaking at different times
    gets assigned different speaker IDs.
    
    Args:
        turns: List of speaker turns
        audio_path: Path to audio file
        similarity_threshold: Cosine similarity threshold for merging (0-1)
        
    Returns:
        Tuple of (updated_turns, speaker_mapping)
        speaker_mapping maps old speaker IDs to new (merged) IDs
    """
    try:
        from pyannote.audio import Model, Inference
        import torch
    except ImportError:
        print("⚠️ Cannot load embedding model, skipping speaker merging")
        return turns, {}
    
    speakers = list(set(t.speaker for t in turns))
    
    if len(speakers) <= 1:
        return turns, {}
    
    try:
        print("🔗 Loading speaker embedding model...")
        embedding_model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=config.HF_TOKEN
        )
        inference = Inference(embedding_model, window="whole")
        
        # Load audio
        import librosa
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Get embeddings for each speaker
        speaker_embeddings = {}
        
        for speaker in speakers:
            # Concatenate all segments for this speaker
            speaker_audio = []
            for turn in turns:
                if turn.speaker == speaker:
                    start_sample = int(turn.start * sr)
                    end_sample = int(turn.end * sr)
                    speaker_audio.extend(y[start_sample:end_sample].tolist())
            
            if len(speaker_audio) < sr * 0.5:  # Need at least 0.5s
                continue
            
            # Get embedding
            speaker_audio_np = np.array(speaker_audio, dtype=np.float32)
            # Save to temp file for inference
            temp_audio_path = audio_path.replace(".wav", f"_{speaker}.wav")
            import soundfile as sf
            sf.write(temp_audio_path, speaker_audio_np, sr)
            
            try:
                embedding = inference(temp_audio_path)
                speaker_embeddings[speaker] = embedding
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
        
        if len(speaker_embeddings) < 2:
            return turns, {}
        
        # Compare embeddings and find similar speakers
        speaker_mapping = {s: s for s in speakers}  # Default: map to self
        processed = set()
        
        speaker_list = list(speaker_embeddings.keys())
        for i, speaker_a in enumerate(speaker_list):
            if speaker_a in processed:
                continue
            
            for speaker_b in speaker_list[i+1:]:
                if speaker_b in processed:
                    continue
                
                # Calculate cosine similarity
                emb_a = speaker_embeddings[speaker_a]
                emb_b = speaker_embeddings[speaker_b]
                
                similarity = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
                
                if similarity > similarity_threshold:
                    print(f"🔗 Merging {speaker_b} → {speaker_a} (similarity: {similarity:.2f})")
                    speaker_mapping[speaker_b] = speaker_a
                    processed.add(speaker_b)
        
        # Apply mapping to turns
        updated_turns = []
        for turn in turns:
            new_speaker = speaker_mapping.get(turn.speaker, turn.speaker)
            updated_turns.append(SpeakerTurn(
                start=turn.start,
                end=turn.end,
                speaker=new_speaker
            ))
        
        return updated_turns, speaker_mapping
        
    except Exception as e:
        print(f"⚠️ Speaker merging failed: {e}")
        return turns, {}


def analyze_all_speakers(
    audio_path: str,
    turns: list[SpeakerTurn]
) -> dict[str, SpeakerInfo]:
    """
    Analyze all speakers to get gender and other info.
    
    Args:
        audio_path: Path to audio file
        turns: List of speaker turns
        
    Returns:
        Dict mapping speaker ID to SpeakerInfo
    """
    speakers = list(set(t.speaker for t in turns))
    speaker_info = {}
    
    for speaker in speakers:
        # Calculate total duration
        total_duration = sum(
            t.end - t.start
            for t in turns
            if t.speaker == speaker
        )
        
        # Analyze pitch for gender
        avg_pitch, gender = analyze_speaker_pitch(audio_path, turns, speaker)
        
        speaker_info[speaker] = SpeakerInfo(
            id=speaker,
            gender=gender,
            avg_pitch=avg_pitch,
            total_duration=total_duration
        )
        
        print(f"🎤 {speaker}: {gender} (pitch={avg_pitch:.0f}Hz, duration={total_duration:.1f}s)")
    
    return speaker_info


def diarize_video(
    video_path: str,
    temp_dir: str,
    channel_settings: Optional[dict] = None
) -> Optional[DiarizationResult]:
    """
    Full diarization pipeline for a video.
    
    Includes:
    - Speaker diarization (who speaks when)
    - Speaker merging (same voice across gaps = same ID)
    - Gender detection via pitch analysis
    
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
        
        initial_speakers = list(set(t.speaker for t in turns))
        print(f"🎤 Initial speakers detected: {len(initial_speakers)}")
        
        # Merge similar speakers (same voice across gaps)
        if len(initial_speakers) > 1:
            print("🔗 Checking for duplicate speakers (same voice, different IDs)...")
            turns, speaker_mapping = merge_similar_speakers(turns, audio_path)
            
            if speaker_mapping:
                merged_count = len([k for k, v in speaker_mapping.items() if k != v])
                if merged_count > 0:
                    print(f"🔗 Merged {merged_count} duplicate speaker(s)")
        
        # Get updated speaker list after merging
        speakers = list(set(t.speaker for t in turns))
        print(f"🎤 Final speakers: {len(speakers)}")
        
        # Determine primary speaker (who talks most)
        primary = determine_primary_speaker(turns, config.PRIMARY_SPEAKER_STRATEGY)
        
        # Analyze each speaker (gender detection via pitch)
        print("🎤 Analyzing speaker voices...")
        speaker_info = analyze_all_speakers(audio_path, turns)
        
        return DiarizationResult(
            turns=turns,
            primary_speaker=primary,
            speakers=speakers,
            speaker_info=speaker_info
        )
        
    except DiarizationError as e:
        print(f"⚠️ Diarization failed: {e}")
        print("   Continuing without speaker diarization...")
        return None
    except Exception as e:
        print(f"⚠️ Unexpected diarization error: {e}")
        print("   Continuing without speaker diarization...")
        return None

