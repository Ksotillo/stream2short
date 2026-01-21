"""Groq API transcription for Stream2Short Worker.

Uses Groq's blazing-fast Whisper API for near-instant transcription.
Falls back to local faster-whisper if Groq is not configured.

Now uses smart chunking for natural subtitle breaks instead of fixed word counts.
"""

import os
from pathlib import Path
from typing import Optional
from groq import Groq
from config import config
from smart_chunker import smart_chunk_transcript


_groq_client: Optional[Groq] = None


def is_groq_available() -> bool:
    """Check if Groq API is configured."""
    return bool(config.GROQ_API_KEY)


def get_groq_client() -> Groq:
    """Get or create Groq client instance."""
    global _groq_client
    
    if _groq_client is None:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not configured")
        
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
        print(f"✅ Groq client initialized (model: {config.GROQ_MODEL})")
    
    return _groq_client


def transcribe_with_groq(
    audio_path: str,
    enable_smart_chunking: bool = True,
    enable_emphasis: bool = False,
) -> tuple[list[dict], str]:
    """
    Transcribe audio using Groq's Whisper API with smart chunking.
    
    Uses intelligent subtitle segmentation that respects:
    - Punctuation as natural break points
    - Max ~30 characters per line
    - Duration: 0.8–2.2 seconds per chunk
    - Optional keyword emphasis
    
    Args:
        audio_path: Path to audio file (wav, mp3, flac, etc.)
        enable_smart_chunking: Use smart chunking (True) or legacy 3-word (False)
        enable_emphasis: Highlight keywords in subtitles
        
    Returns:
        Tuple of (segments list, transcript text)
        Each segment has: start, end, text keys
    """
    client = get_groq_client()
    
    print(f"🚀 Transcribing with Groq ({config.GROQ_MODEL}): {audio_path}")
    
    # Check file size (Groq limit: 25MB free tier)
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"📁 Audio file size: {file_size_mb:.1f} MB")
    
    if file_size_mb > 25:
        raise ValueError(f"Audio file too large for Groq ({file_size_mb:.1f}MB > 25MB limit)")
    
    # Call Groq API with word-level timestamps
    # Note: Don't set language param - let Whisper auto-detect for multilingual support
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=config.GROQ_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            # language parameter omitted - Whisper auto-detects the language
            temperature=0.0,
        )
    
    print(f"📝 Groq transcription complete!")
    
    # Collect word-level data for smart chunking
    all_words = []
    full_text_parts = []
    
    # Check if we have word-level timestamps
    if hasattr(transcription, 'words') and transcription.words:
        words = transcription.words
        print(f"📝 Got {len(words)} word-level timestamps")
        
        # Normalize word data
        for w in words:
            word_text = w.word.strip() if hasattr(w, 'word') else w.get('word', '').strip()
            start = w.start if hasattr(w, 'start') else w.get('start', 0)
            end = w.end if hasattr(w, 'end') else w.get('end', start + 0.3)
            
            if word_text:
                all_words.append({
                    'word': word_text,
                    'start': float(start),
                    'end': float(end),
                })
                full_text_parts.append(word_text)
    
    elif hasattr(transcription, 'segments') and transcription.segments:
        # Fallback to segment-level timestamps - synthesize word timing
        segments = transcription.segments
        print(f"📝 Got {len(segments)} segment-level timestamps (synthesizing word timing)")
        
        for segment in segments:
            text = segment.text.strip() if hasattr(segment, 'text') else segment.get('text', '').strip()
            if not text:
                continue
            
            start = segment.start if hasattr(segment, 'start') else segment.get('start', 0)
            end = segment.end if hasattr(segment, 'end') else segment.get('end', start + 1)
            
            words = text.split()
            if not words:
                continue
                
            duration = end - start
            time_per_word = duration / len(words)
            
            for j, word_text in enumerate(words):
                word_start = start + (j * time_per_word)
                word_end = start + ((j + 1) * time_per_word)
                
                all_words.append({
                    'word': word_text,
                    'start': float(word_start),
                    'end': float(word_end),
                })
                full_text_parts.append(word_text)
    
    else:
        # Last resort: just use the text with estimated timing
        text = transcription.text if hasattr(transcription, 'text') else ""
        print(f"⚠️ No timestamps available, estimating timing")
        
        if text:
            words = text.split()
            # Estimate ~0.3 seconds per word
            for j, word_text in enumerate(words):
                word_start = j * 0.3
                word_end = (j + 1) * 0.3
                
                all_words.append({
                    'word': word_text,
                    'start': float(word_start),
                    'end': float(word_end),
                })
                full_text_parts.append(word_text)
    
    # Apply smart chunking or legacy chunking
    if enable_smart_chunking and all_words:
        result_segments = smart_chunk_transcript(
            words=all_words,
            enable_emphasis=enable_emphasis,
        )
        print(f"✅ Smart chunking: {len(all_words)} words → {len(result_segments)} chunks")
    else:
        # Legacy 3-word chunking fallback
        result_segments = _legacy_chunk_words(all_words, words_per_chunk=3)
        print(f"✅ Legacy chunking: {len(all_words)} words → {len(result_segments)} chunks")
    
    full_text = " ".join(full_text_parts)
    
    return result_segments, full_text


def _legacy_chunk_words(words: list[dict], words_per_chunk: int = 3) -> list[dict]:
    """
    Legacy fixed-word-count chunking (for backwards compatibility).
    
    Args:
        words: List of word dicts with 'word', 'start', 'end'
        words_per_chunk: Number of words per subtitle
        
    Returns:
        List of segment dicts
    """
    result_segments = []
    
    for i in range(0, len(words), words_per_chunk):
        chunk = words[i:i + words_per_chunk]
        if not chunk:
            continue
        
        text = " ".join(w.get('word', '') for w in chunk).strip()
        
        if text:
            result_segments.append({
                'start': chunk[0].get('start', 0),
                'end': chunk[-1].get('end', chunk[0].get('start', 0) + 0.5),
                'text': text,
            })
    
    return result_segments


def transcribe_with_segments(
    video_path: str,
    audio_path: str = None,
    enable_smart_chunking: bool = True,
    enable_emphasis: bool = False,
) -> tuple[list[dict], str]:
    """
    Transcribe using Groq if available, otherwise fall back to local Whisper.
    
    This is the main entry point that automatically chooses the best method.
    Now uses smart chunking by default for natural subtitle breaks.
    
    Args:
        video_path: Path to video file (used if audio_path not provided)
        audio_path: Optional path to preprocessed audio file
        enable_smart_chunking: Use intelligent chunking (default True)
        enable_emphasis: Highlight keywords (default False)
        
    Returns:
        Tuple of (segments list, transcript text)
    """
    input_path = audio_path if audio_path else video_path
    
    if is_groq_available():
        try:
            return transcribe_with_groq(
                input_path,
                enable_smart_chunking=enable_smart_chunking,
                enable_emphasis=enable_emphasis,
            )
        except Exception as e:
            print(f"⚠️ Groq transcription failed: {e}")
            print("🔄 Falling back to local Whisper...")
    else:
        print("ℹ️ GROQ_API_KEY not set, using local Whisper")
    
    # Fall back to local transcription
    from transcribe import transcribe_video_with_segments
    return transcribe_video_with_segments(
        video_path,
        audio_path=audio_path,
        enable_smart_chunking=enable_smart_chunking,
        enable_emphasis=enable_emphasis,
    )

