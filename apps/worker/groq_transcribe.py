"""Groq API transcription for Stream2Short Worker.

Uses Groq's blazing-fast Whisper API for near-instant transcription.
Falls back to local faster-whisper if Groq is not configured.
"""

import os
from pathlib import Path
from typing import Optional
from groq import Groq
from config import config


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
    words_per_subtitle: int = 3,
) -> tuple[list[dict], str]:
    """
    Transcribe audio using Groq's Whisper API.
    
    Args:
        audio_path: Path to audio file (wav, mp3, flac, etc.)
        words_per_subtitle: Max words per subtitle chunk
        
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
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=config.GROQ_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language="en",  # Can be made configurable
            temperature=0.0,
        )
    
    print(f"📝 Groq transcription complete!")
    
    # Process the response
    result_segments = []
    full_text_parts = []
    
    # Check if we have word-level timestamps
    if hasattr(transcription, 'words') and transcription.words:
        words = transcription.words
        print(f"📝 Got {len(words)} word-level timestamps")
        
        # Group words into subtitle chunks
        for i in range(0, len(words), words_per_subtitle):
            chunk = words[i:i + words_per_subtitle]
            if not chunk:
                continue
            
            # Build text from chunk
            text = " ".join(w.word.strip() if hasattr(w, 'word') else w.get('word', '').strip() 
                          for w in chunk).strip()
            
            if text:
                # Get timestamps
                first_word = chunk[0]
                last_word = chunk[-1]
                
                start = first_word.start if hasattr(first_word, 'start') else first_word.get('start', 0)
                end = last_word.end if hasattr(last_word, 'end') else last_word.get('end', start + 0.5)
                
                result_segments.append({
                    'start': float(start),
                    'end': float(end),
                    'text': text,
                })
                full_text_parts.append(text)
    
    elif hasattr(transcription, 'segments') and transcription.segments:
        # Fallback to segment-level timestamps
        segments = transcription.segments
        print(f"📝 Got {len(segments)} segment-level timestamps (no word-level)")
        
        for segment in segments:
            text = segment.text.strip() if hasattr(segment, 'text') else segment.get('text', '').strip()
            if not text:
                continue
            
            start = segment.start if hasattr(segment, 'start') else segment.get('start', 0)
            end = segment.end if hasattr(segment, 'end') else segment.get('end', start + 1)
            
            # Split into word chunks
            words = text.split()
            duration = end - start
            time_per_word = duration / len(words) if words else duration
            
            for j in range(0, len(words), words_per_subtitle):
                chunk_words = words[j:j + words_per_subtitle]
                chunk_start = start + (j * time_per_word)
                chunk_end = min(start + ((j + len(chunk_words)) * time_per_word), end)
                chunk_text = " ".join(chunk_words)
                
                if chunk_text:
                    result_segments.append({
                        'start': float(chunk_start),
                        'end': float(chunk_end),
                        'text': chunk_text,
                    })
                    full_text_parts.append(chunk_text)
    
    else:
        # Last resort: just use the text
        text = transcription.text if hasattr(transcription, 'text') else ""
        print(f"⚠️ No timestamps available, using text only")
        
        if text:
            words = text.split()
            # Estimate ~0.3 seconds per word
            for j in range(0, len(words), words_per_subtitle):
                chunk_words = words[j:j + words_per_subtitle]
                chunk_start = j * 0.3
                chunk_end = (j + len(chunk_words)) * 0.3
                chunk_text = " ".join(chunk_words)
                
                if chunk_text:
                    result_segments.append({
                        'start': float(chunk_start),
                        'end': float(chunk_end),
                        'text': chunk_text,
                    })
                    full_text_parts.append(chunk_text)
    
    full_text = " ".join(full_text_parts)
    print(f"✅ Created {len(result_segments)} subtitle segments via Groq")
    
    return result_segments, full_text


def transcribe_with_segments(
    video_path: str,
    words_per_subtitle: int = 3,
    audio_path: str = None,
) -> tuple[list[dict], str]:
    """
    Transcribe using Groq if available, otherwise fall back to local Whisper.
    
    This is the main entry point that automatically chooses the best method.
    
    Args:
        video_path: Path to video file (used if audio_path not provided)
        words_per_subtitle: Max words per subtitle chunk
        audio_path: Optional path to preprocessed audio file
        
    Returns:
        Tuple of (segments list, transcript text)
    """
    input_path = audio_path if audio_path else video_path
    
    if is_groq_available():
        try:
            return transcribe_with_groq(input_path, words_per_subtitle)
        except Exception as e:
            print(f"⚠️ Groq transcription failed: {e}")
            print("🔄 Falling back to local Whisper...")
    else:
        print("ℹ️ GROQ_API_KEY not set, using local Whisper")
    
    # Fall back to local transcription
    from transcribe import transcribe_video_with_segments
    return transcribe_video_with_segments(video_path, words_per_subtitle, audio_path)

