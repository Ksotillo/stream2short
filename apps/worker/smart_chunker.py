"""Smart subtitle chunking for Stream2Short Worker.

Replaces fixed "N-word chunks" with intelligent segmentation rules:
- Break on punctuation (. , ! ? ; :)
- Max ~30 characters per line
- Max 2 lines per subtitle
- Duration: 0.8–2.2 seconds per chunk (ADAPTIVE based on speech rate)
- Optional keyword emphasis (highlights important words)
- Non-overlap post-processor ensures no subtitle collisions

This creates more natural, readable subtitles that respect sentence structure
and reading pace, adapting to fast or slow speakers.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple


# Configuration constants
DEFAULT_MAX_CHARS_PER_LINE = 30
DEFAULT_MAX_LINES = 2
DEFAULT_MIN_DURATION = 0.8  # seconds
DEFAULT_MAX_DURATION = 2.2  # seconds
DEFAULT_MIN_CHARS = 5  # Minimum chars before allowing a break

# Adaptive speech rate thresholds (words per second)
SPEECH_RATE_SLOW = 2.5      # Below this = slow speaker
SPEECH_RATE_FAST = 4.0      # Above this = fast speaker

# Minimum gap between subtitles to prevent overlap (seconds)
MIN_SUBTITLE_GAP = 0.05  # 50ms

# Punctuation that signals a natural break point
STRONG_BREAK_PUNCTUATION = {'.', '!', '?'}  # End of sentence
MEDIUM_BREAK_PUNCTUATION = {',', ';', ':'}  # Clause breaks
WEAK_BREAK_PUNCTUATION = {'-', '–', '—'}    # Dashes

# Words that commonly indicate emphasis (for optional highlighting)
EMPHASIS_KEYWORDS = {
    # Exclamations
    'wow', 'omg', 'damn', 'holy', 'what', 'yes', 'no', 'wait', 'stop', 
    'look', 'oh', 'nice', 'sick', 'insane', 'crazy', 'actually', 'literally',
    # Gaming terms
    'gg', 'ez', 'rip', 'pog', 'clutch', 'win', 'lose', 'kill', 'dead',
    'lets', "let's", 'go', 'run', 'push', 'back', 'help',
    # Intensifiers
    'really', 'very', 'super', 'so', 'just', 'never', 'always', 'ever',
}


@dataclass
class Word:
    """Represents a word with timing information."""
    text: str
    start: float
    end: float
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    @property
    def has_strong_punctuation(self) -> bool:
        """Check if word ends with sentence-ending punctuation."""
        return any(self.text.rstrip().endswith(p) for p in STRONG_BREAK_PUNCTUATION)
    
    @property
    def has_medium_punctuation(self) -> bool:
        """Check if word ends with clause-break punctuation."""
        return any(self.text.rstrip().endswith(p) for p in MEDIUM_BREAK_PUNCTUATION)
    
    @property
    def has_weak_punctuation(self) -> bool:
        """Check if word ends with weak break punctuation."""
        return any(self.text.rstrip().endswith(p) for p in WEAK_BREAK_PUNCTUATION)
    
    @property
    def is_emphasis_word(self) -> bool:
        """Check if this word should be emphasized."""
        clean = re.sub(r'[^\w]', '', self.text.lower())
        return clean in EMPHASIS_KEYWORDS


@dataclass 
class ChunkConfig:
    """Configuration for smart chunking behavior."""
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE
    max_lines: int = DEFAULT_MAX_LINES
    min_duration: float = DEFAULT_MIN_DURATION
    max_duration: float = DEFAULT_MAX_DURATION
    min_chars: int = DEFAULT_MIN_CHARS
    enable_emphasis: bool = False  # Highlight keywords
    
    @property
    def max_chars_total(self) -> int:
        """Maximum characters across all lines."""
        return self.max_chars_per_line * self.max_lines


def calculate_speech_rate(words: list) -> Tuple[float, str]:
    """
    Calculate speech rate from word timestamps.
    
    Args:
        words: List of word dicts with timing info
        
    Returns:
        Tuple of (words_per_second, rate_category)
        rate_category is 'slow', 'normal', or 'fast'
    """
    if not words or len(words) < 2:
        return 3.0, 'normal'  # Default to normal
    
    # Get first and last word timestamps
    first_word = words[0]
    last_word = words[-1]
    
    # Extract start/end times
    if hasattr(first_word, 'start'):
        start_time = first_word.start
        end_time = last_word.end if hasattr(last_word, 'end') else last_word.start + 0.3
    elif isinstance(first_word, dict):
        start_time = first_word.get('start', 0)
        end_time = last_word.get('end', last_word.get('start', 0) + 0.3)
    else:
        return 3.0, 'normal'
    
    total_duration = end_time - start_time
    if total_duration <= 0:
        return 3.0, 'normal'
    
    words_per_second = len(words) / total_duration
    
    if words_per_second < SPEECH_RATE_SLOW:
        category = 'slow'
    elif words_per_second > SPEECH_RATE_FAST:
        category = 'fast'
    else:
        category = 'normal'
    
    return words_per_second, category


def get_adaptive_config(
    speech_rate: float,
    rate_category: str,
    enable_emphasis: bool = False,
) -> ChunkConfig:
    """
    Create a ChunkConfig adapted to the speech rate.
    
    Fast speakers get shorter chunks to prevent overlap.
    Slow speakers get longer chunks for natural reading.
    
    Args:
        speech_rate: Words per second
        rate_category: 'slow', 'normal', or 'fast'
        enable_emphasis: Whether to enable keyword emphasis
        
    Returns:
        ChunkConfig tuned for the speech rate
    """
    if rate_category == 'slow':
        # Slow speech: longer chunks, more characters
        return ChunkConfig(
            max_chars_per_line=35,
            max_lines=2,
            min_duration=1.0,
            max_duration=2.5,
            min_chars=6,
            enable_emphasis=enable_emphasis,
        )
    elif rate_category == 'fast':
        # Fast speech: shorter chunks, fewer characters
        # This prevents subtitle overlap by creating more frequent, shorter segments
        return ChunkConfig(
            max_chars_per_line=25,
            max_lines=2,
            min_duration=0.5,
            max_duration=1.2,  # Much shorter for fast speech
            min_chars=4,
            enable_emphasis=enable_emphasis,
        )
    else:
        # Normal speech: default settings
        return ChunkConfig(
            max_chars_per_line=DEFAULT_MAX_CHARS_PER_LINE,
            max_lines=DEFAULT_MAX_LINES,
            min_duration=DEFAULT_MIN_DURATION,
            max_duration=DEFAULT_MAX_DURATION,
            min_chars=DEFAULT_MIN_CHARS,
            enable_emphasis=enable_emphasis,
        )


def smart_chunk_words(
    words: list[dict],
    config: Optional[ChunkConfig] = None,
    adaptive: bool = True,
    fix_overlaps: bool = True,
) -> list[dict]:
    """
    Convert word-level timestamps into smart subtitle chunks.
    
    Uses intelligent rules to create natural-reading subtitles:
    - Respects punctuation as natural break points
    - Limits character count per line
    - Ensures reasonable duration per chunk
    - Optionally marks emphasis words
    - ADAPTIVE: Adjusts chunk duration based on speech rate
    - NON-OVERLAP: Ensures no subtitle collisions
    
    Args:
        words: List of word dicts with 'word'/'text', 'start', 'end' keys
              (supports both Groq and faster-whisper formats)
        config: ChunkConfig with customization options (if None and adaptive=True,
                will auto-generate based on speech rate)
        adaptive: If True and no config provided, adapts chunk settings to speech rate
        fix_overlaps: If True, runs non-overlap post-processor
        
    Returns:
        List of segment dicts with 'start', 'end', 'text', and optionally 'emphasis_indices'
    """
    if not words:
        return []
    
    # Calculate speech rate for logging and adaptive config
    speech_rate, rate_category = calculate_speech_rate(words)
    
    # Use adaptive config if no config provided and adaptive mode is enabled
    if config is None:
        if adaptive:
            config = get_adaptive_config(speech_rate, rate_category)
            print(f"🎤 Speech rate: {speech_rate:.1f} words/sec ({rate_category}) → "
                  f"max_duration={config.max_duration}s, max_chars={config.max_chars_per_line}")
        else:
            config = ChunkConfig()
    
    # Normalize words to Word objects
    normalized_words = _normalize_words(words)
    
    if not normalized_words:
        return []
    
    # Build chunks using smart rules
    chunks = []
    current_chunk: list[Word] = []
    current_chars = 0
    
    for i, word in enumerate(normalized_words):
        word_text = word.text.strip()
        word_chars = len(word_text) + (1 if current_chunk else 0)  # +1 for space
        
        # Calculate what adding this word would mean
        new_chars = current_chars + word_chars
        new_duration = (word.end - current_chunk[0].start) if current_chunk else word.duration
        
        # Determine if we should break before this word
        should_break = False
        
        if current_chunk:
            # Check duration constraints
            if new_duration > config.max_duration:
                should_break = True
            
            # Check character limit
            elif new_chars > config.max_chars_total:
                should_break = True
            
            # Check for natural break after previous word (punctuation)
            prev_word = current_chunk[-1]
            
            if prev_word.has_strong_punctuation:
                # Strong punctuation: always break (end of sentence)
                should_break = True
            
            elif prev_word.has_medium_punctuation:
                # Medium punctuation: break if chunk is substantial enough
                chunk_duration = word.start - current_chunk[0].start
                if chunk_duration >= config.min_duration and current_chars >= config.min_chars:
                    should_break = True
            
            elif prev_word.has_weak_punctuation:
                # Weak punctuation: only break if we're approaching limits
                if new_chars > config.max_chars_per_line or new_duration > config.max_duration * 0.8:
                    should_break = True
        
        # Execute break if needed
        if should_break and current_chunk:
            chunk = _finalize_chunk(current_chunk, config)
            if chunk:
                chunks.append(chunk)
            current_chunk = []
            current_chars = 0
        
        # Add word to current chunk
        current_chunk.append(word)
        current_chars = sum(len(w.text.strip()) for w in current_chunk) + len(current_chunk) - 1
    
    # Finalize last chunk
    if current_chunk:
        chunk = _finalize_chunk(current_chunk, config)
        if chunk:
            chunks.append(chunk)
    
    # Post-process: merge very short chunks, split very long ones
    chunks = _post_process_chunks(chunks, config)
    
    # Fix any overlapping subtitles
    if fix_overlaps:
        chunks = fix_overlapping_subtitles(chunks)
    
    print(f"📝 Smart chunking: {len(words)} words → {len(chunks)} chunks")
    
    return chunks


def _normalize_words(words: list[dict]) -> list[Word]:
    """
    Normalize word data from different transcription formats.
    
    Supports:
    - Groq format: {'word': 'text', 'start': 0.0, 'end': 0.5}
    - faster-whisper: objects with .word, .start, .end attributes
    - Generic dict: {'text': 'text', 'start': 0.0, 'end': 0.5}
    """
    normalized = []
    
    for w in words:
        # Handle object attributes (faster-whisper)
        if hasattr(w, 'word'):
            text = w.word
            start = w.start
            end = w.end
        # Handle Groq dict format
        elif isinstance(w, dict):
            text = w.get('word') or w.get('text', '')
            start = w.get('start', 0)
            end = w.get('end', start + 0.3)
        else:
            continue
        
        text = str(text).strip()
        if text:
            normalized.append(Word(
                text=text,
                start=float(start),
                end=float(end),
            ))
    
    return normalized


def _finalize_chunk(words: list[Word], config: ChunkConfig) -> Optional[dict]:
    """
    Convert a list of Word objects into a final chunk dict.
    
    Applies text formatting and identifies emphasis words if enabled.
    """
    if not words:
        return None
    
    text = " ".join(w.text.strip() for w in words)
    
    if not text.strip():
        return None
    
    chunk = {
        'start': words[0].start,
        'end': words[-1].end,
        'text': text,
    }
    
    # Add emphasis information if enabled
    if config.enable_emphasis:
        emphasis_indices = []
        for i, word in enumerate(words):
            if word.is_emphasis_word:
                emphasis_indices.append(i)
        
        if emphasis_indices:
            chunk['emphasis_indices'] = emphasis_indices
    
    return chunk


def _post_process_chunks(chunks: list[dict], config: ChunkConfig) -> list[dict]:
    """
    Post-process chunks to handle edge cases.
    
    - Merges very short chunks (< min_duration) with neighbors
    - Ensures no chunk exceeds max constraints
    """
    if len(chunks) <= 1:
        return chunks
    
    processed = []
    i = 0
    
    while i < len(chunks):
        chunk = chunks[i]
        duration = chunk['end'] - chunk['start']
        
        # If chunk is too short and can be merged with next
        if duration < config.min_duration and i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            combined_duration = next_chunk['end'] - chunk['start']
            combined_text = chunk['text'] + " " + next_chunk['text']
            
            # Merge if combined would be within limits
            if (combined_duration <= config.max_duration and 
                len(combined_text) <= config.max_chars_total):
                merged = {
                    'start': chunk['start'],
                    'end': next_chunk['end'],
                    'text': combined_text,
                }
                
                # Merge emphasis indices
                if 'emphasis_indices' in chunk or 'emphasis_indices' in next_chunk:
                    merged['emphasis_indices'] = chunk.get('emphasis_indices', []).copy()
                    if 'emphasis_indices' in next_chunk:
                        # Offset indices for merged words
                        word_count = len(chunk['text'].split())
                        for idx in next_chunk['emphasis_indices']:
                            merged['emphasis_indices'].append(idx + word_count)
                
                processed.append(merged)
                i += 2  # Skip next chunk since we merged it
                continue
        
        processed.append(chunk)
        i += 1
    
    return processed


def fix_overlapping_subtitles(
    chunks: list[dict],
    min_gap: float = MIN_SUBTITLE_GAP,
) -> list[dict]:
    """
    Post-processor to fix overlapping subtitles.
    
    Ensures no two subtitles display at the same time by:
    1. Detecting overlaps (current.end > next.start)
    2. Trimming current.end to create a gap before next.start
    3. If trimming would make a subtitle too short (<0.2s), log a warning
    
    Args:
        chunks: List of chunk dicts with 'start', 'end', 'text' keys
        min_gap: Minimum gap between subtitles (default 50ms)
        
    Returns:
        Fixed chunks with no overlaps
    """
    if len(chunks) <= 1:
        return chunks
    
    fixed = []
    overlaps_fixed = 0
    
    for i, chunk in enumerate(chunks):
        fixed_chunk = chunk.copy()
        
        # Check for overlap with next chunk
        if i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            current_end = fixed_chunk['end']
            next_start = next_chunk['start']
            
            # Overlap detected
            if current_end > next_start - min_gap:
                # Calculate new end time (with gap)
                new_end = next_start - min_gap
                
                # Check if this would make the subtitle too short
                current_start = fixed_chunk['start']
                new_duration = new_end - current_start
                
                if new_duration >= 0.2:  # Minimum readable duration
                    fixed_chunk['end'] = new_end
                    overlaps_fixed += 1
                else:
                    # Subtitle would be too short - keep original but log
                    # This is a rare edge case with extremely fast speech
                    print(f"⚠️ Subtitle overlap: cannot fix without making subtitle too short "
                          f"(would be {new_duration:.2f}s)")
        
        fixed.append(fixed_chunk)
    
    if overlaps_fixed > 0:
        print(f"🔧 Fixed {overlaps_fixed} overlapping subtitle(s)")
    
    return fixed


def format_chunk_for_display(
    chunk: dict,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
) -> str:
    """
    Format chunk text with line breaks for subtitle display.
    
    Splits text into lines of max_chars_per_line, breaking at word boundaries.
    
    Args:
        chunk: Chunk dict with 'text' key
        max_chars_per_line: Maximum characters per line
        
    Returns:
        Formatted text with \\N line breaks for ASS format
    """
    text = chunk.get('text', '')
    words = text.split()
    
    if not words:
        return ''
    
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_len = len(word)
        space_len = 1 if current_line else 0
        
        if current_length + space_len + word_len > max_chars_per_line and current_line:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_len
        else:
            current_line.append(word)
            current_length += space_len + word_len
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Limit to 2 lines
    lines = lines[:2]
    
    return '\\N'.join(lines)


def add_emphasis_markup(
    text: str,
    emphasis_indices: list[int],
    color: str = "&H0000FFFF",  # Yellow in ASS BGR format
) -> str:
    """
    Add ASS markup to emphasize specific words.
    
    Args:
        text: Original text
        emphasis_indices: Word indices to emphasize
        color: ASS color code for emphasis
        
    Returns:
        Text with ASS color tags for emphasis
    """
    if not emphasis_indices:
        return text
    
    words = text.split()
    for i in emphasis_indices:
        if 0 <= i < len(words):
            # Wrap word in color tags
            words[i] = f"{{\\c{color}}}{words[i]}{{\\c}}"
    
    return ' '.join(words)


# Convenience function for pipeline integration
def smart_chunk_transcript(
    words: list,
    enable_emphasis: bool = False,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_LINE,
    max_lines: int = DEFAULT_MAX_LINES,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    adaptive: bool = True,
    fix_overlaps: bool = True,
) -> list[dict]:
    """
    High-level function for transcription pipeline integration.
    
    Drop-in replacement for fixed-word chunking with adaptive speech rate
    detection and overlap prevention.
    
    Args:
        words: Word-level timestamps from transcription
        enable_emphasis: Whether to mark emphasis words
        max_chars_per_line: Max characters per subtitle line (ignored if adaptive=True)
        max_lines: Max lines per subtitle
        min_duration: Minimum subtitle duration, seconds (ignored if adaptive=True)
        max_duration: Maximum subtitle duration, seconds (ignored if adaptive=True)
        adaptive: If True, auto-adjusts chunk settings based on speech rate.
                  Fast speakers get shorter chunks to prevent overlap.
                  Slow speakers get longer chunks for natural reading.
        fix_overlaps: If True, runs non-overlap post-processor to ensure
                      no two subtitles display simultaneously.
        
    Returns:
        List of segment dicts compatible with existing pipeline
    """
    if adaptive:
        # Let smart_chunk_words auto-detect speech rate and configure
        return smart_chunk_words(
            words,
            config=None,
            adaptive=True,
            fix_overlaps=fix_overlaps,
        )
    else:
        # Use provided config values
        config = ChunkConfig(
            max_chars_per_line=max_chars_per_line,
            max_lines=max_lines,
            min_duration=min_duration,
            max_duration=max_duration,
            enable_emphasis=enable_emphasis,
        )
        return smart_chunk_words(words, config, adaptive=False, fix_overlaps=fix_overlaps)

