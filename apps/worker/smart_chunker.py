"""Smart subtitle chunking for Stream2Short Worker.

Replaces fixed "N-word chunks" with intelligent segmentation rules:
- Break on punctuation (. , ! ? ; :)
- Max ~30 characters per line
- Max 2 lines per subtitle
- Duration: 0.8–2.2 seconds per chunk
- Optional keyword emphasis (highlights important words)

This creates more natural, readable subtitles that respect sentence structure
and reading pace.
"""

import re
from dataclasses import dataclass
from typing import Optional


# Configuration constants
DEFAULT_MAX_CHARS_PER_LINE = 30
DEFAULT_MAX_LINES = 2
DEFAULT_MIN_DURATION = 0.8  # seconds
DEFAULT_MAX_DURATION = 2.2  # seconds
DEFAULT_MIN_CHARS = 5  # Minimum chars before allowing a break

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


def smart_chunk_words(
    words: list[dict],
    config: Optional[ChunkConfig] = None,
) -> list[dict]:
    """
    Convert word-level timestamps into smart subtitle chunks.
    
    Uses intelligent rules to create natural-reading subtitles:
    - Respects punctuation as natural break points
    - Limits character count per line
    - Ensures reasonable duration per chunk
    - Optionally marks emphasis words
    
    Args:
        words: List of word dicts with 'word'/'text', 'start', 'end' keys
              (supports both Groq and faster-whisper formats)
        config: ChunkConfig with customization options
        
    Returns:
        List of segment dicts with 'start', 'end', 'text', and optionally 'emphasis_indices'
    """
    if config is None:
        config = ChunkConfig()
    
    if not words:
        return []
    
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
) -> list[dict]:
    """
    High-level function for transcription pipeline integration.
    
    Drop-in replacement for fixed-word chunking.
    
    Args:
        words: Word-level timestamps from transcription
        enable_emphasis: Whether to mark emphasis words
        max_chars_per_line: Max characters per subtitle line
        max_lines: Max lines per subtitle
        min_duration: Minimum subtitle duration (seconds)
        max_duration: Maximum subtitle duration (seconds)
        
    Returns:
        List of segment dicts compatible with existing pipeline
    """
    config = ChunkConfig(
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
        min_duration=min_duration,
        max_duration=max_duration,
        enable_emphasis=enable_emphasis,
    )
    
    return smart_chunk_words(words, config)

