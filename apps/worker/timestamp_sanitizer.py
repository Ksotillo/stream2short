"""Timestamp sanitizer for word-level transcription timestamps.

Whisper (via Groq or local) sometimes produces incorrect word-level timestamps:
- Words pinned to 0.0s when they're actually spoken much later
- Words with timestamps that don't correspond to any speech segment

This module cross-references word-level timestamps against segment-level
timestamps and ONLY fixes clearly broken ones — it does not aggressively
clamp words that already have reasonable timing.

Key principle: if a word's timestamp already falls within (or near) any
segment's time range, it's probably correct. Only fix timestamps that are
clearly orphaned (not near any segment).
"""

from typing import Optional


# A word is "orphaned" if it's more than this far from any segment boundary
ORPHAN_THRESHOLD = 1.5  # seconds — word must be >1.5s from nearest segment

# A word at the very start (< this time) with a large gap to next word is suspicious
EARLY_WORD_THRESHOLD = 0.5  # seconds — words starting before 0.5s are suspect


def sanitize_word_timestamps(
    words: list[dict],
    segments: list[dict],
    debug: bool = False,
) -> list[dict]:
    """
    Fix ONLY clearly broken word timestamps using segments as reference.

    Conservative approach — only fixes words whose timestamps are clearly wrong:
    1. Words that don't fall within or near ANY segment (orphaned)
    2. Words stuck at 0.0s when no segment starts that early

    Does NOT fix words that already have reasonable timing, even if they
    don't perfectly match their text-matched segment. This prevents
    breaking correct timestamps like "carro" at 3.5s being clamped to 7s.

    Args:
        words: List of word dicts with 'word', 'start', 'end'
        segments: List of segment dicts with 'text', 'start', 'end'
        debug: Print detailed fix info

    Returns:
        Sanitized copy of words list with corrected timestamps
    """
    if not words or not segments:
        return words

    fixes_applied = 0
    sanitized = []

    for i, word in enumerate(words):
        fixed_word = word.copy()

        # Check if this word's timestamp is "orphaned" — not near any segment
        is_orphaned, nearest_seg, distance = _check_orphaned(fixed_word, segments)

        if is_orphaned and nearest_seg:
            # This word's timestamp is clearly wrong — fix it
            old_start = fixed_word['start']

            # Find the best time to place this word
            fixed_word = _fix_orphaned_word(fixed_word, i, words, segments, debug)
            fixes_applied += 1

            if debug:
                print(f"  🔧 Orphaned word '{fixed_word['word']}' at {old_start:.2f}s "
                      f"(nearest segment at {nearest_seg['start']:.2f}s, dist={distance:.1f}s) "
                      f"→ moved to {fixed_word['start']:.2f}s")

        # Ensure end > start (basic sanity)
        if fixed_word['end'] <= fixed_word['start']:
            fixed_word['end'] = fixed_word['start'] + 0.15

        sanitized.append(fixed_word)

    # Second pass: fix early-start words (0.0s with big gap to next word)
    early_fixes = _fix_early_start_words(sanitized, segments, debug=debug)
    fixes_applied += early_fixes

    if fixes_applied > 0:
        print(f"🔧 Timestamp sanitizer: fixed {fixes_applied} word timestamp(s)")
    else:
        if debug:
            print(f"✅ Timestamp sanitizer: all {len(words)} word timestamps look correct")

    return sanitized


def _check_orphaned(
    word: dict,
    segments: list[dict],
) -> tuple[bool, Optional[dict], float]:
    """
    Check if a word's timestamp is orphaned (not near any segment).

    A word is orphaned if its start time is more than ORPHAN_THRESHOLD
    seconds away from the nearest segment's time range.

    Returns:
        (is_orphaned, nearest_segment, distance_to_nearest)
    """
    word_start = word['start']
    best_seg = None
    best_dist = float('inf')

    for seg in segments:
        seg_start = seg['start']
        seg_end = seg['end']

        # Word falls inside this segment — definitely not orphaned
        if seg_start - 0.3 <= word_start <= seg_end + 0.3:
            return False, seg, 0.0

        # Calculate distance to segment
        dist = min(abs(word_start - seg_start), abs(word_start - seg_end))
        if dist < best_dist:
            best_dist = dist
            best_seg = seg

    is_orphaned = best_dist > ORPHAN_THRESHOLD
    return is_orphaned, best_seg, best_dist


def _fix_orphaned_word(
    word: dict,
    word_idx: int,
    all_words: list[dict],
    segments: list[dict],
    debug: bool = False,
) -> dict:
    """
    Fix an orphaned word by finding the right segment for it via text matching.

    Strategy:
    1. Find which segment contains this word's text
    2. Place the word at a reasonable position within that segment
    """
    word_text_clean = word['word'].strip().lower().strip('.,!?;:¿¡"\'')
    fixed = word.copy()

    # Find the segment that contains this word by text
    best_seg = None
    for seg in segments:
        seg_text_clean = seg['text'].strip().lower()
        if word_text_clean in seg_text_clean:
            best_seg = seg
            break

    if not best_seg:
        # Fallback: use nearest segment by time
        _, best_seg, _ = _check_orphaned(word, segments)

    if best_seg:
        # Place word at segment start (it will be ordered by the chunker later)
        # Use the segment's start as reference, offset slightly based on
        # neighboring words that are already in this segment
        fixed['start'] = best_seg['start']
        fixed['end'] = min(best_seg['start'] + 0.3, best_seg['end'])

        # If the previous word is also in this time range, place after it
        if word_idx > 0:
            prev = all_words[word_idx - 1]
            if abs(prev['end'] - best_seg['start']) < 2.0:
                fixed['start'] = prev['end']
                fixed['end'] = fixed['start'] + 0.3

    return fixed


def _fix_early_start_words(
    words: list[dict],
    segments: list[dict],
    debug: bool = False,
) -> int:
    """
    Fix words stuck at the very beginning (near 0.0s) when speech
    doesn't actually start that early.

    This catches the classic Whisper bug: first word gets start=0.0
    but actual speech begins seconds later.
    """
    if not words or not segments:
        return 0

    # Find when speech actually starts (earliest segment)
    earliest_segment_start = min(seg['start'] for seg in segments)

    fixes = 0

    for i, word in enumerate(words):
        # Only look at words near the start of the audio
        if word['start'] > EARLY_WORD_THRESHOLD:
            break

        # If this word starts before the earliest segment and there's a gap
        # to the next word, it's likely a Whisper hallucination at 0.0s
        if word['start'] < earliest_segment_start - ORPHAN_THRESHOLD:
            # Check gap to next word
            if i + 1 < len(words):
                next_start = words[i + 1]['start']
                gap = next_start - word['end']

                if gap > ORPHAN_THRESHOLD:
                    old_start = word['start']

                    # Move word to just before the next word
                    word_duration = max(0.15, min(word['end'] - word['start'], 0.4))
                    word['start'] = max(0, next_start - word_duration)
                    word['end'] = next_start

                    if debug:
                        print(f"  🔧 Early word '{word['word']}' at {old_start:.2f}s "
                              f"moved to {word['start']:.2f}s (speech starts at "
                              f"{earliest_segment_start:.1f}s, gap to next={gap:.1f}s)")
                    fixes += 1

    return fixes


def extract_segments_from_groq(transcription) -> list[dict]:
    """
    Extract segment-level timestamps from a Groq transcription response.

    Groq returns segments alongside words when requested with
    timestamp_granularities=["word", "segment"].

    Args:
        transcription: Groq API transcription response object

    Returns:
        List of segment dicts with 'text', 'start', 'end'
    """
    segments = []

    if hasattr(transcription, 'segments') and transcription.segments:
        for seg in transcription.segments:
            text = seg.text.strip() if hasattr(seg, 'text') else seg.get('text', '').strip()
            start = seg.start if hasattr(seg, 'start') else seg.get('start', 0)
            end = seg.end if hasattr(seg, 'end') else seg.get('end', start + 1)

            if text:
                segments.append({
                    'text': text,
                    'start': float(start),
                    'end': float(end),
                })

    return segments


def extract_segments_from_whisper(segments_list: list) -> list[dict]:
    """
    Extract segment-level timestamps from a faster-whisper segments list.

    Args:
        segments_list: List of faster-whisper Segment objects

    Returns:
        List of segment dicts with 'text', 'start', 'end'
    """
    segments = []

    for seg in segments_list:
        text = seg.text.strip() if hasattr(seg, 'text') else ''
        start = float(seg.start) if hasattr(seg, 'start') else 0
        end = float(seg.end) if hasattr(seg, 'end') else start + 1

        if text:
            segments.append({
                'text': text,
                'start': start,
                'end': end,
            })

    return segments
