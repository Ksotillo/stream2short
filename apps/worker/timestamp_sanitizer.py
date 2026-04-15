"""Timestamp sanitizer for word-level transcription timestamps.

Whisper (via Groq or local) sometimes produces incorrect word-level timestamps:
- Words pinned to 0.0s when they're actually spoken much later
- Words with timestamps outside their parent segment boundaries
- Suspicious gaps between consecutive words

This module cross-references word-level timestamps against segment-level
timestamps and fixes anomalies before the chunker sees them.
"""

from typing import Optional


# A word is considered "suspicious" if the gap to the next word exceeds this
SUSPICIOUS_GAP_THRESHOLD = 2.0  # seconds

# If a word starts before its segment, it's clearly wrong
# (This is the core fix for the "subtitle appears at 0:00" bug)


def sanitize_word_timestamps(
    words: list[dict],
    segments: list[dict],
    debug: bool = False,
) -> list[dict]:
    """
    Fix word-level timestamps using segment-level timestamps as ground truth.

    Whisper's segment timestamps are generally more reliable than word timestamps.
    This function ensures every word falls within its parent segment's time range.

    Algorithm:
    1. For each word, find the segment it belongs to (by text matching + proximity)
    2. Clamp word.start to be >= segment.start
    3. Clamp word.end to be <= segment.end
    4. Detect suspicious gaps and redistribute timestamps

    Args:
        words: List of word dicts with 'word', 'start', 'end'
        segments: List of segment dicts with 'text', 'start', 'end'
        debug: Print detailed fix info

    Returns:
        Sanitized copy of words list with corrected timestamps
    """
    if not words or not segments:
        return words

    # Build segment lookup: for each segment, track which words belong to it
    word_to_segment = _assign_words_to_segments(words, segments)

    fixes_applied = 0
    sanitized = []

    for i, word in enumerate(words):
        fixed_word = word.copy()
        seg = word_to_segment.get(i)

        if seg:
            seg_start = seg['start']
            seg_end = seg['end']

            # Clamp word start to segment start
            if fixed_word['start'] < seg_start:
                if debug:
                    print(f"  🔧 Word '{fixed_word['word']}' start {fixed_word['start']:.2f}s "
                          f"< segment start {seg_start:.2f}s → clamped")
                fixed_word['start'] = seg_start
                fixes_applied += 1

            # Clamp word end to segment end
            if fixed_word['end'] > seg_end:
                if debug:
                    print(f"  🔧 Word '{fixed_word['word']}' end {fixed_word['end']:.2f}s "
                          f"> segment end {seg_end:.2f}s → clamped")
                fixed_word['end'] = seg_end
                fixes_applied += 1

        # Ensure end > start (basic sanity)
        if fixed_word['end'] <= fixed_word['start']:
            fixed_word['end'] = fixed_word['start'] + 0.15

        sanitized.append(fixed_word)

    # Second pass: fix suspicious gaps between consecutive words
    gap_fixes = _fix_suspicious_gaps(sanitized, debug=debug)
    fixes_applied += gap_fixes

    # Third pass: ensure monotonic timestamps (each word starts >= previous word's start)
    mono_fixes = _ensure_monotonic(sanitized, debug=debug)
    fixes_applied += mono_fixes

    if fixes_applied > 0:
        print(f"🔧 Timestamp sanitizer: fixed {fixes_applied} word timestamp(s)")
    else:
        if debug:
            print(f"✅ Timestamp sanitizer: all {len(words)} word timestamps look correct")

    return sanitized


def _assign_words_to_segments(
    words: list[dict],
    segments: list[dict],
) -> dict[int, dict]:
    """
    Assign each word to its parent segment.

    Uses a two-pass approach:
    1. Try text matching (rebuild segment text from consecutive words)
    2. Fall back to time proximity (find closest segment by timestamp)

    Returns:
        Dict mapping word index -> segment dict
    """
    if not segments:
        return {}

    word_to_segment = {}

    # Approach: walk through segments and words in order.
    # For each segment, consume words whose text matches the segment's words.
    seg_idx = 0
    word_idx = 0

    # Precompute: split each segment's text into individual words for matching
    seg_words_list = []
    for seg in segments:
        seg_text = seg.get('text', '').strip()
        seg_word_tokens = seg_text.split()
        seg_words_list.append(seg_word_tokens)

    for seg_idx, seg in enumerate(segments):
        seg_word_tokens = seg_words_list[seg_idx]
        if not seg_word_tokens:
            continue

        # Try to match words sequentially
        matched_count = 0
        scan_idx = word_idx

        for seg_word_token in seg_word_tokens:
            if scan_idx >= len(words):
                break

            # Fuzzy match: strip punctuation and compare lowercase
            word_clean = words[scan_idx]['word'].strip().lower().strip('.,!?;:¿¡"\'')
            seg_clean = seg_word_token.strip().lower().strip('.,!?;:¿¡"\'')

            if word_clean == seg_clean or seg_clean.startswith(word_clean) or word_clean.startswith(seg_clean):
                word_to_segment[scan_idx] = seg
                scan_idx += 1
                matched_count += 1
            else:
                # Words might not perfectly match (punctuation differences, etc.)
                # Still assign if we're within the segment's time range
                if seg['start'] <= words[scan_idx]['start'] <= seg['end'] + 0.5:
                    word_to_segment[scan_idx] = seg
                    scan_idx += 1
                    matched_count += 1
                else:
                    break

        if matched_count > 0:
            word_idx = scan_idx

    # Fallback: assign any unmatched words to the nearest segment by time
    for i in range(len(words)):
        if i not in word_to_segment:
            word_start = words[i]['start']
            best_seg = None
            best_dist = float('inf')

            for seg in segments:
                # Distance: how far is the word from this segment's time range?
                if seg['start'] <= word_start <= seg['end']:
                    best_seg = seg
                    break
                dist = min(abs(word_start - seg['start']), abs(word_start - seg['end']))
                if dist < best_dist:
                    best_dist = dist
                    best_seg = seg

            if best_seg:
                word_to_segment[i] = best_seg

    return word_to_segment


def _fix_suspicious_gaps(
    words: list[dict],
    debug: bool = False,
) -> int:
    """
    Detect and fix suspicious gaps between consecutive words.

    Example: word[0] = {start: 0.0, end: 0.3}, word[1] = {start: 7.2, end: 7.5}
    The 7-second gap is clearly wrong. Word[0] should start near word[1].

    Fix: redistribute the first word's timing to just before the second word.
    """
    fixes = 0

    for i in range(len(words) - 1):
        current = words[i]
        next_word = words[i + 1]

        gap = next_word['start'] - current['end']

        if gap > SUSPICIOUS_GAP_THRESHOLD:
            # This word's timestamp is likely wrong
            # Move it to just before the next word, with estimated duration
            word_duration = current['end'] - current['start']
            word_duration = max(0.15, min(word_duration, 0.5))  # Sane duration

            new_start = next_word['start'] - word_duration
            new_end = next_word['start']

            if debug:
                print(f"  🔧 Suspicious gap: '{current['word']}' ends at {current['end']:.2f}s "
                      f"but next word starts at {next_word['start']:.2f}s "
                      f"(gap={gap:.1f}s) → moved to {new_start:.2f}s")

            current['start'] = max(0, new_start)
            current['end'] = max(current['start'] + 0.1, new_end)
            fixes += 1

    return fixes


def _ensure_monotonic(
    words: list[dict],
    debug: bool = False,
) -> int:
    """
    Ensure word timestamps are monotonically non-decreasing.

    After clamping and gap fixes, some words might have start times
    that come before the previous word. Fix by pushing them forward.
    """
    fixes = 0

    for i in range(1, len(words)):
        prev_end = words[i - 1]['end']
        if words[i]['start'] < prev_end:
            if debug:
                print(f"  🔧 Non-monotonic: '{words[i]['word']}' start {words[i]['start']:.2f}s "
                      f"< prev end {prev_end:.2f}s → adjusted")
            words[i]['start'] = prev_end
            if words[i]['end'] <= words[i]['start']:
                words[i]['end'] = words[i]['start'] + 0.15
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
