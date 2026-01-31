-- Phase: Transcript Editing Feature
-- Allow streamers to edit transcripts and re-render with corrected subtitles

-- ============================================================================
-- CLIP_JOBS: Add transcript segments storage
-- ============================================================================

-- Store transcript segments with timing information (for editing/re-rendering)
-- Format: [{ "start": 0.0, "end": 1.5, "text": "Hello", "speaker": "SPEAKER_0", "is_primary": true }, ...]
ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS transcript_segments JSONB;

-- Track when transcript was last edited (for UI indication)
ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS transcript_edited_at TIMESTAMPTZ;

-- Index for jobs with edited transcripts (useful for filtering)
CREATE INDEX IF NOT EXISTS idx_clip_jobs_transcript_edited 
    ON clip_jobs(transcript_edited_at) 
    WHERE transcript_edited_at IS NOT NULL;

-- ============================================================================
-- COMMENTS for documentation
-- ============================================================================

COMMENT ON COLUMN clip_jobs.transcript_segments IS 
    'JSONB array of transcript segments with timing. Each segment has: start (float), end (float), text (string), speaker (optional), is_primary (optional boolean)';

COMMENT ON COLUMN clip_jobs.transcript_edited_at IS 
    'Timestamp of when the transcript was last manually edited. NULL means not edited.';


