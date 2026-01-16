-- Phase 1: Dashboard & Review System
-- Add review columns and job_events table for debugging/progress tracking

-- ============================================================================
-- CLIP_JOBS: Add review and stage tracking columns
-- ============================================================================

-- Review status: pending (needs review), approved, rejected
ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS review_status TEXT DEFAULT 'pending';

-- Review notes (optional comments from reviewer)
ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS review_notes TEXT;

-- Timestamp of when review was done
ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;

-- Track last completed stage (for retry hints)
-- Values: download, transcribe, render, upload
ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS last_stage TEXT;

-- Render preset used for this job
ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS render_preset TEXT DEFAULT 'default';

-- Transcript text (for dashboard preview and searching)
ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS transcript_text TEXT;

-- URL to version without subtitles (we already have final_video_url for with subs)
ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS no_subtitles_url TEXT;

-- Add constraint for valid review_status values
ALTER TABLE clip_jobs DROP CONSTRAINT IF EXISTS valid_review_status;
ALTER TABLE clip_jobs ADD CONSTRAINT valid_review_status 
    CHECK (review_status IS NULL OR review_status IN ('pending', 'approved', 'rejected'));

-- Add constraint for valid last_stage values
ALTER TABLE clip_jobs DROP CONSTRAINT IF EXISTS valid_last_stage;
ALTER TABLE clip_jobs ADD CONSTRAINT valid_last_stage 
    CHECK (last_stage IS NULL OR last_stage IN ('download', 'transcribe', 'render', 'upload'));

-- Add constraint for valid render_preset values
ALTER TABLE clip_jobs DROP CONSTRAINT IF EXISTS valid_render_preset;
ALTER TABLE clip_jobs ADD CONSTRAINT valid_render_preset 
    CHECK (render_preset IS NULL OR render_preset IN ('default', 'clean', 'boxed', 'minimal', 'bold'));

-- ============================================================================
-- JOB_EVENTS: Event/log tracking for debugging and progress
-- ============================================================================

CREATE TABLE IF NOT EXISTS job_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES clip_jobs(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    level TEXT NOT NULL DEFAULT 'info',      -- info, warn, error
    stage TEXT,                               -- download, transcribe, render, upload
    message TEXT NOT NULL,
    data JSONB DEFAULT '{}'::jsonb,          -- Additional structured data
    
    -- Constraint for valid level values
    CONSTRAINT valid_event_level CHECK (level IN ('info', 'warn', 'error'))
);

-- Index for fetching events by job (most common query)
CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id, created_at DESC);

-- Index for filtering by level (e.g., show only errors)
CREATE INDEX IF NOT EXISTS idx_job_events_level ON job_events(level);

-- ============================================================================
-- Index for review status filtering
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_clip_jobs_review_status ON clip_jobs(review_status)
    WHERE review_status IS NOT NULL;

