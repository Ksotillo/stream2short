-- Anti-Spam / Dedupe Constraints
-- Prevents duplicate processing of the same Twitch clip

-- ============================================================================
-- STEP 1: Clean up existing duplicates before adding constraint
-- ============================================================================
-- For any duplicate twitch_clip_id, keep only the MOST RECENT job
-- Set older duplicates to NULL so they don't block the constraint

-- First, identify and nullify duplicates (keep the newest one)
UPDATE clip_jobs
SET twitch_clip_id = NULL
WHERE id IN (
    SELECT id FROM (
        SELECT 
            id,
            twitch_clip_id,
            ROW_NUMBER() OVER (
                PARTITION BY twitch_clip_id 
                ORDER BY created_at DESC
            ) as rn
        FROM clip_jobs
        WHERE twitch_clip_id IS NOT NULL
    ) ranked
    WHERE rn > 1  -- All but the most recent
);

-- ============================================================================
-- STEP 2: PARTIAL UNIQUE INDEX ON twitch_clip_id
-- ============================================================================
-- Only enforce uniqueness when twitch_clip_id is NOT NULL
-- This allows multiple jobs with NULL clip_id (e.g., jobs that failed before clip creation)
CREATE UNIQUE INDEX IF NOT EXISTS idx_clip_jobs_twitch_clip_id_unique
ON clip_jobs(twitch_clip_id)
WHERE twitch_clip_id IS NOT NULL;

-- ============================================================================
-- COMPOSITE INDEX FOR COOLDOWN QUERIES
-- ============================================================================
-- Optimize queries that check recent jobs by channel + time window
CREATE INDEX IF NOT EXISTS idx_clip_jobs_channel_created_at
ON clip_jobs(channel_id, created_at DESC);

-- Optimize queries that check recent jobs by channel + user + time window
CREATE INDEX IF NOT EXISTS idx_clip_jobs_channel_user_created_at
ON clip_jobs(channel_id, requested_by, created_at DESC);

-- ============================================================================
-- INDEX FOR ACTIVE JOB QUERIES
-- ============================================================================
-- Optimize queries that check for non-completed jobs by channel
CREATE INDEX IF NOT EXISTS idx_clip_jobs_channel_status
ON clip_jobs(channel_id, status)
WHERE status NOT IN ('ready', 'failed');

