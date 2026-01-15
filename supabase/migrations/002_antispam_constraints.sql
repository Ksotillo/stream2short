-- Anti-Spam / Dedupe Constraints
-- Prevents duplicate processing of the same Twitch clip

-- ============================================================================
-- PARTIAL UNIQUE INDEX ON twitch_clip_id
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

