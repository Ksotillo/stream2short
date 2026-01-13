-- Twitch Clips MVP - Initial Schema
-- Multi-streamer support with channels, oauth_tokens, and clip_jobs

-- ============================================================================
-- CHANNELS TABLE
-- Stores one record per streamer (tenant)
-- ============================================================================
-- 
-- Settings JSON schema:
-- {
--   "enable_diarization": boolean  -- Override global ENABLE_DIARIZATION for this channel
-- }
--
CREATE TABLE IF NOT EXISTS channels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    twitch_broadcaster_id TEXT UNIQUE NOT NULL,
    twitch_login TEXT,
    display_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    settings JSONB DEFAULT '{}'::jsonb  -- Per-channel settings (see schema above)
);

-- Index for looking up by login name (used by StreamElements)
CREATE INDEX IF NOT EXISTS idx_channels_twitch_login ON channels(twitch_login);

-- ============================================================================
-- OAUTH_TOKENS TABLE
-- Stores OAuth tokens per channel (refreshable)
-- ============================================================================
CREATE TABLE IF NOT EXISTS oauth_tokens (
    channel_id UUID PRIMARY KEY REFERENCES channels(id) ON DELETE CASCADE,
    access_token TEXT NOT NULL,
    refresh_token TEXT NOT NULL,
    scopes TEXT[] NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- CLIP_JOBS TABLE
-- Stores jobs and their processing progress
-- ============================================================================
CREATE TABLE IF NOT EXISTS clip_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_id UUID NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    requested_by TEXT,
    source TEXT NOT NULL DEFAULT 'streamelements',
    status TEXT NOT NULL DEFAULT 'queued',
    attempt_count INT NOT NULL DEFAULT 0,
    twitch_clip_id TEXT,
    twitch_clip_url TEXT,
    raw_video_path TEXT,
    final_video_path TEXT,
    final_video_url TEXT,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraint to ensure valid status values
    CONSTRAINT valid_status CHECK (status IN (
        'queued',
        'creating_clip',
        'waiting_clip',
        'downloading',
        'transcribing',
        'rendering',
        'uploading',
        'ready',
        'failed'
    ))
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_clip_jobs_channel_created ON clip_jobs(channel_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_clip_jobs_status ON clip_jobs(status);
CREATE INDEX IF NOT EXISTS idx_clip_jobs_twitch_clip_id ON clip_jobs(twitch_clip_id);

-- ============================================================================
-- TRIGGER: Auto-update updated_at timestamp
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to oauth_tokens
DROP TRIGGER IF EXISTS update_oauth_tokens_updated_at ON oauth_tokens;
CREATE TRIGGER update_oauth_tokens_updated_at
    BEFORE UPDATE ON oauth_tokens
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to clip_jobs
DROP TRIGGER IF EXISTS update_clip_jobs_updated_at ON clip_jobs;
CREATE TRIGGER update_clip_jobs_updated_at
    BEFORE UPDATE ON clip_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- ROW LEVEL SECURITY (Optional - enable if using Supabase Auth later)
-- ============================================================================
-- ALTER TABLE channels ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE oauth_tokens ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE clip_jobs ENABLE ROW LEVEL SECURITY;

