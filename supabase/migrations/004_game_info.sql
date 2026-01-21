-- Add game/category info and thumbnail to clip_jobs
-- This enables filtering clips by game and displaying thumbnails

-- ============================================================================
-- CLIP_JOBS: Add game and thumbnail columns
-- ============================================================================

-- Game/category ID from Twitch (e.g., "33214" for Fortnite)
ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS game_id TEXT;

-- Game/category name for display (e.g., "Fortnite")
ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS game_name TEXT;

-- Twitch clip thumbnail URL for preview
ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS thumbnail_url TEXT;

-- Index for filtering by game
CREATE INDEX IF NOT EXISTS idx_clip_jobs_game_id ON clip_jobs(game_id)
    WHERE game_id IS NOT NULL;

-- Index for filtering by game name (for search)
CREATE INDEX IF NOT EXISTS idx_clip_jobs_game_name ON clip_jobs(game_name)
    WHERE game_name IS NOT NULL;

-- Composite index for common dashboard queries (channel + game + status)
CREATE INDEX IF NOT EXISTS idx_clip_jobs_channel_game ON clip_jobs(channel_id, game_id, status);

