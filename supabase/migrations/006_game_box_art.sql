-- Add game box art URL to clip_jobs
-- Twitch provides box art for games which we can display in the dashboard

ALTER TABLE clip_jobs ADD COLUMN IF NOT EXISTS game_box_art_url TEXT;

COMMENT ON COLUMN clip_jobs.game_box_art_url IS 
    'Twitch game box art URL (template with {width}x{height} placeholders)';
