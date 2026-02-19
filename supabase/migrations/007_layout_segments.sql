-- Migration 007: Store temporal layout segment data per clip job
--
-- layout_segments stores the output of detect_layout_segments():
-- a JSON array of { start_time, end_time, layout, webcam_region } objects
-- that describe how the clip's layout changes over time.
--
-- Example (corner_overlay → full_cam near end of clip):
-- [
--   { "start_time": 0, "end_time": 38.5, "layout": "SPLIT",    "webcam_region": {...} },
--   { "start_time": 38.5, "end_time": 60, "layout": "FULL_CAM", "webcam_region": {...} }
-- ]
--
-- NULL means temporal analysis was not run (clips processed before this migration).
-- [] means analysis ran but found no transitions (single stable layout).

ALTER TABLE clip_jobs
  ADD COLUMN IF NOT EXISTS layout_segments JSONB DEFAULT NULL;

COMMENT ON COLUMN clip_jobs.layout_segments IS
  'Temporal layout segments detected by detect_layout_segments(). '
  'Array of {start_time, end_time, layout, webcam_region} or NULL if not yet analyzed.';
