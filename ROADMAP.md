# Stream2Short Roadmap

> **Goal:** Implement features **incrementally** (phase-by-phase) on top of the existing MVP.
>
> **Principles:**
> - Make changes **small, testable, and reversible**
> - Each phase must include **acceptance criteria**
> - Maintain **multi-streamer** behavior
> - Introduce new features behind **feature flags** / per-channel settings

---

## ✅ Completed (MVP Baseline)

### Core Pipeline
- [x] Twitch OAuth integration
- [x] StreamElements `!clip` command support
- [x] Clip creation via Twitch API
- [x] Video download via yt-dlp
- [x] AI transcription (faster-whisper)
- [x] Vertical video rendering (FFmpeg)
- [x] Google Drive upload (Shared Drives)
- [x] Multi-streamer support

### Subtitle Enhancements
- [x] Speaker diarization (pyannote.audio)
- [x] Gender-based subtitle colors
- [x] Speaker merging (same voice detection)
- [x] Zoom-in animation effect
- [x] TikTok-style formatting (2-3 words, ALL CAPS)

### Video Enhancements
- [x] Webcam detection and split layout
- [x] Face-tight cropping
- [x] Two output versions (with/without subtitles)

### Infrastructure
- [x] Anti-spam cooldowns
- [x] Duplicate clip prevention
- [x] Disk space management
- [x] Audio preprocessing (loudness normalization)
- [x] Guaranteed temp file cleanup

### API Endpoints
- [x] `GET /se/clip` - StreamElements trigger
- [x] `POST /api/clip` - Direct clip creation (requires LIVE)
- [x] `POST /api/process-clip` - Process existing clip (offline)
- [x] `GET /jobs` - List jobs for channel
- [x] `GET /jobs/:id` - Get job details

---

## 🎯 Phase 1 — "Usable Daily": Dashboard + Notifications + Re-renders

**Why:** Turn the pipeline into a workflow creators/editors can actually use.

### 1.1 Next.js Dashboard (`apps/dashboard`)
- **Status:** ✅ Complete
- **Effort:** ~4-6 hours

**Pages:**
| Route | Description |
|-------|-------------|
| `/` | Channel selection (or default if only one) |
| `/jobs` | Job list with filters: channel, status, date range |
| `/jobs/[id]` | Job detail with video preview, transcript, actions |

**Features:**
- Video preview (WITH_SUBTITLES + WITHOUT_SUBTITLES side by side)
- Transcript preview with speaker labels
- Actions: Approve / Reject / Retry / Re-render
- Filter by channel (multi-streamer support)

**Auth Options:**
- **Option A (fastest):** Shared password via `DASHBOARD_PASSWORD` env var
- **Option B (proper):** Supabase Auth + RLS (can defer to Phase 2)

### 1.2 New API Endpoints
- **Status:** ✅ Complete
- **Effort:** ~2 hours

```
GET  /api/channels              → List connected channels
GET  /api/jobs?channel=&status= → Paginated job list
GET  /api/jobs/:id              → Job details (exists, enhance)
POST /api/jobs/:id/review       → { decision: "approved"|"rejected", notes? }
POST /api/jobs/:id/retry        → { from_stage?: "download"|"transcribe"|"render"|"upload" }
POST /api/jobs/:id/rerender     → { preset: "clean"|"boxed"|... }
```

**Security:** Add `DASHBOARD_API_KEY` header requirement for dashboard endpoints.

### 1.3 Database Schema Updates
- **Status:** ✅ Complete
- **Effort:** ~30 min

**Add to `clip_jobs`:**
```sql
ALTER TABLE clip_jobs ADD COLUMN review_status TEXT;      -- pending|approved|rejected
ALTER TABLE clip_jobs ADD COLUMN review_notes TEXT;
ALTER TABLE clip_jobs ADD COLUMN reviewed_at TIMESTAMPTZ;
ALTER TABLE clip_jobs ADD COLUMN last_stage TEXT;         -- for retry hints
ALTER TABLE clip_jobs ADD COLUMN render_preset TEXT DEFAULT 'default';
```

**Optional `job_events` table (for debugging/progress):**
```sql
CREATE TABLE job_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id UUID REFERENCES clip_jobs(id),
  created_at TIMESTAMPTZ DEFAULT now(),
  level TEXT,      -- info|warn|error
  stage TEXT,
  message TEXT,
  data JSONB
);
```

### 1.4 Discord Notifications
- **Status:** ⏭️ Skipped (for later)
- **Effort:** ~1 hour

**Features:**
- Webhook notification on job `ready`
- Include: streamer name, clip URL, Google Drive link, dashboard link
- Thumbnail preview in embed

**Env vars:**
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
PUBLIC_DASHBOARD_URL=https://your-domain.com/dashboard
```

### 1.5 Retry + Re-render Mechanics
- **Status:** ✅ Complete
- **Effort:** ~2 hours

**Retry:** Re-queue same job, worker resumes from failed stage.

**Re-render:** Skip download/transcribe, only re-render with new preset:
- `clean` - White text, outline, subtle shadow
- `boxed` - Rounded rectangle background
- `minimal` - Small, unobtrusive

### Phase 1 Acceptance Criteria
- [x] Dashboard shows jobs for multiple streamers
- [x] Can preview WITH/WITHOUT subtitles
- [x] Can approve/reject with notes
- [x] Can retry a failed job from specific stage
- [x] Can re-render with different preset without re-downloading
- [ ] ~~Discord notification fires on job completion~~ (skipped for later)

---

## 🎨 Phase 2 — "Quality + Consistency": Caption Presets + Smart Chunking

**Why:** Most perceived quality comes from captions. Make it consistent and configurable.

### 2.1 Caption Preset System
- **Status:** Not Started
- **Effort:** ~3 hours

**Presets:**
| Preset | Description |
|--------|-------------|
| `clean` | White text, outline, subtle shadow |
| `boxed` | Rounded rectangle background, high contrast |
| `karaoke-lite` | Highlight current phrase |
| `creator-bold` | Bigger font, punchier outline |
| `gaming` | Neon glow effects |
| `minimal` | Small, unobtrusive |

**Configurable per preset:**
- Font size, family, color
- Outline thickness & color
- Shadow distance & color
- Margins (top, bottom, sides)
- All-caps toggle
- Highlight color (optional)

**Store:** `channels.settings.subtitle_preset`

### 2.2 Smart Chunking
- **Status:** Not Started
- **Effort:** ~2 hours

Replace fixed "3-word chunks" with intelligent rules:
- Break on punctuation (. , ! ?)
- Max characters per line (~30)
- Max 2 lines per subtitle
- Duration: 0.8–2.2 seconds per chunk
- Emphasis: highlight 1 keyword per caption (optional)

### 2.3 Platform Safe Zones
- **Status:** Not Started
- **Effort:** ~1 hour

**Profiles:**
| Platform | Top Reserved | Bottom Reserved | Side Margins |
|----------|-------------|-----------------|--------------|
| TikTok | 150px | 280px | 50px |
| Reels | 120px | 320px | 40px |
| Shorts | 100px | 200px | 40px |
| Generic | 50px | 100px | 30px |

**Store:** `channels.settings.safe_zone_profile`

### Phase 2 Acceptance Criteria
- [ ] Changing `subtitle_preset` changes rendering without code changes
- [ ] Captions never overlap platform UI elements
- [ ] Chunking looks natural (phrases, punctuation) vs mechanical

---

## 🎬 Phase 3 — "Better Vertical Framing": Templates + Face Tracking

**Why:** Cropping/layout is the next quality jump. Make it deterministic and configurable.

### 3.1 Template System
- **Status:** Not Started
- **Effort:** ~4 hours

**Templates:**
| Template | Description |
|----------|-------------|
| `center_crop` | Simple center crop (existing) |
| `split_webcam` | Webcam top, game bottom (existing) |
| `blur_bg` | Blurred full frame as background, sharp crop overlay |
| `face_track` | Dynamic crop following speaker's face |
| `pip` | Picture-in-picture (small webcam corner) |

**Store:** `channels.settings.render_template`

**Auto-detection:** If webcam detected and template is `auto`, choose `split_webcam`.

### 3.2 Face Tracking (MediaPipe)
- **Status:** Not Started
- **Effort:** ~4-6 hours

**Implementation:**
- Use MediaPipe face detection
- Sample frames at 5-10 fps
- Compute crop window center, smooth with EMA
- Generate dynamic FFmpeg crop parameters

**Env:** `ENABLE_FACE_TRACKING=true`

### 3.3 Blur Background Template
- **Status:** Not Started
- **Effort:** ~2 hours

**Implementation:**
- Scale source to fill 1080x1920
- Apply gaussian blur (radius ~30)
- Overlay sharp cropped content centered

### Phase 3 Acceptance Criteria
- [ ] Each template produces stable, readable 9:16 output
- [ ] Face tracking keeps speaker centered without jitter
- [ ] Blur BG looks good for gameplay-heavy clips

---

## 📦 Phase 4 — "Scale + Reliability": Observability + Storage Abstraction

**Why:** Prepare for multiple streamers, higher volume, different storage needs.

### 4.1 Enhanced Observability
- **Status:** Not Started
- **Effort:** ~2 hours

**Metrics to track:**
- Step timings (download, transcribe, render, upload)
- FFmpeg stderr captured
- Model versions used
- Upload durations
- Error rates by stage

**Store in:** `job_events` table or `clip_jobs.metrics` JSONB

**Dashboard shows:**
- Failures by stage (pie chart)
- Average stage time (bar chart)
- Success rate over time

### 4.2 Storage Provider Abstraction
- **Status:** Not Started
- **Effort:** ~3 hours

**Providers:**
- `gdrive` (existing)
- `s3` (AWS/MinIO)
- `azure_blob`
- `local` (for dev/testing)

**Store in DB:**
```sql
ALTER TABLE clip_jobs ADD COLUMN storage_provider TEXT DEFAULT 'gdrive';
ALTER TABLE clip_jobs ADD COLUMN storage_id TEXT;      -- fileId/object key
ALTER TABLE clip_jobs ADD COLUMN storage_url TEXT;     -- derived URL
```

### 4.3 Enhanced Rate Limiting
- **Status:** Not Started
- **Effort:** ~1 hour

**Add:**
- Per-channel max jobs/hour
- Per-channel max jobs/day
- Configurable via `channels.settings`

### Phase 4 Acceptance Criteria
- [ ] Can diagnose failures from dashboard without Docker logs
- [ ] Switching storage provider requires only env var change
- [ ] Rate limits prevent abuse without blocking legitimate use

---

## 🚀 Phase 5 — "Content Engine": VOD Extraction + Auto-Titles + Quality Gate

**Why:** Go beyond clips to full content creation workflow.

### 5.1 VOD Segment Extraction
- **Status:** Not Started
- **Effort:** ~3 hours

**Endpoint:** `POST /api/extract-vod`
```json
{
  "vod_id": "1234567890",
  "start_time": "01:23:45",
  "duration": 90,
  "channel": "streamer_login"
}
```

**Features:**
- Extract any segment from VODs (up to 5 minutes)
- Works when streamer is offline
- Same processing pipeline

### 5.2 Quality Gate
- **Status:** Not Started
- **Effort:** ~2 hours

**Before full render, detect:**
- Too quiet (low audio levels)
- Too short (< 5 seconds)
- Too much silence (> 50%)
- No speech detected

**Mark jobs as:** `discarded` with reason

### 5.3 Auto-Generated Metadata
- **Status:** Not Started
- **Effort:** ~2 hours

**Generate:**
- Suggested filename (from transcript)
- Suggested title
- Suggested caption/hashtags
- Keywords for searchability

**Store:**
```sql
ALTER TABLE clip_jobs ADD COLUMN suggested_title TEXT;
ALTER TABLE clip_jobs ADD COLUMN suggested_caption TEXT;
ALTER TABLE clip_jobs ADD COLUMN suggested_tags TEXT[];
```

### 5.4 Thumbnail Generation
- **Status:** Not Started
- **Effort:** ~1 hour

**Features:**
- Extract best frame (peak audio/action)
- Save alongside video in storage
- Multiple sizes: 1080x1920, 720x1280, 360x640

### Phase 5 Acceptance Criteria
- [ ] Can extract VOD segments without Twitch clip limitations
- [ ] Low-quality clips auto-discarded with clear reason
- [ ] Generated titles/captions are usable starting points

---

## 🌟 Phase 6 — "Full Automation": Auto-Upload + Learning Loop

**Why:** Eliminate manual steps entirely.

### 6.1 Auto-Upload to Platforms
- **Status:** Not Started
- **Effort:** ~6-8 hours

**Platforms:**
| Platform | API | Notes |
|----------|-----|-------|
| TikTok | Content Posting API | Requires app approval |
| YouTube Shorts | YouTube Data API v3 | OAuth required |
| Instagram Reels | Instagram Graph API | Business account only |

**Features:**
- Draft mode (upload but don't publish)
- Scheduled publishing
- Per-channel platform configuration

### 6.2 Learning Loop
- **Status:** Not Started
- **Effort:** ~3 hours

**Use review outcomes to:**
- Track which presets get approved most
- Suggest default preset per channel
- Identify common rejection reasons

### 6.3 Background Music
- **Status:** Not Started
- **Effort:** ~3 hours

**Features:**
- Library of royalty-free tracks
- Auto-duck music during speech
- Configurable volume levels
- Per-channel music preferences

### Phase 6 Acceptance Criteria
- [ ] Clips auto-upload to configured platforms
- [ ] System learns from approvals/rejections
- [ ] Background music enhances without overpowering speech

---

## 📋 Implementation Order

### Quick Wins (Do First)
1. Discord Notifications (Phase 1.4) - ~1 hour
2. Schema updates (Phase 1.3) - ~30 min
3. Retry mechanics (Phase 1.5) - ~2 hours

### Core Value (Phase 1 Complete)
4. Dashboard API endpoints (Phase 1.2) - ~2 hours
5. Next.js Dashboard (Phase 1.1) - ~4-6 hours

### Quality Polish (Phase 2-3)
6. Caption presets (Phase 2.1) - ~3 hours
7. Smart chunking (Phase 2.2) - ~2 hours
8. Safe zones (Phase 2.3) - ~1 hour
9. Template system (Phase 3.1) - ~4 hours

### Scale & Features (Phase 4-6)
10. Observability (Phase 4.1) - ~2 hours
11. VOD extraction (Phase 5.1) - ~3 hours
12. Auto-upload (Phase 6.1) - ~6-8 hours

---

## 🛠 Technical Considerations

| Topic | Notes |
|-------|-------|
| **GPU Support** | Would 10x speed up Whisper and FFmpeg |
| **Horizontal Scaling** | Multiple workers for parallel processing |
| **CDN Integration** | Faster delivery than Google Drive links |
| **Database** | Consider direct PostgreSQL for high volume |
| **Caching** | Redis for transcription results, avoid re-processing |

---

## 📝 Branch Naming Convention

```
phase-1-dashboard
phase-1-notifications
phase-2-caption-presets
phase-3-face-tracking
...
```

**Rule:** One phase = one PR. Validate acceptance criteria before merging.

---

*Last updated: January 2026*
