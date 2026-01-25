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
- [x] AI transcription (~~faster-whisper~~ → **Groq API**)
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
- [x] **NEW:** Gemini Vision AI webcam detection
- [x] **NEW:** FULL_CAM layout mode (when clip is just webcam)
- [x] **NEW:** Layout caching (skip re-detection on re-renders)

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
- [x] `GET /api/jobs` - List jobs for channel
- [x] `GET /api/jobs/:id` - Get job details
- [x] **NEW:** `GET /api/games` - Get games for channel (filtering)

---

## ✅ Phase 1 — "Usable Daily": Dashboard + Notifications + Re-renders

**Status:** ✅ COMPLETE (except Discord notifications - skipped)

**Why:** Turn the pipeline into a workflow creators/editors can actually use.

### 1.1 Next.js Dashboard (`apps/dashboard`)
- **Status:** ✅ Complete
- **Effort:** ~4-6 hours

**Pages:**
| Route | Description |
|-------|-------------|
| `/` | Home with stats + recent clips |
| `/clips` | Clip list with filters: status, game, date |
| `/clips/[id]` | Clip detail with video preview, transcript, actions |
| `/clips/create` | Manual clip processing form |
| `/settings` | Channel settings + StreamElements setup |
| `/profile` | Mobile-only profile page |

**Features:**
- ✅ Video preview (WITH_SUBTITLES + WITHOUT_SUBTITLES)
- ✅ Transcript preview
- ✅ Actions: Approve / Reject / Retry / Re-render
- ✅ Filter by status and game
- ✅ **NEW:** Twitch OAuth authentication (Option B implemented!)
- ✅ **NEW:** Mobile-first responsive design (TikTok/Instagram style)
- ✅ **NEW:** Game category filtering with thumbnails

### 1.2 New API Endpoints
- **Status:** ✅ Complete

```
GET  /api/channels              → List connected channels
GET  /api/jobs?channel=&status=&game_id= → Paginated job list
GET  /api/jobs/:id              → Job details with events
GET  /api/games?channel_id=     → Games for filtering
POST /api/jobs/:id/review       → { decision: "approved"|"rejected", notes? }
POST /api/jobs/:id/retry        → { from_stage?: "download"|"transcribe"|"render"|"upload" }
POST /api/jobs/:id/rerender     → { preset: "clean"|"boxed"|... }
```

**Security:** ✅ `DASHBOARD_API_KEY` header + Twitch OAuth

### 1.3 Database Schema Updates
- **Status:** ✅ Complete

**Added to `clip_jobs`:**
```sql
-- Phase 1 original
ALTER TABLE clip_jobs ADD COLUMN review_status TEXT;
ALTER TABLE clip_jobs ADD COLUMN review_notes TEXT;
ALTER TABLE clip_jobs ADD COLUMN reviewed_at TIMESTAMPTZ;
ALTER TABLE clip_jobs ADD COLUMN last_stage TEXT;
ALTER TABLE clip_jobs ADD COLUMN render_preset TEXT DEFAULT 'default';
ALTER TABLE clip_jobs ADD COLUMN transcript_text TEXT;
ALTER TABLE clip_jobs ADD COLUMN no_subtitles_url TEXT;

-- NEW: Game info (004_game_info.sql)
ALTER TABLE clip_jobs ADD COLUMN game_id TEXT;
ALTER TABLE clip_jobs ADD COLUMN game_name TEXT;
ALTER TABLE clip_jobs ADD COLUMN thumbnail_url TEXT;
```

**`job_events` table:** ✅ Implemented for debugging/progress

### 1.4 Discord Notifications
- **Status:** ⏭️ Skipped (for later)
- **Effort:** ~1 hour

### 1.5 Retry + Re-render Mechanics
- **Status:** ✅ Complete

### Phase 1 Acceptance Criteria
- [x] Dashboard shows jobs for multiple streamers
- [x] Can preview WITH/WITHOUT subtitles
- [x] Can approve/reject with notes
- [x] Can retry a failed job from specific stage
- [x] Can re-render with different preset without re-downloading
- [x] **BONUS:** Twitch OAuth login (streamers see only their clips)
- [x] **BONUS:** Mobile-friendly UI with bottom navigation
- [x] **BONUS:** Game category filtering
- [ ] ~~Discord notification fires on job completion~~ (skipped)

---

## 🎨 Phase 2 — "Quality + Consistency": Caption Presets + Smart Chunking

**Status:** Not Started

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
- **Status:** ✅ Complete
- **Effort:** ~2 hours

Replaced fixed "3-word chunks" with intelligent rules in `smart_chunker.py`:
- ✅ Break on punctuation (. , ! ? ; :) - respects sentence/clause boundaries
- ✅ Max characters per line (~30) - with automatic line breaking
- ✅ Max 2 lines per subtitle
- ✅ Duration: 0.8–2.2 seconds per chunk - natural reading pace
- ✅ Emphasis: highlight keywords per caption (optional, configurable)

**Files:**
- `apps/worker/smart_chunker.py` - Core chunking logic
- `apps/worker/groq_transcribe.py` - Groq integration
- `apps/worker/transcribe.py` - Local Whisper integration
- `apps/worker/ass_writer.py` - Emphasis rendering support

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
- [x] Chunking looks natural (phrases, punctuation) vs mechanical

---

## 🎬 Phase 3 — "Better Vertical Framing": Templates + Face Tracking

**Status:** 🟡 Partially Complete

**Why:** Cropping/layout is the next quality jump. Make it deterministic and configurable.

### 3.1 Template System
- **Status:** 🟡 Partially Complete
- **Effort:** ~4 hours

**Templates:**
| Template | Description | Status |
|----------|-------------|--------|
| `center_crop` | Simple center crop | ✅ Done |
| `split_webcam` | Webcam top, game bottom | ✅ Done |
| `full_cam` | Full-camera vertical crop | ✅ Done |
| `blur_bg` | Blurred full frame as background | ❌ Not done |
| `face_track` | Dynamic crop following speaker's face | ✅ **NEW** (default for FULL_CAM) |
| `pip` | Picture-in-picture (small webcam corner) | ❌ Not done |

**Store:** `channels.settings.render_template`

**Auto-detection:** ✅ Implemented - automatically detects webcam and chooses layout:
- `FULL_CAM` - when webcam covers >70% of frame
- `SPLIT` - when webcam detected in corner
- `NO_WEBCAM` - simple center crop

### 3.2 Face Tracking
- **Status:** ✅ Complete (using OpenCV DNN)
- **Effort:** ~4-6 hours

**What we have:**
- ✅ OpenCV DNN face detection
- ✅ Gemini Vision AI webcam rectangle detection
- ✅ Face-anchored cropping
- ✅ **Dynamic per-frame tracking** with EMA smoothing
- ✅ FFmpeg expression-based interpolation between keyframes
- ✅ Automatic face following for FULL_CAM layout

### 3.3 Blur Background Template
- **Status:** Not Started
- **Effort:** ~2 hours

### Phase 3 Acceptance Criteria
- [x] Split webcam template produces stable 9:16 output
- [x] FULL_CAM mode handles webcam-only clips
- [x] Auto-detection chooses appropriate layout
- [ ] Blur BG template available
- [x] Face tracking keeps speaker centered without jitter

---

## 📦 Phase 4 — "Scale + Reliability": Observability + Storage Abstraction

**Status:** 🟡 Partially Complete

**Why:** Prepare for multiple streamers, higher volume, different storage needs.

### 4.1 Enhanced Observability
- **Status:** 🟡 Partially Complete
- **Effort:** ~2 hours

**What we have:**
- ✅ `job_events` table for logging
- ✅ Stage tracking (`last_stage`)
- ✅ Error capture in database
- ❌ Dashboard charts (failures by stage, timing)
- ❌ FFmpeg stderr captured

### 4.2 Storage Provider Abstraction
- **Status:** Not Started
- **Effort:** ~3 hours

**Providers:**
- ✅ `gdrive` (implemented)
- ❌ `s3` (AWS/MinIO)
- ❌ `azure_blob`
- ❌ `local` (for dev/testing)

### 4.3 Enhanced Rate Limiting
- **Status:** ✅ Complete
- **Effort:** ~1 hour

**Implemented:**
- ✅ Per-channel cooldown
- ✅ Per-user cooldown
- ✅ Block on active job
- ✅ Duplicate clip prevention
- ✅ Human-readable error messages

### Phase 4 Acceptance Criteria
- [x] Basic event logging to database
- [x] Rate limits prevent abuse
- [ ] Can diagnose failures from dashboard without Docker logs
- [ ] Switching storage provider requires only env var change

---

## 🚀 Phase 5 — "Content Engine": VOD Extraction + Auto-Titles + Quality Gate

**Status:** 🟡 Partially Complete

### 5.1 VOD Segment Extraction
- **Status:** Not Started
- **Effort:** ~3 hours

### 5.2 Quality Gate
- **Status:** Not Started
- **Effort:** ~2 hours

### 5.3 Auto-Generated Metadata
- **Status:** 🟡 Partially Complete
- **Effort:** ~2 hours

**What we have:**
- ✅ Filename generated from transcript words
- ✅ `transcript_text` stored for preview/search
- ❌ Suggested title generation
- ❌ Suggested caption/hashtags
- ❌ Keywords extraction

### 5.4 Thumbnail Generation
- **Status:** 🟡 Partially Complete
- **Effort:** ~1 hour

**What we have:**
- ✅ Twitch thumbnail URL stored (`thumbnail_url`)
- ❌ Custom thumbnail generation
- ❌ Multiple sizes

### Phase 5 Acceptance Criteria
- [ ] Can extract VOD segments without Twitch clip limitations
- [ ] Low-quality clips auto-discarded with clear reason
- [x] Filename generated from transcript
- [x] Thumbnail available for preview

---

## 🌟 Phase 6 — "Full Automation": Auto-Upload + Learning Loop

**Status:** Not Started

### 6.1 Auto-Upload to Platforms
- **Status:** Not Started
- **Effort:** ~6-8 hours

**Platforms:**
| Platform | API | Notes |
|----------|-----|-------|
| TikTok | Content Posting API | Requires app approval |
| YouTube Shorts | YouTube Data API v3 | OAuth required |
| Instagram Reels | Instagram Graph API | Business account only |

### 6.2 Learning Loop
- **Status:** Not Started
- **Effort:** ~3 hours

### 6.3 Background Music
- **Status:** Not Started
- **Effort:** ~3 hours

### Phase 6 Acceptance Criteria
- [ ] Clips auto-upload to configured platforms
- [ ] System learns from approvals/rejections
- [ ] Background music enhances without overpowering speech

---

## 🆕 Unplanned Features (Added Since Roadmap)

These features were implemented outside the original roadmap:

| Feature | Description | Status |
|---------|-------------|--------|
| **Groq Transcription** | Cloud-based Whisper API (faster than local) | ✅ Done |
| **Gemini Vision** | AI webcam detection with frame analysis | ✅ Done |
| **FULL_CAM Mode** | Layout for webcam-only clips | ✅ Done |
| **Dynamic Face Tracking** | Follow speaker's face throughout video | ✅ Done |
| **Game Categories** | Track and filter by game/category | ✅ Done |
| **Twitch OAuth Dashboard** | Proper auth instead of shared password | ✅ Done |
| **Mobile UI Redesign** | TikTok/Instagram-style mobile experience | ✅ Done |
| **Layout Caching** | Skip re-detection on re-renders | ✅ Done |
| **Force Reprocess** | Bypass duplicate detection for testing | ✅ Done |

---

## 📋 Updated Implementation Order

### ✅ Completed
1. ~~Discord Notifications (Phase 1.4)~~ - Skipped
2. ✅ Schema updates (Phase 1.3)
3. ✅ Retry mechanics (Phase 1.5)
4. ✅ Dashboard API endpoints (Phase 1.2)
5. ✅ Next.js Dashboard (Phase 1.1)
6. ✅ Partial template system (Phase 3.1)
7. ✅ Basic observability (Phase 4.1)
8. ✅ Rate limiting (Phase 4.3)

### Next Up (Suggested)
1. Caption presets (Phase 2.1) - ~3 hours
2. Smart chunking (Phase 2.2) - ~2 hours
3. Safe zones (Phase 2.3) - ~1 hour
4. Blur background template (Phase 3.3) - ~2 hours
5. Discord notifications (Phase 1.4) - ~1 hour

### Future
6. VOD extraction (Phase 5.1) - ~3 hours
7. Storage abstraction (Phase 4.2) - ~3 hours
8. Auto-upload (Phase 6.1) - ~6-8 hours

---

## 🛠 Technical Considerations

| Topic | Notes |
|-------|-------|
| **GPU Support** | Would 10x speed up FFmpeg (Whisper now uses Groq cloud) |
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
