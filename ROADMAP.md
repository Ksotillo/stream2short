# Stream2Short Roadmap

Future improvements and feature ideas for Stream2Short.

## 🎯 High Impact, Low Effort

### Discord/Webhook Notifications
- **Status:** Not Started
- **Effort:** ~1 hour
- **Description:** Send webhook notification when clip processing is complete
- **Features:**
  - Discord webhook integration
  - Include clip title, Google Drive link, thumbnail preview
  - Configurable webhook URL per channel
- **Why:** Know when clips are ready without checking logs or Drive

### Simple Web Dashboard
- **Status:** Not Started
- **Effort:** ~2-3 hours
- **Description:** Basic web UI to monitor and manage clips
- **Features:**
  - List recent jobs with status (queued, processing, ready, failed)
  - View Google Drive links for completed clips
  - Retry failed jobs with one click
  - Filter by channel/date
- **Why:** Better visibility than checking Docker logs

### Thumbnail Generation
- **Status:** Not Started
- **Effort:** ~1 hour
- **Description:** Auto-extract the best frame as a thumbnail
- **Features:**
  - Extract frame at peak action/speech
  - Save alongside video in Google Drive
  - Multiple sizes (1080x1920, 720x1280)
- **Why:** Ready for upload to social platforms

---

## 🚀 High Impact, Medium Effort

### VOD Segment Extraction
- **Status:** Not Started
- **Effort:** ~2 hours
- **Description:** Extract any segment from VODs without Twitch's 60s clip limit
- **Features:**
  - New endpoint: `POST /api/extract-vod`
  - Specify VOD ID, start time, duration (up to 5 minutes)
  - Works even when streamer is offline
  - Same processing pipeline (transcription, vertical render, etc.)
- **Why:** Full control over clip length, not limited by Twitch clips
- **Example:**
  ```json
  POST /api/extract-vod
  {
    "vod_id": "1234567890",
    "start_time": "01:23:45",
    "duration": 90
  }
  ```

### Custom Watermark/Branding
- **Status:** Not Started
- **Effort:** ~2 hours
- **Description:** Add logos, intros, and outros to clips
- **Features:**
  - Overlay logo/watermark (configurable position, size, opacity)
  - Add intro video (3-5 seconds)
  - Add outro video with call-to-action
  - Per-channel branding settings
- **Why:** Professional look, brand recognition

### Multiple Output Presets
- **Status:** Not Started
- **Effort:** ~2 hours
- **Description:** Generate clips optimized for different platforms
- **Features:**
  - TikTok: 1080x1920, 9:16, max 3 minutes
  - YouTube Shorts: 1080x1920, 9:16, max 60 seconds
  - Instagram Reels: 1080x1920, 9:16, max 90 seconds
  - Twitter/X: 1280x720, 16:9
  - Custom presets
- **Why:** Each platform has different specs and preferences

### Auto-Upload to Platforms
- **Status:** Not Started
- **Effort:** ~4-6 hours
- **Description:** Automatically upload processed clips to social platforms
- **Features:**
  - TikTok upload via API
  - YouTube Shorts upload via API
  - Instagram Reels (requires business account)
  - Configurable per channel
  - Draft mode option (upload but don't publish)
- **Why:** Eliminate manual upload step entirely

---

## 💡 Nice to Have

### Subtitle Style Presets
- **Status:** Not Started
- **Description:** Pre-configured subtitle styles
- **Presets:**
  - "TikTok Viral" - Bold, animated, colorful
  - "News/Professional" - Clean, readable, minimal
  - "Gaming" - Neon, glowing effects
  - "Minimal" - Small, unobtrusive
- **Why:** Quick style changes without config diving

### Background Music
- **Status:** Not Started
- **Description:** Add royalty-free music under speech
- **Features:**
  - Library of royalty-free tracks
  - Auto-duck music during speech
  - Configurable volume levels
- **Why:** More engaging content

### Clip Trimming
- **Status:** Not Started
- **Description:** Adjust start/end points before processing
- **Features:**
  - Preview clip in browser
  - Drag handles to trim
  - Save trimmed version
- **Why:** Fine-tune clips before rendering

### Batch Processing UI
- **Status:** Not Started
- **Description:** Process multiple clips at once
- **Features:**
  - Select multiple existing clips
  - Queue all for processing
  - Track progress of batch
- **Why:** Efficient for catching up on old clips

### Analytics Dashboard
- **Status:** Not Started
- **Description:** Track usage and performance metrics
- **Features:**
  - Processing times per clip
  - Storage usage over time
  - Clips per channel/day
  - Error rates and common failures
- **Why:** Understand usage patterns, optimize resources

---

## ✅ Completed Features

### Core MVP
- [x] Twitch OAuth integration
- [x] StreamElements `!clip` command support
- [x] Clip creation via Twitch API
- [x] Video download via yt-dlp
- [x] AI transcription (Whisper)
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
- [x] `/se/clip` - StreamElements trigger
- [x] `/api/clip` - Direct clip creation (requires LIVE)
- [x] `/api/process-clip` - Process existing clip (no LIVE needed)
- [x] `/jobs` - List/query jobs

---

## 📝 Notes

### Priority Order (Recommended)
1. Discord Notifications - Quick win, big QoL improvement
2. VOD Segment Extraction - Removes Twitch clip limitations
3. Simple Dashboard - Better monitoring and management
4. Custom Branding - Professional output
5. Auto-Upload - Full automation

### Technical Considerations
- **GPU Support:** Would significantly speed up Whisper and video rendering
- **Horizontal Scaling:** Multiple workers for parallel processing
- **CDN Integration:** Faster delivery than Google Drive links
- **Database:** Consider PostgreSQL directly vs Supabase for high volume

---

*Last updated: January 2026*

