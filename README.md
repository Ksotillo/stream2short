# Stream2Short

Turn Twitch clips into social-ready vertical videos with automatic subtitles.

A **!clip** command in your Twitch chat creates a vertical 9:16 video with burned-in subtitles, ready for TikTok, YouTube Shorts, or Instagram Reels.

## Features

- 🎬 **Automatic clip creation** via StreamElements `!clip` command
- 📱 **Vertical video rendering** (1080x1920) optimized for social platforms
- 📝 **AI-powered subtitles** using Whisper for accurate transcription
- 🎤 **Speaker diarization** - color subtitles by speaker (white/yellow)
- 📷 **Webcam detection** - YOLOv8 model (98% accuracy, ~0.92 mean IoU) auto-detects webcam overlay and creates the right layout
- 👥 **Multi-streamer support** - works for multiple connected streamers
- 📦 **Google Drive storage** - organized by streamer and date
- 🔄 **Job queue** - handles multiple clip requests efficiently
- 🖥️ **Dashboard** - web UI to review, approve, retry, and re-render clips

## Folder Structure in Google Drive

Clips are automatically organized:

```
Stream2Short/
├── StreamerName/
│   ├── 2026-01-10/
│   │   ├── clip_abc123_final.mp4
│   │   └── clip_def456_final.mp4
│   └── 2026-01-11/
│       └── clip_xyz789_final.mp4
└── AnotherStreamer/
    └── 2026-01-10/
        └── clip_ghi012_final.mp4
```

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  StreamElements │────▶│   Fastify    │────▶│     Redis       │
│   !clip command │     │     API      │     │     Queue       │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                               │                      │
                               │                      ▼
                        ┌──────┴──────┐     ┌─────────────────┐
                        │  Supabase   │◀────│  Python Worker  │
                        │  Database   │     │                 │
                        └─────────────┘                                                 │  • Twitch API   │
                                            │  • Whisper      │
                                            │  • YOLOv8       │
                                            │  • FFmpeg       │
                                            │  • Google Drive │
                                            └─────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A Twitch application ([dev.twitch.tv](https://dev.twitch.tv/console/apps))
- A Supabase project ([supabase.com](https://supabase.com))
- A Google Cloud project with Drive API enabled

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/yourusername/stream2short.git
cd stream2short

# Copy environment template
cp .env.example .env
```

### 2. Set Up Google Drive

1. **Create a Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select an existing one

2. **Enable the Google Drive API**
   - Go to APIs & Services → Library
   - Search for "Google Drive API" and enable it

3. **Create a Service Account**
   - Go to APIs & Services → Credentials
   - Click "Create Credentials" → "Service Account"
   - Give it a name (e.g., "stream2short-uploader")
   - Click "Create and Continue" (skip optional steps)

4. **Download the JSON Key**
   - Click on your new service account
   - Go to "Keys" tab → "Add Key" → "Create new key"
   - Choose JSON format and download
   - Save it as `./credentials/service-account.json`

5. **Share Your Google Drive Folder**
   - Create a folder in your Google Drive called "Stream2Short"
   - Right-click → Share → Add the service account email (looks like `name@project.iam.gserviceaccount.com`)
   - Give it "Editor" access
   - Copy the folder ID from the URL: `https://drive.google.com/drive/folders/THIS_IS_THE_FOLDER_ID`
   - Add the folder ID to your `.env` as `GOOGLE_DRIVE_FOLDER_ID`

### 3. Configure Environment

Edit `.env` with your credentials:

```bash
# Required
TWITCH_CLIENT_ID=your_client_id
TWITCH_CLIENT_SECRET=your_client_secret
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SE_SHARED_SECRET=your_random_secret  # openssl rand -hex 32

# Google Drive
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_from_step_5
```

### 4. Set Up Database

Run the SQL migration in your Supabase dashboard:

1. Go to Supabase → SQL Editor
2. Copy contents of `supabase/migrations/001_initial_schema.sql`
3. Run the query

Or use Supabase CLI:

```bash
supabase db push
```

### 5. Start Services

```bash
docker compose up --build
```

This starts:
- **Redis** on port 6379
- **API** on port 3000
- **Worker** (background processor)

### 6. Connect Your Twitch Account

Visit: `http://localhost:3000/auth/twitch/start`

Complete the OAuth flow to connect your Twitch channel.

### 7. Configure StreamElements

In StreamElements, create a custom command:

**Command:** `!clip`

**Response:**
```
$(customapi https://YOUR_DOMAIN/se/clip?channel=$(channel)&user=$(user)&secret=YOUR_SHARED_SECRET)
```

For local testing, use [ngrok](https://ngrok.com) or [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/):

```bash
ngrok http 3000
# or
cloudflared tunnel --url http://localhost:3000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/auth/twitch/start` | GET | Start OAuth flow |
| `/auth/twitch/callback` | GET | OAuth callback |
| `/se/clip` | GET | StreamElements trigger (requires LIVE) |
| `/api/clip` | POST | Create NEW clip (requires LIVE) |
| `/api/process-clip` | POST | Process EXISTING clip (no need to be LIVE!) |
| `/jobs` | GET | List jobs for a channel |
| `/jobs/:id` | GET | Get job details |
| `/jobs/:id/signed-url` | GET | Get video download URL |

### Process Existing Clips (No Need to Be Live!)

Use this endpoint to process any existing Twitch clip without being live:

```bash
# Using a clip URL
curl -X POST http://localhost:3000/api/process-clip \
  -H "Content-Type: application/json" \
  -d '{"clip_url": "https://clips.twitch.tv/YourClipSlug"}'

# Or using just the clip ID
curl -X POST http://localhost:3000/api/process-clip \
  -H "Content-Type: application/json" \
  -d '{"clip_id": "YourClipSlug"}'
```

**Response:**
```json
{
  "success": true,
  "job_id": "uuid",
  "clip_id": "YourClipSlug",
  "clip_url": "https://www.twitch.tv/channel/clip/YourClipSlug",
  "clip_title": "Amazing Play!",
  "broadcaster": "streamer_name",
  "message": "Processing clip \"Amazing Play!\" - no need to be live!"
}
```

> **Note:** The clip's broadcaster must have connected their Twitch account via `/auth/twitch/start` for their clips to be processed.

### Query Jobs

```bash
# List recent jobs for a channel
curl "http://localhost:3000/jobs?channel=your_channel_name"

# Get specific job
curl "http://localhost:3000/jobs/job-uuid"
```

## Development

### Run API locally (without Docker)

```bash
cd apps/api
npm install
npm run dev
```

### Run Worker locally (without Docker)

```bash
cd apps/worker
pip install -r requirements.txt
python main.py
```

### Project Structure

```
stream2short/
├── apps/
│   ├── api/                 # Fastify API (TypeScript)
│   │   ├── src/
│   │   │   ├── routes/      # API routes
│   │   │   ├── config.ts    # Configuration
│   │   │   ├── db.ts        # Supabase client
│   │   │   ├── queue.ts     # Redis queue
│   │   │   └── twitch.ts    # Twitch OAuth
│   │   └── Dockerfile
│   │
│   └── worker/              # Python worker
│       ├── config.py        # Configuration
│       ├── db.py            # Database operations
│       ├── twitch_api.py    # Twitch API client
│       ├── transcribe.py    # Whisper transcription
│       ├── video.py         # FFmpeg processing
│       ├── webcam_detect.py # Webcam overlay detection (orchestrator)
│       ├── webcam_locator/  # YOLOv8 detection engine (fine-tuned)
│       ├── webcam_locator_bridge.py  # Adapter: locator → WebcamRegion
│       ├── models/
│       │   └── webcam_yolov8n.pt     # Trained YOLO weights (~6MB)
│       ├── storage.py       # Google Drive upload
│       ├── pipeline.py      # Processing pipeline
│       ├── main.py          # Worker entry point
│       └── Dockerfile
│
├── credentials/             # Google service account (gitignored)
├── supabase/
│   └── migrations/          # Database migrations
│
├── docker-compose.yml
├── .env.example
└── README.md
```

## Job Status Flow

```
queued → creating_clip → waiting_clip → downloading → transcribing → rendering → uploading → ready
                                                                                            ↓
                                                                                         failed
```

## Webcam Detection

The worker automatically detects whether a stream clip contains a webcam overlay and positions it correctly in the vertical frame.

### Detection Pipeline

Detection runs as a waterfall — the highest-priority strategy that returns a result wins:

| Priority | Strategy | Accuracy | Speed |
|---|---|---|---|
| 1 | **YOLOv8** (fine-tuned on Twitch clips) | Mean IoU 0.921, 98% found | ~1–2s |
| 2 | **Gemini Vision API** (if `GEMINI_API_KEY` set) | Good rough localization | ~3–6s |
| 3 | **OpenCV** (face detection + edge heuristics) | Moderate | ~1–2s |

If no strategy finds a webcam, the clip is rendered as a simple center crop.

### Layout Types

| Layout | Description |
|---|---|
| `SPLIT` | Webcam in a corner or side of the frame alongside gameplay |
| `FULL_CAM` | Clip is entirely webcam — rendered full-frame with face tracking |
| `NO_WEBCAM` | No webcam found — simple center crop of gameplay |

### YOLOv8 Model

The model (`models/webcam_yolov8n.pt`) is a YOLOv8 Nano fine-tuned on 53 hand-labeled Twitch stream clips covering a variety of streamers, games, and webcam layouts. Trained for 80 epochs with transfer learning from COCO weights.

**Supported webcam types detected:** `side_box`, `corner_overlay`, `full_cam`, `top_band`, `bottom_band`, `center_box`

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | If set, enables Gemini as Strategy 2 fallback |

### Troubleshooting Webcam Detection

```bash
# Watch detection logs for the next job
docker compose logs -f worker

# Look for these lines:
# ✅ [Strategy 0] YOLO webcam_locator: side_box @ (980,10) 290x220  pos=top-right  conf=0.87
# ℹ️ [Strategy 0] YOLO: no webcam detected, trying Gemini...
# ⚠️ [Strategy 0] YOLO failed (...), falling back to Gemini+OpenCV
```

If YOLO consistently misses a specific layout, the model can be retrained using the `webcam-locator` companion project.

---

## Troubleshooting

### OAuth fails

- Verify `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET`
- Check redirect URI matches exactly (including trailing slash)
- Ensure Twitch app is not in "Testing" mode if you want others to connect

### Clip not available after polling

- Twitch clips take a few seconds to process
- The worker retries automatically
- If it fails repeatedly, the streamer might not be live

### FFmpeg fails

- Check worker logs: `docker compose logs worker`
- Ensure the downloaded clip is valid
- Try increasing `CLIP_POLL_MAX_ATTEMPTS` if clips aren't ready

### Google Drive upload fails

- Verify service account JSON is in `./credentials/service-account.json`
- Check that the service account email has access to the target folder
- Ensure the Google Drive API is enabled in your Google Cloud project
- Check worker logs: `docker compose logs worker`

### Token refresh fails

- Streamer may need to re-authenticate
- Check if Twitch app permissions changed
- Visit `/auth/twitch/start` to reconnect

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `API_PORT` | No | API port (default: 3000) |
| `BASE_URL` | No | Public URL for OAuth callbacks |
| `SE_SHARED_SECRET` | Yes | Secret for StreamElements validation |
| `TWITCH_CLIENT_ID` | Yes | Twitch app client ID |
| `TWITCH_CLIENT_SECRET` | Yes | Twitch app client secret |
| `TWITCH_REDIRECT_URI` | No | OAuth redirect URI |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service role key |
| `REDIS_URL` | No | Redis connection URL |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | No* | Path to service account JSON |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | No* | Service account JSON as string |
| `GOOGLE_DRIVE_FOLDER_ID` | No | Root folder ID for uploads |
| `WHISPER_MODEL` | No | Whisper model size (default: small) |
| `MAX_ATTEMPTS` | No | Max job retry attempts (default: 2) |

*One of `GOOGLE_SERVICE_ACCOUNT_FILE` or `GOOGLE_SERVICE_ACCOUNT_JSON` is required for uploads.

### Disk Space Management

Prevents disk space exhaustion from accumulated temp files:

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_DISK_SPACE_GB` | 2.0 | Minimum free disk space required before processing |
| `CLEANUP_TEMP` | true | Auto-delete temp files after each job |

**How it works:**

1. **Pre-job disk check:** Before processing starts, the worker verifies there's at least `MIN_DISK_SPACE_GB` free. If not, the job fails immediately with a clear error.

2. **Guaranteed cleanup:** Each job uses its own temp directory. The cleanup happens automatically when the job finishes - even if it crashes or errors out.

3. **Stale directory cleanup:** On startup, the worker removes any temp directories older than 24 hours (leftovers from crashed containers).

**Disk usage per clip (approximate):**
- Downloaded video: 20-100MB
- Extracted audio (WAV): 10-50MB
- Subtitle files: <1MB
- Two rendered outputs: 40-200MB each
- **Total per job:** ~100-500MB (cleaned up after upload)

**Troubleshooting low disk space:**
```bash
# Check disk usage on the VM
df -h

# Manually clear temp files
rm -rf /tmp/stream2short/*

# Check Docker volumes
docker system df
docker system prune -a  # Remove unused containers/images
```

### Anti-Spam / Cooldown Settings

Prevents chat spam and duplicate processing:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHANNEL_COOLDOWN_SECONDS` | 30 | Minimum seconds between clips for the same channel |
| `USER_COOLDOWN_SECONDS` | 60 | Minimum seconds between clips for the same user |
| `BLOCK_ON_ACTIVE_JOB` | true | Block new clips if one is already processing |
| `BLOCK_DUPLICATE_CLIPS` | true | Prevent re-processing the same Twitch clip |

**How it works:**

1. **Per-channel cooldown (30s default):** If anyone created a clip for this channel in the last 30 seconds, new requests are rejected. This prevents multiple users from spamming `!clip` at the same time.

2. **Per-user cooldown (60s default):** The same user can't trigger clips more than once per minute. Prevents individual spam.

3. **Active job blocking:** If a clip is currently being processed for the channel, new requests are rejected. Only one clip at a time per channel.

4. **Duplicate clip prevention:** Once a Twitch clip ID has been processed, it won't be processed again. The `/api/process-clip` endpoint returns the existing job info if you try to re-process.

**Example responses when rate-limited:**

```
⏳ Please wait 15s before creating another clip.
⏳ A clip is already being processed. Please wait.
```

**Database migration:** Run `supabase/migrations/002_antispam_constraints.sql` in your Supabase SQL editor to add the unique constraint on `twitch_clip_id`.

### Audio Preprocessing

Improves transcription accuracy and speaker detection:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AUDIO_NORMALIZATION` | true | Apply EBU R128 loudness normalization |

**How it works:**

1. **Single extraction:** Audio is extracted once and shared by both Whisper (transcription) and pyannote (diarization)

2. **Loudness normalization:** Uses FFmpeg's `loudnorm` filter (EBU R128 standard):
   - Target: -16 LUFS (optimal for speech)
   - Consistent levels regardless of source volume

3. **High-pass filter:** Removes low-frequency noise below 80Hz

4. **Consistent format:** 16kHz mono WAV (required by both Whisper and pyannote)

**Benefits:**
- Improved speech recognition accuracy
- Better speaker separation in diarization
- Consistent results across clips with varying audio levels

### Speaker Diarization (Optional)

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | If diarization | Hugging Face read token |
| `ENABLE_DIARIZATION` | No | Enable speaker diarization (default: false) |
| `DIARIZATION_MODEL` | No | pyannote model (default: pyannote/speaker-diarization-3.1) |
| `PRIMARY_SPEAKER_STRATEGY` | No | "most_time" or "first" (default: most_time) |

**Diarization Setup:**
1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
2. Accept model conditions for:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Create a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set `HF_TOKEN` and `ENABLE_DIARIZATION=true`

**Per-channel toggle:** Update a channel's `settings` JSON in Supabase to override the global setting:

```sql
-- Enable diarization for a specific channel
UPDATE channels 
SET settings = jsonb_set(settings, '{enable_diarization}', 'true')
WHERE twitch_login = 'streamer_name';

-- Disable diarization for a specific channel (even if globally enabled)
UPDATE channels 
SET settings = jsonb_set(settings, '{enable_diarization}', 'false')
WHERE twitch_login = 'streamer_name';
```

## Dashboard

The dashboard is a web UI for managing and reviewing clips. It's designed to be deployed on **Vercel** (free).

### Deploy to Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New Project**
2. Import your GitHub repo
3. Set **Root Directory** to `apps/dashboard`
4. Add environment variables:

| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_API_URL` | Your API URL (e.g., `https://your-vm:3443`) |
| `DASHBOARD_API_KEY` | Same secret key as in your API's `.env` |

5. Deploy!

### Access

- **URL:** Your Vercel URL (e.g., `https://stream2short-dashboard.vercel.app`)
- **Auth:** Requires `DASHBOARD_API_KEY` for API calls

### Features

| Page | Description |
|------|-------------|
| `/` | Overview with stats and recent jobs |
| `/jobs` | Full job list with filters (channel, status, review) |
| `/jobs/[id]` | Job details with video preview, transcript, actions |

### Actions

| Action | When Available | Description |
|--------|----------------|-------------|
| **Approve** | Job ready | Mark clip as approved |
| **Reject** | Job ready | Mark as rejected with notes |
| **Retry** | Job failed | Re-queue job from start |
| **Re-render** | Job ready | Re-render with different preset |

### Environment Variables

```bash
# API (.env)
DASHBOARD_API_KEY=your-secret-key     # Required to access dashboard endpoints
PUBLIC_DASHBOARD_URL=http://...       # Used in notifications (optional)
```

### Database Migration

Run this migration to add review columns:

```sql
-- supabase/migrations/003_dashboard_review.sql
-- Adds: review_status, review_notes, reviewed_at, last_stage, render_preset
```

## License

MIT
