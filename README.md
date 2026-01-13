# Stream2Short

Turn Twitch clips into social-ready vertical videos with automatic subtitles.

A **!clip** command in your Twitch chat creates a vertical 9:16 video with burned-in subtitles, ready for TikTok, YouTube Shorts, or Instagram Reels.

## Features

- 🎬 **Automatic clip creation** via StreamElements `!clip` command
- 📱 **Vertical video rendering** (1080x1920) optimized for social platforms
- 📝 **AI-powered subtitles** using Whisper for accurate transcription
- 🎤 **Speaker diarization** - color subtitles by speaker (white/yellow)
- 📷 **Webcam detection** - auto-detects face and creates split layout
- 👥 **Multi-streamer support** - works for multiple connected streamers
- 📦 **Google Drive storage** - organized by streamer and date
- 🔄 **Job queue** - handles multiple clip requests efficiently

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
                        └─────────────┘     │  • Twitch API   │
                                            │  • Whisper      │
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

## License

MIT
