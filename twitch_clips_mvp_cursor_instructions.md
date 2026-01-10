# Twitch “!clip → social-ready vertical” MVP (Cursor Implementation Guide)

> **Project choices (confirmed):**
> - ✅ **Multi-streamer from day 1**
> - ✅ **Minimal Supabase schema**
> - ✅ **StreamElements `$(customapi ...)` trigger**
> - ✅ **Monorepo**
> - ✅ Existing Docker setup already in the repo (extend it; don’t replace it)
> - ✅ Future-friendly for an optional **Next.js dashboard**

---

## 0) What Cursor should do (high-level)

Build a monorepo MVP that lets a Twitch streamer’s chat run `!clip`, which triggers a backend job that:

1. Creates a Twitch clip (for that streamer)
2. Waits until the clip is available
3. Fetches an official download URL for the clip
4. Downloads the MP4
5. Generates:
   - Vertical 9:16 version (1080x1920)
   - Burned-in subtitles (Whisper transcript → SRT/ASS → FFmpeg)
6. Uploads the final MP4 to private storage
7. Stores a job record in Supabase so an editor can review & upload manually

**Important:** StreamElements `$(customapi)` is **GET-only**, so the trigger endpoint must return immediately with a short plain-text message like “Queued ✅ job=…”. The heavy work runs asynchronously in a worker.

---

## 1) Monorepo layout (add to existing repo)

Create (or align to) this structure:

```
/
  apps/
    api/                 # TypeScript Fastify API (webhook + OAuth + queue producer)
    worker/              # Python worker (Whisper + FFmpeg + uploads)
    dashboard/           # (Optional later) Next.js app for reviewing jobs
  packages/
    shared/              # shared types/constants (optional for MVP)
  supabase/
    migrations/          # SQL migrations for minimal schema
  infra/                 # docker/compose additions if you keep infra separate
  docker-compose.yml     # already exists; extend it, don’t rewrite
  .env.example
  README.md
```

If the repo already has a different layout, **adapt the changes** but preserve the intent:
- API + Worker as separately buildable services
- Shared env
- One compose file to run everything locally

---

## 2) Services to add (to existing Docker Compose)

Add/ensure these services exist in your existing compose:

- `redis` (queue broker)
- `api` (Fastify)
- `worker` (Python; consumes queue)
- (optional) `dashboard` (Next.js placeholder; can be added later)

**Do not remove** existing services. Add new ones.

### Compose conventions
- Use `.env` for secrets
- Bind mount code for local dev (optional)
- Use named volumes for caches/temp if needed

**Queue recommendation:** BullMQ (Redis) from the API; Worker can either:
- (A) consume BullMQ directly (Python implementation) — *not ideal for MVP*
- (B) Have the API enqueue jobs in Redis *and* expose a “claim job” endpoint for the worker — *OK*
- (C) Use Supabase as the queue (polling) — *simplest MVP, lowest infra*
- (D) Use a Redis list/stream that Python reads from — *simple and language-agnostic*

✅ For this MVP, choose **(D) Redis list** because it’s easiest cross-language:

- API: `LPUSH clip_jobs_queue <job_id>`
- Worker: `BRPOP clip_jobs_queue 0` (blocking pop)

Retries can be implemented by re-queueing and tracking attempt count in Supabase.

---

## 3) Minimal Supabase schema (required for multi-streamer)

Create a migration SQL file in `supabase/migrations/` that defines:

### 3.1 `channels`
Stores one record per streamer (tenant).

Fields:
- `id` UUID PK (default gen)
- `twitch_broadcaster_id` TEXT UNIQUE NOT NULL
- `twitch_login` TEXT
- `display_name` TEXT
- `created_at` TIMESTAMPTZ default now()
- `settings` JSONB default `{}`  
  (keep minimal; later add template/subtitle preset here)

### 3.2 `oauth_tokens`
Stores OAuth tokens per channel (refreshable).

Fields:
- `channel_id` UUID PK FK → channels(id)
- `access_token` TEXT NOT NULL
- `refresh_token` TEXT NOT NULL
- `scopes` TEXT[] NOT NULL
- `expires_at` TIMESTAMPTZ NOT NULL
- `updated_at` TIMESTAMPTZ default now()

> **Note:** Keep tokens in DB for MVP. Later you can encrypt refresh tokens at the app layer.

### 3.3 `clip_jobs`
Stores jobs and progress.

Fields:
- `id` UUID PK default gen
- `channel_id` UUID FK → channels(id)
- `requested_by` TEXT
- `source` TEXT NOT NULL default 'streamelements'  -- or 'clip_url'
- `status` TEXT NOT NULL  -- queued|creating_clip|waiting_clip|downloading|transcribing|rendering|uploading|ready|failed
- `attempt_count` INT NOT NULL default 0
- `twitch_clip_id` TEXT
- `twitch_clip_url` TEXT
- `raw_video_path` TEXT
- `final_video_path` TEXT
- `final_video_url` TEXT
- `error` TEXT
- `created_at` TIMESTAMPTZ default now()
- `updated_at` TIMESTAMPTZ default now()

Indexes:
- `(channel_id, created_at desc)`
- `(status)`
- `(twitch_clip_id)` optional

Cursor: implement the SQL migration and also an `update_updated_at` trigger.

---

## 4) Twitch OAuth (multi-streamer onboarding)

### 4.1 API endpoints (apps/api)
Create these endpoints:

- `GET /auth/twitch/start`
  - params: `channel` (optional)  
  - redirects to Twitch OAuth authorize URL
  - `state` contains a signed payload or random nonce stored server-side

- `GET /auth/twitch/callback`
  - exchanges code for tokens
  - calls Twitch “Get Users” to identify broadcaster user
  - upserts `channels`
  - upserts `oauth_tokens`
  - responds with a success page (“Connected. You may close this tab.”)

**Scopes**
For MVP you need the scopes required to:
- Create clips
- Fetch clip download URL
- (Optional) post a chat message back

Implement scopes as a constant list and store granted scopes.

**Token refresh**
Implement a helper: `getValidAccessToken(channel_id)`:
- If `expires_at` is near (e.g., < 2 minutes), refresh using Twitch token endpoint
- Update DB
- Return valid token

---

## 5) StreamElements trigger design (GET-only)

### 5.1 API endpoint
Create a GET endpoint:

- `GET /se/clip`
  - query params:
    - `broadcaster` (preferred) OR `channel` (login)
    - `user` (who requested; passed by StreamElements)
    - `secret` or `sig` (to prevent random internet abuse)

Behavior:
1. Validate `secret` (simple shared secret in querystring is OK for MVP)
2. Find the channel record by `twitch_login` or `twitch_broadcaster_id`
   - If not found: return “Channel not connected. Have streamer auth first.”
3. Insert `clip_jobs` row with status `queued`
4. Push job ID into Redis queue: `LPUSH clip_jobs_queue <job_id>`
5. Return plain text:  
   - `Queued ✅ job=<uuid>. It will appear in the review queue soon.`

**Important:** Always return HTTP 200 with a short string.

### 5.2 StreamElements command
In StreamElements, create a custom command like:

`!clip` → `$(customapi https://YOUR_DOMAIN/se/clip?channel=$(channel)&user=$(user)&secret=YOUR_SHARED_SECRET)`

If `$(channel)` returns the login name, map it to `channels.twitch_login`.

---

## 6) Worker pipeline (apps/worker)

### 6.1 Worker loop
Implement a Python service that:

- Connects to Redis
- `BRPOP clip_jobs_queue 0` to receive a `job_id`
- Loads the job + channel + tokens from Supabase (via Supabase REST or direct Postgres connection)
- Runs the pipeline steps
- Updates `clip_jobs.status` after every step
- On failure:
  - set `status=failed`, `error=<message>`
  - optionally requeue if `attempt_count < MAX_ATTEMPTS`

### 6.2 Pipeline steps (MVP)

#### Step A: Create clip
From worker (or from API—choose one; **prefer worker** to keep API fast):
- call Twitch “Create Clip” for the channel
- store `twitch_clip_id`
- store `twitch_clip_url` if returned
- update status: `creating_clip` → `waiting_clip`

#### Step B: Poll until clip is available
- poll “Get Clips” for that `clip_id` every ~1s up to ~15s
- if not found: fail with a clear error

#### Step C: Get clip download URL
- call Twitch “Get Clip Download”
- select the best quality mp4 URL
- download to local disk, e.g. `/tmp/<job_id>/raw.mp4`
- store `raw_video_path`

#### Step D: Transcribe (Whisper)
Use `faster-whisper`:
- model: `small` (MVP balance) or `medium` if you can afford
- output SRT with timestamps
- store `captions.srt`

#### Step E: Render vertical + subtitles (FFmpeg)
Produce 1080x1920 output:

**Simple vertical template (MVP)**
- Center-crop to 9:16:
  - scale to fill height, crop width OR vice-versa depending on aspect

Example FFmpeg filter chain (adjust if needed):

- `scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920`

Then burn subtitles:
- easiest: use SRT via `subtitles=` filter
- better styling: convert to ASS later (optional)

Output: `/tmp/<job_id>/final.mp4`

#### Step F: Upload final mp4
Upload to S3-compatible storage (recommended):
- key: `final/<broadcaster_id>/<job_id>.mp4`
- return a private URL or store the key + generate signed URL on-demand

Update job:
- `final_video_path` and/or `final_video_url`
- status: `ready`

---

## 7) Storage (S3/R2) for MVP

Implement a tiny upload helper in the worker:
- Use AWS SDK compatible (boto3) with:
  - `S3_ENDPOINT`
  - `S3_ACCESS_KEY`
  - `S3_SECRET_KEY`
  - `S3_BUCKET`
- Make uploads private by default
- Store object key in DB
- Optional: API endpoint to generate a signed URL for a job

---

## 8) API endpoints for review (minimal)

Add these endpoints to `apps/api`:

- `GET /jobs?channel=<login_or_id>&limit=50`
  - returns recent jobs with status and URLs/paths
- `GET /jobs/:id`
  - returns one job
- (optional) `GET /jobs/:id/signed-url`
  - returns a signed URL for playback/download if your bucket is private

> This is enough to later plug into a Next.js dashboard.

---

## 9) Environment variables (root .env.example)

Create or update `.env.example` with:

### API
- `API_PORT=...`
- `BASE_URL=https://yourdomain`
- `SE_SHARED_SECRET=...`

### Twitch
- `TWITCH_CLIENT_ID=...`
- `TWITCH_CLIENT_SECRET=...`
- `TWITCH_REDIRECT_URI=${BASE_URL}/auth/twitch/callback`

### Supabase
- `SUPABASE_URL=...`
- `SUPABASE_SERVICE_ROLE_KEY=...` (server-side only)
- `SUPABASE_ANON_KEY=...` (if needed for dashboard later)

### Redis
- `REDIS_URL=redis://redis:6379/0`

### Storage (S3/R2)
- `S3_ENDPOINT=...`
- `S3_BUCKET=...`
- `S3_ACCESS_KEY=...`
- `S3_SECRET_KEY=...`
- `S3_REGION=auto` (R2) or your region

### Worker tuning
- `WHISPER_MODEL=small`
- `MAX_ATTEMPTS=2`

---

## 10) Local dev checklist (what Cursor should document in README)

1. Copy `.env.example` → `.env` and fill values
2. Run compose:
   - `docker compose up --build`
3. Apply Supabase migration:
   - Either via Supabase CLI or manual SQL in Supabase dashboard
4. Connect a streamer:
   - Visit `http://localhost:<API_PORT>/auth/twitch/start`
   - Complete OAuth
5. Configure StreamElements command:
   - `!clip` uses `$(customapi ...)` against your public URL (ngrok/cloudflared for local tests)
6. Trigger:
   - run `!clip` in chat and confirm:
     - job created in `clip_jobs`
     - worker processes it
     - final mp4 uploaded
     - job becomes `ready`

---

## 11) MVP constraints / non-goals (state clearly)

MVP includes:
- One vertical render template (center crop)
- Burned subtitles (basic)
- Private storage upload
- Review queue in Supabase (no UI required yet)

MVP excludes (for now):
- Face tracking / smart cropping
- Two-pane gameplay layouts
- Automatic virality scoring / LLM “quality gate”
- Auto-publishing to TikTok/Reels/Shorts
- Full dashboard (but design endpoints to support it)

---

## 12) Implementation notes (Cursor should follow)

- Keep API fast and return immediately for `/se/clip`
- Worker is responsible for heavy work
- Always update job status per step
- Write clear logs
- Ensure per-channel token refresh works; never assume tokens are valid
- Store all job outputs under channel-specific paths (multi-tenant separation)

---

## 13) Deliverables Cursor must produce

1. **Supabase migration** in `supabase/migrations/`
2. **Fastify API** in `apps/api`:
   - Twitch OAuth endpoints
   - `/se/clip` endpoint
   - job query endpoints
   - Redis queue push
3. **Python worker** in `apps/worker`:
   - Redis BRPOP loop
   - Twitch clip create + poll + download URL
   - Download mp4
   - faster-whisper transcription
   - FFmpeg render to vertical with burned subs
   - Upload to S3/R2
4. **Docker updates**:
   - add `api`, `worker`, `redis` services
   - do not break existing docker setup
5. **README updates**:
   - setup steps
   - StreamElements command snippet
   - troubleshooting section

---

## 14) (Optional) Next.js dashboard future plan (do not implement now)

Later, add `apps/dashboard` with:
- Auth (Supabase auth or simple password for MVP)
- Job list by channel
- Video preview using signed URLs
- Approve/reject buttons (future)

For now, only ensure the API has endpoints that a dashboard can call.

---

## 15) Cursor execution instructions (important)

**Cursor:**
- Do NOT create a brand-new repo.
- Do NOT delete or rewrite the existing Docker setup.
- Add only the needed services/files.
- Keep changes scoped and incremental.
- Ensure the system works for **multiple streamers** (multi-tenant):
  - tokens + jobs tied to `channels`
  - storage keys include broadcaster ID

---

## 16) Acceptance criteria (what “done” means)

A) A streamer can OAuth-connect successfully and creates a `channels` + `oauth_tokens` record.

B) StreamElements `!clip` command hits `/se/clip` and immediately returns “Queued ✅ …”.

C) A job row appears in `clip_jobs` with `status=queued`, then progresses.

D) The worker produces a `final.mp4` vertical video with burned subtitles and uploads it.

E) The job ends in `status=ready` with `final_video_url` (or storage key) stored.

F) This works for **Streamer A** and **Streamer B** independently (no cross-talk).

---

## 17) Troubleshooting checklist (include in README)

- OAuth fails → reminder: verify client id/secret/redirect URI
- Token refresh fails → log response and mark channel disconnected
- Clip not available after polling → retry create clip once
- Download URL missing → verify scopes
- FFmpeg fails → log command + stderr in job.error
- Upload fails → verify endpoint/bucket/keys

---

### Done. Cursor can now implement the MVP from this document.
