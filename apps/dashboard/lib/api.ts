const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000'
const API_KEY = process.env.DASHBOARD_API_KEY || ''

interface FetchOptions {
  method?: string
  body?: unknown
}

async function fetchAPI<T>(endpoint: string, options: FetchOptions = {}): Promise<T> {
  const res = await fetch(`${API_URL}${endpoint}`, {
    method: options.method || 'GET',
    headers: {
      'Content-Type': 'application/json',
      'x-dashboard-api-key': API_KEY,
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
    cache: 'no-store',
  })
  
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.error || `API Error: ${res.status}`)
  }
  
  return res.json()
}

// Types
export interface TranscriptSegment {
  start: number
  end: number
  text: string
  speaker?: string
  is_primary?: boolean
}

export interface Job {
  id: string
  channel_id: string
  twitch_clip_id: string
  twitch_clip_url: string
  status: string
  final_video_url: string | null
  no_subtitles_url: string | null
  transcript_text: string | null
  transcript_segments: TranscriptSegment[] | null
  transcript_edited_at: string | null
  error: string | null
  requested_by: string | null
  review_status: string | null
  review_notes: string | null
  reviewed_at: string | null
  render_preset: string
  last_stage: string | null
  game_id: string | null
  game_name: string | null
  thumbnail_url: string | null
  created_at: string
  updated_at: string
}

export interface Game {
  game_id: string
  game_name: string
  box_art_url: string | null
  count: number
}

// Helper to format Twitch box art URL with dimensions
export function formatBoxArtUrl(url: string | null, width = 40, height = 53): string | null {
  if (!url) return null
  return url.replace('{width}', width.toString()).replace('{height}', height.toString())
}

export interface JobEvent {
  id: string
  job_id: string
  level: string
  stage: string | null
  message: string
  data: unknown
  created_at: string
}

export interface Channel {
  id: string
  twitch_broadcaster_id: string
  twitch_login: string
  display_name: string
  created_at: string
}

// API Functions
export async function getChannels(): Promise<{ channels: Channel[] }> {
  return fetchAPI('/api/channels')
}

export async function getJobs(params: {
  channel_id?: string
  status?: string
  review_status?: string
  game_id?: string
  limit?: number
  cursor?: string
} = {}): Promise<{ 
  jobs: Job[]
  pagination: { next_cursor: string | null; has_more: boolean }
}> {
  const searchParams = new URLSearchParams()
  if (params.channel_id) searchParams.set('channel_id', params.channel_id)
  if (params.status) searchParams.set('status', params.status)
  if (params.review_status) searchParams.set('review_status', params.review_status)
  if (params.game_id) searchParams.set('game_id', params.game_id)
  if (params.limit) searchParams.set('limit', params.limit.toString())
  if (params.cursor) searchParams.set('cursor', params.cursor)
  
  const query = searchParams.toString()
  return fetchAPI(`/api/jobs${query ? `?${query}` : ''}`)
}

export async function getGames(channelId: string): Promise<{ games: Game[] }> {
  return fetchAPI(`/api/games?channel_id=${channelId}`)
}

export async function getJob(id: string, includeEvents = false): Promise<{ 
  job: Job
  events?: JobEvent[] 
}> {
  return fetchAPI(`/api/jobs/${id}${includeEvents ? '?include_events=true' : ''}`)
}

export async function reviewJob(
  id: string, 
  decision: 'approved' | 'rejected',
  notes?: string
): Promise<{ success: boolean; message: string }> {
  // Use local Next.js API route (keeps API key server-side)
  const res = await fetch('/api/clips/review', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ jobId: id, decision, notes }),
  })
  
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.error || `API Error: ${res.status}`)
  }
  
  return res.json()
}

export async function retryJob(id: string): Promise<{ success: boolean; message: string }> {
  // Use local Next.js API route (keeps API key server-side)
  const res = await fetch('/api/clips/retry', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ jobId: id }),
  })
  
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.error || `API Error: ${res.status}`)
  }
  
  return res.json()
}

export async function rerenderJob(
  id: string,
  preset: string
): Promise<{ success: boolean; message: string }> {
  // Use local Next.js API route (keeps API key server-side)
  const res = await fetch('/api/clips/rerender', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ jobId: id, preset }),
  })
  
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.error || `API Error: ${res.status}`)
  }
  
  return res.json()
}

export async function processClip(
  clipUrl: string,
  requestedBy?: string
): Promise<{ 
  success: boolean
  message: string
  job_id?: string
  twitch_clip_url?: string
  error?: string
}> {
  return fetchAPI('/process-clip', {
    method: 'POST',
    body: { 
      clip_url: clipUrl,
      requested_by: requestedBy || 'dashboard',
    },
  })
}

export async function updateTranscript(
  id: string,
  segments: TranscriptSegment[]
): Promise<{ 
  success: boolean
  message: string
  segment_count: number
  transcript_text: string
}> {
  // Use local Next.js API route (keeps API key server-side)
  const res = await fetch('/api/clips/transcript', {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ jobId: id, segments }),
  })
  
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.error || `API Error: ${res.status}`)
  }
  
  return res.json()
}
