const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000'
const API_KEY = process.env.DASHBOARD_API_KEY || ''

interface FetchOptions extends RequestInit {
  body?: any
}

async function apiFetch<T>(endpoint: string, options: FetchOptions = {}): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(API_KEY ? { 'x-dashboard-api-key': API_KEY } : {}),
    ...((options.headers as Record<string, string>) || {}),
  }

  const res = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers,
    body: options.body ? JSON.stringify(options.body) : undefined,
    cache: 'no-store',
  })

  if (!res.ok) {
    const error = await res.json().catch(() => ({ error: 'Unknown error' }))
    throw new Error(error.error || `API error: ${res.status}`)
  }

  return res.json()
}

// Types
export interface Channel {
  id: string
  twitch_broadcaster_id: string
  twitch_login: string | null
  display_name: string | null
  created_at: string
  settings: Record<string, unknown>
}

export interface Job {
  id: string
  channel_id: string
  requested_by: string | null
  source: string
  status: string
  attempt_count: number
  twitch_clip_id: string | null
  twitch_clip_url: string | null
  raw_video_path: string | null
  final_video_path: string | null
  final_video_url: string | null
  no_subtitles_url: string | null
  error: string | null
  created_at: string
  updated_at: string
  review_status: 'pending' | 'approved' | 'rejected' | null
  review_notes: string | null
  reviewed_at: string | null
  last_stage: string | null
  render_preset: string
  transcript_text: string | null
}

export interface JobEvent {
  id: string
  job_id: string
  created_at: string
  level: 'info' | 'warn' | 'error'
  stage: string | null
  message: string
  data: Record<string, unknown>
}

// API Functions
export async function getChannels(): Promise<{ channels: Channel[] }> {
  return apiFetch('/api/channels')
}

export interface JobsFilters {
  channel_id?: string
  status?: string
  review_status?: string
  limit?: number
  cursor?: string
}

export async function getJobs(filters: JobsFilters = {}): Promise<{
  jobs: Job[]
  pagination: { next_cursor: string | null; has_more: boolean }
}> {
  const params = new URLSearchParams()
  if (filters.channel_id) params.set('channel_id', filters.channel_id)
  if (filters.status) params.set('status', filters.status)
  if (filters.review_status) params.set('review_status', filters.review_status)
  if (filters.limit) params.set('limit', String(filters.limit))
  if (filters.cursor) params.set('cursor', filters.cursor)

  const query = params.toString()
  return apiFetch(`/api/jobs${query ? `?${query}` : ''}`)
}

export async function getJob(id: string, includeEvents = false): Promise<{
  job: Job
  events?: JobEvent[]
}> {
  return apiFetch(`/api/jobs/${id}${includeEvents ? '?include_events=true' : ''}`)
}

export async function reviewJob(
  id: string,
  decision: 'approved' | 'rejected',
  notes?: string
): Promise<{ success: boolean; message: string }> {
  return apiFetch(`/api/jobs/${id}/review`, {
    method: 'POST',
    body: { decision, notes },
  })
}

export async function retryJob(
  id: string,
  fromStage?: string
): Promise<{ success: boolean; message: string }> {
  return apiFetch(`/api/jobs/${id}/retry`, {
    method: 'POST',
    body: { from_stage: fromStage },
  })
}

export async function rerenderJob(
  id: string,
  preset: string
): Promise<{ success: boolean; message: string }> {
  return apiFetch(`/api/jobs/${id}/rerender`, {
    method: 'POST',
    body: { preset },
  })
}

