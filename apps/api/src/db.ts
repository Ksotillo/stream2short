import { createClient } from '@supabase/supabase-js';
import { config } from './config.js';

// Database types
export interface Channel {
  id: string;
  twitch_broadcaster_id: string;
  twitch_login: string | null;
  display_name: string | null;
  created_at: string;
  settings: Record<string, unknown>;
}

export interface OAuthToken {
  channel_id: string;
  access_token: string;
  refresh_token: string;
  scopes: string[];
  expires_at: string;
  updated_at: string;
}

export type JobStatus = 
  | 'queued'
  | 'creating_clip'
  | 'waiting_clip'
  | 'downloading'
  | 'transcribing'
  | 'rendering'
  | 'uploading'
  | 'ready'
  | 'failed';

export type ReviewStatus = 'pending' | 'approved' | 'rejected';
export type LastStage = 'download' | 'transcribe' | 'render' | 'upload';
export type RenderPreset = 'default' | 'clean' | 'boxed' | 'minimal' | 'bold';

export interface ClipJob {
  id: string;
  channel_id: string;
  requested_by: string | null;
  source: string;
  status: JobStatus;
  attempt_count: number;
  twitch_clip_id: string | null;
  twitch_clip_url: string | null;
  raw_video_path: string | null;
  final_video_path: string | null;
  final_video_url: string | null;
  no_subtitles_url: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  // Phase 1: Review & dashboard fields
  review_status: ReviewStatus | null;
  review_notes: string | null;
  reviewed_at: string | null;
  last_stage: LastStage | null;
  render_preset: RenderPreset;
  transcript_text: string | null;
}

export interface JobEvent {
  id: string;
  job_id: string;
  created_at: string;
  level: 'info' | 'warn' | 'error';
  stage: string | null;
  message: string;
  data: Record<string, unknown>;
}

// Create Supabase client with service role key (server-side only)
export const supabase = createClient(
  config.supabase.url,
  config.supabase.serviceRoleKey,
  {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  }
);

// Channel operations
export async function getChannelByLogin(login: string): Promise<Channel | null> {
  const { data, error } = await supabase
    .from('channels')
    .select('*')
    .eq('twitch_login', login.toLowerCase())
    .single();
  
  if (error) return null;
  return data as Channel;
}

export async function getChannelByBroadcasterId(broadcasterId: string): Promise<Channel | null> {
  const { data, error } = await supabase
    .from('channels')
    .select('*')
    .eq('twitch_broadcaster_id', broadcasterId)
    .single();
  
  if (error) return null;
  return data as Channel;
}

export async function upsertChannel(channel: Partial<Channel> & { twitch_broadcaster_id: string }): Promise<Channel> {
  const { data, error } = await supabase
    .from('channels')
    .upsert(channel, { onConflict: 'twitch_broadcaster_id' })
    .select()
    .single();
  
  if (error) throw new Error(`Failed to upsert channel: ${error.message}`);
  return data as Channel;
}

// Token operations
export async function getTokens(channelId: string): Promise<OAuthToken | null> {
  const { data, error } = await supabase
    .from('oauth_tokens')
    .select('*')
    .eq('channel_id', channelId)
    .single();
  
  if (error) return null;
  return data as OAuthToken;
}

export async function upsertTokens(tokens: OAuthToken): Promise<void> {
  const { error } = await supabase
    .from('oauth_tokens')
    .upsert(tokens, { onConflict: 'channel_id' });
  
  if (error) throw new Error(`Failed to upsert tokens: ${error.message}`);
}

// Job operations
export async function createJob(job: Partial<ClipJob>): Promise<ClipJob> {
  const { data, error } = await supabase
    .from('clip_jobs')
    .insert(job)
    .select()
    .single();
  
  if (error) throw new Error(`Failed to create job: ${error.message}`);
  return data as ClipJob;
}

export async function getJob(jobId: string): Promise<ClipJob | null> {
  const { data, error } = await supabase
    .from('clip_jobs')
    .select('*')
    .eq('id', jobId)
    .single();
  
  if (error) return null;
  return data as ClipJob;
}

export async function getJobsByChannel(channelId: string, limit = 50): Promise<ClipJob[]> {
  const { data, error } = await supabase
    .from('clip_jobs')
    .select('*')
    .eq('channel_id', channelId)
    .order('created_at', { ascending: false })
    .limit(limit);
  
  if (error) return [];
  return data as ClipJob[];
}

// ============================================================================
// Phase 1: Dashboard & Review Functions
// ============================================================================

/**
 * Get all connected channels
 */
export async function getAllChannels(): Promise<Channel[]> {
  const { data, error } = await supabase
    .from('channels')
    .select('*')
    .order('created_at', { ascending: false });
  
  if (error) return [];
  return data as Channel[];
}

/**
 * Filter options for getJobsWithFilters
 */
export interface JobFilters {
  channelId?: string;
  status?: JobStatus | JobStatus[];
  reviewStatus?: ReviewStatus | ReviewStatus[];
  gameId?: string;
  gameName?: string;
  dateFrom?: string;
  dateTo?: string;
  limit?: number;
  cursor?: string; // job ID for cursor-based pagination
}

/**
 * Get jobs with advanced filters and pagination
 */
export async function getJobsWithFilters(filters: JobFilters): Promise<{ jobs: ClipJob[]; nextCursor: string | null }> {
  let query = supabase
    .from('clip_jobs')
    .select('*')
    .order('created_at', { ascending: false });
  
  // Channel filter
  if (filters.channelId) {
    query = query.eq('channel_id', filters.channelId);
  }
  
  // Status filter (single or multiple)
  if (filters.status) {
    if (Array.isArray(filters.status)) {
      query = query.in('status', filters.status);
    } else {
      query = query.eq('status', filters.status);
    }
  }
  
  // Review status filter
  if (filters.reviewStatus) {
    if (Array.isArray(filters.reviewStatus)) {
      query = query.in('review_status', filters.reviewStatus);
    } else {
      query = query.eq('review_status', filters.reviewStatus);
    }
  }
  
  // Game filter
  if (filters.gameId) {
    query = query.eq('game_id', filters.gameId);
  }
  if (filters.gameName) {
    query = query.ilike('game_name', `%${filters.gameName}%`);
  }
  
  // Date range filters
  if (filters.dateFrom) {
    query = query.gte('created_at', filters.dateFrom);
  }
  if (filters.dateTo) {
    query = query.lte('created_at', filters.dateTo);
  }
  
  // Cursor-based pagination (get items after cursor)
  if (filters.cursor) {
    const cursorJob = await getJob(filters.cursor);
    if (cursorJob) {
      query = query.lt('created_at', cursorJob.created_at);
    }
  }
  
  // Limit (+1 to detect if there are more results)
  const limit = filters.limit || 50;
  query = query.limit(limit + 1);
  
  const { data, error } = await query;
  
  if (error) return { jobs: [], nextCursor: null };
  
  const jobs = data as ClipJob[];
  
  // Check if there are more results
  let nextCursor: string | null = null;
  if (jobs.length > limit) {
    const lastJob = jobs.pop(); // Remove the extra item
    nextCursor = lastJob?.id || null;
  }
  
  return { jobs, nextCursor };
}

/**
 * Get unique games for a channel (for filter UI)
 */
export async function getGamesForChannel(channelId: string): Promise<Array<{ game_id: string; game_name: string; count: number }>> {
  // Get all games with counts
  const { data, error } = await supabase
    .from('clip_jobs')
    .select('game_id, game_name')
    .eq('channel_id', channelId)
    .not('game_id', 'is', null)
    .not('game_name', 'is', null);
  
  if (error || !data) return [];
  
  // Aggregate by game
  const gameMap = new Map<string, { game_id: string; game_name: string; count: number }>();
  
  for (const job of data) {
    const key = job.game_id;
    if (!key) continue;
    
    if (gameMap.has(key)) {
      gameMap.get(key)!.count++;
    } else {
      gameMap.set(key, {
        game_id: job.game_id,
        game_name: job.game_name || 'Unknown Game',
        count: 1,
      });
    }
  }
  
  // Sort by count (most clips first)
  return Array.from(gameMap.values()).sort((a, b) => b.count - a.count);
}

/**
 * Review a job (approve/reject)
 */
export async function reviewJob(
  jobId: string,
  decision: ReviewStatus,
  notes?: string
): Promise<void> {
  const { error } = await supabase
    .from('clip_jobs')
    .update({
      review_status: decision,
      review_notes: notes || null,
      reviewed_at: new Date().toISOString(),
    })
    .eq('id', jobId);
  
  if (error) throw new Error(`Failed to review job: ${error.message}`);
}

/**
 * Reset job for retry (clear error, reset status)
 */
export async function resetJobForRetry(
  jobId: string,
  fromStage?: LastStage
): Promise<void> {
  const updates: Partial<ClipJob> = {
    status: 'queued',
    error: null,
  };
  
  // If retrying from a specific stage, we keep data from previous stages
  // Otherwise, start fresh
  if (!fromStage || fromStage === 'download') {
    updates.raw_video_path = null;
    updates.final_video_path = null;
    updates.final_video_url = null;
    updates.no_subtitles_url = null;
    updates.transcript_text = null;
    updates.last_stage = null;
  }
  
  const { error } = await supabase
    .from('clip_jobs')
    .update(updates)
    .eq('id', jobId);
  
  if (error) throw new Error(`Failed to reset job: ${error.message}`);
}

/**
 * Update job's render preset (for re-render)
 */
export async function updateJobPreset(
  jobId: string,
  preset: RenderPreset
): Promise<void> {
  const { error } = await supabase
    .from('clip_jobs')
    .update({
      render_preset: preset,
      status: 'queued', // Re-queue for rendering
      final_video_path: null,
      final_video_url: null,
      no_subtitles_url: null,
    })
    .eq('id', jobId);
  
  if (error) throw new Error(`Failed to update preset: ${error.message}`);
}

/**
 * Create a job event (for logging/debugging)
 */
export async function createJobEvent(
  jobId: string,
  level: 'info' | 'warn' | 'error',
  message: string,
  stage?: string,
  data?: Record<string, unknown>
): Promise<void> {
  const { error } = await supabase
    .from('job_events')
    .insert({
      job_id: jobId,
      level,
      message,
      stage: stage || null,
      data: data || {},
    });
  
  if (error) throw new Error(`Failed to create job event: ${error.message}`);
}

/**
 * Get events for a job
 */
export async function getJobEvents(jobId: string): Promise<JobEvent[]> {
  const { data, error } = await supabase
    .from('job_events')
    .select('*')
    .eq('job_id', jobId)
    .order('created_at', { ascending: true });
  
  if (error) return [];
  return data as JobEvent[];
}

export async function updateJob(jobId: string, updates: Partial<ClipJob>): Promise<void> {
  const { error } = await supabase
    .from('clip_jobs')
    .update(updates)
    .eq('id', jobId);
  
  if (error) throw new Error(`Failed to update job: ${error.message}`);
}

// ============================================================================
// Anti-Spam / Cooldown Functions
// ============================================================================

/**
 * Check if there's a recent job for this channel (within cooldown period).
 * Used for per-channel cooldown.
 */
export async function getRecentJobForChannel(
  channelId: string,
  withinSeconds: number
): Promise<ClipJob | null> {
  const cutoff = new Date(Date.now() - withinSeconds * 1000).toISOString();
  
  const { data, error } = await supabase
    .from('clip_jobs')
    .select('*')
    .eq('channel_id', channelId)
    .gte('created_at', cutoff)
    .order('created_at', { ascending: false })
    .limit(1)
    .maybeSingle();
  
  if (error) return null;
  return data as ClipJob | null;
}

/**
 * Check if there's a recent job by this user for this channel (within cooldown period).
 * Used for per-user cooldown.
 */
export async function getRecentJobByUser(
  channelId: string,
  requestedBy: string,
  withinSeconds: number
): Promise<ClipJob | null> {
  const cutoff = new Date(Date.now() - withinSeconds * 1000).toISOString();
  
  const { data, error } = await supabase
    .from('clip_jobs')
    .select('*')
    .eq('channel_id', channelId)
    .eq('requested_by', requestedBy)
    .gte('created_at', cutoff)
    .order('created_at', { ascending: false })
    .limit(1)
    .maybeSingle();
  
  if (error) return null;
  return data as ClipJob | null;
}

/**
 * Check if there's an active (queued or processing) job for this channel.
 * Active statuses are anything not 'ready' or 'failed'.
 * 
 * Only considers jobs created within the last 10 minutes as "active".
 * Older jobs are likely stuck and shouldn't block new requests.
 */
export async function getActiveJobForChannel(
  channelId: string,
  maxAgeMinutes: number = 10
): Promise<ClipJob | null> {
  // Only consider jobs created in the last N minutes as "active"
  // Older jobs are likely stuck and shouldn't block new requests
  const cutoff = new Date(Date.now() - maxAgeMinutes * 60 * 1000).toISOString();
  
  const { data, error } = await supabase
    .from('clip_jobs')
    .select('*')
    .eq('channel_id', channelId)
    .not('status', 'in', '("ready","failed")')
    .gte('created_at', cutoff)  // Only recent jobs
    .order('created_at', { ascending: false })
    .limit(1)
    .maybeSingle();
  
  if (error) return null;
  return data as ClipJob | null;
}

/**
 * Check if we already have a job for this Twitch clip ID.
 * Used to prevent duplicate processing of the same clip.
 */
export async function getJobByClipId(twitchClipId: string): Promise<ClipJob | null> {
  const { data, error } = await supabase
    .from('clip_jobs')
    .select('*')
    .eq('twitch_clip_id', twitchClipId)
    .limit(1)
    .maybeSingle();
  
  if (error) return null;
  return data as ClipJob | null;
}

/**
 * Delete a job by ID (for force reprocessing).
 */
export async function deleteJob(jobId: string): Promise<void> {
  // First delete any job events
  await supabase
    .from('job_events')
    .delete()
    .eq('job_id', jobId);
  
  // Then delete the job
  const { error } = await supabase
    .from('clip_jobs')
    .delete()
    .eq('id', jobId);
  
  if (error) {
    throw new Error(`Failed to delete job: ${error.message}`);
  }
}

/**
 * Result of anti-spam checks.
 */
export interface CooldownCheckResult {
  allowed: boolean;
  reason?: string;
  waitSeconds?: number;
  existingJob?: ClipJob;
}

/**
 * Comprehensive anti-spam check for clip creation.
 * Checks: channel cooldown, user cooldown, active job, duplicate clip.
 */
export async function checkCooldowns(
  channelId: string,
  requestedBy: string | null,
  twitchClipId: string | null,
  options: {
    channelCooldownSeconds: number;
    userCooldownSeconds: number;
    blockOnActiveJob: boolean;
    blockDuplicateClips: boolean;
  }
): Promise<CooldownCheckResult> {
  // 1. Check for duplicate clip ID
  if (options.blockDuplicateClips && twitchClipId) {
    const existingJob = await getJobByClipId(twitchClipId);
    if (existingJob) {
      const statusMsg = existingJob.status === 'ready' 
        ? 'Check your clips list to find it!' 
        : `Current status: ${existingJob.status}`;
      return {
        allowed: false,
        reason: `This clip has already been processed. ${statusMsg}`,
        existingJob,
      };
    }
  }
  
  // 2. Check for active job on this channel
  if (options.blockOnActiveJob) {
    const activeJob = await getActiveJobForChannel(channelId);
    if (activeJob) {
      return {
        allowed: false,
        reason: 'Another clip is currently being processed for this channel. Please wait for it to finish before creating a new one.',
        existingJob: activeJob,
      };
    }
  }
  
  // 3. Check channel cooldown
  if (options.channelCooldownSeconds > 0) {
    const recentJob = await getRecentJobForChannel(channelId, options.channelCooldownSeconds);
    if (recentJob) {
      const jobAge = (Date.now() - new Date(recentJob.created_at).getTime()) / 1000;
      const waitSeconds = Math.ceil(options.channelCooldownSeconds - jobAge);
      return {
        allowed: false,
        reason: `This channel just created a clip. Please wait ${waitSeconds} seconds before creating another one.`,
        waitSeconds,
        existingJob: recentJob,
      };
    }
  }
  
  // 4. Check per-user cooldown
  if (options.userCooldownSeconds > 0 && requestedBy) {
    const recentUserJob = await getRecentJobByUser(channelId, requestedBy, options.userCooldownSeconds);
    if (recentUserJob) {
      const jobAge = (Date.now() - new Date(recentUserJob.created_at).getTime()) / 1000;
      const waitSeconds = Math.ceil(options.userCooldownSeconds - jobAge);
      return {
        allowed: false,
        reason: `You just created a clip. Please wait ${waitSeconds} seconds before creating another one.`,
        waitSeconds,
        existingJob: recentUserJob,
      };
    }
  }
  
  // All checks passed
  return { allowed: true };
}

