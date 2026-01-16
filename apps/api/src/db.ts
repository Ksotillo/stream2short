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
  error: string | null;
  created_at: string;
  updated_at: string;
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
      return {
        allowed: false,
        reason: 'This clip has already been processed',
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
        reason: 'A clip is already being processed for this channel',
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
        reason: `Channel cooldown active`,
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
        reason: `User cooldown active`,
        waitSeconds,
        existingJob: recentUserJob,
      };
    }
  }
  
  // All checks passed
  return { allowed: true };
}

