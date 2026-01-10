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

