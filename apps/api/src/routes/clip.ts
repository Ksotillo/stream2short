import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { config } from '../config.js';
import { 
  getChannelByLogin, 
  getChannelByBroadcasterId, 
  createJob, 
  updateJob,
  checkCooldowns,
  getJobByClipId,
} from '../db.js';
import { enqueueJob } from '../queue.js';
import { getValidAccessToken, createClip, getClip } from '../twitch.js';

// Extract clip ID from various URL formats
function extractClipId(input: string): string {
  // If it's already just a clip ID (no URL structure)
  if (!input.includes('/') && !input.includes('.')) {
    return input;
  }
  
  // Various Twitch clip URL formats:
  // https://clips.twitch.tv/ClipSlug
  // https://www.twitch.tv/channel/clip/ClipSlug
  // https://www.twitch.tv/channel/clip/ClipSlug?filter=clips&range=7d&sort=time
  
  const patterns = [
    /clips\.twitch\.tv\/([a-zA-Z0-9_-]+)/,
    /twitch\.tv\/\w+\/clip\/([a-zA-Z0-9_-]+)/,
  ];
  
  for (const pattern of patterns) {
    const match = input.match(pattern);
    if (match) {
      return match[1];
    }
  }
  
  // Fallback: return as-is (might be just a slug)
  return input;
}

export async function clipRoutes(fastify: FastifyInstance): Promise<void> {
  // StreamElements trigger endpoint (GET-only)
  // Called via $(customapi https://domain/se/clip?channel=$(channel)&user=$(user)&secret=SECRET)
  fastify.get('/se/clip', async (
    request: FastifyRequest<{
      Querystring: {
        channel?: string;
        broadcaster?: string;
        user?: string;
        secret?: string;
      };
    }>,
    reply: FastifyReply
  ) => {
    const { channel, broadcaster, user, secret } = request.query;
    
    // Always return plain text for StreamElements
    reply.type('text/plain');
    
    // Validate shared secret
    if (!secret || secret !== config.seSharedSecret) {
      fastify.log.warn('Invalid or missing secret in /se/clip request');
      return reply.send('Unauthorized request.');
    }
    
    // Find channel by login or broadcaster ID
    let channelRecord = null;
    
    if (broadcaster) {
      channelRecord = await getChannelByBroadcasterId(broadcaster);
    } else if (channel) {
      // StreamElements $(channel) returns the channel login name
      channelRecord = await getChannelByLogin(channel.toLowerCase());
    }
    
    if (!channelRecord) {
      fastify.log.warn(`Channel not found: ${broadcaster || channel}`);
      return reply.send('Channel not connected. Have streamer auth first at /auth/twitch/start');
    }
    
    try {
      // ====================================================================
      // ANTI-SPAM CHECKS (before creating clip)
      // ====================================================================
      const cooldownResult = await checkCooldowns(
        channelRecord.id,
        user || null,
        null, // No clip ID yet - we're creating a new one
        {
          channelCooldownSeconds: config.cooldown.channelCooldownSeconds,
          userCooldownSeconds: config.cooldown.userCooldownSeconds,
          blockOnActiveJob: config.cooldown.blockOnActiveJob,
          blockDuplicateClips: false, // Not applicable for new clips
        }
      );
      
      if (!cooldownResult.allowed) {
        fastify.log.info(`Clip request blocked: ${cooldownResult.reason} (channel: ${channelRecord.twitch_login}, user: ${user})`);
        
        if (cooldownResult.waitSeconds) {
          return reply.send(`⏳ Please wait ${cooldownResult.waitSeconds}s before creating another clip.`);
        }
        
        if (cooldownResult.reason?.includes('already being processed')) {
          return reply.send(`⏳ A clip is already being processed. Please wait.`);
        }
        
        return reply.send(`⏳ ${cooldownResult.reason}`);
      }
      
      // Get valid access token for the channel
      const accessToken = await getValidAccessToken(channelRecord.id);
      
      // Create clip on Twitch immediately
      fastify.log.info(`Creating clip for channel ${channelRecord.twitch_login}...`);
      const clipData = await createClip(channelRecord.twitch_broadcaster_id, accessToken);
      
      // Build the clip URL (edit_url redirects to clip page)
      // Format: https://clips.twitch.tv/{clip_id}
      const clipUrl = `https://clips.twitch.tv/${clipData.id}`;
      
      // Create job record with clip already created
      const job = await createJob({
        channel_id: channelRecord.id,
        requested_by: user || null,
        source: 'streamelements',
        status: 'waiting_clip',  // Skip to waiting stage
        twitch_clip_id: clipData.id,
      });
      
      // Push to Redis queue for background processing
      await enqueueJob(job.id);
      
      fastify.log.info(`Clip created: ${clipData.id} for channel ${channelRecord.twitch_login} by ${user}`);
      
      // Return the clip URL immediately!
      return reply.send(`Clip created! 🎬 ${clipUrl}`);
    } catch (err: any) {
      // Handle specific Twitch errors
      const errorMsg = err?.message || String(err);
      
      if (errorMsg.includes('Channel offline')) {
        return reply.send('❌ Channel must be LIVE to create clips!');
      }
      
      if (errorMsg.includes('Token refresh failed') || errorMsg.includes('re-authenticate')) {
        return reply.send('❌ Auth expired. Streamer needs to reconnect at /auth/twitch/start');
      }
      
      fastify.log.error({ err }, 'Failed to create clip');
      return reply.send('❌ Failed to create clip. Try again later.');
    }
  });
  
  // Direct clip trigger (can be used for testing or alternative integrations)
  // NOTE: This creates a NEW clip - requires channel to be LIVE
  fastify.post('/api/clip', async (
    request: FastifyRequest<{
      Body: {
        channel_id?: string;
        broadcaster_id?: string;
        channel?: string;
        requested_by?: string;
      };
    }>,
    reply: FastifyReply
  ) => {
    const { channel_id, broadcaster_id, channel, requested_by } = request.body || {};
    
    let channelRecord = null;
    
    if (channel_id) {
      // Direct channel ID lookup would need a new function, use broadcaster for now
      channelRecord = await getChannelByBroadcasterId(channel_id);
    } else if (broadcaster_id) {
      channelRecord = await getChannelByBroadcasterId(broadcaster_id);
    } else if (channel) {
      channelRecord = await getChannelByLogin(channel.toLowerCase());
    }
    
    if (!channelRecord) {
      return reply.status(404).send({ error: 'Channel not found or not connected' });
    }
    
    try {
      // Anti-spam checks
      const cooldownResult = await checkCooldowns(
        channelRecord.id,
        requested_by || null,
        null,
        {
          channelCooldownSeconds: config.cooldown.channelCooldownSeconds,
          userCooldownSeconds: config.cooldown.userCooldownSeconds,
          blockOnActiveJob: config.cooldown.blockOnActiveJob,
          blockDuplicateClips: false,
        }
      );
      
      if (!cooldownResult.allowed) {
        return reply.status(429).send({
          error: 'Rate limited',
          reason: cooldownResult.reason,
          wait_seconds: cooldownResult.waitSeconds,
        });
      }
      
      const job = await createJob({
        channel_id: channelRecord.id,
        requested_by: requested_by || null,
        source: 'api',
        status: 'queued',
      });
      
      await enqueueJob(job.id);
      
      return reply.send({
        success: true,
        job_id: job.id,
        message: 'Clip job queued successfully',
      });
    } catch (err) {
      fastify.log.error({ err }, 'Failed to queue clip job');
      return reply.status(500).send({ error: 'Failed to queue clip job' });
    }
  });

  // ============================================================================
  // Process an EXISTING Twitch clip (no need to be LIVE!)
  // ============================================================================
  // This endpoint processes an already-existing clip through the pipeline:
  // - Skips clip creation (since clip already exists)
  // - Downloads, transcribes, renders vertical, uploads to Google Drive
  // 
  // Usage:
  //   POST /api/process-clip
  //   {
  //     "clip_url": "https://clips.twitch.tv/ClipSlug",
  //     // OR
  //     "clip_id": "ClipSlug",
  //     "force": true  // Optional: bypass duplicate check (for testing)
  //   }
  // ============================================================================
  fastify.post('/api/process-clip', async (
    request: FastifyRequest<{
      Body: {
        clip_url?: string;
        clip_id?: string;
        requested_by?: string;
        force?: boolean;
      };
    }>,
    reply: FastifyReply
  ) => {
    const { clip_url, clip_id, requested_by, force } = request.body || {};
    
    // Validate input
    if (!clip_url && !clip_id) {
      return reply.status(400).send({ 
        error: 'Missing required field: provide either clip_url or clip_id',
        example: {
          clip_url: 'https://clips.twitch.tv/YourClipSlug',
          // OR
          clip_id: 'YourClipSlug',
        }
      });
    }
    
    // Extract the clip ID from URL if needed
    const extractedClipId = extractClipId(clip_url || clip_id || '');
    
    if (!extractedClipId) {
      return reply.status(400).send({ error: 'Invalid clip URL or ID' });
    }
    
    fastify.log.info(`Processing existing clip: ${extractedClipId}`);
    
    try {
      // We need to fetch clip details to find the broadcaster
      // For this, we need ANY valid access token from a connected channel
      // We'll use the clip's broadcaster_id to find their channel
      
      // First, try to get clip info using app credentials
      // Since getClip requires an access token, we need to find a connected channel first
      // Then verify the clip belongs to them
      
      // For now, require the channel to be specified or we'll try to detect from clip
      // We can use Twitch's public API to get basic clip info
      
      // Fetch clip using Twitch API (need an access token from ANY connected channel)
      // TODO: Could optimize by using app access token instead
      
      const response = await fetch(
        `https://api.twitch.tv/helix/clips?id=${extractedClipId}`,
        {
          headers: {
            'Client-Id': config.twitch.clientId,
            // Use app access token if available, or we'll need to get one
            'Authorization': `Bearer ${await getAppAccessToken()}`,
          },
        }
      );
      
      if (!response.ok) {
        const errorText = await response.text();
        fastify.log.error(`Failed to fetch clip ${extractedClipId}: ${errorText}`);
        return reply.status(404).send({ error: 'Clip not found or Twitch API error' });
      }
      
      const data = await response.json() as { data: Array<{
        id: string;
        url: string;
        broadcaster_id: string;
        broadcaster_name: string;
        title: string;
        created_at: string;
        thumbnail_url: string;
        duration: number;
      }> };
      
      if (!data.data || data.data.length === 0) {
        return reply.status(404).send({ error: 'Clip not found' });
      }
      
      const clipInfo = data.data[0];
      fastify.log.info(`Found clip: "${clipInfo.title}" by ${clipInfo.broadcaster_name}`);
      
      // Find the channel in our database
      const channelRecord = await getChannelByBroadcasterId(clipInfo.broadcaster_id);
      
      if (!channelRecord) {
        return reply.status(400).send({ 
          error: `Channel "${clipInfo.broadcaster_name}" is not connected. The streamer needs to authenticate first at /auth/twitch/start`,
          broadcaster_id: clipInfo.broadcaster_id,
          broadcaster_name: clipInfo.broadcaster_name,
        });
      }
      
      // ====================================================================
      // ANTI-SPAM CHECKS (including duplicate clip check)
      // Skip duplicate check if force=true (for testing/reprocessing)
      // ====================================================================
      if (force) {
        fastify.log.info(`Force mode enabled - skipping duplicate check for clip ${clipInfo.id}`);
      }
      
      const cooldownResult = await checkCooldowns(
        channelRecord.id,
        requested_by || 'api-process-clip',
        clipInfo.id, // Check for duplicate clip ID
        {
          channelCooldownSeconds: 0, // No channel cooldown for process-clip
          userCooldownSeconds: 0,    // No user cooldown for process-clip
          blockOnActiveJob: config.cooldown.blockOnActiveJob,
          blockDuplicateClips: force ? false : config.cooldown.blockDuplicateClips, // Skip if force=true
        }
      );
      
      if (!cooldownResult.allowed) {
        // For duplicate clips, return the existing job info
        if (cooldownResult.existingJob && cooldownResult.reason?.includes('already been processed')) {
          return reply.status(409).send({
            error: cooldownResult.reason,
            existing_job_id: cooldownResult.existingJob.id,
            existing_job_status: cooldownResult.existingJob.status,
            clip_id: clipInfo.id,
          });
        }
        
        return reply.status(429).send({
          error: 'Rate limited',
          reason: cooldownResult.reason,
        });
      }
      
      // Create job with clip info already populated (skips creation stage)
      const job = await createJob({
        channel_id: channelRecord.id,
        requested_by: requested_by || 'api-process-clip',
        source: 'api',
        status: 'waiting_clip',  // Skip to waiting stage (clip already exists)
        twitch_clip_id: clipInfo.id,
      });
      
      // Queue the job
      await enqueueJob(job.id);
      
      fastify.log.info(`Queued existing clip for processing: ${job.id} (clip: ${clipInfo.id})`);
      
      return reply.send({
        success: true,
        job_id: job.id,
        clip_id: clipInfo.id,
        clip_url: clipInfo.url,
        clip_title: clipInfo.title,
        broadcaster: clipInfo.broadcaster_name,
        message: `Processing clip "${clipInfo.title}" - no need to be live!`,
      });
      
    } catch (err: any) {
      fastify.log.error({ err }, 'Failed to process existing clip');
      return reply.status(500).send({ 
        error: 'Failed to process clip', 
        details: err?.message || String(err),
      });
    }
  });
}

// ============================================================================
// App Access Token (Client Credentials Flow)
// ============================================================================
// Used for API calls that don't require user authorization
let cachedAppToken: { token: string; expiresAt: Date } | null = null;

async function getAppAccessToken(): Promise<string> {
  // Check if we have a valid cached token
  if (cachedAppToken && cachedAppToken.expiresAt > new Date()) {
    return cachedAppToken.token;
  }
  
  // Get new app access token using client credentials
  const response = await fetch('https://id.twitch.tv/oauth2/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      client_id: config.twitch.clientId,
      client_secret: config.twitch.clientSecret,
      grant_type: 'client_credentials',
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to get app access token: ${await response.text()}`);
  }
  
  const data = await response.json() as { 
    access_token: string; 
    expires_in: number;
    token_type: string;
  };
  
  // Cache the token (with 1 minute buffer)
  cachedAppToken = {
    token: data.access_token,
    expiresAt: new Date(Date.now() + (data.expires_in - 60) * 1000),
  };
  
  return data.access_token;
}

