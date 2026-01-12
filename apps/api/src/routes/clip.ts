import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { config } from '../config.js';
import { getChannelByLogin, getChannelByBroadcasterId, createJob, updateJob } from '../db.js';
import { enqueueJob } from '../queue.js';
import { getValidAccessToken, createClip } from '../twitch.js';

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
      // Get valid access token for the channel
      const accessToken = await getValidAccessToken(channelRecord.id);
      
      // Create clip on Twitch immediately
      fastify.log.info(`Creating clip for channel ${channelRecord.twitch_login}...`);
      const clipData = await createClip(channelRecord.twitch_broadcaster_id, accessToken);
      
      // Build the clip URL (edit_url redirects to clip page)
      // Format: https://clips.twitch.tv/{channel}/clip/{clip_id}
      const clipUrl = `https://clips.twitch.tv/${channelRecord.twitch_login}/clip/${clipData.id}`;
      
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
}

