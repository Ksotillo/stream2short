import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { getJob, getJobsByChannel, getChannelByLogin, getChannelByBroadcasterId } from '../db.js';

export async function jobRoutes(fastify: FastifyInstance): Promise<void> {
  // Get jobs for a channel
  fastify.get('/jobs', async (
    request: FastifyRequest<{
      Querystring: {
        channel?: string;
        broadcaster_id?: string;
        limit?: string;
      };
    }>,
    reply: FastifyReply
  ) => {
    const { channel, broadcaster_id, limit } = request.query;
    
    let channelRecord = null;
    
    if (broadcaster_id) {
      channelRecord = await getChannelByBroadcasterId(broadcaster_id);
    } else if (channel) {
      channelRecord = await getChannelByLogin(channel.toLowerCase());
    }
    
    if (!channelRecord) {
      return reply.status(404).send({ error: 'Channel not found' });
    }
    
    const jobs = await getJobsByChannel(
      channelRecord.id,
      limit ? parseInt(limit, 10) : 50
    );
    
    return reply.send({
      channel: {
        id: channelRecord.id,
        twitch_login: channelRecord.twitch_login,
        display_name: channelRecord.display_name,
      },
      jobs,
    });
  });
  
  // Get a single job by ID
  fastify.get('/jobs/:id', async (
    request: FastifyRequest<{
      Params: {
        id: string;
      };
    }>,
    reply: FastifyReply
  ) => {
    const { id } = request.params;
    
    const job = await getJob(id);
    
    if (!job) {
      return reply.status(404).send({ error: 'Job not found' });
    }
    
    return reply.send({ job });
  });
  
  // Get signed URL for a job's video (placeholder for S3 signed URLs)
  fastify.get('/jobs/:id/signed-url', async (
    request: FastifyRequest<{
      Params: {
        id: string;
      };
    }>,
    reply: FastifyReply
  ) => {
    const { id } = request.params;
    
    const job = await getJob(id);
    
    if (!job) {
      return reply.status(404).send({ error: 'Job not found' });
    }
    
    if (job.status !== 'ready' || !job.final_video_url) {
      return reply.status(400).send({ 
        error: 'Video not ready',
        status: job.status,
      });
    }
    
    // For MVP, we'll return the stored URL directly
    // In production, generate a signed URL here
    return reply.send({
      url: job.final_video_url,
      expires_in: 3600, // 1 hour (placeholder)
    });
  });
}

