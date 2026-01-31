import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { config } from '../config.js';
import {
  getAllChannels,
  getJob,
  getJobsWithFilters,
  getJobEvents,
  reviewJob,
  resetJobForRetry,
  updateJobPreset,
  updateTranscriptSegments,
  getGamesForChannel,
  JobFilters,
  JobStatus,
  ReviewStatus,
  LastStage,
  RenderPreset,
  TranscriptSegment,
} from '../db.js';
import { enqueueJob } from '../queue.js';

// Middleware to verify dashboard API key
async function verifyDashboardApiKey(
  request: FastifyRequest,
  reply: FastifyReply
): Promise<void> {
  const apiKey = request.headers['x-dashboard-api-key'] as string;
  
  if (!config.dashboardApiKey) {
    // If no API key configured, allow access (for development)
    request.log.warn('DASHBOARD_API_KEY not configured - allowing unauthenticated access');
    return;
  }
  
  if (!apiKey || apiKey !== config.dashboardApiKey) {
    reply.status(401).send({ error: 'Invalid or missing API key' });
    return;
  }
}

export async function dashboardRoutes(fastify: FastifyInstance): Promise<void> {
  // Apply API key verification to all routes in this plugin
  fastify.addHook('preHandler', verifyDashboardApiKey);
  
  // ============================================================================
  // GET /api/channels - List all connected channels
  // ============================================================================
  fastify.get('/api/channels', async (request, reply) => {
    const channels = await getAllChannels();
    
    return reply.send({
      channels: channels.map(ch => ({
        id: ch.id,
        twitch_broadcaster_id: ch.twitch_broadcaster_id,
        twitch_login: ch.twitch_login,
        display_name: ch.display_name,
        created_at: ch.created_at,
        settings: ch.settings,
      })),
    });
  });
  
  // ============================================================================
  // GET /api/jobs - List jobs with filters and pagination
  // ============================================================================
  fastify.get('/api/jobs', async (
    request: FastifyRequest<{
      Querystring: {
        channel?: string;
        channel_id?: string;
        status?: string;
        review_status?: string;
        game_id?: string;
        game_name?: string;
        date_from?: string;
        date_to?: string;
        limit?: string;
        cursor?: string;
      };
    }>,
    reply
  ) => {
    const { channel, channel_id, status, review_status, game_id, game_name, date_from, date_to, limit, cursor } = request.query;
    
    const filters: JobFilters = {};
    
    // Channel filter (by ID or login)
    if (channel_id) {
      filters.channelId = channel_id;
    } else if (channel) {
      // If channel login provided, we need to look up the ID
      // For simplicity, the frontend should use channel_id
      filters.channelId = channel;
    }
    
    // Status filter (comma-separated for multiple)
    if (status) {
      const statuses = status.split(',') as JobStatus[];
      filters.status = statuses.length === 1 ? statuses[0] : statuses;
    }
    
    // Review status filter
    if (review_status) {
      const reviewStatuses = review_status.split(',') as ReviewStatus[];
      filters.reviewStatus = reviewStatuses.length === 1 ? reviewStatuses[0] : reviewStatuses;
    }
    
    // Game filter
    if (game_id) filters.gameId = game_id;
    if (game_name) filters.gameName = game_name;
    
    // Date range
    if (date_from) filters.dateFrom = date_from;
    if (date_to) filters.dateTo = date_to;
    
    // Pagination
    if (limit) filters.limit = parseInt(limit, 10);
    if (cursor) filters.cursor = cursor;
    
    const { jobs, nextCursor } = await getJobsWithFilters(filters);
    
    return reply.send({
      jobs,
      pagination: {
        next_cursor: nextCursor,
        has_more: !!nextCursor,
      },
    });
  });
  
  // ============================================================================
  // GET /api/games - Get unique games for a channel (for filter UI)
  // ============================================================================
  fastify.get('/api/games', async (
    request: FastifyRequest<{
      Querystring: {
        channel_id: string;
      };
    }>,
    reply
  ) => {
    const { channel_id } = request.query;
    
    if (!channel_id) {
      return reply.status(400).send({ error: 'channel_id is required' });
    }
    
    const games = await getGamesForChannel(channel_id);
    
    return reply.send({ games });
  });
  
  // ============================================================================
  // GET /api/jobs/:id - Get job details with events
  // ============================================================================
  fastify.get('/api/jobs/:id', async (
    request: FastifyRequest<{
      Params: { id: string };
      Querystring: { include_events?: string };
    }>,
    reply
  ) => {
    const { id } = request.params;
    const includeEvents = request.query.include_events === 'true';
    
    const job = await getJob(id);
    
    if (!job) {
      return reply.status(404).send({ error: 'Job not found' });
    }
    
    const response: { job: typeof job; events?: Awaited<ReturnType<typeof getJobEvents>> } = { job };
    
    if (includeEvents) {
      response.events = await getJobEvents(id);
    }
    
    return reply.send(response);
  });
  
  // ============================================================================
  // GET /api/jobs/:id/events - Get job events
  // ============================================================================
  fastify.get('/api/jobs/:id/events', async (
    request: FastifyRequest<{
      Params: { id: string };
    }>,
    reply
  ) => {
    const { id } = request.params;
    
    const job = await getJob(id);
    if (!job) {
      return reply.status(404).send({ error: 'Job not found' });
    }
    
    const events = await getJobEvents(id);
    
    return reply.send({ events });
  });
  
  // ============================================================================
  // POST /api/jobs/:id/review - Review a job (approve/reject)
  // ============================================================================
  fastify.post('/api/jobs/:id/review', async (
    request: FastifyRequest<{
      Params: { id: string };
      Body: {
        decision: ReviewStatus;
        notes?: string;
      };
    }>,
    reply
  ) => {
    const { id } = request.params;
    const { decision, notes } = request.body;
    
    if (!decision || !['approved', 'rejected'].includes(decision)) {
      return reply.status(400).send({ error: 'Invalid decision. Must be "approved" or "rejected"' });
    }
    
    const job = await getJob(id);
    if (!job) {
      return reply.status(404).send({ error: 'Job not found' });
    }
    
    await reviewJob(id, decision, notes);
    
    fastify.log.info(`Job ${id} reviewed: ${decision}`);
    
    return reply.send({
      success: true,
      message: `Job ${decision}`,
    });
  });
  
  // ============================================================================
  // POST /api/jobs/:id/retry - Retry a failed job
  // ============================================================================
  fastify.post('/api/jobs/:id/retry', async (
    request: FastifyRequest<{
      Params: { id: string };
      Body: {
        from_stage?: LastStage;
      };
    }>,
    reply
  ) => {
    const { id } = request.params;
    const { from_stage } = request.body || {};
    
    const job = await getJob(id);
    if (!job) {
      return reply.status(404).send({ error: 'Job not found' });
    }
    
    // Only allow retry of failed jobs
    if (job.status !== 'failed') {
      return reply.status(400).send({ 
        error: 'Can only retry failed jobs',
        current_status: job.status,
      });
    }
    
    // Reset job for retry
    await resetJobForRetry(id, from_stage);
    
    // Re-queue the job
    await enqueueJob(id);
    
    fastify.log.info(`Job ${id} queued for retry from stage: ${from_stage || 'beginning'}`);
    
    return reply.send({
      success: true,
      message: `Job queued for retry`,
      from_stage: from_stage || 'download',
    });
  });
  
  // ============================================================================
  // POST /api/jobs/:id/rerender - Re-render with different preset
  // ============================================================================
  fastify.post('/api/jobs/:id/rerender', async (
    request: FastifyRequest<{
      Params: { id: string };
      Body: {
        preset: RenderPreset;
      };
    }>,
    reply
  ) => {
    const { id } = request.params;
    const { preset } = request.body;
    
    const validPresets: RenderPreset[] = ['default', 'clean', 'boxed', 'minimal', 'bold'];
    
    if (!preset || !validPresets.includes(preset)) {
      return reply.status(400).send({ 
        error: 'Invalid preset',
        valid_presets: validPresets,
      });
    }
    
    const job = await getJob(id);
    if (!job) {
      return reply.status(404).send({ error: 'Job not found' });
    }
    
    // Only allow re-render of completed jobs
    if (job.status !== 'ready') {
      return reply.status(400).send({ 
        error: 'Can only re-render completed jobs',
        current_status: job.status,
      });
    }
    
    // Check if raw video still exists (needed for re-render)
    if (!job.raw_video_path) {
      return reply.status(400).send({ 
        error: 'Raw video no longer available for re-render',
      });
    }
    
    // Update preset and re-queue
    await updateJobPreset(id, preset);
    
    // Re-queue the job
    await enqueueJob(id);
    
    fastify.log.info(`Job ${id} queued for re-render with preset: ${preset}`);
    
    return reply.send({
      success: true,
      message: `Job queued for re-render`,
      preset,
    });
  });
  
  // ============================================================================
  // PATCH /api/jobs/:id/transcript - Update transcript segments (for editing)
  // ============================================================================
  fastify.patch('/api/jobs/:id/transcript', async (
    request: FastifyRequest<{
      Params: { id: string };
      Body: {
        segments: TranscriptSegment[];
      };
    }>,
    reply
  ) => {
    const { id } = request.params;
    const { segments } = request.body;
    
    // Validate segments
    if (!segments || !Array.isArray(segments)) {
      return reply.status(400).send({ error: 'segments must be an array' });
    }
    
    // Validate each segment has required fields
    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      if (typeof seg.start !== 'number' || typeof seg.end !== 'number' || typeof seg.text !== 'string') {
        return reply.status(400).send({ 
          error: `Invalid segment at index ${i}: must have start (number), end (number), and text (string)`,
        });
      }
    }
    
    const job = await getJob(id);
    if (!job) {
      return reply.status(404).send({ error: 'Job not found' });
    }
    
    // Update the segments
    await updateTranscriptSegments(id, segments);
    
    // Reconstruct plain text for response
    const transcriptText = segments.map(s => s.text).join(' ');
    
    fastify.log.info(`Job ${id} transcript updated with ${segments.length} segments`);
    
    return reply.send({
      success: true,
      message: 'Transcript updated successfully',
      segment_count: segments.length,
      transcript_text: transcriptText,
    });
  });
}

