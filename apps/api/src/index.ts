import Fastify from 'fastify';
import cors from '@fastify/cors';
import { config, validateConfig } from './config.js';
import { authRoutes } from './routes/auth.js';
import { clipRoutes } from './routes/clip.js';
import { jobRoutes } from './routes/jobs.js';
import { dashboardRoutes } from './routes/dashboard.js';
import { getRedis, closeRedis, getQueueLength } from './queue.js';

const fastify = Fastify({
  logger: {
    level: 'info',
  },
});

// Register CORS
await fastify.register(cors, {
  origin: true,
  credentials: true,
});

// Health check endpoint
fastify.get('/health', async () => {
  let redisConnected = false;
  let queueLength = 0;
  
  try {
    const redis = getRedis();
    await redis.ping();
    redisConnected = true;
    queueLength = await getQueueLength();
  } catch {
    // Redis not connected
  }
  
  return {
    status: 'ok',
    timestamp: new Date().toISOString(),
    redis: redisConnected ? 'connected' : 'disconnected',
    queue_length: queueLength,
  };
});

// Register routes
await fastify.register(authRoutes);
await fastify.register(clipRoutes);
await fastify.register(jobRoutes);
await fastify.register(dashboardRoutes);

// Graceful shutdown
const shutdown = async () => {
  console.log('\n🛑 Shutting down...');
  await closeRedis();
  await fastify.close();
  process.exit(0);
};

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// Start server
const start = async () => {
  try {
    validateConfig();
    
    await fastify.listen({ port: config.port, host: '0.0.0.0' });
    
    console.log(`
╔═══════════════════════════════════════════════════════════╗
║                   Stream2Short API                        ║
╠═══════════════════════════════════════════════════════════╣
║  Server running on port ${config.port}                           ║
║                                                           ║
║  Public Endpoints:                                        ║
║  • GET  /health              - Health check               ║
║  • GET  /auth/twitch/start   - Start OAuth flow           ║
║  • GET  /auth/twitch/callback - OAuth callback            ║
║  • GET  /se/clip             - StreamElements trigger     ║
║  • POST /api/clip            - Create NEW clip (LIVE)     ║
║  • POST /api/process-clip    - Process EXISTING clip      ║
║  • GET  /jobs                - List jobs for channel      ║
║  • GET  /jobs/:id            - Get job details            ║
║                                                           ║
║  Dashboard Endpoints (require API key):                   ║
║  • GET  /api/channels        - List all channels          ║
║  • GET  /api/jobs            - Jobs with filters          ║
║  • GET  /api/jobs/:id        - Job details + events       ║
║  • POST /api/jobs/:id/review - Approve/reject job         ║
║  • POST /api/jobs/:id/retry  - Retry failed job           ║
║  • POST /api/jobs/:id/rerender - Re-render with preset    ║
╚═══════════════════════════════════════════════════════════╝
`);
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();

