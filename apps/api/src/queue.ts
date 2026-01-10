import Redis from 'ioredis';
import { config } from './config.js';

let redis: Redis | null = null;

export function getRedis(): Redis {
  if (!redis) {
    redis = new Redis(config.redisUrl, {
      maxRetriesPerRequest: 3,
      retryStrategy(times) {
        const delay = Math.min(times * 50, 2000);
        return delay;
      },
    });
    
    redis.on('error', (err) => {
      console.error('Redis connection error:', err);
    });
    
    redis.on('connect', () => {
      console.log('✅ Connected to Redis');
    });
  }
  
  return redis;
}

// Push a job ID to the queue
export async function enqueueJob(jobId: string): Promise<void> {
  const redis = getRedis();
  await redis.lpush(config.queueName, jobId);
  console.log(`📤 Enqueued job ${jobId}`);
}

// Get queue length (for monitoring)
export async function getQueueLength(): Promise<number> {
  const redis = getRedis();
  return redis.llen(config.queueName);
}

// Graceful shutdown
export async function closeRedis(): Promise<void> {
  if (redis) {
    await redis.quit();
    redis = null;
  }
}

