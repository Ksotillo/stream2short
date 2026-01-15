import dotenv from 'dotenv';

dotenv.config({ path: '../../.env' });

export const config = {
  // Server
  port: parseInt(process.env.API_PORT || '3000', 10),
  baseUrl: process.env.BASE_URL || 'http://localhost:3000',
  
  // StreamElements
  seSharedSecret: process.env.SE_SHARED_SECRET || '',
  
  // Twitch OAuth
  twitch: {
    clientId: process.env.TWITCH_CLIENT_ID || '',
    clientSecret: process.env.TWITCH_CLIENT_SECRET || '',
    redirectUri: process.env.TWITCH_REDIRECT_URI || '',
    // Scopes needed for clip creation and management
    scopes: [
      'clips:edit',        // Create clips
      'user:read:email',   // Get user info
    ],
  },
  
  // Supabase
  supabase: {
    url: process.env.SUPABASE_URL || '',
    serviceRoleKey: process.env.SUPABASE_SERVICE_ROLE_KEY || '',
  },
  
  // Redis
  redisUrl: process.env.REDIS_URL || 'redis://localhost:6379/0',
  
  // Queue
  queueName: 'clip_jobs_queue',
  
  // ============================================================================
  // Anti-Spam / Cooldown Settings
  // ============================================================================
  cooldown: {
    // Per-channel cooldown: minimum seconds between clips for the SAME channel
    // Prevents multiple users from spamming !clip at once
    channelCooldownSeconds: parseInt(process.env.CHANNEL_COOLDOWN_SECONDS || '30', 10),
    
    // Per-user cooldown: minimum seconds between clips for the SAME user on a channel
    // Prevents a single user from spamming !clip
    userCooldownSeconds: parseInt(process.env.USER_COOLDOWN_SECONDS || '60', 10),
    
    // Block if there's already an active (queued/processing) job for the channel
    blockOnActiveJob: process.env.BLOCK_ON_ACTIVE_JOB !== 'false',
    
    // Block duplicate clip processing (same twitch_clip_id)
    blockDuplicateClips: process.env.BLOCK_DUPLICATE_CLIPS !== 'false',
  },
};

// Validate required config
export function validateConfig(): void {
  const required = [
    ['TWITCH_CLIENT_ID', config.twitch.clientId],
    ['TWITCH_CLIENT_SECRET', config.twitch.clientSecret],
    ['SUPABASE_URL', config.supabase.url],
    ['SUPABASE_SERVICE_ROLE_KEY', config.supabase.serviceRoleKey],
    ['SE_SHARED_SECRET', config.seSharedSecret],
  ];
  
  const missing = required.filter(([, value]) => !value).map(([name]) => name);
  
  if (missing.length > 0) {
    console.warn(`⚠️  Missing environment variables: ${missing.join(', ')}`);
    console.warn('Some features may not work correctly.');
  }
}

