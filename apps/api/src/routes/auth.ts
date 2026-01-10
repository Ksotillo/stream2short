import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import crypto from 'crypto';
import { config } from '../config.js';
import { getAuthUrl, exchangeCodeForTokens, getTwitchUser } from '../twitch.js';
import { upsertChannel, upsertTokens, type OAuthToken } from '../db.js';

// Simple in-memory state store (use Redis in production for multi-instance)
const stateStore = new Map<string, { expires: number }>();

// Clean expired states periodically
setInterval(() => {
  const now = Date.now();
  for (const [state, data] of stateStore.entries()) {
    if (data.expires < now) {
      stateStore.delete(state);
    }
  }
}, 60000);

export async function authRoutes(fastify: FastifyInstance): Promise<void> {
  // Start OAuth flow
  fastify.get('/auth/twitch/start', async (request: FastifyRequest, reply: FastifyReply) => {
    // Generate a random state for CSRF protection
    const state = crypto.randomBytes(16).toString('hex');
    
    // Store state with 10 minute expiry
    stateStore.set(state, { expires: Date.now() + 10 * 60 * 1000 });
    
    const authUrl = getAuthUrl(state);
    
    return reply.redirect(authUrl);
  });
  
  // OAuth callback
  fastify.get('/auth/twitch/callback', async (
    request: FastifyRequest<{
      Querystring: {
        code?: string;
        state?: string;
        error?: string;
        error_description?: string;
      };
    }>,
    reply: FastifyReply
  ) => {
    const { code, state, error, error_description } = request.query;
    
    // Handle OAuth errors
    if (error) {
      fastify.log.error(`OAuth error: ${error} - ${error_description}`);
      return reply.type('text/html').send(`
        <!DOCTYPE html>
        <html>
        <head><title>Connection Failed</title></head>
        <body style="font-family: system-ui; padding: 40px; text-align: center;">
          <h1>❌ Connection Failed</h1>
          <p>${error_description || error}</p>
          <p>Please try again or contact support.</p>
        </body>
        </html>
      `);
    }
    
    // Validate state
    if (!state || !stateStore.has(state)) {
      return reply.type('text/html').send(`
        <!DOCTYPE html>
        <html>
        <head><title>Invalid Request</title></head>
        <body style="font-family: system-ui; padding: 40px; text-align: center;">
          <h1>❌ Invalid Request</h1>
          <p>The authentication request expired or is invalid. Please try again.</p>
        </body>
        </html>
      `);
    }
    
    // Remove used state
    stateStore.delete(state);
    
    if (!code) {
      return reply.type('text/html').send(`
        <!DOCTYPE html>
        <html>
        <head><title>Missing Code</title></head>
        <body style="font-family: system-ui; padding: 40px; text-align: center;">
          <h1>❌ Missing Authorization Code</h1>
          <p>No authorization code was provided. Please try again.</p>
        </body>
        </html>
      `);
    }
    
    try {
      // Exchange code for tokens
      const tokens = await exchangeCodeForTokens(code);
      
      // Get user info
      const user = await getTwitchUser(tokens.access_token);
      
      fastify.log.info(`User authenticated: ${user.login} (${user.id})`);
      
      // Upsert channel
      const channel = await upsertChannel({
        twitch_broadcaster_id: user.id,
        twitch_login: user.login.toLowerCase(),
        display_name: user.display_name,
      });
      
      // Upsert tokens
      const oauthTokens: OAuthToken = {
        channel_id: channel.id,
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
        scopes: tokens.scope,
        expires_at: new Date(Date.now() + tokens.expires_in * 1000).toISOString(),
        updated_at: new Date().toISOString(),
      };
      
      await upsertTokens(oauthTokens);
      
      fastify.log.info(`Channel ${channel.twitch_login} connected successfully`);
      
      return reply.type('text/html').send(`
        <!DOCTYPE html>
        <html>
        <head>
          <title>Connected!</title>
          <style>
            body {
              font-family: system-ui, -apple-system, sans-serif;
              background: linear-gradient(135deg, #9146FF 0%, #6441A5 100%);
              min-height: 100vh;
              display: flex;
              align-items: center;
              justify-content: center;
              margin: 0;
              color: white;
            }
            .card {
              background: white;
              color: #333;
              padding: 40px 60px;
              border-radius: 16px;
              box-shadow: 0 20px 60px rgba(0,0,0,0.3);
              text-align: center;
            }
            h1 { color: #9146FF; margin-bottom: 8px; }
            .channel { font-size: 1.2em; font-weight: bold; color: #6441A5; }
            .info { margin-top: 20px; color: #666; font-size: 0.9em; }
          </style>
        </head>
        <body>
          <div class="card">
            <h1>✅ Connected!</h1>
            <p class="channel">${user.display_name}</p>
            <p>Your Twitch channel is now connected to Stream2Short.</p>
            <p class="info">You can close this tab now.</p>
          </div>
        </body>
        </html>
      `);
    } catch (err) {
      fastify.log.error({ err }, 'OAuth callback error');
      
      return reply.type('text/html').send(`
        <!DOCTYPE html>
        <html>
        <head><title>Connection Failed</title></head>
        <body style="font-family: system-ui; padding: 40px; text-align: center;">
          <h1>❌ Connection Failed</h1>
          <p>An error occurred while connecting your account.</p>
          <p style="color: #666; font-size: 0.9em;">${err instanceof Error ? err.message : 'Unknown error'}</p>
        </body>
        </html>
      `);
    }
  });
}

