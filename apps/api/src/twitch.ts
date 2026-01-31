import { config } from './config.js';
import { getTokens, upsertTokens, type OAuthToken } from './db.js';

const TWITCH_AUTH_URL = 'https://id.twitch.tv/oauth2';
const TWITCH_API_URL = 'https://api.twitch.tv/helix';

interface TwitchTokenResponse {
  access_token: string;
  refresh_token: string;
  expires_in: number;
  scope: string[];
  token_type: string;
}

interface TwitchUser {
  id: string;
  login: string;
  display_name: string;
  email?: string;
  profile_image_url?: string;
}

// Generate OAuth authorization URL
export function getAuthUrl(state: string): string {
  const params = new URLSearchParams({
    client_id: config.twitch.clientId,
    redirect_uri: config.twitch.redirectUri || `${config.baseUrl}/auth/twitch/callback`,
    response_type: 'code',
    scope: config.twitch.scopes.join(' '),
    state,
  });
  
  return `${TWITCH_AUTH_URL}/authorize?${params.toString()}`;
}

// Exchange authorization code for tokens
export async function exchangeCodeForTokens(code: string): Promise<TwitchTokenResponse> {
  const response = await fetch(`${TWITCH_AUTH_URL}/token`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      client_id: config.twitch.clientId,
      client_secret: config.twitch.clientSecret,
      code,
      grant_type: 'authorization_code',
      redirect_uri: config.twitch.redirectUri || `${config.baseUrl}/auth/twitch/callback`,
    }),
  });
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to exchange code: ${error}`);
  }
  
  return response.json() as Promise<TwitchTokenResponse>;
}

// Refresh access token
export async function refreshAccessToken(refreshToken: string): Promise<TwitchTokenResponse> {
  const response = await fetch(`${TWITCH_AUTH_URL}/token`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      client_id: config.twitch.clientId,
      client_secret: config.twitch.clientSecret,
      refresh_token: refreshToken,
      grant_type: 'refresh_token',
    }),
  });
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to refresh token: ${error}`);
  }
  
  return response.json() as Promise<TwitchTokenResponse>;
}

// Get user info from access token
export async function getTwitchUser(accessToken: string): Promise<TwitchUser> {
  const response = await fetch(`${TWITCH_API_URL}/users`, {
    headers: {
      'Authorization': `Bearer ${accessToken}`,
      'Client-Id': config.twitch.clientId,
    },
  });
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to get user: ${error}`);
  }
  
  const data = await response.json() as { data: TwitchUser[] };
  if (!data.data || data.data.length === 0) {
    throw new Error('No user data returned');
  }
  
  return data.data[0];
}

// Get a valid access token for a channel (refresh if needed)
export async function getValidAccessToken(channelId: string): Promise<string> {
  const tokens = await getTokens(channelId);
  
  if (!tokens) {
    throw new Error('No tokens found for channel');
  }
  
  const expiresAt = new Date(tokens.expires_at);
  const now = new Date();
  const twoMinutesFromNow = new Date(now.getTime() + 2 * 60 * 1000);
  
  // If token expires in more than 2 minutes, it's still valid
  if (expiresAt > twoMinutesFromNow) {
    return tokens.access_token;
  }
  
  // Token is expired or expiring soon, refresh it
  console.log(`Refreshing token for channel ${channelId}`);
  
  try {
    const newTokens = await refreshAccessToken(tokens.refresh_token);
    
    const updatedTokens: OAuthToken = {
      channel_id: channelId,
      access_token: newTokens.access_token,
      refresh_token: newTokens.refresh_token,
      scopes: newTokens.scope,
      expires_at: new Date(Date.now() + newTokens.expires_in * 1000).toISOString(),
      updated_at: new Date().toISOString(),
    };
    
    await upsertTokens(updatedTokens);
    
    return newTokens.access_token;
  } catch (error) {
    console.error('Failed to refresh token:', error);
    throw new Error('Token refresh failed. Channel may need to re-authenticate.');
  }
}

// Create a clip for a broadcaster
export async function createClip(broadcasterId: string, accessToken: string): Promise<{ id: string; edit_url: string }> {
  const response = await fetch(`${TWITCH_API_URL}/clips?broadcaster_id=${broadcasterId}`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${accessToken}`,
      'Client-Id': config.twitch.clientId,
    },
  });
  
  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to create clip: ${error}`);
  }
  
  const data = await response.json() as { data: Array<{ id: string; edit_url: string }> };
  if (!data.data || data.data.length === 0) {
    throw new Error('No clip data returned');
  }
  
  return data.data[0];
}

// Get clip details
export async function getClip(clipId: string, accessToken: string): Promise<{
  id: string;
  url: string;
  embed_url: string;
  broadcaster_id: string;
  creator_id: string;
  video_id: string;
  game_id: string;
  language: string;
  title: string;
  view_count: number;
  created_at: string;
  thumbnail_url: string;
  duration: number;
} | null> {
  const response = await fetch(`${TWITCH_API_URL}/clips?id=${clipId}`, {
    headers: {
      'Authorization': `Bearer ${accessToken}`,
      'Client-Id': config.twitch.clientId,
    },
  });
  
  if (!response.ok) {
    return null;
  }
  
  const data = await response.json() as { data: Array<{
    id: string;
    url: string;
    embed_url: string;
    broadcaster_id: string;
    creator_id: string;
    video_id: string;
    game_id: string;
    language: string;
    title: string;
    view_count: number;
    created_at: string;
    thumbnail_url: string;
    duration: number;
  }> };
  
  if (!data.data || data.data.length === 0) {
    return null;
  }
  
  return data.data[0];
}

// Get clip download URL from thumbnail URL
// Twitch doesn't provide a direct download URL, but we can derive it from the thumbnail
export function getClipDownloadUrl(thumbnailUrl: string): string {
  // Thumbnail URL format: https://clips-media-assets2.twitch.tv/AT-cm%7C{clip_id}-preview-480x272.jpg
  // Video URL format: https://clips-media-assets2.twitch.tv/AT-cm%7C{clip_id}.mp4
  // Or: https://production.assets.clips.twitchcdn.net/{slug}-offset-{offset}.mp4
  
  // Remove the preview suffix and change extension
  const downloadUrl = thumbnailUrl
    .replace(/-preview-\d+x\d+\.jpg$/, '.mp4')
    .replace(/-preview\.jpg$/, '.mp4');
  
  return downloadUrl;
}

// ============================================================================
// App Access Token (for API calls that don't require user auth)
// ============================================================================
let cachedAppToken: { token: string; expiresAt: Date } | null = null;

export async function getAppAccessToken(): Promise<string> {
  // Check if we have a valid cached token
  if (cachedAppToken && cachedAppToken.expiresAt > new Date()) {
    return cachedAppToken.token;
  }
  
  // Get new app access token using client credentials
  const response = await fetch(`${TWITCH_AUTH_URL}/token`, {
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

// Get game info by ID (includes box art)
export async function getGameInfo(gameId: string): Promise<{
  id: string;
  name: string;
  box_art_url: string;
} | null> {
  const accessToken = await getAppAccessToken();
  
  const response = await fetch(`${TWITCH_API_URL}/games?id=${gameId}`, {
    headers: {
      'Authorization': `Bearer ${accessToken}`,
      'Client-Id': config.twitch.clientId,
    },
  });
  
  if (!response.ok) {
    return null;
  }
  
  const data = await response.json() as { data: Array<{
    id: string;
    name: string;
    box_art_url: string;
  }> };
  
  if (!data.data || data.data.length === 0) {
    return null;
  }
  
  return data.data[0];
}

// Get multiple games info by IDs
export async function getGamesInfo(gameIds: string[]): Promise<Map<string, {
  id: string;
  name: string;
  box_art_url: string;
}>> {
  if (gameIds.length === 0) {
    return new Map();
  }
  
  const accessToken = await getAppAccessToken();
  
  // Twitch allows up to 100 IDs per request
  const params = new URLSearchParams();
  gameIds.slice(0, 100).forEach(id => params.append('id', id));
  
  const response = await fetch(`${TWITCH_API_URL}/games?${params.toString()}`, {
    headers: {
      'Authorization': `Bearer ${accessToken}`,
      'Client-Id': config.twitch.clientId,
    },
  });
  
  if (!response.ok) {
    return new Map();
  }
  
  const data = await response.json() as { data: Array<{
    id: string;
    name: string;
    box_art_url: string;
  }> };
  
  const result = new Map<string, { id: string; name: string; box_art_url: string }>();
  for (const game of data.data || []) {
    result.set(game.id, game);
  }
  
  return result;
}

