"""Twitch API client for Stream2Short Worker."""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional
import httpx
from config import config
from db import get_tokens, update_tokens

TWITCH_AUTH_URL = "https://id.twitch.tv/oauth2"
TWITCH_API_URL = "https://api.twitch.tv/helix"


class TwitchAPIError(Exception):
    """Twitch API error."""
    pass


def get_valid_access_token(channel_id: str) -> str:
    """
    Get a valid access token for a channel, refreshing if necessary.
    
    Args:
        channel_id: The channel's database ID
        
    Returns:
        Valid access token
        
    Raises:
        TwitchAPIError: If tokens not found or refresh fails
    """
    tokens = get_tokens(channel_id)
    
    if not tokens:
        raise TwitchAPIError(f"No tokens found for channel {channel_id}")
    
    expires_at = datetime.fromisoformat(tokens["expires_at"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    
    # If token expires in more than 2 minutes, it's still valid
    if expires_at > now + timedelta(minutes=2):
        return tokens["access_token"]
    
    # Token is expired or expiring soon, refresh it
    print(f"🔄 Refreshing token for channel {channel_id}")
    
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{TWITCH_AUTH_URL}/token",
                data={
                    "client_id": config.TWITCH_CLIENT_ID,
                    "client_secret": config.TWITCH_CLIENT_SECRET,
                    "refresh_token": tokens["refresh_token"],
                    "grant_type": "refresh_token",
                },
            )
            response.raise_for_status()
            data = response.json()
        
        new_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=data["expires_in"])).isoformat()
        
        update_tokens(
            channel_id=channel_id,
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=new_expires_at,
            scopes=data.get("scope", []),
        )
        
        return data["access_token"]
        
    except httpx.HTTPError as e:
        raise TwitchAPIError(f"Failed to refresh token: {e}")


def create_clip(broadcaster_id: str, access_token: str) -> dict:
    """
    Create a clip for a broadcaster.
    
    Args:
        broadcaster_id: Twitch broadcaster ID
        access_token: Valid access token
        
    Returns:
        Dict with 'id' and 'edit_url'
        
    Raises:
        TwitchAPIError: If clip creation fails
    """
    with httpx.Client() as client:
        response = client.post(
            f"{TWITCH_API_URL}/clips",
            params={"broadcaster_id": broadcaster_id},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Client-Id": config.TWITCH_CLIENT_ID,
            },
        )
        
        if not response.is_success:
            raise TwitchAPIError(f"Failed to create clip: {response.status_code} - {response.text}")
        
        data = response.json()
        
        if not data.get("data"):
            raise TwitchAPIError("No clip data returned from Twitch")
        
        return data["data"][0]


def get_clip(clip_id: str, access_token: str) -> Optional[dict]:
    """
    Get clip details.
    
    Args:
        clip_id: Twitch clip ID
        access_token: Valid access token
        
    Returns:
        Clip data dict or None if not found
    """
    with httpx.Client() as client:
        response = client.get(
            f"{TWITCH_API_URL}/clips",
            params={"id": clip_id},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Client-Id": config.TWITCH_CLIENT_ID,
            },
        )
        
        if not response.is_success:
            return None
        
        data = response.json()
        
        if not data.get("data"):
            return None
        
        return data["data"][0]


def wait_for_clip(clip_id: str, access_token: str) -> dict:
    """
    Poll until clip is available.
    
    Args:
        clip_id: Twitch clip ID
        access_token: Valid access token
        
    Returns:
        Clip data dict
        
    Raises:
        TwitchAPIError: If clip not available after max attempts
    """
    for attempt in range(config.CLIP_POLL_MAX_ATTEMPTS):
        clip = get_clip(clip_id, access_token)
        
        if clip:
            print(f"✅ Clip {clip_id} is now available")
            return clip
        
        print(f"⏳ Waiting for clip {clip_id} (attempt {attempt + 1}/{config.CLIP_POLL_MAX_ATTEMPTS})")
        time.sleep(config.CLIP_POLL_INTERVAL)
    
    raise TwitchAPIError(f"Clip {clip_id} not available after {config.CLIP_POLL_MAX_ATTEMPTS} attempts")


def get_clip_download_url(thumbnail_url: str) -> str:
    """
    Derive clip download URL from thumbnail URL.
    
    Twitch doesn't provide a direct download URL, but we can derive it from the thumbnail.
    Thumbnail: https://clips-media-assets2.twitch.tv/AT-cm%7C{id}-preview-480x272.jpg
    Video: https://clips-media-assets2.twitch.tv/AT-cm%7C{id}.mp4
    
    Args:
        thumbnail_url: Clip thumbnail URL
        
    Returns:
        MP4 download URL
    """
    import re
    
    # Remove preview suffix and change to .mp4
    download_url = re.sub(r"-preview-\d+x\d+\.jpg$", ".mp4", thumbnail_url)
    download_url = re.sub(r"-preview\.jpg$", ".mp4", download_url)
    
    return download_url


def download_clip(url: str, output_path: str) -> None:
    """
    Download a clip to a local file.
    
    Args:
        url: Clip download URL
        output_path: Local file path to save to
        
    Raises:
        TwitchAPIError: If download fails
    """
    print(f"📥 Downloading clip from {url}")
    
    with httpx.Client(timeout=60.0) as client:
        response = client.get(url, follow_redirects=True)
        
        if not response.is_success:
            raise TwitchAPIError(f"Failed to download clip: {response.status_code}")
        
        with open(output_path, "wb") as f:
            f.write(response.content)
    
    print(f"✅ Downloaded clip to {output_path}")

