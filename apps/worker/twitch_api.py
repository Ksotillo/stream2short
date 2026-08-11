"""Twitch API client for Stream2Short Worker."""

import os
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


def get_clip_download_urls(thumbnail_url: str) -> list[str]:
    """
    Get possible download URLs from thumbnail URL.
    
    Twitch has multiple thumbnail/video URL formats that change over time.
    This function returns multiple possible URLs to try.
    
    Args:
        thumbnail_url: Clip thumbnail URL
        
    Returns:
        List of possible MP4 download URLs to try
    """
    import re
    
    urls = []
    
    # New Twitch format (static-cdn.jtvnw.net)
    # Thumbnail: https://static-cdn.jtvnw.net/twitch-clips-thumbnails-prod/{slug}/{uuid}/preview-480x272.jpg
    new_format_match = re.search(
        r"twitch-clips-thumbnails-prod/([^/]+)/([^/]+)/preview",
        thumbnail_url
    )
    
    if new_format_match:
        slug = new_format_match.group(1)
        uuid = new_format_match.group(2)
        
        # Try various CDN patterns
        urls.extend([
            f"https://production.assets.clips.twitchcdn.net/{uuid}-offset-0.mp4",
            f"https://production.assets.clips.twitchcdn.net/{uuid}.mp4",
            f"https://clips-media-assets2.twitch.tv/{uuid}.mp4",
            f"https://clips-media-assets2.twitch.tv/{uuid}-offset-0.mp4",
            f"https://production.assets.clips.twitchcdn.net/v2/media/{uuid}/vod/1080.mp4",
            f"https://production.assets.clips.twitchcdn.net/v2/media/{uuid}/vod/720.mp4",
            f"https://production.assets.clips.twitchcdn.net/v2/media/{uuid}/vod/480.mp4",
            f"https://production.assets.clips.twitchcdn.net/v2/media/{uuid}/vod/360.mp4",
        ])
    
    # Old format - try removing preview suffix
    old_format_url = re.sub(r"-preview-\d+x\d+\.jpg$", ".mp4", thumbnail_url)
    if old_format_url != thumbnail_url:
        urls.append(old_format_url)
    
    old_format_url2 = re.sub(r"-preview\.jpg$", ".mp4", thumbnail_url)
    if old_format_url2 != thumbnail_url:
        urls.append(old_format_url2)
    
    return urls


def get_clip_download_url(thumbnail_url: str) -> str:
    """
    Legacy function - returns first URL candidate.
    Use get_clip_download_urls for multiple options.
    """
    urls = get_clip_download_urls(thumbnail_url)
    return urls[0] if urls else thumbnail_url.replace(".jpg", ".mp4")


# Twitch's public web player client ID (stable for many years, used by the
# actual twitch.tv player — NOT our app's client ID)
TWITCH_GQL_URL = "https://gql.twitch.tv/gql"
TWITCH_PLAYER_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"

# Known persisted-query hashes for VideoAccessToken_Clip, tried in order.
# Each entry: (sha256Hash, variables_builder)
_CLIP_TOKEN_QUERY_HASHES = [
    # Newer hash observed in the Twitch player (2026)
    ("4f35f1ac933d76b1da008c806cd5546a7534dfaff83e033a422a81f24e5991b3",
     lambda slug: {"slug": slug, "platform": "web"}),
    # Classic hash, stable for many years
    ("36b89d2507fce29e5ca551df756d27c1cfe079e2609642b4390aa4c35796eb11",
     lambda slug: {"slug": slug}),
]


def _extract_clip_slug(clip_url: str) -> str:
    """
    Extract the clip slug from a Twitch clip URL.

    Supports:
    - https://www.twitch.tv/channel/clip/SlugHere
    - https://clips.twitch.tv/SlugHere
    """
    from urllib.parse import urlparse

    path = urlparse(clip_url).path.strip("/")
    parts = path.split("/")

    if "clip" in parts:
        # /channel/clip/SlugHere
        idx = parts.index("clip")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    # clips.twitch.tv/SlugHere — slug is the whole path
    return parts[-1]


def download_clip_direct_gql(clip_url: str, output_path: str) -> None:
    """
    Download a Twitch clip via Twitch's public GraphQL player API.

    Fallback for when yt-dlp's extractor breaks (Twitch rotates the GraphQL
    hashes yt-dlp uses for metadata; the player's VideoAccessToken_Clip query
    is far more stable since the real twitch.tv player depends on it).

    Args:
        clip_url: Twitch clip page URL
        output_path: Local file path to save to

    Raises:
        TwitchAPIError: If download fails
    """
    from urllib.parse import quote

    slug = _extract_clip_slug(clip_url)
    print(f"📥 Downloading clip via direct GQL API (slug: {slug})")

    clip_data = None
    last_error = None

    with httpx.Client(timeout=30) as client:
        for query_hash, build_variables in _CLIP_TOKEN_QUERY_HASHES:
            body = {
                "operationName": "VideoAccessToken_Clip",
                "variables": build_variables(slug),
                "extensions": {
                    "persistedQuery": {"version": 1, "sha256Hash": query_hash}
                },
            }

            response = client.post(
                TWITCH_GQL_URL,
                json=body,
                headers={"Client-ID": TWITCH_PLAYER_CLIENT_ID},
            )

            if not response.is_success:
                last_error = f"GQL HTTP {response.status_code}"
                continue

            data = response.json()

            if data.get("errors"):
                last_error = f"GQL error: {data['errors'][0].get('message', 'unknown')}"
                continue

            clip = (data.get("data") or {}).get("clip")
            if clip and clip.get("playbackAccessToken") and clip.get("videoQualities"):
                clip_data = clip
                break

            last_error = "GQL response missing clip/token data"

        if clip_data is None:
            raise TwitchAPIError(f"Direct GQL clip lookup failed: {last_error}")

        # Pick the highest quality source
        qualities = clip_data["videoQualities"]
        try:
            qualities = sorted(
                qualities,
                key=lambda q: int(q.get("quality", 0) or 0),
                reverse=True,
            )
        except (ValueError, TypeError):
            pass  # keep API order (usually highest first)

        source_url = qualities[0]["sourceURL"]
        token = clip_data["playbackAccessToken"]
        download_url = (
            f"{source_url}?sig={token['signature']}&token={quote(token['value'])}"
        )

        print(f"   🎞️ Quality: {qualities[0].get('quality', '?')}p")

        # Stream the MP4 to disk
        with client.stream("GET", download_url) as video_response:
            if not video_response.is_success:
                raise TwitchAPIError(
                    f"Clip video download failed: HTTP {video_response.status_code}"
                )
            with open(output_path, "wb") as f:
                for chunk in video_response.iter_bytes(chunk_size=65536):
                    f.write(chunk)

    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    if file_size < 10000:
        raise TwitchAPIError(f"Downloaded file too small ({file_size} bytes)")

    print(f"✅ Downloaded clip via direct GQL ({file_size} bytes) to {output_path}")


def download_clip_with_ytdlp(clip_url: str, output_path: str) -> None:
    """
    Download a Twitch clip using yt-dlp.
    
    Args:
        clip_url: Twitch clip page URL (e.g., https://www.twitch.tv/channel/clip/ClipSlug)
        output_path: Local file path to save to
        
    Raises:
        TwitchAPIError: If download fails
    """
    import subprocess
    import shutil
    
    print(f"📥 Downloading clip with yt-dlp: {clip_url}")
    
    # Check if yt-dlp is available
    if not shutil.which("yt-dlp"):
        raise TwitchAPIError("yt-dlp not found. Please install it.")
    
    try:
        # Run yt-dlp to download the clip at best quality
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "best",  # Best single format (Twitch clips are usually single file)
                "-o", output_path,
                "--no-playlist",
                "--no-warnings",
                clip_url,
            ],
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise TwitchAPIError(f"yt-dlp failed: {error_msg}")
        
        # Verify file exists and has content
        import os
        if not os.path.exists(output_path):
            raise TwitchAPIError(f"Download completed but file not found: {output_path}")
        
        file_size = os.path.getsize(output_path)
        if file_size < 10000:  # Less than 10KB is suspicious
            raise TwitchAPIError(f"Downloaded file too small ({file_size} bytes)")
        
        print(f"✅ Downloaded clip ({file_size} bytes) to {output_path}")
        
    except subprocess.TimeoutExpired:
        raise TwitchAPIError("yt-dlp download timed out after 120 seconds")
    except FileNotFoundError:
        raise TwitchAPIError("yt-dlp not found. Please install it.")


def get_game_info(game_id: str, access_token: str) -> Optional[dict]:
    """
    Get game/category info by ID.
    
    Args:
        game_id: Twitch game/category ID
        access_token: Valid access token
        
    Returns:
        Game data dict with 'id', 'name', 'box_art_url' or None if not found
    """
    if not game_id:
        return None
    
    with httpx.Client() as client:
        response = client.get(
            f"{TWITCH_API_URL}/games",
            params={"id": game_id},
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


def _download_clip_with_fallback(clip_url: str, output_path: str) -> None:
    """
    Download a clip: try yt-dlp first, fall back to direct GQL API.

    yt-dlp breaks whenever Twitch rotates their GraphQL operation hashes
    (recurring KeyError('data') issue). The direct GQL fallback uses the
    player's access-token query which is much more stable.
    """
    try:
        download_clip_with_ytdlp(clip_url, output_path)
        return
    except TwitchAPIError as e:
        print(f"⚠️ yt-dlp failed: {e}")
        print("🔄 Falling back to direct Twitch GQL download...")

    download_clip_direct_gql(clip_url, output_path)


def download_clip(clip_url_or_thumbnail: str, output_path: str, clip_page_url: str = None) -> None:
    """
    Download a clip to a local file.
    
    Uses yt-dlp with a direct Twitch GQL API fallback.
    
    Args:
        clip_url_or_thumbnail: Either the clip page URL or thumbnail URL
        output_path: Local file path to save to
        clip_page_url: Optional direct clip page URL
        
    Raises:
        TwitchAPIError: If download fails
    """
    # If we have a clip page URL, use it directly
    if clip_page_url:
        _download_clip_with_fallback(clip_page_url, output_path)
        return
    
    # If this looks like a Twitch clip page URL, use it
    if "twitch.tv" in clip_url_or_thumbnail and "clip" in clip_url_or_thumbnail:
        _download_clip_with_fallback(clip_url_or_thumbnail, output_path)
        return
    
    # Otherwise, we can't download from just a thumbnail URL anymore
    raise TwitchAPIError(
        "Cannot download clip: need a Twitch clip page URL. "
        f"Got: {clip_url_or_thumbnail}"
    )

