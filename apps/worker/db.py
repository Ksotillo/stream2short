"""Database operations for Stream2Short Worker."""

from typing import Optional, Any
from supabase import create_client, Client
from config import config

# Initialize Supabase client
supabase: Client = create_client(
    config.SUPABASE_URL,
    config.SUPABASE_SERVICE_ROLE_KEY
)


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    """Get a job by ID."""
    response = supabase.table("clip_jobs").select("*").eq("id", job_id).single().execute()
    return response.data


def get_channel(channel_id: str) -> Optional[dict[str, Any]]:
    """Get a channel by ID."""
    response = supabase.table("channels").select("*").eq("id", channel_id).single().execute()
    return response.data


def get_tokens(channel_id: str) -> Optional[dict[str, Any]]:
    """Get OAuth tokens for a channel."""
    response = supabase.table("oauth_tokens").select("*").eq("channel_id", channel_id).single().execute()
    return response.data


def update_tokens(channel_id: str, access_token: str, refresh_token: str, expires_at: str, scopes: list[str]) -> None:
    """Update OAuth tokens for a channel."""
    supabase.table("oauth_tokens").upsert({
        "channel_id": channel_id,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "scopes": scopes,
    }).execute()


def update_job(job_id: str, **updates: Any) -> None:
    """Update a job with the given fields."""
    supabase.table("clip_jobs").update(updates).eq("id", job_id).execute()


def update_job_status(job_id: str, status: str) -> None:
    """Update job status."""
    update_job(job_id, status=status)


def mark_job_failed(job_id: str, error: str) -> None:
    """Mark a job as failed with an error message."""
    update_job(job_id, status="failed", error=error)


def increment_attempt_count(job_id: str) -> int:
    """Increment and return the attempt count for a job."""
    job = get_job(job_id)
    if not job:
        return 0
    new_count = job.get("attempt_count", 0) + 1
    update_job(job_id, attempt_count=new_count)
    return new_count

