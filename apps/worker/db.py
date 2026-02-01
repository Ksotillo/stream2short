"""Database operations for Stream2Short Worker."""

import time
from typing import Optional, Any, Callable, TypeVar
from supabase import create_client, Client
from config import config

T = TypeVar('T')


def get_client() -> Client:
    """Get a fresh Supabase client.
    
    Creating a new client for each operation ensures we don't have stale connections
    during long-running jobs (rendering can take 10+ minutes).
    """
    return create_client(
        config.SUPABASE_URL,
        config.SUPABASE_SERVICE_ROLE_KEY
    )


def with_retry(operation: Callable[[], T], max_retries: int = 3, delay: float = 1.0) -> T:
    """Execute an operation with retry logic for connection errors.
    
    Args:
        operation: Function to execute
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds (doubles each retry)
    
    Returns:
        Result of the operation
        
    Raises:
        Last exception if all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return operation()
        except (BrokenPipeError, ConnectionError, OSError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                print(f"⚠️ Database connection error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"   Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Database operation failed after {max_retries} attempts: {e}")
        except Exception as e:
            # For other exceptions, don't retry
            raise e
    
    raise last_error


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    """Get a job by ID."""
    def _query():
        client = get_client()
        response = client.table("clip_jobs").select("*").eq("id", job_id).single().execute()
        return response.data
    
    return with_retry(_query)


def get_channel(channel_id: str) -> Optional[dict[str, Any]]:
    """Get a channel by ID."""
    def _query():
        client = get_client()
        response = client.table("channels").select("*").eq("id", channel_id).single().execute()
        return response.data
    
    return with_retry(_query)


def get_tokens(channel_id: str) -> Optional[dict[str, Any]]:
    """Get OAuth tokens for a channel."""
    def _query():
        client = get_client()
        response = client.table("oauth_tokens").select("*").eq("channel_id", channel_id).single().execute()
        return response.data
    
    return with_retry(_query)


def update_tokens(channel_id: str, access_token: str, refresh_token: str, expires_at: str, scopes: list[str]) -> None:
    """Update OAuth tokens for a channel."""
    def _query():
        client = get_client()
        client.table("oauth_tokens").upsert({
            "channel_id": channel_id,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "scopes": scopes,
        }).execute()
    
    with_retry(_query)


def update_job(job_id: str, **updates: Any) -> None:
    """Update a job with the given fields."""
    def _query():
        client = get_client()
        client.table("clip_jobs").update(updates).eq("id", job_id).execute()
    
    with_retry(_query)


def update_job_status(job_id: str, status: str) -> None:
    """Update job status."""
    update_job(job_id, status=status)


def mark_job_failed(job_id: str, error: str, last_stage: str = None) -> None:
    """Mark a job as failed with an error message and last completed stage."""
    updates = {"status": "failed", "error": error}
    if last_stage:
        updates["last_stage"] = last_stage
    update_job(job_id, **updates)


def update_last_stage(job_id: str, stage: str) -> None:
    """Update the last completed stage for a job."""
    update_job(job_id, last_stage=stage)


def log_job_event(
    job_id: str,
    level: str,
    message: str,
    stage: str = None,
    data: dict = None
) -> None:
    """Log an event for a job (for debugging/progress tracking)."""
    def _query():
        client = get_client()
        client.table("job_events").insert({
            "job_id": job_id,
            "level": level,
            "message": message,
            "stage": stage,
            "data": data or {},
        }).execute()
    
    try:
        with_retry(_query)
    except Exception as e:
        print(f"⚠️ Failed to log job event: {e}")


def increment_attempt_count(job_id: str) -> int:
    """Increment and return the attempt count for a job."""
    job = get_job(job_id)
    if not job:
        return 0
    new_count = job.get("attempt_count", 0) + 1
    update_job(job_id, attempt_count=new_count)
    return new_count
