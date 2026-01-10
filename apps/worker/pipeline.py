"""Clip processing pipeline for Stream2Short Worker."""

import os
import shutil
from pathlib import Path
from typing import Any

from config import config
from db import get_job, get_channel, update_job, update_job_status, mark_job_failed
from twitch_api import (
    get_valid_access_token,
    create_clip,
    wait_for_clip,
    get_clip_download_url,
    download_clip,
    TwitchAPIError,
)
from transcribe import transcribe_video
from video import render_vertical_video, VideoProcessingError
from storage import upload_file


class PipelineError(Exception):
    """Pipeline processing error."""
    pass


def process_job(job_id: str) -> None:
    """
    Process a clip job through all pipeline stages.
    
    Stages:
    1. Create clip (Twitch API)
    2. Wait for clip to be available
    3. Download clip
    4. Transcribe audio (Whisper)
    5. Render vertical video with subtitles (FFmpeg)
    6. Upload to Google Drive
    
    Args:
        job_id: UUID of the job to process
    """
    print(f"\n{'='*60}")
    print(f"🎬 Processing job: {job_id}")
    print(f"{'='*60}\n")
    
    # Load job data
    job = get_job(job_id)
    if not job:
        print(f"❌ Job {job_id} not found")
        return
    
    # Load channel data
    channel = get_channel(job["channel_id"])
    if not channel:
        mark_job_failed(job_id, "Channel not found")
        return
    
    broadcaster_id = channel["twitch_broadcaster_id"]
    
    # Create temp directory for this job
    temp_dir = Path(config.TEMP_DIR) / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get valid access token
        access_token = get_valid_access_token(job["channel_id"])
        
        # Stage 1: Create clip
        update_job_status(job_id, "creating_clip")
        print("📹 Stage 1: Creating clip...")
        
        clip_data = create_clip(broadcaster_id, access_token)
        clip_id = clip_data["id"]
        
        update_job(job_id, twitch_clip_id=clip_id)
        print(f"✅ Clip created: {clip_id}")
        
        # Stage 2: Wait for clip to be available
        update_job_status(job_id, "waiting_clip")
        print("⏳ Stage 2: Waiting for clip...")
        
        clip_info = wait_for_clip(clip_id, access_token)
        
        update_job(job_id, twitch_clip_url=clip_info.get("url"))
        print(f"✅ Clip available: {clip_info.get('url')}")
        
        # Stage 3: Download clip
        update_job_status(job_id, "downloading")
        print("📥 Stage 3: Downloading clip...")
        
        thumbnail_url = clip_info.get("thumbnail_url", "")
        download_url = get_clip_download_url(thumbnail_url)
        
        raw_video_path = str(temp_dir / "raw.mp4")
        download_clip(download_url, raw_video_path)
        
        update_job(job_id, raw_video_path=raw_video_path)
        print(f"✅ Downloaded to: {raw_video_path}")
        
        # Stage 4: Transcribe
        update_job_status(job_id, "transcribing")
        print("🎙️ Stage 4: Transcribing audio...")
        
        srt_path = str(temp_dir / "captions.srt")
        transcribe_video(raw_video_path, srt_path)
        print(f"✅ Transcription saved to: {srt_path}")
        
        # Stage 5: Render vertical video with subtitles
        update_job_status(job_id, "rendering")
        print("🎬 Stage 5: Rendering vertical video...")
        
        final_video_path = str(temp_dir / "final.mp4")
        render_vertical_video(
            input_path=raw_video_path,
            output_path=final_video_path,
            subtitle_path=srt_path,
        )
        
        update_job(job_id, final_video_path=final_video_path)
        print(f"✅ Rendered to: {final_video_path}")
        
        # Stage 6: Upload to Google Drive
        update_job_status(job_id, "uploading")
        print("📤 Stage 6: Uploading to Google Drive...")
        
        # Use display_name or twitch_login for folder name
        streamer_name = channel.get("display_name") or channel.get("twitch_login") or broadcaster_id
        
        # Check if Google Drive is configured
        if config.GOOGLE_SERVICE_ACCOUNT_FILE or config.GOOGLE_SERVICE_ACCOUNT_JSON:
            upload_result = upload_file(
                local_path=final_video_path,
                streamer_name=streamer_name,
                job_id=job_id,
            )
            final_url = upload_result.get("webViewLink") or upload_result.get("webContentLink", "")
            drive_file_id = upload_result.get("id", "")
            drive_path = upload_result.get("path", "")
            
            print(f"📁 Saved to: {drive_path}")
        else:
            # No Google Drive configured, keep local path
            print("⚠️ Google Drive not configured, keeping local file")
            final_url = final_video_path
        
        # Mark job as ready
        update_job(
            job_id,
            status="ready",
            final_video_url=final_url,
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Job {job_id} completed successfully!")
        print(f"📍 Final video: {final_url}")
        print(f"{'='*60}\n")
        
    except TwitchAPIError as e:
        error_msg = f"Twitch API error: {e}"
        print(f"❌ {error_msg}")
        mark_job_failed(job_id, error_msg)
        
    except VideoProcessingError as e:
        error_msg = f"Video processing error: {e}"
        print(f"❌ {error_msg}")
        mark_job_failed(job_id, error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}: {e}"
        print(f"❌ {error_msg}")
        mark_job_failed(job_id, error_msg)
        
    finally:
        # Cleanup temp directory (optional - keep for debugging)
        if os.getenv("CLEANUP_TEMP", "false").lower() == "true":
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"🧹 Cleaned up temp directory: {temp_dir}")

