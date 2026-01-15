"""Clip processing pipeline for Stream2Short Worker."""

import os
import shutil
from pathlib import Path
from typing import Any, Optional

from config import config
from db import get_job, get_channel, update_job, update_job_status, mark_job_failed
from twitch_api import (
    get_valid_access_token,
    create_clip,
    wait_for_clip,
    download_clip,
    TwitchAPIError,
)
from transcribe import transcribe_video, transcribe_video_with_segments
from video import render_vertical_video, render_video_auto, VideoProcessingError
from storage import upload_file, SharedDriveError
from diarization import (
    diarize_video,
    assign_speakers_to_segments,
    DiarizationResult,
)
from ass_writer import segments_to_ass_with_diarization


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
        
        # Check if clip was already created by the API
        clip_id = job.get("twitch_clip_id")
        
        if clip_id:
            # Clip already created by API, skip to Stage 2
            print(f"📹 Clip already created by API: {clip_id}")
        else:
            # Stage 1: Create clip (fallback for direct API calls)
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
        
        # Get the clip page URL (e.g., https://www.twitch.tv/channel/clip/ClipSlug)
        clip_page_url = clip_info.get("url", "")
        print(f"🔗 Clip URL: {clip_page_url}")
        
        raw_video_path = str(temp_dir / "raw.mp4")
        # Use yt-dlp to download the clip reliably
        download_clip(clip_page_url, raw_video_path)
        
        update_job(job_id, raw_video_path=raw_video_path)
        print(f"✅ Downloaded to: {raw_video_path}")
        
        # Stage 4: Transcribe + Diarization
        update_job_status(job_id, "transcribing")
        print("🎙️ Stage 4: Transcribing audio...")
        
        # Get channel settings for per-channel diarization toggle
        channel_settings = channel.get("settings", {}) or {}
        
        # Get transcript segments
        segments, transcript_text_raw = transcribe_video_with_segments(raw_video_path)
        print(f"✅ Transcription complete: {len(segments)} segments")
        
        # Stage 4b: Speaker diarization (if enabled)
        diarization_result: Optional[DiarizationResult] = None
        has_diarization = False
        
        try:
            diarization_result = diarize_video(
                video_path=raw_video_path,
                temp_dir=str(temp_dir),
                channel_settings=channel_settings,
            )
            
            if diarization_result:
                has_diarization = True
                print(f"🎤 Diarization complete: {len(diarization_result.speakers)} speakers detected")
                
                # Assign speakers to transcript segments
                segments = assign_speakers_to_segments(
                    segments=segments,
                    turns=diarization_result.turns,
                    primary_speaker=diarization_result.primary_speaker,
                )
                
                # NOTE: We no longer merge segments - keep short 2-3 word chunks for TikTok style
                # merge_adjacent_segments was causing subtitles to become long sentences again
                
                # Count speaker distribution
                primary_count = sum(1 for s in segments if s.get('is_primary', True))
                other_count = len(segments) - primary_count
                speakers_found = list(set(s.get('speaker', 'UNKNOWN') for s in segments))
                print(f"📊 Speaker distribution: {primary_count} primary, {other_count} other ({len(speakers_found)} speakers)")
        except Exception as e:
            print(f"⚠️ Diarization skipped: {e}")
        
        # Generate ASS subtitles with speaker coloring
        subtitle_path = str(temp_dir / "captions.ass")
        segments_to_ass_with_diarization(
            segments=segments,
            output_path=subtitle_path,
            has_diarization=has_diarization,
        )
        print(f"✅ Subtitles saved to: {subtitle_path}")
        
        # Stage 5: Render vertical videos (TWO versions: with and without subtitles)
        update_job_status(job_id, "rendering")
        print("🎬 Stage 5: Rendering vertical videos (2 versions)...")
        
        # Version 1: WITHOUT subtitles (raw vertical crop)
        video_no_subs_path = str(temp_dir / "final_no_subs.mp4")
        print("🎬 Rendering version WITHOUT subtitles...")
        render_video_auto(
            input_path=raw_video_path,
            output_path=video_no_subs_path,
            subtitle_path=None,  # No subtitles
            enable_webcam_detection=True,
        )
        print(f"✅ Rendered (no subs): {video_no_subs_path}")
        
        # Version 2: WITH subtitles
        video_with_subs_path = str(temp_dir / "final_with_subs.mp4")
        print("🎬 Rendering version WITH subtitles...")
        render_video_auto(
            input_path=raw_video_path,
            output_path=video_with_subs_path,
            subtitle_path=subtitle_path,
            enable_webcam_detection=True,
        )
        print(f"✅ Rendered (with subs): {video_with_subs_path}")
        
        update_job(job_id, final_video_path=video_with_subs_path)
        
        # Stage 6: Upload BOTH versions to Google Drive
        update_job_status(job_id, "uploading")
        print("📤 Stage 6: Uploading to Google Drive (2 versions)...")
        
        # Use display_name or twitch_login for folder name
        streamer_name = channel.get("display_name") or channel.get("twitch_login") or broadcaster_id
        
        # Use transcript text from transcription stage
        transcript_text = transcript_text_raw
        
        # Get clip timestamp from clip_info
        clip_timestamp = clip_info.get("created_at", "")
        
        # Check if Google Drive is configured
        if config.GOOGLE_SERVICE_ACCOUNT_FILE or config.GOOGLE_SERVICE_ACCOUNT_JSON:
            # Upload version WITHOUT subtitles
            print("📤 Uploading WITHOUT_SUBTITLES version...")
            upload_result_no_subs = upload_file(
                local_path=video_no_subs_path,
                streamer_name=streamer_name,
                job_id=job_id,
                transcript_text=transcript_text,
                clip_timestamp=clip_timestamp,
                version_suffix="WITHOUT_SUBTITLES",
            )
            print(f"📁 Saved: {upload_result_no_subs.get('path', '')}")
            
            # Upload version WITH subtitles
            print("📤 Uploading WITH_SUBTITLES version...")
            upload_result_with_subs = upload_file(
                local_path=video_with_subs_path,
                streamer_name=streamer_name,
                job_id=job_id,
                transcript_text=transcript_text,
                clip_timestamp=clip_timestamp,
                version_suffix="WITH_SUBTITLES",
            )
            print(f"📁 Saved: {upload_result_with_subs.get('path', '')}")
            
            # Use the WITH_SUBTITLES version as the primary URL
            final_url = upload_result_with_subs.get("webViewLink") or upload_result_with_subs.get("webContentLink", "")
            drive_path = upload_result_with_subs.get("path", "")
            
            print(f"📁 Both versions saved to: {'/'.join(drive_path.split('/')[:-1])}/")
        else:
            # No Google Drive configured, keep local path
            print("⚠️ Google Drive not configured, keeping local files")
            final_url = video_with_subs_path
        
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
    
    except SharedDriveError as e:
        error_msg = f"Google Drive error: {e}"
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

