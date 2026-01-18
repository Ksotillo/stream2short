"""Clip processing pipeline for Stream2Short Worker."""

import os
import shutil
from pathlib import Path
from typing import Any, Optional

from config import config
from db import (
    get_job, get_channel, update_job, update_job_status, mark_job_failed,
    update_last_stage, log_job_event
)
from twitch_api import (
    get_valid_access_token,
    create_clip,
    wait_for_clip,
    download_clip,
    TwitchAPIError,
)
from transcribe import transcribe_video
from groq_transcribe import transcribe_with_segments
from video import render_vertical_video, render_video_auto, VideoProcessingError
from storage import upload_file, SharedDriveError
from diarization import (
    diarize_video,
    assign_speakers_to_segments,
    DiarizationResult,
)
from ass_writer import segments_to_ass_with_diarization
from disk_utils import (
    job_temp_directory,
    DiskSpaceError,
    cleanup_old_temp_directories,
    print_disk_status,
)
from audio_preprocess import (
    preprocess_audio_for_pipeline,
    AudioPreprocessError,
)


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
    
    # Use context manager for temp directory (guarantees cleanup)
    try:
        with job_temp_directory(job_id) as (temp_dir, disk_stats):
            _process_job_stages(job_id, job, channel, broadcaster_id, temp_dir)
    except DiskSpaceError as e:
        error_msg = f"Disk space error: {e}"
        print(f"❌ {error_msg}")
        mark_job_failed(job_id, error_msg)
    except Exception as e:
        # Any other error - cleanup still happens via context manager
        error_msg = f"Unexpected error: {type(e).__name__}: {e}"
        print(f"❌ {error_msg}")
        mark_job_failed(job_id, error_msg)


def _should_skip_to_stage(job: dict, target_stage: str) -> bool:
    """
    Check if we should skip to a specific stage based on job state.
    Used for retry from a specific stage.
    """
    last_stage = job.get("last_stage")
    if not last_stage:
        return False
    
    # Stage order
    stage_order = ["download", "transcribe", "render", "upload"]
    
    try:
        last_idx = stage_order.index(last_stage)
        target_idx = stage_order.index(target_stage)
        return target_idx <= last_idx
    except ValueError:
        return False


def _process_job_stages(
    job_id: str,
    job: dict,
    channel: dict,
    broadcaster_id: str,
    temp_dir: Path
) -> None:
    """
    Internal function that runs the actual processing stages.
    Separated from process_job to work with the cleanup context manager.
    
    Supports retry from a specific stage by checking job.last_stage.
    """
    current_stage = None
    
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
        current_stage = "download"
        raw_video_path = job.get("raw_video_path")
        
        # Check if we can skip download (retry from later stage)
        if raw_video_path and os.path.exists(raw_video_path):
            print(f"⏭️ Skipping download - raw video exists: {raw_video_path}")
            log_job_event(job_id, "info", "Skipping download - raw video exists", "download")
        else:
            update_job_status(job_id, "downloading")
            print("📥 Stage 3: Downloading clip...")
            log_job_event(job_id, "info", "Starting download", "download")
            
            # Get the clip page URL (e.g., https://www.twitch.tv/channel/clip/ClipSlug)
            clip_page_url = clip_info.get("url", "")
            print(f"🔗 Clip URL: {clip_page_url}")
            
            raw_video_path = str(temp_dir / "raw.mp4")
            # Use yt-dlp to download the clip reliably
            download_clip(clip_page_url, raw_video_path)
            
            update_job(job_id, raw_video_path=raw_video_path)
            update_last_stage(job_id, "download")
            log_job_event(job_id, "info", f"Downloaded to: {raw_video_path}", "download")
            print(f"✅ Downloaded to: {raw_video_path}")
        
        # Stage 3b: Audio preprocessing (normalization + extraction)
        # Single extraction used for both transcription and diarization
        print("🔊 Stage 3b: Preprocessing audio...")
        try:
            preprocessed_audio = preprocess_audio_for_pipeline(
                video_path=raw_video_path,
                temp_dir=str(temp_dir),
            )
            print(f"✅ Audio preprocessed: {preprocessed_audio}")
        except AudioPreprocessError as e:
            print(f"⚠️ Audio preprocessing failed: {e}")
            print("   Falling back to direct video processing...")
            preprocessed_audio = None
        
        # Stage 4: Transcribe + Diarization
        current_stage = "transcribe"
        update_job_status(job_id, "transcribing")
        print("🎙️ Stage 4: Transcribing audio...")
        log_job_event(job_id, "info", "Starting transcription", "transcribe")
        
        # Get channel settings for per-channel diarization toggle
        channel_settings = channel.get("settings", {}) or {}
        
        # Get transcript segments (using preprocessed audio if available)
        # Uses Groq API if configured, otherwise falls back to local Whisper
        segments, transcript_text_raw = transcribe_with_segments(
            video_path=raw_video_path,
            audio_path=preprocessed_audio,  # Use normalized audio
        )
        
        # Store transcript text for dashboard preview
        update_job(job_id, transcript_text=transcript_text_raw)
        update_last_stage(job_id, "transcribe")
        log_job_event(job_id, "info", f"Transcription complete: {len(segments)} segments", "transcribe")
        print(f"✅ Transcription complete: {len(segments)} segments")
        
        # Stage 4b: Speaker diarization (if enabled)
        diarization_result: Optional[DiarizationResult] = None
        has_diarization = False
        speaker_info = {}  # Dict of speaker_id -> SpeakerInfo for gender-based colors
        
        try:
            diarization_result = diarize_video(
                video_path=raw_video_path,
                temp_dir=str(temp_dir),
                channel_settings=channel_settings,
                audio_path=preprocessed_audio,  # Use same preprocessed audio
            )
            
            if diarization_result:
                has_diarization = True
                print(f"🎤 Diarization complete: {len(diarization_result.speakers)} speakers detected")
                
                # Get speaker info (gender, pitch, etc.) for color assignment
                speaker_info = diarization_result.speaker_info
                
                # Log gender detection results
                for spk_id, info in speaker_info.items():
                    print(f"   🎤 {spk_id}: {info.gender} (avg pitch: {info.avg_pitch:.0f}Hz)")
                
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
        
        # Generate ASS subtitles with speaker coloring (gender-based colors)
        subtitle_path = str(temp_dir / "captions.ass")
        segments_to_ass_with_diarization(
            segments=segments,
            output_path=subtitle_path,
            has_diarization=has_diarization,
            speaker_info=speaker_info,  # Pass speaker info for gender-based colors
        )
        print(f"✅ Subtitles saved to: {subtitle_path}")
        
        # Stage 5: Render vertical videos (TWO versions: with and without subtitles)
        current_stage = "render"
        update_job_status(job_id, "rendering")
        
        # Get render preset (default or specified by re-render request)
        render_preset = job.get("render_preset", "default")
        print(f"🎬 Stage 5: Rendering vertical videos (preset: {render_preset})...")
        log_job_event(job_id, "info", f"Starting render with preset: {render_preset}", "render")
        
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
        update_last_stage(job_id, "render")
        log_job_event(job_id, "info", "Render complete (both versions)", "render")
        
        # Stage 6: Upload BOTH versions to Google Drive
        current_stage = "upload"
        update_job_status(job_id, "uploading")
        print("📤 Stage 6: Uploading to Google Drive (2 versions)...")
        log_job_event(job_id, "info", "Starting upload to Google Drive", "upload")
        
        # Use display_name or twitch_login for folder name
        streamer_name = channel.get("display_name") or channel.get("twitch_login") or broadcaster_id
        
        # Use transcript text from transcription stage
        transcript_text = transcript_text_raw
        
        # Get clip timestamp from clip_info
        clip_timestamp = clip_info.get("created_at", "")
        
        # Variables to store URLs
        final_url = ""
        no_subs_url = ""
        
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
            no_subs_url = upload_result_no_subs.get("webViewLink") or upload_result_no_subs.get("webContentLink", "")
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
            final_url = upload_result_with_subs.get("webViewLink") or upload_result_with_subs.get("webContentLink", "")
            drive_path = upload_result_with_subs.get("path", "")
            
            print(f"📁 Both versions saved to: {'/'.join(drive_path.split('/')[:-1])}/")
            log_job_event(job_id, "info", f"Uploaded both versions to Google Drive", "upload", {
                "with_subtitles": final_url,
                "without_subtitles": no_subs_url,
            })
        else:
            # No Google Drive configured, keep local path
            print("⚠️ Google Drive not configured, keeping local files")
            final_url = video_with_subs_path
            no_subs_url = video_no_subs_path
            log_job_event(job_id, "warn", "Google Drive not configured, keeping local files", "upload")
        
        update_last_stage(job_id, "upload")
        
        # Mark job as ready with both URLs
        update_job(
            job_id,
            status="ready",
            final_video_url=final_url,
            no_subtitles_url=no_subs_url,
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Job {job_id} completed successfully!")
        print(f"📍 Final video: {final_url}")
        print(f"{'='*60}\n")
        
    except TwitchAPIError as e:
        error_msg = f"Twitch API error: {e}"
        print(f"❌ {error_msg}")
        log_job_event(job_id, "error", error_msg, current_stage)
        mark_job_failed(job_id, error_msg, current_stage)
        
    except VideoProcessingError as e:
        error_msg = f"Video processing error: {e}"
        print(f"❌ {error_msg}")
        log_job_event(job_id, "error", error_msg, current_stage)
        mark_job_failed(job_id, error_msg, current_stage)
    
    except SharedDriveError as e:
        error_msg = f"Google Drive error: {e}"
        print(f"❌ {error_msg}")
        log_job_event(job_id, "error", error_msg, current_stage)
        mark_job_failed(job_id, error_msg, current_stage)
        
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}: {e}"
        print(f"❌ {error_msg}")
        log_job_event(job_id, "error", error_msg, current_stage)
        mark_job_failed(job_id, error_msg, current_stage)
        raise  # Re-raise to trigger cleanup in context manager

