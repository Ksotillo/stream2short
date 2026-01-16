"""Configuration for Stream2Short Worker."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from root .env
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


class Config:
    """Worker configuration."""
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    QUEUE_NAME: str = "clip_jobs_queue"
    
    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    
    # Twitch
    TWITCH_CLIENT_ID: str = os.getenv("TWITCH_CLIENT_ID", "")
    TWITCH_CLIENT_SECRET: str = os.getenv("TWITCH_CLIENT_SECRET", "")
    
    # Google Drive Storage (Shared Drive)
    GOOGLE_SERVICE_ACCOUNT_FILE: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "")
    GOOGLE_SERVICE_ACCOUNT_JSON: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    
    # Shared Drive settings (required for service accounts)
    GDRIVE_SHARED_DRIVE_ID: str = os.getenv("GDRIVE_SHARED_DRIVE_ID", "")
    GDRIVE_PARENT_FOLDER_ID: str = os.getenv("GDRIVE_PARENT_FOLDER_ID", "")  # Optional specific folder
    GDRIVE_PARENT_FOLDER_NAME: str = os.getenv("GDRIVE_PARENT_FOLDER_NAME", "Stream2Short Uploads")
    
    # Legacy (deprecated, use GDRIVE_SHARED_DRIVE_ID instead)
    GOOGLE_DRIVE_FOLDER_ID: str = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
    
    # Worker settings
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")
    MAX_ATTEMPTS: int = int(os.getenv("MAX_ATTEMPTS", "2"))
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp/stream2short")
    
    # Disk space management
    MIN_DISK_SPACE_GB: float = float(os.getenv("MIN_DISK_SPACE_GB", "2.0"))  # Minimum free space required
    CLEANUP_TEMP: bool = os.getenv("CLEANUP_TEMP", "true").lower() == "true"  # Default to cleanup
    
    # Clip polling
    CLIP_POLL_INTERVAL: float = 1.0  # seconds
    CLIP_POLL_MAX_ATTEMPTS: int = 15
    
    # Audio preprocessing
    ENABLE_AUDIO_NORMALIZATION: bool = os.getenv("ENABLE_AUDIO_NORMALIZATION", "true").lower() == "true"
    
    # Speaker Diarization (pyannote.audio)
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")  # Hugging Face token for gated models
    ENABLE_DIARIZATION: bool = os.getenv("ENABLE_DIARIZATION", "false").lower() == "true"
    DIARIZATION_MODEL: str = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    DIARIZATION_SEGMENTATION: str = os.getenv("DIARIZATION_SEGMENTATION", "pyannote/segmentation-3.0")
    PRIMARY_SPEAKER_STRATEGY: str = os.getenv("PRIMARY_SPEAKER_STRATEGY", "most_time")  # "most_time" or "first"
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration. Returns list of missing vars."""
        required = [
            ("SUPABASE_URL", cls.SUPABASE_URL),
            ("SUPABASE_SERVICE_ROLE_KEY", cls.SUPABASE_SERVICE_ROLE_KEY),
            ("TWITCH_CLIENT_ID", cls.TWITCH_CLIENT_ID),
            ("TWITCH_CLIENT_SECRET", cls.TWITCH_CLIENT_SECRET),
        ]
        return [name for name, value in required if not value]


config = Config()

