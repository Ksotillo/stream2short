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
    
    # Clip polling
    CLIP_POLL_INTERVAL: float = 1.0  # seconds
    CLIP_POLL_MAX_ATTEMPTS: int = 15
    
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

