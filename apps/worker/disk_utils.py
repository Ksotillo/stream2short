"""Disk space management utilities for Stream2Short Worker.

Provides:
- Disk space checks before processing
- Per-job temp directory with guaranteed cleanup
- Size tracking and logging
"""

import os
import shutil
from pathlib import Path
from contextlib import contextmanager
from typing import Generator
from dataclasses import dataclass

from config import config


class DiskSpaceError(Exception):
    """Raised when there's not enough disk space to process a job."""
    pass


@dataclass
class DiskUsageStats:
    """Statistics about disk usage for a job."""
    initial_free_gb: float
    final_free_gb: float
    job_size_mb: float
    files_count: int


def get_free_disk_space_gb(path: str = None) -> float:
    """
    Get free disk space in gigabytes for the given path.
    
    Args:
        path: Path to check (defaults to TEMP_DIR)
        
    Returns:
        Free space in GB
    """
    if path is None:
        path = config.TEMP_DIR
    
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    
    try:
        stat = shutil.disk_usage(path)
        return stat.free / (1024 ** 3)  # Convert bytes to GB
    except Exception as e:
        print(f"⚠️ Failed to get disk usage for {path}: {e}")
        return 0.0


def check_disk_space(required_gb: float = None) -> tuple[bool, float]:
    """
    Check if there's enough free disk space to process a job.
    
    Args:
        required_gb: Minimum required space in GB (defaults to config.MIN_DISK_SPACE_GB)
        
    Returns:
        Tuple of (has_enough_space, current_free_gb)
    """
    if required_gb is None:
        required_gb = config.MIN_DISK_SPACE_GB
    
    free_gb = get_free_disk_space_gb()
    has_enough = free_gb >= required_gb
    
    return has_enough, free_gb


def get_directory_size_mb(path: str) -> float:
    """
    Get total size of a directory in megabytes.
    
    Args:
        path: Directory path
        
    Returns:
        Size in MB
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except Exception:
        pass
    
    return total_size / (1024 ** 2)  # Convert bytes to MB


def count_files_in_directory(path: str) -> int:
    """
    Count total files in a directory (recursively).
    
    Args:
        path: Directory path
        
    Returns:
        File count
    """
    count = 0
    try:
        for _, _, filenames in os.walk(path):
            count += len(filenames)
    except Exception:
        pass
    return count


def cleanup_temp_directory(path: str, force: bool = False) -> bool:
    """
    Clean up a temporary directory.
    
    Args:
        path: Directory to remove
        force: Force removal even if CLEANUP_TEMP is false
        
    Returns:
        True if cleaned up, False otherwise
    """
    if not os.path.exists(path):
        return True
    
    # Check if cleanup is enabled (unless forced)
    if not force and not config.CLEANUP_TEMP:
        return False
    
    try:
        size_mb = get_directory_size_mb(path)
        file_count = count_files_in_directory(path)
        
        shutil.rmtree(path, ignore_errors=True)
        
        if not os.path.exists(path):
            print(f"🧹 Cleaned up temp directory: {path} ({size_mb:.1f}MB, {file_count} files)")
            return True
        else:
            print(f"⚠️ Failed to fully cleanup: {path}")
            return False
    except Exception as e:
        print(f"⚠️ Error during cleanup of {path}: {e}")
        return False


def cleanup_old_temp_directories(max_age_hours: int = 24) -> int:
    """
    Clean up old temp directories that may have been left behind from crashed jobs.
    
    Args:
        max_age_hours: Max age in hours before directory is considered stale
        
    Returns:
        Number of directories cleaned up
    """
    import time
    
    temp_root = Path(config.TEMP_DIR)
    if not temp_root.exists():
        return 0
    
    max_age_seconds = max_age_hours * 3600
    now = time.time()
    cleaned = 0
    
    try:
        for item in temp_root.iterdir():
            if item.is_dir():
                try:
                    # Check directory age
                    mtime = item.stat().st_mtime
                    age = now - mtime
                    
                    if age > max_age_seconds:
                        size_mb = get_directory_size_mb(str(item))
                        shutil.rmtree(item, ignore_errors=True)
                        print(f"🧹 Cleaned stale temp: {item.name} ({size_mb:.1f}MB, {age/3600:.1f}h old)")
                        cleaned += 1
                except Exception as e:
                    print(f"⚠️ Error checking/cleaning {item}: {e}")
    except Exception as e:
        print(f"⚠️ Error scanning temp directory: {e}")
    
    return cleaned


@contextmanager
def job_temp_directory(job_id: str) -> Generator[tuple[Path, DiskUsageStats], None, None]:
    """
    Context manager that creates a per-job temp directory with guaranteed cleanup.
    
    Usage:
        with job_temp_directory(job_id) as (temp_dir, stats):
            # ... process job using temp_dir ...
        # Cleanup happens automatically, even on exceptions
    
    Args:
        job_id: Unique job identifier
        
    Yields:
        Tuple of (temp_dir_path, stats_object)
        
    Raises:
        DiskSpaceError: If there's not enough disk space
    """
    # Check disk space before starting
    has_space, free_gb = check_disk_space()
    
    if not has_space:
        raise DiskSpaceError(
            f"Not enough disk space! "
            f"Free: {free_gb:.2f}GB, Required: {config.MIN_DISK_SPACE_GB:.1f}GB. "
            f"Clean up old files or increase disk space."
        )
    
    initial_free_gb = free_gb
    print(f"💾 Disk space: {free_gb:.2f}GB free (min: {config.MIN_DISK_SPACE_GB:.1f}GB)")
    
    # Create temp directory
    temp_dir = Path(config.TEMP_DIR) / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Stats object to track usage
    stats = DiskUsageStats(
        initial_free_gb=initial_free_gb,
        final_free_gb=0.0,
        job_size_mb=0.0,
        files_count=0
    )
    
    try:
        yield temp_dir, stats
        
    finally:
        # Always calculate final stats
        stats.job_size_mb = get_directory_size_mb(str(temp_dir))
        stats.files_count = count_files_in_directory(str(temp_dir))
        stats.final_free_gb = get_free_disk_space_gb()
        
        # Log disk usage
        print(f"💾 Job used: {stats.job_size_mb:.1f}MB ({stats.files_count} files)")
        print(f"💾 Disk space: {stats.final_free_gb:.2f}GB free (was {stats.initial_free_gb:.2f}GB)")
        
        # Always cleanup temp directory (unless CLEANUP_TEMP is false)
        cleanup_temp_directory(str(temp_dir))


def print_disk_status() -> None:
    """Print current disk status for debugging."""
    free_gb = get_free_disk_space_gb()
    temp_root = Path(config.TEMP_DIR)
    
    print(f"\n📊 Disk Status:")
    print(f"   Free space: {free_gb:.2f}GB")
    print(f"   Min required: {config.MIN_DISK_SPACE_GB:.1f}GB")
    print(f"   Temp directory: {config.TEMP_DIR}")
    print(f"   Auto cleanup: {'enabled' if config.CLEANUP_TEMP else 'disabled'}")
    
    if temp_root.exists():
        total_temp_mb = get_directory_size_mb(str(temp_root))
        job_dirs = list(temp_root.iterdir()) if temp_root.exists() else []
        print(f"   Temp usage: {total_temp_mb:.1f}MB ({len(job_dirs)} job directories)")
    else:
        print(f"   Temp usage: 0MB (directory doesn't exist)")
    
    print()

