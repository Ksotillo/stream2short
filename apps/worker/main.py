#!/usr/bin/env python3
"""Stream2Short Worker - Main entry point.

Consumes jobs from Redis queue and processes them through the clip pipeline.
"""

import signal
import sys
import redis
from config import config
from pipeline import process_job
from db import increment_attempt_count, mark_job_failed
from disk_utils import (
    print_disk_status,
    cleanup_old_temp_directories,
    check_disk_space,
)

# Global flag for graceful shutdown
running = True


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global running
    print("\n🛑 Shutdown signal received, finishing current job...")
    running = False


def requeue_job(redis_client: redis.Redis, job_id: str) -> None:
    """Re-queue a job for retry."""
    redis_client.lpush(config.QUEUE_NAME, job_id)
    print(f"🔄 Re-queued job {job_id}")


def main():
    """Main worker loop."""
    global running
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Validate configuration
    missing = config.validate()
    if missing:
        print(f"❌ Missing required configuration: {', '.join(missing)}")
        print("Please set the required environment variables.")
        sys.exit(1)
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║               Stream2Short Worker                         ║
╠═══════════════════════════════════════════════════════════╣
║  Starting up...                                           ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    # Print disk status and check space
    print_disk_status()
    
    has_space, free_gb = check_disk_space()
    if not has_space:
        print(f"⚠️ Warning: Low disk space! {free_gb:.2f}GB free, {config.MIN_DISK_SPACE_GB:.1f}GB required")
        print("   Worker will fail jobs until more space is available.")
    
    # Cleanup stale temp directories from crashed jobs
    stale_cleaned = cleanup_old_temp_directories(max_age_hours=24)
    if stale_cleaned > 0:
        print(f"🧹 Cleaned {stale_cleaned} stale temp directories from previous runs")
    
    # Connect to Redis
    try:
        redis_client = redis.from_url(config.REDIS_URL)
        redis_client.ping()
        print(f"✅ Connected to Redis: {config.REDIS_URL}")
    except redis.ConnectionError as e:
        print(f"❌ Failed to connect to Redis: {e}")
        sys.exit(1)
    
    print(f"👂 Listening for jobs on queue: {config.QUEUE_NAME}")
    print("Press Ctrl+C to stop\n")
    
    while running:
        try:
            # Blocking pop with 5 second timeout
            # This allows us to check the running flag periodically
            result = redis_client.brpop(config.QUEUE_NAME, timeout=5)
            
            if result is None:
                # Timeout, just continue to check running flag
                continue
            
            _, job_id_bytes = result
            job_id = job_id_bytes.decode("utf-8")
            
            print(f"\n📥 Received job: {job_id}")
            
            # Increment attempt count
            attempt_count = increment_attempt_count(job_id)
            
            if attempt_count > config.MAX_ATTEMPTS:
                print(f"❌ Job {job_id} exceeded max attempts ({config.MAX_ATTEMPTS})")
                mark_job_failed(job_id, f"Exceeded max attempts ({config.MAX_ATTEMPTS})")
                continue
            
            print(f"🔄 Attempt {attempt_count}/{config.MAX_ATTEMPTS}")
            
            # Process the job
            try:
                process_job(job_id)
            except Exception as e:
                print(f"❌ Job {job_id} failed: {e}")
                
                # Re-queue if under max attempts
                if attempt_count < config.MAX_ATTEMPTS:
                    requeue_job(redis_client, job_id)
                else:
                    mark_job_failed(job_id, str(e))
                    
        except redis.ConnectionError as e:
            print(f"⚠️ Redis connection error: {e}")
            print("Attempting to reconnect in 5 seconds...")
            import time
            time.sleep(5)
            try:
                redis_client = redis.from_url(config.REDIS_URL)
                redis_client.ping()
                print("✅ Reconnected to Redis")
            except redis.ConnectionError:
                print("❌ Failed to reconnect")
                
        except Exception as e:
            print(f"❌ Unexpected error in worker loop: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n👋 Worker stopped gracefully")


if __name__ == "__main__":
    main()

