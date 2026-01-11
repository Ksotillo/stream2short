"""Google Drive storage operations for Stream2Short Worker."""

import os
import json
from datetime import datetime
from typing import Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from config import config

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.file']

_drive_service = None


def get_drive_service():
    """
    Get or create a Google Drive service instance.
    
    Uses service account credentials from JSON file or environment variable.
    """
    global _drive_service
    
    if _drive_service is not None:
        return _drive_service
    
    # Try to load credentials from JSON file first
    if config.GOOGLE_SERVICE_ACCOUNT_FILE and os.path.exists(config.GOOGLE_SERVICE_ACCOUNT_FILE):
        credentials = service_account.Credentials.from_service_account_file(
            config.GOOGLE_SERVICE_ACCOUNT_FILE,
            scopes=SCOPES
        )
    elif config.GOOGLE_SERVICE_ACCOUNT_JSON:
        # Load from JSON string (for Docker/environment variable)
        service_account_info = json.loads(config.GOOGLE_SERVICE_ACCOUNT_JSON)
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPES
        )
    else:
        raise ValueError(
            "Google Drive credentials not configured. "
            "Set GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON"
        )
    
    _drive_service = build('drive', 'v3', credentials=credentials)
    print("✅ Connected to Google Drive")
    
    return _drive_service


def find_or_create_folder(name: str, parent_id: Optional[str] = None) -> str:
    """
    Find a folder by name or create it if it doesn't exist.
    
    Args:
        name: Folder name
        parent_id: Parent folder ID (None for root or shared folder)
        
    Returns:
        Folder ID
    """
    service = get_drive_service()
    
    # Use the configured root folder if no parent specified
    if parent_id is None:
        parent_id = config.GOOGLE_DRIVE_FOLDER_ID
    
    if not parent_id:
        raise ValueError("GOOGLE_DRIVE_FOLDER_ID must be set - service accounts cannot upload to their own Drive")
    
    # Search for existing folder
    query = f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    query += f" and '{parent_id}' in parents"
    
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)',
        pageSize=1,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    
    files = results.get('files', [])
    
    if files:
        return files[0]['id']
    
    # Create new folder inside the parent
    folder_metadata = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    
    folder = service.files().create(
        body=folder_metadata,
        fields='id',
        supportsAllDrives=True
    ).execute()
    
    print(f"📁 Created folder: {name}")
    
    return folder['id']


def upload_file(
    local_path: str,
    streamer_name: str,
    job_id: str,
    filename: str = "final.mp4",
) -> dict:
    """
    Upload a file to Google Drive with organized folder structure.
    
    Structure: Root / {streamer_name} / {date} / {filename}
    
    Args:
        local_path: Path to local file
        streamer_name: Streamer's Twitch login/display name
        job_id: Job UUID (used in filename)
        filename: Output filename
        
    Returns:
        Dict with 'id', 'name', 'webViewLink', 'webContentLink'
    """
    service = get_drive_service()
    
    # Get today's date for folder organization
    date_folder = datetime.now().strftime('%Y-%m-%d')
    
    print(f"📤 Uploading to Google Drive: {streamer_name}/{date_folder}/{filename}")
    
    # Create folder structure: Root -> Streamer -> Date
    streamer_folder_id = find_or_create_folder(streamer_name)
    date_folder_id = find_or_create_folder(date_folder, streamer_folder_id)
    
    # Prepare file metadata
    # Use job_id in filename to ensure uniqueness
    final_filename = f"clip_{job_id[:8]}_{filename}"
    
    file_metadata = {
        'name': final_filename,
        'parents': [date_folder_id]
    }
    
    # Upload file
    media = MediaFileUpload(
        local_path,
        mimetype='video/mp4',
        resumable=True
    )
    
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, name, webViewLink, webContentLink',
        supportsAllDrives=True
    ).execute()
    
    print(f"✅ Uploaded: {file['name']} (ID: {file['id']})")
    
    # Make file accessible via link (anyone with link can view)
    try:
        service.permissions().create(
            fileId=file['id'],
            body={
                'type': 'anyone',
                'role': 'reader'
            },
            supportsAllDrives=True
        ).execute()
        print("🔗 File shared with link access")
    except Exception as e:
        print(f"⚠️ Could not set sharing permissions: {e}")
    
    return {
        'id': file['id'],
        'name': file['name'],
        'webViewLink': file.get('webViewLink', ''),
        'webContentLink': file.get('webContentLink', ''),
        'path': f"{streamer_name}/{date_folder}/{final_filename}"
    }


def get_file_link(file_id: str) -> str:
    """
    Get a direct download link for a file.
    
    Args:
        file_id: Google Drive file ID
        
    Returns:
        Direct download URL
    """
    # Google Drive direct download format
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def delete_file(file_id: str) -> None:
    """
    Delete a file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
    """
    service = get_drive_service()
    service.files().delete(fileId=file_id).execute()
    print(f"🗑️ Deleted file: {file_id}")


def list_streamer_clips(streamer_name: str, limit: int = 50) -> list:
    """
    List all clips for a streamer.
    
    Args:
        streamer_name: Streamer's name (folder name)
        limit: Maximum number of results
        
    Returns:
        List of file metadata dicts
    """
    service = get_drive_service()
    
    # Find streamer folder
    try:
        streamer_folder_id = find_or_create_folder(streamer_name)
    except Exception:
        return []
    
    # List all video files in streamer's folder (recursive)
    query = (
        f"'{streamer_folder_id}' in parents or "
        f"mimeType = 'video/mp4' and trashed = false"
    )
    
    results = service.files().list(
        q=f"mimeType = 'video/mp4' and trashed = false",
        spaces='drive',
        fields='files(id, name, webViewLink, webContentLink, createdTime, size)',
        pageSize=limit,
        orderBy='createdTime desc'
    ).execute()
    
    return results.get('files', [])
