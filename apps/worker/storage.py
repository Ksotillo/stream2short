"""Google Drive Shared Drive storage operations for Stream2Short Worker.

Service accounts cannot upload to personal Google Drive (no storage quota).
This module uploads to a Shared Drive where the service account is a member.
"""

import os
import json
from datetime import datetime
from typing import Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from config import config

# Google Drive API scopes - full drive access for Shared Drives
SCOPES = ['https://www.googleapis.com/auth/drive']

_drive_service = None


class SharedDriveError(Exception):
    """Shared Drive operation error with helpful messages."""
    pass


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


def verify_shared_drive_access(shared_drive_id: str) -> dict:
    """
    Verify the service account has access to the Shared Drive.
    
    Args:
        shared_drive_id: The Shared Drive ID
        
    Returns:
        Shared Drive metadata
        
    Raises:
        SharedDriveError: If access denied or not found
    """
    service = get_drive_service()
    
    try:
        drive_info = service.drives().get(driveId=shared_drive_id).execute()
        print(f"✅ Verified access to Shared Drive: {drive_info.get('name', shared_drive_id)}")
        return drive_info
    except HttpError as e:
        if e.resp.status == 404:
            raise SharedDriveError(
                f"Shared Drive '{shared_drive_id}' not found. "
                f"Confirm the Shared Drive ID is correct."
            )
        elif e.resp.status == 403:
            raise SharedDriveError(
                f"Access denied to Shared Drive '{shared_drive_id}'. "
                f"Confirm the service account is a member of the Shared Drive "
                f"with 'Content manager' or higher permissions."
            )
        else:
            raise SharedDriveError(f"Failed to access Shared Drive: {e}")


def find_folder_in_shared_drive(
    shared_drive_id: str,
    folder_name: str,
    parent_id: Optional[str] = None
) -> Optional[str]:
    """
    Find a folder by name inside a Shared Drive.
    
    Args:
        shared_drive_id: The Shared Drive ID
        folder_name: Name of the folder to find
        parent_id: Optional parent folder ID (if None, searches in drive root)
        
    Returns:
        Folder ID if found, None otherwise
    """
    service = get_drive_service()
    
    # Build query
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    try:
        results = service.files().list(
            q=query,
            corpora="drive",
            driveId=shared_drive_id,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="files(id, name)",
            pageSize=1
        ).execute()
        
        files = results.get('files', [])
        if files:
            print(f"📁 Found existing folder: {folder_name}")
            return files[0]['id']
        return None
        
    except HttpError as e:
        print(f"⚠️ Error searching for folder: {e}")
        return None


def create_folder_in_shared_drive(
    shared_drive_id: str,
    folder_name: str,
    parent_id: Optional[str] = None
) -> str:
    """
    Create a folder inside a Shared Drive.
    
    Args:
        shared_drive_id: The Shared Drive ID
        folder_name: Name for the new folder
        parent_id: Parent folder ID (if None, creates in drive root)
        
    Returns:
        New folder ID
        
    Raises:
        SharedDriveError: If creation fails
    """
    service = get_drive_service()
    
    # Parent is either the specified folder or the Shared Drive root
    parent = parent_id or shared_drive_id
    
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent]
    }
    
    try:
        folder = service.files().create(
            body=folder_metadata,
            fields='id, name',
            supportsAllDrives=True
        ).execute()
        
        print(f"📁 Created folder: {folder_name} (ID: {folder['id']})")
        return folder['id']
        
    except HttpError as e:
        if e.resp.status == 403:
            raise SharedDriveError(
                f"Permission denied creating folder '{folder_name}'. "
                f"Confirm service account has 'Content manager' or higher role "
                f"on Shared Drive '{shared_drive_id}'."
            )
        raise SharedDriveError(f"Failed to create folder: {e}")


def ensure_shared_drive_folder(
    shared_drive_id: str,
    folder_name: str,
    parent_folder_id: Optional[str] = None
) -> str:
    """
    Ensure a folder exists in the Shared Drive, creating it if needed.
    
    Args:
        shared_drive_id: The Shared Drive ID
        folder_name: Name of the folder to find/create
        parent_folder_id: Optional specific parent folder ID
        
    Returns:
        Folder ID (existing or newly created)
    """
    # First verify we have access to the Shared Drive
    verify_shared_drive_access(shared_drive_id)
    
    # If a specific parent folder ID is provided, validate it exists
    if parent_folder_id:
        service = get_drive_service()
        try:
            service.files().get(
                fileId=parent_folder_id,
                supportsAllDrives=True,
                fields='id, name'
            ).execute()
            print(f"✅ Using configured parent folder: {parent_folder_id}")
            return parent_folder_id
        except HttpError:
            raise SharedDriveError(
                f"Configured GDRIVE_PARENT_FOLDER_ID '{parent_folder_id}' not found or not accessible. "
                f"Confirm the folder exists in Shared Drive '{shared_drive_id}'."
            )
    
    # Search for existing folder in Shared Drive root
    existing_folder_id = find_folder_in_shared_drive(shared_drive_id, folder_name)
    
    if existing_folder_id:
        return existing_folder_id
    
    # Create the folder in Shared Drive root
    return create_folder_in_shared_drive(shared_drive_id, folder_name)


def upload_file(
    local_path: str,
    streamer_name: str,
    job_id: str,
    filename: str = "final.mp4",
) -> dict:
    """
    Upload a file to the Shared Drive.
    
    Args:
        local_path: Path to local file
        streamer_name: Streamer's Twitch login/display name
        job_id: Job UUID (used in filename)
        filename: Output filename
        
    Returns:
        Dict with 'id', 'name', 'webViewLink', 'webContentLink', 'path'
        
    Raises:
        SharedDriveError: If upload fails
    """
    service = get_drive_service()
    
    # Validate Shared Drive is configured
    shared_drive_id = config.GDRIVE_SHARED_DRIVE_ID
    if not shared_drive_id:
        raise SharedDriveError(
            "GDRIVE_SHARED_DRIVE_ID is not set. "
            "Service accounts require a Shared Drive to upload files."
        )
    
    # Ensure destination folder exists
    destination_folder_id = ensure_shared_drive_folder(
        shared_drive_id=shared_drive_id,
        folder_name=config.GDRIVE_PARENT_FOLDER_NAME,
        parent_folder_id=config.GDRIVE_PARENT_FOLDER_ID or None
    )
    
    # Build descriptive filename
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    final_filename = f"{streamer_name}_{date_str}_{job_id[:8]}.mp4"
    
    print(f"📤 Uploading to Shared Drive: {final_filename}")
    
    # Prepare file metadata
    file_metadata = {
        'name': final_filename,
        'parents': [destination_folder_id]
    }
    
    # Upload with resumable upload for large files
    media = MediaFileUpload(
        local_path,
        mimetype='video/mp4',
        resumable=True
    )
    
    try:
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink, webContentLink',
            supportsAllDrives=True
        ).execute()
        
        print(f"✅ Uploaded: {file['name']} (ID: {file['id']})")
        
        # Files in Shared Drives inherit the drive's sharing settings
        # No need to manually set permissions
        
        return {
            'id': file['id'],
            'name': file['name'],
            'webViewLink': file.get('webViewLink', ''),
            'webContentLink': file.get('webContentLink', ''),
            'path': f"{config.GDRIVE_PARENT_FOLDER_NAME}/{final_filename}"
        }
        
    except HttpError as e:
        if e.resp.status == 403:
            error_content = e.content.decode() if e.content else str(e)
            if 'storageQuotaExceeded' in error_content:
                raise SharedDriveError(
                    "Storage quota exceeded. This should not happen with Shared Drives. "
                    f"Confirm file is being uploaded to Shared Drive '{shared_drive_id}' "
                    f"and not to the service account's personal Drive."
                )
            raise SharedDriveError(
                f"Permission denied uploading to Shared Drive. "
                f"Confirm service account is a member of Shared Drive '{shared_drive_id}' "
                f"with 'Content manager' or higher permissions. Error: {e}"
            )
        elif e.resp.status == 404:
            raise SharedDriveError(
                f"Destination folder '{destination_folder_id}' not found. "
                f"Confirm the folder exists in Shared Drive '{shared_drive_id}'."
            )
        raise SharedDriveError(f"Upload failed: {e}")


def verify_setup() -> dict:
    """
    Verify Google Drive Shared Drive setup is correct.
    
    Returns:
        Dict with verification results
        
    Useful for healthchecks and debugging.
    """
    results = {
        'shared_drive_configured': False,
        'shared_drive_accessible': False,
        'shared_drive_name': None,
        'destination_folder_id': None,
        'destination_folder_name': None,
        'error': None
    }
    
    shared_drive_id = config.GDRIVE_SHARED_DRIVE_ID
    
    if not shared_drive_id:
        results['error'] = "GDRIVE_SHARED_DRIVE_ID not configured"
        return results
    
    results['shared_drive_configured'] = True
    
    try:
        # Verify Shared Drive access
        drive_info = verify_shared_drive_access(shared_drive_id)
        results['shared_drive_accessible'] = True
        results['shared_drive_name'] = drive_info.get('name')
        
        # Ensure destination folder
        folder_id = ensure_shared_drive_folder(
            shared_drive_id=shared_drive_id,
            folder_name=config.GDRIVE_PARENT_FOLDER_NAME,
            parent_folder_id=config.GDRIVE_PARENT_FOLDER_ID or None
        )
        results['destination_folder_id'] = folder_id
        results['destination_folder_name'] = config.GDRIVE_PARENT_FOLDER_NAME
        
    except SharedDriveError as e:
        results['error'] = str(e)
    except Exception as e:
        results['error'] = f"Unexpected error: {e}"
    
    return results


def get_file_link(file_id: str) -> str:
    """
    Get a direct download link for a file.
    
    Args:
        file_id: Google Drive file ID
        
    Returns:
        Direct download URL
    """
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def delete_file(file_id: str) -> None:
    """
    Delete a file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
    """
    service = get_drive_service()
    service.files().delete(fileId=file_id, supportsAllDrives=True).execute()
    print(f"🗑️ Deleted file: {file_id}")
