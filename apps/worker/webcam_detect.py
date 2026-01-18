"""Webcam/face detection for Stream2Short Worker.

Detects the webcam overlay position in stream recordings and returns
the region coordinates for video compositing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class WebcamRegion:
    """Represents a detected webcam region in the video."""
    
    def __init__(self, x: int, y: int, width: int, height: int, position: str):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.position = position  # 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    
    def __repr__(self):
        return f"WebcamRegion({self.position}: x={self.x}, y={self.y}, w={self.width}, h={self.height})"
    
    def to_ffmpeg_crop(self) -> str:
        """Return FFmpeg crop filter string."""
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


def extract_frame(video_path: str, time_sec: float = 5.0) -> Optional[np.ndarray]:
    """
    Extract a single frame from a video at the specified time.
    
    Args:
        video_path: Path to the video file
        time_sec: Time in seconds to extract frame from
        
    Returns:
        Frame as numpy array (BGR) or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"⚠️ Could not open video: {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30  # Default fallback
    
    # Calculate target frame number
    target_frame = int(time_sec * fps)
    target_frame = min(target_frame, total_frames - 1)  # Don't exceed video length
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"⚠️ Could not read frame at {time_sec}s")
        return None
    
    return frame


def detect_faces_in_region(
    frame: np.ndarray,
    region_name: str,
    x_start: int,
    y_start: int,
    width: int,
    height: int,
    face_cascade: cv2.CascadeClassifier,
    padding_ratio: float = 2.0,  # How much padding around face (2.0 = 200% extra - much more zoomed out)
) -> Optional[WebcamRegion]:
    """
    Detect faces in a specific region of the frame.
    
    Returns WebcamRegion with generous padding around face to capture
    the full webcam frame, not just a close-up of the face.
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Crop the region
    region = frame[y_start:y_start+height, x_start:x_start+width]
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(50, 50),  # Min face size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        # Found at least one face - get the largest one
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = largest_face
        
        # Calculate crop around face with GENEROUS padding
        # Goal: Capture the full webcam frame, not just the face
        # The face coordinates are relative to the region, convert to frame coordinates
        face_center_x = x_start + fx + fw // 2
        face_center_y = y_start + fy + fh // 2
        
        # Calculate crop size with generous padding
        # We want to see shoulders/torso, not just face
        face_size = max(fw, fh)
        
        # Much larger multiplier to zoom out and show more of the person
        crop_height = int(face_size * (1 + padding_ratio * 2))  # Face + lots of padding
        
        # Make the crop wider than tall (16:9 webcam aspect ratio)
        crop_width = int(crop_height * 16 / 9)
        
        # Shift the center down slightly to show more body (face typically in upper third of webcam)
        adjusted_center_y = face_center_y + int(face_size * 0.3)
        
        # Calculate crop coordinates centered on adjusted position
        crop_x = face_center_x - crop_width // 2
        crop_y = adjusted_center_y - crop_height // 2
        
        # Clamp to frame boundaries
        crop_x = max(0, min(crop_x, frame_width - crop_width))
        crop_y = max(0, min(crop_y, frame_height - crop_height))
        crop_width = min(crop_width, frame_width - crop_x)
        crop_height = min(crop_height, frame_height - crop_y)
        
        # Ensure we have reasonable dimensions
        if crop_width < 100 or crop_height < 100:
            print(f"  ⚠️ Crop too small in {region_name}, skipping")
            return None
        
        print(f"  ✅ Face detected in {region_name}: face={fw}x{fh}, crop={crop_width}x{crop_height} (zoomed out)")
        
        return WebcamRegion(
            x=crop_x,
            y=crop_y,
            width=crop_width,
            height=crop_height,
            position=region_name
        )
    
    return None


def detect_webcam_region(
    video_path: str,
    sample_times: list[float] = [3.0, 10.0, 15.0],
) -> Optional[WebcamRegion]:
    """
    Detect the webcam/facecam region in a video.
    
    Samples multiple frames and looks for faces in the corner regions.
    Returns the region where a face is most consistently detected.
    
    Args:
        video_path: Path to the video file
        sample_times: List of times (in seconds) to sample frames from
        
    Returns:
        WebcamRegion if webcam detected, None otherwise
    """
    print(f"🔍 Detecting webcam region in: {video_path}")
    
    # Load OpenCV's pre-trained face detector
    # Try multiple paths for the cascade file
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    ]
    
    face_cascade = None
    for path in cascade_paths:
        if Path(path).exists():
            face_cascade = cv2.CascadeClassifier(path)
            break
    
    if face_cascade is None or face_cascade.empty():
        print("⚠️ Could not load face cascade classifier")
        return None
    
    # Track detections per region
    region_detections: dict[str, int] = {
        'top-left': 0,
        'top-right': 0,
        'bottom-left': 0,
        'bottom-right': 0,
    }
    
    detected_regions: dict[str, WebcamRegion] = {}
    
    for sample_time in sample_times:
        frame = extract_frame(video_path, sample_time)
        if frame is None:
            continue
        
        frame_height, frame_width = frame.shape[:2]
        
        # Define corner regions (approximately 30% of each dimension)
        region_width = int(frame_width * 0.35)
        region_height = int(frame_height * 0.40)
        
        print(f"  📸 Sampling frame at {sample_time}s ({frame_width}x{frame_height})")
        
        # Check each corner region
        corners = {
            'top-left': (0, 0),
            'top-right': (frame_width - region_width, 0),
            'bottom-left': (0, frame_height - region_height),
            'bottom-right': (frame_width - region_width, frame_height - region_height),
        }
        
        for region_name, (x_start, y_start) in corners.items():
            result = detect_faces_in_region(
                frame=frame,
                region_name=region_name,
                x_start=x_start,
                y_start=y_start,
                width=region_width,
                height=region_height,
                face_cascade=face_cascade,
            )
            
            if result:
                region_detections[region_name] += 1
                detected_regions[region_name] = result
    
    # Find region with most consistent detections
    if not any(region_detections.values()):
        print("❌ No webcam/face detected in any corner")
        return None
    
    best_region = max(region_detections, key=region_detections.get)
    detection_count = region_detections[best_region]
    
    if detection_count == 0:
        print("❌ No consistent face detections")
        return None
    
    result = detected_regions[best_region]
    print(f"✅ Webcam detected: {result}")
    return result


def get_video_dimensions(video_path: str) -> Tuple[int, int]:
    """Get video width and height."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


if __name__ == "__main__":
    # Test with a video file
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        result = detect_webcam_region(video_path)
        print(f"\nResult: {result}")

