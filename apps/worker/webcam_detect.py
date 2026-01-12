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
) -> bool:
    """
    Detect if there are faces in a specific region of the frame.
    
    Returns True if face found, False otherwise.
    """
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
        print(f"  ✅ Face detected in {region_name}: {len(faces)} face(s)")
        return True
    
    return False


def get_corner_webcam_region(
    frame_width: int,
    frame_height: int,
    corner: str,
    webcam_width_ratio: float = 0.28,   # Webcam is ~28% of stream width
    webcam_height_ratio: float = 0.38,  # Webcam is ~38% of stream height
) -> WebcamRegion:
    """
    Get a webcam region anchored to a specific corner.
    
    Args:
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        corner: Which corner ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        webcam_width_ratio: How much of frame width the webcam takes (default 28%)
        webcam_height_ratio: How much of frame height the webcam takes (default 38%)
        
    Returns:
        WebcamRegion anchored to the specified corner
    """
    webcam_w = int(frame_width * webcam_width_ratio)
    webcam_h = int(frame_height * webcam_height_ratio)
    
    if corner == 'top-left':
        return WebcamRegion(x=0, y=0, width=webcam_w, height=webcam_h, position=corner)
    elif corner == 'top-right':
        return WebcamRegion(x=frame_width - webcam_w, y=0, width=webcam_w, height=webcam_h, position=corner)
    elif corner == 'bottom-left':
        return WebcamRegion(x=0, y=frame_height - webcam_h, width=webcam_w, height=webcam_h, position=corner)
    elif corner == 'bottom-right':
        return WebcamRegion(x=frame_width - webcam_w, y=frame_height - webcam_h, width=webcam_w, height=webcam_h, position=corner)
    else:
        raise ValueError(f"Unknown corner: {corner}")


def detect_webcam_region(
    video_path: str,
    sample_times: list[float] = [3.0, 10.0, 15.0],
    webcam_width_ratio: float = 0.28,   # Webcam typically ~28% of stream width
    webcam_height_ratio: float = 0.38,  # Webcam typically ~38% of stream height
) -> Optional[WebcamRegion]:
    """
    Detect the webcam/facecam region in a video.
    
    Strategy:
    1. Sample multiple frames and look for faces in corner regions
    2. Once we know WHICH corner has a face, return a fixed webcam crop from that corner
    3. This ensures we capture the FULL webcam overlay, not just the face
    
    Args:
        video_path: Path to the video file
        sample_times: List of times (in seconds) to sample frames from
        webcam_width_ratio: Expected webcam width as ratio of frame (default 28%)
        webcam_height_ratio: Expected webcam height as ratio of frame (default 38%)
        
    Returns:
        WebcamRegion if webcam detected, None otherwise
    """
    print(f"🔍 Detecting webcam region in: {video_path}")
    
    # Load OpenCV's pre-trained face detector
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
    
    # Track detections per corner
    corner_detections: dict[str, int] = {
        'top-left': 0,
        'top-right': 0,
        'bottom-left': 0,
        'bottom-right': 0,
    }
    
    frame_width = 0
    frame_height = 0
    
    for sample_time in sample_times:
        frame = extract_frame(video_path, sample_time)
        if frame is None:
            continue
        
        frame_height, frame_width = frame.shape[:2]
        
        # Search area is slightly larger than webcam to ensure we find faces
        search_width = int(frame_width * 0.40)
        search_height = int(frame_height * 0.45)
        
        print(f"  📸 Sampling frame at {sample_time}s ({frame_width}x{frame_height})")
        
        # Check each corner region for faces
        corners = {
            'top-left': (0, 0),
            'top-right': (frame_width - search_width, 0),
            'bottom-left': (0, frame_height - search_height),
            'bottom-right': (frame_width - search_width, frame_height - search_height),
        }
        
        for corner_name, (x_start, y_start) in corners.items():
            has_face = detect_faces_in_region(
                frame=frame,
                region_name=corner_name,
                x_start=x_start,
                y_start=y_start,
                width=search_width,
                height=search_height,
                face_cascade=face_cascade,
            )
            
            if has_face:
                corner_detections[corner_name] += 1
    
    # Find corner with most consistent face detections
    if not any(corner_detections.values()):
        print("❌ No webcam/face detected in any corner")
        return None
    
    best_corner = max(corner_detections, key=corner_detections.get)
    detection_count = corner_detections[best_corner]
    
    if detection_count == 0:
        print("❌ No consistent face detections")
        return None
    
    # Now return a fixed webcam region anchored to that corner
    result = get_corner_webcam_region(
        frame_width=frame_width,
        frame_height=frame_height,
        corner=best_corner,
        webcam_width_ratio=webcam_width_ratio,
        webcam_height_ratio=webcam_height_ratio,
    )
    
    print(f"✅ Webcam detected in {best_corner}: {result}")
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

