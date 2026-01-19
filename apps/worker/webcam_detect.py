"""Webcam/face detection for Stream2Short Worker.

Detects the webcam overlay position in stream recordings and returns
the region coordinates for video compositing.

Strategy:
1. Use Gemini Vision AI to detect IF a webcam exists and WHICH corner
2. Refine the bounding box using OpenCV edge/contour detection
3. Fall back to OpenCV face detection if Gemini unavailable
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict


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


def refine_webcam_bbox(
    frame_bgr: np.ndarray,
    corner: str,
    video_width: int,
    video_height: int,
    roi_ratio: float = 0.45,
) -> Optional[Dict[str, int]]:
    """
    Refine webcam bounding box using edge detection and contour analysis.
    
    Gemini often returns face-like square bboxes instead of the actual webcam overlay.
    This function analyzes the corner region to find the true rectangular overlay.
    
    Args:
        frame_bgr: Full video frame (BGR)
        corner: Which corner to search ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        video_width: Full frame width
        video_height: Full frame height
        roi_ratio: What fraction of the frame to use as ROI (0.45 = 45%)
    
    Returns:
        Dict with x, y, width, height in full-frame coordinates, or None
    """
    print(f"  🔬 Refining bbox in {corner} using edge detection...")
    
    # Calculate ROI bounds
    roi_w = int(video_width * roi_ratio)
    roi_h = int(video_height * roi_ratio)
    
    if corner == 'top-left':
        roi_x, roi_y = 0, 0
    elif corner == 'top-right':
        roi_x, roi_y = video_width - roi_w, 0
    elif corner == 'bottom-left':
        roi_x, roi_y = 0, video_height - roi_h
    elif corner == 'bottom-right':
        roi_x, roi_y = video_width - roi_w, video_height - roi_h
    else:
        print(f"  ⚠️ Unknown corner: {corner}")
        return None
    
    # Extract ROI
    roi = frame_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].copy()
    roi_area = roi_w * roi_h
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Morphological close to connect broken edges
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"  ⚠️ No contours found in {corner} ROI")
        return None
    
    # Filter and score candidates
    candidates = []
    min_area = roi_area * 0.05  # At least 5% of ROI
    
    for contour in contours:
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get bounding rectangle
        bx, by, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        
        # Filter by area
        if area < min_area:
            continue
        
        # Filter by aspect ratio (typical webcam: 1.1 to 2.6)
        ar = bw / bh if bh > 0 else 0
        if ar < 1.1 or ar > 2.6:
            continue
        
        # Calculate corner distance (prefer boxes closer to the actual corner)
        if corner == 'top-left':
            corner_dist = bx + by
        elif corner == 'top-right':
            corner_dist = (roi_w - (bx + bw)) + by
        elif corner == 'bottom-left':
            corner_dist = bx + (roi_h - (by + bh))
        elif corner == 'bottom-right':
            corner_dist = (roi_w - (bx + bw)) + (roi_h - (by + bh))
        else:
            corner_dist = bx + by
        
        # Score: prefer larger area and closer to corner
        # Normalize distance by ROI diagonal
        roi_diag = np.sqrt(roi_w**2 + roi_h**2)
        dist_score = 1 - (corner_dist / roi_diag)  # 0-1, higher = closer to corner
        area_score = area / roi_area  # 0-1, higher = larger
        
        # Combined score (weight area more)
        score = area_score * 0.7 + dist_score * 0.3
        
        candidates.append({
            'x': bx,
            'y': by,
            'width': bw,
            'height': bh,
            'area': area,
            'ar': ar,
            'corner_dist': corner_dist,
            'score': score,
            'vertices': len(approx)
        })
        
        print(f"     Candidate: {bw}x{bh} at ({bx},{by}), AR={ar:.2f}, dist={corner_dist}, score={score:.3f}")
    
    if not candidates:
        print(f"  ⚠️ No valid rectangle candidates in {corner} ROI")
        return None
    
    # Sort by score (highest first)
    candidates.sort(key=lambda c: c['score'], reverse=True)
    best = candidates[0]
    
    # Convert to full-frame coordinates
    full_x = roi_x + best['x']
    full_y = roi_y + best['y']
    
    print(f"  ✅ Refined bbox: {best['width']}x{best['height']} at ({full_x},{full_y}), AR={best['ar']:.2f}")
    
    return {
        'x': full_x,
        'y': full_y,
        'width': best['width'],
        'height': best['height'],
        'corner': corner
    }


def is_bbox_suspicious(bbox: Dict[str, int], video_width: int, video_height: int) -> Tuple[bool, str]:
    """
    Check if Gemini's bbox looks suspicious (likely a face crop, not webcam overlay).
    
    Args:
        bbox: Dict with x, y, width, height
        video_width: Full frame width
        video_height: Full frame height
    
    Returns:
        Tuple of (is_suspicious, reason)
    """
    w = bbox.get('width', 0)
    h = bbox.get('height', 0)
    
    if w <= 0 or h <= 0:
        return True, "invalid dimensions"
    
    ar = w / h
    area = w * h
    frame_area = video_width * video_height
    area_ratio = area / frame_area
    
    # Check 1: Near-square aspect ratio (faces are ~1:1, webcams are wider)
    if 0.85 <= ar <= 1.15:
        return True, f"near-square AR={ar:.2f} (likely face crop)"
    
    # Check 2: Too small (less than 3% of frame)
    if area_ratio < 0.03:
        return True, f"too small ({area_ratio*100:.1f}% of frame)"
    
    # Check 3: Inverted aspect ratio (taller than wide)
    if ar < 1.0:
        return True, f"inverted AR={ar:.2f} (taller than wide)"
    
    return False, "looks valid"


def create_fallback_widescreen_bbox(
    corner: str,
    gemini_height: int,
    video_width: int,
    video_height: int,
    target_ar: float = 16/9,
    max_ratio: float = 0.50,
) -> Dict[str, int]:
    """
    Create a fallback widescreen bbox when refinement fails.
    
    Uses the height from Gemini (or a reasonable default) and computes
    width to achieve target aspect ratio (16:9).
    
    Args:
        corner: Which corner to anchor the bbox
        gemini_height: Height from Gemini's bbox (or 0 for default)
        video_width: Full frame width
        video_height: Full frame height
        target_ar: Target aspect ratio (width/height)
        max_ratio: Max fraction of frame for any dimension
    
    Returns:
        Dict with x, y, width, height
    """
    print(f"  🎯 Creating fallback widescreen bbox in {corner}...")
    
    # Use Gemini height or reasonable default (20% of frame height)
    h = gemini_height if gemini_height > 50 else int(video_height * 0.20)
    
    # Compute width for target AR
    w = int(h * target_ar)
    
    # Clamp to max ratio
    max_w = int(video_width * max_ratio)
    max_h = int(video_height * max_ratio)
    
    if w > max_w:
        w = max_w
        h = int(w / target_ar)
    if h > max_h:
        h = max_h
        w = int(h * target_ar)
    
    # Make even
    w = w - (w % 2)
    h = h - (h % 2)
    
    # Anchor to corner
    if corner == 'top-left':
        x, y = 0, 0
    elif corner == 'top-right':
        x, y = video_width - w, 0
    elif corner == 'bottom-left':
        x, y = 0, video_height - h
    elif corner == 'bottom-right':
        x, y = video_width - w, video_height - h
    else:
        x, y = video_width - w, 0  # Default to top-right
    
    print(f"  📐 Fallback bbox: {w}x{h} at ({x},{y}), AR={w/h:.2f}")
    
    return {
        'x': x,
        'y': y,
        'width': w,
        'height': h,
        'corner': corner
    }


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
    temp_dir: str = "/tmp",
) -> Optional[WebcamRegion]:
    """
    Detect the webcam/facecam region in a video.
    
    First tries Gemini Vision AI for accurate webcam rectangle detection.
    Falls back to OpenCV face detection if Gemini is not configured or fails.
    
    Args:
        video_path: Path to the video file
        sample_times: List of times (in seconds) to sample frames from
        temp_dir: Directory for temporary frame extraction
        
    Returns:
        WebcamRegion if webcam detected, None otherwise
    """
    print(f"🔍 Detecting webcam region in: {video_path}")
    
    # =========================================================================
    # Strategy 1: Try Gemini Vision AI first (more accurate)
    # =========================================================================
    gemini_attempted = False
    gemini_explicitly_no_webcam = False
    
    try:
        from gemini_vision import detect_webcam_with_gemini, extract_frame_for_analysis, save_debug_frame_with_box
        from config import config
        
        if config.GEMINI_API_KEY:
            gemini_attempted = True
            print("  🤖 Attempting Gemini Vision detection...")
            
            # Get video dimensions
            width, height = get_video_dimensions(video_path)
            
            # Try multiple frames for better detection
            timestamps_to_try = [3.0, 8.0, 15.0]
            
            for ts in timestamps_to_try:
                frame_path = f"{temp_dir}/webcam_detect_frame_{int(ts)}.jpg"
                print(f"  📸 Analyzing frame at {ts}s...")
                
                if extract_frame_for_analysis(video_path, frame_path, timestamp=ts):
                    result = detect_webcam_with_gemini(frame_path, width, height)
                    
                    # Clean up frame
                    try:
                        import os
                        os.remove(frame_path)
                    except:
                        pass
                    
                    if result:
                        if result.get('no_webcam_confirmed'):
                            # Gemini explicitly said there's no webcam - trust it, skip OpenCV
                            print("  ℹ️ Gemini confirmed: NO webcam in this video")
                            gemini_explicitly_no_webcam = True
                            break  # Don't try more frames
                        else:
                            print(f"  ✅ Gemini found webcam: {result}")
                            
                            # Use corner from Gemini if available
                            position = result.get('corner', 'top-left')
                            if position == 'unknown':
                                # Fallback: determine from coordinates
                                if result['x'] > width / 2:
                                    position = 'top-right' if result['y'] < height / 2 else 'bottom-right'
                                elif result['y'] > height / 2:
                                    position = 'bottom-left'
                                else:
                                    position = 'top-left'
                            
                            cam_x = result['x']
                            cam_y = result['y']
                            cam_w = result['width']
                            cam_h = result['height']
                            
                            # ============================================================
                            # CHECK IF GEMINI BBOX IS SUSPICIOUS (face crop vs overlay)
                            # ============================================================
                            gemini_ar = cam_w / cam_h if cam_h > 0 else 0
                            is_suspicious, suspicious_reason = is_bbox_suspicious(result, width, height)
                            
                            print(f"  📐 Gemini bbox: {cam_w}x{cam_h} at ({cam_x},{cam_y}), AR={gemini_ar:.2f}")
                            
                            if is_suspicious:
                                print(f"  ⚠️ Gemini bbox suspicious: {suspicious_reason}")
                                print(f"  🔬 Running OpenCV refinement in {position} corner...")
                                
                                # Load the frame for refinement
                                frame_for_refine = cv2.imread(frame_path)
                                refined_bbox = None
                                
                                if frame_for_refine is not None:
                                    refined_bbox = refine_webcam_bbox(
                                        frame_for_refine, position, width, height
                                    )
                                
                                if refined_bbox:
                                    # Use refined bbox
                                    print(f"  ✅ Using refined bbox instead of Gemini's")
                                    cam_x = refined_bbox['x']
                                    cam_y = refined_bbox['y']
                                    cam_w = refined_bbox['width']
                                    cam_h = refined_bbox['height']
                                else:
                                    # Refinement failed - use fallback heuristic
                                    print(f"  ⚠️ Refinement failed, using fallback widescreen bbox")
                                    fallback = create_fallback_widescreen_bbox(
                                        position, cam_h, width, height
                                    )
                                    cam_x = fallback['x']
                                    cam_y = fallback['y']
                                    cam_w = fallback['width']
                                    cam_h = fallback['height']
                            else:
                                print(f"  ✅ Gemini bbox looks valid (not suspicious)")
                            
                            # ============================================================
                            # APPLY PADDING (after refinement/fallback)
                            # ============================================================
                            # Add 15% padding on each side
                            padding_x = int(cam_w * 0.15)
                            padding_y = int(cam_h * 0.15)
                            
                            # Expand region
                            new_x = cam_x - padding_x
                            new_y = cam_y - padding_y
                            new_w = cam_w + (padding_x * 2)
                            new_h = cam_h + (padding_y * 2)
                            
                            # Clamp to video bounds based on corner position
                            if 'right' in position:
                                # Right-side: ensure we don't go past right edge
                                if new_x + new_w > width:
                                    new_x = width - new_w
                                if new_x < 0:
                                    new_w = new_w + new_x  # Reduce width
                                    new_x = 0
                            else:
                                # Left-side: ensure we don't go past left edge
                                if new_x < 0:
                                    new_x = 0
                                if new_x + new_w > width:
                                    new_w = width - new_x
                            
                            if 'bottom' in position:
                                if new_y + new_h > height:
                                    new_y = height - new_h
                                if new_y < 0:
                                    new_h = new_h + new_y
                                    new_y = 0
                            else:
                                if new_y < 0:
                                    new_y = 0
                                if new_y + new_h > height:
                                    new_h = height - new_y
                            
                            print(f"  📐 Final bbox: {new_w}x{new_h} at ({new_x},{new_y}) (with padding)")
                            
                            # Save debug frame showing where webcam was detected
                            debug_frame_path = f"{temp_dir}/debug_webcam_detection.jpg"
                            save_debug_frame_with_box(
                                frame_path, debug_frame_path,
                                new_x, new_y, new_w, new_h
                            )
                            
                            return WebcamRegion(
                                x=new_x,
                                y=new_y,
                                width=new_w,
                                height=new_h,
                                position=position
                            )
                else:
                    print(f"  ⚠️ Could not extract frame at {ts}s")
            
            if not gemini_explicitly_no_webcam:
                print("  ⚠️ Gemini didn't find webcam, falling back to face detection")
    except ImportError:
        print("  ⚠️ Gemini module not available")
    except Exception as e:
        print(f"  ⚠️ Gemini detection failed: {e}")
    
    # If Gemini explicitly said "no webcam", trust it and skip OpenCV
    # (OpenCV has too many false positives with game graphics)
    if gemini_explicitly_no_webcam:
        print("  📹 Skipping OpenCV (Gemini confirmed no webcam) → using simple crop")
        return None
    
    # =========================================================================
    # Strategy 2: Fall back to OpenCV face detection
    # Only if Gemini wasn't configured or had an API error (not "no webcam")
    # =========================================================================
    print("  🔍 Using OpenCV face detection...")
    
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

