#!/usr/bin/env python3
"""
Debug script for webcam bbox detection.

Tests the webcam detection pipeline on a video file and outputs debug images
showing the detected bbox, variance maps, and refinement results.

Usage:
    python scripts/debug_webcam_bbox.py <video_path> [--timestamps 3,10,15] [--output-dir /tmp/debug]

Example:
    python scripts/debug_webcam_bbox.py /path/to/clip.mp4 --timestamps 3,10,15
"""

import argparse
import os
import sys
import tempfile

# Add the worker directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apps', 'worker'))

import cv2
import numpy as np

from webcam_detect import (
    extract_frame,
    detect_face_dnn,
    refine_temporal_stability,
    refine_edge_scan_from_face,
    refine_side_box_tight_edges,
    refine_side_box_bbox_contours,
    refine_side_box_bbox,
)

try:
    from gemini_vision import detect_webcam_with_gemini
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️ Gemini vision not available, will use manual bbox")


def get_video_dimensions(video_path: str) -> tuple:
    """Get video dimensions using ffprobe."""
    import subprocess
    import json
    
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video':
            return stream['width'], stream['height']
    
    return 1920, 1080  # Default fallback


def draw_bbox_on_frame(frame, bbox, color, label, thickness=2):
    """Draw a bounding box with label on a frame."""
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Add label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    cv2.putText(frame, label, (x, y - 5), font, font_scale, color, 2)


def main():
    parser = argparse.ArgumentParser(description='Debug webcam bbox detection')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--timestamps', default='3,10,15',
                        help='Comma-separated timestamps to sample (default: 3,10,15)')
    parser.add_argument('--output-dir', default='/tmp/debug_webcam',
                        help='Output directory for debug images')
    parser.add_argument('--gemini-bbox', default=None,
                        help='Manual Gemini bbox as x,y,w,h (e.g., 800,100,350,250)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Video not found: {args.video_path}")
        return 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    timestamps = [float(t.strip()) for t in args.timestamps.split(',')]
    print(f"⏱️ Timestamps: {timestamps}")
    
    # Get video dimensions
    width, height = get_video_dimensions(args.video_path)
    print(f"📐 Video dimensions: {width}x{height}")
    
    # Extract first frame for Gemini detection
    frame = extract_frame(args.video_path, timestamps[0])
    if frame is None:
        print(f"❌ Could not extract frame at {timestamps[0]}s")
        return 1
    
    # Save original frame
    cv2.imwrite(os.path.join(args.output_dir, 'original_frame.jpg'), frame)
    print(f"📸 Saved original frame")
    
    # Get Gemini bbox (or use manual)
    gemini_bbox = None
    
    if args.gemini_bbox:
        parts = [int(p) for p in args.gemini_bbox.split(',')]
        gemini_bbox = {'x': parts[0], 'y': parts[1], 'width': parts[2], 'height': parts[3]}
        print(f"📦 Using manual bbox: {gemini_bbox}")
    elif HAS_GEMINI:
        # Save frame temporarily for Gemini
        temp_frame_path = os.path.join(args.output_dir, 'temp_frame.jpg')
        cv2.imwrite(temp_frame_path, frame)
        
        print("🤖 Running Gemini detection...")
        result = detect_webcam_with_gemini(temp_frame_path, width, height)
        
        if result and not result.get('no_webcam_confirmed'):
            gemini_bbox = {
                'x': result['x'],
                'y': result['y'],
                'width': result['width'],
                'height': result['height']
            }
            print(f"📦 Gemini bbox: {gemini_bbox}")
            print(f"   Type: {result.get('type')}, Confidence: {result.get('confidence')}")
        else:
            print("❌ Gemini did not detect webcam")
    
    if gemini_bbox is None:
        print("❌ No bbox available. Use --gemini-bbox to provide one manually.")
        return 1
    
    # Detect face
    face_center = None
    face = detect_face_dnn(
        frame,
        gemini_bbox['x'] - 50,
        gemini_bbox['y'] - 50,
        gemini_bbox['width'] + 100,
        gemini_bbox['height'] + 100
    )
    
    if face:
        face_center = (face.center_x, face.center_y)
        print(f"👤 Face detected: center=({face.center_x}, {face.center_y}), size={face.width}x{face.height}")
    else:
        print("⚠️ No face detected")
    
    # Draw Gemini bbox on frame
    frame_with_boxes = frame.copy()
    draw_bbox_on_frame(frame_with_boxes, gemini_bbox, (0, 0, 255), 'Gemini', 2)
    if face:
        cv2.rectangle(frame_with_boxes, 
                     (face.x, face.y), 
                     (face.x + face.width, face.y + face.height),
                     (255, 0, 0), 2)
        cv2.putText(frame_with_boxes, 'Face', (face.x, face.y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Test each refinement method
    print("\n" + "="*60)
    print("TESTING REFINEMENT METHODS")
    print("="*60)
    
    methods_results = {}
    
    # Method 1: Temporal stability
    print("\n📊 Method 1: temporal_stability")
    result = refine_temporal_stability(
        args.video_path, gemini_bbox, width, height,
        timestamps=timestamps, debug=True, debug_dir=args.output_dir
    )
    methods_results['temporal_stability'] = result
    if result:
        draw_bbox_on_frame(frame_with_boxes, result, (0, 255, 0), 'Temporal', 2)
    
    # Method 2: Edge scan
    if face_center:
        print("\n📊 Method 2: edge_scan")
        result = refine_edge_scan_from_face(
            frame, gemini_bbox, face_center, width, height,
            debug=True, debug_dir=args.output_dir
        )
        methods_results['edge_scan'] = result
        if result:
            draw_bbox_on_frame(frame_with_boxes, result, (255, 255, 0), 'EdgeScan', 2)
    
    # Method 3: Tight edges
    print("\n📊 Method 3: tight_edges")
    result = refine_side_box_tight_edges(
        frame, gemini_bbox, face_center, width, height,
        debug=True, debug_dir=args.output_dir
    )
    methods_results['tight_edges'] = result
    if result:
        draw_bbox_on_frame(frame_with_boxes, result, (0, 255, 255), 'TightEdges', 2)
    
    # Method 4: Contours
    print("\n📊 Method 4: contours")
    result = refine_side_box_bbox_contours(
        frame, gemini_bbox, face_center, width, height, debug=True
    )
    methods_results['contours'] = result
    if result:
        draw_bbox_on_frame(frame_with_boxes, result, (255, 0, 255), 'Contours', 2)
    
    # Save comparison image
    cv2.imwrite(os.path.join(args.output_dir, 'all_methods_comparison.jpg'), frame_with_boxes)
    print(f"\n📸 Saved: {os.path.join(args.output_dir, 'all_methods_comparison.jpg')}")
    
    # Full pipeline test
    print("\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)
    
    final_result = refine_side_box_bbox(
        frame, gemini_bbox, face_center, width, height,
        debug=True, debug_dir=args.output_dir, video_path=args.video_path
    )
    
    # Create final comparison
    final_frame = frame.copy()
    draw_bbox_on_frame(final_frame, gemini_bbox, (0, 0, 255), 'Gemini (input)', 3)
    if final_result:
        draw_bbox_on_frame(final_frame, final_result, (0, 255, 0), 'Final (output)', 3)
    if face:
        cv2.circle(final_frame, (face.center_x, face.center_y), 8, (255, 0, 0), -1)
    
    cv2.imwrite(os.path.join(args.output_dir, 'final_result.jpg'), final_frame)
    print(f"\n📸 Saved: {os.path.join(args.output_dir, 'final_result.jpg')}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Gemini bbox: {gemini_bbox['width']}x{gemini_bbox['height']} at ({gemini_bbox['x']},{gemini_bbox['y']})")
    
    gemini_area = gemini_bbox['width'] * gemini_bbox['height']
    
    for name, result in methods_results.items():
        if result:
            area = result['width'] * result['height']
            ratio = area / gemini_area
            status = "✅" if ratio <= 1.35 else "⚠️ EXPANDED"
            print(f"  {name}: {result['width']}x{result['height']} at ({result['x']},{result['y']}) [{ratio:.2f}x] {status}")
        else:
            print(f"  {name}: FAILED")
    
    if final_result:
        area = final_result['width'] * final_result['height']
        ratio = area / gemini_area
        print(f"\n✅ FINAL: {final_result['width']}x{final_result['height']} at ({final_result['x']},{final_result['y']}) [{ratio:.2f}x]")
    else:
        print(f"\n⚠️ FINAL: Using Gemini bbox (no refinement succeeded)")
    
    print(f"\n📁 Debug images saved to: {args.output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
