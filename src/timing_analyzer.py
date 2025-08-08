"""
Timing analysis and duplicate frame detection
"""
import logging
import numpy as np
import cv2

def analyze_timing_changes(timecode_path, fps=25, rife_mode="off"):
    """Analyze timing changes and detect duplicate/missing frames based on mode."""
    with open(timecode_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if not line.startswith('#')]
    
    timestamps = [int(line) for line in lines if line.isdigit()]
    if len(timestamps) < 10:
        return []
    
    if rife_mode == "off":
        return []
    elif rife_mode == "maximum":
        logging.info("Maximum mode: will interpolate entire video")
        return [{'start_frame': 0, 'end_frame': len(timestamps), 'reason': 'maximum', 'type': 'full_video'}]
    
    # Analyze frame duplication and timing irregularities
    original_interval = 1000 / fps  # Expected interval in ms
    problem_segments = []
    
    # Different sensitivity thresholds
    if rife_mode == "precision":
        threshold = 0.05  # 5% deviation - very sensitive
        merge_distance = 2  # Smaller merge distance for precision
    else:  # adaptive
        threshold = 0.15  # 15% deviation - moderate sensitivity
        merge_distance = 5  # Larger merge distance for adaptive
    
    # Detect different types of timing issues
    duplicate_frames = []  # Frames with identical or too-close timestamps
    fast_segments = []     # Segments that are too fast (frame drops)
    slow_segments = []     # Segments that are too slow (frame duplication)
    
    logging.info(f"🔍 Analyzing frame timing: expected_interval={original_interval:.1f}ms")
    
    for i in range(1, len(timestamps)):
        actual_interval = timestamps[i] - timestamps[i-1]
        
        if actual_interval <= 1:
            # Duplicate or nearly duplicate frame timestamps
            duplicate_frames.append({
                'frame': i,
                'prev_frame': i-1,
                'interval': actual_interval,
                'type': 'duplicate'
            })
        elif actual_interval > 0:
            speed_ratio = original_interval / actual_interval
            deviation = abs(speed_ratio - 1.0)
            
            if deviation > threshold:
                frame_data = {
                    'frame': i,
                    'deviation': deviation,
                    'actual_interval': actual_interval,
                    'expected_interval': original_interval,
                    'speed_ratio': speed_ratio
                }
                
                if speed_ratio > 1.0:
                    # Video is too fast here (frames were dropped)
                    frame_data['type'] = 'fast'
                    fast_segments.append(frame_data)
                else:
                    # Video is too slow here (frames were duplicated)
                    frame_data['type'] = 'slow' 
                    slow_segments.append(frame_data)
    
    # Log analysis results
    logging.info(f"📊 Frame analysis results:")
    logging.info(f"   • Duplicate frames: {len(duplicate_frames)}")
    logging.info(f"   • Fast segments (dropped frames): {len(fast_segments)}")
    logging.info(f"   • Slow segments (duplicated frames): {len(slow_segments)}")
    
    # Focus on segments that need interpolation (duplicates and slow segments)
    interpolation_candidates = duplicate_frames + slow_segments + fast_segments
    
    if not interpolation_candidates:
        logging.info(f"{rife_mode.title()} mode: no timing issues detected above {threshold:.1%} threshold")
        return []
    
    # Group nearby problem frames into segments for interpolation
    segments = []
    current_segment = None
    
    # Sort by frame number
    interpolation_candidates.sort(key=lambda x: x['frame'])
    
    for frame_data in interpolation_candidates:
        frame = frame_data['frame']
        
        if current_segment is None:
            current_segment = {
                'start_frame': max(0, frame - merge_distance),
                'end_frame': min(len(timestamps), frame + merge_distance),
                'issues': [frame_data],
                'primary_type': frame_data['type']
            }
        elif frame <= current_segment['end_frame'] + merge_distance:
            # Extend current segment
            current_segment['end_frame'] = min(len(timestamps), max(current_segment['end_frame'], frame + merge_distance))
            current_segment['issues'].append(frame_data)
        else:
            # Finalize current segment
            segments.append(current_segment)
            current_segment = {
                'start_frame': max(0, frame - merge_distance),
                'end_frame': min(len(timestamps), frame + merge_distance),
                'issues': [frame_data],
                'primary_type': frame_data['type']
            }
    
    # Add last segment
    if current_segment:
        segments.append(current_segment)
    
    # Log detailed segment information
    total_affected_frames = sum(seg['end_frame'] - seg['start_frame'] for seg in segments)
    coverage_pct = (total_affected_frames / len(timestamps)) * 100
    
    logging.info(f"🎯 {rife_mode.title()} interpolation plan:")
    logging.info(f"   • {len(segments)} segments need interpolation")
    logging.info(f"   • Coverage: {coverage_pct:.1f}% of video ({total_affected_frames}/{len(timestamps)} frames)")
    
    for i, seg in enumerate(segments):
        seg_size = seg['end_frame'] - seg['start_frame']
        issue_types = [issue['type'] for issue in seg['issues']]
        type_summary = ', '.join(set(issue_types))
        logging.info(f"   Segment {i+1}: frames {seg['start_frame']}-{seg['end_frame']} ({seg_size} frames)")
        logging.info(f"      → Issues: {type_summary} ({len(seg['issues'])} problem frames)")
    
    return segments

def detect_duplicate_frames(frames, similarity_threshold=0.01):
    """
    Detect duplicate frames by comparing consecutive frames.
    Returns a dictionary mapping frame indices to duplicate info.
    """
    duplicate_map = {}
    
    for i in range(1, len(frames)):
        # Convert frames to grayscale for comparison
        frame1_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        # Calculate difference
        diff = cv2.absdiff(frame1_gray, frame2_gray)
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        if similarity > (1.0 - similarity_threshold):
            duplicate_map[i] = {
                'is_duplicate': True,
                'similarity': similarity,
                'reference_frame': i-1
            }
        
        if i % 100 == 0:
            logging.info(f"Duplicate detection: {i}/{len(frames)} frames processed")
    
    return duplicate_map

def _create_variation(frame):
    """Create subtle variation of a frame to avoid exact duplication."""
    # Apply very subtle noise
    noise = np.random.randint(-2, 3, frame.shape, dtype=np.int8)
    varied_frame = cv2.add(frame, noise.astype(frame.dtype))
    return varied_frame