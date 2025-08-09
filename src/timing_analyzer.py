"""
Timing analysis and duplicate frame detection
"""
import logging
import numpy as np
import cv2

def analyze_timing_changes(timecode_path, fps=25, rife_mode="off", video_path=None):
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
    elif rife_mode == "diagnostic":
        logging.info("Diagnostic mode: using ultra-sensitive detection")
        # Continue with diagnostic detection below
    
    # Analyze frame duplication and timing irregularities
    original_interval = 1000 / fps  # Expected interval in ms
    problem_segments = []
    
    # Different sensitivity thresholds
    if rife_mode == "diagnostic":
        threshold = 0.10  # 10% deviation - balanced for diagnostic
        merge_distance = 3   # Moderate merge distance
    elif rife_mode == "precision":
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
    
    # Add visual duplicate detection if video path provided
    visual_duplicates = []
    if video_path:
        logging.info("🎥 Analyzing video frames for visual duplicates...")
        visual_duplicates = detect_visual_duplicates_from_video(video_path, rife_mode)
        logging.info(f"   • Found {len(visual_duplicates)} visual duplicate frames")
    
    # Focus on segments that need interpolation (duplicates and slow segments)
    interpolation_candidates = duplicate_frames + slow_segments + fast_segments + visual_duplicates
    
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

def detect_visual_duplicates_from_video(video_path, mode="adaptive"):
    """
    Advanced visual duplicate detection using multiple comparison methods.
    Detects frozen/stuck frames even with slight variations.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adaptive thresholds based on mode
    if mode == "diagnostic":
        # Balanced diagnostic mode - sensitive but not excessive
        pixel_threshold = 0.03      # 3% pixel difference max - balanced
        structural_threshold = 0.97 # 97% structural similarity min - reasonable
        check_distance = 3          # Check 3 frames back - focused
    elif mode == "precision":
        pixel_threshold = 0.02      # 2% pixel difference max
        structural_threshold = 0.98 # 98% structural similarity min
        check_distance = 2
    else:  # adaptive
        pixel_threshold = 0.05      # 5% pixel difference max  
        structural_threshold = 0.95 # 95% structural similarity min
        check_distance = 5
    
    logging.info(f"🎥 Advanced freeze detection (mode: {mode})")
    logging.info(f"   Pixel threshold: {pixel_threshold:.1%}, Structural: {structural_threshold:.1%}")
    
    visual_duplicates = []
    
    # Process video frame by frame without loading all into memory
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Keep sliding window of recent frames for comparison
    frame_window = []
    window_size = min(check_distance + 1, 10)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert and resize for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize for faster comparison (maintain aspect ratio)
        height, width = gray.shape
        if width > 480:  # Scale down large frames
            scale = 480.0 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height))
        
        # Add to sliding window
        frame_window.append({
            'index': frame_idx,
            'frame': gray,
            'mean': np.mean(gray),
            'std': np.std(gray)
        })
        
        # Keep only recent frames
        if len(frame_window) > window_size:
            frame_window.pop(0)
        
        # Compare with recent frames
        if frame_idx > 0:
            current_frame_data = frame_window[-1]
            
            for i in range(len(frame_window) - 2, -1, -1):  # Go backwards through window
                ref_frame_data = frame_window[i]
                distance = frame_idx - ref_frame_data['index']
                
                if distance > check_distance:
                    break
                
                # Quick statistical comparison first
                mean_diff = abs(current_frame_data['mean'] - ref_frame_data['mean'])
                std_diff = abs(current_frame_data['std'] - ref_frame_data['std'])
                
                # If stats are very different, skip detailed comparison
                # More lenient thresholds for diagnostic mode
                mean_threshold = 5 if mode == "diagnostic" else 10
                std_threshold = 10 if mode == "diagnostic" else 15
                
                if mean_diff > mean_threshold or std_diff > std_threshold:
                    continue
                
                # Detailed pixel comparison
                current_frame = current_frame_data['frame']
                ref_frame = ref_frame_data['frame']
                
                # Multiple comparison methods
                freeze_detected = False
                
                # Method 1: Simple pixel difference
                diff = cv2.absdiff(current_frame, ref_frame)
                pixel_diff_ratio = np.mean(diff) / 255.0
                
                # For diagnostic mode, also check if there's actual motion (not just noise)
                if mode == "diagnostic":
                    # Only mark as freeze if VERY similar (less than 1% difference)
                    freeze_threshold = 0.01  # 1% difference for true freeze
                else:
                    freeze_threshold = pixel_threshold
                
                if pixel_diff_ratio < freeze_threshold:
                    freeze_detected = True
                    method = "pixel_diff"
                    score = 1.0 - pixel_diff_ratio
                
                # Method 2: Structural similarity (if pixel diff is borderline)
                elif pixel_diff_ratio < pixel_threshold * 2:
                    try:
                        # Calculate SSIM-like measure
                        mean1, mean2 = np.mean(current_frame), np.mean(ref_frame)
                        var1, var2 = np.var(current_frame), np.var(ref_frame)
                        covar = np.mean((current_frame - mean1) * (ref_frame - mean2))
                        
                        c1, c2 = 0.01, 0.03
                        ssim = ((2 * mean1 * mean2 + c1) * (2 * covar + c2)) / \
                               ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
                        
                        if ssim > structural_threshold:
                            freeze_detected = True
                            method = "structural"
                            score = ssim
                            
                    except:
                        pass  # Skip SSIM if calculation fails
                
                if freeze_detected:
                    visual_duplicates.append({
                        'frame': frame_idx,
                        'reference_frame': ref_frame_data['index'],
                        'similarity': score,
                        'pixel_diff': pixel_diff_ratio,
                        'type': 'visual_freeze',
                        'method': method,
                        'distance': distance
                    })
                    
                    # Log detailed info for first few detections in diagnostic mode
                    if mode == "diagnostic" and len(visual_duplicates) <= 5:
                        logging.info(f"      Freeze detected: frame {frame_idx} → {ref_frame_data['index']}, "
                                   f"method={method}, pixel_diff={pixel_diff_ratio:.3f}, score={score:.3f}")
                    
                    break  # Found freeze, stop checking this frame
        
        frame_idx += 1
        
        # Progress update
        if frame_idx % 100 == 0:
            logging.info(f"   Analyzing: {frame_idx}/{total_frames} frames ({len(visual_duplicates)} freezes found)")
    
    cap.release()
    
    # Remove duplicate detections (same frame detected multiple times)
    unique_duplicates = []
    seen_frames = set()
    
    for dup in visual_duplicates:
        if dup['frame'] not in seen_frames:
            unique_duplicates.append(dup)
            seen_frames.add(dup['frame'])
    
    logging.info(f"🎭 Advanced freeze detection complete!")
    logging.info(f"   • Found {len(unique_duplicates)} frozen/duplicate frames")
    logging.info(f"   • Detection rate: {len(unique_duplicates)/total_frames*100:.1f}% of video")
    
    # Show detailed examples
    if unique_duplicates:
        method_counts = {}
        for dup in unique_duplicates:
            method = dup.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        logging.info(f"   • Detection methods: {dict(method_counts)}")
        logging.info("   • Examples found:")
        
        for i, dup in enumerate(unique_duplicates[:3]):  # Show first 3
            logging.info(f"      Frame {dup['frame']} → {dup['reference_frame']} "
                        f"({dup['method']}: {dup['similarity']:.3f}, "
                        f"pixel_diff: {dup.get('pixel_diff', 0):.3f})")
        
        if len(unique_duplicates) > 3:
            logging.info(f"      ... and {len(unique_duplicates) - 3} more frozen frames")
    
    return unique_duplicates

def _create_variation(frame):
    """Create subtle variation of a frame to avoid exact duplication."""
    # Apply very subtle noise
    noise = np.random.randint(-2, 3, frame.shape, dtype=np.int8)
    varied_frame = cv2.add(frame, noise.astype(frame.dtype))
    return varied_frame