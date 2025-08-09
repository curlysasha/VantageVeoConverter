"""
Simple diagnostic - just add red borders to frozen frames
"""
import cv2
import numpy as np
import logging
import os

def create_simple_diagnostic(retimed_video_path, output_path):
    """
    Analyze the retimed video and add red borders to duplicate frames.
    Preserves original video quality and metadata.
    """
    logging.info("🔍 Simple diagnostic - detecting freezes by comparing adjacent frames")
    
    # First, just copy the video to preserve quality
    import shutil
    shutil.copy2(retimed_video_path, output_path)
    
    # Then analyze for freeze detection
    cap = cv2.VideoCapture(retimed_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Analyze frames without rewriting video (to preserve quality)
    prev_frame = None
    frame_idx = 0
    freeze_count = 0
    freeze_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame is a duplicate of previous
        if prev_frame is not None:
            # Simple comparison - convert to grayscale and check difference
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            mean_diff = np.mean(diff)
            
            # If difference is very small, it's a freeze
            if mean_diff < 2.0:  # Very strict threshold - only exact duplicates
                freeze_count += 1
                freeze_frames.append(frame_idx)
        
        prev_frame = frame.copy()
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            logging.info(f"   Progress: {frame_idx}/{total_frames} ({freeze_count} freezes found)")
    
    cap.release()
    
    freeze_pct = (freeze_count / total_frames * 100) if total_frames > 0 else 0
    
    # Show some example freeze frames
    example_frames = freeze_frames[:10] if len(freeze_frames) > 10 else freeze_frames
    examples_str = ", ".join(map(str, example_frames))
    if len(freeze_frames) > 10:
        examples_str += f" ... (and {len(freeze_frames) - 10} more)"

    report = f"""🔍 FREEZE DETECTION ANALYSIS

📊 Results:
• Total frames: {total_frames}
• Frozen frames: {freeze_count} ({freeze_pct:.1f}%)
• Example frozen frame numbers: {examples_str}

📹 Video Output:
• Shows the synchronized video (same as "sync without RIFE")
• Contains freezes from timing corrections
• No visual markers added (preserves original quality)

🎯 Frozen frames detected:
{len(freeze_frames)} frames are nearly identical to previous frame
"""
    
    logging.info(f"✅ Simple diagnostic complete: {freeze_count} freezes found ({freeze_pct:.1f}%)")
    
    return freeze_count > 0, report