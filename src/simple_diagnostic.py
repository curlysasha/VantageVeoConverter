"""
Simple diagnostic - just add red borders to frozen frames
"""
import cv2
import numpy as np
import logging
import os

def create_simple_diagnostic(retimed_video_path, output_path):
    """
    Simply copy the retimed video and add red borders to duplicate frames.
    Super simple approach - no complex analysis.
    """
    logging.info("🔍 Simple diagnostic - detecting freezes by comparing adjacent frames")
    
    # Open videos
    cap = cv2.VideoCapture(retimed_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    prev_frame = None
    frame_idx = 0
    freeze_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame is a duplicate of previous
        is_freeze = False
        if prev_frame is not None:
            # Simple comparison - convert to grayscale and check difference
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            mean_diff = np.mean(diff)
            
            # If difference is very small, it's a freeze
            if mean_diff < 2.0:  # Very strict threshold - only exact duplicates
                is_freeze = True
                freeze_count += 1
        
        # Mark frame if it's a freeze
        if is_freeze:
            # Red border
            cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 10)
            
            # Add text
            cv2.putText(frame, f"FREEZE {frame_idx}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        out.write(frame)
        prev_frame = frame.copy()
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            logging.info(f"   Progress: {frame_idx}/{total_frames} ({freeze_count} freezes found)")
    
    cap.release()
    out.release()
    
    freeze_pct = (freeze_count / total_frames * 100) if total_frames > 0 else 0
    
    report = f"""🔍 SIMPLE DIAGNOSTIC REPORT

📊 Results:
• Total frames: {total_frames}
• Frozen frames: {freeze_count} ({freeze_pct:.1f}%)

Red frames = exact duplicates of previous frame (freezes)
"""
    
    logging.info(f"✅ Simple diagnostic complete: {freeze_count} freezes found ({freeze_pct:.1f}%)")
    
    return freeze_count > 0, report