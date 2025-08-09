"""
Create side-by-side comparison of synchronized video with and without diagnostic markers
"""
import cv2
import numpy as np
import logging

def create_comparison_diagnostic(synchronized_video_path, output_path):
    """
    Create top-bottom comparison:
    Top: original synchronized video
    Bottom: same video with red markers on frozen frames
    """
    logging.info("🔍 Creating top-bottom diagnostic comparison...")
    
    # Open video
    cap = cv2.VideoCapture(synchronized_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video - twice as tall for top-bottom
    new_height = height * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, new_height))
    
    prev_frame = None
    frame_idx = 0
    freeze_count = 0
    freeze_frames = []
    
    logging.info(f"Processing {total_frames} frames for comparison...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame is frozen
        is_freeze = False
        if prev_frame is not None:
            # Compare with previous frame
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray1, gray2)
            mean_diff = np.mean(diff)
            
            if mean_diff < 2.0:  # Very strict threshold for freezes
                is_freeze = True
                freeze_count += 1
                freeze_frames.append(frame_idx)
        
        # Create top frame (original)
        top_frame = frame.copy()
        
        # Add label to top
        cv2.putText(top_frame, "SYNCHRONIZED", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Create bottom frame (with markers if frozen)
        bottom_frame = frame.copy()
        
        if is_freeze:
            # Add thick red border
            cv2.rectangle(bottom_frame, (0, 0), (width-1, height-1), (0, 0, 255), 15)
            
            # Add red overlay
            overlay = bottom_frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
            bottom_frame = cv2.addWeighted(bottom_frame, 0.7, overlay, 0.3, 0)
            
            # Add freeze text
            cv2.putText(bottom_frame, f"FREEZE {frame_idx}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        # Add label to bottom
        cv2.putText(bottom_frame, "WITH MARKERS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combine top-bottom
        comparison_frame = np.vstack([top_frame, bottom_frame])
        
        # Add separator line between top and bottom
        cv2.line(comparison_frame, (0, height), (width, height), (255, 255, 255), 2)
        
        out.write(comparison_frame)
        prev_frame = frame.copy()
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            progress = (frame_idx / total_frames) * 100
            logging.info(f"   Progress: {progress:.1f}% ({freeze_count} freezes found)")
    
    cap.release()
    out.release()
    
    freeze_pct = (freeze_count / total_frames * 100) if total_frames > 0 else 0
    
    # Show some example freeze frames
    example_frames = freeze_frames[:10] if len(freeze_frames) > 10 else freeze_frames
    examples_str = ", ".join(map(str, example_frames))
    if len(freeze_frames) > 10:
        examples_str += f" ... (and {len(freeze_frames) - 10} more)"

    report = f"""🔍 TOP-BOTTOM DIAGNOSTIC COMPARISON

📊 Analysis Results:
• Total frames: {total_frames}
• Frozen frames: {freeze_count} ({freeze_pct:.1f}%)
• Example frozen frame numbers: {examples_str}

🎬 Video Layout:
┌─────────────┐
│ SYNCHRONIZED│ <- Top: Clean synchronized video
│   (clean)   │
├─────────────┤ <- White separator line
│ WITH MARKERS│ <- Bottom: Same video + red markers
│ (red borders│    on frozen frames
│  on freezes)│
└─────────────┘

🔴 Red frames on bottom = detected freezes
⚪ White line = separator between original and diagnostic

This shows exactly where timing corrections created frozen frames!
"""
    
    logging.info(f"✅ Side-by-side diagnostic created: {freeze_count} freezes marked ({freeze_pct:.1f}%)")
    
    return freeze_count > 0, report