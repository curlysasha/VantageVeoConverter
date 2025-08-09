"""
Triple video diagnostic: Original + Freeze Detection + AI Repair
"""
import cv2
import numpy as np
import logging
from .timecode_freeze_predictor import predict_freezes_from_timecodes
from .ai_freeze_repair import repair_freezes_with_rife

def create_triple_diagnostic(synchronized_video_path, timecode_path, output_path, rife_model):
    """
    Создает диагностику из 3 видео:
    Top: Synchronized video (clean)
    Middle: Freeze detection with markers  
    Bottom: AI-repaired video with RIFE
    """
    logging.info("🎬 Creating triple diagnostic: Original + Detection + AI Repair...")
    
    # 1. Predict freezes from timecodes
    freeze_segments = predict_freezes_from_timecodes(timecode_path)
    
    if not freeze_segments:
        logging.info("No freezes predicted - creating simple comparison")
        return create_no_freezes_diagnostic(synchronized_video_path, output_path)
    
    # Create set of predicted freeze frames
    predicted_freeze_frames = set()
    frame_predictions = {}
    
    for seg in freeze_segments:
        for pred in seg['predictions']:
            frame_num = pred['frame']
            predicted_freeze_frames.add(frame_num)
            frame_predictions[frame_num] = pred
    
    logging.info(f"Predicted {len(predicted_freeze_frames)} freeze frames")
    
    # 2. Create AI-repaired video
    import tempfile
    import os
    
    temp_dir = os.path.dirname(output_path)
    repaired_fd, repaired_video_path = tempfile.mkstemp(suffix="_ai_repaired.mp4", dir=temp_dir)
    os.close(repaired_fd)
    
    try:
        repair_success = repair_freezes_with_rife(
            synchronized_video_path, 
            freeze_segments, 
            repaired_video_path, 
            rife_model
        )
        
        if not repair_success:
            logging.error("AI repair failed")
            return False, "AI repair failed"
        
        # 3. Create triple comparison video
        success, report = create_triple_comparison_video(
            synchronized_video_path,      # Top: original synchronized
            predicted_freeze_frames,      # Middle: with freeze markers
            frame_predictions,
            repaired_video_path,          # Bottom: AI repaired
            output_path
        )
        
        return success, report
        
    finally:
        # Cleanup temporary file
        try:
            if os.path.exists(repaired_video_path):
                os.unlink(repaired_video_path)
        except:
            pass

def create_triple_comparison_video(original_path, freeze_frames, frame_predictions, repaired_path, output_path):
    """
    Create 3-panel comparison video.
    """
    logging.info("🎨 Creating triple comparison video...")
    
    # Open all videos
    cap_original = cv2.VideoCapture(original_path)
    cap_repaired = cv2.VideoCapture(repaired_path)
    
    fps = cap_original.get(cv2.CAP_PROP_FPS)
    width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video - triple height
    new_height = height * 3
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, new_height))
    
    frame_idx = 0
    marked_count = 0
    repaired_count = len(freeze_frames)
    
    logging.info(f"Creating triple comparison: {total_frames} frames")
    
    while True:
        ret1, frame_original = cap_original.read()
        ret2, frame_repaired = cap_repaired.read()
        
        if not ret1 or not ret2:
            break
        
        # Top panel: Clean synchronized video
        top_frame = frame_original.copy()
        cv2.putText(top_frame, "SYNCHRONIZED ORIGINAL", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Middle panel: Freeze detection markers
        middle_frame = frame_original.copy()
        
        if frame_idx in freeze_frames:
            pred_info = frame_predictions[frame_idx]
            marked_count += 1
            
            # Color based on severity
            if pred_info['severity'] == 'HIGH':
                color = (0, 0, 255)      # Red
                thickness = 15
            elif pred_info['severity'] == 'MEDIUM':
                color = (0, 128, 255)    # Orange  
                thickness = 12
            else:  # LOW
                color = (0, 255, 255)    # Yellow
                thickness = 8
            
            # Add border
            cv2.rectangle(middle_frame, (0, 0), (width-1, height-1), color, thickness)
            
            # Add prediction text
            cv2.putText(middle_frame, f"FREEZE {frame_idx}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(middle_frame, f"{pred_info['severity']}: {pred_info['type']}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.putText(middle_frame, "FREEZE DETECTION", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Bottom panel: AI repaired video
        bottom_frame = frame_repaired.copy()
        cv2.putText(bottom_frame, "AI REPAIRED (RIFE)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if frame_idx in freeze_frames:
            # Mark repaired frame with green accent
            cv2.rectangle(bottom_frame, (0, 0), (width-1, height-1), (0, 255, 0), 3)
            cv2.putText(bottom_frame, f"REPAIRED {frame_idx}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine all three panels
        triple_frame = np.vstack([top_frame, middle_frame, bottom_frame])
        
        # Add separator lines
        cv2.line(triple_frame, (0, height), (width, height), (255, 255, 255), 2)
        cv2.line(triple_frame, (0, height * 2), (width, height * 2), (255, 255, 255), 2)
        
        out.write(triple_frame)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            progress = (frame_idx / total_frames) * 100
            logging.info(f"   Progress: {progress:.1f}% ({marked_count} freezes marked)")
    
    cap_original.release()
    cap_repaired.release()
    out.release()
    
    # Create detailed report
    marked_pct = (marked_count / total_frames * 100) if total_frames > 0 else 0
    
    report = f"""🎬 TRIPLE DIAGNOSTIC COMPLETE!

📊 Analysis Results:
• Total frames: {total_frames}
• Detected freezes: {marked_count} ({marked_pct:.1f}%)
• AI repaired frames: {repaired_count}

🎨 Triple Video Layout:
┌─────────────────────────┐
│    SYNCHRONIZED         │
│      ORIGINAL           │
├─────────────────────────┤
│   FREEZE DETECTION      │
│   (with markers)        │
├─────────────────────────┤
│    AI REPAIRED          │
│     (RIFE)              │
└─────────────────────────┘

🤖 AI Method: RIFE point interpolation
⚡ Strategy: Replace frozen frames with AI-generated intermediate frames
✨ Compare all three to see the improvement!"""
    
    logging.info(f"✅ Triple diagnostic complete: {marked_count} freezes detected and repaired")
    
    return True, report

def create_no_freezes_diagnostic(video_path, output_path):
    """Create diagnostic when no freezes are detected."""
    logging.info("Creating no-freezes diagnostic")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Simple copy with "NO FREEZES" overlay
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add "NO FREEZES" text
        cv2.putText(frame, "NO FREEZES DETECTED", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(frame, "VIDEO IS ALREADY SMOOTH", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    report = "✅ No freezes detected - video is already smooth!"
    return True, report