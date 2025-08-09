"""
Predictive diagnostic using timecode analysis instead of frame comparison
"""
import cv2
import numpy as np
import logging
from .timecode_freeze_predictor import predict_freezes_from_timecodes, create_prediction_report

def create_predictive_diagnostic(synchronized_video_path, timecode_path, output_path):
    """
    Create diagnostic video using TIMECODE PREDICTION instead of frame analysis.
    Shows exactly where timecode analysis predicts freezes will occur.
    """
    logging.info("🔮 Creating predictive diagnostic from timecode analysis...")
    
    # Predict freezes from timecodes FIRST
    freeze_segments = predict_freezes_from_timecodes(timecode_path)
    
    if not freeze_segments:
        logging.info("No freezes predicted - creating clean comparison")
    
    # Create set of predicted freeze frames
    predicted_freeze_frames = set()
    frame_predictions = {}  # frame -> prediction info
    
    for seg in freeze_segments:
        for pred in seg['predictions']:
            frame_num = pred['frame']
            predicted_freeze_frames.add(frame_num)
            frame_predictions[frame_num] = pred
    
    logging.info(f"Predicted {len(predicted_freeze_frames)} freeze frames")
    
    # Open video
    cap = cv2.VideoCapture(synchronized_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video - top-bottom comparison
    new_height = height * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, new_height))
    
    frame_idx = 0
    marked_count = 0
    
    logging.info(f"Creating comparison video: {total_frames} frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create top frame (clean)
        top_frame = frame.copy()
        cv2.putText(top_frame, "SYNCHRONIZED", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Create bottom frame (with prediction markers)
        bottom_frame = frame.copy()
        
        # Check if this frame is predicted to freeze
        if frame_idx in predicted_freeze_frames:
            pred_info = frame_predictions[frame_idx]
            marked_count += 1
            
            # Color based on severity
            if pred_info['severity'] == 'HIGH':
                color = (0, 0, 255)      # Red
                thickness = 20
            elif pred_info['severity'] == 'MEDIUM':
                color = (0, 128, 255)    # Orange
                thickness = 15
            else:  # LOW
                color = (0, 255, 255)    # Yellow
                thickness = 10
            
            # Add border
            cv2.rectangle(bottom_frame, (0, 0), (width-1, height-1), color, thickness)
            
            # Add overlay
            overlay = bottom_frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), color, -1)
            bottom_frame = cv2.addWeighted(bottom_frame, 0.75, overlay, 0.25, 0)
            
            # Add prediction text
            cv2.putText(bottom_frame, f"PREDICTED FREEZE {frame_idx}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add severity and reason
            cv2.putText(bottom_frame, f"{pred_info['severity']}: {pred_info['type']}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add deviation info
            deviation_text = f"Dev: {pred_info['deviation_pct']:.0f}%"
            cv2.putText(bottom_frame, deviation_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add label to bottom
        cv2.putText(bottom_frame, "PREDICTED FREEZES", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combine top-bottom
        comparison_frame = np.vstack([top_frame, bottom_frame])
        
        # Add separator line
        cv2.line(comparison_frame, (0, height), (width, height), (255, 255, 255), 2)
        
        out.write(comparison_frame)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            progress = (frame_idx / total_frames) * 100
            logging.info(f"   Progress: {progress:.1f}% ({marked_count} predictions marked)")
    
    cap.release()
    out.release()
    
    marked_pct = (marked_count / total_frames * 100) if total_frames > 0 else 0
    
    # Create detailed report
    report = create_prediction_report(freeze_segments)
    
    additional_info = f"""

📊 Visualization Results:
• Total frames: {total_frames}
• Marked frames: {marked_count} ({marked_pct:.1f}%)

🎨 Color Coding:
🔴 Red borders = HIGH severity (>80% timing deviation)
🟠 Orange borders = MEDIUM severity (50-80% deviation)  
🟡 Yellow borders = LOW severity (30-50% deviation)

⚡ This prediction was made BEFORE analyzing actual video frames!
Compare with actual freeze detection to validate accuracy."""
    
    full_report = report + additional_info
    
    logging.info(f"✅ Predictive diagnostic complete: {marked_count} frames marked ({marked_pct:.1f}%)")
    
    return marked_count > 0, full_report