"""
Create physically retimed video with actual frame duplicates
"""
import cv2
import numpy as np
import logging

def create_physical_retime(input_video_path, timecode_path, output_path):
    """
    Create video with ACTUAL frame duplicates based on timecodes.
    This creates a CFR video where frames are physically duplicated.
    """
    logging.info("🎬 Creating physically retimed video with frame duplicates...")
    
    # Read timecodes
    with open(timecode_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if not line.startswith('#')]
    
    timestamps_ms = [int(line) for line in lines if line.isdigit()]
    
    if len(timestamps_ms) < 2:
        logging.error("Not enough timecodes!")
        return False
    
    logging.info(f"Loaded {len(timestamps_ms)} timecodes")
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Input video: {total_input_frames} frames at {input_fps} FPS")
    
    # Load all frames
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    
    # Calculate target FPS (use input FPS)
    target_fps = input_fps
    target_frame_duration_ms = 1000.0 / target_fps
    
    logging.info(f"Target: {target_fps} FPS, {target_frame_duration_ms:.1f}ms per frame")
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    # Generate output frames based on timecodes
    output_frame_count = 0
    duplicate_count = 0
    
    # Calculate total duration
    total_duration_ms = timestamps_ms[-1] - timestamps_ms[0]
    expected_output_frames = int(total_duration_ms / target_frame_duration_ms)
    
    logging.info(f"Expected output frames: ~{expected_output_frames}")
    
    for output_time_ms in np.arange(timestamps_ms[0], timestamps_ms[-1], target_frame_duration_ms):
        # Find which input frame this output time corresponds to
        input_frame_idx = 0
        
        # Find the timecode bracket this output time falls into
        for i in range(len(timestamps_ms) - 1):
            if timestamps_ms[i] <= output_time_ms < timestamps_ms[i + 1]:
                input_frame_idx = i
                break
        
        # Ensure frame index is valid
        if input_frame_idx >= len(all_frames):
            input_frame_idx = len(all_frames) - 1
        
        # Check if this is a duplicate
        is_duplicate = False
        if output_frame_count > 0:
            # Check if we're using the same frame as the previous output
            prev_output_time = output_time_ms - target_frame_duration_ms
            prev_input_frame_idx = 0
            
            for i in range(len(timestamps_ms) - 1):
                if timestamps_ms[i] <= prev_output_time < timestamps_ms[i + 1]:
                    prev_input_frame_idx = i
                    break
                    
            if prev_input_frame_idx >= len(all_frames):
                prev_input_frame_idx = len(all_frames) - 1
                
            if input_frame_idx == prev_input_frame_idx:
                is_duplicate = True
                duplicate_count += 1
        
        # Write the frame
        out.write(all_frames[input_frame_idx])
        output_frame_count += 1
        
        if output_frame_count % 100 == 0:
            logging.info(f"   Progress: {output_frame_count} frames ({duplicate_count} duplicates)")
    
    out.release()
    
    duplicate_pct = (duplicate_count / output_frame_count * 100) if output_frame_count > 0 else 0
    
    logging.info(f"✅ Physical retime complete!")
    logging.info(f"   Output frames: {output_frame_count}")
    logging.info(f"   Duplicated frames: {duplicate_count} ({duplicate_pct:.1f}%)")
    
    return True