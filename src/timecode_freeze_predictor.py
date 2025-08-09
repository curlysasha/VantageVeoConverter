"""
Predict freeze frames directly from timecode analysis
"""
import logging
import numpy as np

def predict_freezes_from_timecodes(timecode_path, fps=24.0):
    """
    Предсказать фризы анализируя ТОЛЬКО timecode файл.
    Возвращает точные номера кадров где будут фризы.
    """
    logging.info("🔮 Predicting freezes from timecode analysis...")
    
    # Read timecodes
    with open(timecode_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if not line.startswith('#')]
    
    timestamps_ms = [int(line) for line in lines if line.isdigit()]
    
    if len(timestamps_ms) < 2:
        logging.warning("Not enough timecodes for analysis")
        return []
    
    logging.info(f"Analyzing {len(timestamps_ms)} timestamps...")
    
    # Calculate expected frame interval
    expected_interval_ms = 1000.0 / fps
    logging.info(f"Expected interval: {expected_interval_ms:.1f}ms per frame")
    
    # Simulate physical frame duplication to find REAL duplicates
    freeze_predictions = []
    target_frame_duration_ms = expected_interval_ms
    
    # Simulate which input frames will be used for each output moment
    frame_usage = {}  # output_frame_idx -> input_frame_idx
    output_frame_idx = 0
    
    for output_time_ms in np.arange(timestamps_ms[0], timestamps_ms[-1], target_frame_duration_ms):
        # Find which input frame this output time corresponds to
        input_frame_idx = 0
        
        # Find the timecode bracket this output time falls into
        for i in range(len(timestamps_ms) - 1):
            if timestamps_ms[i] <= output_time_ms < timestamps_ms[i + 1]:
                input_frame_idx = i
                break
        
        # Ensure frame index is valid
        if input_frame_idx >= len(timestamps_ms) - 1:
            input_frame_idx = len(timestamps_ms) - 2
            
        frame_usage[output_frame_idx] = input_frame_idx
        output_frame_idx += 1
    
    logging.info(f"Simulated {len(frame_usage)} output frames")
    
    # Now detect actual duplicates by checking frame_usage
    prev_input_frame = None
    duplicate_count = 0
    
    for out_idx in sorted(frame_usage.keys()):
        input_frame = frame_usage[out_idx]
        
        if prev_input_frame is not None and input_frame == prev_input_frame:
            # This is a REAL duplicate frame!
            duplicate_count += 1
            
            # Calculate timing info for this frame
            actual_output_time = timestamps_ms[0] + (out_idx * target_frame_duration_ms)
            
            # Find the interval that caused this duplicate
            interval_idx = input_frame
            if interval_idx < len(timestamps_ms) - 1:
                actual_interval = timestamps_ms[interval_idx + 1] - timestamps_ms[interval_idx]
                deviation_pct = abs(actual_interval - expected_interval_ms) / expected_interval_ms
                
                freeze_info = {
                    'frame': out_idx,
                    'type': 'FREEZE_DUPLICATE',
                    'actual_interval_ms': actual_interval,
                    'expected_interval_ms': expected_interval_ms,
                    'deviation_pct': deviation_pct * 100,
                    'severity': 'HIGH' if deviation_pct > 1.0 else 'MEDIUM',
                    'reason': f'Duplicate frame (input {input_frame}): {actual_interval:.1f}ms interval',
                    'input_frame_used': input_frame
                }
                
                freeze_predictions.append(freeze_info)
        
        prev_input_frame = input_frame
    
    logging.info(f"Found {duplicate_count} actual duplicate frames")
    
    # Group consecutive freeze predictions
    grouped_freezes = []
    if freeze_predictions:
        current_group = [freeze_predictions[0]]
        
        for freeze in freeze_predictions[1:]:
            # If consecutive frames, add to current group
            if freeze['frame'] - current_group[-1]['frame'] <= 1:
                current_group.append(freeze)
            else:
                # Start new group
                grouped_freezes.append(current_group)
                current_group = [freeze]
        
        # Add last group
        grouped_freezes.append(current_group)
    
    # Convert groups to freeze segments
    freeze_segments = []
    for group in grouped_freezes:
        start_frame = group[0]['frame']
        end_frame = group[-1]['frame']
        
        # Calculate average severity
        severities = [f['severity'] for f in group]
        severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for s in severities:
            severity_counts[s] += 1
        
        avg_severity = max(severity_counts.keys(), key=lambda k: severity_counts[k])
        
        freeze_segments.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frame_count': end_frame - start_frame + 1,
            'severity': avg_severity,
            'predictions': group,
            'summary': f"{len(group)} timing issues in frames {start_frame}-{end_frame}"
        })
    
    # Log results
    total_affected_frames = sum(len(seg['predictions']) for seg in freeze_segments)
    logging.info(f"🔮 Freeze prediction complete!")
    logging.info(f"   • Predicted freeze segments: {len(freeze_segments)}")
    logging.info(f"   • Affected frames: {total_affected_frames}")
    
    # Show detailed predictions
    for i, seg in enumerate(freeze_segments):
        logging.info(f"   Segment {i+1}: frames {seg['start_frame']}-{seg['end_frame']} "
                    f"({seg['severity']} severity, {seg['frame_count']} frames)")
        
        # Show first few examples
        for j, pred in enumerate(seg['predictions'][:3]):
            logging.info(f"      Frame {pred['frame']}: {pred['reason']}")
        
        if len(seg['predictions']) > 3:
            logging.info(f"      ... and {len(seg['predictions']) - 3} more")
    
    return freeze_segments

def create_prediction_report(freeze_segments):
    """Create detailed report of freeze predictions."""
    if not freeze_segments:
        return "🔮 No timing issues predicted from timecode analysis!"
    
    total_segments = len(freeze_segments)
    total_frames = sum(seg['frame_count'] for seg in freeze_segments)
    
    # Count by severity
    severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for seg in freeze_segments:
        severity_counts[seg['severity']] += seg['frame_count']
    
    report = f"""🔮 TIMECODE FREEZE PREDICTION

📊 Predicted Results:
• Freeze segments: {total_segments}
• Affected frames: {total_frames}
• High severity: {severity_counts['HIGH']} frames
• Medium severity: {severity_counts['MEDIUM']} frames  
• Low severity: {severity_counts['LOW']} frames

🎯 This prediction is based on timecode analysis BEFORE creating video.
Red frames in diagnostic = predicted freeze locations!

📋 Segment Details:"""
    
    for i, seg in enumerate(freeze_segments[:5]):  # Show first 5
        report += f"\n• Segment {i+1}: frames {seg['start_frame']}-{seg['end_frame']} ({seg['severity']})"
    
    if len(freeze_segments) > 5:
        report += f"\n• ... and {len(freeze_segments) - 5} more segments"
    
    return report