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
    
    freeze_predictions = []
    
    # Analyze each frame interval
    for i in range(len(timestamps_ms) - 1):
        current_time = timestamps_ms[i]
        next_time = timestamps_ms[i + 1]
        actual_interval = next_time - current_time
        
        # Calculate deviation from expected
        deviation_pct = abs(actual_interval - expected_interval_ms) / expected_interval_ms
        
        # Predict freeze types
        freeze_info = None
        
        if actual_interval < expected_interval_ms * 0.5:
            # Very short interval - will cause duplicate frame
            freeze_info = {
                'frame': i + 1,
                'type': 'SHORT_INTERVAL',
                'actual_interval_ms': actual_interval,
                'expected_interval_ms': expected_interval_ms,
                'deviation_pct': deviation_pct * 100,
                'severity': 'HIGH' if deviation_pct > 0.8 else 'MEDIUM',
                'reason': f'Interval too short: {actual_interval:.1f}ms (expected {expected_interval_ms:.1f}ms)'
            }
        
        elif actual_interval > expected_interval_ms * 2.0:
            # Very long interval - will cause frame skip/stretch
            freeze_info = {
                'frame': i + 1,
                'type': 'LONG_INTERVAL', 
                'actual_interval_ms': actual_interval,
                'expected_interval_ms': expected_interval_ms,
                'deviation_pct': deviation_pct * 100,
                'severity': 'HIGH' if deviation_pct > 3.0 else 'MEDIUM',
                'reason': f'Interval too long: {actual_interval:.1f}ms (expected {expected_interval_ms:.1f}ms)'
            }
        
        elif deviation_pct > 0.3:  # 30% deviation
            # Moderate timing issue
            freeze_info = {
                'frame': i + 1,
                'type': 'TIMING_DEVIATION',
                'actual_interval_ms': actual_interval,
                'expected_interval_ms': expected_interval_ms, 
                'deviation_pct': deviation_pct * 100,
                'severity': 'LOW',
                'reason': f'Timing deviation: {deviation_pct*100:.1f}%'
            }
        
        if freeze_info:
            freeze_predictions.append(freeze_info)
    
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