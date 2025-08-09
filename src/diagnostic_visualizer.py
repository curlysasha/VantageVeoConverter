"""
Diagnostic visualizer for problem frame detection
"""
import cv2
import numpy as np
import logging
from .timing_analyzer import analyze_timing_changes

def create_diagnostic_video(input_video_path, output_path, timecode_path, rife_mode="adaptive"):
    """
    Create diagnostic video with visual markers on problem frames.
    Helps verify if detection is working correctly.
    """
    logging.info("🔍 Starting diagnostic video creation...")
    
    # Get problem segments
    problem_segments = analyze_timing_changes(timecode_path, rife_mode=rife_mode, video_path=input_video_path)
    
    if not problem_segments:
        logging.warning("No problem segments detected!")
    
    # Create set of problem frames for quick lookup
    problem_frames = set()
    for seg in problem_segments:
        for frame_idx in range(seg['start_frame'], seg['end_frame'] + 1):
            problem_frames.add(frame_idx)
    
    # Also get individual frame issues for detailed info
    frame_issues = {}
    for seg in problem_segments:
        if 'issues' in seg:
            for issue in seg['issues']:
                frame_idx = issue.get('frame')
                if frame_idx:
                    frame_issues[frame_idx] = issue
    
    logging.info(f"📊 Diagnostic info:")
    logging.info(f"   • Total problem segments: {len(problem_segments)}")
    logging.info(f"   • Total problem frames: {len(problem_frames)}")
    logging.info(f"   • Frames with detailed issues: {len(frame_issues)}")
    
    # Open video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    marked_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame is problematic
        if frame_idx in problem_frames:
            # Mark the frame with visual indicators
            marked_frame = mark_problem_frame(frame, frame_idx, frame_issues.get(frame_idx))
            out.write(marked_frame)
            marked_count += 1
        else:
            # Normal frame - write as is
            out.write(frame)
        
        frame_idx += 1
        
        # Progress update
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames) * 100
            logging.info(f"   Progress: {progress:.1f}% ({marked_count} frames marked)")
    
    cap.release()
    out.release()
    
    # Final statistics
    marked_pct = (marked_count / total_frames) * 100 if total_frames > 0 else 0
    
    logging.info(f"✅ Diagnostic video created!")
    logging.info(f"📊 Results:")
    logging.info(f"   • Marked {marked_count} frames ({marked_pct:.1f}%)")
    logging.info(f"   • Total frames: {total_frames}")
    
    # Create detailed report
    report = generate_diagnostic_report(problem_segments, frame_issues, marked_count, total_frames)
    
    return marked_count > 0, report

def mark_problem_frame(frame, frame_idx, issue_info=None):
    """
    Add visual markers to problem frame.
    """
    marked_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Red border for problem frames
    border_thickness = 10
    cv2.rectangle(marked_frame, (0, 0), (width-1, height-1), 
                  (0, 0, 255), border_thickness)
    
    # Add semi-transparent red overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
    marked_frame = cv2.addWeighted(marked_frame, 0.7, overlay, 0.3, 0)
    
    # Add text overlay with frame info
    text_bg_height = 120
    text_bg = np.zeros((text_bg_height, width, 3), dtype=np.uint8)
    text_bg[:] = (0, 0, 0)  # Black background
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 255)  # Red text
    thickness = 2
    
    # Frame number
    text1 = f"PROBLEM FRAME: {frame_idx}"
    cv2.putText(text_bg, text1, (10, 30), font, font_scale, color, thickness)
    
    # Issue type if available
    if issue_info:
        issue_type = issue_info.get('type', 'unknown')
        if issue_type == 'visual_freeze':
            ref_frame = issue_info.get('reference_frame', 'N/A')
            similarity = issue_info.get('similarity', 0)
            text2 = f"FREEZE DETECTED: matches frame {ref_frame}"
            text3 = f"Similarity: {similarity:.1%}"
        elif issue_type == 'visual_duplicate':
            ref_frame = issue_info.get('reference_frame', 'N/A')
            similarity = issue_info.get('similarity', 0)
            text2 = f"DUPLICATE: matches frame {ref_frame}"
            text3 = f"Similarity: {similarity:.1%}"
        elif issue_type == 'duplicate':
            text2 = "DUPLICATE TIMESTAMP"
            text3 = f"Interval: {issue_info.get('interval', 0)}ms"
        elif issue_type == 'slow':
            text2 = "SLOW MOTION DETECTED"
            text3 = f"Speed: {issue_info.get('speed_ratio', 1):.2f}x"
        elif issue_type == 'fast':
            text2 = "FAST MOTION DETECTED"
            text3 = f"Speed: {issue_info.get('speed_ratio', 1):.2f}x"
        else:
            text2 = f"Type: {issue_type}"
            text3 = ""
        
        cv2.putText(text_bg, text2, (10, 60), font, font_scale, (255, 255, 255), thickness)
        if text3:
            cv2.putText(text_bg, text3, (10, 90), font, font_scale, (255, 255, 255), thickness)
    else:
        text2 = "In problem segment"
        cv2.putText(text_bg, text2, (10, 60), font, font_scale, (255, 255, 255), thickness)
    
    # Overlay text background on frame
    marked_frame[0:text_bg_height, :] = cv2.addWeighted(
        marked_frame[0:text_bg_height, :], 0.3, text_bg, 0.7, 0
    )
    
    # Add corner markers
    marker_size = 50
    marker_thickness = 5
    marker_color = (0, 255, 255)  # Yellow
    
    # Top-left corner
    cv2.line(marked_frame, (0, marker_size), (0, 0), marker_color, marker_thickness)
    cv2.line(marked_frame, (0, 0), (marker_size, 0), marker_color, marker_thickness)
    
    # Top-right corner
    cv2.line(marked_frame, (width - marker_size, 0), (width, 0), marker_color, marker_thickness)
    cv2.line(marked_frame, (width, 0), (width, marker_size), marker_color, marker_thickness)
    
    # Bottom-left corner
    cv2.line(marked_frame, (0, height - marker_size), (0, height), marker_color, marker_thickness)
    cv2.line(marked_frame, (0, height), (marker_size, height), marker_color, marker_thickness)
    
    # Bottom-right corner
    cv2.line(marked_frame, (width - marker_size, height), (width, height), marker_color, marker_thickness)
    cv2.line(marked_frame, (width, height), (width, height - marker_size), marker_color, marker_thickness)
    
    return marked_frame

def generate_diagnostic_report(problem_segments, frame_issues, marked_count, total_frames):
    """
    Generate detailed diagnostic report.
    """
    marked_pct = (marked_count / total_frames) * 100 if total_frames > 0 else 0
    
    # Count issue types
    issue_types = {}
    for issue in frame_issues.values():
        issue_type = issue.get('type', 'unknown')
        issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
    
    report = f"""🔍 DIAGNOSTIC REPORT
    
📊 Summary:
• Total frames analyzed: {total_frames}
• Problem frames detected: {marked_count} ({marked_pct:.1f}%)
• Problem segments: {len(problem_segments)}

🎯 Issue Types Found:
"""
    
    for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / marked_count * 100) if marked_count > 0 else 0
        if issue_type == 'visual_freeze':
            report += f"• FREEZES: {count} frames ({percentage:.1f}%)\n"
        elif issue_type == 'visual_duplicate':
            report += f"• VISUAL DUPLICATES: {count} frames ({percentage:.1f}%)\n"
        elif issue_type == 'duplicate':
            report += f"• TIMESTAMP DUPLICATES: {count} frames ({percentage:.1f}%)\n"
        elif issue_type == 'slow':
            report += f"• SLOW MOTION: {count} frames ({percentage:.1f}%)\n"
        elif issue_type == 'fast':
            report += f"• FAST MOTION: {count} frames ({percentage:.1f}%)\n"
        else:
            report += f"• {issue_type.upper()}: {count} frames ({percentage:.1f}%)\n"
    
    report += f"""
📝 Visual Markers:
• RED BORDER = Problem frame detected
• RED OVERLAY = Frame needs interpolation
• YELLOW CORNERS = Visual emphasis
• TEXT OVERLAY = Detailed issue information

🎬 Next Steps:
1. Review the diagnostic video to verify detection accuracy
2. If detection is correct → Run interpolation mode
3. If detection is wrong → Adjust sensitivity settings
"""
    
    return report