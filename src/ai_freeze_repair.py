"""
AI-powered freeze repair using RIFE interpolation
"""
import cv2
import numpy as np
import logging
import torch
from .timecode_freeze_predictor import predict_freezes_from_timecodes
from .real_rife import RealRIFE

def repair_freezes_with_rife(video_path, freeze_predictions, output_path, rife_model):
    """
    Точечное исправление замороженных кадров с помощью RIFE.
    Заменяет только дубликаты на AI-интерполированные кадры.
    """
    logging.info("🤖 Starting AI freeze repair with RIFE...")
    
    if not freeze_predictions:
        logging.info("No freezes to repair - copying original video")
        import shutil
        shutil.copy2(video_path, output_path)
        return True
    
    # Initialize REAL RIFE once for all frames
    real_rife = RealRIFE()
    logging.info(f"REAL RIFE available: {real_rife.available}")
    
    # Create set of frames that need repair
    frames_to_repair = set()
    for segment in freeze_predictions:
        for pred in segment['predictions']:
            frames_to_repair.add(pred['frame'])
    
    logging.info(f"Will repair {len(frames_to_repair)} frozen frames")
    logging.info(f"Frames to repair: {sorted(list(frames_to_repair))[:10]}...")  # Show first 10
    
    # Group consecutive frozen frames into sequences
    freeze_sequences = group_consecutive_frames(frames_to_repair)
    logging.info(f"Detected {len(freeze_sequences)} freeze sequences")
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load all frames
    all_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        frame_idx += 1
    cap.release()
    
    logging.info(f"📊 FRAME COUNT ANALYSIS:")
    logging.info(f"   Input video: {video_path}")
    logging.info(f"   Total frames loaded: {len(all_frames)}")
    logging.info(f"   Original total_frames: {total_frames}")
    logging.info(f"   FPS: {fps}")
    logging.info(f"   Video duration: {len(all_frames)/fps:.3f} seconds")
    
    # Create output video with better codec and explicit flush
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        logging.error("❌ VideoWriter failed to open!")
        raise Exception("VideoWriter initialization failed")
    
    logging.info(f"✅ VideoWriter opened successfully:")
    
    repaired_count = 0
    written_frames = 0
    
    logging.info(f"🔧 Starting frame processing loop for {len(all_frames)} frames...")
    
    # Process each frame
    for frame_idx in range(len(all_frames)):
        current_frame = all_frames[frame_idx]
        
        if frame_idx in frames_to_repair:
            # Find which sequence this frame belongs to
            sequence_info = find_frame_in_sequences(frame_idx, freeze_sequences)
            
            if sequence_info:
                seq_start, seq_end, position = sequence_info
                prev_frame, next_frame = find_neighbor_frames(all_frames, seq_start, frames_to_repair, seq_end)
                
                if prev_frame is not None and next_frame is not None:
                    # Calculate proper timestep for this frame within sequence
                    sequence_length = seq_end - seq_start + 1
                    timestep = (position + 1) / (sequence_length + 1)
                    
                    logging.info(f"   Repairing frame {frame_idx} (seq {seq_start}-{seq_end}, pos {position}, timestep {timestep:.3f})")
                    
                    try:
                        interpolated = interpolate_with_timestep_rife(prev_frame, next_frame, timestep, real_rife)
                        if interpolated is not None:
                            current_frame = interpolated
                            repaired_count += 1
                            logging.info(f"   ✅ Successfully repaired frame {frame_idx} using RIFE at timestep {timestep:.3f}")
                        else:
                            logging.warning(f"   ❌ RIFE returned None for frame {frame_idx}, using original")
                    except Exception as e:
                        logging.error(f"   ❌ RIFE error for frame {frame_idx}: {e}")
                else:
                    logging.warning(f"   ❌ Cannot find neighbors for sequence {seq_start}-{seq_end} - skipping repair")
            else:
                logging.warning(f"   ❌ Frame {frame_idx} not found in sequences - skipping repair")
        
        # Write frame with verification
        success = out.write(current_frame)
        if not success:
            logging.warning(f"⚠️ Failed to write frame {frame_idx}!")
        written_frames += 1
        
        if frame_idx % 50 == 0:
            logging.info(f"   Progress: {frame_idx+1}/{len(all_frames)} frames processed, {written_frames} written, {repaired_count} repaired")
    
    # Force flush and release
    out.release()
    del out  # Explicit cleanup
    
    logging.info(f"🔧 Frame processing loop completed:")
    logging.info(f"   Total frames processed: {len(all_frames)}")
    logging.info(f"   Total frames written: {written_frames}")
    logging.info(f"   Should match: {len(all_frames) == written_frames}")
    
    # Wait a moment for file system to flush
    import time
    time.sleep(0.1)
    
    # Check final frame counts
    cap_check = cv2.VideoCapture(output_path)
    if not cap_check.isOpened():
        logging.error(f"❌ Cannot reopen output video: {output_path}")
        return False
        
    output_frame_count = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = cap_check.get(cv2.CAP_PROP_FPS)
    cap_check.release()
    
    # If frame count mismatch, try to fix with FFmpeg remux
    if abs(len(all_frames) - output_frame_count) > 0:
        logging.warning(f"⚠️ Frame count mismatch detected, attempting FFmpeg fix...")
        temp_path = output_path.replace('.mp4', '_temp_remux.mp4')
        
        try:
            import subprocess
            # Remux without re-encoding to preserve all frames
            cmd = ['ffmpeg', '-y', '-i', output_path, '-c', 'copy', '-avoid_negative_ts', 'make_zero', temp_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                import shutil
                shutil.move(temp_path, output_path)
                logging.info("✅ FFmpeg remux completed")
                
                # Recheck frame count
                cap_check = cv2.VideoCapture(output_path)
                output_frame_count = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
                output_fps = cap_check.get(cv2.CAP_PROP_FPS)
                cap_check.release()
            else:
                logging.warning(f"⚠️ FFmpeg remux failed: {result.stderr}")
        except Exception as e:
            logging.warning(f"⚠️ FFmpeg fix failed: {e}")
    
    repair_pct = (repaired_count / len(frames_to_repair) * 100) if frames_to_repair else 0
    
    logging.info(f"📊 FINAL FRAME COUNT COMPARISON:")
    logging.info(f"   INPUT video frames: {len(all_frames)}")
    logging.info(f"   WRITTEN frames: {written_frames}")  
    logging.info(f"   OUTPUT video frames (OpenCV): {output_frame_count}")
    logging.info(f"   OUTPUT FPS: {output_fps}")
    logging.info(f"   INPUT duration: {len(all_frames)/fps:.3f}s")
    logging.info(f"   OUTPUT duration: {output_frame_count/output_fps:.3f}s")
    logging.info(f"   Frame loss: {len(all_frames) - output_frame_count}")
    logging.info(f"   Time difference: {(len(all_frames)/fps) - (output_frame_count/output_fps):.3f}s")
    logging.info(f"✅ AI repair complete! Repaired: {repaired_count}/{len(frames_to_repair)} frames ({repair_pct:.1f}%)")
    
    return True

def group_consecutive_frames(frame_set):
    """
    Группировать последовательные замороженные кадры в последовательности.
    Возвращает список кортежей (start_frame, end_frame).
    """
    if not frame_set:
        return []
    
    frames = sorted(list(frame_set))
    sequences = []
    current_start = frames[0]
    current_end = frames[0]
    
    for i in range(1, len(frames)):
        if frames[i] == current_end + 1:
            # Consecutive frame - extend current sequence
            current_end = frames[i]
        else:
            # Gap found - save current sequence and start new one
            sequences.append((current_start, current_end))
            current_start = frames[i]
            current_end = frames[i]
    
    # Don't forget the last sequence
    sequences.append((current_start, current_end))
    
    return sequences

def find_frame_in_sequences(frame_idx, sequences):
    """
    Найти в какой последовательности находится кадр.
    Возвращает (seq_start, seq_end, position_in_sequence) или None.
    """
    for seq_start, seq_end in sequences:
        if seq_start <= frame_idx <= seq_end:
            position = frame_idx - seq_start
            return seq_start, seq_end, position
    return None

def find_neighbor_frames(all_frames, seq_start, frozen_frames, seq_end):
    """
    Найти ближайшие незамороженные кадры до и после последовательности.
    """
    prev_frame = None
    next_frame = None
    prev_idx = -1
    next_idx = -1
    
    # Поиск предыдущего незамороженного кадра (before sequence)
    for i in range(seq_start - 1, -1, -1):
        if i not in frozen_frames:
            prev_frame = all_frames[i]
            prev_idx = i
            break
    
    # Поиск следующего незамороженного кадра (after sequence)
    for i in range(seq_end + 1, len(all_frames)):
        if i not in frozen_frames:
            next_frame = all_frames[i]
            next_idx = i
            break
    
    logging.info(f"     Sequence: {seq_start}-{seq_end}, Prev: {prev_idx}, Next: {next_idx}")
    
    return prev_frame, next_frame

def interpolate_with_timestep_rife(prev_frame, next_frame, timestep, real_rife):
    """
    Интерполировать один кадр с указанным timestep используя НАСТОЯЩИЙ RIFE.
    """
    try:
        if real_rife and real_rife.available:
            # Use custom interpolation with specific timestep
            interpolated_frame = real_rife.interpolate_at_timestep(prev_frame, next_frame, timestep)
            if interpolated_frame is not None:
                return interpolated_frame
        
        raise Exception("❌ REAL RIFE not available!")
        
    except Exception as e:
        logging.error(f"REAL RIFE timestep interpolation failed: {e}")
        raise Exception(f"❌ REAL RIFE timestep interpolation failed: {e}")

def interpolate_with_real_rife(prev_frame, next_frame, real_rife):
    """
    Интерполировать один кадр между двумя используя НАСТОЯЩИЙ RIFE из оригинального репозитория.
    """
    return interpolate_with_timestep_rife(prev_frame, next_frame, 0.5, real_rife)

def create_ai_repair_report(repaired_count, total_freezes):
    """Create detailed repair report."""
    if total_freezes == 0:
        return "🤖 No freezes detected - no AI repair needed!"
    
    repair_pct = (repaired_count / total_freezes * 100) if total_freezes > 0 else 0
    
    report = f"""🤖 AI FREEZE REPAIR COMPLETE!

📊 Repair Results:
• Detected freezes: {total_freezes}
• Successfully repaired: {repaired_count} ({repair_pct:.1f}%)
• Failed repairs: {total_freezes - repaired_count}

⚡ Method: RIFE point interpolation
🎯 Strategy: Replace frozen frames with AI-generated intermediate frames
🔧 Neighbors: Uses closest non-frozen frames for interpolation

✨ Result: Smooth video with AI-repaired freeze points!"""
    
    return report