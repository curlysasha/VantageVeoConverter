"""
Video processing and interpolation
"""
import logging
import cv2
import torch
import numpy as np
import shutil
from .timing_analyzer import detect_duplicate_frames, _create_variation

def interpolate_video(input_video_path, problem_segments, output_path, rife_mode, rife_model):
    """Interpolate video with smart duplicate frame replacement."""
    if not problem_segments or rife_mode == "off":
        shutil.copy2(input_video_path, output_path)
        return False
    
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load all frames first for analysis
    logging.info("🎬 Pre-loading frames for duplicate detection...")
    all_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    
    logging.info(f"📊 Loaded {len(all_frames)} frames for analysis")
    
    # Detect duplicate frames by comparing consecutive frames
    duplicate_map = detect_duplicate_frames(all_frames)
    duplicate_count = len(duplicate_map)
    logging.info(f"🔍 Found {duplicate_count} duplicate/similar frames ({duplicate_count/len(all_frames)*100:.1f}%)")
    
    # Create output video with duplicate replacement
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    replaced_count = 0
    interpolated_count = 0
    
    try:
        logging.info(f"🚀 Starting {rife_mode} with duplicate replacement")
        if torch.cuda.is_available():
            logging.info(f"💾 GPU acceleration enabled")
        
        for current_frame in range(len(all_frames)):
            frame = all_frames[current_frame]
            
            # Check if this frame is a duplicate and should be replaced
            if current_frame in duplicate_map and current_frame > 0:
                # This is a duplicate frame - replace it with interpolation
                ref_frame_idx = duplicate_map[current_frame]['reference_frame']
                ref_frame = all_frames[ref_frame_idx]
                
                # Find next non-duplicate frame for interpolation target
                target_frame_idx = current_frame + 1
                while (target_frame_idx < len(all_frames) and 
                       target_frame_idx in duplicate_map):
                    target_frame_idx += 1
                
                if target_frame_idx < len(all_frames):
                    target_frame = all_frames[target_frame_idx]
                    
                    try:
                        # Create interpolated frame to replace duplicate
                        interpolation_factor = current_frame - ref_frame_idx
                        max_interpolation = target_frame_idx - ref_frame_idx
                        
                        if max_interpolation > 0:
                            timestep = interpolation_factor / max_interpolation
                            
                            # Use RIFE to create replacement frame
                            interpolated_frames = rife_model.interpolate_frames(
                                ref_frame, target_frame, 1
                            )
                            
                            if interpolated_frames:
                                # Use interpolated frame instead of duplicate
                                replacement_frame = interpolated_frames[0]
                                out.write(replacement_frame)
                                replaced_count += 1
                                
                                if current_frame % 50 == 0:
                                    logging.info(f"🔄 Replaced duplicate frame {current_frame} with interpolation")
                            else:
                                # Fallback: use original frame with slight modification
                                modified_frame = _create_variation(frame)
                                out.write(modified_frame)
                                replaced_count += 1
                        else:
                            out.write(frame)
                    
                    except Exception as e:
                        logging.warning(f"Failed to replace duplicate frame {current_frame}: {e}")
                        out.write(frame)
                else:
                    # No target frame found, use original
                    out.write(frame)
            else:
                # Normal frame or first frame - write as is
                out.write(frame)
            
            # Additional interpolation for problem segments (if not maximum mode)
            if rife_mode != "maximum":
                needs_extra_interpolation = any(
                    seg['start_frame'] <= current_frame <= seg['end_frame']
                    for seg in problem_segments
                )
                
                if (needs_extra_interpolation and 
                    current_frame > 0 and 
                    current_frame not in duplicate_map):
                    
                    try:
                        prev_frame = all_frames[current_frame - 1]
                        extra_interpolated = rife_model.interpolate_frames(
                            prev_frame, frame, 1
                        )
                        
                        for extra_frame in extra_interpolated:
                            out.write(extra_frame)
                            interpolated_count += 1
                            
                    except Exception as e:
                        logging.debug(f"Extra interpolation failed at {current_frame}: {e}")
            
            # Progress logging
            if current_frame % 100 == 0:
                progress = (current_frame / len(all_frames)) * 100
                logging.info(f"Progress: {progress:.1f}% - Replaced: {replaced_count}, Added: {interpolated_count}")
                
                if torch.cuda.is_available() and current_frame % 500 == 0:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**2
                    logging.info(f"💾 GPU Memory: {gpu_memory:.1f}MB")
    
    finally:
        out.release()
    
    # Results
    total_modifications = replaced_count + interpolated_count
    logging.info(f"✅ Interpolation complete!")
    logging.info(f"🔄 Replaced {replaced_count} duplicate frames")
    logging.info(f"➕ Added {interpolated_count} extra interpolated frames")
    logging.info(f"🎯 Total improvements: {total_modifications} frames")
    logging.info(f"📈 Improvement rate: {total_modifications/len(all_frames)*100:.1f}%")
    
    return True

def regenerate_timecodes_for_interpolated_video(original_video_path, interpolated_video_path, original_timecode_path, new_timecode_path):
    """Пересчитывает timecode для интерполированного видео."""
    
    # Получаем информацию о видео
    cap_orig = cv2.VideoCapture(original_video_path)
    orig_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap_orig.get(cv2.CAP_PROP_FPS)
    cap_orig.release()
    
    cap_interp = cv2.VideoCapture(interpolated_video_path)
    interp_frames = int(cap_interp.get(cv2.CAP_PROP_FRAME_COUNT))
    interp_fps = cap_interp.get(cv2.CAP_PROP_FPS)
    cap_interp.release()
    
    logging.info(f"Original: {orig_frames} frames at {orig_fps} FPS")
    logging.info(f"Interpolated: {interp_frames} frames at {interp_fps} FPS")
    
    # Читаем оригинальные timecodes
    with open(original_timecode_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if not line.startswith('#')]
    
    orig_timestamps = [int(line) for line in lines if line.isdigit()]
    
    if len(orig_timestamps) != orig_frames:
        logging.warning(f"Timecode mismatch: {len(orig_timestamps)} timestamps vs {orig_frames} frames")
    
    # Создаем новые timecodes для интерполированного видео
    # Простой подход: равномерно распределяем интерполированные кадры
    ratio = interp_frames / orig_frames
    
    new_timestamps = []
    for i in range(interp_frames):
        # Находим соответствующий оригинальный кадр
        orig_frame_idx = int(i / ratio)
        orig_frame_idx = min(orig_frame_idx, len(orig_timestamps) - 1)
        
        # Интерполируем время между соседними оригинальными кадрами
        if orig_frame_idx < len(orig_timestamps) - 1:
            # Позиция между кадрами (0.0 - 1.0)
            sub_position = (i / ratio) - orig_frame_idx
            
            # Интерполируем время
            start_time = orig_timestamps[orig_frame_idx]
            end_time = orig_timestamps[orig_frame_idx + 1]
            interpolated_time = start_time + (end_time - start_time) * sub_position
        else:
            # Последний кадр
            interpolated_time = orig_timestamps[-1]
        
        new_timestamps.append(int(interpolated_time))
    
    # Санитизация: убеждаемся что времена монотонно возрастают
    for i in range(1, len(new_timestamps)):
        if new_timestamps[i] <= new_timestamps[i-1]:
            new_timestamps[i] = new_timestamps[i-1] + 1
    
    # Записываем новый timecode файл
    with open(new_timecode_path, 'w') as f:
        f.write("# timecode format v2\n")
        f.write(f"# Generated for interpolated video: {interp_frames} frames\n")
        for timestamp in new_timestamps:
            f.write(f"{timestamp}\n")
    
    logging.info(f"Generated {len(new_timestamps)} timecodes for interpolated video")