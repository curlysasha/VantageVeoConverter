"""
Video processing and interpolation - Smart version
"""
import logging
import shutil
import cv2
import os
import tempfile
from .motion_preserving_interpolator import MotionPreservingInterpolator

def interpolate_video(input_video_path, problem_segments, output_path, rife_mode, rife_model):
    """Motion-preserving video interpolation using dedicated processor."""
    if rife_mode == "off":
        shutil.copy2(input_video_path, output_path)
        return False
    
    logging.info(f"🎭 Starting {rife_mode} motion-preserving interpolation")
    
    if not problem_segments:
        logging.info("ℹ️  No timing problems detected - copying original video")
        shutil.copy2(input_video_path, output_path)
        return False
    
    # Use the motion-preserving interpolator directly
    interpolator = MotionPreservingInterpolator(device="cuda" if rife_model.device == "cuda" else "cpu")
    
    try:
        result = interpolator.process_video_motion_preserving(
            input_video_path, 
            output_path, 
            problem_segments, 
            mode=rife_mode
        )
        
        frames_fixed = result.get("frames_fixed", 0)
        return frames_fixed > 0
        
    except Exception as e:
        logging.error(f"Motion-preserving interpolation failed: {e}")
        # Fallback - copy original
        shutil.copy2(input_video_path, output_path)
        return False

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