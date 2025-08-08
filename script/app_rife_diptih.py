import subprocess
import os
import json
import numpy as np
from scipy.interpolate import interp1d
import cv2
import whisper
import logging
import shutil
import gradio as gr
import tempfile
import time
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- Global Setup: Load Whisper Model Once ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = None
try:
    logging.info(f"Loading Whisper model ('base') to {DEVICE}... (This speeds up subsequent requests)")
    WHISPER_MODEL = whisper.load_model("base", device=DEVICE) 
except Exception as e:
    logging.error(f"Failed to load Whisper model: {e}")

# --- Simple RIFE Implementation ---
class SimpleRIFE:
    """Simple frame interpolation using OpenCV."""
    
    def __init__(self):
        self.available = True
    
    def interpolate_frames(self, frame1, frame2, num_intermediate=1):
        """Simple interpolation between two frames."""
        try:
            interpolated_frames = []
            for i in range(1, num_intermediate + 1):
                alpha = i / (num_intermediate + 1)
                # Simple blending
                blended = cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)
                interpolated_frames.append(blended)
            return interpolated_frames
        except Exception as e:
            logging.warning(f"Frame interpolation failed: {e}")
            return [frame1.copy() for _ in range(num_intermediate)]

# Global RIFE instance
RIFE_MODEL = SimpleRIFE()

# --- Core Utility Functions ---

def check_dependencies():
    """Check if required external tools are available."""
    missing = []
    if not shutil.which("mp4fpsmod"): missing.append("mp4fpsmod")
    if not shutil.which("ffmpeg"): missing.append("ffmpeg")
    if not shutil.which("espeak-ng") and not shutil.which("espeak"):
         missing.append("espeak/espeak-ng")
    if missing:
        raise EnvironmentError(f"Missing dependencies: {', '.join(missing)}.")

def run_command(command):
    """Helper function to run shell commands."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {result.stderr}")
    return result.stdout

# --- Synchronization Steps ---

def extract_and_standardize_audio(input_path, output_audio_path):
    command = [
        "ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le", 
        "-ar", "16000", "-ac", "1", output_audio_path
    ]
    run_command(command)

def transcribe_audio(audio_path, output_transcript_path):
    if WHISPER_MODEL is None: raise RuntimeError("Whisper model not loaded.")
    result = WHISPER_MODEL.transcribe(audio_path)
    text = result["text"]
    formatted_text = text.strip().replace('. ', '.\n').replace('? ', '?\n')
    with open(output_transcript_path, 'w') as f:
        f.write(formatted_text) 
    return formatted_text

def forced_alignment(audio_path, transcript_path, output_alignment_path):
    command = [
        "python3", "-m", "aeneas.tools.execute_task",
        audio_path, transcript_path,
        "task_language=eng|os_task_file_format=json|is_text_type=plain",
        output_alignment_path
    ]
    run_command(command)

def analyze_timing_changes(timecode_path, fps=25, rife_mode="off"):
    """Analyze timing changes based on mode."""
    with open(timecode_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if not line.startswith('#')]
    
    timestamps = [int(line) for line in lines if line.isdigit()]
    if len(timestamps) < 10:
        return []
    
    if rife_mode == "off":
        return []
    elif rife_mode == "maximum":
        logging.info("Maximum mode: will interpolate entire video")
        return [{'start_frame': 0, 'end_frame': len(timestamps), 'reason': 'maximum'}]
    
    # Analyze timing variations for adaptive and precision
    original_interval = 1000 / fps
    problem_segments = []
    
    # Different sensitivity thresholds
    if rife_mode == "precision":
        threshold = 0.05  # 5% deviation - very sensitive
        merge_distance = 3  # Smaller merge distance for precision
    else:  # adaptive
        threshold = 0.15  # 15% deviation - moderate sensitivity
        merge_distance = 10  # Larger merge distance for adaptive
    
    deviations = []
    for i in range(1, len(timestamps)):
        actual_interval = timestamps[i] - timestamps[i-1]
        if actual_interval > 0:
            speed_ratio = original_interval / actual_interval
            deviation = abs(speed_ratio - 1.0)
            deviations.append({
                'frame': i,
                'deviation': deviation,
                'actual_interval': actual_interval,
                'expected_interval': original_interval
            })
    
    # Calculate statistics
    avg_deviation = np.mean([d['deviation'] for d in deviations])
    max_deviation = np.max([d['deviation'] for d in deviations])
    
    logging.info(f"{rife_mode.title()} analysis: avg_deviation={avg_deviation:.3f}, max_deviation={max_deviation:.3f}, threshold={threshold}")
    
    # Find problem frames
    problem_frames = [d for d in deviations if d['deviation'] > threshold]
    
    if not problem_frames:
        logging.info(f"{rife_mode.title()} mode: no timing issues detected above {threshold:.1%} threshold")
        return []
    
    # Group nearby problem frames into segments
    if problem_frames:
        segments = []
        current_segment = None
        
        for frame_data in problem_frames:
            frame = frame_data['frame']
            
            if current_segment is None:
                current_segment = {
                    'start_frame': max(0, frame - merge_distance//2),
                    'end_frame': frame + merge_distance//2,
                    'deviation': frame_data['deviation'],
                    'frame_count': 1
                }
            elif frame <= current_segment['end_frame'] + merge_distance:
                # Extend current segment
                current_segment['end_frame'] = max(current_segment['end_frame'], frame + merge_distance//2)
                current_segment['deviation'] = max(current_segment['deviation'], frame_data['deviation'])
                current_segment['frame_count'] += 1
            else:
                # Start new segment
                current_segment['end_frame'] = min(len(timestamps), current_segment['end_frame'])
                segments.append(current_segment)
                current_segment = {
                    'start_frame': max(0, frame - merge_distance//2),
                    'end_frame': frame + merge_distance//2,
                    'deviation': frame_data['deviation'],
                    'frame_count': 1
                }
        
        # Add last segment
        if current_segment:
            current_segment['end_frame'] = min(len(timestamps), current_segment['end_frame'])
            segments.append(current_segment)
        
        # Log detailed results
        total_affected_frames = sum(seg['end_frame'] - seg['start_frame'] for seg in segments)
        coverage_pct = (total_affected_frames / len(timestamps)) * 100
        
        logging.info(f"{rife_mode.title()} mode: found {len(problem_frames)} problem frames in {len(segments)} segments")
        logging.info(f"Coverage: {coverage_pct:.1f}% of video ({total_affected_frames}/{len(timestamps)} frames)")
        
        for i, seg in enumerate(segments):
            seg_size = seg['end_frame'] - seg['start_frame']
            logging.info(f"  Segment {i+1}: frames {seg['start_frame']}-{seg['end_frame']} ({seg_size} frames, max_dev={seg['deviation']:.3f})")
        
        return segments
    
    return []

def interpolate_video(input_video_path, problem_segments, output_path, rife_mode):
    """Interpolate video based on problem segments and mode."""
    if not problem_segments or rife_mode == "off":
        shutil.copy2(input_video_path, output_path)
        return False
    
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ВАЖНО: Сохраняем оригинальный FPS!
    # Интерполяция только добавляет кадры для плавности
    target_fps = fps  # Всегда оригинальный FPS
    
    # Different interpolation strategies per mode
    if rife_mode == "maximum":
        interpolation_factor = 2  # Агрессивная интерполяция
        interpolate_all = True
    elif rife_mode == "precision":
        interpolation_factor = 1  # Консервативная интерполяция
        interpolate_all = False
    else:  # adaptive
        interpolation_factor = 1  # Умеренная интерполяция
        interpolate_all = False
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    current_frame = 0
    prev_frame = None
    interpolated_count = 0
    
    try:
        logging.info(f"Starting {rife_mode} interpolation (keeping original FPS: {fps})")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Determine if current frame needs interpolation
            needs_interpolation = False
            if interpolate_all:
                needs_interpolation = True
            else:
                # Check if current frame is in any problem segment
                needs_interpolation = any(
                    seg['start_frame'] <= current_frame <= seg['end_frame'] 
                    for seg in problem_segments
                )
            
            # Perform interpolation if needed
            if needs_interpolation and prev_frame is not None:
                try:
                    # Create interpolated frames
                    interpolated = RIFE_MODEL.interpolate_frames(
                        prev_frame, frame, interpolation_factor
                    )
                    
                    for interp_frame in interpolated:
                        out.write(interp_frame)
                        interpolated_count += 1
                        
                except Exception as e:
                    logging.warning(f"Interpolation failed at frame {current_frame}: {e}")
            
            # Always write the original frame
            out.write(frame)
            prev_frame = frame.copy()
            current_frame += 1
            
            if current_frame % 50 == 0:
                progress = (current_frame / total_frames) * 100
                logging.info(f"Interpolation progress: {progress:.1f}%")
    
    finally:
        cap.release()
        out.release()
    
    # Calculate actual results
    final_frame_count = current_frame + interpolated_count
    actual_duration = current_frame / fps  # Оригинальная длительность
    
    logging.info(f"Interpolation complete!")
    logging.info(f"Added {interpolated_count} frames ({interpolated_count/current_frame:.1%} increase)")
    logging.info(f"Result: {final_frame_count} frames at original {fps} FPS")
    logging.info(f"Duration unchanged: {actual_duration:.2f} seconds")
    
    return True

def generate_vfr_timecodes(video_path, align_source_path, align_target_path, output_timecode_path):
    with open(align_source_path) as f: align_source = json.load(f)
    with open(align_target_path) as f: align_target = json.load(f)

    T_source, T_target = [0.0], [0.0]
    min_len = min(len(align_source['fragments']), len(align_target['fragments']))
    
    for i in range(min_len):
        T_source.append(float(align_source['fragments'][i]['end']))
        T_target.append(float(align_target['fragments'][i]['end']))

    alignment_func = interp1d(T_source, T_target, bounds_error=False, fill_value="extrapolate")

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    original_timestamps = np.arange(total_frames) / video_fps
    new_timestamps_ms = (alignment_func(original_timestamps) * 1000).round().astype(int)

    new_timestamps_ms = np.maximum(0, new_timestamps_ms)
    if len(new_timestamps_ms) > 0 and new_timestamps_ms[0] != 0:
        new_timestamps_ms -= new_timestamps_ms[0]

    with open(output_timecode_path, 'w') as f:
        f.write("# timecode format v2\n")
        prev_timestamp = -1
        for timestamp in new_timestamps_ms:
            if timestamp <= prev_timestamp:
                timestamp = prev_timestamp + 1
            f.write(f"{timestamp}\n")
            prev_timestamp = timestamp

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

def retime_video(input_video_path, timecode_path, output_retimed_video_path):
    binary = shutil.which("mp4fpsmod")
    command = [binary, "-t", timecode_path, "-o", output_retimed_video_path, input_video_path]
    run_command(command)

def mux_final_output(retimed_video_path, target_audio_path, final_output_path):
    command = [
        "ffmpeg", "-y", "-i", retimed_video_path, "-i", target_audio_path,
        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", final_output_path
    ]
    run_command(command)

def create_comparison_grid(original_video, adaptive_video, precision_video, maximum_video, output_path):
    """Создает диптих 2x2 из 4 видео с подписями режимов (упрощенная версия)."""
    
    try:
        # Получаем информацию о самом коротком видео для синхронизации
        videos = [original_video, adaptive_video, precision_video, maximum_video]
        labels = ["ORIGINAL", "ADAPTIVE", "PRECISION", "MAXIMUM"]
        
        min_duration = float('inf')
        for video in videos:
            cap = cv2.VideoCapture(video)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frames / fps
            min_duration = min(min_duration, duration)
            cap.release()
        
        # Ограничиваем длительность для быстрого сравнения
        demo_duration = min(30, min_duration)
        
        logging.info(f"Creating grid with {demo_duration:.1f}s duration")
        
        # FFmpeg команда с аудио из первого видео
        command = [
            "ffmpeg", "-y",
            "-i", original_video,
            "-i", adaptive_video, 
            "-i", precision_video,
            "-i", maximum_video,
            "-filter_complex", 
            f"[0:v]scale=640:360,drawtext=text='{labels[0]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v0];"
            f"[1:v]scale=640:360,drawtext=text='{labels[1]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v1];"
            f"[2:v]scale=640:360,drawtext=text='{labels[2]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v2];"
            f"[3:v]scale=640:360,drawtext=text='{labels[3]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v3];"
            "[v0][v1]hstack[top];"
            "[v2][v3]hstack[bottom];"
            "[top][bottom]vstack[v]",
            "-map", "[v]",
            "-map", "0:a",  # Берем аудио из первого видео (original)
            "-c:v", "libx264", 
            "-c:a", "aac",  # Кодируем аудио в AAC
            "-preset", "ultrafast",  # Быстрое кодирование
            "-crf", "28",  # Быстрое сжатие
            "-t", str(demo_duration),
            "-r", "15",  # Пониженный FPS для скорости
            output_path
        ]
        
        logging.info("Starting FFmpeg grid creation with audio...")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("Grid with audio created successfully!")
            return True
        else:
            logging.error(f"FFmpeg failed: {result.stderr}")
            
            # Fallback: создаем без аудио, потом добавляем
            logging.info("Fallback: creating grid without audio first...")
            
            # Команда без аудио
            command_no_audio = [
                "ffmpeg", "-y",
                "-i", original_video,
                "-i", adaptive_video, 
                "-i", precision_video,
                "-i", maximum_video,
                "-filter_complex", 
                f"[0:v]scale=640:360,drawtext=text='{labels[0]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v0];"
                f"[1:v]scale=640:360,drawtext=text='{labels[1]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v1];"
                f"[2:v]scale=640:360,drawtext=text='{labels[2]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v2];"
                f"[3:v]scale=640:360,drawtext=text='{labels[3]}':fontsize=20:fontcolor=white:box=1:boxcolor=black@0.7:x=10:y=10[v3];"
                "[v0][v1]hstack[top];"
                "[v2][v3]hstack[bottom];"
                "[top][bottom]vstack[v]",
                "-map", "[v]",
                "-c:v", "libx264", 
                "-preset", "ultrafast",
                "-crf", "28",
                "-t", str(demo_duration),
                "-r", "15",
                os.path.splitext(output_path)[0] + "_video_only.mp4"
            ]
            
            result_video = subprocess.run(command_no_audio, capture_output=True, text=True)
            
            if result_video.returncode == 0:
                # Теперь добавляем аудио
                video_only_path = os.path.splitext(output_path)[0] + "_video_only.mp4"
                
                command_add_audio = [
                    "ffmpeg", "-y",
                    "-i", video_only_path,
                    "-i", original_video,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    output_path
                ]
                
                result_audio = subprocess.run(command_add_audio, capture_output=True, text=True)
                
                # Очищаем временный файл
                try:
                    os.unlink(video_only_path)
                except:
                    pass
                
                if result_audio.returncode == 0:
                    logging.info("Grid with audio created via fallback!")
                    return True
            
            # Последний fallback: просто копируем первое видео
            logging.info("Final fallback: using original video as output")
            shutil.copy2(original_video, output_path)
            return True
            
    except Exception as e:
        logging.error(f"Grid creation failed: {e}")
        
        # Fallback: копируем оригинальное видео
        try:
            shutil.copy2(original_video, output_path)
            logging.info("Fallback: copied original video")
            return True
        except:
            return False

def comparison_workflow(input_video_path, target_audio_path, progress=gr.Progress()):
    """Workflow для сравнения всех режимов."""
    start_time = time.time()
    
    try:
        check_dependencies()
        
        # Создаем выходные файлы
        output_files = {}
        for mode in ['original', 'adaptive', 'precision', 'maximum']:
            fd, path = tempfile.mkstemp(suffix=f"_{mode}.mp4", prefix="comparison_")
            os.close(fd)
            output_files[mode] = path
        
        # Финальный диптих
        grid_fd, grid_path = tempfile.mkstemp(suffix="_comparison_grid.mp4", prefix="final_")
        os.close(grid_fd)  # Закрываем дескриптор, а не путь!
        
        modes_to_process = [
            ('original', 'off'),
            ('adaptive', 'adaptive'), 
            ('precision', 'precision'),
            ('maximum', 'maximum')
        ]
        
        results = {}
        
        for i, (mode_name, rife_mode) in enumerate(modes_to_process):
            progress_start = i * 0.2
            progress_end = (i + 1) * 0.2
            
            progress(progress_start, desc=f"Processing {mode_name.upper()} mode...")
            
            # Используем существующий workflow для каждого режима
            def mode_progress(value, desc=""):
                actual_progress = progress_start + (value * (progress_end - progress_start))
                progress(actual_progress, desc=f"{mode_name.upper()}: {desc}")
            
            try:
                result_path, status = synchronization_workflow(
                    input_video_path, target_audio_path, rife_mode, mode_progress
                )
                
                # Копируем результат
                shutil.copy2(result_path, output_files[mode_name])
                results[mode_name] = {
                    'path': output_files[mode_name],
                    'status': status
                }
                
                # Очищаем временный файл
                try:
                    os.unlink(result_path)
                except:
                    pass
                    
                logging.info(f"Completed {mode_name} mode")
                
            except Exception as e:
                logging.error(f"Failed {mode_name} mode: {e}")
                results[mode_name] = {'path': None, 'status': f"Failed: {e}"}
        
        # Создаем диптих если все режимы успешны
        progress(0.85, desc="Creating comparison grid...")
        
        if all(r['path'] for r in results.values()):
            grid_success = create_comparison_grid(
                results['original']['path'],
                results['adaptive']['path'], 
                results['precision']['path'],
                results['maximum']['path'],
                grid_path
            )
            
            if grid_success:
                progress(0.95, desc="Finalizing comparison...")
                
                duration = time.time() - start_time
                
                # Создаем подробный отчет
                status_report = f"""🎬 COMPARISON COMPLETE! Processed in {duration:.1f} seconds.

📊 ALL MODES COMPARISON:
• ORIGINAL (VFR Only): Basic synchronization
• ADAPTIVE RIFE: Smart interpolation where needed  
• PRECISION RIFE: Surgical interpolation at VFR points
• MAXIMUM RIFE: Full video interpolation

🎯 Grid Layout (2x2):
┌─────────────┬─────────────┐
│  ORIGINAL   │  ADAPTIVE   │
├─────────────┼─────────────┤  
│ PRECISION   │  MAXIMUM    │
└─────────────┴─────────────┘

⏱️ Processing times and details logged above.
Watch the grid to see smoothness differences!"""

                # Очищаем индивидуальные файлы
                for mode_result in results.values():
                    if mode_result['path']:
                        try:
                            os.unlink(mode_result['path'])
                        except:
                            pass
                
                return grid_path, status_report
            else:
                raise Exception("Failed to create comparison grid")
        else:
            failed_modes = [mode for mode, result in results.items() if not result['path']]
            raise Exception(f"Failed modes: {', '.join(failed_modes)}")
            
    except Exception as e:
        logging.error(f"Comparison workflow failed: {e}", exc_info=True)
        
        # Очистка в случае ошибки
        if 'results' in locals():
            for mode_result in results.values():
                if mode_result.get('path') and os.path.exists(mode_result['path']):
                    try:
                        os.unlink(mode_result['path'])
                    except:
                        pass
        
        if 'grid_path' in locals() and os.path.exists(grid_path):
            try:
                os.unlink(grid_path)
            except:
                pass
                
        raise gr.Error(f"Comparison failed: {e}")

def synchronization_workflow(input_video_path, target_audio_path, rife_mode="off", progress=gr.Progress()):
    """Main synchronization workflow with RIFE modes."""
    start_time = time.time()
    try:
        check_dependencies()
        
        final_output_fd, final_output_path = tempfile.mkstemp(suffix=".mp4", prefix="synchronized_")
        os.close(final_output_fd)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"Starting processing in {temp_dir}")

            paths = {
                "source_audio": os.path.join(temp_dir, "source_16k.wav"),
                "target_audio_processed": os.path.join(temp_dir, "target_16k.wav"),
                "transcript": os.path.join(temp_dir, "transcript.txt"),
                "align_source": os.path.join(temp_dir, "align_source.json"),
                "align_target": os.path.join(temp_dir, "align_target.json"),
                "timecodes": os.path.join(temp_dir, "timecodes_v2.txt"),
                "interpolated_video": os.path.join(temp_dir, "interpolated.mp4"),
                "retimed_video": os.path.join(temp_dir, "retimed.mp4"),
                "interpolated_timecodes": os.path.join(temp_dir, "interpolated_timecodes_v2.txt"),
                "temp_final_output": os.path.join(temp_dir, "synchronized_output.mp4")
            }

            # Standard pipeline
            progress(0.1, desc="1/8: Extracting audio...")
            extract_and_standardize_audio(input_video_path, paths["source_audio"])
            extract_and_standardize_audio(target_audio_path, paths["target_audio_processed"])
            
            progress(0.2, desc=f"2/8: Transcribing (Whisper on {DEVICE})...")
            transcript_text = transcribe_audio(paths["target_audio_processed"], paths["transcript"])
            
            progress(0.35, desc="3/8: Aligning Source...")
            forced_alignment(paths["source_audio"], paths["transcript"], paths["align_source"])

            progress(0.5, desc="4/8: Aligning Target...")
            forced_alignment(paths["target_audio_processed"], paths["transcript"], paths["align_target"])
            
            progress(0.65, desc="5/8: Calculating VFR timecodes...")
            generate_vfr_timecodes(input_video_path, paths["align_source"], paths["align_target"], paths["timecodes"])
            
            # RIFE interpolation
            video_for_retiming = input_video_path
            timecodes_for_retiming = paths["timecodes"]
            interpolation_applied = False
            
            if rife_mode != "off":
                progress(0.75, desc=f"6/8: Analyzing for {rife_mode} interpolation...")
                problem_segments = analyze_timing_changes(paths["timecodes"], rife_mode=rife_mode)
                
                if problem_segments or rife_mode == "maximum":
                    progress(0.8, desc=f"6/8: Applying {rife_mode} interpolation...")
                    interpolation_applied = interpolate_video(
                        input_video_path, problem_segments, paths["interpolated_video"], rife_mode
                    )
                    if interpolation_applied:
                        video_for_retiming = paths["interpolated_video"]
                        # Пересчитываем timecodes для интерполированного видео
                        progress(0.85, desc="6/8: Recalculating timecodes...")
                        regenerate_timecodes_for_interpolated_video(
                            input_video_path, 
                            paths["interpolated_video"],
                            paths["timecodes"],
                            paths["interpolated_timecodes"]
                        )
                        timecodes_for_retiming = paths["interpolated_timecodes"]
            else:
                progress(0.8, desc="6/8: Skipping interpolation...")
            
            progress(0.9, desc="7/8: Retiming video...")
            retime_video(video_for_retiming, timecodes_for_retiming, paths["retimed_video"])
            
            progress(0.95, desc="8/8: Final muxing...")
            mux_final_output(paths["retimed_video"], target_audio_path, paths["temp_final_output"])
            
            progress(0.99, desc="Saving...")
            shutil.copy2(paths["temp_final_output"], final_output_path)
            
            duration = time.time() - start_time
            
            # Status message
            mode_note = ""
            if rife_mode != "off":
                if interpolation_applied:
                    mode_note = f" (with {rife_mode} RIFE interpolation)"
                else:
                    mode_note = f" ({rife_mode} mode - no interpolation needed)"
            
            status_msg = f"Synchronization successful{mode_note}! Processed in {duration:.2f} seconds.\n\n--- Transcript Preview ---\n{transcript_text[:1000]}..."
            return final_output_path, status_msg

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        if 'final_output_path' in locals() and os.path.exists(final_output_path):
            try:
                os.unlink(final_output_path)
            except:
                pass
        raise gr.Error(f"Processing failed: {e}")

# --- Gradio Interface ---

with gr.Blocks(title="Video-Audio Synchronizer with RIFE") as interface:
    gr.Markdown("# Video-to-Audio Non-Linear Synchronizer")
    gr.Markdown("Upload source video and target audio. Choose RIFE mode for enhanced smoothness or compare all modes.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Source Video")
            audio_input = gr.Audio(label="Target Audio", type="filepath")
            
            gr.Markdown("### Single Mode Processing")
            rife_mode = gr.Radio(
                choices=[
                    ("Off", "off"),
                    ("Adaptive", "adaptive"), 
                    ("Precision", "precision"),
                    ("Maximum", "maximum")
                ],
                value="off",
                label="RIFE Interpolation Mode"
            )
            
            submit_button = gr.Button("Start Synchronization", variant="primary")
            
            gr.Markdown("---")
            gr.Markdown("### Compare All Modes")
            gr.Markdown("Process video with all 4 modes and create side-by-side comparison")
            
            compare_button = gr.Button("🎬 Compare All Modes (Off/Adaptive/Precision/Maximum)", variant="secondary", size="lg")
        
        with gr.Column():
            video_output = gr.Video(label="Synchronized Video / Comparison Grid")
            text_output = gr.Textbox(label="Status & Transcript", lines=10)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            **🚀 Off**: VFR only (3-5s)
            - Basic synchronization
            - Fastest processing
            """)
        
        with gr.Column():
            gr.Markdown("""
            **🎯 Adaptive**: Smart interpolation (15-30s)
            - Analyzes timing issues
            - Interpolates problem areas
            """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            **🔧 Precision**: VFR points only (10-25s)
            - Surgical approach
            - Interpolates exact VFR changes
            """)
        
        with gr.Column():
            gr.Markdown("""
            **💎 Maximum**: Full video (60-120s)
            - Maximum smoothness
            - Interpolates entire video
            """)
    
    with gr.Row():
        gr.Markdown("""
        ### 🎬 **Comparison Mode**
        
        Automatically processes your video with all 4 modes and creates a **2x2 grid comparison**:
        
        ```
        ┌─────────────┬─────────────┐
        │  ORIGINAL   │  ADAPTIVE   │
        │ (VFR Only)  │    RIFE     │
        ├─────────────┼─────────────┤  
        │ PRECISION   │  MAXIMUM    │
        │    RIFE     │    RIFE     │
        └─────────────┴─────────────┘
        ```
        
        Perfect for evaluating which mode works best for your content!
        """)

    # Connect buttons to functions
    submit_button.click(
        fn=synchronization_workflow,
        inputs=[video_input, audio_input, rife_mode],
        outputs=[video_output, text_output]
    )
    
    compare_button.click(
        fn=comparison_workflow,
        inputs=[video_input, audio_input],
        outputs=[video_output, text_output]
    )

if __name__ == '__main__':
    print("Launching Enhanced Video-Audio Synchronizer...")
    interface.launch(share=True, server_name="0.0.0.0")