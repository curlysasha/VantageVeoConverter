"""
Audio synchronization utilities
"""
import subprocess
import os
import json
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import savgol_filter
import cv2
import whisper
import logging
import warnings

# Import binary utilities
from .binary_utils import get_ffmpeg, get_ffprobe, get_mp4fpsmod

def extract_and_standardize_audio(input_path, output_audio_path):
    """Extract and standardize audio to 16kHz mono PCM."""
    ffmpeg_path = get_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found! Place ffmpeg binary in bin/ directory")
    
    command = [
        ffmpeg_path, "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le", 
        "-ar", "16000", "-ac", "1", output_audio_path
    ]
    run_command(command)

def transcribe_audio(audio_path, output_transcript_path, whisper_model):
    """Transcribe audio using Whisper."""
    if whisper_model is None: 
        raise RuntimeError("Whisper model not loaded.")
    
    result = whisper_model.transcribe(audio_path)
    text = result["text"]
    formatted_text = text.strip().replace('. ', '.\n').replace('? ', '?\n')
    with open(output_transcript_path, 'w') as f:
        f.write(formatted_text) 
    return formatted_text

def forced_alignment(audio_path, transcript_path, output_alignment_path):
    """Perform forced alignment using Aeneas."""
    # Check if aeneas is available
    try:
        import aeneas
        logging.info("Aeneas module available")
    except ImportError:
        error_msg = """
❌ Aeneas not installed! Please install it:

Quick fix:
  pip install numpy>=1.24,<2.3 scipy>=1.9.0
  pip install --no-build-isolation aeneas>=1.7.3

Or run the automatic installer:
  python install_dependencies.py

System dependencies (Ubuntu/Debian):
  sudo apt-get install espeak libasound2-dev libespeak-dev
"""
        logging.error(error_msg)
        raise RuntimeError("Aeneas not available. Run: python install_dependencies.py")
    
    command = [
        "python3", "-m", "aeneas.tools.execute_task",
        audio_path, transcript_path,
        "task_language=eng|os_task_file_format=json|is_text_type=plain",
        output_alignment_path
    ]
    run_command(command)

def validate_video_format(video_path):
    """Validate video format and codec compatibility."""
    logging.info("Validating video format...")
    
    ffprobe_path = get_ffprobe()
    if not ffprobe_path:
        raise RuntimeError("ffprobe not found! Place ffprobe binary in bin/ directory")
    
    command = [
        ffprobe_path, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,codec_type,r_frame_rate,avg_frame_rate",
        "-show_entries", "format=format_name,duration",
        "-of", "json", video_path
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        format_name = info.get('format', {}).get('format_name', '').lower()
        if 'mp4' not in format_name and 'mov' not in format_name:
            warnings.warn(f"Video format '{format_name}' may not be compatible with mp4fpsmod. MP4/MOV recommended.")
        
        stream = info.get('streams', [{}])[0]
        codec = stream.get('codec_name', 'unknown')
        
        logging.info(f"Video format: {format_name}, Codec: {codec}")
        return info
        
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        logging.warning(f"Could not validate video format: {e}")
        return None

def detect_vfr_and_get_fps(video_path, video_info=None):
    """Detect if video has variable frame rate and get accurate FPS."""
    logging.info("Detecting frame rate...")
    
    if video_info:
        stream = video_info.get('streams', [{}])[0]
        r_frame_rate = stream.get('r_frame_rate', '')
        avg_frame_rate = stream.get('avg_frame_rate', '')
        
        def parse_fps(fps_str):
            if '/' in fps_str:
                num, den = fps_str.split('/')
                return float(num) / float(den) if float(den) != 0 else 0
            return 0
        
        r_fps = parse_fps(r_frame_rate)
        avg_fps = parse_fps(avg_frame_rate)
        
        if r_fps > 0 and avg_fps > 0:
            fps_diff = abs(r_fps - avg_fps)
            if fps_diff > 0.01:
                logging.warning(f"Variable frame rate detected! r_fps={r_fps:.3f}, avg_fps={avg_fps:.3f}")
                logging.info("Using average frame rate for processing")
                return avg_fps, True
            else:
                logging.info(f"Constant frame rate: {r_fps:.3f} fps")
                return r_fps, False
    
    cap = cv2.VideoCapture(video_path)
    opencv_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 100:
        logging.info("Calculating actual FPS from frame timestamps...")
        timestamps = []
        sample_size = min(100, total_frames)
        frame_step = total_frames // sample_size
        
        for i in range(0, sample_size * frame_step, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamps.append(timestamp)
        
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            avg_interval = np.mean(intervals)
            calculated_fps = (1.0 / avg_interval) / frame_step if avg_interval > 0 else opencv_fps
            
            fps_variance = np.std(intervals) / avg_interval if avg_interval > 0 else 0
            is_vfr = fps_variance > 0.1
            
            if is_vfr:
                logging.warning(f"Variable frame rate detected through timestamp analysis!")
            
            logging.info(f"Calculated FPS: {calculated_fps:.3f}, OpenCV FPS: {opencv_fps:.3f}")
            
            if abs(calculated_fps - opencv_fps) > 1.0:
                logging.info(f"Using calculated FPS: {calculated_fps:.3f}")
                cap.release()
                return calculated_fps, is_vfr
    
    cap.release()
    logging.info(f"Using OpenCV detected FPS: {opencv_fps:.3f}")
    return opencv_fps, False

def generate_vfr_timecodes(video_path, align_source_path, align_target_path, output_timecode_path,
                          smooth_interpolation=True, progress_callback=None):
    """Enhanced VFR timecode generation with smoothing and better edge handling."""
    logging.info("Generating VFR timecodes with improvements...")
    
    with open(align_source_path) as f: align_source = json.load(f)
    with open(align_target_path) as f: align_target = json.load(f)

    T_source, T_target = [0.0], [0.0]
    min_len = min(len(align_source['fragments']), len(align_target['fragments']))
    
    for i in range(min_len):
        T_source.append(float(align_source['fragments'][i]['end']))
        T_target.append(float(align_target['fragments'][i]['end']))

    # Get video details with improved FPS detection
    video_info = validate_video_format(video_path)
    video_fps, is_vfr = detect_vfr_and_get_fps(video_path, video_info)
    
    if is_vfr:
        logging.warning("Processing variable frame rate video - results may vary")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps
    cap.release()

    # Improved edge handling
    if T_source[-1] < video_duration:
        T_source.append(video_duration)
        if len(T_target) > 1:
            rate = (T_target[-1] - T_target[-2]) / (T_source[-1] - T_source[-2])
            T_target.append(T_target[-1] + rate * (video_duration - T_source[-2]))
        else:
            T_target.append(video_duration)
    
    # Enhanced interpolation
    if smooth_interpolation and len(T_source) > 3:
        try:
            alignment_func = CubicSpline(T_source, T_target, bc_type='natural', extrapolate=False)
            logging.info("Using cubic spline interpolation for smoother time warping")
        except:
            alignment_func = interp1d(T_source, T_target, kind='linear', 
                                    bounds_error=False, fill_value=(T_target[0], T_target[-1]))
            logging.info("Using linear interpolation")
    else:
        alignment_func = interp1d(T_source, T_target, kind='linear',
                                bounds_error=False, fill_value=(T_target[0], T_target[-1]))

    # Map frames with progress logging
    original_timestamps = np.arange(total_frames) / video_fps
    log_interval = max(1, total_frames // 20)
    
    new_timestamps = []
    for i, t in enumerate(original_timestamps):
        new_t = alignment_func(t)
        new_timestamps.append(new_t)
        
        if progress_callback and i % log_interval == 0:
            progress = (i / total_frames) * 100
            progress_callback(progress, f"Processing frame {i}/{total_frames}")
            logging.info(f"Timecode generation: {progress:.1f}% complete")
    
    new_timestamps = np.array(new_timestamps)
    
    # Apply smoothing filter
    if smooth_interpolation and len(new_timestamps) > 51:
        try:
            window_length = min(51, len(new_timestamps) if len(new_timestamps) % 2 == 1 else len(new_timestamps) - 1)
            new_timestamps = savgol_filter(new_timestamps, window_length, 3)
            logging.info("Applied smoothing filter to timestamps")
        except:
            logging.warning("Could not apply smoothing filter")
    
    new_timestamps_ms = (new_timestamps * 1000).round().astype(int)

    new_timestamps_ms = np.maximum(0, new_timestamps_ms)
    if len(new_timestamps_ms) > 0 and new_timestamps_ms[0] != 0:
        new_timestamps_ms -= new_timestamps_ms[0]

    with open(output_timecode_path, 'w') as f:
        f.write("# timecode format v2\n")
        prev_timestamp = -1
        collision_count = 0
        
        for timestamp in new_timestamps_ms:
            if timestamp <= prev_timestamp:
                timestamp = prev_timestamp + 1
                collision_count += 1
            f.write(f"{timestamp}\n")
            prev_timestamp = timestamp
        
        if collision_count > 0:
            logging.warning(f"Resolved {collision_count} timestamp collisions")

def retime_video(input_video_path, timecode_path, output_retimed_video_path):
    """Apply timestamps using mp4fpsmod."""
    binary = get_mp4fpsmod()
    
    if not binary:
        logging.error("mp4fpsmod not found! Please place binary in project/bin/ directory")
        # Fallback - just copy the original
        import shutil
        shutil.copy2(input_video_path, output_retimed_video_path)
        return
    
    logging.info(f"Running mp4fpsmod: {binary}")
    command = [binary, "-t", timecode_path, "-o", output_retimed_video_path, input_video_path]
    
    try:
        run_command(command)
        # Verify the output
        if os.path.exists(output_retimed_video_path):
            size = os.path.getsize(output_retimed_video_path)
            logging.info(f"mp4fpsmod completed. Output size: {size} bytes")
        else:
            logging.error("mp4fpsmod failed to create output file!")
    except Exception as e:
        logging.error(f"mp4fpsmod error: {e}")
        # Fallback - copy original
        shutil.copy2(input_video_path, output_retimed_video_path)

def mux_final_output(retimed_video_path, target_audio_path, final_output_path):
    """Combine retimed video with target audio."""
    logging.info("🔊 mux_final_output: Starting audio muxing...")
    logging.info(f"   Video input: {retimed_video_path}")
    logging.info(f"   Audio input: {target_audio_path}")
    logging.info(f"   Final output: {final_output_path}")
    
    # Check audio in target file first
    try:
        import json
        ffprobe_path = get_ffprobe()
        if not ffprobe_path:
            raise RuntimeError("ffprobe not found! Place ffprobe binary in bin/ directory")
        audio_check_cmd = [ffprobe_path, '-v', 'quiet', '-print_format', 'json', '-show_streams', target_audio_path]
        audio_check = subprocess.run(audio_check_cmd, capture_output=True, text=True, timeout=30)
        
        if audio_check.returncode == 0:
            probe_data = json.loads(audio_check.stdout)
            audio_streams = [s for s in probe_data.get('streams', []) if s.get('codec_type') == 'audio']
            logging.info(f"🔍 Target audio streams: {len(audio_streams)}")
            for i, stream in enumerate(audio_streams):
                logging.info(f"   Stream {i}: {stream.get('codec_name', 'unknown')} @ {stream.get('bit_rate', 'unknown')} bps")
        else:
            logging.warning(f"⚠️ Could not probe target audio: {audio_check.stderr}")
    except Exception as e:
        logging.warning(f"⚠️ Audio probe failed: {e}")
    
    ffmpeg_path = get_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found! Place ffmpeg binary in bin/ directory")
    
    command = [
        ffmpeg_path, "-y", "-v", "info",  # More verbose output
        "-i", retimed_video_path, "-i", target_audio_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",  # Set audio bitrate
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", final_output_path
    ]
    
    logging.info(f"🔧 FFmpeg command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        
        logging.info(f"📊 FFmpeg return code: {result.returncode}")
        if result.stdout:
            logging.info(f"📊 FFmpeg STDOUT:\n{result.stdout}")
        if result.stderr:
            logging.info(f"📊 FFmpeg STDERR:\n{result.stderr}")
        
        if result.returncode != 0:
            logging.error(f"❌ FFmpeg failed: {result.stderr}")
            raise RuntimeError(f"Command failed: {result.stderr}")
        else:
            logging.info("✅ mux_final_output completed successfully")
            
        return result.stdout
        
    except subprocess.TimeoutExpired:
        logging.error("❌ FFmpeg timeout in mux_final_output")
        raise RuntimeError("FFmpeg timeout")
    except Exception as e:
        logging.error(f"❌ mux_final_output error: {e}")
        raise

def run_command(command):
    """Helper function to run shell commands."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {result.stderr}")
    return result.stdout